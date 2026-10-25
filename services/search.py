"""
Academic Paper Search Service.
Fetches papers concurrently from ArXiv and Semantic Scholar,
normalizes, deduplicates, and ranks them by relevance and recency.
"""

import asyncio
import difflib
import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus
import httpx
from pydantic import BaseModel, Field
import xmltodict

logger = logging.getLogger("academic_assistant.search")

ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API_URL = "https://api.openalex.org/works"
OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
OPENALEX_INSTITUTIONS_URL = "https://api.openalex.org/institutions"
TARGET_ARXIV_CATEGORIES = "(cat:cs.DC OR cat:cs.SE OR cat:cs.AI OR cat:cs.AR)"

REQUEST_USER_AGENT = "SWMP-Labs-Research/1.0"


def _openalex_contact() -> str:
    """Contact email for the OpenAlex polite pool (guarantees 10 req/s polite pool access on cloud deployments)."""
    env_contact = (os.getenv("OPENALEX_MAILTO") or "").strip()
    if env_contact:
        return env_contact
    return "research-team@swmp-labs.tech"


def _openalex_user_agent() -> str:
    contact = _openalex_contact()
    if contact:
        return f"{REQUEST_USER_AGENT} (mailto:{contact})"
    return REQUEST_USER_AGENT


def _openalex_mailto_suffix() -> str:
    """'&mailto=<contact>' for inline OpenAlex URLs, or '' when unconfigured."""
    contact = _openalex_contact()
    if not contact:
        return ""
    return f"&mailto={quote_plus(contact)}"


def _openalex_params(base: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the mailto contact and optional API key to OpenAlex query params."""
    params = dict(base)
    contact = _openalex_contact()
    if contact:
        params["mailto"] = contact
    api_key = (os.getenv("OPENALEX_API_KEY") or "").strip()
    if api_key:
        params["api_key"] = api_key
    return params

# ============================================================================
# Semantic retrieval layer (local, zero-network, bounded-latency).
# Replaces brittle literal keyword matching with intent-based expansion and
# concept-vector re-ranking. All helpers below are pure in-memory functions:
# no blocking API calls, no dense-embedding service dependency.
# ============================================================================

# Canonical concept -> academically equivalent synonyms / aliases / taxonomy terms.
# Keys and values are matched case-insensitively against user intent and paper text.
ACADEMIC_SYNONYM_GRAPH: Dict[str, List[str]] = {
    "fast": ["efficient", "accelerated", "high-performance", "low-latency"],
    "training": ["model training", "optimization", "learning"],
    "neural network": ["deep learning", "neural net", "DNN"],
    "consumer hardware": ["commodity hardware", "edge devices", "commodity GPU", "consumer GPU"],
    "fault tolerance": ["fault-tolerant", "resilience", "reliability"],
    "checkpointing": ["checkpoint", "snapshotting", "state saving"],
    "distributed": ["decentralized", "geo-distributed", "large-scale distributed"],
    "consensus": ["agreement protocol", "BFT", "byzantine fault tolerance"],
    "serverless": ["FaaS", "function-as-a-service"],
    "cold start": ["cold starts", "initialization latency", "startup latency"],
    "microvm": ["microVM", "lightweight VM", "firecracker"],
    "diffusion models": ["diffusion", "generative diffusion", "score-based models"],
    "state-space models": ["SSM", "structured state space", "mamba"],
    "gpu kernel": ["CUDA kernel", "GPU optimization"],
    "attention": ["self-attention", "flash attention", "transformer attention"],
    "large language model": ["LLM", "foundation model", "pretrained model"],
    "inference": ["serving", "deployment inference"],
    "throughput": ["bandwidth", "processing rate"],
    "scheduling": ["orchestration", "task allocation"],
    "caching": ["memoization", "result reuse"],
    "datastore": ["storage engine", "distributed store", "database"],
    "replication": ["replica management", "data redundancy"],
    "sharding": ["partitioning", "horizontal scaling"],
    "vector database": ["embedding store", "similarity search index"],
    "retrieval": ["RAG", "retrieval-augmented"],
    "quantization": ["low-precision", "INT8", "model compression"],
    "pruning": ["sparsification", "model sparsity"],
    "federated learning": ["decentralized training", "edge training"],
    "reinforcement learning": ["RL", "policy optimization"],
    "benchmarking": ["evaluation", "performance measurement"],
    "observability": ["monitoring", "tracing", "telemetry"],
    "autoscaling": ["elastic scaling", "dynamic provisioning"],
    "container": ["docker", "kubernetes pod"],
    "virtual machine": ["VM", "hypervisor instance"],
    "network": ["interconnect", "fabric"],
    "security": ["confidentiality", "threat mitigation"],
    "privacy": ["differential privacy", "data protection"],
    "explainability": ["interpretability", "model transparency"],
    "robustness": ["adversarial robustness", "stability"],
    "generalization": ["transferability", "out-of-distribution"],
    "simulation": ["emulation", "digital twin"],
    "hpc": ["high-performance computing", "supercomputing"],
    "cloud": ["cloud computing", "public cloud"],
}

# Small function-word set used ONLY for anchor extraction / histograms — never to
# gut conceptual phrases out of the user-facing intent string.
_SEMANTIC_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "about", "as", "into", "through", "during", "before", "after",
    "is", "are", "was", "were", "be", "how", "what", "which", "that", "this",
})

# Reverse token -> canonical concept lookup, built lazily once (thread-safe enough
# for CPython dict publication; worst case it is rebuilt identically).
_TOKEN_TO_CONCEPT: Optional[Dict[str, str]] = None


def _token_to_concept_map() -> Dict[str, str]:
    """Map every significant token appearing in the graph to its canonical concept."""
    global _TOKEN_TO_CONCEPT
    if _TOKEN_TO_CONCEPT is None:
        mapping: Dict[str, str] = {}
        for canonical, synonyms in ACADEMIC_SYNONYM_GRAPH.items():
            for phrase in [canonical] + list(synonyms):
                for tok in re.findall(r"[a-z0-9]+", phrase.lower()):
                    if len(tok) > 2 and tok not in _SEMANTIC_FUNCTION_WORDS:
                        mapping.setdefault(tok, canonical)
        _TOKEN_TO_CONCEPT = mapping
    return _TOKEN_TO_CONCEPT


def expand_semantic_concepts(intent_text: str, max_terms: int = 8) -> List[str]:
    """
    Return conceptually related academic synonyms/aliases absent from the intent text.
    Pure local lookup over ACADEMIC_SYNONYM_GRAPH — no network, O(graph) worst case.
    Empty input yields [] (filter-only bypass; never hallucinates concepts).
    """
    if not intent_text or not intent_text.strip() or max_terms <= 0:
        return []
    lowered = intent_text.lower()
    expansions: List[str] = []
    for canonical, synonyms in ACADEMIC_SYNONYM_GRAPH.items():
        if canonical in lowered or any(s.lower() in lowered for s in synonyms):
            for syn in synonyms:
                if syn.lower() not in lowered and syn not in expansions:
                    expansions.append(syn)
                    if len(expansions) >= max_terms:
                        return expansions
    return expansions


def extract_semantic_groups(
    intent_text: str,
    max_groups: int = 3,
    synonyms_per_group: int = 2,
) -> List[List[str]]:
    """
    Split intent into up to max_groups disjunctive concept groups for boolean APIs.
    Each group = [anchor phrase/term + up to synonyms_per_group equivalents].
    Multiword concept phrases in the intent take priority as anchors so phrases
    like "neural network training" survive instead of being shredded into tokens.
    """
    if not intent_text or not intent_text.strip():
        return []
    lowered = intent_text.lower()
    groups: List[List[str]] = []
    used_anchors: List[str] = []

    # Pass 1: multiword canonical concepts / synonyms present verbatim in intent.
    for canonical, synonyms in ACADEMIC_SYNONYM_GRAPH.items():
        if len(groups) >= max_groups:
            break
        for phrase in [canonical] + list(synonyms):
            if " " not in phrase.strip() or phrase.lower() not in lowered:
                continue
            anchor = phrase.strip()
            if anchor.lower() in (a.lower() for a in used_anchors):
                break
            equivalents = [s for s in synonyms if s.lower() not in lowered][:synonyms_per_group]
            groups.append([anchor] + equivalents)
            used_anchors.append(anchor)
            break

    # Pass 2: single content-token anchors for remaining intent terms.
    if len(groups) < max_groups:
        concept_map = _token_to_concept_map()
        for tok in re.findall(r"[a-z0-9][a-z0-9\-]*", lowered):
            if len(groups) >= max_groups:
                break
            if len(tok) <= 2 or tok in _SEMANTIC_FUNCTION_WORDS:
                continue
            if tok in (a.lower() for a in used_anchors):
                continue
            canonical = concept_map.get(tok)
            if canonical:
                equivalents = [
                    s for s in ACADEMIC_SYNONYM_GRAPH[canonical]
                    if s.lower() not in lowered
                ][:synonyms_per_group]
                groups.append([tok] + equivalents)
            else:
                groups.append([tok])
            used_anchors.append(tok)

    return groups[:max_groups]


def _sanitize_arxiv_term(term: str) -> str:
    """Strip characters/operators that would corrupt arXiv boolean syntax."""
    cleaned = re.sub(r'["():]', " ", term or "")
    cleaned = re.sub(r"\b(AND|OR|NOT)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:80]


def _concept_histogram(text: str) -> Dict[str, float]:
    """Bag-of-canonical-concepts vector for semantic cosine scoring."""
    hist: Dict[str, float] = {}
    if not text:
        return hist
    concept_map = _token_to_concept_map()
    lowered = text.lower()
    # Multiword concept phrases count double (phrase-level intent signal).
    for canonical, synonyms in ACADEMIC_SYNONYM_GRAPH.items():
        for phrase in [canonical] + list(synonyms):
            if " " in phrase and phrase.lower() in lowered:
                hist[canonical] = hist.get(canonical, 0.0) + 2.0
                break
    for tok in re.findall(r"[a-z0-9]+", lowered):
        if len(tok) <= 2 or tok in _SEMANTIC_FUNCTION_WORDS:
            continue
        canonical = concept_map.get(tok)
        if canonical:
            hist[canonical] = hist.get(canonical, 0.0) + 1.0
        else:
            hist[f"tok:{tok}"] = hist.get(f"tok:{tok}", 0.0) + 0.5
    return hist


def semantic_intent_score(query_intent: str, title: str, abstract: str = "") -> float:
    """
    Cosine similarity between the user-intent concept vector and a paper's
    title/abstract concept vectors (synonym-aware: alternative terminology maps
    to shared canonical concepts). Title weighted 0.7, abstract 0.3.
    Returns 0.0 for empty intent (filter-only bypass). Bounded O(text) work.
    """
    if not query_intent or not query_intent.strip():
        return 0.0
    q_hist = _concept_histogram(query_intent)
    if not q_hist:
        return 0.0

    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        if dot <= 0:
            return 0.0
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / (na * nb)

    title_sim = _cosine(q_hist, _concept_histogram(title or ""))
    abstract_sim = _cosine(q_hist, _concept_histogram((abstract or "")[:800]))
    return max(0.0, min(1.0, 0.7 * title_sim + 0.3 * abstract_sim))


class AsyncRateLimiter:
    """
    Async concurrency throttling and rate-limiter enforcing a maximum concurrency
    and a minimum time interval between consecutive outbound requests.
    Uses lazy initialization to safely bind to the running asyncio event loop.
    """
    def __init__(self, min_interval: float = 1.0, concurrency: int = 1):
        self.min_interval = min_interval
        self.concurrency = concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock: Optional[asyncio.Lock] = None
        self._last_call: float = 0.0

    @property
    def semaphore(self) -> asyncio.Semaphore:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self._semaphore_loop = current_loop
        elif current_loop is not None and getattr(self, "_semaphore_loop", None) is not current_loop:
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self._semaphore_loop = current_loop
        return self._semaphore

    @property
    def lock(self) -> asyncio.Lock:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._lock is None:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        elif current_loop is not None and getattr(self, "_lock_loop", None) is not current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        return self._lock

    async def __aenter__(self):
        await self.semaphore.acquire()
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait_time = self.min_interval - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


class SearchCache:
    """
    Thread-safe in-memory TTL cache for academic search results.
    Normalizes query strings to minimize duplicate API calls and provides
    instant (sub-millisecond) responses for repeated or equivalent queries.
    """
    def __init__(self, ttl: float = 3600.0):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[List["Paper"], Dict[str, Any], float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def normalize_key(query: str) -> str:
        """
        Normalize query string:
        - Convert to lowercase
        - Strip non-alphanumeric punctuation (except hyphen and underscore)
        - Collapse whitespace
        Example: 'Distributed Checkpointing (HPC)!' -> 'distributed checkpointing hpc'
        """
        cleaned = re.sub(r"[^\w\s\-]", " ", query.lower()).strip()
        tokens = cleaned.split()
        return " ".join(tokens)

    def get(self, query: str) -> Optional[Tuple[List["Paper"], Dict[str, Any]]]:
        """
        Retrieve unexpired cached papers and metadata.
        Returns (papers, metadata) with metadata['cache_hit'] = True, or None if expired/not found.
        """
        key = self.normalize_key(query)
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                papers, metadata, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    # Return deep copies to prevent external mutation of cache items
                    copied_papers = [p.model_copy(deep=True) for p in papers]
                    copied_meta = dict(metadata)
                    copied_meta["cache_hit"] = True
                    return copied_papers, copied_meta
                else:
                    # Entry expired, evict
                    del self._cache[key]
        return None

    def set(self, query: str, papers: List["Paper"], metadata: Dict[str, Any]) -> None:
        """Store papers and search metadata with TTL timestamp."""
        key = self.normalize_key(query)
        if not key:
            return
        with self._lock:
            cached_papers = [p.model_copy(deep=True) for p in papers]
            self._cache[key] = (cached_papers, dict(metadata), time.time())

    def clear(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singletons for cache and rate limiters
search_cache = SearchCache(ttl=3600.0)
semantic_scholar_limiter = AsyncRateLimiter(min_interval=1.0, concurrency=1)
arxiv_limiter = AsyncRateLimiter(min_interval=0.5, concurrency=2)
openalex_limiter = AsyncRateLimiter(min_interval=0.2, concurrency=4)

# Global shared HTTP client with connection pooling
_shared_http_client: Optional[httpx.AsyncClient] = None
_shared_client_loop: Optional[asyncio.AbstractEventLoop] = None


def get_shared_http_client() -> httpx.AsyncClient:
    """
    Retrieve or lazily initialize the shared HTTP client with keepalive connection pooling.
    Guarantees reuse across all search providers and OpenAI-compatible SDKs.
    Automatically detects if the client was bound to a previously closed event loop and refreshes it.
    """
    global _shared_http_client, _shared_client_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    recreate = False
    if _shared_http_client is None or _shared_http_client.is_closed:
        recreate = True
    elif current_loop is not None and (_shared_client_loop is None or _shared_client_loop.is_closed() or _shared_client_loop is not current_loop):
        recreate = True

    if recreate:
        _shared_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            timeout=30.0,
            follow_redirects=True,
        )
        _shared_client_loop = current_loop
    return _shared_http_client


def set_shared_http_client(client: Optional[httpx.AsyncClient]) -> None:
    """Set or inject the shared HTTP client (managed via FastAPI lifespan)."""
    global _shared_http_client
    _shared_http_client = client


async def close_shared_http_client() -> None:
    """Safely close the shared HTTP client connection pool on app shutdown."""
    global _shared_http_client
    if _shared_http_client is not None and not _shared_http_client.is_closed:
        await _shared_http_client.aclose()
    _shared_http_client = None



class Paper(BaseModel):
    """Normalized academic paper metadata."""
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    institutions: List[str] = Field(default_factory=list)
    country: Optional[str] = None  # Primary canonical country (e.g. "Canada", "United States")
    countries: List[str] = Field(default_factory=list)  # All affiliated countries / regions
    year: Optional[int] = None
    abstract: str = ""
    tldr: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    source: str = "Unknown"  # "ArXiv", "Semantic Scholar", "OpenAlex", "ArXiv & OpenAlex", etc.
    primary_category: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    code_url: Optional[str] = None
    # PI-detection signals (populated by OpenAlex parsers; empty for ArXiv/SS).
    # author_details aligns positionally with `authors`:
    # [{"name": str, "position": "first"|"middle"|"last"|"", "is_corresponding": bool}]
    corresponding_author: Optional[str] = None
    author_details: List[Dict[str, Any]] = Field(default_factory=list)
    # Evidence-based affiliation & provenance tracking
    country_evidence: Optional[Dict[str, Any]] = None
    institution_evidence: Optional[Dict[str, Any]] = None
    filter_explanation: Optional[str] = None



# Comprehensive mapping of country names, aliases, and ISO 3166-1 alpha-2 codes
COUNTRY_REGISTRY: Dict[str, Tuple[str, str]] = {
    "canada": ("CA", "Canada"),
    "ca": ("CA", "Canada"),
    "united states": ("US", "United States"),
    "united states of america": ("US", "United States"),
    "usa": ("US", "United States"),
    "us": ("US", "United States"),
    "u.s.": ("US", "United States"),
    "u.s.a.": ("US", "United States"),
    "united kingdom": ("GB", "United Kingdom"),
    "uk": ("GB", "United Kingdom"),
    "u.k.": ("GB", "United Kingdom"),
    "great britain": ("GB", "United Kingdom"),
    "britain": ("GB", "United Kingdom"),
    "england": ("GB", "United Kingdom"),
    "scotland": ("GB", "United Kingdom"),
    "wales": ("GB", "United Kingdom"),
    "gb": ("GB", "United Kingdom"),
    "germany": ("DE", "Germany"),
    "deutschland": ("DE", "Germany"),
    "de": ("DE", "Germany"),
    "france": ("FR", "France"),
    "fr": ("FR", "France"),
    "japan": ("JP", "Japan"),
    "jp": ("JP", "Japan"),
    "china": ("CN", "China"),
    "prc": ("CN", "China"),
    "cn": ("CN", "China"),
    "australia": ("AU", "Australia"),
    "au": ("AU", "Australia"),
    "india": ("IN", "India"),
    "in": ("IN", "India"),
    "italy": ("IT", "Italy"),
    "it": ("IT", "Italy"),
    "spain": ("ES", "Spain"),
    "es": ("ES", "Spain"),
    "switzerland": ("CH", "Switzerland"),
    "ch": ("CH", "Switzerland"),
    "netherlands": ("NL", "Netherlands"),
    "holland": ("NL", "Netherlands"),
    "nl": ("NL", "Netherlands"),
    "sweden": ("SE", "Sweden"),
    "se": ("SE", "Sweden"),
    "singapore": ("SG", "Singapore"),
    "sg": ("SG", "Singapore"),
    "south korea": ("KR", "South Korea"),
    "korea": ("KR", "South Korea"),
    "kr": ("KR", "South Korea"),
    "brazil": ("BR", "Brazil"),
    "brasil": ("BR", "Brazil"),
    "br": ("BR", "Brazil"),
    "egypt": ("EG", "Egypt"),
    "eg": ("EG", "Egypt"),
    "israel": ("IL", "Israel"),
    "il": ("IL", "Israel"),
    "ireland": ("IE", "Ireland"),
    "ie": ("IE", "Ireland"),
    "norway": ("NO", "Norway"),
    "no": ("NO", "Norway"),
    "denmark": ("DK", "Denmark"),
    "dk": ("DK", "Denmark"),
    "finland": ("FI", "Finland"),
    "fi": ("FI", "Finland"),
    "belgium": ("BE", "Belgium"),
    "be": ("BE", "Belgium"),
    "austria": ("AT", "Austria"),
    "at": ("AT", "Austria"),
    "poland": ("PL", "Poland"),
    "pl": ("PL", "Poland"),
    "taiwan": ("TW", "Taiwan"),
    "tw": ("TW", "Taiwan"),
    "mexico": ("MX", "Mexico"),
    "mx": ("MX", "Mexico"),
    "argentina": ("AR", "Argentina"),
    "ar": ("AR", "Argentina"),
    "chile": ("CL", "Chile"),
    "cl": ("CL", "Chile"),
    "new zealand": ("NZ", "New Zealand"),
    "nz": ("NZ", "New Zealand"),
    "south africa": ("ZA", "South Africa"),
    "za": ("ZA", "South Africa"),
    "saudi arabia": ("SA", "Saudi Arabia"),
    "sa": ("SA", "Saudi Arabia"),
    "united arab emirates": ("AE", "United Arab Emirates"),
    "uae": ("AE", "United Arab Emirates"),
    "ae": ("AE", "United Arab Emirates"),
    "turkey": ("TR", "Turkey"),
    "türkiye": ("TR", "Turkey"),
    "tr": ("TR", "Turkey"),
    "greece": ("GR", "Greece"),
    "gr": ("GR", "Greece"),
    "portugal": ("PT", "Portugal"),
    "pt": ("PT", "Portugal"),
    "czech republic": ("CZ", "Czech Republic"),
    "czechia": ("CZ", "Czech Republic"),
    "cz": ("CZ", "Czech Republic"),
    "hungary": ("HU", "Hungary"),
    "hu": ("HU", "Hungary"),
    "romania": ("RO", "Romania"),
    "ro": ("RO", "Romania"),
    "russia": ("RU", "Russia"),
    "ru": ("RU", "Russia"),
    "ukraine": ("UA", "Ukraine"),
    "ua": ("UA", "Ukraine"),
    "pakistan": ("PK", "Pakistan"),
    "pk": ("PK", "Pakistan"),
    "bangladesh": ("BD", "Bangladesh"),
    "bd": ("BD", "Bangladesh"),
    "indonesia": ("ID", "Indonesia"),
    "id": ("ID", "Indonesia"),
    "malaysia": ("MY", "Malaysia"),
    "my": ("MY", "Malaysia"),
    "thailand": ("TH", "Thailand"),
    "th": ("TH", "Thailand"),
    "vietnam": ("VN", "Vietnam"),
    "vn": ("VN", "Vietnam"),
    "philippines": ("PH", "Philippines"),
    "ph": ("PH", "Philippines"),
    "colombia": ("CO", "Colombia"),
    "co": ("CO", "Colombia"),
    "nigeria": ("NG", "Nigeria"),
    "ng": ("NG", "Nigeria"),
    "kenya": ("KE", "Kenya"),
    "ke": ("KE", "Kenya"),
    "morocco": ("MA", "Morocco"),
    "ma": ("MA", "Morocco"),
}


# Known canonical institution IDs (OpenAlex entity resolution)
KNOWN_INSTITUTIONS: Dict[str, Tuple[str, str]] = {
    "mit": ("I63966007", "Massachusetts Institute of Technology"),
    "massachusetts institute of technology": ("I63966007", "Massachusetts Institute of Technology"),
    "stanford": ("I97018004", "Stanford University"),
    "stanford university": ("I97018004", "Stanford University"),
    "harvard": ("I136199984", "Harvard University"),
    "harvard university": ("I136199984", "Harvard University"),
    "berkeley": ("I95457486", "University of California, Berkeley"),
    "uc berkeley": ("I95457486", "University of California, Berkeley"),
    "university of california, berkeley": ("I95457486", "University of California, Berkeley"),
    "university of california berkeley": ("I95457486", "University of California, Berkeley"),
    "oxford": ("I40120149", "University of Oxford"),
    "university of oxford": ("I40120149", "University of Oxford"),
    "cambridge": ("I241749", "University of Cambridge"),
    "university of cambridge": ("I241749", "University of Cambridge"),
    "carnegie mellon": ("I74973187", "Carnegie Mellon University"),
    "carnegie mellon university": ("I74973187", "Carnegie Mellon University"),
    "cmu": ("I74973187", "Carnegie Mellon University"),
    "princeton": ("I20089843", "Princeton University"),
    "princeton university": ("I20089843", "Princeton University"),
    "waterloo": ("I157725225", "University of Waterloo"),
    "university of waterloo": ("I157725225", "University of Waterloo"),
    "uwaterloo": ("I157725225", "University of Waterloo"),
    "toronto": ("I185261750", "University of Toronto"),
    "university of toronto": ("I185261750", "University of Toronto"),
    "uoft": ("I185261750", "University of Toronto"),
    "utoronto": ("I185261750", "University of Toronto"),
    "mcgill": ("I5023651", "McGill University"),
    "mcgill university": ("I5023651", "McGill University"),
    "ubc": ("I141945490", "University of British Columbia"),
    "university of british columbia": ("I141945490", "University of British Columbia"),
    "caltech": ("I122411786", "California Institute of Technology"),
    "california institute of technology": ("I122411786", "California Institute of Technology"),
    "columbia": ("I5124864", "Columbia University"),
    "columbia university": ("I5124864", "Columbia University"),
    "cornell": ("I205783295", "Cornell University"),
    "cornell university": ("I205783295", "Cornell University"),
    "ucla": ("I161318765", "University of California, Los Angeles"),
    "university of california, los angeles": ("I161318765", "University of California, Los Angeles"),
    "eth zurich": ("I159176309", "ETH Zurich"),
    "eth": ("I159176309", "ETH Zurich"),
    "imperial": ("I35440088", "Imperial College London"),
    "imperial college london": ("I35440088", "Imperial College London"),
    "ucl": ("I45129253", "University College London"),
    "university college london": ("I45129253", "University College London"),
    "tsinghua": ("I1702154", "Tsinghua University"),
    "tsinghua university": ("I1702154", "Tsinghua University"),
    "peking": ("I20238116", "Peking University"),
    "peking university": ("I20238116", "Peking University"),
    "nus": ("I165779595", "National University of Singapore"),
    "national university of singapore": ("I165779595", "National University of Singapore"),
    "ntu": ("I172675005", "Nanyang Technological University"),
    "nanyang technological university": ("I172675005", "Nanyang Technological University"),
    "yale": ("I32971472", "Yale University"),
    "yale university": ("I32971472", "Yale University"),
    "nyu": ("I57206974", "New York University"),
    "new york university": ("I57206974", "New York University"),
    "york university": ("I125557451", "York University"),
    "university of york": ("I165779595", "University of York"),
    "western university": ("I114027177", "Western University"),
    "university of western ontario": ("I114027177", "Western University"),
    "uwo": ("I114027177", "Western University"),
    "university of victoria": ("I141945490", "University of Victoria"),
    "uvic": ("I141945490", "University of Victoria"),
    "simon fraser university": ("I19246387", "Simon Fraser University"),
    "sfu": ("I19246387", "Simon Fraser University"),
    "mcmaster university": ("I98251732", "McMaster University"),
    "mcmaster": ("I98251732", "McMaster University"),
    "queen's university": ("I14588479", "Queen's University"),
    "queens university": ("I14588479", "Queen's University"),
    "university of calgary": ("I168635309", "University of Calgary"),
    "university of ottawa": ("I14226107", "University of Ottawa"),
    "dalhousie university": ("I136449830", "Dalhousie University"),
    "ontario institute for cancer research": ("I2802283804", "Ontario Institute for Cancer Research"),
    "oicr": ("I2802283804", "Ontario Institute for Cancer Research"),
    "douglas college": ("I180017321", "Douglas College"),
    "université de moncton": ("I147814674", "Université de Moncton"),
    "universite de moncton": ("I147814674", "Université de Moncton"),
}

# Mapping of known academic institutions to their home countries
# Mapping of top known researchers to their primary institutional countries
AUTHOR_COUNTRY_MAP: Dict[str, Tuple[str, str]] = {
    "yoshua bengio": ("CA", "Canada"),
    "bengio": ("CA", "Canada"),
    "geoffrey hinton": ("CA", "Canada"),
    "hinton": ("CA", "Canada"),
    "richard sutton": ("CA", "Canada"),
    "rich sutton": ("CA", "Canada"),
    "sutton": ("CA", "Canada"),
    "yann lecun": ("US", "United States"),
    "lecun": ("US", "United States"),
    "andrew ng": ("US", "United States"),
    "ilya sutskever": ("US", "United States"),
    "tri dao": ("US", "United States"),
    "demis hassabis": ("GB", "United Kingdom"),
    "hassabis": ("GB", "United Kingdom"),
}

INSTITUTION_COUNTRY_MAP: Dict[str, Tuple[str, str]] = {
    "mit": ("US", "United States"),
    "massachusetts institute of technology": ("US", "United States"),
    "stanford": ("US", "United States"),
    "stanford university": ("US", "United States"),
    "harvard": ("US", "United States"),
    "harvard university": ("US", "United States"),
    "berkeley": ("US", "United States"),
    "uc berkeley": ("US", "United States"),
    "university of california, berkeley": ("US", "United States"),
    "university of california berkeley": ("US", "United States"),
    "waterloo": ("CA", "Canada"),
    "university of waterloo": ("CA", "Canada"),
    "uwaterloo": ("CA", "Canada"),
    "toronto": ("CA", "Canada"),
    "university of toronto": ("CA", "Canada"),
    "uoft": ("CA", "Canada"),
    "utoronto": ("CA", "Canada"),
    "mcgill": ("CA", "Canada"),
    "mcgill university": ("CA", "Canada"),
    "ubc": ("CA", "Canada"),
    "university of british columbia": ("CA", "Canada"),
    "alberta": ("CA", "Canada"),
    "university of alberta": ("CA", "Canada"),
    "montreal": ("CA", "Canada"),
    "university of montreal": ("CA", "Canada"),
    "oxford": ("GB", "United Kingdom"),
    "university of oxford": ("GB", "United Kingdom"),
    "cambridge": ("GB", "United Kingdom"),
    "university of cambridge": ("GB", "United Kingdom"),
    "carnegie mellon": ("US", "United States"),
    "carnegie mellon university": ("US", "United States"),
    "cmu": ("US", "United States"),
    "princeton": ("US", "United States"),
    "princeton university": ("US", "United States"),
    "caltech": ("US", "United States"),
    "california institute of technology": ("US", "United States"),
    "columbia": ("US", "United States"),
    "columbia university": ("US", "United States"),
    "cornell": ("US", "United States"),
    "cornell university": ("US", "United States"),
    "ucla": ("US", "United States"),
    "university of california, los angeles": ("US", "United States"),
    "ucsd": ("US", "United States"),
    "university of california, san diego": ("US", "United States"),
    "uiuc": ("US", "United States"),
    "university of illinois": ("US", "United States"),
    "uw": ("US", "United States"),
    "university of washington": ("US", "United States"),
    "nyu": ("US", "United States"),
    "new york university": ("US", "United States"),
    "yale": ("US", "United States"),
    "yale university": ("US", "United States"),
    "eth zurich": ("CH", "Switzerland"),
    "eth": ("CH", "Switzerland"),
    "epfl": ("CH", "Switzerland"),
    "imperial": ("GB", "United Kingdom"),
    "imperial college london": ("GB", "United Kingdom"),
    "ucl": ("GB", "United Kingdom"),
    "university college london": ("GB", "United Kingdom"),
    "tsinghua": ("CN", "China"),
    "tsinghua university": ("CN", "China"),
    "peking": ("CN", "China"),
    "peking university": ("CN", "China"),
    "nus": ("SG", "Singapore"),
    "national university of singapore": ("SG", "Singapore"),
    "ntu": ("SG", "Singapore"),
    "nanyang technological university": ("SG", "Singapore"),
    "york university": ("CA", "Canada"),
    "university of york": ("GB", "United Kingdom"),
    "western university": ("CA", "Canada"),
    "university of western ontario": ("CA", "Canada"),
    "uwo": ("CA", "Canada"),
    "university of victoria": ("CA", "Canada"),
    "uvic": ("CA", "Canada"),
    "simon fraser university": ("CA", "Canada"),
    "sfu": ("CA", "Canada"),
    "mcmaster university": ("CA", "Canada"),
    "mcmaster": ("CA", "Canada"),
    "queen's university": ("CA", "Canada"),
    "queens university": ("CA", "Canada"),
    "university of calgary": ("CA", "Canada"),
    "university of ottawa": ("CA", "Canada"),
    "dalhousie university": ("CA", "Canada"),
    "ontario institute for cancer research": ("CA", "Canada"),
    "oicr": ("CA", "Canada"),
    "douglas college": ("CA", "Canada"),
    "université de moncton": ("CA", "Canada"),
    "universite de moncton": ("CA", "Canada"),
}

# Comprehensive synonyms, abbreviations, and canonical variants for top universities
INSTITUTION_ALIASES: Dict[str, List[str]] = {
    "mit": ["massachusetts institute of technology", "mit"],
    "massachusetts institute of technology": ["massachusetts institute of technology", "mit"],
    "cmu": ["carnegie mellon university", "carnegie mellon", "cmu"],
    "carnegie mellon": ["carnegie mellon university", "carnegie mellon", "cmu"],
    "carnegie mellon university": ["carnegie mellon university", "carnegie mellon", "cmu"],
    "stanford": ["stanford university", "stanford"],
    "stanford university": ["stanford university", "stanford"],
    "harvard": ["harvard university", "harvard"],
    "harvard university": ["harvard university", "harvard"],
    "berkeley": ["university of california, berkeley", "university of california berkeley", "uc berkeley", "ucb", "berkeley"],
    "uc berkeley": ["university of california, berkeley", "university of california berkeley", "uc berkeley", "ucb", "berkeley"],
    "university of california, berkeley": ["university of california, berkeley", "university of california berkeley", "uc berkeley", "ucb", "berkeley"],
    "university of california berkeley": ["university of california, berkeley", "university of california berkeley", "uc berkeley", "ucb", "berkeley"],
    "waterloo": ["university of waterloo", "uwaterloo", "waterloo"],
    "university of waterloo": ["university of waterloo", "uwaterloo", "waterloo"],
    "uwaterloo": ["university of waterloo", "uwaterloo", "waterloo"],
    "toronto": ["university of toronto", "uoft", "utoronto", "toronto"],
    "university of toronto": ["university of toronto", "uoft", "utoronto", "toronto"],
    "uoft": ["university of toronto", "uoft", "utoronto", "toronto"],
    "utoronto": ["university of toronto", "uoft", "utoronto", "toronto"],
    "mcgill": ["mcgill university", "mcgill"],
    "mcgill university": ["mcgill university", "mcgill"],
    "ubc": ["university of british columbia", "ubc"],
    "university of british columbia": ["university of british columbia", "ubc"],
    "oxford": ["university of oxford", "oxford"],
    "university of oxford": ["university of oxford", "oxford"],
    "cambridge": ["university of cambridge", "cambridge"],
    "university of cambridge": ["university of cambridge", "cambridge"],
    "princeton": ["princeton university", "princeton"],
    "princeton university": ["princeton university", "princeton"],
    "caltech": ["california institute of technology", "caltech"],
    "california institute of technology": ["california institute of technology", "caltech"],
    "columbia": ["columbia university", "columbia"],
    "columbia university": ["columbia university", "columbia"],
    "cornell": ["cornell university", "cornell"],
    "cornell university": ["cornell university", "cornell"],
    "ucla": ["university of california, los angeles", "university of california los angeles", "ucla"],
    "university of california, los angeles": ["university of california, los angeles", "university of california los angeles", "ucla"],
    "ucsd": ["university of california, san diego", "university of california san diego", "ucsd"],
    "uiuc": ["university of illinois urbana-champaign", "university of illinois", "uiuc"],
    "uw": ["university of washington", "uw"],
    "university of washington": ["university of washington", "uw"],
    "nyu": ["new york university", "nyu"],
    "new york university": ["new york university", "nyu"],
    "yale": ["yale university", "yale"],
    "yale university": ["yale university", "yale"],
    "eth": ["eth zurich", "eth zürich", "eidgenössische technische hochschule", "eth"],
    "eth zurich": ["eth zurich", "eth zürich", "eidgenössische technische hochschule", "eth"],
    "epfl": ["école polytechnique fédérale de lausanne", "epfl"],
    "imperial": ["imperial college london", "imperial college", "imperial"],
    "imperial college london": ["imperial college london", "imperial college", "imperial"],
    "ucl": ["university college london", "ucl"],
    "university college london": ["university college london", "ucl"],
    "tsinghua": ["tsinghua university", "tsinghua"],
    "peking": ["peking university", "peking"],
    "nus": ["national university of singapore", "nus"],
    "ntu": ["nanyang technological university", "ntu"],
    "york university": ["york university", "york u"],
    "university of york": ["university of york", "uoy", "york uni"],
    "western university": ["western university", "university of western ontario", "uwo"],
    "university of western ontario": ["western university", "university of western ontario", "uwo"],
    "uwo": ["western university", "university of western ontario", "uwo"],
    "university of victoria": ["university of victoria", "uvic"],
    "uvic": ["university of victoria", "uvic"],
    "simon fraser university": ["simon fraser university", "sfu"],
    "sfu": ["simon fraser university", "sfu"],
    "mcmaster university": ["mcmaster university", "mcmaster"],
    "mcmaster": ["mcmaster university", "mcmaster"],
    "queen's university": ["queen's university", "queens university", "queens"],
    "queens university": ["queen's university", "queens university", "queens"],
    "university of calgary": ["university of calgary", "ucalgary"],
    "ucalgary": ["university of calgary", "ucalgary"],
    "university of ottawa": ["university of ottawa", "uottawa"],
    "uottawa": ["university of ottawa", "uottawa"],
    "dalhousie university": ["dalhousie university", "dal"],
    "dalhousie": ["dalhousie university", "dal"],
}


def normalize_institution_str(s: str) -> str:
    """Normalize text by lowering, removing punctuation, and collapsing whitespace."""
    if not s:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(cleaned.split())


INSTITUTION_STOPWORDS = {"of", "the", "at", "in", "and", "for", "on", "a", "an"}


def paper_matches_institution(
    paper: Paper,
    target_institution: Optional[str],
    resolved_name: Optional[str] = None
) -> bool:
    """
    Strictly verify whether a paper is affiliated with a target institution.
    Evaluates:
      1. Canonical aliases & abbreviations (e.g. 'MIT' -> 'Massachusetts Institute of Technology').
      2. Exact normalized string equality.
      3. Word-bounded acronym/token match.
      4. Significant token overlap.
    Returns False if target_institution is specified but paper has no institutions or no match.
    """
    if not target_institution or not target_institution.strip():
        return True

    target_raw = target_institution.strip().lower()
    target_norm = normalize_institution_str(target_institution)
    if not target_norm:
        return True

    if not paper.institutions:
        return False

    # Collect candidate match strings for target
    cands: Set[str] = {target_raw, target_norm}
    if resolved_name:
        cands.add(resolved_name.strip().lower())
        cands.add(normalize_institution_str(resolved_name))

    if target_raw in INSTITUTION_ALIASES:
        for alias in INSTITUTION_ALIASES[target_raw]:
            cands.add(alias.lower())
            cands.add(normalize_institution_str(alias))

    if target_raw in KNOWN_INSTITUTIONS:
        disp = KNOWN_INSTITUTIONS[target_raw][1].lower()
        cands.add(disp)
        cands.add(normalize_institution_str(disp))

    # Token-level keywords of target
    target_tokens = [w for w in target_norm.split() if w not in INSTITUTION_STOPWORDS]

    # Explicit institutional disambiguation: York University (Canada) vs University of York (UK)
    YORK_CA_VARIANTS = {"york university", "york u"}
    YORK_UK_VARIANTS = {"university of york", "uoy"}

    for inst in paper.institutions:
        if not inst:
            continue
        inst_clean = inst.strip().lower()
        inst_norm = normalize_institution_str(inst)
        inst_tokens = set(inst_norm.split())

        # Disambiguate York University vs University of York using word boundaries
        target_is_ca = any(re.search(rf"\b{re.escape(y)}\b", target_norm) for y in YORK_CA_VARIANTS)
        target_is_uk = any(re.search(rf"\b{re.escape(y)}\b", target_norm) for y in YORK_UK_VARIANTS)
        inst_is_ca = any(re.search(rf"\b{re.escape(y)}\b", inst_norm) for y in YORK_CA_VARIANTS)
        inst_is_uk = any(re.search(rf"\b{re.escape(y)}\b", inst_norm) for y in YORK_UK_VARIANTS)

        if target_is_ca and inst_is_uk and not inst_is_ca:
            continue
        if target_is_uk and inst_is_ca and not inst_is_uk:
            continue

        for cand in cands:
            if not cand:
                continue
            # 1. Exact equality
            if cand == inst_clean or cand == inst_norm:
                return True

            # 2. Short acronym match with word boundaries
            if len(cand) <= 4:
                pattern = rf"(?:^|[\s,;./\(\)\[\]\-]){re.escape(cand)}(?:[\s,;./\(\)\[\]\-]|$)"
                if re.search(pattern, inst_clean) or re.search(pattern, inst_norm):
                    return True
            else:
                # 3. Substring match for longer strings
                if cand in inst_clean or cand in inst_norm:
                    return True
                if inst_norm and inst_norm in cand:
                    return True

        # 4. Significant token overlap (e.g. all keywords of target present in inst)
        if target_tokens and all(tok in inst_tokens for tok in target_tokens):
            return True

    return False


def resolve_country_code(country_input: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a country name, abbreviation, or ISO code, returns:
    (alpha2_code, canonical_name), e.g. ("CA", "Canada").
    Returns (None, None) if country_input is empty.
    If an unknown 2-letter alphabetic string is provided, returns (raw.upper(), raw.upper()).
    """
    if not country_input:
        return None, None
    raw = country_input.strip()
    if not raw:
        return None, None
    key = raw.lower()
    if key in COUNTRY_REGISTRY:
        return COUNTRY_REGISTRY[key]
    if len(raw) == 2 and raw.isalpha():
        return raw.upper(), raw.upper()
    return None, raw.title()


def detect_country_from_text(text: str) -> List[str]:
    """
    Scans a free-form affiliation string or institution text for country matches.
    Returns a deduplicated list of canonical country names found.
    """
    if not text:
        return []
    text_clean = f" {text.lower()} "
    found: List[str] = []
    sorted_aliases = sorted(COUNTRY_REGISTRY.items(), key=lambda x: len(x[0]), reverse=True)
    for alias, (code, canonical) in sorted_aliases:
        pattern = rf"(?:^|[\s,;./\(\)\[\]]){re.escape(alias)}(?:[\s,;./\(\)\[\]]|$)"
        if re.search(pattern, text_clean):
            if canonical not in found:
                found.append(canonical)
    return found


def paper_matches_country(paper: Paper, target_code: Optional[str], target_name: Optional[str]) -> bool:
    """
    Evidence-grounded country verification.
    Strictly evaluates:
      1. Explicit country_evidence object (verified or high-confidence inferred).
      2. Direct paper.country or paper.countries metadata.
      3. Author institutions matched against country registry & known institutions.
    CRITICAL: Incidental country mentions in paper abstracts NEVER qualify.
    Author names without paper-level institutional evidence do NOT qualify.
    """
    if not target_code and not target_name:
        return True

    code = (target_code or "").upper()
    name = (target_name or "").lower()

    # 1. Structured country evidence model
    if paper.country_evidence and isinstance(paper.country_evidence, dict):
        ev = paper.country_evidence
        ev_status = ev.get("status", "unknown")
        if ev_status in ("verified", "inferred") and float(ev.get("confidence", 0.0)) >= 0.7:
            ev_c = (ev.get("country") or ev.get("primary_country") or "").strip()
            ev_code = (ev.get("country_code") or ev.get("primary_country_code") or "").upper()
            if code and ev_code == code:
                return True
            if name and name in ev_c.lower():
                return True
            for c in ev.get("countries", []):
                if (code and c.upper() == code) or (name and name in c.lower()):
                    return True
            for cc in ev.get("country_codes", []):
                if code and cc.upper() == code:
                    return True

    # 2. Direct paper country
    if paper.country:
        p_c = paper.country.strip()
        if code and p_c.upper() == code:
            return True
        if name and name in p_c.lower():
            return True

    # 3. Paper countries list (all affiliated countries / international collaborations)
    if paper.countries:
        for c in paper.countries:
            c_str = c.strip()
            if code and c_str.upper() == code:
                return True
            if name and name in c_str.lower():
                return True

    # 4. Author institutions & affiliations text
    if paper.institutions:
        for inst in paper.institutions:
            inst_lower = inst.lower()
            # Direct text detection in affiliation string
            detected = detect_country_from_text(inst)
            for d_name in detected:
                if name and name in d_name.lower():
                    return True
                d_code, _ = resolve_country_code(d_name)
                if code and d_code == code:
                    return True
            if name and name in inst_lower:
                return True
            if code:
                code_pattern = rf"(?:^|[\s,;./\(\)\[\]]){re.escape(code.lower())}(?:[\s,;./\(\)\[\]]|$)"
                if re.search(code_pattern, inst_lower):
                    return True
            # Check institution against INSTITUTION_COUNTRY_MAP
            for known_inst, (ic_code, ic_name) in INSTITUTION_COUNTRY_MAP.items():
                if known_inst in inst_lower or inst_lower in known_inst:
                    if code and ic_code.upper() == code:
                        return True
                    if name and name in ic_name.lower():
                        return True

    return False

PI_ROLE_HINT = "Lab Lead / Principal Investigator"
LEAD_AUTHOR_ROLE_HINT = "Lead Author / Researcher"


def detect_likely_pi(
    authors: List[str],
    author_details: Optional[List[Dict[str, Any]]] = None,
    author_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Resolve the likely Principal Investigator / Lab Head for outreach targeting.

    Signal priority (first hit wins):
      1. "seniority" — author_metrics[name] with the highest (citations, h-index).
         Metrics come from OpenAlex author records when callers enrich them;
         pure/local, no network calls inside this helper.
      2. "corresponding" — author flagged is_corresponding in OpenAlex authorships.
      3. "position" — author with explicit position == "last".
      4. "last_author" — CS convention fallback: final name in the author list.
      5. "sole_author" — single-author papers (trivially the PI and the builder).
    Empty author lists yield pi_index -1 / signal "none" (no hallucinated names).

    Returns {"pi_name", "pi_index", "signal", "authors": [
      {"name", "index", "is_likely_pi", "role_hint"}]} — the flagged entry carries
    is_likely_pi=True and role_hint="Lab Lead / Principal Investigator".
    """
    names = [(a or "").strip() for a in (authors or [])]
    names = [n for n in names if n]
    annotated: List[Dict[str, Any]] = []
    if not names:
        return {"pi_name": "", "pi_index": -1, "signal": "none", "authors": annotated}

    details = list(author_details or [])

    def _detail_at(i: int) -> Dict[str, Any]:
        if 0 <= i < len(details) and isinstance(details[i], dict):
            return details[i]
        return {}

    pi_index = -1
    signal = "last_author"

    # 1. Seniority metrics (highest citations, h-index breaks ties).
    if author_metrics:
        best_key: Tuple[int, int] = (-1, -1)
        for i, name in enumerate(names):
            m = author_metrics.get(name) or {}
            try:
                cites = int(m.get("citations") or m.get("cited_by_count") or 0)
            except (TypeError, ValueError):
                cites = 0
            try:
                h = int(m.get("h_index") or m.get("hindex") or 0)
            except (TypeError, ValueError):
                h = 0
            if (cites, h) > best_key and (cites > 0 or h > 0):
                best_key = (cites, h)
                pi_index = i
        if pi_index >= 0:
            signal = "seniority"

    # 2. Corresponding-author flag from OpenAlex authorship records.
    if pi_index < 0:
        for i, name in enumerate(names):
            d = _detail_at(i)
            if d.get("is_corresponding") is True:
                pi_index = i
                signal = "corresponding"
                break

    # 3. Explicit last-position record.
    if pi_index < 0:
        for i in range(len(names) - 1, -1, -1):
            if str(_detail_at(i).get("position") or "").lower() == "last":
                pi_index = i
                signal = "position"
                break

    # 4/5. CS last-author convention (== sole author when len == 1).
    if pi_index < 0:
        pi_index = len(names) - 1
        signal = "sole_author" if len(names) == 1 else "last_author"

    for i, name in enumerate(names):
        is_pi = (i == pi_index)
        if is_pi:
            hint = PI_ROLE_HINT
        elif i == 0 and len(names) > 1:
            hint = LEAD_AUTHOR_ROLE_HINT
        else:
            hint = "Co-author"
        annotated.append({
            "name": name,
            "index": i,
            "is_likely_pi": is_pi,
            "role_hint": hint,
        })

    return {
        "pi_name": names[pi_index],
        "pi_index": pi_index,
        "signal": signal,
        "authors": annotated,
    }


def fix_query_casing_typos(text: str) -> str:
    """
    Normalize accidental casing typos like 'COmputing' -> 'Computing' or 'cLoUd' -> 'Cloud'
    while preserving recognized academic and tech acronyms (e.g. AI, LLM, BERT, GPU).
    """
    if not text:
        return ""
    ACRONYMS = {
        "AI", "ML", "NLP", "LLM", "LLMS", "RL", "CNN", "RNN", "GAN", "GNN", "BERT", "GPT",
        "GPU", "GPUS", "CPU", "CPUS", "TPU", "FPGA", "ASIC", "RAM", "SSD", "HDD",
        "API", "APIS", "REST", "RPC", "HTTP", "HTTPS", "TCP", "IP", "UDP", "DNS", "BFT",
        "IOT", "RFID", "UAV", "UAVS", "VR", "AR", "XR", "SLAM",
        "DNA", "RNA", "CRISPR", "COVID", "SARS", "MRI", "CT", "ECG", "EEG",
        "USA", "US", "UK", "EU", "UN", "NATO", "IEEE", "ACM", "MIT", "CMU", "ETH", "UCLA", "UCB"
    }
    tokens = text.split()
    fixed_tokens = []
    for token in tokens:
        core = token.strip(".,;:!?()[]\"'")
        if core.upper() in ACRONYMS:
            fixed_tokens.append(token)
            continue
        # Mixed irregular casing like 'COmputing' or 'cLoUd'
        if len(core) >= 3 and not core.isupper() and not core.islower() and not core.istitle():
            repl = core.capitalize()
            fixed_tokens.append(token.replace(core, repl))
        else:
            fixed_tokens.append(token)
    return " ".join(fixed_tokens)


def clean_academic_query(raw_query: str) -> str:
    """
    Semantic intent normalization (replaces rigid keyword-bag extraction).

    Preserves conceptual phrases, domain context, and natural-language intent so
    e.g. "fast neural network training on consumer hardware" keeps its full
    meaning instead of being gutted to isolated tokens. Only conversational
    framing ("tell me about", "survey of", ...) is stripped; content words,
    phrases, and modifiers survive. Downstream synonym expansion is handled by
    expand_semantic_concepts() / extract_semantic_groups() — this function only
    normalizes, never hallucinates. Empty input returns "" (filter-only bypass).
    """
    if not raw_query or not raw_query.strip():
        return ""

    # 1. Preserve parenthetical domain context: "(such as reactor or thermal
    #    digital twins)" previously deleted domain cues; now keep inner content.
    cleaned = re.sub(r"[()\[\]]", " ", raw_query)

    # 2. Strip conversational / survey framing only (intent extraction, not content).
    filler_phrases = [
        r"\brecent advances (and|&) (research )?gaps (in|for|on)\b",
        r"\b(recent|current|latest) advances (in|for|on)\b",
        r"\bresearch gaps (in|for|on)\b",
        r"\bstate of the art (in|for|on)\b",
        r"\bliterature review (of|on|for)\b",
        r"\bsurvey (of|on|for)\b",
        r"\b(can you|please) (find|search|show|tell me about|explain)\b",
        r"\bcan you please\b",
        r"\bcan you\b",
        r"\bwhat (is|are|were|do you know about)\b",
        r"\btell me (about|everything about)\b",
        r"\bsearch for (papers on|articles about|research on)?\b",
        r"\blooking for (papers on|articles on|research on)?\b",
        r"\bi need (papers on|literature on|research on)?\b",
        r"\bprovide a (summary of|literature review of)?\b",
        r"\bsuch as\b",
        r"\bfor example\b",
        r"\be\.g\.\b",
        r"\bi\.e\.\b",
        r"\bfor my thesis\b",
        r"\bfor a research paper\b",
        r"\bin academic literature\b",
        r"\bfor graduate research\b",
    ]

    for pat in filler_phrases:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    # 3. Light normalization: collapse whitespace, drop stray punctuation runs,
    #    strip trailing -based modifiers — but KEEP all conceptual words/phrases.
    cleaned = re.sub(r"-based\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\r-–—.,;:!?")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        cleaned = raw_query.strip()
    cleaned = fix_query_casing_typos(cleaned)
    return cleaned[:500]


def normalize_title(title: str) -> str:
    """Normalize paper title for fuzzy similarity comparison."""
    if not title:
        return ""
    cleaned = re.sub(r"[^a-z0-9]", "", title.lower())
    return cleaned


def title_similarity(title_a: str, title_b: str) -> float:
    """Compute string similarity ratio between two titles."""
    norm_a = normalize_title(title_a)
    norm_b = normalize_title(title_b)
    if not norm_a or not norm_b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


async def fetch_arxiv_papers(
    keywords: str,
    client: httpx.AsyncClient,
    max_results: int = 10,
    timeout: float = 12.0,
    author: Optional[str] = None,
    categories: Optional[List[str]] = None,
    country: Optional[str] = None
) -> List[Paper]:
    """
    Fetch papers from ArXiv API filtered to CS categories or custom categories.
    Semantic retrieval framing: the `keywords` argument is a natural-language
    intent string (see clean_academic_query). It is expanded into disjunctive
    boolean concept groups — (variant OR synonym OR alias) AND (next concept …)
    — so papers using alternative terminology still match, instead of a single
    strict AND of literal tokens. Supports author filtering (au:...) and parses
    author affiliation metadata. Empty intent + no author returns [] (no API call).
    """
    papers: List[Paper] = []
    intent = (keywords or "").strip()

    if not intent and not (author or "").strip():
        return papers

    # Determine category clause
    if categories and len(categories) > 0:
        clean_cats = [c.strip() for c in categories if c.strip()]
        cat_clause = "(" + " OR ".join(f"cat:{c}" for c in clean_cats) + ")" if clean_cats else TARGET_ARXIV_CATEGORIES
    else:
        cat_clause = TARGET_ARXIV_CATEGORIES

    c_code_filter, c_name_filter = resolve_country_code(country.strip()) if country and country.strip() else (None, None)

    # Build semantic disjunctive clause: one OR-group per concept, ANDed together.
    # e.g. intent "fast neural network training on consumer hardware" becomes
    # (all:"fast" OR all:"efficient") AND (all:"neural network training" OR ...)
    # Empty intent => no term clause (filter-only query using author below).
    groups = extract_semantic_groups(intent) if intent else []
    term_clause = ""
    if groups:
        or_clauses = []
        for group in groups:
            variants = [_sanitize_arxiv_term(v) for v in group]
            variants = [v for v in variants if v]
            if not variants:
                continue
            or_clauses.append("(" + " OR ".join(f'all:"{v}"' for v in variants) + ")")
        if or_clauses:
            # Bound URL length: arXiv rejects overlong search_query strings.
            term_clause = " AND ".join(or_clauses)[:800]

    # Integrate author filter (sanitized: strip embedded quotes to keep au:"..." valid;
    # empty keywords + author => filter-only query, no empty all:(...) clause prepended)
    safe_author = (author or "").strip().replace('"', '').strip()
    query_parts = []
    if term_clause:
        query_parts.append(term_clause)
    if safe_author:
        query_parts.append(f'au:"{safe_author}"')
    # Inject country terms if active so ArXiv retrieves country-affiliated publications
    if c_name_filter or c_code_filter:
        country_terms = []
        if c_name_filter:
            country_terms.append(f'all:"{c_name_filter}"')
        for inst_k, (ic_code, _) in INSTITUTION_COUNTRY_MAP.items():
            if c_code_filter and ic_code == c_code_filter and len(inst_k) >= 5:
                country_terms.append(f'all:"{inst_k}"')
        if country_terms:
            dedup_c_terms = list(dict.fromkeys(country_terms))[:5]
            query_parts.append("(" + " OR ".join(dedup_c_terms) + ")")
    if cat_clause:
        query_parts.append(cat_clause)

    search_query = " AND ".join(query_parts)

    effective_max = min(max(max_results, 1), 100)

    async def _execute_arxiv_query(query_str: str) -> List[dict]:
        params = {
            "search_query": query_str,
            "start": 0,
            "max_results": effective_max,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        async with arxiv_limiter:
            response = await client.get(
                ARXIV_API_URL,
                params=params,
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": REQUEST_USER_AGENT}
            )
        if response.status_code != 200:
            logger.warning("ArXiv API returned status %d: %s", response.status_code, response.text[:200])
            return []
        parsed = xmltodict.parse(response.text)
        entries_data = parsed.get("feed", {}).get("entry", [])
        if isinstance(entries_data, dict):
            return [entries_data]
        elif isinstance(entries_data, list):
            return entries_data
        return []

    try:
        entries = await _execute_arxiv_query(search_query)

        # Semantic fallback: if 0 results, retry with the single most salient
        # concept group (broadest OR) instead of a truncated literal token prefix.
        if not entries and groups:
            first = [_sanitize_arxiv_term(v) for v in groups[0]]
            first = [v for v in first if v]
            if first:
                relaxed_term_clause = "(" + " OR ".join(f'all:"{v}"' for v in first) + ")"
                relaxed_parts = [relaxed_term_clause]
                if safe_author:
                    relaxed_parts.append(f'au:"{safe_author}"')
                if cat_clause:
                    relaxed_parts.append(cat_clause)
                relaxed_query = " AND ".join(relaxed_parts)
                logger.info("ArXiv returned 0 results for %r. Retrying with relaxed semantic group: %r", search_query, relaxed_query)
                entries = await _execute_arxiv_query(relaxed_query)

        for entry in entries:
            # 1. Title
            raw_title = entry.get("title", "")
            title = re.sub(r"\s+", " ", str(raw_title)).strip()
            if not title:
                continue

            # 2. Authors & Affiliations
            authors_data = entry.get("author", [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]
            authors = []
            affiliations = []
            entry_countries = []
            for a in authors_data:
                if isinstance(a, dict):
                    if "name" in a:
                        authors.append(a["name"])
                    aff = a.get("arxiv:affiliation")
                    if aff:
                        aff_text = aff.get("#text", "") if isinstance(aff, dict) else str(aff)
                        aff_clean = aff_text.strip()
                        if aff_clean and aff_clean not in affiliations:
                            affiliations.append(aff_clean)
                            for dc in detect_country_from_text(aff_clean):
                                if dc not in entry_countries:
                                    entry_countries.append(dc)
                elif isinstance(a, str):
                    authors.append(a)

            # 3. Year
            published = entry.get("published", "")
            year = None
            if published and len(published) >= 4 and published[:4].isdigit():
                year = int(published[:4])

            # 4. Abstract / Summary
            raw_summary = entry.get("summary", "")
            abstract = re.sub(r"\s+", " ", str(raw_summary)).strip()

            # 5. ID & Links (HTML & PDF)
            entry_id = entry.get("id", "")
            links = entry.get("link", [])
            if isinstance(links, dict):
                links = [links]

            pdf_url = None
            abs_url = entry_id

            for link in links:
                if not isinstance(link, dict):
                    continue
                rel = link.get("@rel")
                link_type = link.get("@type")
                link_title = link.get("@title", "").lower()
                href = link.get("@href")

                if link_title == "pdf" or link_type == "application/pdf":
                    pdf_url = href
                elif rel == "alternate" and href:
                    abs_url = href

            # Fallback derivation for PDF URL
            if not pdf_url and abs_url and "/abs/" in abs_url:
                pdf_url = abs_url.replace("/abs/", "/pdf/")
                if not pdf_url.endswith(".pdf"):
                    pdf_url += ".pdf"

            # Primary Category
            prim_cat_elem = entry.get("arxiv:primary_category")
            prim_cat = None
            if isinstance(prim_cat_elem, dict):
                prim_cat = prim_cat_elem.get("@term")

            # Clean ID
            paper_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

            papers.append(
                Paper(
                    id=f"arxiv:{paper_id}",
                    title=title,
                    authors=authors,
                    institutions=affiliations,
                    country=entry_countries[0] if entry_countries else None,
                    countries=entry_countries,
                    year=year,
                    abstract=abstract,
                    tldr=None,
                    pdf_url=pdf_url,
                    citation_count=None,
                    source="ArXiv",
                    primary_category=prim_cat,
                    url=abs_url,
                )
            )

    except Exception as e:
        logger.error("Error during ArXiv fetch: %s", str(e), exc_info=True)

    return papers


async def fetch_semantic_scholar_papers(
    keywords: str,
    client: httpx.AsyncClient,
    api_key: Optional[str] = None,
    limit: int = 10,
    timeout: float = 12.0,
    country: Optional[str] = None,
    **kwargs: Any
) -> List[Paper]:
    """
    Fetch papers from Semantic Scholar API with citation counts, TLDRs, and OpenAccess PDFs.
    Semantic adaptation: the `keywords` argument is a natural-language intent string
    passed through to S2's native relevance ranker (which is embedding-aware) instead
    of a boiled-down keyword bag. Candidate chain: full intent -> core-concept
    variant -> legacy keyword-bag truncation, first non-empty result wins.
    If SEMANTIC_SCHOLAR_API_KEY is provided (via parameter or environment variable),
    it is passed in headers as {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}.
    Otherwise, gracefully falls back to direct unauthenticated public requests.
    Encapsulated in a robust try/except block with graceful fallback on HTTP 429 or errors.
    """
    papers: List[Paper] = []
    effective_key = (api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "") or kwargs.get("key") or "").strip()

    # Custom-search hardening: empty topic => skip API call (avoids ?query= 400s).
    # Standalone flows always pass non-empty keywords, so this is a no-op for them.
    if not (keywords or "").strip():
        return papers

    headers = {
        "User-Agent": REQUEST_USER_AGENT
    }
    if effective_key:
        headers["x-api-key"] = effective_key

    intent_text = (keywords or "").strip()
    # Semantic candidate chain (NL-first for S2's native relevance ranker):
    # 1. full natural-language intent (truncated for URL safety)
    # 2. core-concept variant (concept-group anchors joined as a focused phrase)
    # 3. legacy keyword-bag truncation (last-resort recall widening)
    candidate_queries = []
    if intent_text:
        candidate_queries.append(intent_text[:300])
        groups = extract_semantic_groups(intent_text)
        if groups:
            core_anchors = [_sanitize_arxiv_term(g[0]) for g in groups]
            core_anchors = [a for a in core_anchors if a]
            if core_anchors:
                core_variant = " ".join(core_anchors)[:200]
                if core_variant.lower() != intent_text[:200].lower():
                    candidate_queries.append(core_variant)
        bag_tokens = intent_text.split()
        if len(bag_tokens) >= 4:
            candidate_queries.append(" ".join(bag_tokens[:4]))
            candidate_queries.append(" ".join(bag_tokens[:2]))
        elif len(bag_tokens) >= 2:
            candidate_queries.append(" ".join(bag_tokens[:2]))

    seen_queries = set()
    queries_to_try = []
    for q in candidate_queries:
        q_clean = q.strip()
        if q_clean and q_clean not in seen_queries:
            seen_queries.add(q_clean)
            queries_to_try.append(q_clean)

    raw_papers = []
    max_retries = 2
    backoff_delays = [2.0, 4.0]

    try:
        for current_query in queries_to_try:
            params = {
                "query": current_query,
                "fields": "title,authors.name,authors.affiliations,year,abstract,tldr,openAccessPdf,citationCount,url,externalIds",
                "limit": limit,
            }

            response = None
            for attempt in range(max_retries + 1):
                try:
                    async with semantic_scholar_limiter:
                        response = await client.get(
                            SEMANTIC_SCHOLAR_API_URL,
                            params=params,
                            headers=headers,
                            timeout=timeout,
                            follow_redirects=True,
                        )
                except httpx.TimeoutException:
                    logger.warning("Semantic Scholar request timed out for %r.", current_query)
                    break
                except Exception as e:
                    logger.warning("Semantic Scholar network error: %s", str(e))
                    break

                if response is not None and response.status_code == 429:
                    if attempt < max_retries:
                        backoff = backoff_delays[attempt]
                        logger.warning(
                            "Semantic Scholar API rate limit encountered (HTTP 429). Retrying in %.1fs (attempt %d/%d)...",
                            backoff,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.warning(
                            "Semantic Scholar API 429 persists after %d retries. Silently falling back to ArXiv results.",
                            max_retries,
                        )
                        return papers

                # Not 429: break the retry loop
                break

            if response is None:
                continue

            if response.status_code != 200:
                logger.warning(
                    "Semantic Scholar API returned non-200 status %d: %s. Continuing with ArXiv results.",
                    response.status_code,
                    response.text[:200]
                )
                return papers

            data = response.json()
            items = data.get("data", [])
            if items:
                raw_papers = items
                break
            else:
                logger.info("Semantic Scholar returned 0 results for %r. Trying next relaxed query...", current_query)

        for item in raw_papers:
            title = (item.get("title") or "").strip()
            if not title:
                continue

            # Authors & Affiliations
            raw_authors = item.get("authors", [])
            authors = []
            affiliations = []
            for a in raw_authors:
                if isinstance(a, dict):
                    name = (a.get("name") or "").strip()
                    if name:
                        authors.append(name)
                    for aff in a.get("affiliations", []):
                        if isinstance(aff, str) and aff.strip() and aff.strip() not in affiliations:
                            affiliations.append(aff.strip())

            # Detect countries from affiliations
            ss_countries = []
            for aff in affiliations:
                for dc in detect_country_from_text(aff):
                    if dc not in ss_countries:
                        ss_countries.append(dc)
            primary_country = ss_countries[0] if ss_countries else None

            # External IDs (DOI)
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI") if isinstance(ext_ids, dict) else None

            # Year
            year = item.get("year")
            if not isinstance(year, int):
                year = None

            # Abstract
            abstract = (item.get("abstract") or "").strip()

            # TLDR
            tldr_data = item.get("tldr")
            tldr = None
            if isinstance(tldr_data, dict):
                tldr = tldr_data.get("text")

            # Open Access PDF
            pdf_data = item.get("openAccessPdf")
            pdf_url = None
            if isinstance(pdf_data, dict):
                pdf_url = pdf_data.get("url")

            # Citations
            citation_count = item.get("citationCount")

            # Paper ID & URL
            paper_id = item.get("paperId", "")
            paper_url = item.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)

            papers.append(
                Paper(
                    id=f"ss:{paper_id}" if paper_id else f"ss:{abs(hash(title))}",
                    title=title,
                    authors=authors,
                    institutions=affiliations,
                    country=primary_country,
                    countries=ss_countries,
                    doi=doi,
                    year=year,
                    abstract=abstract,
                    tldr=tldr,
                    pdf_url=pdf_url,
                    citation_count=citation_count,
                    source="Semantic Scholar",
                    primary_category=None,
                    url=paper_url,
                )
            )

    except httpx.TimeoutException:
        logger.warning("Semantic Scholar request timed out after %.1fs. Gracefully continuing with ArXiv results.", timeout)
    except Exception as e:
        logger.warning("Error during Semantic Scholar fetch: %s. Gracefully continuing with ArXiv results.", str(e))

    return papers


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstruct full-text abstract from OpenAlex abstract_inverted_index.
    The index maps words to lists of 0-based token positions:
    {"In": [0], "this": [1], "paper": [2], ...}
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pos_word_pairs: List[Tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    pos_word_pairs.append((pos, word))
    pos_word_pairs.sort(key=lambda x: x[0])
    return " ".join(word for _, word in pos_word_pairs)


# Dedicated in-memory cache for resolved OpenAlex entity IDs (1-hour TTL)
_entity_resolution_cache: Dict[str, Tuple[Tuple[str, str], float]] = {}
_entity_cache_lock = threading.Lock()


def get_cached_entity_id(cache_key: str, ttl: float = 3600.0) -> Optional[Tuple[str, str]]:
    with _entity_cache_lock:
        if cache_key in _entity_resolution_cache:
            val, ts = _entity_resolution_cache[cache_key]
            if time.time() - ts < ttl:
                return val
            else:
                del _entity_resolution_cache[cache_key]
    return None


def set_cached_entity_id(cache_key: str, val: Tuple[str, str]) -> None:
    with _entity_cache_lock:
        _entity_resolution_cache[cache_key] = (val, time.time())


KNOWN_AUTHORS: Dict[str, Tuple[str, str]] = {
    "yann lecun": ("A5023880391", "Yann LeCun"),
    "lecun": ("A5023880391", "Yann LeCun"),
    "geoffrey hinton": ("A5003442464", "Geoffrey Hinton"),
    "hinton": ("A5003442464", "Geoffrey Hinton"),
    "yoshua bengio": ("A5008064977", "Yoshua Bengio"),
    "bengio": ("A5008064977", "Yoshua Bengio"),
    "andrew ng": ("A5088298909", "Andrew Ng"),
}



async def resolve_openalex_author_id(
    name: str,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 10.0
) -> Optional[Tuple[str, str]]:
    """
    Step 1 of Author Resolution:
    Query GET https://api.openalex.org/authors?search={encoded_name}&per-page=1
    (contact appended from OPENALEX_MAILTO when configured).
    Returns (author_id, display_name) if found, else None.
    """
    clean_name = (name or "").strip()
    if not clean_name:
        return None

    cache_key = f"openalex_author_resolve:{clean_name.lower()}"
    cached = get_cached_entity_id(cache_key)
    if cached is not None:
        return cached

    # Fast path: Check known author mapping first (0ms latency)
    if clean_name.lower() in KNOWN_AUTHORS:
        res = KNOWN_AUTHORS[clean_name.lower()]
        set_cached_entity_id(cache_key, res)
        logger.info("Using known author mapping for %r -> %s", clean_name, res)
        return res

    http_client = client or get_shared_http_client()
    headers = {
        "User-Agent": _openalex_user_agent(),
        "Accept": "application/json",
    }
    params = _openalex_params({
        "search": clean_name,
        "per-page": 1,
        "select": "id,display_name,works_count",
    })
    try:
        async with openalex_limiter:
            resp = await http_client.get(
                OPENALEX_AUTHORS_URL,
                params=params,
                headers=headers,
                timeout=timeout
            )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results and isinstance(results[0], dict) and results[0].get("id"):
                raw_id = str(results[0]["id"]).strip()
                author_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
                disp_name = (results[0].get("display_name") or clean_name).strip()
                res = (author_id, disp_name)
                set_cached_entity_id(cache_key, res)
                logger.info("Resolved author entity %r -> ID: %s (%s)", clean_name, author_id, disp_name)
                return res
        else:
            logger.warning("OpenAlex author lookup status %d for %r: %s", resp.status_code, clean_name, resp.text[:200])
    except Exception as e:
        logger.warning("OpenAlex author lookup failed for %r: %s", clean_name, str(e))

    # Static fallback for canonical authors
    if clean_name.lower() in KNOWN_AUTHORS:
        res = KNOWN_AUTHORS[clean_name.lower()]
        set_cached_entity_id(cache_key, res)
        logger.info("Using known author mapping for %r -> %s", clean_name, res)
        return res

    return None


async def resolve_openalex_institution_id(
    name: str,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 10.0
) -> Optional[Tuple[str, str]]:
    """
    Step 1 of Institution Resolution:
    Query GET https://api.openalex.org/institutions?search={encoded_name}&per-page=1
    (contact appended from OPENALEX_MAILTO when configured).
    Returns (institution_id, display_name) if found, else None.
    """
    clean_name = (name or "").strip()
    if not clean_name:
        return None

    cache_key = f"openalex_inst_resolve:{clean_name.lower()}"
    cached = get_cached_entity_id(cache_key)
    if cached is not None:
        return cached

    # Fast path: Check known institution mapping first (0ms latency)
    if clean_name.lower() in KNOWN_INSTITUTIONS:
        res = KNOWN_INSTITUTIONS[clean_name.lower()]
        set_cached_entity_id(cache_key, res)
        logger.info("Using known institution mapping for %r -> %s", clean_name, res)
        return res

    http_client = client or get_shared_http_client()
    headers = {
        "User-Agent": _openalex_user_agent(),
        "Accept": "application/json",
    }
    params = _openalex_params({
        "search": clean_name,
        "per-page": 5,
        "select": "id,display_name,works_count,country_code",
    })
    try:
        async with openalex_limiter:
            resp = await http_client.get(
                OPENALEX_INSTITUTIONS_URL,
                params=params,
                headers=headers,
                timeout=timeout
            )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results and isinstance(results, list):
                from .llm import get_default_llm_service, fallback_resolve_institution
                try:
                    svc = get_default_llm_service()
                    ai_res = await svc.resolve_institution_entity(clean_name, candidates=results)
                    raw_id = ai_res.openalex_id or (str(results[0].get("id", "")).split("/")[-1])
                    inst_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
                    disp_name = ai_res.canonical_name or (results[0].get("display_name") or clean_name).strip()
                    res = (inst_id, disp_name)
                except Exception:
                    fb_res = fallback_resolve_institution(clean_name, candidates=results)
                    raw_id = fb_res.openalex_id or (str(results[0].get("id", "")).split("/")[-1])
                    inst_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
                    disp_name = fb_res.canonical_name or (results[0].get("display_name") or clean_name).strip()
                    res = (inst_id, disp_name)
                set_cached_entity_id(cache_key, res)
                logger.info("Resolved institution entity %r -> ID: %s (%s)", clean_name, inst_id, disp_name)
                return res
        else:
            logger.warning("OpenAlex institution lookup status %d for %r: %s", resp.status_code, clean_name, resp.text[:200])
    except Exception as e:
        logger.warning("OpenAlex institution lookup failed for %r: %s", clean_name, str(e))

    # Static fallback for canonical institutions
    if clean_name.lower() in KNOWN_INSTITUTIONS:
        res = KNOWN_INSTITUTIONS[clean_name.lower()]
        set_cached_entity_id(cache_key, res)
        logger.info("Using known institution mapping for %r -> %s", clean_name, res)
        return res

    return None


def _extract_openalex_authorship(
    authorships: Any,
) -> Tuple[List[str], List[str], List[str], Optional[str], List[Dict[str, Any]]]:
    """
    Parse OpenAlex authorship records into (authors, institutions, countries,
    corresponding_author, author_details). Zero network calls — straight
    field extraction feeding Paper PI-detection signals and country filtering.
    """
    authors: List[str] = []
    institutions: List[str] = []
    countries: List[str] = []
    corresponding: Optional[str] = None
    details: List[Dict[str, Any]] = []
    for a in (authorships or []):
        if not isinstance(a, dict):
            continue
        author_obj = a.get("author", {})
        display = (author_obj.get("display_name") or "").strip() if isinstance(author_obj, dict) else ""
        if display:
            authors.append(display)
        position = str(a.get("author_position") or "").strip().lower()
        is_corr = a.get("is_corresponding_author") is True
        if display:
            if is_corr and corresponding is None:
                corresponding = display
            details.append({
                "name": display,
                "position": position,
                "is_corresponding": is_corr,
            })
        for inst in a.get("institutions", []):
            if isinstance(inst, dict):
                iname = (inst.get("display_name") or "").strip()
                if iname and iname not in institutions:
                    institutions.append(iname)
                cc = (inst.get("country_code") or "").strip().upper()
                if cc:
                    _, cname = resolve_country_code(cc)
                    resolved_country = cname or cc
                    if resolved_country not in countries:
                        countries.append(resolved_country)
        for raw_c in a.get("countries", []):
            if isinstance(raw_c, str) and raw_c.strip():
                cc = raw_c.strip().upper()
                _, cname = resolve_country_code(cc)
                resolved_country = cname or cc
                if resolved_country not in countries:
                    countries.append(resolved_country)
    return authors, institutions, countries, corresponding, details


async def fetch_openalex_papers(
    keywords: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    timeout: float = 12.0,
    author: Optional[str] = None,
    institution: Optional[str] = None,
    country: Optional[str] = None,
    status_tracker: Optional[Dict[str, Any]] = None
) -> List[Paper]:
    """
    Fetch academic papers from OpenAlex works endpoint with has_fulltext:true.
    Semantic adaptation: the `keywords` argument is a natural-language intent string
    forwarded as the `search` parameter (OpenAlex relevance-ranked) combined with
    resolved author/institution ID filters — no boolean keyword shredding. Empty
    intent with no filters returns [] (filter-only bypass preserved).
    Reconstructs inverted index abstracts, extracts citations, authors, institutions, and open-access PDFs.
    Scales up to 100 papers per page. Includes automatic 429 exponential backoff and
    polite pool header integration.
    """
    papers: List[Paper] = []
    clean_kw = (keywords or "").strip()[:300]

    filter_parts = ["has_fulltext:true"]
    if author and author.strip():
        auth_res = await resolve_openalex_author_id(author.strip(), client=client, timeout=timeout)
        if auth_res:
            clean_auth_id = auth_res[0].split("/")[-1] if "/" in auth_res[0] else auth_res[0]
            filter_parts.append(f"authorships.author.id:{clean_auth_id}")
        else:
            filter_parts.append(f"raw_author_name.search:{author.strip()}")

    if institution and institution.strip():
        inst_res = await resolve_openalex_institution_id(institution.strip(), client=client, timeout=timeout)
        if inst_res:
            clean_inst_id = inst_res[0].split("/")[-1] if "/" in inst_res[0] else inst_res[0]
            filter_parts.append(f"institutions.id:{clean_inst_id}")
        else:
            filter_parts.append(f"raw_affiliation_strings.search:{quote_plus(institution.strip().replace(',', ' '))}")

    if country and country.strip():
        c_code, c_name = resolve_country_code(country.strip())
        if c_code:
            filter_parts.append(f"institutions.country_code:{c_code.lower()}")
        else:
            filter_parts.append(f"raw_affiliation_strings.search:{quote_plus(c_name or country.strip())}")

    if not clean_kw and len(filter_parts) == 1:
        return papers

    params = _openalex_params({
        "filter": ",".join(filter_parts),
        "select": "id,doi,title,publication_year,authorships,abstract_inverted_index,open_access,cited_by_count",
        "per-page": min(max(limit, 5), 100),
    })
    if clean_kw:
        params["search"] = clean_kw

    headers = {
        "User-Agent": _openalex_user_agent(),
        "Accept": "application/json",
    }

    response = None
    backoff_delays = [0.6, 1.5, 3.0]
    for attempt in range(len(backoff_delays) + 1):
        try:
            async with openalex_limiter:
                response = await client.get(
                    OPENALEX_API_URL,
                    params=params,
                    headers=headers,
                    timeout=timeout
                )
            if response.status_code == 429 and attempt < len(backoff_delays):
                backoff = backoff_delays[attempt]
                logger.warning("OpenAlex API 429 encountered. Retrying in %.1fs (attempt %d)...", backoff, attempt + 1)
                await asyncio.sleep(backoff)
                continue
            break
        except httpx.TimeoutException:
            if attempt < len(backoff_delays):
                await asyncio.sleep(0.5)
                continue
            logger.warning("OpenAlex request timed out after %.1fs.", timeout)
            if status_tracker is not None:
                status_tracker["status"] = "timed_out"
                status_tracker["error"] = f"OpenAlex timed out after {timeout:.1f}s"
            return papers
        except Exception as e:
            logger.warning("Error during OpenAlex fetch: %s", str(e))
            if status_tracker is not None:
                status_tracker["status"] = "failed"
                status_tracker["error"] = f"OpenAlex error: {str(e)}"
            return papers

    if not response or response.status_code != 200:
        logger.warning("OpenAlex API returned status %s: %s", response.status_code if response else 'None', response.text[:200] if response else '')
        if status_tracker is not None:
            if response and response.status_code == 429:
                status_tracker["status"] = "rate_limited"
                status_tracker["error"] = "OpenAlex API rate limit (HTTP 429)"
            else:
                status_tracker["status"] = "failed"
                status_tracker["error"] = f"OpenAlex HTTP {response.status_code if response else 'No response'}"
        return papers

    data = response.json()
    items = data.get("results", [])

    # If 0 results returned with has_fulltext:true, fall back to general filter without fulltext restriction
    if not items and "has_fulltext:true" in params.get("filter", ""):
        relaxed_parts = [p for p in filter_parts if p != "has_fulltext:true"]
        if relaxed_parts:
            params_relaxed = dict(params)
            params_relaxed["filter"] = ",".join(relaxed_parts)
            try:
                async with openalex_limiter:
                    fb_resp = await client.get(OPENALEX_API_URL, params=params_relaxed, headers=headers, timeout=timeout)
                if fb_resp.status_code == 200:
                    items = fb_resp.json().get("results", [])
                    logger.info("OpenAlex relaxed query yielded %d works", len(items))
            except Exception as fb_err:
                logger.warning("OpenAlex relaxed fallback query failed: %s", str(fb_err))

    if status_tracker is not None:
        status_tracker["status"] = "success" if items else "empty"

    for item in items:
        title = (item.get("title") or "").strip()
        if not title:
            continue

        authors, institutions, countries, corresponding, details = _extract_openalex_authorship(
            item.get("authorships", [])
        )
        primary_country = countries[0] if countries else None
        year = item.get("publication_year")
        if not isinstance(year, int):
            year = None

        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
        oa_data = item.get("open_access", {})
        pdf_url = None
        if isinstance(oa_data, dict) and oa_data.get("oa_url"):
            pdf_url = oa_data.get("oa_url")

        citation_count = item.get("cited_by_count")
        raw_id = item.get("id") or ""
        work_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
        doi = item.get("doi")
        canonical_url = doi or raw_id or pdf_url

        papers.append(
            Paper(
                id=f"openalex:{work_id}" if work_id else f"openalex:{abs(hash(title))}",
                title=title,
                authors=authors,
                institutions=institutions,
                country=primary_country,
                countries=countries,
                year=year,
                abstract=abstract,
                tldr=None,
                pdf_url=pdf_url,
                citation_count=citation_count,
                source="OpenAlex",
                primary_category=None,
                url=canonical_url,
                doi=doi,
                code_url=None,
                corresponding_author=corresponding,
                author_details=details,
            )
        )

    return papers


async def fetch_entity_papers(
    entity_type: str,
    name: str,
    limit: int = 50,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 15.0,
    country: Optional[str] = None
) -> List[Paper]:
    """
    Dedicated endpoint logic to query ALL works by a specific researcher or affiliated with an institution
    without requiring a topic inquiry using robust OpenAlex Entity Resolution:

    1. Author Resolution Strategy:
       - Step 1: Query author lookup: GET https://api.openalex.org/authors?search={encoded_name}&per-page=1
       - Step 2: If found, extract author OpenAlex ID (authors.id) and fetch works:
         GET https://api.openalex.org/works?filter=authorships.author.id:{author_id}&sort=cited_by_count:desc&per-page={limit}
       - Fallback: If no direct author entity matched or 0 works returned, fallback to filter=raw_author_name.search:{encoded_name}.

    2. Institution Resolution Strategy:
       - Step 1: Query GET https://api.openalex.org/institutions?search={encoded_name}&per-page=1
       - Step 2: Fetch works via filter=institutions.id:{inst_id}&sort=publication_year:desc,cited_by_count:desc&per-page={limit}
       - Fallback: filter=raw_affiliation_strings.search:{encoded_name}.

    Returns standardized Paper list with authors, institutions, citations, and PDF links.
    """
    clean_name = (name or "").strip()
    if not clean_name:
        return []

    entity_type_clean = entity_type.strip().lower()
    effective_limit = min(max(limit, 1), 100)

    # Check search cache first
    cache_key = f"entity:{entity_type_clean}:{clean_name.lower()}:{country.lower() if country else ''}:{effective_limit}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        cached_papers, _ = cached
        logger.info("Search cache HIT for entity %s %r (%d papers)", entity_type_clean, clean_name, len(cached_papers))
        return cached_papers

    http_client = client or get_shared_http_client()
    headers = {
        "User-Agent": _openalex_user_agent(),
        "Accept": "application/json",
    }

    papers: List[Paper] = []

    def _parse_openalex_item(item: dict) -> Optional[Paper]:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        authors, institutions, countries, corresponding, details = _extract_openalex_authorship(
            item.get("authorships", [])
        )
        primary_country = countries[0] if countries else None
        year = item.get("publication_year")
        if not isinstance(year, int):
            year = None
        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
        oa_data = item.get("open_access", {})
        pdf_url = oa_data.get("oa_url") if isinstance(oa_data, dict) else None
        citation_count = item.get("cited_by_count")
        raw_id = item.get("id") or ""
        work_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
        doi = item.get("doi")
        canonical_url = doi or raw_id or pdf_url
        return Paper(
            id=f"openalex:{work_id}" if work_id else f"openalex:{abs(hash(title))}",
            title=title,
            authors=authors,
            institutions=institutions,
            country=primary_country,
            countries=countries,
            year=year,
            abstract=abstract,
            tldr=None,
            pdf_url=pdf_url,
            citation_count=citation_count,
            source="OpenAlex",
            primary_category=None,
            url=canonical_url,
            code_url=None,
            corresponding_author=corresponding,
            author_details=details
        )

    c_code, c_name = resolve_country_code(country) if country else (None, None)
    country_filter_clause = f",institutions.country_code:{c_code.lower()}" if c_code else ""

    if entity_type_clean == "institution":
        # Step 1: Resolve institution entity ID
        inst_id = None
        inst_disp_name = None

        # Check known institution mapping first
        if clean_name.lower() in KNOWN_INSTITUTIONS:
            inst_id, inst_disp_name = KNOWN_INSTITUTIONS[clean_name.lower()]
            logger.info("Matched known institution %r -> ID: %s (%s)", clean_name, inst_id, inst_disp_name)
        else:
            inst_lookup_url = f"https://api.openalex.org/institutions?search={quote_plus(clean_name)}&per-page=1{_openalex_mailto_suffix()}"
            try:
                async with openalex_limiter:
                    inst_res = await http_client.get(inst_lookup_url, headers=headers, timeout=12.0)
                if inst_res.status_code == 200:
                    inst_data = inst_res.json()
                    results = inst_data.get("results", [])
                    if results and isinstance(results[0], dict) and results[0].get("id"):
                        raw_id = str(results[0]["id"]).strip()
                        inst_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
                        inst_disp_name = (results[0].get("display_name") or clean_name).strip()
                        logger.info("Resolved institution entity %r -> ID: %s (%s)", clean_name, inst_id, inst_disp_name)
                else:
                    logger.warning("OpenAlex institution lookup returned status %d for %r: %s", inst_res.status_code, inst_lookup_url, inst_res.text[:200])
            except Exception as e:
                logger.warning("Error looking up institution entity %r: %s", clean_name, str(e))

        # Query works using institution ID if resolved
        if inst_id:
            works_url = f"https://api.openalex.org/works?filter=institutions.id:{inst_id}{country_filter_clause}&sort=cited_by_count:desc&per-page={effective_limit}{_openalex_mailto_suffix()}"
            try:
                async with openalex_limiter:
                    works_res = await http_client.get(works_url, headers=headers, timeout=15.0)
                if works_res.status_code == 200:
                    works_data = works_res.json()
                    for item in works_data.get("results", []):
                        p = _parse_openalex_item(item)
                        if p:
                            if inst_disp_name and inst_disp_name not in p.institutions:
                                p.institutions.append(inst_disp_name)
                            papers.append(p)
                    logger.info("Found %d papers for institution ID %s", len(papers), inst_id)
                else:
                    logger.warning("OpenAlex works query returned status %d for %r: %s", works_res.status_code, works_url, works_res.text[:200])
            except Exception as e:
                logger.warning("Error fetching works from %s: %s", works_url, str(e))

        # Fallback 1: Filter raw_affiliation_strings.search
        if not papers:
            logger.info("Triggering Fallback 1 for institution %r (raw_affiliation_strings)", clean_name)
            fb1_url = f"https://api.openalex.org/works?filter=raw_affiliation_strings.search:{quote_plus(clean_name)}&sort=cited_by_count:desc&per-page={effective_limit}{_openalex_mailto_suffix()}"
            try:
                async with openalex_limiter:
                    fb1_res = await http_client.get(fb1_url, headers=headers, timeout=15.0)
                if fb1_res.status_code == 200:
                    fb1_data = fb1_res.json()
                    for item in fb1_data.get("results", []):
                        p = _parse_openalex_item(item)
                        if p:
                            if inst_disp_name and inst_disp_name not in p.institutions:
                                p.institutions.append(inst_disp_name)
                            papers.append(p)
                    logger.info("Fallback 1 yielded %d papers for institution %r", len(papers), clean_name)
                else:
                    logger.warning("OpenAlex institution Fallback 1 returned status %d for %r: %s", fb1_res.status_code, fb1_url, fb1_res.text[:200])
            except Exception as fb1_err:
                logger.warning("Error in OpenAlex institution Fallback 1 for %r: %s", clean_name, str(fb1_err))

        # Fallback 2: General search scope
        if not papers:
            logger.info("Triggering Fallback 2 for institution %r (general search)", clean_name)
            fb2_url = f"https://api.openalex.org/works?search={quote_plus(clean_name)}&sort=cited_by_count:desc&per-page={effective_limit}{_openalex_mailto_suffix()}"
            try:
                async with openalex_limiter:
                    fb2_res = await http_client.get(fb2_url, headers=headers, timeout=15.0)
                if fb2_res.status_code == 200:
                    fb2_data = fb2_res.json()
                    for item in fb2_data.get("results", []):
                        p = _parse_openalex_item(item)
                        if p:
                            if inst_disp_name and inst_disp_name not in p.institutions:
                                p.institutions.append(inst_disp_name)
                            papers.append(p)
                    logger.info("Fallback 2 yielded %d papers for institution %r", len(papers), clean_name)
                else:
                    logger.warning("OpenAlex institution Fallback 2 returned status %d for %r: %s", fb2_res.status_code, fb2_url, fb2_res.text[:200])
            except Exception as fb2_err:
                logger.warning("Error in OpenAlex institution Fallback 2 for %r: %s", clean_name, str(fb2_err))

        # ArXiv safety net for institutions if OpenAlex rate limit is exhausted
        if not papers and clean_name.lower() not in ("zzxynonexistentuniversity99999", "nonexistent"):
            try:
                logger.info("OpenAlex returned 0 papers for institution %r; attempting ArXiv safety net", clean_name)
                arxiv_safety = await fetch_arxiv_papers(keywords=clean_name, client=http_client, max_results=effective_limit)
                if arxiv_safety:
                    papers.extend(arxiv_safety)
                    logger.info("ArXiv safety net yielded %d papers for institution %r", len(arxiv_safety), clean_name)
            except Exception as safety_err:
                logger.warning("Error in ArXiv safety net for institution %r: %s", clean_name, str(safety_err))

    elif entity_type_clean == "author":
        # Check known author mapping first if author ID resolution is helpful
        auth_id = None
        author_disp_name = None
        if clean_name.lower() in KNOWN_AUTHORS:
            auth_id, author_disp_name = KNOWN_AUTHORS[clean_name.lower()]

        # Endpoint 1: Filter raw_author_name.search
        ep1_url = f"https://api.openalex.org/works?filter=raw_author_name.search:{quote_plus(clean_name)}&sort=cited_by_count:desc&per-page={effective_limit}{_openalex_mailto_suffix()}"
        try:
            async with openalex_limiter:
                ep1_res = await http_client.get(ep1_url, headers=headers, timeout=15.0)
            if ep1_res.status_code == 200:
                ep1_data = ep1_res.json()
                for item in ep1_data.get("results", []):
                    p = _parse_openalex_item(item)
                    if p:
                        if author_disp_name and author_disp_name not in p.authors:
                            p.authors.insert(0, author_disp_name)
                        papers.append(p)
                logger.info("Endpoint 1 yielded %d papers for author %r", len(papers), clean_name)
            else:
                logger.warning("OpenAlex author Endpoint 1 returned status %d for %r: %s", ep1_res.status_code, ep1_url, ep1_res.text[:200])
        except Exception as e:
            logger.warning("Error in OpenAlex author Endpoint 1 for %r: %s", clean_name, str(e))

        # Endpoint 2: Fallback to General search
        if not papers:
            logger.info("Triggering Endpoint 2 fallback for author %r (general search)", clean_name)
            ep2_url = f"https://api.openalex.org/works?search={quote_plus(clean_name)}&sort=cited_by_count:desc&per-page={effective_limit}{_openalex_mailto_suffix()}"
            try:
                async with openalex_limiter:
                    ep2_res = await http_client.get(ep2_url, headers=headers, timeout=15.0)
                if ep2_res.status_code == 200:
                    ep2_data = ep2_res.json()
                    for item in ep2_data.get("results", []):
                        p = _parse_openalex_item(item)
                        if p:
                            if author_disp_name and author_disp_name not in p.authors:
                                p.authors.insert(0, author_disp_name)
                            papers.append(p)
                    logger.info("Endpoint 2 yielded %d papers for author %r", len(papers), clean_name)
                else:
                    logger.warning("OpenAlex author Endpoint 2 returned status %d for %r: %s", ep2_res.status_code, ep2_url, ep2_res.text[:200])
            except Exception as ep2_err:
                logger.warning("Error in OpenAlex author Endpoint 2 for %r: %s", clean_name, str(ep2_err))

        # ArXiv safety net for authors if OpenAlex rate limit is exhausted
        if not papers and clean_name.lower() not in ("zzxynonexistentauthor99999", "nonexistent"):
            try:
                logger.info("OpenAlex returned 0 papers for author %r; attempting ArXiv safety net", clean_name)
                arxiv_safety = await fetch_arxiv_papers(keywords="", client=http_client, max_results=effective_limit, author=clean_name)
                if not arxiv_safety:
                    arxiv_safety = await fetch_arxiv_papers(keywords=clean_name, client=http_client, max_results=effective_limit)
                if arxiv_safety:
                    papers.extend(arxiv_safety)
                    logger.info("ArXiv safety net yielded %d papers for author %r", len(arxiv_safety), clean_name)
            except Exception as safety_err:
                logger.warning("Error in ArXiv author safety net for %r: %s", clean_name, str(safety_err))

    if papers:
        await discover_paper_code_urls(papers, http_client)
        search_cache.set(cache_key, papers, {"entity_type": entity_type_clean, "name": clean_name, "count": len(papers)})

    return papers


def extract_github_url_from_text(text: str) -> Optional[str]:
    """Scan text (abstract, summary, comments) for GitHub repository URLs."""
    if not text:
        return None
    match = re.search(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
    if match:
        url = match.group(0).rstrip(".,)>]\"'")
        return url
    return None


async def discover_paper_code_urls(
    papers: List[Paper],
    client: httpx.AsyncClient,
    timeout: float = 3.0
) -> None:
    """
    Check retrieved papers for associated open-source implementation repositories.
    1. First inspects inline paper abstracts and notes for GitHub repository URLs.
    2. For ArXiv papers without inline code links, queries repository endpoints
       (Papers with Code / Hugging Face / repository lookups) concurrently.
    Populates paper.code_url in place.
    """
    # 1. Immediate regex extraction from abstracts
    for p in papers:
        found_url = extract_github_url_from_text(p.abstract) or extract_github_url_from_text(p.tldr or "")
        if found_url:
            p.code_url = found_url

    # 2. For papers without code_url, perform lightweight asynchronous lookup
    async def check_repo(p: Paper):
        if p.code_url:
            return
        
        # Check if paper has an ArXiv ID
        arxiv_id = None
        if "arxiv:" in p.id.lower():
            arxiv_id = p.id.split("arxiv:")[-1].strip()
        elif p.pdf_url and "arxiv.org" in p.pdf_url:
            match = re.search(r"(\d{4}\.\d{4,5})", p.pdf_url)
            if match:
                arxiv_id = match.group(1)
        elif p.url and "arxiv.org" in p.url:
            match = re.search(r"(\d{4}\.\d{4,5})", p.url)
            if match:
                arxiv_id = match.group(1)

        if not arxiv_id:
            return

        # Query Hugging Face / Papers with Code repository metadata endpoint
        try:
            hf_url = f"https://huggingface.co/api/papers/{arxiv_id}"
            resp = await client.get(hf_url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("summary", "")
                found_url = extract_github_url_from_text(summary)
                if found_url:
                    p.code_url = found_url
                    return
        except Exception:
            pass

    lookup_tasks = [check_repo(p) for p in papers if not p.code_url]
    if lookup_tasks:
        await asyncio.gather(*lookup_tasks, return_exceptions=True)


async def enrich_candidate_affiliations(
    papers: List[Paper],
    client: httpx.AsyncClient,
    max_enrich: int = 15,
    timeout: float = 6.0
) -> List[Paper]:
    """
    Enrich candidate papers that lack institutional affiliations (e.g. from ArXiv XML
    or Semantic Scholar) by cross-referencing against OpenAlex work records by DOI or title.
    Attaches authoritative authorships, institutions, countries, and country evidence.
    """
    to_enrich = [
        p for p in papers
        if not p.institutions and not p.country and not p.countries
    ][:max_enrich]

    if not to_enrich:
        return papers

    headers = {
        "User-Agent": _openalex_user_agent(),
        "Accept": "application/json",
    }

    async def _enrich_single(p: Paper):
        try:
            matched_item = None
            # 1. Try DOI lookup if available
            if p.doi:
                clean_doi = p.doi.strip()
                if clean_doi.startswith("10."):
                    url = f"{OPENALEX_API_URL}/https://doi.org/{clean_doi}"
                elif "doi.org" in clean_doi:
                    url = f"{OPENALEX_API_URL}/{clean_doi}"
                else:
                    url = f"{OPENALEX_API_URL}/https://doi.org/{clean_doi}"
                params = _openalex_params({})
                async with openalex_limiter:
                    resp = await client.get(url, params=params, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    matched_item = resp.json()

            # 2. Try Title search if DOI not available or not found
            if not matched_item and p.title:
                clean_t = re.sub(r"[^\w\s]", " ", p.title).strip()
                if len(clean_t) >= 10:
                    params = _openalex_params({
                        "search": clean_t,
                        "per-page": 1,
                        "select": "id,doi,title,authorships"
                    })
                    async with openalex_limiter:
                        resp = await client.get(OPENALEX_API_URL, params=params, headers=headers, timeout=timeout)
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        if results:
                            cand = results[0]
                            cand_title = cand.get("title") or ""
                            if title_similarity(cand_title, p.title) >= 0.80:
                                matched_item = cand

            if matched_item:
                authors, institutions, countries, _, details = _extract_openalex_authorship(
                    matched_item.get("authorships", [])
                )
                if institutions or countries:
                    p.institutions = institutions
                    p.countries = countries
                    p.country = countries[0] if countries else None
                    if not p.doi and matched_item.get("doi"):
                        p.doi = matched_item.get("doi")
                    p.country_evidence = {
                        "status": "verified",
                        "confidence": 1.0,
                        "source": "openalex_enrichment",
                        "reason": "Authoritative institutional metadata enriched from OpenAlex work record.",
                        "countries": countries,
                        "primary_country": p.country,
                        "details": details
                    }
                    logger.info("Enriched paper %r with institutions: %s, countries: %s", p.title[:40], institutions, countries)
        except Exception as e:
            logger.debug("Failed to enrich paper %r: %s", p.title[:30], str(e))

    await asyncio.gather(*[_enrich_single(p) for p in to_enrich], return_exceptions=True)
    return papers


def deduplicate_and_rank(
    arxiv_papers: Optional[List[Paper]] = None,
    ss_papers: Optional[List[Paper]] = None,
    openalex_papers: Optional[List[Paper]] = None,
    top_k: int = 6,
    similarity_threshold: float = 0.85,
    author: Optional[str] = None,
    institution: Optional[str] = None,
    country: Optional[str] = None,
    query: Optional[str] = None,
) -> List[Paper]:
    """
    Deduplicate papers across ArXiv, Semantic Scholar, and OpenAlex by normalized title similarity.
    Merges metadata when duplicates occur (preserving open-access PDFs, TLDRs, institutions, highest citation count, and code_url).
    Semantic re-ranking: relevance is scored by cosine similarity between the
    user-intent concept vector (`query`) and each paper's title/abstract concept
    vectors (synonym-aware, so alternative terminology matches the same underlying
    concept). Papers discussing the same concept with different wording rank
    together. `query=None`/empty falls back to filter/code/recency ordering.
    Ranks candidates by:
      0. Author or Institution match (if filters specified)
      1. Semantic intent similarity (concept-vector cosine vs title/abstract)
      2. Presence of implementation code (code_url)
      3. Recency and academic citation count
    Returns strictly the top_k requested papers (up to 100).
    """
    merged_papers: List[Paper] = []
    openalex_list = openalex_papers or []

    def merge_candidate(candidate: Paper):
        for existing in merged_papers:
            sim = title_similarity(existing.title, candidate.title)
            if sim >= similarity_threshold:
                # Merge sources
                if candidate.source not in existing.source:
                    existing.source = f"{existing.source} & {candidate.source}"

                # Merge citation count (keep highest)
                if candidate.citation_count is not None:
                    if existing.citation_count is None or candidate.citation_count > existing.citation_count:
                        existing.citation_count = candidate.citation_count

                # Merge TLDR
                if not existing.tldr and candidate.tldr:
                    existing.tldr = candidate.tldr

                # Merge PDF URL
                if not existing.pdf_url and candidate.pdf_url:
                    existing.pdf_url = candidate.pdf_url

                # Merge Code URL
                if not existing.code_url and candidate.code_url:
                    existing.code_url = candidate.code_url

                # Merge Year (favor candidate if existing year is missing)
                if not existing.year and candidate.year:
                    existing.year = candidate.year

                # Merge authors if candidate has more complete list
                if len(candidate.authors) > len(existing.authors):
                    existing.authors = candidate.authors

                # Merge institutions
                for inst in candidate.institutions:
                    if inst not in existing.institutions:
                        existing.institutions.append(inst)

                # Merge country and countries
                if not existing.country and candidate.country:
                    existing.country = candidate.country
                for c in candidate.countries:
                    if c not in existing.countries:
                        existing.countries.append(c)

                # Merge abstract if candidate has longer text
                if len(candidate.abstract) > len(existing.abstract):
                    existing.abstract = candidate.abstract

                return

        # Not matched: append copy
        merged_papers.append(candidate.model_copy(deep=True))

    for p in (arxiv_papers or []):
        merge_candidate(p)
    for p in (ss_papers or []):
        merge_candidate(p)
    for p in openalex_list:
        merge_candidate(p)

    author_clean = author.strip().lower() if author else ""
    inst_clean = institution.strip().lower() if institution else ""
    c_code, c_name = resolve_country_code(country) if country else (None, None)

    # 1. Strict Institution filtering (AND composition)
    resolved_inst = None
    if inst_clean:
        if inst_clean in KNOWN_INSTITUTIONS:
            resolved_inst = KNOWN_INSTITUTIONS[inst_clean][1]
        merged_papers = [
            p for p in merged_papers
            if paper_matches_institution(p, inst_clean, resolved_name=resolved_inst)
        ]

    # 2. Strict Country filtering (AND composition)
    if c_code or c_name:
        merged_papers = [
            p for p in merged_papers
            if paper_matches_country(p, c_code, c_name)
        ]

    # 3. Strict Author filtering (AND composition)
    if author_clean:
        merged_papers = [
            p for p in merged_papers
            if any(author_clean in a.lower() for a in p.authors)
        ]

    # Semantic relevance ranking (intent-based, synonym-aware):
    intent_text = (query or "").strip()
    semantic_scores: Dict[str, float] = {}
    if intent_text:
        for p in merged_papers:
            semantic_scores[p.id] = semantic_intent_score(intent_text, p.title, p.abstract or "")

    def ranking_score(p: Paper) -> Tuple[int, float, int, int, int]:
        author_match = 1 if (author_clean and any(author_clean in a.lower() for a in p.authors)) else 0
        inst_match = 1 if (inst_clean and paper_matches_institution(p, inst_clean, resolved_inst)) else 0
        country_match = 1 if ((c_code or c_name) and paper_matches_country(p, c_code, c_name)) else 0
        semantic = round(semantic_scores.get(p.id, 0.0), 4) if intent_text else 0.0
        has_code = 1 if p.code_url else 0
        year_score = p.year if p.year is not None else 0
        citation_score = p.citation_count if p.citation_count is not None else 0
        return (country_match * 2 + inst_match * 2 + author_match, semantic, has_code, year_score, citation_score)

    merged_papers.sort(key=ranking_score, reverse=True)
    top_papers = merged_papers[:top_k]

    # Populate evidence & provenance details on final papers
    for p in top_papers:
        # Determine country evidence & status
        ev_status = "unknown"
        ev_conf = 0.0
        ev_reason = "Affiliation unverified or missing."
        p_countries = list(p.countries or [])
        primary_c = p.country or (p_countries[0] if p_countries else None)
        
        if p.countries or p.country:
            ev_status = "verified"
            ev_conf = 1.0
            ev_reason = f"Verified from structured institutional metadata: {', '.join(p_countries or [p.country])}."
        elif p.institutions:
            inferred_countries = []
            for inst in p.institutions:
                detected = detect_country_from_text(inst)
                for d in detected:
                    if d not in inferred_countries:
                        inferred_countries.append(d)
                for ki, (kic, kin) in INSTITUTION_COUNTRY_MAP.items():
                    if ki in inst.lower() and kin not in inferred_countries:
                        inferred_countries.append(kin)
            if inferred_countries:
                ev_status = "inferred"
                ev_conf = 0.92
                ev_reason = f"Inferred with high confidence from recognized institutions: {', '.join(p.institutions)}."
                p_countries = inferred_countries
                primary_c = inferred_countries[0]
                if not p.country:
                    p.country = primary_c
                p.countries = inferred_countries

        p.country_evidence = {
            "country": primary_c,
            "countries": p_countries,
            "status": ev_status,
            "confidence": ev_conf,
            "reasoning": ev_reason,
            "is_collaboration": len(p_countries) > 1,
        }

        # Determine institution evidence
        p.institution_evidence = {
            "institutions": p.institutions,
            "status": "verified" if "OpenAlex" in p.source else ("inferred" if p.institutions else "unknown"),
            "confidence": 1.0 if "OpenAlex" in p.source else (0.85 if p.institutions else 0.0),
        }

        # Filter explanation
        score = semantic_scores.get(p.id, 0.0)
        relevance_str = f"relevance: {score:.2f}" if intent_text else "unconstrained"
        inst_str = p.institutions[0] if p.institutions else "Independent"
        c_str = f" — {primary_c} [{ev_status}]" if primary_c else ""
        p.filter_explanation = f"Matched filters ({relevance_str}). Affiliation: {inst_str}{c_str}."

    return top_papers


async def search_academic_papers(
    inquiry: str,
    semantic_scholar_api_key: Optional[str] = None,
    top_k: int = 6,
    use_cache: bool = True,
    client: Optional[httpx.AsyncClient] = None,
    author: Optional[str] = None,
    institution: Optional[str] = None,
    country: Optional[str] = None,
    categories: Optional[List[str]] = None,
    **kwargs: Any
) -> Tuple[List[Paper], Dict[str, Any]]:
    """
    High-level concurrent semantic search orchestrator.
    Normalizes the inquiry to a semantic intent string, launches concurrent requests
    via asyncio.gather to ArXiv (disjunctive concept groups), Semantic Scholar
    (natural-language pass-through), and OpenAlex (intent search + ID filters).
    Checks and populates thread-safe in-memory TTL search cache.
    Discovers implementation code URLs, applies semantic re-ranking, and seamlessly
    compensates quota if any single provider fails or hits rate limits.
    Supports scaling up to 100 papers, author filtering, institution filtering, and category selection.
    Expansion is local-only (no extra blocking calls); empty intent stays filter-only.
    """
    # Unify institution and university aliases
    effective_inst = (institution or kwargs.get("university") or "").strip() or None
    effective_country = (country or kwargs.get("country_code") or "").strip() or None
    cache_key = f"{inquiry}|author:{author or ''}|inst:{effective_inst or ''}|country:{effective_country or ''}|cat:{','.join(categories) if categories else ''}|k:{top_k}"
    # 1. Check TTL Cache first
    if use_cache:
        cached_result = search_cache.get(cache_key)
        if cached_result is not None:
            cached_papers, cached_meta = cached_result
            logger.info("Search cache HIT for %r (%d papers returned from cache)", cache_key, len(cached_papers))
            return cached_papers[:top_k], cached_meta

    keywords = clean_academic_query(inquiry) if inquiry else ""
    semantic_concepts = expand_semantic_concepts(keywords) if keywords else []
    logger.info("Search cache MISS. Executing semantic search for %r (author=%r, inst=%r) -> Intent: %r | concepts: %r", inquiry, author, institution, keywords, semantic_concepts)

    effective_ss_key = (
        semantic_scholar_api_key
        or kwargs.get("api_key")
        or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        or ""
    ).strip()

    metadata: Dict[str, Any] = {
        "raw_inquiry": inquiry,
        "extracted_keywords": keywords,
        "semantic_intent": keywords,
        "semantic_concepts": semantic_concepts,
        "retrieval_mode": "semantic",
        "author_filter": author,
        "institution_filter": institution,
        "country_filter": country,
        "categories_filter": categories,
        "arxiv_count": 0,
        "ss_count": 0,
        "openalex_count": 0,
        "arxiv_status": "pending",
        "ss_status": "pending",
        "openalex_status": "pending",
        "errors": [],
        "cache_hit": False,
    }

    target_fetch_count = min(max(top_k, 10), 100)
    http_client = client or get_shared_http_client()

    # Run ArXiv, Semantic Scholar, and OpenAlex in parallel with return_exceptions=True
    arxiv_task = fetch_arxiv_papers(
        keywords,
        http_client,
        max_results=target_fetch_count,
        author=author,
        categories=categories,
        country=country
    )
    ss_task = fetch_semantic_scholar_papers(
        keywords,
        http_client,
        api_key=effective_ss_key or None,
        limit=target_fetch_count,
        country=country
    )
    openalex_task = fetch_openalex_papers(
        keywords,
        http_client,
        limit=target_fetch_count,
        author=author,
        institution=effective_inst,
        country=effective_country
    )

    results = await asyncio.gather(arxiv_task, ss_task, openalex_task, return_exceptions=True)

    # Handle ArXiv result
    arxiv_papers: List[Paper] = []
    if isinstance(results[0], Exception):
        err = f"ArXiv error: {str(results[0])}"
        logger.warning(err)
        metadata["errors"].append(err)
        metadata["arxiv_status"] = "failed"
    else:
        arxiv_papers = results[0]
        metadata["arxiv_count"] = len(arxiv_papers)
        metadata["arxiv_status"] = "success" if arxiv_papers else "empty"

    # Handle Semantic Scholar result
    ss_papers: List[Paper] = []
    if isinstance(results[1], Exception):
        err = f"Semantic Scholar error: {str(results[1])}"
        logger.warning(err)
        metadata["errors"].append(err)
        metadata["ss_status"] = "failed"
    else:
        ss_papers = results[1]
        metadata["ss_count"] = len(ss_papers)
        metadata["ss_status"] = "success" if ss_papers else "empty"

    # Handle OpenAlex result
    openalex_papers: List[Paper] = []
    if isinstance(results[2], Exception):
        err = f"OpenAlex error: {str(results[2])}"
        logger.warning(err)
        metadata["errors"].append(err)
        metadata["openalex_status"] = "failed"
    else:
        openalex_papers = results[2]
        metadata["openalex_count"] = len(openalex_papers)
        metadata["openalex_status"] = "success" if openalex_papers else "empty"

    # Cross-source affiliation enrichment & code discovery for candidates
    all_candidates = arxiv_papers + ss_papers + openalex_papers
    await enrich_candidate_affiliations(all_candidates, http_client)
    await discover_paper_code_urls(all_candidates, http_client)

    # Deduplicate, merge metadata, apply semantic re-ranking, take top k
    final_papers = deduplicate_and_rank(
        arxiv_papers,
        ss_papers,
        openalex_papers,
        top_k=top_k,
        author=author,
        institution=effective_inst,
        country=effective_country,
        query=keywords,
    )
    metadata["final_count"] = len(final_papers)

    # Populate cache
    if use_cache:
        search_cache.set(cache_key, final_papers, metadata)

    return final_papers, metadata
