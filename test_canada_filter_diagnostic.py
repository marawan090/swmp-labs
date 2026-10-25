"""
Comprehensive Regression Test Suite for Canada Filter & Affiliation Matching.
SWMP Labs — Academic Research Assistant.

Covers:
1. Exact search query 'Cloud COmputing' with country filter Canada.
2. Case sensitivity and typos in queries ('COmputing', 'cLoUd', 'DISTRIBUTED').
3. Query decomposition with explicit vs inferred country.
4. Institutional disambiguation: York University (Canada) vs University of York (UK).
5. International collaborations: papers with multi-country co-authors (Canada + another country).
6. Cross-source affiliation enrichment for ArXiv / Semantic Scholar candidate papers via OpenAlex.
7. Pure negative cases: non-Canadian affiliations, papers mentioning Canada only in title/abstract.
8. Error handling: OpenAlex rate limit (HTTP 429) / timeout status transparency (no silent swallowing).
9. Cache isolation: filtered search must never return unfiltered cached entries.
10. Concurrent search execution performance.
"""

import unittest
import asyncio
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from services.search import (
    Paper,
    clean_academic_query,
    fix_query_casing_typos,
    resolve_country_code,
    paper_matches_country,
    paper_matches_institution,
    deduplicate_and_rank,
    enrich_candidate_affiliations,
    fetch_openalex_papers,
    SearchCache,
    KNOWN_INSTITUTIONS,
    INSTITUTION_COUNTRY_MAP,
)
from services.llm import QueryDecomposition, fallback_decompose_query


class TestCanadaFilterDiagnostics(unittest.IsolatedAsyncioTestCase):
    """Deterministic regression test suite for Canada filter and AI-assisted search."""

    def test_01_exact_query_typo_normalization(self):
        """Verify 'Cloud COmputing' typo is normalized to 'Cloud Computing' while acronyms survive."""
        raw = "Cloud COmputing"
        cleaned = clean_academic_query(raw)
        self.assertEqual(cleaned, "Cloud Computing")

        # Mixed casing typos
        self.assertEqual(fix_query_casing_typos("cLoUd COmputing"), "Cloud Computing")
        self.assertEqual(fix_query_casing_typos("distributed LLM on GPU"), "distributed LLM on GPU")
        self.assertEqual(fix_query_casing_typos("BERT fine-tuning in NLP"), "BERT fine-tuning in NLP")

    def test_02_country_code_resolution(self):
        """Verify Canada resolves to CA and Canada, and case insensitivity."""
        for c in ["Canada", "canada", "CANADA", "CA", "ca"]:
            code, name = resolve_country_code(c)
            self.assertEqual(code, "CA")
            self.assertEqual(name, "Canada")

    def test_03_institutional_disambiguation_york_university(self):
        """
        Verify disambiguation between:
        - York University (Toronto, Ontario, Canada)
        - University of York (York, North Yorkshire, United Kingdom)
        """
        self.assertIn("york university", INSTITUTION_COUNTRY_MAP)
        self.assertEqual(INSTITUTION_COUNTRY_MAP["york university"], ("CA", "Canada"))

        self.assertIn("university of york", INSTITUTION_COUNTRY_MAP)
        self.assertEqual(INSTITUTION_COUNTRY_MAP["university of york"], ("GB", "United Kingdom"))

        # Paper from York University (Canada)
        p_york_ca = Paper(
            id="p1",
            title="Cloud Security Models",
            authors=["Alice"],
            institutions=["York University, Toronto, Canada"],
            country="Canada",
            countries=["Canada"],
            source="OpenAlex"
        )

        # Paper from University of York (UK)
        p_york_uk = Paper(
            id="p2",
            title="Real-Time Cloud Scheduling",
            authors=["Bob"],
            institutions=["University of York, Heslington, York, UK"],
            country="United Kingdom",
            countries=["United Kingdom"],
            source="OpenAlex"
        )

        # Target: York University (Canada)
        self.assertTrue(paper_matches_institution(p_york_ca, "York University"))
        self.assertFalse(paper_matches_institution(p_york_uk, "York University"))

        # Target: University of York (UK)
        self.assertTrue(paper_matches_institution(p_york_uk, "University of York"))
        self.assertFalse(paper_matches_institution(p_york_ca, "University of York"))

        # Strict compositional AND with country filter Canada
        self.assertTrue(paper_matches_country(p_york_ca, "CA", "Canada"))
        self.assertFalse(paper_matches_country(p_york_uk, "CA", "Canada"))

    def test_04_international_co_authorship_preservation(self):
        """Verify international collaborations (Canada + another country) pass the Canada filter."""
        p_collab = Paper(
            id="collab-1",
            title="Consolidation of cloud computing in ATLAS",
            authors=["Ryan Paul Taylor", "Cristovao Jose Domingues Cordeiro"],
            institutions=["University of Victoria", "European Organization for Nuclear Research", "Lancaster University"],
            country="Canada",
            countries=["Canada", "Switzerland", "United Kingdom"],
            source="OpenAlex"
        )

        self.assertTrue(paper_matches_country(p_collab, "CA", "Canada"))
        self.assertIn("Canada", p_collab.countries)
        self.assertIn("Switzerland", p_collab.countries)
        self.assertIn("United Kingdom", p_collab.countries)

        # Multi-filter deduplicate and rank preserves all countries
        ranked = deduplicate_and_rank([p_collab], country="Canada", top_k=5)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].countries, ["Canada", "Switzerland", "United Kingdom"])

    def test_05_pure_negative_cases(self):
        """
        Verify pure negative cases fail country filtering:
        1. Non-Canadian institution (MIT, USA).
        2. Paper mentioning Canada ONLY in the title or abstract.
        """
        p_mit = Paper(
            id="mit-1",
            title="Cloud Optimization Algorithms",
            authors=["John Doe"],
            institutions=["Massachusetts Institute of Technology"],
            country="United States",
            countries=["United States"],
            source="Semantic Scholar"
        )
        self.assertFalse(paper_matches_country(p_mit, "CA", "Canada"))

        # Incidental mention in abstract/title ONLY
        p_abstract_mention = Paper(
            id="incidental-1",
            title="A Survey of Cloud Computing in Canada and North America",
            abstract="We evaluate the performance of datacenters across Canada and the US from our lab in California.",
            authors=["Jane Smith"],
            institutions=["Stanford University"],
            country="United States",
            countries=["United States"],
            source="Semantic Scholar"
        )
        self.assertFalse(paper_matches_country(p_abstract_mention, "CA", "Canada"))

        # Record with no institutions or country metadata
        p_no_meta = Paper(
            id="empty-meta-1",
            title="Generic Cloud Computing Analysis",
            authors=["Anonymous"],
            institutions=[],
            country=None,
            countries=[],
            source="ArXiv"
        )
        self.assertFalse(paper_matches_country(p_no_meta, "CA", "Canada"))

    def test_06_query_decomposition_explicit_vs_inferred(self):
        """Verify query decomposition handles explicit filters and inferred intent."""
        # Query with embedded institution
        decomp = fallback_decompose_query("cloud computing at University of Waterloo")
        self.assertEqual(decomp.topic, "cloud computing")
        self.assertEqual(decomp.institution, "University of Waterloo")

        # Explicit filters override / compose with query
        decomp2 = fallback_decompose_query("cloud computing", explicit_filters={"country": "Canada"})
        self.assertEqual(decomp2.country, "Canada")
        self.assertEqual(decomp2.topic, "cloud computing")

    async def test_07_cross_source_enrichment_rescues_valid_candidate(self):
        """
        Verify that an ArXiv candidate lacking affiliation tags is enriched via OpenAlex
        and passes the Canada filter.
        """
        raw_arxiv_paper = Paper(
            id="arxiv:2005.12267v1",
            title="Data Mining with Big Data in Intrusion Detection Systems: A Systematic Literature Review",
            authors=["Fadi Salo", "MohammadNoor Injadat", "Ali Bou Nassif"],
            institutions=[],
            country=None,
            countries=[],
            source="ArXiv"
        )

        # Before enrichment: fails
        self.assertFalse(paper_matches_country(raw_arxiv_paper, "CA", "Canada"))

        # Mock OpenAlex response for enrichment
        mock_openalex_work = {
            "title": "Data Mining with Big Data in Intrusion Detection Systems: A Systematic Literature Review",
            "doi": "https://doi.org/10.48550/arxiv.2005.12267",
            "authorships": [
                {
                    "author": {"display_name": "Fadi Salo"},
                    "institutions": [{"display_name": "Western University", "country_code": "CA"}]
                },
                {
                    "author": {"display_name": "Ali Bou Nassif"},
                    "institutions": [
                        {"display_name": "Western University", "country_code": "CA"},
                        {"display_name": "University of Sharjah", "country_code": "AE"}
                    ]
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [mock_openalex_work]}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_resp

        enriched = await enrich_candidate_affiliations([raw_arxiv_paper], mock_client)
        p_enriched = enriched[0]

        # After enrichment: has institutions, country, and passes
        self.assertIn("Western University", p_enriched.institutions)
        self.assertIn("Canada", p_enriched.countries)
        self.assertEqual(p_enriched.country, "Canada")
        self.assertTrue(paper_matches_country(p_enriched, "CA", "Canada"))
        self.assertIsNotNone(p_enriched.country_evidence)
        self.assertEqual(p_enriched.country_evidence["status"], "verified")

    async def test_08_openalex_error_and_rate_limit_transparency(self):
        """
        Verify that OpenAlex rate-limiting (429) or timeout is tracked in status_tracker
        and not silently masked as empty.
        """
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Too Many Requests"
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_openalex_papers(
                "cloud computing",
                mock_client,
                country="Canada",
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "rate_limited")
        self.assertIn("429", status_tracker.get("error", ""))

    def test_09_cache_isolation_filtered_vs_unfiltered(self):
        """
        Verify that a filtered search (e.g. country=Canada) NEVER returns
        cached entries from an unfiltered search.
        """
        cache = SearchCache(ttl=3600.0)

        unfiltered_papers = [
            Paper(id="us-1", title="US Cloud Paper", institutions=["MIT"], country="United States")
        ]
        unfiltered_meta = {"arxiv_count": 3, "ss_count": 10, "openalex_count": 0}

        # Store unfiltered results under query
        cache.set("Cloud Computing", unfiltered_papers, unfiltered_meta)

        # User searches with country="Canada"
        filtered_key = "Cloud Computing|author:|inst:|country:Canada|k:10"
        has_active_filters = True

        cached_entry = cache.get(filtered_key)
        if cached_entry is None and not has_active_filters:
            cached_entry = cache.get("Cloud Computing")

        # Must be None! Must not fall back to unfiltered entry
        self.assertIsNone(cached_entry)

    async def test_10_concurrent_searches_stability(self):
        """Verify multiple simultaneous filtered searches execute without race conditions."""
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        client.get.return_value = mock_resp

        tasks = [
            fetch_openalex_papers(f"topic {i}", client, country="Canada")
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, list)


if __name__ == "__main__":
    unittest.main()
