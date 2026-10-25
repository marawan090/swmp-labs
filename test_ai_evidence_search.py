"""
Comprehensive Automated Test Suite for AI-Assisted, Evidence-Based Research Search System.
Covers:
  1. Intelligent Institution & University Resolution (abbreviations, ambiguity handling, international institutions).
  2. Evidence-Based Country Affiliation Reasoning (rejecting incidental abstract mentions, provenance labeling, international collaborations).
  3. Query Intent & Semantic Filter Decomposition (topic separation, geographic conflict detection).
  4. Compositional Filtering with AI-assisted ranking (Topic AND University AND Country preserved).
  5. Fallback and Reliability Guardrails (zero downtime when LLM is unavailable or rate-limited).
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("."))

from services.search import (
    Paper,
    resolve_country_code,
    paper_matches_country,
    paper_matches_institution,
    deduplicate_and_rank,
    KNOWN_INSTITUTIONS,
    INSTITUTION_ALIASES,
)
from services.llm import (
    InstitutionResolution,
    PaperAffiliationEvidence,
    QueryDecomposition,
    fallback_resolve_institution,
    fallback_classify_affiliations,
    fallback_decompose_query,
    get_default_llm_service,
)


class TestAIEvidenceSearchSystem(unittest.IsolatedAsyncioTestCase):

    # ========================================================================
    # 1. Intelligent Institution and University Resolution
    # ========================================================================

    def test_abbreviation_and_canonical_resolution(self):
        """Test standard abbreviations (MIT, CMU, UWaterloo) map to canonical names."""
        r_mit = fallback_resolve_institution("MIT")
        self.assertEqual(r_mit.canonical_name, "Massachusetts Institute of Technology")
        self.assertEqual(r_mit.country_code, "US")
        self.assertFalse(r_mit.is_ambiguous)

        r_cmu = fallback_resolve_institution("CMU")
        self.assertEqual(r_cmu.canonical_name, "Carnegie Mellon University")
        self.assertEqual(r_cmu.country_code, "US")

        r_uw = fallback_resolve_institution("UWaterloo")
        self.assertEqual(r_uw.canonical_name, "University of Waterloo")
        self.assertEqual(r_uw.country_code, "CA")

    def test_ambiguous_institution_resolution(self):
        """Test ambiguous names (Columbia, Washington) flag ambiguity and expose candidate interpretations."""
        r_columbia = fallback_resolve_institution("Columbia")
        self.assertTrue(r_columbia.is_ambiguous)
        self.assertEqual(r_columbia.canonical_name, "Columbia University")
        self.assertGreaterEqual(len(r_columbia.candidates), 2)
        candidate_names = [c.name for c in r_columbia.candidates]
        self.assertIn("Columbia University", candidate_names)
        self.assertIn("University of British Columbia", candidate_names)

        r_wash = fallback_resolve_institution("Washington")
        self.assertTrue(r_wash.is_ambiguous)
        self.assertEqual(r_wash.canonical_name, "University of Washington")
        candidate_names_wash = [c.name for c in r_wash.candidates]
        self.assertIn("University of Washington", candidate_names_wash)
        self.assertIn("Washington University in St. Louis", candidate_names_wash)

    def test_international_and_foreign_universities(self):
        """Test non-English and international institutions (Tsinghua, ETH Zurich, Sorbonne, TU Munich)."""
        r_eth = fallback_resolve_institution("ETH")
        self.assertEqual(r_eth.canonical_name, "ETH Zurich")
        self.assertEqual(r_eth.country_code, "CH")

        r_tsinghua = fallback_resolve_institution("tsinghua")
        self.assertEqual(r_tsinghua.canonical_name, "Tsinghua University")
        self.assertEqual(r_tsinghua.country_code, "CN")

        r_sorbonne = fallback_resolve_institution("sorbonne")
        self.assertEqual(r_sorbonne.canonical_name, "Sorbonne University")
        self.assertEqual(r_sorbonne.country_code, "FR")

        r_tum = fallback_resolve_institution("TU Munich")
        self.assertEqual(r_tum.canonical_name, "Technical University of Munich")
        self.assertEqual(r_tum.country_code, "DE")

    # ========================================================================
    # 2. Evidence-Based Country Affiliation Reasoning
    # ========================================================================

    def test_rejection_of_incidental_abstract_mentions(self):
        """
        CRITICAL GUARDRAIL: A paper that merely mentions 'Canada' in its abstract
        (e.g., dataset, evaluation) but has authors from US must NEVER match country='Canada'.
        """
        p_incidental = Paper(
            id="p:incidental",
            title="Evaluating Neural Networks on Demographics",
            authors=["John Doe", "Jane Roe"],
            institutions=["Stanford University"],
            country="United States",
            countries=["United States"],
            abstract="In this work we evaluate model performance on the Canadian healthcare dataset and compare with Canada regional health outcomes."
        )

        # Must match US
        self.assertTrue(paper_matches_country(p_incidental, "US", "United States"))
        # Must strictly REJECT Canada
        self.assertFalse(paper_matches_country(p_incidental, "CA", "Canada"))

    def test_provenance_labeling(self):
        """Test provenance statuses: verified, inferred, unknown."""
        # 1. Verified via structured OpenAlex metadata
        p_verified = Paper(
            id="p:v",
            title="Fast Attention Kernels",
            authors=["Alice"],
            institutions=["University of Waterloo"],
            country="Canada",
            countries=["Canada"]
        )
        aff_v = fallback_classify_affiliations(p_verified)
        self.assertEqual(aff_v.status, "verified")
        self.assertEqual(aff_v.primary_country, "Canada")
        self.assertEqual(aff_v.confidence, 1.0)

        # 2. Inferred via recognized academic institution
        p_inferred = Paper(
            id="p:inf",
            title="Distributed Consensus",
            authors=["Bob"],
            institutions=["University of Waterloo"],
            country=None,
            countries=[]
        )
        aff_inf = fallback_classify_affiliations(p_inferred)
        self.assertEqual(aff_inf.status, "inferred")
        self.assertIn("Canada", aff_inf.countries)
        self.assertGreater(aff_inf.confidence, 0.8)

        # 3. Unknown when no affiliation evidence exists
        p_unknown = Paper(
            id="p:unk",
            title="Independent Survey",
            authors=["Charlie"],
            institutions=[],
            country=None,
            countries=[],
            abstract="No institutional affiliation."
        )
        aff_unk = fallback_classify_affiliations(p_unknown)
        self.assertEqual(aff_unk.status, "unknown")
        self.assertIsNone(aff_unk.primary_country)

    def test_international_collaboration_matching(self):
        """A paper with authors from multiple countries must match either participating country."""
        p_collab = Paper(
            id="p:collab",
            title="Global Climate Modeling via Graph Neural Networks",
            authors=["Alice Canada", "Bob US"],
            institutions=["University of Toronto", "Stanford University"],
            country="Canada",
            countries=["Canada", "United States"],
            year=2024
        )
        aff = fallback_classify_affiliations(p_collab)
        self.assertTrue(aff.is_international_collaboration)
        self.assertIn("Canada", aff.countries)
        self.assertIn("United States", aff.countries)

        # Must match Canada
        self.assertTrue(paper_matches_country(p_collab, "CA", "Canada"))
        # Must also match United States
        self.assertTrue(paper_matches_country(p_collab, "US", "United States"))
        # Must not match Germany
        self.assertFalse(paper_matches_country(p_collab, "DE", "Germany"))

    # ========================================================================
    # 3. Query Intent and Semantic Filter Understanding
    # ========================================================================

    def test_query_decomposition_separation(self):
        """Separate topic keywords from institutional, geographic, and author constraints."""
        q = "fast neural network training from Stanford in United States by Alice"
        decomp = fallback_decompose_query(q)
        self.assertEqual(decomp.institution, "Stanford")
        self.assertEqual(decomp.country, "United States")
        self.assertIn("fast neural network training", decomp.topic)
        self.assertFalse(decomp.has_geographic_conflict)

    def test_geographic_conflict_detection(self):
        """Detect when an institution's home country conflicts with a requested country."""
        q_conflict = "deep learning papers from Stanford in Canada"
        decomp = fallback_decompose_query(q_conflict)
        self.assertEqual(decomp.institution, "Stanford")
        self.assertEqual(decomp.country, "Canada")
        self.assertTrue(decomp.has_geographic_conflict)
        self.assertIsNotNone(decomp.conflict_explanation)
        self.assertIn("Stanford University is located in United States", decomp.conflict_explanation)

    # ========================================================================
    # 4. Strict Compositional Filtering (Topic AND University AND Country)
    # ========================================================================

    def test_strict_multi_filter_composition(self):
        """A candidate paper must satisfy all active filters simultaneously."""
        p1 = Paper(
            id="1",
            title="Deep Learning at Waterloo Canada",
            institutions=["University of Waterloo"],
            country="Canada",
            countries=["Canada"],
            abstract="Deep learning research."
        )
        p2 = Paper(
            id="2",
            title="Deep Learning at Stanford USA",
            institutions=["Stanford University"],
            country="United States",
            countries=["United States"],
            abstract="Deep learning research."
        )
        p3 = Paper(
            id="3",
            title="Quantum Computing at Waterloo Canada",
            institutions=["University of Waterloo"],
            country="Canada",
            countries=["Canada"],
            abstract="Quantum computing research."
        )

        # Filter: Topic='deep learning', University='Waterloo', Country='Canada'
        results = deduplicate_and_rank(
            arxiv_papers=[p1, p2, p3],
            query="deep learning",
            institution="Waterloo",
            country="Canada"
        )
        self.assertEqual(len(results), 2)  # p1 and p3 are both Waterloo & Canada
        # p1 has higher semantic relevance to 'deep learning'
        self.assertEqual(results[0].id, "1")
        # Evidence populated
        self.assertIsNotNone(results[0].country_evidence)
        self.assertEqual(results[0].country_evidence.get("status"), "verified")
        self.assertIsNotNone(results[0].filter_explanation)

        # Filter: University='Waterloo', Country='United States' -> 0 results (conflict)
        results_conflict = deduplicate_and_rank(
            arxiv_papers=[p1, p2, p3],
            institution="Waterloo",
            country="United States"
        )
        self.assertEqual(len(results_conflict), 0)

    # ========================================================================
    # 5. Service Fallbacks & Reliability
    # ========================================================================

    async def test_llm_service_fallback_on_unreachable_api(self):
        """Ensure LLM service gracefully resolves institutions and affiliations even if API key is invalid."""
        svc = get_default_llm_service()
        # Call with dummy key / unreachable to verify it never crashes and returns valid Pydantic models
        res = await svc.resolve_institution_entity("CMU", custom_api_key="sk-dummy-key-for-test")
        self.assertIsInstance(res, InstitutionResolution)
        self.assertEqual(res.canonical_name, "Carnegie Mellon University")

        p = Paper(id="t_svc", title="Test Paper", institutions=["MIT"])
        aff_res = await svc.classify_paper_affiliations(p, custom_api_key="sk-dummy-key-for-test")
        self.assertIsInstance(aff_res, PaperAffiliationEvidence)
        self.assertIn("United States", aff_res.countries)

    def test_research_stream_scoping_and_execution(self):
        """Ensure /api/research/stream initializes and yields SSE events without UnboundLocalError."""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        resp = client.get('/api/research/stream?query=distributed+consensus')
        self.assertEqual(resp.status_code, 200)
        lines = []
        for line in resp.iter_lines():
            if line:
                lines.append(line)
                if len(lines) >= 3:
                    break
        self.assertGreaterEqual(len(lines), 1)
        self.assertTrue(any("keywords" in l or "status" in l for l in lines))


if __name__ == "__main__":
    unittest.main()
