import unittest
from services.search import (
    Paper,
    resolve_country_code,
    paper_matches_country,
    paper_matches_institution,
    deduplicate_and_rank,
    KNOWN_INSTITUTIONS,
    INSTITUTION_ALIASES,
    INSTITUTION_COUNTRY_MAP,
)
from main import SearchRequest, SavedPaperCreateRequest


class TestFiltersComposition(unittest.TestCase):

    def setUp(self):
        self.p_mit = Paper(
            id="test:1",
            title="Quantum Circuit Optimization on Superconducting Qubits",
            authors=["Alice Smith", "Bob Jones"],
            institutions=["Massachusetts Institute of Technology"],
            country="United States",
            countries=["United States"],
            year=2024,
            abstract="Quantum computing optimization algorithms at MIT."
        )

        self.p_waterloo = Paper(
            id="test:2",
            title="Post-Quantum Cryptographic Protocols",
            authors=["David Brown"],
            institutions=["University of Waterloo"],
            country="Canada",
            countries=["Canada"],
            year=2023,
            abstract="Secure cryptography developed at University of Waterloo in Canada."
        )

        self.p_stanford = Paper(
            id="test:3",
            title="Deep Learning Foundation Models for Computer Vision",
            authors=["Carol White"],
            institutions=["Stanford University"],
            country="United States",
            countries=["United States"],
            year=2024,
            abstract="Novel vision architectures from Stanford AI lab."
        )

        self.p_toronto = Paper(
            id="test:4",
            title="Neural Information Processing Systems",
            authors=["Geoffrey Hinton"],
            institutions=["University of Toronto"],
            country="Canada",
            countries=["Canada"],
            year=2022,
            abstract="Foundations of deep learning at University of Toronto."
        )

        self.p_no_inst = Paper(
            id="test:5",
            title="General Quantum Information Theory",
            authors=["Anonymous Researcher"],
            institutions=[],
            country=None,
            countries=[],
            year=2021,
            abstract="Theoretical survey of quantum gates with no institutional affiliation."
        )

        self.p_summit = Paper(
            id="test:6",
            title="High Performance Parallel Matrix Algorithms",
            authors=["Elena Rostova"],
            institutions=["Summit Computing Center"],
            country="United States",
            countries=["United States"],
            year=2023,
            abstract="Benchmarking on the Summit cluster."
        )

    # 1. Institution Matching & Canonical Alias Resolution
    def test_mit_alias_matching(self):
        """Verify 'MIT' matches 'Massachusetts Institute of Technology'."""
        self.assertTrue(paper_matches_institution(self.p_mit, "MIT"))
        self.assertTrue(paper_matches_institution(self.p_mit, "mit"))
        self.assertTrue(paper_matches_institution(self.p_mit, "Massachusetts Institute of Technology"))
        self.assertFalse(paper_matches_institution(self.p_stanford, "MIT"))

    def test_short_acronym_boundary_matching(self):
        """Verify 'MIT' does not falsely match 'Summit Computing Center'."""
        self.assertFalse(paper_matches_institution(self.p_summit, "MIT"))
        self.assertFalse(paper_matches_institution(self.p_summit, "mit"))

    def test_waterloo_alias_matching(self):
        """Verify 'Waterloo' and 'UWaterloo' match 'University of Waterloo'."""
        self.assertTrue(paper_matches_institution(self.p_waterloo, "Waterloo"))
        self.assertTrue(paper_matches_institution(self.p_waterloo, "uwaterloo"))
        self.assertTrue(paper_matches_institution(self.p_waterloo, "University of Waterloo"))
        self.assertFalse(paper_matches_institution(self.p_mit, "Waterloo"))

    def test_empty_institution_rejected_when_filter_active(self):
        """Papers with no institution must be strictly excluded when university filter is applied."""
        self.assertFalse(paper_matches_institution(self.p_no_inst, "MIT"))
        self.assertFalse(paper_matches_institution(self.p_no_inst, "Waterloo"))
        self.assertFalse(paper_matches_institution(self.p_no_inst, "Stanford"))

    # 2. Country Matching & Institution Location Inference
    def test_country_matching(self):
        """Verify country matching by name and ISO code."""
        self.assertTrue(paper_matches_country(self.p_waterloo, "CA", "Canada"))
        self.assertTrue(paper_matches_country(self.p_mit, "US", "United States"))
        self.assertFalse(paper_matches_country(self.p_waterloo, "US", "United States"))
        self.assertFalse(paper_matches_country(self.p_mit, "CA", "Canada"))

    def test_institution_country_map_inference(self):
        """Verify paper with Canadian institution matches Canada even if country was unset."""
        p_waterloo_no_country = Paper(
            id="test:7",
            title="Quantum Teleportation",
            institutions=["University of Waterloo"]
        )
        self.assertTrue(paper_matches_country(p_waterloo_no_country, "CA", "Canada"))
        self.assertFalse(paper_matches_country(p_waterloo_no_country, "US", "United States"))

    # 3. Compositional Filtering (AND Logic)
    def test_composition_topic_and_university(self):
        """Topic = 'Quantum' AND University = 'MIT'."""
        candidates = [self.p_mit, self.p_waterloo, self.p_stanford, self.p_no_inst]
        results = deduplicate_and_rank(
            arxiv_papers=candidates,
            query="Quantum",
            institution="MIT",
            top_k=10
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test:1")
        self.assertIn("Massachusetts Institute of Technology", results[0].institutions)

    def test_composition_topic_and_country(self):
        """Topic = 'Quantum' AND Country = 'Canada'."""
        candidates = [self.p_mit, self.p_waterloo, self.p_stanford, self.p_toronto, self.p_no_inst]
        results = deduplicate_and_rank(
            arxiv_papers=candidates,
            query="Quantum",
            country="Canada",
            top_k=10
        )
        # Only Canadian papers: p_waterloo and p_toronto
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("Canada", r.countries)

    def test_composition_university_and_country(self):
        """University = 'Waterloo' AND Country = 'Canada'."""
        candidates = [self.p_mit, self.p_waterloo, self.p_stanford, self.p_toronto]
        results = deduplicate_and_rank(
            arxiv_papers=candidates,
            institution="Waterloo",
            country="Canada",
            top_k=10
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test:2")
        self.assertIn("University of Waterloo", results[0].institutions)

    def test_composition_mismatch_returns_empty(self):
        """University = 'Waterloo' AND Country = 'USA' must strictly return 0 results (no OR fallback)."""
        candidates = [self.p_mit, self.p_waterloo, self.p_stanford, self.p_toronto]
        results = deduplicate_and_rank(
            arxiv_papers=candidates,
            institution="Waterloo",
            country="USA",
            top_k=10
        )
        self.assertEqual(len(results), 0)

    def test_composition_topic_and_university_and_country(self):
        """Topic = 'AI' AND University = 'Stanford' AND Country = 'USA'."""
        candidates = [self.p_mit, self.p_waterloo, self.p_stanford, self.p_toronto]
        results = deduplicate_and_rank(
            arxiv_papers=candidates,
            query="AI",
            institution="Stanford",
            country="USA",
            top_k=10
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "test:3")
        self.assertIn("Stanford University", results[0].institutions)

    # 4. SearchRequest and SavedPaperCreateRequest Schema Synchronization
    def test_search_request_aliasing(self):
        """SearchRequest properly synchronizes university/institution aliases and cleans empty values."""
        # Using university parameter
        req1 = SearchRequest(query="Quantum", university="MIT")
        self.assertEqual(req1.institution, "MIT")
        self.assertEqual(req1.university, "MIT")

        # Using institution parameter
        req2 = SearchRequest(query="Quantum", institution="Stanford")
        self.assertEqual(req2.institution, "Stanford")
        self.assertEqual(req2.university, "Stanford")

        # Whitespace handling
        req3 = SearchRequest(query="AI", university="   ")
        self.assertIsNone(req3.institution)
        self.assertIsNone(req3.university)

        # Country sanitization
        req4 = SearchRequest(query="AI", country="  Canada  ")
        self.assertEqual(req4.country, "Canada")


if __name__ == "__main__":
    unittest.main()
