import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from services.search import (
    Paper,
    fetch_arxiv_papers,
    fetch_semantic_scholar_papers,
    fetch_openalex_papers,
    deduplicate_and_rank,
    search_academic_papers,
)


class TestProviderStatusSemantics(unittest.IsolatedAsyncioTestCase):
    """
    Exhaustive verification of the 6 canonical provider status outcomes:
    - success_with_results
    - success_empty
    - rate_limited
    - timeout
    - network_error
    - provider_error

    Verifies that HTTP 429 is never labeled as success_empty or success,
    and that healthy providers' valid candidates are never discarded when
    one provider encounters rate limiting or error.
    """

    # -------------------------------------------------------------
    # 1. Semantic Scholar Status Tests
    # -------------------------------------------------------------
    async def test_ss_status_rate_limited(self):
        """HTTP 429 must be explicitly recorded as 'rate_limited' and not masked as empty or success."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = '{"message": "Too Many Requests"}'
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_semantic_scholar_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "rate_limited")
        self.assertIn("429", status_tracker.get("error", ""))

    async def test_ss_status_timeout(self):
        """Timeout must be recorded as 'timeout'."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_semantic_scholar_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "timeout")

    async def test_ss_status_network_error(self):
        """Connection / Network errors must be recorded as 'network_error'."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Failed to establish connection")

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_semantic_scholar_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "network_error")

    async def test_ss_status_provider_error(self):
        """500 / 503 errors must be recorded as 'provider_error'."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "Service Unavailable"
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_semantic_scholar_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "provider_error")

    async def test_ss_status_success_empty(self):
        """HTTP 200 with 0 items must be recorded as 'success_empty'."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_semantic_scholar_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "success_empty")

    async def test_ss_status_success_with_results(self):
        """HTTP 200 with papers must be recorded as 'success_with_results'."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "title": "Cloud Computing: State of the Art",
                    "authors": [{"name": "R. Buyya", "affiliations": ["University of Melbourne"]}],
                    "year": 2009,
                    "paperId": "ss-12345"
                }
            ]
        }
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_semantic_scholar_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 1)
        self.assertEqual(status_tracker.get("status"), "success_with_results")

    # -------------------------------------------------------------
    # 2. ArXiv Status Tests
    # -------------------------------------------------------------
    async def test_arxiv_status_rate_limited(self):
        """ArXiv HTTP 429 recorded as rate_limited."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "Rate limit exceeded"
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        papers = await fetch_arxiv_papers(
            "cloud computing",
            mock_client,
            status_tracker=status_tracker
        )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "rate_limited")

    async def test_arxiv_status_timeout(self):
        """ArXiv timeout recorded as timeout."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("ArXiv connection timed out")

        status_tracker = {}
        papers = await fetch_arxiv_papers(
            "cloud computing",
            mock_client,
            status_tracker=status_tracker
        )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "timeout")

    async def test_arxiv_status_network_error(self):
        """ArXiv network error recorded as network_error."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Network unreachable")

        status_tracker = {}
        papers = await fetch_arxiv_papers(
            "cloud computing",
            mock_client,
            status_tracker=status_tracker
        )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "network_error")

    async def test_arxiv_status_provider_error(self):
        """ArXiv 500 error recorded as provider_error."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        papers = await fetch_arxiv_papers(
            "cloud computing",
            mock_client,
            status_tracker=status_tracker
        )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "provider_error")

    async def test_arxiv_status_success_empty(self):
        """ArXiv returning valid feed with 0 entries recorded as success_empty."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        papers = await fetch_arxiv_papers(
            "cloud computing",
            mock_client,
            status_tracker=status_tracker
        )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "success_empty")

    async def test_arxiv_status_success_with_results(self):
        """ArXiv returning entries recorded as success_with_results."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '''<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2401.00001v1</id>
                <title>Serverless Cloud Execution</title>
                <summary>A paper on serverless computing.</summary>
                <author><name>Alice Smith</name></author>
                <published>2024-01-01T00:00:00Z</published>
            </entry>
        </feed>'''
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        papers = await fetch_arxiv_papers(
            "cloud computing",
            mock_client,
            status_tracker=status_tracker
        )

        self.assertEqual(len(papers), 1)
        self.assertEqual(status_tracker.get("status"), "success_with_results")

    # -------------------------------------------------------------
    # 3. OpenAlex Status Tests
    # -------------------------------------------------------------
    async def test_openalex_status_rate_limited(self):
        """OpenAlex HTTP 429 recorded as rate_limited."""
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
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "rate_limited")

    async def test_openalex_status_timeout(self):
        """OpenAlex timeout recorded as timeout."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.TimeoutException("OpenAlex timed out")

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_openalex_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "timeout")

    async def test_openalex_status_network_error(self):
        """OpenAlex network error recorded as network_error."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.side_effect = httpx.ConnectError("Connection reset by peer")

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_openalex_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "network_error")

    async def test_openalex_status_provider_error(self):
        """OpenAlex 502 Bad Gateway recorded as provider_error."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.text = "Bad Gateway"
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_openalex_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "provider_error")

    async def test_openalex_status_success_empty(self):
        """OpenAlex empty results recorded as success_empty."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        mock_client.get.return_value = mock_resp

        status_tracker = {}
        with patch("asyncio.sleep", new_callable=AsyncMock):
            papers = await fetch_openalex_papers(
                "cloud computing",
                mock_client,
                status_tracker=status_tracker
            )

        self.assertEqual(len(papers), 0)
        self.assertEqual(status_tracker.get("status"), "success_empty")

    # -------------------------------------------------------------
    # 4. Healthy Provider Candidate Preservation Under Provider Degradation
    # -------------------------------------------------------------
    async def test_healthy_provider_preserved_when_peer_is_rate_limited(self):
        """
        When Semantic Scholar and OpenAlex are rate-limited (429),
        candidates from ArXiv MUST NOT be discarded.
        """
        arxiv_candidate = Paper(
            id="arxiv:2401.0001",
            title="Resilient Distributed Systems in the Cloud",
            authors=["Leslie Lamport"],
            institutions=["Microsoft Research"],
            country="United States",
            countries=["United States"],
            year=2024,
            source="ArXiv"
        )

        with patch("services.search.fetch_arxiv_papers", new_callable=AsyncMock) as mock_arxiv, \
             patch("services.search.fetch_semantic_scholar_papers", new_callable=AsyncMock) as mock_ss, \
             patch("services.search.fetch_openalex_papers", new_callable=AsyncMock) as mock_oa, \
             patch("services.search.enrich_candidate_affiliations", new_callable=AsyncMock), \
             patch("services.search.discover_paper_code_urls", new_callable=AsyncMock):

            async def fake_arxiv(*args, **kwargs):
                if kwargs.get("status_tracker") is not None:
                    kwargs["status_tracker"]["status"] = "success_with_results"
                return [arxiv_candidate]

            async def fake_ss(*args, **kwargs):
                if kwargs.get("status_tracker") is not None:
                    kwargs["status_tracker"]["status"] = "rate_limited"
                    kwargs["status_tracker"]["error"] = "Semantic Scholar rate limit (HTTP 429)"
                return []

            async def fake_oa(*args, **kwargs):
                if kwargs.get("status_tracker") is not None:
                    kwargs["status_tracker"]["status"] = "rate_limited"
                    kwargs["status_tracker"]["error"] = "OpenAlex API rate limit (HTTP 429)"
                return []

            mock_arxiv.side_effect = fake_arxiv
            mock_ss.side_effect = fake_ss
            mock_oa.side_effect = fake_oa

            final_papers, metadata = await search_academic_papers(
                "distributed systems",
                top_k=5,
                use_cache=False
            )

            # Assert ArXiv candidate survived
            self.assertEqual(len(final_papers), 1)
            self.assertEqual(final_papers[0].id, "arxiv:2401.0001")

            # Assert statuses are accurate and distinct
            self.assertEqual(metadata["arxiv_status"], "success_with_results")
            self.assertEqual(metadata["ss_status"], "rate_limited")
            self.assertEqual(metadata["openalex_status"], "rate_limited")
            self.assertEqual(metadata["provider_statuses"]["semantic_scholar"], "rate_limited")
            self.assertNotEqual(metadata["ss_status"], "success_empty")
            self.assertNotEqual(metadata["ss_status"], "success")


if __name__ == "__main__":
    unittest.main()
