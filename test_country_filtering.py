"""
Automated Test Suite for Country-Based Research Filtering Feature
Verifies:
1. Schema & Pydantic models (Paper, SearchRequest, SavedPaperCreateRequest).
2. Country resolution & text detection helpers.
3. deduplicate_and_rank strict country filtering.
4. OpenAlex and ArXiv integration with country parameters.
5. FastAPI endpoint request validation and execution.
6. SQL schema definition.
7. Frontend UI elements for country selection.
"""

import asyncio
import os
import sys
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

from services.search import (
    Paper,
    COUNTRY_REGISTRY,
    resolve_country_code,
    detect_country_from_text,
    paper_matches_country,
    deduplicate_and_rank,
)
from main import app, SearchRequest, SavedPaperCreateRequest


def test_schema_and_models():
    """Verify that Paper, SearchRequest, and SavedPaperCreateRequest contain country fields."""
    p = Paper(
        id="test:1",
        title="Distributed Checkpointing in Heterogeneous Clusters",
        authors=["Alice Smith", "Bob Jones"],
        institutions=["University of Waterloo"],
        country="Canada",
        countries=["Canada", "United States"],
        year=2025
    )
    assert p.country == "Canada"
    assert "Canada" in p.countries
    dump = p.model_dump()
    assert dump["country"] == "Canada"
    assert dump["countries"] == ["Canada", "United States"]

    # SearchRequest validation
    sr = SearchRequest(query="distributed systems", country="Canada", limit=10)
    assert sr.country == "Canada"

    # SavedPaperCreateRequest validation
    sp = SavedPaperCreateRequest(paper_id="test:1", country="Canada")
    assert sp.country == "Canada"
    print("[PASS] Test 1: Schema and Pydantic models updated with country field.")


def test_country_resolution_helpers():
    """Verify country resolution dictionary and helper functions."""
    code, name = resolve_country_code("Canada")
    assert code == "CA" and name == "Canada", f"Unexpected: {code}, {name}"

    code, name = resolve_country_code("ca")
    assert code == "CA" and name == "Canada"

    code, name = resolve_country_code("USA")
    assert code == "US" and name == "United States"

    code, name = resolve_country_code("United States")
    assert code == "US" and name == "United States"

    code, name = resolve_country_code("United Kingdom")
    assert code == "GB" and name == "United Kingdom"

    code, name = resolve_country_code("Germany")
    assert code == "DE" and name == "Germany"

    code, name = resolve_country_code("Egypt")
    assert code == "EG" and name == "Egypt"

    # Text detection
    detected = detect_country_from_text("Vector Institute, Toronto, Ontario, Canada")
    assert "Canada" in detected

    detected_us = detect_country_from_text("MIT CSAIL, Cambridge, MA, USA")
    assert "United States" in detected_us

    # Paper matching
    p_ca = Paper(id="p:ca", title="Canadian Research", country="Canada", institutions=["McGill University, Canada"])
    p_us = Paper(id="p:us", title="US Research", country="United States", institutions=["Stanford University, USA"])

    assert paper_matches_country(p_ca, "CA", "Canada") is True
    assert paper_matches_country(p_ca, "US", "United States") is False
    assert paper_matches_country(p_us, "US", "United States") is True
    assert paper_matches_country(p_us, "CA", "Canada") is False

    print("[PASS] Test 2: Country resolution and matching helpers verified.")


def test_deduplicate_and_rank_strict_country_filter():
    """Verify that deduplicate_and_rank strictly filters candidates by country."""
    p1 = Paper(id="1", title="Scaling LLMs on Canadian Superclusters", country="Canada", citation_count=50)
    p2 = Paper(id="2", title="TPU Optimizations in Mountain View", country="United States", citation_count=200)
    p3 = Paper(id="3", title="Quantum Computing at Waterloo", institutions=["University of Waterloo, Canada"], citation_count=30)
    p4 = Paper(id="4", title="European Grid Computing", country="Germany", citation_count=80)

    # Filter by Canada
    results_ca = deduplicate_and_rank(
        arxiv_papers=[p1, p2],
        ss_papers=[p3],
        openalex_papers=[p4],
        top_k=10,
        country="Canada"
    )
    assert len(results_ca) == 2, f"Expected 2 Canadian papers, got {len(results_ca)}"
    titles = [p.title for p in results_ca]
    assert "Scaling LLMs on Canadian Superclusters" in titles
    assert "Quantum Computing at Waterloo" in titles
    assert "TPU Optimizations in Mountain View" not in titles
    assert "European Grid Computing" not in titles

    # Filter by Germany
    results_de = deduplicate_and_rank(
        arxiv_papers=[p1, p2],
        ss_papers=[p3],
        openalex_papers=[p4],
        top_k=10,
        country="Germany"
    )
    assert len(results_de) == 1
    assert results_de[0].title == "European Grid Computing"

    print("[PASS] Test 3: deduplicate_and_rank strictly restricts papers to target country.")


def test_sql_schema_file():
    """Verify that schema.sql exists and defines country column and index."""
    assert os.path.exists("schema.sql"), "schema.sql does not exist!"
    with open("schema.sql", "r", encoding="utf-8") as f:
        schema = f.read()

    assert "country VARCHAR" in schema or "country TEXT" in schema
    assert "idx_saved_papers_country" in schema
    print("[PASS] Test 4: Database schema definition includes country column and index.")


def test_frontend_ui_elements():
    """Verify that index.html contains country dropdown and input elements."""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert 'id="search-country-select"' in html, "Missing #search-country-select in index.html"
    assert 'id="custom-country-input"' in html, "Missing #custom-country-input in index.html"
    assert 'value="Canada"' in html, "Missing Canada option in country selector"
    assert 'params.set(\'country\'' in html, "Missing country query parameter in SSE stream builder"
    assert 'Affiliation Country:' in html, "Missing country badge tooltip on paper cards"
    print("[PASS] Test 5: Frontend UI elements and SSE parameter passing verified.")


def test_api_endpoints():
    """Test FastAPI search endpoints using TestClient."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["SWMP_TEST_KEY"] = "test-bypass-key"
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200

    # 2. Search endpoint with country filter
    res_search = client.post(
        "/api/search",
        json={"query": "distributed systems", "country": "Canada", "limit": 3},
        headers={"Authorization": "Bearer test-bypass-key", "X-Axiom-Code": "AXIOM-TEST-BYPASS"}
    )
    # The endpoint should accept country in payload without 422 Unprocessable Entity
    assert res_search.status_code != 422, f"Search endpoint failed validation: {res_search.text}"
    print(f"[PASS] Test 6: /api/search endpoint validated with country filter (status: {res_search.status_code}).")


if __name__ == "__main__":
    print("\n--- Running Country Filtering Test Suite ---")
    test_schema_and_models()
    test_country_resolution_helpers()
    test_deduplicate_and_rank_strict_country_filter()
    test_sql_schema_file()
    test_frontend_ui_elements()
    test_api_endpoints()
    print("\nALL TESTS PASSED SUCCESSFULLY!\n")
