import asyncio
import json
from services.search import (
    fetch_arxiv_papers,
    fetch_semantic_scholar_papers,
    fetch_openalex_papers,
    enrich_candidate_affiliations,
    paper_matches_country,
    resolve_country_code,
    get_shared_http_client,
    clean_academic_query,
    deduplicate_and_rank,
)

async def run_diagnostics():
    client = get_shared_http_client()
    query = "Cloud COmputing"
    country = "Canada"
    code, name = resolve_country_code(country)
    cleaned = clean_academic_query(query)

    print(f"=== DIAGNOSTIC TRACE FOR QUERY: '{query}' | Country: '{country}' (code={code}, name={name}) ===")
    print(f"Normalized Intent Query: '{cleaned}'\n")

    # 1. Fetch from sources
    print("--- 1. SOURCE CANDIDATE RETRIEVAL ---")
    openalex_status_tracker = {}
    arxiv_papers = await fetch_arxiv_papers(cleaned, client, max_results=10, country=country)
    ss_papers = await fetch_semantic_scholar_papers(cleaned, client, limit=10, country=country)
    openalex_papers = await fetch_openalex_papers(
        cleaned,
        client,
        limit=10,
        country=country,
        status_tracker=openalex_status_tracker
    )

    print(f"ArXiv retrieved: {len(arxiv_papers)} papers")
    print(f"Semantic Scholar retrieved: {len(ss_papers)} papers")
    print(f"OpenAlex retrieved: {len(openalex_papers)} papers (status: {openalex_status_tracker.get('status', 'unknown')})\n")

    raw_candidates = arxiv_papers + ss_papers + openalex_papers
    print(f"Total raw candidates across all sources before enrichment: {len(raw_candidates)}\n")

    # 2. Inspect candidates BEFORE cross-source enrichment
    print("--- 2. CANDIDATES PRE-ENRICHMENT ---")
    pre_enrich_records = []
    for i, p in enumerate(raw_candidates, 1):
        matched = paper_matches_country(p, code, name)
        rejection_reason = "NONE (PASSED)" if matched else (
            "MISSING_AFFILIATION_METADATA: Record has no institution, country, or author affiliations."
            if not p.institutions and not p.country and not p.countries
            else f"AFFILIATION_MISMATCH: Affiliations {p.institutions} / country {p.country} do not match {country}."
        )
        rec = {
            "index": i,
            "source": p.source,
            "id": p.id,
            "title": p.title,
            "doi": p.doi,
            "authors": p.authors[:3],
            "institutions": p.institutions,
            "country": p.country,
            "countries": p.countries,
            "country_evidence": p.country_evidence,
            "country_filter_pass": matched,
            "rejection_reason": rejection_reason,
        }
        pre_enrich_records.append(rec)
        print(f"Candidate #{i}: [{p.source}] {p.title}")
        print(f"  Institutions: {p.institutions} | Country: {p.country} | Filter: {'PASS' if matched else 'FAIL'}")
        print(f"  Reason: {rejection_reason}")

    # 3. Apply Cross-Source Affiliation Enrichment
    print("\n--- 3. CROSS-SOURCE AFFILIATION ENRICHMENT VIA OPENALEX ---")
    enriched_candidates = await enrich_candidate_affiliations(raw_candidates, client)

    # 4. Inspect candidates AFTER cross-source enrichment
    print("\n--- 4. CANDIDATES POST-ENRICHMENT ---")
    post_enrich_records = []
    for i, p in enumerate(enriched_candidates, 1):
        matched = paper_matches_country(p, code, name)
        rejection_reason = "NONE (PASSED)" if matched else (
            "MISSING_AFFILIATION_METADATA: Record has no institution, country, or author affiliations."
            if not p.institutions and not p.country and not p.countries
            else f"AFFILIATION_MISMATCH: Affiliations {p.institutions} / country {p.country} do not match {country}."
        )
        rec = {
            "index": i,
            "source": p.source,
            "id": p.id,
            "title": p.title,
            "doi": p.doi,
            "authors": p.authors[:3],
            "institutions": p.institutions,
            "country": p.country,
            "countries": p.countries,
            "country_evidence": p.country_evidence,
            "country_filter_pass": matched,
            "rejection_reason": rejection_reason,
        }
        post_enrich_records.append(rec)
        print(f"Post-Enrich Candidate #{i}: [{p.source}] {p.title}")
        print(f"  Institutions: {p.institutions} | Country: {p.country} | Countries: {p.countries}")
        print(f"  Evidence: {p.country_evidence.get('reason') if p.country_evidence else 'None'}")
        print(f"  Filter: {'[PASS]' if matched else '[FAIL]'}")
        print(f"  Reason: {rejection_reason}\n")

    # 5. Final Deduplication and Strict Ranking
    print("--- 5. FINAL DEDUPLICATION AND STRICT FILTERING ---")
    final_papers = deduplicate_and_rank(
        arxiv_papers=arxiv_papers,
        ss_papers=ss_papers,
        openalex_papers=openalex_papers,
        country=country,
        query=cleaned,
        top_k=10
    )
    print(f"Final papers matching all strict constraints: {len(final_papers)}")
    for p in final_papers:
        print(f"  * [{p.source}] {p.title} (Country: {p.country}, Inst: {p.institutions})")

    # Save diagnostic records to JSON for report
    with open("diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "country_filter": country,
            "cleaned_query": cleaned,
            "retrieval_counts": {
                "arxiv": len(arxiv_papers),
                "semantic_scholar": len(ss_papers),
                "openalex": len(openalex_papers),
                "openalex_status": openalex_status_tracker.get("status", "unknown"),
            },
            "pre_enrichment_candidates": pre_enrich_records,
            "post_enrichment_candidates": post_enrich_records,
            "final_matching_papers_count": len(final_papers),
            "final_matching_papers": [
                {
                    "title": p.title,
                    "source": p.source,
                    "country": p.country,
                    "countries": p.countries,
                    "institutions": p.institutions,
                    "evidence": p.country_evidence,
                }
                for p in final_papers
            ]
        }, f, indent=2)
    print("\nSaved diagnostic records to diagnostic_results.json")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())
