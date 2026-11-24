from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "IBMTechSupport" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stage2_retrieval import (
    assess_constraints,
    apply_constraint_filter,
    extract_signals,
    merge_hybrid_results,
    rerank_with_constraints,
    rewrite_query,
)


def test_extract_high_value_identifiers_and_product_version():
    s = extract_signals("IBM DataPower 10.0 CVE-2024-12345 APAR IT12345 error ABC1234E")
    assert "datapower" in s.products
    assert "10.0" in s.versions
    assert "CVE-2024-12345" in s.cves
    assert "IT12345" in s.apars
    assert "ABC1234E" in s.errors


def test_cve_and_error_conflict_is_hard():
    q = "DataPower CVE-2024-12345 ABC1234E"
    good = assess_constraints(q, "DataPower bulletin CVE-2024-12345 ABC1234E")
    bad = assess_constraints(q, "DataPower bulletin CVE-2023-99999 XYZ9999E")
    assert good.score > 0
    assert not good.hard_conflict
    assert bad.score < 0
    assert bad.hard_conflict


def test_product_and_version_mismatch_is_soft_not_deleted():
    q = "IBM DataPower 10.0 cannot boot"
    bad = assess_constraints(q, "WebSphere Application Server version 9.0 boot problem")
    assert bad.score < 0
    assert not bad.hard_conflict


def test_query_rewrite_preserves_identifiers():
    q = "Please help. IBM MQ version 9.3 fails with AMQ1234E. Thanks in advance."
    rewritten = rewrite_query(q)
    assert "AMQ1234E" in rewritten
    assert "9.3" in rewritten
    assert "please help" not in rewritten.lower()


def test_merge_keeps_original_and_rewrite_and_dedupes_document():
    chunks = [
        {"filename": "a.txt", "text": "a"},
        {"filename": "b.txt", "text": "b"},
        {"filename": "a.txt", "text": "a2"},
        {"filename": "c.txt", "text": "c"},
    ]
    original = ([], [], [0, 1], {0: 1.0, 1: 0.8})
    rewritten = ([], [], [2, 3], {2: 1.0, 3: 0.9})
    ranking, scores = merge_hybrid_results(original, rewritten, chunks, rewrite_weight=0.7)
    names = [chunks[i]["filename"] for i in ranking]
    assert names.count("a.txt") == 1
    assert "b.txt" in names and "c.txt" in names
    assert scores


def test_constraint_filter_promotes_exact_identifier_match():
    chunks = [
        {"filename": "wrong.txt", "text": "DataPower CVE-2023-99999 XYZ9999E"},
        {"filename": "right.txt", "text": "DataPower CVE-2024-12345 ABC1234E"},
        {"filename": "neutral.txt", "text": "DataPower general troubleshooting"},
    ]
    q = "DataPower CVE-2024-12345 ABC1234E"
    ranking, assessments, _ = apply_constraint_filter(q, [0, 2, 1], chunks, strength=1.0)
    assert ranking[0] == 1
    assert assessments[0].hard_conflict


def test_reranker_constraint_layer_demotes_hard_conflict():
    chunks = [
        {"filename": "wrong.txt", "text": "DataPower CVE-2023-99999 XYZ9999E"},
        {"filename": "right.txt", "text": "DataPower CVE-2024-12345 ABC1234E"},
    ]
    q = "DataPower CVE-2024-12345 ABC1234E"
    # Cross-encoder alone slightly prefers the wrong document.
    raw = {0: 1.1, 1: 1.0}
    ranked = rerank_with_constraints(
        q, [0, 1], raw, chunks,
        candidate_k=2, filter_enabled=True, filter_strength=1.0,
    )
    assert ranked[0]["row_id"] == 1
    assert ranked[1]["hard_conflict"] is True


def test_filter_disabled_preserves_crossencoder_order():
    chunks = [
        {"filename": "a.txt", "text": "DataPower CVE-2023-99999"},
        {"filename": "b.txt", "text": "DataPower CVE-2024-12345"},
    ]
    ranked = rerank_with_constraints(
        "DataPower CVE-2024-12345", [0, 1], {0: 2.0, 1: 1.0}, chunks,
        candidate_k=2, filter_enabled=False, filter_strength=0.0,
    )
    assert [x["row_id"] for x in ranked] == [0, 1]


def test_sqlcode_format_variants_canonicalize_identically():
    a = extract_signals("DB2 failed with SQLCODE=-911")
    b = extract_signals("DB2 failed with SQLCODE -911")
    c = extract_signals("DB2 failed with SQL -911")
    assert a.errors == b.errors == c.errors == frozenset({"SQLCODE-911"})


def test_portal_and_dynamic_workload_console_are_distinct_products():
    q = extract_signals("IBM WebSphere Portal login problem")
    d = extract_signals("Tivoli Dynamic Workload Console connection problem")
    assert "websphere_portal" in q.products
    assert "dynamic_workload_console" in d.products
    check = assess_constraints("IBM WebSphere Portal login problem", "Tivoli Dynamic Workload Console connection problem")
    assert check.score < 0 and not check.hard_conflict
