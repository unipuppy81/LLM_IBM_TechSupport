from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import yaml

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[2]            # IBMTechSupport
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from document_utils import chunk_filename, filename_key
from stage2_retrieval import (
    DEFAULT_SELECTED_CONFIG,
    apply_constraint_filter,
    merge_hybrid_results,
    rerank_with_constraints,
    rewrite_query,
    score_reranker_candidates,
)

VAL_PATH = PROJECT_DIR / "data" / "processed" / "val_106.json"
TEST_PATH = PROJECT_DIR / "data" / "processed" / "test_clean.json"
OUT_DIR = PROJECT_DIR / "evaluate_results" / "106" / "retrieval_stage2"
SUMMARY_PATH = OUT_DIR / "optimization_summary.json"
CASES_PATH = OUT_DIR / "optimization_cases.jsonl"
SELECTED_CONFIG = DEFAULT_SELECTED_CONFIG
VAL_RAG_CONFIG = PROJECT_DIR / "configs" / "rag" / "v5_stage2_106.yaml"
TEST_RAG_CONFIG = PROJECT_DIR / "configs" / "rag" / "v5_stage2_test.yaml"
VAL_RAG_DIR = PROJECT_DIR / "evaluate_results" / "106" / "rag" / "v5_stage2"
VAL_JUDGE_DIR = PROJECT_DIR / "evaluate_results" / "106" / "judge_quality" / "stage2_v313"
FINAL_DIR = PROJECT_DIR / "evaluate_results" / "test_120" / "stage2_final"
CURRENT_V4_METRICS = PROJECT_DIR / "evaluate_results" / "106" / "rag" / "v4" / "metrics.json"
CURRENT_V313_JUDGE = PROJECT_DIR / "evaluate_results" / "106" / "judge_quality" / "regression_106" / "judge_v311_106.jsonl"

FILTER_STRENGTHS = (0.5, 1.0, 1.5)
RERANKER_CANDIDATE_KS = (10, 20, 30, 50)
REWRITE_WEIGHT = 0.7


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_records(path: Path, expected: int | None = None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"데이터 파일 없음: {path}")
    records = _read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"데이터가 비었습니다: {path}")
    if expected is not None and len(records) != expected:
        raise ValueError(f"데이터 개수 불일치: expected={expected}, actual={len(records)}")
    ids = [str(r.get("id")) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("중복 QA ID")
    for row in records:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"질문 누락: {row.get('id')}")
        if not isinstance(row.get("is_impossible"), bool):
            raise ValueError(f"is_impossible 오류: {row.get('id')}")
    return records


def _answerable_records(records: list[dict]) -> list[dict]:
    out = []
    for row in records:
        if row["is_impossible"]:
            continue
        gold = {
            filename_key(c["filename"])
            for c in row.get("contexts", [])
            if isinstance(c, dict) and isinstance(c.get("filename"), str) and c["filename"].strip()
        }
        if not gold:
            raise ValueError(f"Answerable 질문에 gold context 없음: {row['id']}")
        out.append(row)
    if not out:
        raise ValueError("평가 가능한 answerable 질문이 없습니다.")
    return out


def _dcg(found_ranks: list[int]) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in found_ranks)


def retrieval_metrics(records: list[dict], rankings: list[list[int]], chunks: list[dict]) -> tuple[dict, dict]:
    if len(records) != len(rankings) or not records:
        raise ValueError("records/rankings 정렬 오류")

    sums = Counter()
    cases = {}
    for record, ranking in zip(records, rankings):
        gold = {filename_key(c["filename"]) for c in record["contexts"]}
        names = [filename_key(chunk_filename(chunks[row_id])) for row_id in ranking]
        # Defensive de-duplication at document level.
        unique_names = list(dict.fromkeys(names))
        top5 = unique_names[:5]
        hit_ranks = [i for i, name in enumerate(unique_names[:10], 1) if name in gold]
        found5 = set(top5) & gold
        hit1 = bool(set(unique_names[:1]) & gold)
        hit3 = bool(set(unique_names[:3]) & gold)
        hit5 = bool(found5)
        hit10 = bool(set(unique_names[:10]) & gold)
        recall5 = len(found5) / len(gold)
        precision5 = len(found5) / 5.0
        rr10 = 1.0 / hit_ranks[0] if hit_ranks else 0.0
        found_ranks5 = [i for i, name in enumerate(top5, 1) if name in gold]
        ideal = _dcg(list(range(1, min(5, len(gold)) + 1)))
        ndcg5 = _dcg(found_ranks5) / ideal if ideal else 0.0

        sums.update({
            "hit1": float(hit1), "hit3": float(hit3), "hit5": float(hit5), "hit10": float(hit10),
            "recall5": recall5, "precision5": precision5, "rr10": rr10, "ndcg5": ndcg5,
        })
        cases[str(record["id"])] = {
            "hit_at_5": hit5,
            "first_gold_rank": hit_ranks[0] if hit_ranks else None,
            "gold_documents": sorted(gold),
            "top5": top5,
        }

    n = len(records)
    metrics = {
        "answerable_evaluated": n,
        "hit_at_1_percent": round(sums["hit1"] / n * 100, 4),
        "hit_at_3_percent": round(sums["hit3"] / n * 100, 4),
        "hit_at_5_percent": round(sums["hit5"] / n * 100, 4),
        "hit_at_10_percent": round(sums["hit10"] / n * 100, 4),
        "recall_at_5_percent": round(sums["recall5"] / n * 100, 4),
        "precision_at_5": round(sums["precision5"] / n, 6),
        "mrr_at_10": round(sums["rr10"] / n, 6),
        "ndcg_at_5": round(sums["ndcg5"] / n, 6),
    }
    return metrics, cases


def _compare_cases(base_cases: dict, new_cases: dict) -> dict:
    gained = []
    lost = []
    improved_rank = []
    worsened_rank = []
    for rid in base_cases:
        before = base_cases[rid]
        after = new_cases[rid]
        if not before["hit_at_5"] and after["hit_at_5"]:
            gained.append(rid)
        elif before["hit_at_5"] and not after["hit_at_5"]:
            lost.append(rid)
        b = before["first_gold_rank"] or 10**9
        a = after["first_gold_rank"] or 10**9
        if a < b:
            improved_rank.append(rid)
        elif a > b:
            worsened_rank.append(rid)
    return {
        "gained_hit5_ids": gained,
        "lost_hit5_ids": lost,
        "improved_first_gold_rank_ids": improved_rank,
        "worsened_first_gold_rank_ids": worsened_rank,
    }


def _safe_candidate(base: dict, candidate: dict) -> bool:
    # Stage2 changes are accepted only when they do not lose a gold-document
    # Hit@5 or Recall@5 on the development retrieval set.
    return (
        candidate["hit_at_5_percent"] + 1e-9 >= base["hit_at_5_percent"]
        and candidate["recall_at_5_percent"] + 1e-9 >= base["recall_at_5_percent"]
    )


def _ranking_quality_key(metrics: dict):
    return (
        metrics["ndcg_at_5"],
        metrics["mrr_at_10"],
        metrics["hit_at_3_percent"],
        metrics["precision_at_5"],
        metrics["hit_at_5_percent"],
        metrics["recall_at_5_percent"],
    )


def _select_non_regressing(base_name: str, base: dict, candidates: list[dict]) -> dict:
    safe = [c for c in candidates if _safe_candidate(base["metrics"], c["metrics"])]
    if not safe:
        return base
    best = max(safe, key=lambda c: _ranking_quality_key(c["metrics"]))
    # Keep the prior stage when the new stage is metric-identical.  This avoids
    # adding complexity that did not prove useful.
    if _ranking_quality_key(best["metrics"]) <= _ranking_quality_key(base["metrics"]):
        return base
    return best


def _rows_from_rerank(
    questions: list[str],
    rankings: list[list[int]],
    score_maps: list[dict[int, float]],
    chunks: list[dict],
    *,
    candidate_k: int,
    filter_enabled: bool,
    filter_strength: float,
) -> list[list[int]]:
    out = []
    for q, ranking, score_map in zip(questions, rankings, score_maps):
        rows = rerank_with_constraints(
            q, ranking, score_map, chunks,
            candidate_k=candidate_k,
            filter_enabled=filter_enabled,
            filter_strength=filter_strength,
        )
        out.append([r["row_id"] for r in rows])
    return out


def _write_selected_config(selected: dict, metrics: dict, baseline: dict, source_dataset: Path) -> None:
    payload = {
        "version": "stage2-v1",
        "frozen": True,
        "source_dataset": str(source_dataset.resolve()),
        "selection_policy": "No Hit@5/Recall@5 regression; then maximize nDCG@5/MRR@10.",
        "selected": {
            "reranker_enabled": bool(selected["reranker_enabled"]),
            "filter_enabled": bool(selected["filter_enabled"]),
            "filter_strength": float(selected["filter_strength"]),
            "reranker_candidate_k": int(selected["reranker_candidate_k"]),
            "query_rewrite_enabled": bool(selected["query_rewrite_enabled"]),
            "rewrite_weight": float(selected["rewrite_weight"]),
        },
        "runtime": {
            "reranker_batch_size": int(selected["reranker_batch_size"]),
            "max_candidates": 50,
        },
        "selected_metrics": metrics,
        "baseline_reranker_metrics": baseline,
    }
    SELECTED_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = SELECTED_CONFIG.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(SELECTED_CONFIG)


def optimize(args):
    records_all = _validate_records(Path(args.dataset), expected=args.expected_count)
    records = _answerable_records(records_all)
    questions = [r["question"] for r in records]

    print(f"[1/6] Retrieval baseline: total={len(records_all)}, answerable={len(records)}", flush=True)
    from rag_pipeline import load_search_system
    from hybrid_search import search_all
    from reranker_search import load_reranker
    from retrieval_config import load_retrieval_config

    search_system = load_search_system()
    embedding_model, faiss_index, bm25_index, chunks = search_system
    retrieval_cfg = load_retrieval_config()
    batch_size = int(retrieval_cfg["reranker"]["batch_size"])

    start = time.perf_counter()
    original_results = search_all(embedding_model, faiss_index, bm25_index, questions)
    hybrid_rankings = [list(r[2]) for r in original_results]
    hybrid_metrics, hybrid_cases = retrieval_metrics(records, hybrid_rankings, chunks)

    print("[1/6] Reranker baseline scoring (top 50 cached once)", flush=True)
    reranker = load_reranker()
    max_k = min(50, max((len(x) for x in hybrid_rankings), default=0))
    if max_k < 10:
        raise RuntimeError(f"Hybrid 후보 수가 너무 적습니다: {max_k}")
    base_score_maps = score_reranker_candidates(
        reranker, questions, hybrid_rankings, chunks,
        candidate_k=max_k, batch_size=batch_size,
    )
    baseline_rerankings = _rows_from_rerank(
        questions, hybrid_rankings, base_score_maps, chunks,
        candidate_k=10, filter_enabled=False, filter_strength=0.0,
    )
    reranker_metrics, reranker_cases = retrieval_metrics(records, baseline_rerankings, chunks)

    stages = []
    stages.append({"name":"baseline_hybrid", "metrics":hybrid_metrics})
    current = {
        "name": "baseline_reranker_k10",
        "metrics": reranker_metrics,
        "cases": reranker_cases,
        "rankings": hybrid_rankings,
        "score_maps": base_score_maps,
        "reranker_enabled": True,
        "filter_enabled": False,
        "filter_strength": 0.0,
        "reranker_candidate_k": 10,
        "query_rewrite_enabled": False,
        "rewrite_weight": REWRITE_WEIGHT,
        "reranker_batch_size": batch_size,
    }
    stages.append({
        "name": current["name"], "metrics": current["metrics"],
        "vs_hybrid": _compare_cases(hybrid_cases, reranker_cases),
    })

    print("[2/6] Product/version/CVE/APAR/error soft filter", flush=True)
    filter_candidates = []
    for strength in FILTER_STRENGTHS:
        filtered_rankings = []
        for q, ranking in zip(questions, hybrid_rankings):
            # Keep the filter inside the same top-50 candidate pool that was
            # actually CrossEncoder-scored.  This guarantees optimizer/runtime
            # parity and prevents an unscored tail document from entering the
            # tuned top-k merely because conflicts were demoted.
            filtered, _, _ = apply_constraint_filter(
                q, ranking[:max_k], chunks, strength=strength
            )
            filtered_rankings.append(filtered)
        reranked = _rows_from_rerank(
            questions, filtered_rankings, base_score_maps, chunks,
            candidate_k=10, filter_enabled=True, filter_strength=strength,
        )
        metrics, cases = retrieval_metrics(records, reranked, chunks)
        item = {
            "name": f"filter_s{strength:g}_k10", "metrics": metrics, "cases": cases,
            "rankings": filtered_rankings, "score_maps": base_score_maps,
            "reranker_enabled": True,
            "filter_enabled": True, "filter_strength": strength,
            "reranker_candidate_k": 10, "query_rewrite_enabled": False,
            "rewrite_weight": REWRITE_WEIGHT, "reranker_batch_size": batch_size,
        }
        filter_candidates.append(item)
        stages.append({"name":item["name"], "metrics":metrics, "vs_previous":_compare_cases(current["cases"], cases)})
    current = _select_non_regressing(current["name"], current, filter_candidates)

    print(f"[3/6] Reranker candidate_k tuning: {RERANKER_CANDIDATE_KS}", flush=True)
    k_candidates = []
    # If filter stage was not selected, use original rankings.  Otherwise use its
    # filtered rankings; all rows originate from the same original top-50 cache.
    tuning_rankings = current["rankings"]
    for candidate_k in RERANKER_CANDIDATE_KS:
        if candidate_k > max_k:
            continue
        reranked = _rows_from_rerank(
            questions, tuning_rankings, base_score_maps, chunks,
            candidate_k=candidate_k,
            filter_enabled=bool(current["filter_enabled"]),
            filter_strength=float(current["filter_strength"]),
        )
        metrics, cases = retrieval_metrics(records, reranked, chunks)
        item = dict(current)
        item.update({"name":f"reranker_k{candidate_k}", "metrics":metrics, "cases":cases,
                     "reranker_candidate_k":candidate_k})
        k_candidates.append(item)
        stages.append({"name":item["name"], "metrics":metrics, "vs_previous":_compare_cases(current["cases"], cases)})
    current = _select_non_regressing(current["name"], current, k_candidates)

    print("[4/6] Conservative query rewrite (original + rewritten multi-query fusion)", flush=True)
    rewritten_questions = [rewrite_query(q) for q in questions]
    changed = sum(a != b for a, b in zip(questions, rewritten_questions))
    rewrite_candidate = None
    if changed:
        rewritten_results = search_all(embedding_model, faiss_index, bm25_index, rewritten_questions)
        merged_rankings = []
        for original, rewritten in zip(original_results, rewritten_results):
            ranking, _ = merge_hybrid_results(original, rewritten, chunks, rewrite_weight=REWRITE_WEIGHT)
            # Runtime also uses a fixed 50-document pre-reranker pool.
            ranking = ranking[:50]
            if current["filter_enabled"]:
                ranking, _, _ = apply_constraint_filter(
                    questions[len(merged_rankings)], ranking, chunks,
                    strength=float(current["filter_strength"]),
                )
            merged_rankings.append(ranking)
        merged_score_maps = score_reranker_candidates(
            reranker, questions, merged_rankings, chunks,
            candidate_k=min(50, max((len(x) for x in merged_rankings), default=0)),
            batch_size=batch_size,
        )
        reranked = _rows_from_rerank(
            questions, merged_rankings, merged_score_maps, chunks,
            candidate_k=int(current["reranker_candidate_k"]),
            filter_enabled=bool(current["filter_enabled"]),
            filter_strength=float(current["filter_strength"]),
        )
        metrics, cases = retrieval_metrics(records, reranked, chunks)
        rewrite_candidate = dict(current)
        rewrite_candidate.update({
            "name":"query_rewrite_augment", "metrics":metrics, "cases":cases,
            "rankings":merged_rankings, "score_maps":merged_score_maps,
            "query_rewrite_enabled":True, "rewrite_weight":REWRITE_WEIGHT,
        })
        stages.append({"name":"query_rewrite_augment", "metrics":metrics,
                       "changed_queries":changed, "vs_previous":_compare_cases(current["cases"], cases)})
        current = _select_non_regressing(current["name"], current, [rewrite_candidate])
    else:
        stages.append({"name":"query_rewrite_augment", "skipped":"no query changed"})

    print("[5/6] Full retrieval re-evaluation and regression check", flush=True)
    final_metrics = current["metrics"]
    final_cases = current["cases"]
    comparisons = {
        "final_vs_hybrid": _compare_cases(hybrid_cases, final_cases),
        "final_vs_reranker_k10": _compare_cases(reranker_cases, final_cases),
    }
    if not _safe_candidate(reranker_metrics, final_metrics):
        raise RuntimeError("최종 Stage2가 기존 Reranker Hit@5/Recall@5를 회귀했습니다.")

    # A reranker is not automatically better. If the complete Stage2 path loses
    # gold-document Hit@5/Recall@5 versus the original Hybrid baseline, freeze
    # the Hybrid baseline instead of shipping a regression.
    if (
        not _safe_candidate(hybrid_metrics, final_metrics)
        or _ranking_quality_key(final_metrics) <= _ranking_quality_key(hybrid_metrics)
    ):
        current = {
            "name": "fallback_hybrid_baseline",
            "metrics": hybrid_metrics,
            "cases": hybrid_cases,
            "rankings": hybrid_rankings,
            "score_maps": None,
            "reranker_enabled": False,
            "filter_enabled": False,
            "filter_strength": 0.0,
            "reranker_candidate_k": 10,
            "query_rewrite_enabled": False,
            "rewrite_weight": REWRITE_WEIGHT,
            "reranker_batch_size": batch_size,
        }
        final_metrics = hybrid_metrics
        final_cases = hybrid_cases
        comparisons["fallback_reason"] = (
            "Stage2 reranker path did not beat the Hybrid baseline under the frozen selection rule"
        )
        comparisons["final_vs_hybrid"] = _compare_cases(hybrid_cases, final_cases)
        comparisons["final_vs_reranker_k10"] = _compare_cases(reranker_cases, final_cases)

    print("[6/6] Freeze selected retrieval config", flush=True)
    _write_selected_config(current, final_metrics, reranker_metrics, Path(args.dataset))
    config_sha = _sha256(SELECTED_CONFIG)

    case_rows = []
    for record in records:
        rid = str(record["id"])
        case_rows.append({
            "id": rid,
            "question": record["question"],
            "baseline_reranker": reranker_cases[rid],
            "final_stage2": final_cases[rid],
        })
    _atomic_jsonl(CASES_PATH, case_rows)

    summary = {
        "schema": "rag-stage2-optimization-v1",
        "dataset": str(Path(args.dataset).resolve()),
        "dataset_sha256": _sha256(Path(args.dataset)),
        "total_records": len(records_all),
        "answerable_evaluated": len(records),
        "stages": stages,
        "selected_stage": current["name"],
        "selected_config": str(SELECTED_CONFIG),
        "selected_config_sha256": config_sha,
        "selected": {
            "reranker_enabled": current["reranker_enabled"],
            "filter_enabled": current["filter_enabled"],
            "filter_strength": current["filter_strength"],
            "reranker_candidate_k": current["reranker_candidate_k"],
            "query_rewrite_enabled": current["query_rewrite_enabled"],
            "rewrite_weight": current["rewrite_weight"],
        },
        "baseline_hybrid": hybrid_metrics,
        "baseline_reranker_k10": reranker_metrics,
        "final": final_metrics,
        "comparisons": comparisons,
        "selection_rule": "Never reduce Hit@5 or Recall@5 at a stage; among safe candidates maximize nDCG@5, MRR@10, Hit@3, Precision@5.",
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "test_clean_touched": False,
    }
    _atomic_json(SUMMARY_PATH, summary)

    print(json.dumps({
        "baseline_hybrid": hybrid_metrics,
        "baseline_reranker_k10": reranker_metrics,
        "selected_stage": current["name"],
        "selected": summary["selected"],
        "final": final_metrics,
        "lost_hit5_vs_reranker": comparisons["final_vs_reranker_k10"]["lost_hit5_ids"],
        "test_clean_touched": False,
        "summary": str(SUMMARY_PATH),
    }, ensure_ascii=False, indent=2))
    return summary


def _generate_with_config(config_path: Path, expected_count: int, *, reuse: bool = True):
    from evaluate.evaluate_rag import load_config, create_predictions
    from evaluate.common import score_predictions

    config = load_config(config_path)
    dataset_path = PROJECT_DIR / config["dataset"]["path"]
    records = _validate_records(dataset_path, expected=expected_count)
    output_dir = PROJECT_DIR / config["output"]["directory"]
    pred_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "generation_manifest.json"
    expected_manifest = {
        "dataset_sha256": _sha256(dataset_path),
        "rag_config_sha256": _sha256(config_path),
        "retrieval_config_sha256": _sha256(SELECTED_CONFIG) if SELECTED_CONFIG.is_file() else None,
        "expected_count": expected_count,
    }

    if reuse and pred_path.is_file() and metrics_path.is_file() and manifest_path.is_file():
        old_manifest = _read_json(manifest_path)
        rows = _read_jsonl(pred_path)
        if (
            old_manifest == expected_manifest
            and len(rows) == expected_count
            and [str(x["id"]) for x in rows] == [str(x["id"]) for x in records]
        ):
            print(f"[REUSE] 기존 생성 결과 재사용: {pred_path}")
            return pred_path, metrics_path

    rows, performance = create_predictions(records, config)
    metrics = score_predictions(records, rows, "clean", config["inference"]["batch_size"])
    metrics.update({
        "retriever": config["retrieval"]["method"],
        "top_k": config["retrieval"]["top_k"],
        **performance,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(pred_path, rows)
    _atomic_json(metrics_path, metrics)
    _atomic_json(manifest_path, expected_manifest)
    return pred_path, metrics_path


def _judge_grade(predictions: Path, output: Path, *, reuse: bool = True):
    manifest = output.with_name(output.stem + "_manifest.json")
    fingerprint = _sha256(predictions)
    if reuse and output.is_file() and manifest.is_file():
        old = _read_json(manifest)
        if old.get("predictions_sha256") == fingerprint and old.get("judge_schema_version") == "ibm-techqa-judge-claims-v3.13":
            print(f"[REUSE] 기존 Judge 결과 재사용: {output}")
            return output

    from evaluate.judge_quality import judge_quality
    judge_quality.main([
        "grade",
        "--predictions", str(predictions),
        "--output", str(output),
        "--limit", "0",
        "--force",
    ])
    _atomic_json(manifest, {
        "predictions": str(predictions),
        "predictions_sha256": fingerprint,
        "judge_schema_version": "ibm-techqa-judge-claims-v3.13",
    })
    return output


def _judge_summary(judge_path: Path, predictions_path: Path) -> dict:
    judged = _read_jsonl(judge_path)
    predictions = {str(r["id"]): r for r in _read_jsonl(predictions_path)}
    if len(judged) != len(predictions):
        raise ValueError(f"Judge/Prediction 개수 불일치: {len(judged)} vs {len(predictions)}")
    counts = Counter()
    flags = Counter()
    risks = Counter()
    by_label = {"answerable": Counter(), "impossible": Counter()}
    for row in judged:
        rid = str(row["id"])
        if rid not in predictions:
            raise ValueError(f"Judge ID가 predictions에 없음: {rid}")
        score = row.get("score") or {}
        auto_pass = score.get("auto_pass") is True
        counts["auto_pass" if auto_pass else "not_auto_pass"] += 1
        counts[row.get("judge_status", "UNKNOWN")] += 1
        risks[str(score.get("risk_level", "unknown"))] += 1
        for flag in score.get("review_flags") or []:
            flags[str(flag)] += 1
        label = "impossible" if predictions[rid].get("is_impossible") else "answerable"
        by_label[label]["auto_pass" if auto_pass else "not_auto_pass"] += 1
    n = len(judged)
    return {
        "n": n,
        "auto_pass": counts["auto_pass"],
        "auto_pass_rate_percent": round(counts["auto_pass"] / n * 100, 2) if n else 0.0,
        "judge_status_counts": {k:v for k,v in counts.items() if k not in {"auto_pass","not_auto_pass"}},
        "risk_counts": dict(risks),
        "review_flag_counts": dict(flags.most_common()),
        "by_original_answerability_label": {k: dict(v) for k,v in by_label.items()},
    }


def _retrieval_metrics_from_predictions(dataset_path: Path, predictions_path: Path) -> dict:
    records = _validate_records(dataset_path)
    by_id = {str(r["id"]): r for r in _read_jsonl(predictions_path)}
    answerable = _answerable_records(records)
    synthetic_chunks = []
    rankings = []
    # Reuse the generic metric implementation by constructing one row per
    # retrieved document.  This keeps final-test reporting identical to optimize.
    for record in answerable:
        row = by_id.get(str(record["id"]))
        if row is None:
            raise ValueError(f"Prediction 누락: {record['id']}")
        ranking = []
        for doc in row.get("retrieved_documents") or []:
            ranking.append(len(synthetic_chunks))
            synthetic_chunks.append({"filename": doc["filename"], "text": doc.get("text", "x") or "x"})
        rankings.append(ranking)
    return retrieval_metrics(answerable, rankings, synthetic_chunks)[0]


def validate(args):
    if not SUMMARY_PATH.is_file() or not SELECTED_CONFIG.is_file():
        raise FileNotFoundError("먼저 optimize를 실행해 Stage2 retrieval을 동결하세요.")

    print("[7/9] Validation 106 answers regeneration", flush=True)
    pred_path, metrics_path = _generate_with_config(VAL_RAG_CONFIG, 106, reuse=not args.force)

    print("[8/9] Frozen Judge v3.13 grading", flush=True)
    judge_path = VAL_JUDGE_DIR / "judge_v313_stage2_106.jsonl"
    _judge_grade(pred_path, judge_path, reuse=not args.force)
    judge_summary = _judge_summary(judge_path, pred_path)
    retrieval = _retrieval_metrics_from_predictions(VAL_PATH, pred_path)
    generation = _read_json(metrics_path)

    baseline = {}
    if CURRENT_V4_METRICS.is_file():
        baseline["v4_generation_metrics"] = _read_json(CURRENT_V4_METRICS)
    if CURRENT_V313_JUDGE.is_file():
        old_rows = _read_jsonl(CURRENT_V313_JUDGE)
        old_auto = sum((r.get("score") or {}).get("auto_pass") is True for r in old_rows)
        baseline["v4_frozen_judge"] = {
            "n": len(old_rows), "auto_pass": old_auto,
            "auto_pass_rate_percent": round(old_auto / len(old_rows) * 100, 2) if old_rows else 0.0,
        }

    report = {
        "schema": "rag-stage2-validation-v1",
        "retrieval_config": str(SELECTED_CONFIG),
        "retrieval_config_sha256": _sha256(SELECTED_CONFIG),
        "predictions": str(pred_path),
        "predictions_sha256": _sha256(pred_path),
        "generation_metrics": generation,
        "retrieval_metrics": retrieval,
        "frozen_judge_v313": judge_summary,
        "baseline": baseline,
        "test_clean_touched": False,
        "note": "Judge v3.13 is frozen. Human PASS/FAIL labels from old answers are not reused as labels for newly generated answers.",
    }
    report_path = VAL_JUDGE_DIR / "stage2_validation_report.json"
    _atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[VALIDATION 완료] {report_path}")
    print("test_clean.json은 아직 사용하지 않았습니다.")
    return report


def final_test(args):
    validation_report = VAL_JUDGE_DIR / "stage2_validation_report.json"
    if not validation_report.is_file():
        raise FileNotFoundError("먼저 validate를 완료해야 합니다. test_clean은 그 전에는 사용하지 않습니다.")
    validation = _read_json(validation_report)
    if validation.get("retrieval_config_sha256") != _sha256(SELECTED_CONFIG):
        raise RuntimeError("Validation 이후 retrieval 설정이 변경되었습니다. test_clean 실행 금지.")
    if not TEST_PATH.is_file():
        raise FileNotFoundError(f"test_clean.json 없음: {TEST_PATH}")
    _validate_records(TEST_PATH, expected=120)

    final_report = FINAL_DIR / "final_report.json"
    if final_report.exists() and not args.force_final:
        raise FileExistsError(
            f"최종 test_clean 결과가 이미 있습니다: {final_report}\n"
            "최종 테스트는 반복 튜닝에 사용하면 안 됩니다. 정말 재실행할 때만 --force-final을 명시하세요."
        )

    print("[9/9] Untouched test_clean 120 final evaluation", flush=True)
    pred_path, metrics_path = _generate_with_config(TEST_RAG_CONFIG, 120, reuse=True)
    judge_path = FINAL_DIR / "judge_v313_test120.jsonl"
    _judge_grade(pred_path, judge_path, reuse=False)

    report = {
        "schema": "rag-stage2-final-test-v1",
        "dataset": str(TEST_PATH),
        "dataset_sha256": _sha256(TEST_PATH),
        "retrieval_config": str(SELECTED_CONFIG),
        "retrieval_config_sha256": _sha256(SELECTED_CONFIG),
        "validation_report_sha256": _sha256(validation_report),
        "predictions": str(pred_path),
        "predictions_sha256": _sha256(pred_path),
        "generation_metrics": _read_json(metrics_path),
        "retrieval_metrics": _retrieval_metrics_from_predictions(TEST_PATH, pred_path),
        "frozen_judge_v313": _judge_summary(judge_path, pred_path),
        "warning": "This is the final untouched test result. Do not tune retrieval/Judge on these 120 cases.",
    }
    _atomic_json(final_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[FINAL 완료] {final_report}")
    return report


def self_check(_args=None):
    # No model/data required.  This catches the failure modes that are easiest
    # to introduce while editing Stage2 retrieval.
    from stage2_retrieval import assess_constraints, extract_signals, rewrite_query

    q = "IBM DataPower 10.0 reports CVE-2024-12345 and error ABC1234E. How do I fix it?"
    good = "DataPower version 10.0 security bulletin CVE-2024-12345 ABC1234E"
    bad = "WebSphere Application Server version 9.0 CVE-2023-99999 XYZ9999E"
    g = assess_constraints(q, good)
    b = assess_constraints(q, bad)
    assert g.score > 0 and not g.hard_conflict
    assert b.score < 0 and b.hard_conflict
    s = extract_signals(q)
    assert "CVE-2024-12345" in s.cves and "ABC1234E" in s.errors
    rewritten = rewrite_query("Please help. " + q + " Thanks in advance.")
    assert "CVE-2024-12345" in rewritten and "ABC1234E" in rewritten

    # Selection must reject a recall regression even if nDCG appears better.
    base = {"metrics":{"hit_at_5_percent":90.0,"recall_at_5_percent":90.0,"ndcg_at_5":0.5,"mrr_at_10":0.5,"hit_at_3_percent":80.0,"precision_at_5":0.1}}
    bad_c = {"metrics":{"hit_at_5_percent":89.0,"recall_at_5_percent":90.0,"ndcg_at_5":0.9,"mrr_at_10":0.9,"hit_at_3_percent":90.0,"precision_at_5":0.2}}
    assert _select_non_regressing("base", base, [bad_c]) is base
    print("[SELF-CHECK] PASS")


def main(argv=None):
    p = argparse.ArgumentParser(description="RAG 2차 개선: retrieval 최적화 -> 106 검증 -> untouched test 120")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("self-check", help="모델 없이 Stage2 결정론적 검증")
    a.set_defaults(run=self_check)

    a = sub.add_parser("optimize", help="1~6: baseline/filter/reranker/query rewrite/re-evaluate/freeze")
    a.add_argument("--dataset", type=Path, default=VAL_PATH)
    a.add_argument("--expected-count", type=int, default=106)
    a.set_defaults(run=optimize)

    a = sub.add_parser("validate", help="7~8: 106 답변 재생성 + frozen Judge v3.13 평가")
    a.add_argument("--force", action="store_true", help="기존 Stage2 validation 결과도 재생성/재평가")
    a.set_defaults(run=validate)

    a = sub.add_parser("final-test", help="9: validation 이후 untouched test_clean 120 최종 평가")
    a.add_argument("--force-final", action="store_true", help="이미 존재하는 최종 테스트를 의도적으로 재실행")
    a.set_defaults(run=final_test)

    args = p.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    main()
