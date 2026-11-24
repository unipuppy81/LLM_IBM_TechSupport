from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from document_utils import chunk_filename, filename_key
from stage2_retrieval import (
    DEFAULT_SELECTED_CONFIG,
    apply_constraint_filter,
    load_selected_config,
    merge_hybrid_results,
    rerank_with_constraints,
    rewrite_query,
    score_reranker_candidates,
)

DEFAULT_DATASET = PROJECT_DIR / "data" / "processed" / "val_106.json"
OUT_DIR = PROJECT_DIR / "evaluate_results" / "106" / "retrieval_stage3_diagnosis"
JSON_PATH = OUT_DIR / "diagnosis.json"
TXT_PATH = OUT_DIR / "diagnosis.txt"
WIDE_K = 100


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gold_documents(record: dict) -> set[str]:
    return {
        filename_key(c["filename"])
        for c in record.get("contexts", [])
        if isinstance(c, dict) and isinstance(c.get("filename"), str) and c["filename"].strip()
    }


def dedupe_documents(ranking, chunks) -> list[int]:
    seen = set()
    out = []
    for row_id in ranking:
        row_id = int(row_id)
        key = filename_key(chunk_filename(chunks[row_id]))
        if key in seen:
            continue
        seen.add(key)
        out.append(row_id)
    return out


def first_gold_rank(ranking, chunks, gold: set[str], limit: int | None = None):
    unique = dedupe_documents(ranking, chunks)
    if limit is not None:
        unique = unique[:limit]
    for rank, row_id in enumerate(unique, start=1):
        if filename_key(chunk_filename(chunks[row_id])) in gold:
            return rank
    return None


def has_gold(ranking, chunks, gold: set[str], k: int) -> bool:
    rank = first_gold_rank(ranking, chunks, gold, limit=k)
    return rank is not None and rank <= k


def weighted_rrf(bm25_ranking, dense_ranking, *, rrf_k: int, bm25_weight: float, dense_weight: float):
    scores = {}
    for ranking, weight in ((bm25_ranking, bm25_weight), (dense_ranking, dense_weight)):
        for rank, row_id in enumerate(ranking, start=1):
            row_id = int(row_id)
            scores[row_id] = scores.get(row_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores, key=lambda row_id: (-scores[row_id], row_id)), scores


def classify_failure(*, hybrid_rank, candidate_rank, filtered_rank, final_rank, wide_rrf_rank, bm25_rank, dense_rank, candidate_k):
    if final_rank is not None and final_rank <= 5:
        return "HIT"

    if candidate_rank is not None and candidate_rank <= candidate_k and filtered_rank is not None and filtered_rank > candidate_k:
        return "FILTER_DEMOTION"

    if filtered_rank is not None and filtered_rank <= candidate_k:
        return "RERANKER_TOP5_FAILURE"

    if filtered_rank is not None and filtered_rank <= 50:
        return "CANDIDATE_K_CUTOFF"

    if wide_rrf_rank is not None and wide_rrf_rank <= WIDE_K:
        return "HYBRID_CANDIDATE_K_50_LIMIT"

    if bm25_rank is not None or dense_rank is not None:
        return "FUSION_FAILURE"

    return "BASE_RETRIEVER_MISS"


def fmt_rank(value) -> str:
    return "-" if value is None else str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stage2 Retrieval failure diagnosis on val_106")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--wide-k", type=int, default=WIDE_K)
    args = parser.parse_args(argv)

    if args.wide_k < 50:
        parser.error("--wide-k는 50 이상이어야 합니다.")

    dataset_path = args.dataset
    if not dataset_path.is_file():
        raise FileNotFoundError(f"데이터 파일 없음: {dataset_path}")

    all_records = read_json(dataset_path)
    if not isinstance(all_records, list):
        raise ValueError("dataset은 JSON list여야 합니다.")

    records = [r for r in all_records if r.get("is_impossible") is False]
    if not records:
        raise ValueError("answerable 데이터가 없습니다.")
    for record in records:
        if not gold_documents(record):
            raise ValueError(f"Gold context 없음: {record.get('id')}")

    questions = [r["question"] for r in records]
    print(f"[DIAG 1/5] total={len(all_records)}, answerable={len(records)}", flush=True)

    from rag_pipeline import load_search_system
    from hybrid_search import search_all
    from bm25_search import retrieve as bm25_retrieve
    from dense_search import encode_questions
    from reranker_search import load_reranker
    from retrieval_config import load_retrieval_config

    embedding_model, faiss_index, bm25_index, chunks = load_search_system()
    retrieval_cfg = load_retrieval_config()
    selected_cfg = load_selected_config(DEFAULT_SELECTED_CONFIG)
    selected = selected_cfg["selected"]
    runtime = selected_cfg.get("runtime") or {}
    max_candidates = int(runtime.get("max_candidates", 50))
    candidate_k = int(selected["reranker_candidate_k"])
    batch_size = int(runtime.get("reranker_batch_size", retrieval_cfg["reranker"]["batch_size"]))

    print(f"[DIAG 2/5] wide BM25/Dense search: top {args.wide_k}", flush=True)
    retriever, tokenizer, chunk_id_to_row_id = bm25_index

    # Dense is encoded once for all 77 questions.
    vectors = encode_questions(embedding_model, questions)
    _, dense_ids = faiss_index.search(vectors, min(args.wide_k, faiss_index.ntotal))

    wide_bm25_rankings = []
    wide_dense_rankings = []
    wide_rrf_rankings = []

    hybrid_cfg = retrieval_cfg["hybrid"]
    rrf_k = int(hybrid_cfg["rrf_k"])
    bm25_weight = float(hybrid_cfg["bm25_weight"])
    dense_weight = float(hybrid_cfg["dense_weight"])

    for question, d_ids in zip(questions, dense_ids):
        bm25_results = bm25_retrieve(question, retriever, tokenizer, min(args.wide_k, len(chunks)))
        bm25_ranked = [
            int(chunk_id_to_row_id[row["chunk_id"]])
            for row in bm25_results
            if row.get("score", 0.0) > 0 and row.get("chunk_id") in chunk_id_to_row_id
        ]
        dense_ranked = [int(row_id) for row_id in d_ids if int(row_id) >= 0]
        rrf_ranked, _ = weighted_rrf(
            bm25_ranked,
            dense_ranked,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        wide_bm25_rankings.append(bm25_ranked)
        wide_dense_rankings.append(dense_ranked)
        wide_rrf_rankings.append(dedupe_documents(rrf_ranked, chunks))

    print("[DIAG 3/5] reproduce current Stage2 candidate path", flush=True)
    original_results = search_all(embedding_model, faiss_index, bm25_index, questions)
    original_hybrid = [list(result[2]) for result in original_results]

    # Reproduce query-rewrite branch too, even though current selected config is OFF.
    rewritten_questions = [rewrite_query(q) for q in questions]
    rewritten_results = None
    if bool(selected["query_rewrite_enabled"]) and any(a != b for a, b in zip(questions, rewritten_questions)):
        rewritten_results = search_all(embedding_model, faiss_index, bm25_index, rewritten_questions)

    stage2_candidates = []
    filtered_candidates = []
    for i, question in enumerate(questions):
        rewrite_result = None
        if rewritten_results is not None and rewritten_questions[i] != question:
            rewrite_result = rewritten_results[i]
        merged, _ = merge_hybrid_results(
            original_results[i],
            rewrite_result,
            chunks,
            rewrite_weight=float(selected["rewrite_weight"]),
        )
        merged = merged[:max_candidates]
        stage2_candidates.append(list(merged))

        if bool(selected["filter_enabled"]):
            filtered, _, _ = apply_constraint_filter(
                question,
                merged,
                chunks,
                strength=float(selected["filter_strength"]),
            )
        else:
            filtered = list(merged)
        filtered_candidates.append(filtered)

    print(f"[DIAG 4/5] reranker scoring: candidate_k={candidate_k}", flush=True)
    if bool(selected["reranker_enabled"]):
        reranker = load_reranker()
        score_maps = score_reranker_candidates(
            reranker,
            questions,
            filtered_candidates,
            chunks,
            candidate_k=candidate_k,
            batch_size=batch_size,
        )
        final_rankings = []
        for question, ranking, score_map in zip(questions, filtered_candidates, score_maps):
            rows = rerank_with_constraints(
                question,
                ranking,
                score_map,
                chunks,
                candidate_k=candidate_k,
                filter_enabled=bool(selected["filter_enabled"]),
                filter_strength=float(selected["filter_strength"]),
            )
            final_rankings.append([row["row_id"] for row in rows])
    else:
        final_rankings = [ranking[:max_candidates] for ranking in filtered_candidates]

    print("[DIAG 5/5] classify failures", flush=True)
    rows = []
    failure_counts = Counter()
    stage2_hit5 = 0

    availability = Counter()
    for record, bm25_r, dense_r, wide_rrf_r, hybrid_r, candidate_r, filtered_r, final_r in zip(
        records,
        wide_bm25_rankings,
        wide_dense_rankings,
        wide_rrf_rankings,
        original_hybrid,
        stage2_candidates,
        filtered_candidates,
        final_rankings,
    ):
        gold = gold_documents(record)
        bm25_rank = first_gold_rank(bm25_r, chunks, gold, args.wide_k)
        dense_rank = first_gold_rank(dense_r, chunks, gold, args.wide_k)
        wide_rrf_rank = first_gold_rank(wide_rrf_r, chunks, gold, args.wide_k)
        hybrid_rank = first_gold_rank(hybrid_r, chunks, gold, 50)
        candidate_rank = first_gold_rank(candidate_r, chunks, gold, max_candidates)
        filtered_rank = first_gold_rank(filtered_r, chunks, gold, max_candidates)
        final_rank = first_gold_rank(final_r, chunks, gold, candidate_k)

        hit5 = final_rank is not None and final_rank <= 5
        stage2_hit5 += int(hit5)

        for k in (5, 10, 20, 30, 50):
            if filtered_rank is not None and filtered_rank <= k:
                availability[f"filtered_at_{k}"] += 1
        if bm25_rank is not None:
            availability["bm25_at_wide"] += 1
        if dense_rank is not None:
            availability["dense_at_wide"] += 1
        if wide_rrf_rank is not None:
            availability["rrf_at_wide"] += 1

        failure_type = classify_failure(
            hybrid_rank=hybrid_rank,
            candidate_rank=candidate_rank,
            filtered_rank=filtered_rank,
            final_rank=final_rank,
            wide_rrf_rank=wide_rrf_rank,
            bm25_rank=bm25_rank,
            dense_rank=dense_rank,
            candidate_k=candidate_k,
        )
        if not hit5:
            failure_counts[failure_type] += 1

        rows.append({
            "id": str(record["id"]),
            "question": record["question"],
            "gold_documents": sorted(gold),
            "stage2_hit5": hit5,
            "failure_type": failure_type,
            "ranks": {
                f"bm25_top{args.wide_k}": bm25_rank,
                f"dense_top{args.wide_k}": dense_rank,
                f"wide_rrf_top{args.wide_k}": wide_rrf_rank,
                "current_hybrid_top50": hybrid_rank,
                "stage2_candidate_top50": candidate_rank,
                "after_filter_top50": filtered_rank,
                f"reranker_top{candidate_k}": final_rank,
            },
        })

    n = len(records)
    failures = [row for row in rows if not row["stage2_hit5"]]
    hit5_percent = stage2_hit5 / n * 100.0

    availability_summary = {
        "filtered_gold_available_at_5": availability["filtered_at_5"],
        "filtered_gold_available_at_10": availability["filtered_at_10"],
        "filtered_gold_available_at_20": availability["filtered_at_20"],
        "filtered_gold_available_at_30": availability["filtered_at_30"],
        "filtered_gold_available_at_50": availability["filtered_at_50"],
        f"bm25_gold_available_at_{args.wide_k}": availability["bm25_at_wide"],
        f"dense_gold_available_at_{args.wide_k}": availability["dense_at_wide"],
        f"wide_rrf_gold_available_at_{args.wide_k}": availability["rrf_at_wide"],
    }

    payload = {
        "dataset": str(dataset_path.resolve()),
        "answerable": n,
        "current_stage2": {
            "hit5_count": stage2_hit5,
            "hit5_percent": round(hit5_percent, 4),
            "selected_config": selected,
        },
        "target": {
            "minimum_hit5_count_for_95_percent": 74 if n == 77 else int(n * 0.95 + 0.999999),
            "target_percent": 95.0,
        },
        "candidate_availability": availability_summary,
        "failure_type_counts": dict(failure_counts),
        "failures": failures,
        "all_cases": rows,
    }
    write_json(JSON_PATH, payload)

    lines = []
    lines.append("# Retrieval Stage3 Diagnosis")
    lines.append("")
    lines.append(f"answerable: {n}")
    lines.append(f"current Stage2 Hit@5: {stage2_hit5}/{n} = {hit5_percent:.2f}%")
    target_count = payload["target"]["minimum_hit5_count_for_95_percent"]
    lines.append(f"95% target: >= {target_count}/{n}")
    lines.append("")
    lines.append("## Gold candidate availability")
    for key, value in availability_summary.items():
        lines.append(f"{key}: {value}/{n} = {value/n*100:.2f}%")
    lines.append("")
    lines.append("## Failure types")
    if failure_counts:
        for key, value in failure_counts.most_common():
            lines.append(f"{key}: {value}")
    else:
        lines.append("none")
    lines.append("")
    lines.append("## Failed cases")
    lines.append("ID | BM25 | Dense | RRF100 | Hybrid50 | Candidate50 | Filtered50 | Final | Type")
    lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---")
    for row in failures:
        r = row["ranks"]
        lines.append(
            f"{row['id']} | {fmt_rank(r[f'bm25_top{args.wide_k}'])} | "
            f"{fmt_rank(r[f'dense_top{args.wide_k}'])} | {fmt_rank(r[f'wide_rrf_top{args.wide_k}'])} | "
            f"{fmt_rank(r['current_hybrid_top50'])} | {fmt_rank(r['stage2_candidate_top50'])} | "
            f"{fmt_rank(r['after_filter_top50'])} | {fmt_rank(r[f'reranker_top{candidate_k}'])} | "
            f"{row['failure_type']}"
        )

    TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n[CURRENT]")
    print(f"Stage2 Hit@5: {stage2_hit5}/{n} = {hit5_percent:.2f}%")
    print(f"95% target: >= {target_count}/{n}")

    print("\n[CANDIDATE AVAILABILITY]")
    for key, value in availability_summary.items():
        print(f"{key}: {value}/{n} = {value/n*100:.2f}%")

    print("\n[FAILURE TYPES]")
    for key, value in failure_counts.most_common():
        print(f"{key}: {value}")

    print("\n[FAILED CASES]")
    print("ID | BM25 | Dense | RRF100 | Hybrid50 | Candidate50 | Filtered50 | Final | Type")
    for row in failures:
        r = row["ranks"]
        print(
            f"{row['id']} | {fmt_rank(r[f'bm25_top{args.wide_k}'])} | "
            f"{fmt_rank(r[f'dense_top{args.wide_k}'])} | {fmt_rank(r[f'wide_rrf_top{args.wide_k}'])} | "
            f"{fmt_rank(r['current_hybrid_top50'])} | {fmt_rank(r['stage2_candidate_top50'])} | "
            f"{fmt_rank(r['after_filter_top50'])} | {fmt_rank(r[f'reranker_top{candidate_k}'])} | "
            f"{row['failure_type']}"
        )

    print(f"\nJSON: {JSON_PATH}")
    print(f"TXT : {TXT_PATH}")


if __name__ == "__main__":
    main()
