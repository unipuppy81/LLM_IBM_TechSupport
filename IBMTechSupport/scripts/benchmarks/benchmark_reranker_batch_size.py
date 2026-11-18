"""Reranker Batch Size 최적화: 8 / 16 / 32 / 64.

프로젝트 루트 PowerShell:
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_reranker_batch_size.py

기존 실험 재실행:
    python IBMTechSupport/scripts/benchmarks/benchmark_reranker_batch_size.py --overwrite
중단 후 재개:
    python IBMTechSupport/scripts/benchmarks/benchmark_reranker_batch_size.py --resume

Hybrid 검색은 단 한 번 실행하며, 모든 Batch Size에 동일한 질문/청크 쌍을 전달한다.
기존 YAML, 모델, BM25/PCA/FAISS 인덱스와 검색 코드는 변경하지 않는다.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from dense_search import load_retriever
from document_utils import chunk_filename, filename_key
from evaluate.common import load_records, retrieval_scores
from hybrid_search import load_bm25, search_all
from reranker_search import get_chunk_text, load_reranker, rerank_candidates
from retrieval_config import load_retrieval_config

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "reranker_batch_size_comparison.json"
BATCH_SIZES = (8, 16, 32, 64)

# 이전 단계에서 결정한 고정 조건
EXPECTED_HYBRID = {
    "candidate_k": 50,
    "rrf_k": 10,
    "bm25_weight": 0.5,
    "dense_weight": 1.0,
    "max_chunks_per_document": 1,
}
EXPECTED_RERANKER = {
    "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "candidate_k": 10,
    "top_k": 5,
    "batch_size": 16,  # YAML은 변경하지 않고 실험 중에만 override
    "max_length": 512,
}
BASELINE_METRICS = {
    "hit_at_5_percent": 66.67,
    "mrr_at_10": 0.5231,
    "ndcg_at_5": 0.5537,
}


def validate_config(config):
    """Batch Size 외의 조건이 이전 최적화와 같은지 확인한다."""
    for section, expected in (
        ("hybrid", EXPECTED_HYBRID),
        ("reranker", EXPECTED_RERANKER),
    ):
        for key, value in expected.items():
            actual = config[section].get(key)
            if actual != value:
                raise ValueError(f"{section}.{key}: expected={value!r}, actual={actual!r}")


def dataset_fingerprint(records):
    content = [
        {
            "id": row["id"],
            "question": row["question"],
            "is_impossible": row["is_impossible"],
            "contexts": row["contexts"],
        }
        for row in records
    ]
    data = json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def save_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_rankings(records, rankings, chunks):
    """기존 evaluate_retrieval과 동일한 filename 기반 지표."""
    if len(records) != len(rankings):
        raise ValueError("평가 질문 수와 결과 개수가 일치하지 않습니다.")

    totals = Counter()
    for record, row_ids in zip(records, rankings):
        gold = {filename_key(context["filename"]) for context in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 문서가 없습니다: {record['id']}")
        ranked_names = [
            filename_key(chunk_filename(chunks[int(row_id)]))
            for row_id in row_ids[:10]
        ]
        scores = retrieval_scores(gold, ranked_names, k=5)
        for key in ("hit", "recall", "rr", "precision", "ndcg"):
            totals[key] += scores[key]

    n = len(records)
    return {
        "hit_at_5_percent": round(totals["hit"] / n * 100, 2),
        "recall_at_5_percent": round(totals["recall"] / n * 100, 2),
        "mrr_at_10": round(totals["rr"] / n, 4),
        "precision_at_5": round(totals["precision"] / n, 4),
        "ndcg_at_5": round(totals["ndcg"] / n, 4),
    }


def best_result(results):
    """품질 우선, 동률일 때 실측 추론 시간이 빠른 조건을 선택."""
    return max(
        results,
        key=lambda row: (
            row["metrics"]["hit_at_5_percent"],
            row["metrics"]["mrr_at_10"],
            row["metrics"]["ndcg_at_5"],
            -row["rerank_ms_per_query_median"],
            -row["batch_size"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Reranker Batch Size 벤치마크")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats는 1 이상이어야 합니다.")
    if args.resume and args.overwrite:
        parser.error("--resume과 --overwrite는 동시에 사용할 수 없습니다.")
    if args.output.exists() and not (args.resume or args.overwrite):
        parser.error(f"결과 파일이 이미 있습니다: {args.output} (--resume 또는 --overwrite)")
    if args.resume and not args.output.exists():
        parser.error(f"이어 실행할 결과 파일이 없습니다: {args.output}")

    config = load_retrieval_config()
    validate_config(config)

    all_records = load_records("clean")
    records = [r for r in all_records if not r["is_impossible"]]
    if not records:
        raise ValueError("평가할 답변 가능 질문이 없습니다.")
    questions = [r["question"] for r in records]

    signature = {
        "version": 1,
        "dataset_sha256": dataset_fingerprint(all_records),
        "hybrid": EXPECTED_HYBRID,
        "reranker": EXPECTED_RERANKER,
        "batch_sizes": list(BATCH_SIZES),
        "repeats": args.repeats,
    }
    if args.resume:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("저장된 실험 조건이 현재와 다릅니다. 새 실험은 --overwrite 사용.")
    else:
        payload = {
            "signature": signature,
            "dataset": "clean",
            "total": len(all_records),
            "answerable_evaluated": len(records),
            "impossible_skipped": len(all_records) - len(records),
            "results": [],
        }
    completed = {r["batch_size"] for r in payload["results"]}

    print(f"Validation: {len(records)} answerable / {len(all_records)} total")
    print("Hybrid (fixed):", config["hybrid"])
    print("Reranker (fixed):", config["reranker"])
    print("Batch Size tests:", BATCH_SIZES, "| repeats:", args.repeats)
    print("YAML batch_size remains 16; only the experiment overrides it.")

    # 1. Hybrid는 1회 검색. 모든 조건에 동일한 10개 후보를 사용한다.
    embedder, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    start = time.perf_counter()
    results = search_all(embedder, faiss_index, bm25_index, questions)
    hybrid_rankings = [row[2] for row in results]
    retrieval_seconds = time.perf_counter() - start
    candidate_k = EXPECTED_RERANKER["candidate_k"]
    if any(len(ranking) < candidate_k for ranking in hybrid_rankings):
        raise RuntimeError("일부 질문의 Hybrid 후보가 10개 미만입니다.")
    print(f"Shared Hybrid retrieval: {retrieval_seconds:.3f} s")

    # 검색 모델은 사용을 마쳤으므로 가능한 GPU 메모리에서 해제한다.
    del embedder, faiss_index, bm25_index, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Reranker는 한 번 로드. Batch Size만 변경한다.
    reranker = load_reranker()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_pairs = len(questions) * candidate_k

    for idx, batch_size in enumerate(BATCH_SIZES, start=1):
        if batch_size in completed:
            print(f"[{idx}/{len(BATCH_SIZES)}] SKIP (saved): Batch Size={batch_size}")
            continue

        print(f"\n[{idx}/{len(BATCH_SIZES)}] Batch Size={batch_size}", flush=True)
        # 예열 시간은 측정에서 제외한다.
        warmup_pairs = [
            (questions[0], get_chunk_text(chunks[row_id]))
            for row_id in hybrid_rankings[0][:min(batch_size, candidate_k)]
        ]
        reranker.predict(
            warmup_pairs, batch_size=batch_size,
            convert_to_numpy=True, show_progress_bar=False,
        )
        synchronize()

        measured_ms = []
        peak_alloc_gib = []
        metrics = None
        first_rankings = None
        identical_rankings = True

        for repeat in range(1, args.repeats + 1):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            synchronize()
            started = time.perf_counter()
            ranked = rerank_candidates(
                reranker,
                questions,
                hybrid_rankings,
                chunks,
                batch_size=batch_size,
            )
            synchronize()
            elapsed = time.perf_counter() - started
            measured_ms.append(elapsed * 1000 / len(questions))

            if device == "cuda":
                peak_alloc_gib.append(torch.cuda.max_memory_allocated() / (1024**3))

            ranked_ids = [
                [int(item["row_id"]) for item in group] for group in ranked
            ]
            if first_rankings is None:
                first_rankings = ranked_ids
                metrics = evaluate_rankings(records, ranked_ids, chunks)
            elif ranked_ids != first_rankings:
                identical_rankings = False

            print(
                f"  repeat {repeat}/{args.repeats}: "
                f"{measured_ms[-1]:.3f} ms/query",
                flush=True,
            )

        median_ms = round(statistics.median(measured_ms), 3)
        row = {
            "batch_size": batch_size,
            "metrics": metrics,
            "rerank_ms_per_query_median": median_ms,
            "rerank_ms_per_query_repeats": [round(x, 3) for x in measured_ms],
            "peak_cuda_allocated_gib_max": (
                round(max(peak_alloc_gib), 3) if peak_alloc_gib else None
            ),
            "pairs_scored": total_pairs,
            "rankings_consistent_across_repeats": identical_rankings,
            "question_ids": [r["id"] for r in records],
            "rankings_top_10": first_rankings,
        }
        payload["results"].append(row)
        payload["complete"] = len(payload["results"]) == len(BATCH_SIZES)
        payload["best_batch_size_so_far"] = best_result(payload["results"])["batch_size"]
        save_atomic(args.output, payload)

        print(
            f"Batch={batch_size:<2} | "
            f"Hit@5={metrics['hit_at_5_percent']:.2f}% "
            f"MRR@10={metrics['mrr_at_10']:.4f} "
            f"nDCG@5={metrics['ndcg_at_5']:.4f} | "
            f"Median={median_ms:.3f} ms/query | "
            f"Peak VRAM={row['peak_cuda_allocated_gib_max']} GiB | "
            f"Stable rankings={identical_rankings}",
            flush=True,
        )
        print("Saved:", args.output.resolve(), flush=True)

    # 3. 검색 품질을 우선으로 최적 Batch Size 선택.
    best = best_result(payload["results"])
    baseline = next((r for r in payload["results"] if r["batch_size"] == 16), None)
    if baseline:
        ok = all(baseline["metrics"].get(k) == v for k, v in BASELINE_METRICS.items())
        print("Baseline Batch=16 reproduced:", ok)
        if not ok:
            print("WARNING: 이전 기준 성능이 재현되지 않았습니다. 조건과 환경을 확인하세요.")

    print("\nBest Batch Size (validation):", best["batch_size"])
    print("Best metrics:", json.dumps(best["metrics"], ensure_ascii=False))
    print("Completed:", len(payload["results"]), "/", len(BATCH_SIZES))
    print("YAML, models, search scripts and indexes were not modified.")


if __name__ == "__main__":
    main()
