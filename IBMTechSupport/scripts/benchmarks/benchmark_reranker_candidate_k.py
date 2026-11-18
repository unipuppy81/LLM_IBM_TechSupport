"""Reranker Candidate K 비교: 5, 10, 20, 30, 50.

프로젝트 루트 PowerShell 실행:
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_reranker_candidate_k.py

기존 Hybrid 검색 결과는 한 번만 생성하며, 각 K에 대해 Reranker 추론을
별도로 실행한다. YAML, 인덱스, 기존 검색 스크립트는 수정하지 않는다.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dense_search import load_retriever
from document_utils import chunk_filename, filename_key
from evaluate.common import load_records, retrieval_scores
from hybrid_search import load_bm25, search_all
from reranker_search import get_chunk_text, load_reranker
from retrieval_config import load_retrieval_config


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "reranker_candidate_k_comparison.json"
CANDIDATE_K_VALUES = (5, 10, 20, 30, 50)

# 이전 단계에서 확정한 검색 조건을 검증한다.
EXPECTED_HYBRID = {
    "candidate_k": 50,
    "rrf_k": 10,
    "bm25_weight": 0.5,
    "dense_weight": 1.0,
    "max_chunks_per_document": 1,
}
EXPECTED_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
EXPECTED_MAX_LENGTH = 512
EXPECTED_BATCH_SIZE = 16
EXPECTED_TOP_K = 5
BASELINE_METRICS = {"hit_at_5_percent": 66.67, "mrr_at_10": 0.5231, "ndcg_at_5": 0.5537}


def validate_configuration(config):
    """Hybrid 조건이 변경되었으면 다른 실험이 되므로 실행을 중단한다."""
    hybrid = config["hybrid"]
    for name, expected in EXPECTED_HYBRID.items():
        if hybrid.get(name) != expected:
            raise ValueError(
                f"hybrid.{name}: expected={expected}, actual={hybrid.get(name)}"
            )

    reranker = config["reranker"]
    expected = {
        "model": EXPECTED_MODEL,
        "candidate_k": 10,  # 실제 YAML은 기존값 유지. 실험 스크립트에서만 K를 변경.
        "max_length": EXPECTED_MAX_LENGTH,
        "batch_size": EXPECTED_BATCH_SIZE,
        "top_k": EXPECTED_TOP_K,
    }
    for name, value in expected.items():
        if reranker.get(name) != value:
            raise ValueError(
                f"reranker.{name}: expected={value}, actual={reranker.get(name)}"
            )


def rerank_with_k(model, questions, hybrid_rankings, chunks, candidate_k, batch_size):
    """기존 rerank_candidates()의 정렬 규칙을 유지하면서 후보 K만 변경.

    원본 함수는 YAML의 candidate_k=10을 사용하므로 그대로 호출하면
    20/30/50개를 비교할 수 없다. 이 함수는 실험용으로만 사용한다.
    """
    if len(questions) != len(hybrid_rankings):
        raise ValueError("질문 수와 Hybrid 결과 수가 일치하지 않습니다.")

    pairs = []
    positions = []
    counts = []

    for question_index, (question, ranked_ids) in enumerate(zip(questions, hybrid_rankings)):
        selected = ranked_ids[:candidate_k]
        counts.append(len(selected))
        for original_rank, row_id in enumerate(selected):
            pairs.append((question, get_chunk_text(chunks[row_id])))
            positions.append((question_index, int(row_id), original_rank))

    if not pairs or any(count == 0 for count in counts):
        raise ValueError("Reranker 입력 후보가 없는 질문이 있습니다.")

    # CPU 측 준비 및 GPU 추론과 결과 정렬까지 포함해 측정한다.
    predictions = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    scores = np.asarray(predictions).reshape(-1)
    if scores.size != len(positions) or not np.isfinite(scores).all():
        raise ValueError("Reranker 점수 개수가 다르거나 NaN/Inf가 포함되어 있습니다.")

    grouped = [[] for _ in questions]
    for (question_index, row_id, original_rank), score in zip(positions, scores):
        grouped[question_index].append(
            {"row_id": row_id, "score": float(score), "original_rank": original_rank}
        )

    for group in grouped:
        group.sort(key=lambda item: (-item["score"], item["original_rank"]))

    return [[entry["row_id"] for entry in group] for group in grouped], counts


def evaluate_rankings(rankings, records, chunks):
    """기존 evaluate_retrieval.py와 동일한 파일명 기준 지표 계산."""
    if len(rankings) != len(records):
        raise ValueError("평가 질문 수와 순위 결과 수가 다릅니다.")

    totals = Counter()
    for record, row_ids in zip(records, rankings):
        gold = {filename_key(context["filename"]) for context in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 문서가 없습니다: {record['id']}")

        ranked_names = [
            filename_key(chunk_filename(chunks[row_id]))
            for row_id in row_ids[:10]
        ]
        values = retrieval_scores(gold, ranked_names, k=5)
        for key in ("hit", "recall", "rr", "precision", "ndcg"):
            totals[key] += values[key]

    n = len(records)
    return {
        "hit_at_5_percent": round(totals["hit"] / n * 100, 2),
        "recall_at_5_percent": round(totals["recall"] / n * 100, 2),
        "mrr_at_10": round(totals["rr"] / n, 4),
        "precision_at_5": round(totals["precision"] / n, 4),
        "ndcg_at_5": round(totals["ndcg"] / n, 4),
    }


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Reranker Candidate K 벤치마크")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"결과 파일이 이미 존재합니다: {args.output} (재실행: --overwrite)")

    config = load_retrieval_config()
    validate_configuration(config)

    data = load_records("clean")
    records = [row for row in data if not row["is_impossible"]]
    if not records:
        raise ValueError("평가할 답변 가능 질문이 없습니다.")
    questions = [row["question"] for row in records]

    print(f"Validation: {len(records)} answerable / {len(data)} total")
    print("Hybrid (fixed):", config["hybrid"])
    print("Reranker model:", EXPECTED_MODEL)
    print(f"Candidate K: {CANDIDATE_K_VALUES}")
    print(f"Max length: {EXPECTED_MAX_LENGTH} | Batch size: {EXPECTED_BATCH_SIZE}")
    print("YAML reranker.candidate_k remains 10; only experiment uses other K values.")

    # 동일 검색 입력 확보: Dense + BM25 + Hybrid는 딱 한 번 실행.
    embedding_model, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    retrieval_start = time.perf_counter()
    hybrid_results = search_all(embedding_model, faiss_index, bm25_index, questions)
    retrieval_seconds = time.perf_counter() - retrieval_start
    hybrid_rankings = [item[2] for item in hybrid_results]
    print(f"Shared Hybrid retrieval: {retrieval_seconds:.3f} s")

    # Reranker 모델 로딩은 시간 비교에서 제외.
    reranker = load_reranker()

    # 최초 CUDA 커널 실행/모델 준비 시간이 첫 실험에만 추가되지 않도록 사전 예열.
    warmup_pair = (questions[0], get_chunk_text(chunks[hybrid_rankings[0][0]]))
    reranker.predict(
        [warmup_pair], batch_size=1, show_progress_bar=False, convert_to_numpy=True
    )
    synchronize_cuda()

    results = []
    for index, candidate_k in enumerate(CANDIDATE_K_VALUES, start=1):
        synchronize_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        ranked_ids, counts = rerank_with_k(
            reranker, questions, hybrid_rankings, chunks, candidate_k, EXPECTED_BATCH_SIZE
        )

        synchronize_cuda()
        seconds = time.perf_counter() - start
        metrics = evaluate_rankings(ranked_ids, records, chunks)

        result = {
            "candidate_k": candidate_k,
            "metrics": metrics,
            "rerank_seconds": round(seconds, 3),
            "rerank_ms_per_question": round(seconds / len(records) * 1000, 3),
            "pairs_scored": sum(counts),
            "average_candidates": round(sum(counts) / len(counts), 2),
            "queries_with_fewer_candidates": sum(n < candidate_k for n in counts),
            "process_peak_allocated_gpu_gib": (
                round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
                if torch.cuda.is_available() else None
            ),
        }
        results.append(result)

        print(
            f"[{index}/{len(CANDIDATE_K_VALUES)}] K={candidate_k:<2} | "
            f"Hit@5={metrics['hit_at_5_percent']:.2f}% "
            f"MRR@10={metrics['mrr_at_10']:.4f} "
            f"nDCG@5={metrics['ndcg_at_5']:.4f} | "
            f"Rerank={result['rerank_ms_per_question']:.3f} ms/query | "
            f"Pairs={result['pairs_scored']} "
            f"Short={result['queries_with_fewer_candidates']}",
            flush=True,
        )

    # 품질 우선, 동률이면 더 작은 K (실제 추론 비용 절감).
    best = max(
        results,
        key=lambda row: (
            row["metrics"]["hit_at_5_percent"],
            row["metrics"]["mrr_at_10"],
            row["metrics"]["ndcg_at_5"],
            -row["candidate_k"],
        ),
    )

    baseline = next(row for row in results if row["candidate_k"] == 10)
    baseline_matches = all(
        baseline["metrics"][name] == value
        for name, value in BASELINE_METRICS.items()
    )
    if not baseline_matches:
        print("WARNING: K=10 결과가 이전 통합 평가와 다릅니다. 환경/순위를 확인하세요.")

    payload = {
        "dataset": "clean",
        "total": len(data),
        "answerable_evaluated": len(records),
        "impossible_skipped": len(data) - len(records),
        "hybrid_config": config["hybrid"],
        "reranker_model": EXPECTED_MODEL,
        "max_length": EXPECTED_MAX_LENGTH,
        "batch_size": EXPECTED_BATCH_SIZE,
        "top_k": EXPECTED_TOP_K,
        "shared_retrieval_seconds": round(retrieval_seconds, 3),
        "reference_k10_metrics_match": baseline_matches,
        "results": results,
        "best_candidate_k": best["candidate_k"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBest Candidate K:", best["candidate_k"])
    print("Best metrics:", json.dumps(best["metrics"], ensure_ascii=False))
    print("Saved:", args.output.resolve())
    print("Production YAML, indexes, and search scripts were not modified.")


if __name__ == "__main__":
    main()
