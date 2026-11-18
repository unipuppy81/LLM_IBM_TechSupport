"""Hybrid Weighted RRF k 최적화: BM25 가중치 2개 x RRF k 4개.

PowerShell (프로젝트 루트):
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_hybrid_rrf_k.py

기존 BM25/Dense 인덱스, YAML, 검색 코드를 변경하지 않습니다.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from dense_search import load_retriever
from document_utils import chunk_filename, filename_key
from evaluate.common import load_records, retrieval_scores
from hybrid_search import load_bm25, search_all
from reranker_search import load_reranker, rerank_candidates
from retrieval_config import load_retrieval_config


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "hybrid_rrf_k_comparison.json"

BM25_WEIGHTS = (0.5, 1.0)
DENSE_WEIGHT = 1.0
RRF_K_VALUES = (10, 30, 60, 100)
CANDIDATE_K = 50
RERANKER_CANDIDATE_K = 10
BASELINE_RRF_K = 60


def weighted_rrf(bm25_ids, dense_ids, bm25_weight, rrf_k):
    """hybrid_search.py의 Weighted RRF와 동일한 정렬 및 동점 처리."""
    scores = {}
    for ranked, weight in (
        (bm25_ids, bm25_weight),
        (dense_ids, DENSE_WEIGHT),
    ):
        for rank, row_id in enumerate(ranked, start=1):
            scores[row_id] = scores.get(row_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores, key=lambda row_id: (-scores[row_id], row_id))


def evaluate_rankings(rankings, records, chunks):
    """기존 retrieval_scores()를 이용해 동일 지표 산출."""
    if len(rankings) != len(records):
        raise ValueError("검색 결과와 평가 질문 수가 다릅니다.")

    totals = Counter()
    for record, row_ids in zip(records, rankings):
        gold = {filename_key(context["filename"]) for context in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 문서 파일명이 없습니다: {record['id']}")

        ranked_names = [
            filename_key(chunk_filename(chunks[row_id]))
            for row_id in row_ids[:10]
        ]
        metrics = retrieval_scores(gold, ranked_names, k=5)
        for key in ("hit", "recall", "rr", "precision", "ndcg"):
            totals[key] += metrics[key]

    n = len(records)
    return {
        "hit_at_5_percent": round(totals["hit"] / n * 100, 2),
        "recall_at_5_percent": round(totals["recall"] / n * 100, 2),
        "mrr_at_10": round(totals["rr"] / n, 4),
        "precision_at_5": round(totals["precision"] / n, 4),
        "ndcg_at_5": round(totals["ndcg"] / n, 4),
    }


def check_fixed_configuration():
    config = load_retrieval_config()
    hybrid = config["hybrid"]
    reranker = config["reranker"]

    if hybrid["candidate_k"] != CANDIDATE_K:
        raise ValueError(f"hybrid.candidate_k를 {CANDIDATE_K}으로 유지하세요.")
    if hybrid["rrf_k"] != BASELINE_RRF_K:
        raise ValueError(f"실험 중 YAML hybrid.rrf_k는 {BASELINE_RRF_K}으로 유지하세요.")
    if hybrid["bm25_weight"] != 1.0 or hybrid["dense_weight"] != 1.0:
        raise ValueError("실험 중 YAML BM25/Dense 가중치를 각각 1.0으로 유지하세요.")
    if reranker["candidate_k"] != RERANKER_CANDIDATE_K:
        raise ValueError(f"reranker.candidate_k를 {RERANKER_CANDIDATE_K}으로 유지하세요.")
    if not config["dense"]["pca"]["enabled"] or config["dense"]["pca"]["dimension"] != 192:
        raise ValueError("PCA 192차원 설정을 유지하세요.")
    if config["dense"]["faiss"]["index_type"].lower() != "hnsw":
        raise ValueError("FAISS HNSW 설정을 유지하세요.")


def main():
    parser = argparse.ArgumentParser(description="Hybrid RRF k benchmark")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"결과 파일이 이미 있습니다: {args.output} (--overwrite로 재실행)")
    check_fixed_configuration()

    data = load_records("clean")
    records = [record for record in data if not record["is_impossible"]]
    if not records:
        raise ValueError("답변 가능한 평가 질문이 없습니다.")

    questions = [record["question"] for record in records]
    print(f"Validation: {len(records)} questions / {len(data)} total")
    print(f"Candidate K: {CANDIDATE_K} / Reranker candidate K: {RERANKER_CANDIDATE_K}")
    print(f"BM25 weights: {BM25_WEIGHTS} / Dense weight: {DENSE_WEIGHT}")
    print(f"RRF K values: {RRF_K_VALUES}")

    # BM25 및 Dense 검색을 한 번 실행한 뒤 8개 조합에서 동일한 후보 순위를 사용.
    model, index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    raw_results = search_all(model, index, bm25_index, questions)
    retrieval_pairs = [(row[0], row[1]) for row in raw_results]

    reranker = load_reranker()
    results = []
    total_cases = len(BM25_WEIGHTS) * len(RRF_K_VALUES)

    for weight in BM25_WEIGHTS:
        for rrf_k in RRF_K_VALUES:
            hybrid_rankings = [
                weighted_rrf(bm25_ids, dense_ids, weight, rrf_k)
                for bm25_ids, dense_ids in retrieval_pairs
            ]

            # 기존 검색과 정확히 같은 조건에서 검색 순위가 일치하는지 검증.
            if weight == 1.0 and rrf_k == BASELINE_RRF_K:
                if any(ranking != raw[2] for ranking, raw in zip(hybrid_rankings, raw_results)):
                    raise RuntimeError("기존 Hybrid 결과와 RRF 재계산 결과가 일치하지 않습니다.")

            hybrid_metrics = evaluate_rankings(hybrid_rankings, records, chunks)
            reranked = rerank_candidates(reranker, questions, hybrid_rankings, chunks)
            reranker_rankings = [
                [entry["row_id"] for entry in question_results]
                for question_results in reranked
            ]
            reranker_metrics = evaluate_rankings(reranker_rankings, records, chunks)

            row = {
                "bm25_weight": weight,
                "dense_weight": DENSE_WEIGHT,
                "rrf_k": rrf_k,
                "hybrid": hybrid_metrics,
                "reranker": reranker_metrics,
            }
            results.append(row)
            h, r = hybrid_metrics, reranker_metrics
            print(
                f"[{len(results)}/{total_cases}] BM25={weight:.2f} Dense=1.00 "
                f"RRF_K={rrf_k} | "
                f"Hybrid Hit@5={h['hit_at_5_percent']:.2f}% "
                f"MRR@10={h['mrr_at_10']:.4f} nDCG@5={h['ndcg_at_5']:.4f} | "
                f"Reranker Hit@5={r['hit_at_5_percent']:.2f}% "
                f"MRR@10={r['mrr_at_10']:.4f} nDCG@5={r['ndcg_at_5']:.4f}",
                flush=True,
            )

    # 최종 Reranker 품질 우선. 동점이면 기존 RRF k=60에 가까운 설정 우선.
    best = max(
        results,
        key=lambda row: (
            row["reranker"]["hit_at_5_percent"],
            row["reranker"]["mrr_at_10"],
            row["reranker"]["ndcg_at_5"],
            -abs(row["rrf_k"] - BASELINE_RRF_K),
            -abs(row["bm25_weight"] - 1.0),
        ),
    )

    output = {
        "dataset": "clean",
        "total": len(data),
        "answerable_evaluated": len(records),
        "impossible_skipped": len(data) - len(records),
        "candidate_k": CANDIDATE_K,
        "reranker_candidate_k": RERANKER_CANDIDATE_K,
        "bm25_weights": list(BM25_WEIGHTS),
        "dense_weight": DENSE_WEIGHT,
        "rrf_k_values": list(RRF_K_VALUES),
        "results": results,
        "best": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBest: BM25 weight =", best["bm25_weight"], "/ RRF K =", best["rrf_k"])
    print("Best Reranker:", json.dumps(best["reranker"], ensure_ascii=False))
    print("Saved:", args.output.resolve())
    print("YAML, BM25, PCA, and FAISS indexes were not modified.")


if __name__ == "__main__":
    main()
