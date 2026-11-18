
"""BM25 / Dense 가중치 7개 조합 평가 (Hybrid + Reranker).

실행 (프로젝트 루트 PowerShell):
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_hybrid_weights.py

기존 인덱스, YAML, 검색 코드는 변경하지 않습니다.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from document_utils import chunk_filename, filename_key
from evaluate.common import load_records, retrieval_scores
from dense_search import load_retriever
from hybrid_search import load_bm25, search_all
from reranker_search import load_reranker, rerank_candidates
from retrieval_config import load_retrieval_config


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "hybrid_weight_comparison.json"

BM25_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DENSE_WEIGHT = 1.0

EXPECTED_CANDIDATE_K = 50
EXPECTED_RRF_K = 60
EXPECTED_RERANKER_K = 10


def weighted_rrf(bm25_ids, dense_ids, bm25_weight, rrf_k):
    """기존 hybrid_search.search_all()과 동일한 Weighted RRF 계산."""
    scores = {}

    for ranked, weight in (
        (bm25_ids, bm25_weight),
        (dense_ids, DENSE_WEIGHT),
    ):
        for rank, row_id in enumerate(ranked, start=1):
            scores[row_id] = (
                scores.get(row_id, 0.0)
                + weight / (rrf_k + rank)
            )

    return sorted(
        scores,
        key=lambda row_id: (-scores[row_id], row_id)
    )


def evaluate_rankings(rankings, records, chunks):
    """기존 evaluate_retrieval.py와 동일한 파일명 기반 평가."""
    if len(rankings) != len(records):
        raise ValueError("검색 결과 개수와 평가 질문 수가 다릅니다.")

    totals = Counter()

    for record, row_ids in zip(records, rankings):
        gold = {
            filename_key(c["filename"])
            for c in record["contexts"]
        }

        if not gold:
            raise ValueError(f"정답 파일명이 없습니다: {record['id']}")

        ranked_names = [
            filename_key(chunk_filename(chunks[i]))
            for i in row_ids[:10]
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


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 / Dense weight benchmark"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 결과 JSON 덮어쓰기 허용",
    )

    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(
            f"결과 파일이 이미 있습니다: {args.output} "
            "(--overwrite로 재실행)"
        )

    # 실험 고정 조건 검증
    config = load_retrieval_config()
    hybrid = config["hybrid"]
    reranker_config = config["reranker"]

    if hybrid["candidate_k"] != EXPECTED_CANDIDATE_K:
        raise ValueError("hybrid.candidate_k는 50으로 고정해야 합니다.")

    if hybrid["rrf_k"] != EXPECTED_RRF_K:
        raise ValueError("hybrid.rrf_k는 60으로 고정해야 합니다.")

    if reranker_config["candidate_k"] != EXPECTED_RERANKER_K:
        raise ValueError(
            "reranker.candidate_k는 10으로 고정해야 합니다."
        )

    if (
        hybrid["bm25_weight"] != 1.0
        or hybrid["dense_weight"] != 1.0
    ):
        raise ValueError(
            "기준 실험을 위해 YAML 가중치를 모두 1.0으로 유지하세요."
        )

    # 평가 데이터 준비
    data = load_records("clean")
    records = [r for r in data if not r["is_impossible"]]

    if not records:
        raise ValueError("평가할 답변 가능 질문이 없습니다.")

    print(f"Validation: {len(records)} questions / {len(data)} total")
    print(
        f"Candidate K: {EXPECTED_CANDIDATE_K} "
        f"/ RRF K: {EXPECTED_RRF_K}"
    )
    print(
        f"BM25 weights: {BM25_WEIGHTS} "
        f"/ Dense weight: {DENSE_WEIGHT}"
    )

    # 1. Dense 및 BM25 로드
    model, index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)

    questions = [r["question"] for r in records]

    # 2. 실제 검색은 한 번만 수행
    raw_results = search_all(
        model,
        index,
        bm25_index,
        questions,
    )

    # BM25 / Dense 순위를 추출하여 재사용
    retrieval_pairs = [
        (row[0], row[1])
        for row in raw_results
    ]

    # 3. Reranker는 한 번만 로드
    reranker = load_reranker()

    results = []

    # 4. 가중치 7개 조합 실험
    for number, weight in enumerate(BM25_WEIGHTS, start=1):

        # 가중치별 Hybrid 순위 재계산
        hybrid_rankings = [
            weighted_rrf(
                bm25_ids,
                dense_ids,
                weight,
                EXPECTED_RRF_K,
            )
            for bm25_ids, dense_ids in retrieval_pairs
        ]

        # 기준 가중치에서 기존 Hybrid 결과와 일치하는지 검증
        if weight == 1.0:
            if any(
                ranking != raw[2]
                for ranking, raw in zip(hybrid_rankings, raw_results)
            ):
                raise RuntimeError(
                    "기존 Hybrid RRF와 벤치마크 결과가 다릅니다."
                )

        # Hybrid 평가
        hybrid_metrics = evaluate_rankings(
            hybrid_rankings,
            records,
            chunks,
        )

        # Hybrid 상위 10개 후보를 Reranker로 재정렬
        reranked = rerank_candidates(
            reranker,
            questions,
            hybrid_rankings,
            chunks,
        )

        reranker_rankings = [
            [
                entry["row_id"]
                for entry in question_results
            ]
            for question_results in reranked
        ]

        # Reranker 평가
        reranker_metrics = evaluate_rankings(
            reranker_rankings,
            records,
            chunks,
        )

        result = {
            "bm25_weight": weight,
            "dense_weight": DENSE_WEIGHT,
            "hybrid": hybrid_metrics,
            "reranker": reranker_metrics,
        }

        results.append(result)

        h = hybrid_metrics
        r = reranker_metrics

        print(
            f"[{number}/{len(BM25_WEIGHTS)}] "
            f"BM25={weight:.2f}, Dense=1.00 | "
            f"Hybrid Hit@5={h['hit_at_5_percent']:.2f}% "
            f"MRR@10={h['mrr_at_10']:.4f} "
            f"nDCG@5={h['ndcg_at_5']:.4f} | "
            f"Reranker Hit@5={r['hit_at_5_percent']:.2f}% "
            f"MRR@10={r['mrr_at_10']:.4f} "
            f"nDCG@5={r['ndcg_at_5']:.4f}",
            flush=True,
        )

    # 5. 최적 가중치 선택
    # 우선순위: Reranker Hit@5 → MRR@10 → nDCG@5
    best = max(
        results,
        key=lambda row: (
            row["reranker"]["hit_at_5_percent"],
            row["reranker"]["mrr_at_10"],
            row["reranker"]["ndcg_at_5"],
            -abs(row["bm25_weight"] - 1.0),
        ),
    )

    # 6. JSON 결과 저장
    payload = {
        "dataset": "clean",
        "total": len(data),
        "answerable_evaluated": len(records),
        "impossible_skipped": len(data) - len(records),
        "candidate_k": EXPECTED_CANDIDATE_K,
        "rrf_k": EXPECTED_RRF_K,
        "reranker_candidate_k": EXPECTED_RERANKER_K,
        "dense_weight": DENSE_WEIGHT,
        "results": results,
        "best_bm25_weight": best["bm25_weight"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nBest BM25 weight:", best["bm25_weight"])
    print(
        "Best Reranker:",
        json.dumps(best["reranker"], ensure_ascii=False),
    )
    print("Saved:", args.output.resolve())
    print("YAML, BM25, PCA, FAISS indexes unchanged.")


if __name__ == "__main__":
    main()
