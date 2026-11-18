"""Hybrid Fusion 실험: Weighted RRF vs Min-Max 점수 결합.

프로젝트 루트 PowerShell에서 실행:
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_hybrid_fusion.py

- PCA 192 / HNSW(M=32, efSearch=128) 고정
- BM25 / Dense 각각 상위 50개를 한 번만 검색하여 두 Fusion 방식에 재사용
- BM25=0.5, Dense=1.0, RRF K=10; Reranker 입력 상위 10개 고정
- YAML, 모델, 인덱스, 원본 데이터는 변경하지 않음
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

# benchmarks/에서 직접 실행해도 기존 scripts/의 모듈을 import할 수 있도록 설정
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bm25_search import retrieve as bm25_retrieve
from dense_search import encode_questions, load_retriever
from document_utils import chunk_filename, filename_key
from evaluate.common import load_records, retrieval_scores
from hybrid_search import load_bm25
from reranker_search import load_reranker, rerank_candidates
from retrieval_config import load_retrieval_config


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "hybrid_fusion_comparison.json"

BM25_WEIGHT = 0.5
DENSE_WEIGHT = 1.0
RRF_K = 10
CANDIDATE_K = 50
RERANKER_CANDIDATE_K = 10


def validate_settings():
    """앞선 실험에서 고정한 검색 환경을 확인. Hybrid YAML은 수정하지 않음."""
    config = load_retrieval_config()
    dense = config["dense"]
    reranker = config["reranker"]

    pca = dense.get("pca", {})
    faiss_config = dense.get("faiss", {})

    if not pca.get("enabled") or int(pca.get("dimension", 0)) != 192:
        raise ValueError("dense.pca를 enabled=true, dimension=192로 유지해야 합니다.")
    if str(faiss_config.get("index_type", "")).lower() != "hnsw":
        raise ValueError("dense.faiss.index_type=hnsw를 유지해야 합니다.")
    if int(faiss_config.get("m", 0)) != 32:
        raise ValueError("dense.faiss.m=32를 유지해야 합니다.")
    if int(faiss_config.get("ef_search", 0)) != 128:
        raise ValueError("dense.faiss.ef_search=128을 유지해야 합니다.")
    if int(reranker["candidate_k"]) != RERANKER_CANDIDATE_K:
        raise ValueError("reranker.candidate_k=10을 유지해야 합니다.")

    print("Current YAML hybrid settings:", config["hybrid"])
    print("Experiment overrides (YAML unchanged):")
    print(f"  BM25={BM25_WEIGHT}, Dense={DENSE_WEIGHT}, RRF_K={RRF_K}")
    print(f"  Hybrid Candidate K={CANDIDATE_K}, Reranker Candidate K={RERANKER_CANDIDATE_K}")
    print("  Fusion methods: weighted_rrf / minmax_score")


def weighted_rrf(bm25_results, dense_results):
    """기존 hybrid_search.py와 동일한 Weighted RRF 및 동점 정렬."""
    scores = {}
    for results, weight in (
        (bm25_results, BM25_WEIGHT),
        (dense_results, DENSE_WEIGHT),
    ):
        for rank, (row_id, _score) in enumerate(results, 1):
            scores[row_id] = scores.get(row_id, 0.0) + weight / (RRF_K + rank)

    return sorted(scores, key=lambda row_id: (-scores[row_id], row_id))


def normalize_minmax(results):
    """질문별·검색기별 점수를 [0,1]로 정규화.

    BM25는 score > 0 문서만 포함하고, Dense는 반환된 모든 후보를 포함.
    유효 후보가 한 개이거나 후보 점수가 전부 같으면 해당 후보에 1.0을 부여.
    """
    if not results:
        return {}

    values = [float(score) for _row_id, score in results]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("검색 점수에 NaN/Inf가 포함되어 있습니다.")

    low, high = min(values), max(values)
    if high == low:
        return {row_id: 1.0 for row_id, _score in results}

    scale = high - low
    return {row_id: (float(score) - low) / scale for row_id, score in results}


def minmax_score_fusion(bm25_results, dense_results):
    """각 검색기의 원점수를 개별 Min-Max 정규화한 뒤 가중 합산."""
    bm25_normalized = normalize_minmax(bm25_results)
    dense_normalized = normalize_minmax(dense_results)

    scores = {}
    for row_id, score in bm25_normalized.items():
        scores[row_id] = scores.get(row_id, 0.0) + BM25_WEIGHT * score
    for row_id, score in dense_normalized.items():
        scores[row_id] = scores.get(row_id, 0.0) + DENSE_WEIGHT * score

    # 정규화 점수가 정확히 0이어도 검색된 후보라면 결과에 포함됨
    return sorted(scores, key=lambda row_id: (-scores[row_id], row_id))


def retrieve_candidates(index, question_vectors, questions, bm25_index):
    """BM25와 Dense 후보 및 원점수를 1회 수집해 두 방식에서 동일하게 사용."""
    actual_k = min(CANDIDATE_K, index.ntotal)
    retriever, tokenizer, chunk_id_to_row_id = bm25_index

    start = time.perf_counter()
    dense_scores, dense_ids = index.search(question_vectors, actual_k)
    dense_seconds = time.perf_counter() - start
    dense_rankings = []
    for ids, scores in zip(dense_ids, dense_scores):
        dense_rankings.append([
            (int(row_id), float(score))
            for row_id, score in zip(ids, scores)
            if row_id >= 0
        ])

    start = time.perf_counter()
    bm25_rankings = []
    for question in questions:
        docs = bm25_retrieve(question, retriever, tokenizer, actual_k)
        bm25_rankings.append([
            (chunk_id_to_row_id[doc["chunk_id"]], float(doc["score"]))
            for doc in docs
            if doc["score"] > 0  # 기존 hybrid_search.py와 동일한 필터
        ])
    bm25_seconds = time.perf_counter() - start

    return bm25_rankings, dense_rankings, bm25_seconds, dense_seconds


def evaluate_rankings(rankings, records, chunks):
    """evaluate_retrieval.py와 동일한 정답 파일명 기반 평가."""
    if len(rankings) != len(records):
        raise ValueError("검색 결과 개수와 평가 질문 수가 다릅니다.")

    totals = Counter()
    for record, row_ids in zip(records, rankings):
        gold = {filename_key(c["filename"]) for c in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 문서 파일명이 없습니다: {record['id']}")

        ranked_names = [
            filename_key(chunk_filename(chunks[int(row_id)]))
            for row_id in row_ids[:10]
        ]
        metrics = retrieval_scores(gold, ranked_names, k=5)
        for key in ("hit", "recall", "rr", "precision", "ndcg"):
            totals[key] += metrics[key]

    count = len(records)
    return {
        "hit_at_5_percent": round(totals["hit"] / count * 100, 2),
        "recall_at_5_percent": round(totals["recall"] / count * 100, 2),
        "mrr_at_10": round(totals["rr"] / count, 4),
        "precision_at_5": round(totals["precision"] / count, 4),
        "ndcg_at_5": round(totals["ndcg"] / count, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Weighted RRF vs Min-Max score fusion")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="기존 JSON 결과 덮어쓰기")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"기존 결과 파일이 있습니다: {args.output} (--overwrite 사용)")

    validate_settings()
    data = load_records("clean")
    records = [record for record in data if not record["is_impossible"]]
    if not records:
        raise ValueError("답변 가능한 평가 질문이 없습니다.")
    questions = [record["question"] for record in records]
    print(f"\nValidation: {len(records)} questions / {len(data)} total", flush=True)

    model, index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    reranker = load_reranker()

    # 모든 실험에서 같은 질문 임베딩, 같은 BM25/Dense 후보/원점수를 사용
    start = time.perf_counter()
    question_vectors = encode_questions(model, questions)
    embedding_seconds = time.perf_counter() - start

    bm25_rows, dense_rows, bm25_seconds, dense_seconds = retrieve_candidates(
        index, question_vectors, questions, bm25_index
    )
    print(
        f"Shared retrieval: embedding={embedding_seconds:.3f}s, "
        f"BM25={bm25_seconds:.3f}s, Dense={dense_seconds:.3f}s",
        flush=True,
    )

    cases = (
        ("weighted_rrf", weighted_rrf),
        ("minmax_score", minmax_score_fusion),
    )
    results = []
    for case_number, (method, fusion_function) in enumerate(cases, start=1):
        start = time.perf_counter()
        hybrid_rankings = [
            fusion_function(bm25, dense)
            for bm25, dense in zip(bm25_rows, dense_rows)
        ]
        fusion_seconds = time.perf_counter() - start

        hybrid_metrics = evaluate_rankings(hybrid_rankings, records, chunks)

        start = time.perf_counter()
        reranked = rerank_candidates(reranker, questions, hybrid_rankings, chunks)
        reranker_seconds = time.perf_counter() - start
        reranker_rankings = [
            [item["row_id"] for item in question_results]
            for question_results in reranked
        ]
        reranker_metrics = evaluate_rankings(reranker_rankings, records, chunks)

        row = {
            "fusion_method": method,
            "hybrid": hybrid_metrics,
            "reranker": reranker_metrics,
            "timing": {
                "fusion_seconds": round(fusion_seconds, 4),
                "fusion_mean_ms_per_question": round(
                    fusion_seconds * 1000 / len(records), 4
                ),
                "reranker_seconds": round(reranker_seconds, 4),
            },
        }
        results.append(row)
        h, r = hybrid_metrics, reranker_metrics
        print(
            f"[{case_number}/{len(cases)}] {method} | "
            f"Hybrid Hit@5={h['hit_at_5_percent']:.2f}% "
            f"MRR@10={h['mrr_at_10']:.4f} nDCG@5={h['ndcg_at_5']:.4f} | "
            f"Reranker Hit@5={r['hit_at_5_percent']:.2f}% "
            f"MRR@10={r['mrr_at_10']:.4f} nDCG@5={r['ndcg_at_5']:.4f} | "
            f"Fusion={row['timing']['fusion_mean_ms_per_question']:.4f} ms/query",
            flush=True,
        )

    # 우선순위: Reranker Hit@5 → MRR@10 → nDCG@5 → 기존 RRF 우선
    best = max(
        results,
        key=lambda row: (
            row["reranker"]["hit_at_5_percent"],
            row["reranker"]["mrr_at_10"],
            row["reranker"]["ndcg_at_5"],
            row["fusion_method"] == "weighted_rrf",
        ),
    )

    payload = {
        "dataset": "clean",
        "total": len(data),
        "answerable_evaluated": len(records),
        "impossible_skipped": len(data) - len(records),
        "bm25_weight": BM25_WEIGHT,
        "dense_weight": DENSE_WEIGHT,
        "rrf_k": RRF_K,
        "hybrid_candidate_k": CANDIDATE_K,
        "reranker_candidate_k": RERANKER_CANDIDATE_K,
        "minmax_note": (
            "질문별 BM25 / Dense 각각의 상위 후보 원점수를 독립적으로 "
            "[0,1] Min-Max 정규화 후 BM25=0.5, Dense=1.0으로 합산. "
            "BM25 score<=0 후보 제외, Dense 후보 전체 사용. "
            "단일/상수 점수 집합의 정규화 결과는 1.0으로 처리."
        ),
        "timing_note": (
            "같은 질문 임베딩과 검색 후보를 재사용. Fusion 시간은 순위 결합만 측정하며 "
            "BM25/Dense 검색, 임베딩, 모델 로딩 및 평가 시간은 제외. "
            "Reranker 시간은 별도 측정."
        ),
        "shared_timings": {
            "embedding_seconds": round(embedding_seconds, 4),
            "bm25_seconds": round(bm25_seconds, 4),
            "dense_seconds": round(dense_seconds, 4),
        },
        "results": results,
        "best_fusion_method": best["fusion_method"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nBest Fusion:", best["fusion_method"])
    print("Best Reranker:", json.dumps(best["reranker"], ensure_ascii=False))
    print("Saved:", args.output.resolve())
    print("YAML, BM25, PCA and FAISS indexes were not modified.")


if __name__ == "__main__":
    main()
