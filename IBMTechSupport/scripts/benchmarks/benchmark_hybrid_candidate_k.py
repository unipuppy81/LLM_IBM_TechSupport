"""Hybrid Candidate K (20/50/100) 비교: Hybrid / Reranker 성능과 검색 시간.

프로젝트 루트에서 실행:
    python IBMTechSupport/scripts/benchmarks/benchmark_hybrid_candidate_k.py

설정은 앞선 RRF 실험에서 선정한 BM25=0.5, Dense=1.0, RRF_K=10으로 고정.
BM25, Dense, PCA, FAISS 인덱스와 retrieval.yaml을 변경하지 않음.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# scripts/benchmarks/에서 직접 실행해도 상위 scripts/ 모듈을 찾을 수 있도록 처리
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
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "hybrid_candidate_k_comparison.json"

CANDIDATE_K_VALUES = (20, 50, 100)
BM25_WEIGHT = 0.5
DENSE_WEIGHT = 1.0
RRF_K = 10
RERANKER_CANDIDATE_K = 10


def validate_settings():
    """검색 엔진과 Reranker 후보 수가 이전 실험과 일치하는지 확인."""
    config = load_retrieval_config()
    dense = config["dense"]
    reranker = config["reranker"]

    if not dense["pca"]["enabled"] or dense["pca"]["dimension"] != 192:
        raise ValueError("PCA 192차원 설정을 유지하세요.")
    if dense["faiss"]["index_type"].lower() != "hnsw":
        raise ValueError("FAISS HNSW 설정을 유지하세요.")
    if dense["faiss"]["ef_search"] != 128:
        raise ValueError("HNSW ef_search=128을 유지하세요.")
    if reranker["candidate_k"] != RERANKER_CANDIDATE_K:
        raise ValueError("reranker.candidate_k=10을 유지하세요.")

    # hybrid 설정의 기존 값은 읽기만 함. 이번 벤치마크에서는 상수를 사용.
    print("Current YAML hybrid settings:", config["hybrid"])
    print("Experiment overrides (YAML unchanged):")
    print(f"  BM25={BM25_WEIGHT}, Dense={DENSE_WEIGHT}, RRF_K={RRF_K}")
    print(f"  Hybrid Candidate K={CANDIDATE_K_VALUES}")
    print(f"  Reranker Candidate K={RERANKER_CANDIDATE_K}")


def weighted_rrf(bm25_ids, dense_ids):
    """현재 hybrid_search.py와 동일한 RRF 점수 및 동점 정렬 규칙."""
    scores = {}
    for ranked, weight in (
        (bm25_ids, BM25_WEIGHT),
        (dense_ids, DENSE_WEIGHT),
    ):
        for rank, row_id in enumerate(ranked, start=1):
            scores[row_id] = scores.get(row_id, 0.0) + weight / (RRF_K + rank)

    return sorted(scores, key=lambda row_id: (-scores[row_id], row_id))


def evaluate_rankings(rankings, records, chunks):
    """evaluate_retrieval.py와 동일하게 정답 문서 파일명으로 평가."""
    if len(rankings) != len(records):
        raise ValueError("검색 결과 수와 평가 질문 수가 다릅니다.")

    totals = Counter()
    for record, row_ids in zip(records, rankings):
        gold = {filename_key(c["filename"]) for c in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 문서 파일명이 없습니다: {record['id']}")

        ranked = [
            filename_key(chunk_filename(chunks[int(row_id)]))
            for row_id in row_ids[:10]
        ]
        scores = retrieval_scores(gold, ranked, k=5)
        for key in ("hit", "recall", "rr", "precision", "ndcg"):
            totals[key] += scores[key]

    count = len(records)
    return {
        "hit_at_5_percent": round(totals["hit"] / count * 100, 2),
        "recall_at_5_percent": round(totals["recall"] / count * 100, 2),
        "mrr_at_10": round(totals["rr"] / count, 4),
        "precision_at_5": round(totals["precision"] / count, 4),
        "ndcg_at_5": round(totals["ndcg"] / count, 4),
    }


def retrieve_at_k(index, encoded_questions, questions, bm25_index, k):
    """후보 K에 맞춰 BM25와 Dense를 실제 재검색하고 각 검색 시간을 측정."""
    bm25_retriever, tokenizer, chunk_id_to_row_id = bm25_index
    actual_k = min(k, index.ntotal)

    started = time.perf_counter()
    _, dense_row_ids = index.search(encoded_questions, actual_k)
    dense_seconds = time.perf_counter() - started
    dense_rankings = [
        [int(row_id) for row_id in ids if row_id >= 0]
        for ids in dense_row_ids
    ]

    started = time.perf_counter()
    bm25_rankings = []
    for question in questions:
        docs = bm25_retrieve(question, bm25_retriever, tokenizer, actual_k)
        ranked = [
            chunk_id_to_row_id[doc["chunk_id"]]
            for doc in docs
            if doc["score"] > 0
        ]
        bm25_rankings.append(ranked)
    bm25_seconds = time.perf_counter() - started

    return bm25_rankings, dense_rankings, bm25_seconds, dense_seconds


def main():
    parser = argparse.ArgumentParser(description="Hybrid Candidate K benchmark")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="결과 JSON 덮어쓰기")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        parser.error(f"기존 결과 파일이 있습니다: {args.output} (--overwrite 사용)")

    # 모델 및 원본 데이터는 읽기만 하고 저장/변경하지 않음.
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

    # 공통 질문 임베딩은 한 번 생성. 아래 검색 시간에서는 제외.
    started = time.perf_counter()
    encoded_questions = encode_questions(model, questions)
    embedding_seconds = time.perf_counter() - started
    print(f"Question embedding (shared): {embedding_seconds:.3f} s", flush=True)

    results = []
    for case, candidate_k in enumerate(CANDIDATE_K_VALUES, start=1):
        bm25_rows, dense_rows, bm25_seconds, dense_seconds = retrieve_at_k(
            index, encoded_questions, questions, bm25_index, candidate_k
        )

        started = time.perf_counter()
        hybrid_rankings = [
            weighted_rrf(bm25_ids, dense_ids)
            for bm25_ids, dense_ids in zip(bm25_rows, dense_rows)
        ]
        fusion_seconds = time.perf_counter() - started
        hybrid_metrics = evaluate_rankings(hybrid_rankings, records, chunks)

        started = time.perf_counter()
        reranked = rerank_candidates(
            reranker, questions, hybrid_rankings, chunks
        )
        reranker_seconds = time.perf_counter() - started
        reranker_rankings = [
            [item["row_id"] for item in row]
            for row in reranked
        ]
        reranker_metrics = evaluate_rankings(reranker_rankings, records, chunks)

        retrieval_seconds = bm25_seconds + dense_seconds + fusion_seconds
        timing = {
            "bm25_seconds": round(bm25_seconds, 4),
            "dense_seconds": round(dense_seconds, 4),
            "fusion_seconds": round(fusion_seconds, 4),
            "retrieval_seconds_excluding_embedding": round(retrieval_seconds, 4),
            "reranker_seconds": round(reranker_seconds, 4),
            "retrieval_mean_ms_per_question": round(
                retrieval_seconds * 1000 / len(records), 3
            ),
        }
        results.append({
            "candidate_k": candidate_k,
            "hybrid": hybrid_metrics,
            "reranker": reranker_metrics,
            "timing": timing,
        })

        h, r = hybrid_metrics, reranker_metrics
        print(
            f"[{case}/{len(CANDIDATE_K_VALUES)}] Candidate K={candidate_k} | "
            f"Hybrid Hit@5={h['hit_at_5_percent']:.2f}% "
            f"MRR@10={h['mrr_at_10']:.4f} nDCG@5={h['ndcg_at_5']:.4f} | "
            f"Reranker Hit@5={r['hit_at_5_percent']:.2f}% "
            f"MRR@10={r['mrr_at_10']:.4f} nDCG@5={r['ndcg_at_5']:.4f} | "
            f"Retrieval={timing['retrieval_mean_ms_per_question']:.3f} ms/query",
            flush=True,
        )

    # 1. Reranker Hit@5  2. MRR@10  3. nDCG@5  4. 작은 Candidate K
    best = max(
        results,
        key=lambda row: (
            row["reranker"]["hit_at_5_percent"],
            row["reranker"]["mrr_at_10"],
            row["reranker"]["ndcg_at_5"],
            -row["candidate_k"],
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
        "reranker_candidate_k": RERANKER_CANDIDATE_K,
        "candidate_k_values": list(CANDIDATE_K_VALUES),
        "question_embedding_seconds_shared": round(embedding_seconds, 4),
        "timing_note": (
            "각 K에서 BM25/Dense를 재검색한 시간과 RRF 처리 시간을 측정. "
            "질문 임베딩/모델 로딩/평가 시간 제외. 실험 실행 순서 및 "
            "CPU 환경에 따라 지연시간은 달라질 수 있음."
        ),
        "results": results,
        "best": best,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nBest Candidate K:", best["candidate_k"])
    print("Best Reranker:", json.dumps(best["reranker"], ensure_ascii=False))
    print("Saved:", args.output.resolve())
    print("YAML, BM25, PCA and FAISS indexes were not modified.")


if __name__ == "__main__":
    main()
