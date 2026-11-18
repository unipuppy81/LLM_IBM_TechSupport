"""Hybrid 동일 원본 문서 청크 중복 제한 실험.

실행 (프로젝트 루트 PowerShell):
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_hybrid_dedup.py

실험:
    no_limit: 동일 문서 청크 제한 없음 (기준)
    max_1:    문서별 최대 1개 청크
    max_2:    문서별 최대 2개 청크
    max_3:    문서별 최대 3개 청크

RRF 후보 전체에서 문서별 최대 개수를 적용한 뒤, 아래 순위의 다른 문서 청크로
Reranker 입력 10개를 가능한 한 채웁니다. 기준과 실험 모두 동일한 후보로 시작합니다.
기존 YAML, FAISS/BM25 인덱스, 데이터, 실제 검색 코드는 변경하지 않습니다.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# 프로젝트의 scripts/ 모듈을 이 파일만으로 import 가능하게 설정
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
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "hybrid_dedup_comparison.json"

BM25_WEIGHT = 0.5
DENSE_WEIGHT = 1.0
RRF_K = 10
HYBRID_CANDIDATE_K = 50
RERANKER_CANDIDATE_K = 10
CASES = (("no_limit", None), ("max_1", 1), ("max_2", 2), ("max_3", 3))


def validate_config():
    """앞 실험의 비교 조건만 검증. YAML의 Hybrid 값은 덮어쓰지 않음."""
    config = load_retrieval_config()
    dense = config["dense"]
    pca = dense.get("pca", {})
    faiss_config = dense.get("faiss", {})
    reranker = config["reranker"]

    if not pca.get("enabled") or int(pca.get("dimension", -1)) != 192:
        raise ValueError("실험 조건은 PCA enabled=true, dimension=192입니다.")
    if str(faiss_config.get("index_type", "")).lower() != "hnsw":
        raise ValueError("실험 조건은 FAISS HNSW입니다.")
    if int(faiss_config.get("m", 0)) != 32 or int(faiss_config.get("ef_search", 0)) != 128:
        raise ValueError("실험 조건은 HNSW M=32, efSearch=128입니다.")
    if int(reranker["candidate_k"]) != RERANKER_CANDIDATE_K:
        raise ValueError("reranker.candidate_k=10이어야 합니다.")

    print("Current YAML hybrid settings:", config["hybrid"])
    print("Experiment overrides (YAML unchanged):")
    print(f"  BM25={BM25_WEIGHT} Dense={DENSE_WEIGHT} RRF_K={RRF_K}")
    print(f"  Hybrid Candidate K={HYBRID_CANDIDATE_K}, Reranker Candidate K={RERANKER_CANDIDATE_K}")
    print("  Per-document chunk limits: none / 1 / 2 / 3")


def weighted_rrf(bm25_ranked, dense_ranked):
    """기존 Hybrid와 동일한 Weighted RRF 및 동점일 때 row_id 오름차순."""
    scores = {}
    for ranking, weight in (
        (bm25_ranked, BM25_WEIGHT),
        (dense_ranked, DENSE_WEIGHT),
    ):
        for rank, row_id in enumerate(ranking, start=1):
            scores[row_id] = scores.get(row_id, 0.0) + weight / (RRF_K + rank)

    return sorted(scores, key=lambda row_id: (-scores[row_id], row_id))


def collect_base_rankings(model, index, bm25_index, questions):
    """90개 질문의 Dense, BM25 순위를 한 번만 구해 4개 실험에 재사용."""
    k = min(HYBRID_CANDIDATE_K, index.ntotal)
    retriever, tokenizer, chunk_id_to_row_id = bm25_index

    started = time.perf_counter()
    vectors = encode_questions(model, questions)
    embedding_seconds = time.perf_counter() - started

    started = time.perf_counter()
    _, dense_ids = index.search(vectors, k)
    dense_seconds = time.perf_counter() - started
    dense_rankings = [
        [int(row_id) for row_id in row if row_id >= 0]
        for row in dense_ids
    ]

    started = time.perf_counter()
    bm25_rankings = []
    for question in questions:
        documents = bm25_retrieve(question, retriever, tokenizer, k)
        bm25_rankings.append([
            chunk_id_to_row_id[doc["chunk_id"]]
            for doc in documents
            if doc["score"] > 0
        ])
    bm25_seconds = time.perf_counter() - started

    rankings = [
        weighted_rrf(bm25, dense)
        for bm25, dense in zip(bm25_rankings, dense_rankings)
    ]
    return rankings, {
        "embedding_seconds": round(embedding_seconds, 4),
        "dense_seconds": round(dense_seconds, 4),
        "bm25_seconds": round(bm25_seconds, 4),
    }


def make_source_keys(chunks):
    """기존 평가와 같은 filename_key(파일명 소문자) 기준으로 문서 구분."""
    return [filename_key(chunk_filename(chunk)) for chunk in chunks]


def filter_per_document(ranking, source_keys, max_chunks):
    """RRF 순위를 유지하면서 원본 문서별 청크 개수 상한 적용.

    뒤쪽 후보를 계속 확인해 다른 문서 청크로 후보 수를 보충할 수 있도록,
    상위 10개에서 자르지 않고 전체 RRF 후보에 필터를 적용합니다.
    """
    if max_chunks is None:
        return list(ranking)
    if max_chunks < 1:
        raise ValueError("문서별 최대 청크 수는 1 이상이어야 합니다.")

    counts = Counter()
    selected = []
    for row_id in ranking:
        key = source_keys[row_id]
        if counts[key] >= max_chunks:
            continue
        counts[key] += 1
        selected.append(row_id)
    return selected


def evaluate_rankings(rankings, records, source_keys):
    """기존 evaluate_retrieval.py와 동일한 파일명 기반 평가."""
    if len(rankings) != len(records):
        raise ValueError("랭킹 개수와 질문 개수가 다릅니다.")

    totals = Counter()
    for record, ranking in zip(records, rankings):
        gold = {filename_key(context["filename"]) for context in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 파일명이 없습니다: {record['id']}")

        ranked_names = [source_keys[row_id] for row_id in ranking[:10]]
        scores = retrieval_scores(gold, ranked_names, k=5)
        for metric in ("hit", "recall", "rr", "precision", "ndcg"):
            totals[metric] += scores[metric]

    count = len(records)
    return {
        "hit_at_5_percent": round(totals["hit"] / count * 100, 2),
        "recall_at_5_percent": round(totals["recall"] / count * 100, 2),
        "mrr_at_10": round(totals["rr"] / count, 4),
        "precision_at_5": round(totals["precision"] / count, 4),
        "ndcg_at_5": round(totals["ndcg"] / count, 4),
    }


def diversity_statistics(rankings, source_keys, top_k):
    """상위 K 청크의 평균 고유 문서 수와 중복 청크 수 측정."""
    unique_counts = []
    duplicate_counts = []
    short_count = 0

    for ranking in rankings:
        selected = ranking[:top_k]
        unique = len({source_keys[row_id] for row_id in selected})
        unique_counts.append(unique)
        duplicate_counts.append(len(selected) - unique)
        short_count += len(selected) < top_k

    n = len(rankings)
    return {
        f"mean_unique_documents_at_{top_k}": round(sum(unique_counts) / n, 4),
        f"mean_duplicate_chunks_at_{top_k}": round(sum(duplicate_counts) / n, 4),
        f"queries_with_fewer_than_{top_k}_candidates": short_count,
    }


def validate_output_path(output, overwrite):
    output = output.resolve()
    for forbidden in (PROJECT_DIR / "models", PROJECT_DIR / "data"):
        if output.is_relative_to(forbidden.resolve()):
            raise ValueError("결과 파일은 models/ 또는 data/ 아래에 저장할 수 없습니다.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"결과 파일이 이미 있습니다: {output} (--overwrite 사용)")
    return output


def main():
    parser = argparse.ArgumentParser(description="Hybrid 원본 문서별 청크 중복 제한 실험")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="기존 결과 JSON 덮어쓰기")
    args = parser.parse_args()

    output = validate_output_path(args.output, args.overwrite)
    validate_config()

    data = load_records("clean")
    records = [record for record in data if not record["is_impossible"]]
    if not records:
        raise ValueError("평가할 질문이 없습니다.")
    questions = [record["question"] for record in records]
    print(f"\nValidation: {len(records)} questions / {len(data)} total", flush=True)

    # 임베딩 / BM25 / Reranker 로딩은 한 번만 진행
    model, index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    reranker = load_reranker()
    source_keys = make_source_keys(chunks)

    # 네 설정 모두 같은 원본 후보를 사용하여 중복 제한 효과만 비교
    base_rankings, shared_times = collect_base_rankings(model, index, bm25_index, questions)
    print(
        "Shared retrieval: "
        f"embedding={shared_times['embedding_seconds']:.3f}s, "
        f"BM25={shared_times['bm25_seconds']:.3f}s, "
        f"Dense={shared_times['dense_seconds']:.3f}s",
        flush=True,
    )

    results = []
    for i, (name, limit) in enumerate(CASES, start=1):
        started = time.perf_counter()
        filtered = [filter_per_document(ranking, source_keys, limit) for ranking in base_rankings]
        filter_seconds = time.perf_counter() - started

        hybrid_metrics = evaluate_rankings(filtered, records, source_keys)
        hybrid_diversity = diversity_statistics(filtered, source_keys, RERANKER_CANDIDATE_K)

        # 기존 Reranker 함수가 각 순위의 상위 10개 후보를 그대로 처리하도록 함
        reranker_started = time.perf_counter()
        reranked = rerank_candidates(reranker, questions, filtered, chunks)
        reranker_seconds = time.perf_counter() - reranker_started
        reranker_rankings = [
            [int(item["row_id"]) for item in row]
            for row in reranked
        ]
        reranker_metrics = evaluate_rankings(reranker_rankings, records, source_keys)
        reranker_diversity = diversity_statistics(reranker_rankings, source_keys, 5)

        case = {
            "name": name,
            "max_chunks_per_document": limit,
            "hybrid": hybrid_metrics,
            "reranker": reranker_metrics,
            "hybrid_candidate_diversity": hybrid_diversity,
            "reranker_top5_diversity": reranker_diversity,
            "timing": {
                "dedup_seconds": round(filter_seconds, 4),
                "dedup_mean_ms_per_question": round(filter_seconds * 1000 / len(records), 4),
                "reranker_seconds": round(reranker_seconds, 4),
            },
        }
        results.append(case)
        h, r = hybrid_metrics, reranker_metrics
        print(
            f"[{i}/{len(CASES)}] {name:8s} | "
            f"Hybrid Hit@5={h['hit_at_5_percent']:.2f}% "
            f"MRR@10={h['mrr_at_10']:.4f} nDCG@5={h['ndcg_at_5']:.4f} | "
            f"Reranker Hit@5={r['hit_at_5_percent']:.2f}% "
            f"MRR@10={r['mrr_at_10']:.4f} nDCG@5={r['ndcg_at_5']:.4f} | "
            f"UniqueDocs@10={hybrid_diversity['mean_unique_documents_at_10']:.2f} "
            f"Top5Unique={reranker_diversity['mean_unique_documents_at_5']:.2f} "
            f"Candidates<10={hybrid_diversity['queries_with_fewer_than_10_candidates']}",
            flush=True,
        )

    # 우선순위: Reranker Hit@5 > MRR@10 > nDCG@5 > 후보 다양성.
    # 모두 같으면 기존(no_limit) 방식 우선하여 불필요한 변경을 피함.
    best = max(results, key=lambda case: (
        case["reranker"]["hit_at_5_percent"],
        case["reranker"]["mrr_at_10"],
        case["reranker"]["ndcg_at_5"],
        case["hybrid_candidate_diversity"]["mean_unique_documents_at_10"],
        int(case["max_chunks_per_document"] is None),
    ))

    payload = {
        "dataset": "clean",
        "total": len(data),
        "answerable_evaluated": len(records),
        "impossible_skipped": len(data) - len(records),
        "experimental_settings": {
            "bm25_weight": BM25_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "rrf_k": RRF_K,
            "hybrid_candidate_k": HYBRID_CANDIDATE_K,
            "reranker_candidate_k": RERANKER_CANDIDATE_K,
            "dedup_stage": "after_weighted_rrf_before_reranker",
            "document_identity": "filename_key(chunk_filename(chunk))",
        },
        "shared_retrieval_timing": shared_times,
        "results": results,
        "best_case": best["name"],
        "best_max_chunks_per_document": best["max_chunks_per_document"],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBest:", best["name"])
    print("Best Reranker:", json.dumps(best["reranker"], ensure_ascii=False))
    print("Saved:", output)
    print("YAML, BM25, PCA, FAISS indexes unchanged.")


if __name__ == "__main__":
    main()
