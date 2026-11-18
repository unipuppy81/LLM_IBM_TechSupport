"""Reranker max_length 비교 (128 / 256 / 384 / 512 tokens).

프로젝트 루트 PowerShell:
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_reranker_max_length.py

중단 후 이어서 실행:
    python IBMTechSupport/scripts/benchmarks/benchmark_reranker_max_length.py --resume

기존 YAML, Hybrid/Dense/BM25 코드, 검색 인덱스는 수정하지 않는다.
"""

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from dense_search import load_retriever
from document_utils import chunk_filename, filename_key
from evaluate.common import load_records, retrieval_scores
from hybrid_search import load_bm25, search_all
from reranker_search import get_chunk_text, rerank_candidates
from retrieval_config import load_retrieval_config

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "reranker_max_length_comparison.json"
MAX_LENGTH_VALUES = (128, 256, 384, 512)

# 이전 Hybrid / Reranker 최적화에서 확정한 조건.
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
    "batch_size": 16,
    "max_length": 512,  # YAML은 아직 변경하지 않는다.
}
BASELINE_512 = {
    "hit_at_5_percent": 66.67,
    "mrr_at_10": 0.5231,
    "ndcg_at_5": 0.5537,
}


def check_configuration(config):
    for section_name, expected in (
        ("hybrid", EXPECTED_HYBRID),
        ("reranker", EXPECTED_RERANKER),
    ):
        section = config[section_name]
        for key, value in expected.items():
            if section.get(key) != value:
                raise ValueError(
                    f"{section_name}.{key}: 기대={value!r}, 실제={section.get(key)!r}"
                )


def fingerprint(records):
    """실험 재개 시 서로 다른 검증 데이터를 혼용하지 않도록 한다."""
    content = [
        {
            "id": row["id"],
            "question": row["question"],
            "contexts": row["contexts"],
            "is_impossible": row["is_impossible"],
        }
        for row in records
    ]
    blob = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def score_rankings(records, ranked_rows, chunks):
    """프로젝트 evaluate_retrieval과 동일한 파일명 기반 지표."""
    if len(records) != len(ranked_rows):
        raise ValueError("질문과 결과 개수가 일치하지 않습니다.")

    sums = Counter()
    for record, row_ids in zip(records, ranked_rows):
        gold = {filename_key(ctx["filename"]) for ctx in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 근거 문서가 없습니다: {record['id']}")

        names = [
            filename_key(chunk_filename(chunks[int(row_id)]))
            for row_id in row_ids[:10]
        ]
        values = retrieval_scores(gold, names, k=5)
        for name in ("hit", "recall", "rr", "precision", "ndcg"):
            sums[name] += values[name]

    n = len(records)
    return {
        "hit_at_5_percent": round(sums["hit"] / n * 100, 2),
        "recall_at_5_percent": round(sums["recall"] / n * 100, 2),
        "mrr_at_10": round(sums["rr"] / n, 4),
        "precision_at_5": round(sums["precision"] / n, 4),
        "ndcg_at_5": round(sums["ndcg"] / n, 4),
    }


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_atomic(path, payload):
    """매 설정 완료 후 결과를 저장해 중단 시 재개할 수 있도록 한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def choose_best(results):
    """품질 우선. 완전 동률이면 추론이 빠른 조건 선정."""
    return max(
        results,
        key=lambda item: (
            item["metrics"]["hit_at_5_percent"],
            item["metrics"]["mrr_at_10"],
            item["metrics"]["ndcg_at_5"],
            -item["rerank_ms_per_query"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Reranker max_length 벤치마크")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume과 --overwrite는 동시에 사용할 수 없습니다.")
    if args.output.exists() and not (args.resume or args.overwrite):
        parser.error(f"결과 파일이 이미 있습니다: {args.output} (--resume 또는 --overwrite)")
    if args.resume and not args.output.exists():
        parser.error(f"재개할 결과 파일이 없습니다: {args.output}")

    config = load_retrieval_config()
    check_configuration(config)

    data = load_records("clean")
    records = [row for row in data if not row["is_impossible"]]
    if not records:
        raise ValueError("평가할 답변 가능 질문이 없습니다.")
    questions = [row["question"] for row in records]

    signature = {
        "dataset_sha256": fingerprint(data),
        "hybrid": config["hybrid"],
        "reranker": config["reranker"],
        "max_length_values": list(MAX_LENGTH_VALUES),
    }

    if args.resume:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("저장된 실험 조건이 다릅니다. 새 실험은 --overwrite 사용.")
    else:
        payload = {
            "signature": signature,
            "dataset": "clean",
            "total": len(data),
            "answerable_evaluated": len(records),
            "impossible_skipped": len(data) - len(records),
            "results": [],
        }

    completed = {row["max_length"] for row in payload["results"]}

    print(f"Validation: {len(records)} answerable / {len(data)} total")
    print("Hybrid (fixed):", config["hybrid"])
    print("Reranker model:", EXPECTED_RERANKER["model"])
    print("Reranker Candidate K: 10 | Batch size: 16 | Top K: 5")
    print("Max Length tests:", MAX_LENGTH_VALUES)
    print("YAML max_length remains 512; experiment changes model input length only.")

    # 1. Hybrid 검색 1회: 모든 Max Length에서 완전히 같은 후보 사용.
    embedder, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    start = time.perf_counter()
    hybrid_results = search_all(embedder, faiss_index, bm25_index, questions)
    hybrid_rankings = [item[2] for item in hybrid_results]
    retrieval_seconds = time.perf_counter() - start
    if any(len(row) < EXPECTED_RERANKER["candidate_k"] for row in hybrid_rankings):
        raise RuntimeError("일부 질문의 Hybrid 후보가 10개 미만입니다.")
    print(f"Shared Hybrid retrieval: {retrieval_seconds:.3f} s")

    # 검색용 임베딩 모델 GPU 메모리 해제. 추론 시간에는 검색 비용 포함하지 않음.
    del embedder, faiss_index, bm25_index, hybrid_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pair_count = len(records) * EXPECTED_RERANKER["candidate_k"]

    # 2. 매 길이에서 같은 모델을 로드하되 max_length만 변경.
    for number, max_length in enumerate(MAX_LENGTH_VALUES, start=1):
        if max_length in completed:
            print(f"[{number}/{len(MAX_LENGTH_VALUES)}] SKIP (saved): max_length={max_length}")
            continue

        print(f"\n[{number}/{len(MAX_LENGTH_VALUES)}] Max Length: {max_length}", flush=True)
        load_start = time.perf_counter()
        reranker = CrossEncoder(
            EXPECTED_RERANKER["model"], device=device, max_length=max_length
        )
        load_seconds = time.perf_counter() - load_start

        # 예열 및 CUDA 동기화: 모델 로드와 첫 실행 초기화 시간을 측정에서 제외.
        warmup_pair = [(questions[0], get_chunk_text(chunks[hybrid_rankings[0][0]]))]
        reranker.predict(
            warmup_pair, batch_size=1,
            show_progress_bar=False, convert_to_numpy=True,
        )
        synchronize()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # 기존 rerank_candidates()를 그대로 사용: 점수 및 동점 정렬 규칙 일치.
        start = time.perf_counter()
        reranked = rerank_candidates(
            reranker,
            questions,
            hybrid_rankings,
            chunks,
            batch_size=EXPECTED_RERANKER["batch_size"],
        )
        synchronize()
        seconds = time.perf_counter() - start

        rankings = [[int(item["row_id"]) for item in items] for items in reranked]
        metrics = score_rankings(records, rankings, chunks)
        peak_vram = (
            round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
            if device == "cuda" else None
        )

        result = {
            "max_length": max_length,
            "metrics": metrics,
            "model_load_seconds": round(load_seconds, 3),
            "rerank_seconds": round(seconds, 3),
            "rerank_ms_per_query": round(seconds * 1000 / len(records), 3),
            "peak_cuda_allocated_gib": peak_vram,
            "pairs_scored": pair_count,
            "question_ids": [row["id"] for row in records],
            "rankings_top_10": rankings,
        }
        payload["results"].append(result)
        payload["best_max_length_so_far"] = choose_best(payload["results"])["max_length"]
        payload["complete"] = len(payload["results"]) == len(MAX_LENGTH_VALUES)
        save_atomic(args.output, payload)

        print(
            f"MaxLength={max_length:<3} | "
            f"Hit@5={metrics['hit_at_5_percent']:.2f}% "
            f"MRR@10={metrics['mrr_at_10']:.4f} "
            f"nDCG@5={metrics['ndcg_at_5']:.4f} | "
            f"Rerank={result['rerank_ms_per_query']:.3f} ms/query | "
            f"Peak VRAM={peak_vram} GiB",
            flush=True,
        )
        print("Saved:", args.output.resolve(), flush=True)

        del reranker, reranked
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    best = choose_best(payload["results"])
    baseline = next((x for x in payload["results"] if x["max_length"] == 512), None)
    if baseline:
        match = all(baseline["metrics"].get(key) == value for key, value in BASELINE_512.items())
        print("Baseline 512 reproduced:", match)
        if not match:
            print("WARNING: 512 기준 결과가 이전 실험과 다릅니다. 환경/설정을 확인하세요.")

    print("\nBest Max Length (validation):", best["max_length"])
    print("Best metrics:", json.dumps(best["metrics"], ensure_ascii=False))
    print("Completed:", len(payload["results"]), "/", len(MAX_LENGTH_VALUES))
    print("YAML, BM25, PCA, FAISS indexes and production scripts were not modified.")


if __name__ == "__main__":
    main()
