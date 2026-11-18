"""Compare CrossEncoder reranker models with the current optimized Hybrid retriever.

Run from repository root (PowerShell):
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_reranker_models.py

The source retrieval.yaml, indexes and production search code are NOT modified.
Results: IBMTechSupport/results/reranker_model_comparison.json

After interruption, repeat with --resume; to rerun from scratch, use --overwrite.
"""

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from dense_search import load_retriever
from document_utils import chunk_filename, filename_key
from evaluate.common import load_records, retrieval_scores
from hybrid_search import load_bm25, search_all
from reranker_search import get_chunk_text, rerank_candidates
from retrieval_config import load_retrieval_config

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_DIR / "results" / "reranker_model_comparison.json"

MODELS = (
    "cross-encoder/ms-marco-MiniLM-L6-v2",  # current baseline
    "cross-encoder/ms-marco-MiniLM-L4-v2",  # smaller / faster candidate
    "cross-encoder/ms-marco-MiniLM-L12-v2", # deeper candidate
    "BAAI/bge-reranker-base",               # different model family
)

EXPECTED_HYBRID = {
    "candidate_k": 50,
    "rrf_k": 10,
    "bm25_weight": 0.5,
    "dense_weight": 1.0,
    "max_chunks_per_document": 1,
}
EXPECTED_RERANKER = {"candidate_k": 10, "top_k": 5, "batch_size": 16, "max_length": 512}


def validate_config(config, batch_size):
    """Ensure that only the reranker model varies between runs."""
    hybrid = config["hybrid"]
    reranker = config["reranker"]
    for key, value in EXPECTED_HYBRID.items():
        if hybrid.get(key) != value:
            raise ValueError(f"hybrid.{key} must be {value!r}, got {hybrid.get(key)!r}")
    for key, value in EXPECTED_RERANKER.items():
        if key == "batch_size":
            continue  # --batch-size overrides consistently for every model
        if reranker.get(key) != value:
            raise ValueError(f"reranker.{key} must be {value!r}, got {reranker.get(key)!r}")
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")


def dataset_fingerprint(records):
    """Prevent --resume from mixing different validation sets."""
    content = [
        {"id": x["id"], "question": x["question"],
         "contexts": x["contexts"], "is_impossible": x["is_impossible"]}
        for x in records
    ]
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metrics_from_rankings(records, rankings, chunks):
    """Use the project's existing filename-based retrieval metrics."""
    if len(records) != len(rankings):
        raise ValueError("record count != ranking count")

    sums = Counter()
    for record, ranking in zip(records, rankings):
        gold = {filename_key(c["filename"]) for c in record["contexts"]}
        if not gold:
            raise ValueError(f"Missing gold filename: {record['id']}")
        names = [filename_key(chunk_filename(chunks[int(i)])) for i in ranking[:10]]
        scores = retrieval_scores(gold, names, 5)
        for key in ("hit", "recall", "rr", "precision", "ndcg"):
            sums[key] += scores[key]

    n = len(records)
    return {
        "hit_at_5_percent": round(100 * sums["hit"] / n, 2),
        "recall_at_5_percent": round(100 * sums["recall"] / n, 2),
        "mrr_at_10": round(sums["rr"] / n, 4),
        "precision_at_5": round(sums["precision"] / n, 4),
        "ndcg_at_5": round(sums["ndcg"] / n, 4),
    }


def save_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def select_best(results):
    """Ranking priority: Hit@5, MRR@10, nDCG@5; speed breaks exact ties."""
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
    parser = argparse.ArgumentParser(description="Reranker model comparison")
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help="Hugging Face CrossEncoder model names")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be combined")
    if len(set(args.models)) != len(args.models):
        parser.error("duplicate model names in --models")
    if not args.models:
        parser.error("at least one model is required")
    if args.output.exists() and not (args.resume or args.overwrite):
        parser.error(f"Output already exists: {args.output} (use --resume or --overwrite)")
    if args.resume and not args.output.exists():
        parser.error(f"No results to resume: {args.output}")

    config = load_retrieval_config()
    validate_config(config, args.batch_size)
    all_records = load_records("clean")
    records = [item for item in all_records if not item["is_impossible"]]
    if not records:
        raise ValueError("No answerable questions")
    ids = [r["id"] for r in records]
    signature = {
        "dataset_sha256": dataset_fingerprint(all_records),
        "hybrid": config["hybrid"],
        "reranker_candidate_k": EXPECTED_RERANKER["candidate_k"],
        "top_k": EXPECTED_RERANKER["top_k"],
        "max_length": EXPECTED_RERANKER["max_length"],
        "batch_size": args.batch_size,
        "models": args.models,
    }

    if args.resume:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if payload.get("signature") != signature:
            raise ValueError("Saved benchmark settings do not match; use --overwrite to restart")
    else:
        payload = {
            "signature": signature,
            "dataset": "clean",
            "total": len(all_records),
            "answerable_evaluated": len(records),
            "impossible_skipped": len(all_records) - len(records),
            "results": [],
        }

    completed = {r["model"] for r in payload["results"]}
    print(f"Validation: {len(records)} answerable / {len(all_records)} total")
    print(f"Hybrid: {config['hybrid']}")
    print(f"Reranker candidates: 10, max_length: 512, batch_size: {args.batch_size}")
    print(f"Models: {args.models}")
    print("Ranking scores are not compared across models; only per-query order is compared.")

    # Search only once: all reranker models see the same 10 candidates.
    embedder, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    questions = [r["question"] for r in records]
    retrieval_start = time.perf_counter()
    hybrid_results = search_all(embedder, faiss_index, bm25_index, questions)
    hybrid_rankings = [r[2] for r in hybrid_results]
    retrieval_seconds = time.perf_counter() - retrieval_start

    required_k = EXPECTED_RERANKER["candidate_k"]
    if any(len(row) < required_k for row in hybrid_rankings):
        raise RuntimeError("Fewer than 10 Hybrid candidates for at least one question")
    print(f"Shared retrieval: {retrieval_seconds:.3f} s | questions={len(questions)}")

    # Release the embedding model before loading alternative CrossEncoders.
    del embedder, faiss_index, bm25_index, hybrid_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_pairs = len(questions) * required_k

    for position, model_name in enumerate(args.models, start=1):
        if model_name in completed:
            print(f"[{position}/{len(args.models)}] SKIP (saved): {model_name}")
            continue

        print(f"\n[{position}/{len(args.models)}] Model: {model_name}", flush=True)
        load_start = time.perf_counter()
        cross_encoder = CrossEncoder(model_name, device=device,
                                     max_length=EXPECTED_RERANKER["max_length"])
        load_seconds = time.perf_counter() - load_start

        # Warm-up is excluded from latency measurement.
        sample_pair = [(questions[0], get_chunk_text(chunks[hybrid_rankings[0][0]]))]
        cross_encoder.predict(sample_pair, batch_size=1,
                              show_progress_bar=False, convert_to_numpy=True)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        # Reuse the actual production function and its tie-breaking behavior.
        reranked = rerank_candidates(
            cross_encoder, questions, hybrid_rankings, chunks,
            batch_size=args.batch_size,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        rerank_seconds = time.perf_counter() - start

        rankings = [[int(item["row_id"]) for item in row] for row in reranked]
        metrics = metrics_from_rankings(records, rankings, chunks)
        peak_memory_gb = (
            round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            if device == "cuda" else None
        )

        result = {
            "model": model_name,
            "metrics": metrics,
            "model_load_seconds": round(load_seconds, 3),
            "rerank_seconds": round(rerank_seconds, 3),
            "rerank_ms_per_query": round(rerank_seconds * 1000 / len(questions), 3),
            "pairs_per_second": round(total_pairs / rerank_seconds, 2),
            "peak_cuda_allocated_gb": peak_memory_gb,
            "question_ids": ids,
            "rankings_top_10": rankings,
        }
        payload["results"].append(result)
        best = select_best(payload["results"])
        payload["best_model_so_far"] = best["model"]
        payload["complete"] = len(payload["results"]) == len(args.models)
        save_atomic(args.output, payload)

        print(
            f"Hit@5={metrics['hit_at_5_percent']:.2f}% | "
            f"MRR@10={metrics['mrr_at_10']:.4f} | "
            f"nDCG@5={metrics['ndcg_at_5']:.4f} | "
            f"Rerank={result['rerank_ms_per_query']:.3f} ms/query | "
            f"Peak VRAM={peak_memory_gb} GiB",
            flush=True,
        )
        print(f"Saved: {args.output}", flush=True)

        del cross_encoder, reranked
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    best = select_best(payload["results"])
    print("\nBest model (validation):", best["model"])
    print("Best metrics:", json.dumps(best["metrics"], ensure_ascii=False))
    print("Completed:", len(payload["results"]), "/", len(args.models))
    print("Production YAML, indexes, and code were not changed.")


if __name__ == "__main__":
    main()
