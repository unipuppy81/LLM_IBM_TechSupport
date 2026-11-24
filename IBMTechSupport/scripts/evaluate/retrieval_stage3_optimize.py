from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import yaml

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from document_utils import chunk_filename, filename_key
from stage2_retrieval import (
    apply_constraint_filter,
    document_search_text,
    get_chunk_text,
)

DEFAULT_DATASET = PROJECT_DIR / "data" / "processed" / "val_106.json"
OUT_DIR = PROJECT_DIR / "evaluate_results" / "106" / "retrieval_stage3"
RESULT_JSON = OUT_DIR / "optimization.json"
TOP_CONFIGS_JSON = OUT_DIR / "top_configs.json"
SELECTED_YAML = PROJECT_DIR / "configs" / "rag" / "retrieval_stage3_candidate.yaml"

WIDE_K = 100
RRF_K_VALUES = (5, 10, 20, 40)
BM25_WEIGHTS = (0.25, 0.5, 0.75, 1.0)
FILTER_STRENGTHS = (0.0, 0.5, 1.0)
CANDIDATE_K_VALUES = (20, 30, 50)
PRIOR_ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
INPUT_VARIANTS = ("text", "metadata")
TARGET_PERCENT = 95.0


# Windows spawn-safe multiprocessing worker state.
# Only compact candidate chunks are copied to workers; the full 111k corpus is not.
_MP_RECORDS = None
_MP_QUESTIONS = None
_MP_CHUNKS = None
_MP_BM25_RANKINGS = None
_MP_DENSE_RANKINGS = None
_MP_SCORE_MAPS_BY_VARIANT = None


def _init_grid_worker(
    records,
    questions,
    compact_chunks,
    bm25_rankings,
    dense_rankings,
    score_maps_by_variant,
):
    global _MP_RECORDS
    global _MP_QUESTIONS
    global _MP_CHUNKS
    global _MP_BM25_RANKINGS
    global _MP_DENSE_RANKINGS
    global _MP_SCORE_MAPS_BY_VARIANT

    _MP_RECORDS = records
    _MP_QUESTIONS = questions
    _MP_CHUNKS = compact_chunks
    _MP_BM25_RANKINGS = bm25_rankings
    _MP_DENSE_RANKINGS = dense_rankings
    _MP_SCORE_MAPS_BY_VARIANT = score_maps_by_variant


def _evaluate_config_batch(config_batch):
    if _MP_RECORDS is None:
        raise RuntimeError("multiprocessing worker was not initialized")

    batch_results = []
    for config in config_batch:
        metrics, contribs, _ = evaluate_config(
            _MP_RECORDS,
            _MP_QUESTIONS,
            _MP_CHUNKS,
            _MP_BM25_RANKINGS,
            _MP_DENSE_RANKINGS,
            _MP_SCORE_MAPS_BY_VARIANT[config["reranker_input"]],
            config,
        )
        batch_results.append({
            "config": config,
            "metrics": metrics,
            "contribs": contribs,
        })
    return batch_results


def _batched(values, size):
    return [values[i:i + size] for i in range(0, len(values), size)]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def gold_documents(record: dict) -> set[str]:
    return {
        filename_key(c["filename"])
        for c in record.get("contexts", [])
        if isinstance(c, dict)
        and isinstance(c.get("filename"), str)
        and c["filename"].strip()
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


def weighted_rrf(
    bm25_ranking,
    dense_ranking,
    *,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float = 1.0,
):
    scores = {}
    for ranking, weight in (
        (bm25_ranking, bm25_weight),
        (dense_ranking, dense_weight),
    ):
        for rank, row_id in enumerate(ranking, start=1):
            row_id = int(row_id)
            scores[row_id] = scores.get(row_id, 0.0) + weight / (rrf_k + rank)
    ranked = sorted(scores, key=lambda rid: (-scores[rid], rid))
    return ranked, scores


def zscore_map(values: dict[int, float], ids: list[int]) -> dict[int, float]:
    selected = [float(values[rid]) for rid in ids]
    if not selected:
        return {}
    mu = mean(selected)
    sigma = pstdev(selected)
    if sigma < 1e-8:
        return {rid: 0.0 for rid in ids}
    return {rid: (float(values[rid]) - mu) / sigma for rid in ids}


def score_union(
    reranker,
    questions: list[str],
    union_rankings: list[list[int]],
    chunks: list[dict],
    *,
    variant: str,
    batch_size: int,
) -> list[dict[int, float]]:
    pairs = []
    positions = []

    if variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")

    for q_idx, (question, ranking) in enumerate(zip(questions, union_rankings)):
        for row_id in ranking:
            if variant == "text":
                doc = get_chunk_text(chunks[row_id])
            else:
                doc = document_search_text(chunks[row_id])
            pairs.append((question, doc))
            positions.append((q_idx, int(row_id)))

    raw = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    scores = np.asarray(raw, dtype=float).reshape(-1)

    if len(scores) != len(positions) or not np.isfinite(scores).all():
        raise RuntimeError("Reranker score count/value error")

    grouped = [dict() for _ in questions]
    for (q_idx, row_id), score in zip(positions, scores):
        grouped[q_idx][row_id] = float(score)
    return grouped


def rerank_blended(
    question: str,
    candidate_ranking: list[int],
    adjusted_prior: dict[int, float],
    reranker_scores: dict[int, float],
    *,
    candidate_k: int,
    prior_alpha: float,
) -> list[int]:
    subset = [
        rid
        for rid in candidate_ranking[:candidate_k]
        if rid in reranker_scores and rid in adjusted_prior
    ]
    if not subset:
        return []

    ce_z = zscore_map(reranker_scores, subset)
    prior_z = zscore_map(adjusted_prior, subset)

    scored = []
    for original_rank, row_id in enumerate(subset):
        score = ce_z[row_id] + prior_alpha * prior_z[row_id]
        scored.append((float(score), original_rank, row_id))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [row_id for _, _, row_id in scored]


def case_contribution(record: dict, ranking: list[int], chunks: list[dict]) -> dict:
    gold = gold_documents(record)
    names = []
    seen = set()
    for row_id in ranking:
        name = filename_key(chunk_filename(chunks[row_id]))
        if name in seen:
            continue
        seen.add(name)
        names.append(name)

    top5 = names[:5]
    found5 = set(top5) & gold
    hit_ranks10 = [i for i, name in enumerate(names[:10], 1) if name in gold]
    found_ranks5 = [i for i, name in enumerate(top5, 1) if name in gold]

    def dcg(ranks):
        return sum(1.0 / math.log2(rank + 1) for rank in ranks)

    ideal = dcg(list(range(1, min(5, len(gold)) + 1)))
    first_rank = hit_ranks10[0] if hit_ranks10 else None

    return {
        "id": str(record["id"]),
        "hit1": float(bool(set(names[:1]) & gold)),
        "hit3": float(bool(set(names[:3]) & gold)),
        "hit5": float(bool(found5)),
        "hit10": float(bool(set(names[:10]) & gold)),
        "recall5": len(found5) / len(gold),
        "precision5": len(found5) / 5.0,
        "rr10": (1.0 / first_rank) if first_rank else 0.0,
        "ndcg5": (dcg(found_ranks5) / ideal) if ideal else 0.0,
        "first_gold_rank": first_rank,
    }


def aggregate(contribs: list[dict], indices: list[int] | None = None) -> dict:
    rows = contribs if indices is None else [contribs[i] for i in indices]
    n = len(rows)
    if n == 0:
        raise ValueError("empty metric rows")

    def avg(key):
        return sum(float(r[key]) for r in rows) / n

    return {
        "answerable_evaluated": n,
        "hit_at_1_percent": round(avg("hit1") * 100, 4),
        "hit_at_3_percent": round(avg("hit3") * 100, 4),
        "hit_at_5_percent": round(avg("hit5") * 100, 4),
        "hit_at_10_percent": round(avg("hit10") * 100, 4),
        "recall_at_5_percent": round(avg("recall5") * 100, 4),
        "precision_at_5": round(avg("precision5"), 6),
        "mrr_at_10": round(avg("rr10"), 6),
        "ndcg_at_5": round(avg("ndcg5"), 6),
        "hit5_count": int(round(sum(float(r["hit5"]) for r in rows))),
    }


def selection_key(metrics: dict, config: dict):
    # Primary objective is Hit@5.  Recall then ranking quality.
    # Smaller candidate_k is preferred only after metric ties.
    return (
        metrics["hit_at_5_percent"],
        metrics["recall_at_5_percent"],
        metrics["hit_at_3_percent"],
        metrics["mrr_at_10"],
        metrics["ndcg_at_5"],
        metrics["hit_at_1_percent"],
        -int(config["candidate_k"]),
        -float(config["prior_alpha"]),
        -float(config["filter_strength"]),
        -int(config["rrf_k"]),
        -float(config["bm25_weight"]),
        1 if config["reranker_input"] == "text" else 0,
    )


def evaluate_config(
    records,
    questions,
    chunks,
    bm25_rankings,
    dense_rankings,
    score_maps,
    config,
):
    rankings = []
    contribs = []

    for i, (record, question, bm25_ranking, dense_ranking) in enumerate(
        zip(records, questions, bm25_rankings, dense_rankings)
    ):
        rrf_ranking, _ = weighted_rrf(
            bm25_ranking,
            dense_ranking,
            rrf_k=int(config["rrf_k"]),
            bm25_weight=float(config["bm25_weight"]),
            dense_weight=1.0,
        )
        rrf_ranking = dedupe_documents(rrf_ranking, chunks)[:WIDE_K]

        filtered, _, adjusted = apply_constraint_filter(
            question,
            rrf_ranking,
            chunks,
            strength=float(config["filter_strength"]),
        )

        final = rerank_blended(
            question,
            filtered,
            adjusted,
            score_maps[i],
            candidate_k=int(config["candidate_k"]),
            prior_alpha=float(config["prior_alpha"]),
        )

        rankings.append(final)
        contribs.append(case_contribution(record, final, chunks))

    return aggregate(contribs), contribs, rankings


def build_union_rankings(bm25_rankings, dense_rankings):
    out = []
    for bm25_r, dense_r in zip(bm25_rankings, dense_rankings):
        # Score every row that any RRF configuration can possibly use.
        union = list(dict.fromkeys([*bm25_r[:WIDE_K], *dense_r[:WIDE_K]]))
        out.append(union)
    return out


def cross_validation_estimate(config_results: list[dict], n_cases: int, folds: int = 5) -> dict:
    fold_indices = [[] for _ in range(folds)]
    for idx in range(n_cases):
        fold_indices[idx % folds].append(idx)

    heldout_contribs = [None] * n_cases
    selections = []

    all_indices = set(range(n_cases))
    for fold_id, valid_idx in enumerate(fold_indices):
        valid_set = set(valid_idx)
        train_idx = sorted(all_indices - valid_set)

        best = None
        best_key = None
        for result in config_results:
            train_metrics = aggregate(result["contribs"], train_idx)
            key = selection_key(train_metrics, result["config"])
            if best is None or key > best_key:
                best = result
                best_key = key

        valid_metrics = aggregate(best["contribs"], valid_idx)
        for idx in valid_idx:
            heldout_contribs[idx] = best["contribs"][idx]

        selections.append({
            "fold": fold_id + 1,
            "validation_size": len(valid_idx),
            "selected_config": best["config"],
            "heldout_metrics": valid_metrics,
        })

    if any(row is None for row in heldout_contribs):
        raise RuntimeError("CV heldout rows incomplete")

    return {
        "folds": folds,
        "aggregate_heldout_metrics": aggregate(heldout_contribs),
        "fold_selections": selections,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrieval Stage3 optimizer")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=None)
    default_workers = max(1, min(12, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"grid-search worker processes (default: {default_workers})",
    )
    parser.add_argument(
        "--mp-batch-size",
        type=int,
        default=12,
        help="configs evaluated per multiprocessing task",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.mp_batch_size < 1:
        parser.error("--mp-batch-size must be >= 1")

    if not args.dataset.is_file():
        raise FileNotFoundError(f"dataset not found: {args.dataset}")

    all_records = read_json(args.dataset)
    records = [r for r in all_records if r.get("is_impossible") is False]
    if not records:
        raise ValueError("no answerable records")

    for record in records:
        if not gold_documents(record):
            raise ValueError(f"gold context missing: {record.get('id')}")

    questions = [r["question"] for r in records]
    print(f"[STAGE3 1/6] total={len(all_records)}, answerable={len(records)}", flush=True)

    from rag_pipeline import load_search_system
    from bm25_search import retrieve as bm25_retrieve
    from dense_search import encode_questions
    from reranker_search import load_reranker
    from retrieval_config import load_retrieval_config

    embedding_model, faiss_index, bm25_index, chunks = load_search_system()
    retrieval_cfg = load_retrieval_config()
    batch_size = args.batch_size or int(retrieval_cfg["reranker"]["batch_size"])

    print(f"[STAGE3 2/6] wide BM25/Dense retrieval: top {WIDE_K}", flush=True)
    retriever, tokenizer, chunk_id_to_row_id = bm25_index

    vectors = encode_questions(embedding_model, questions)
    _, dense_ids = faiss_index.search(vectors, min(WIDE_K, faiss_index.ntotal))

    bm25_rankings = []
    dense_rankings = []
    for question, d_ids in zip(questions, dense_ids):
        bm25_results = bm25_retrieve(
            question,
            retriever,
            tokenizer,
            min(WIDE_K, len(chunks)),
        )
        bm25_ranked = [
            int(chunk_id_to_row_id[row["chunk_id"]])
            for row in bm25_results
            if row.get("score", 0.0) > 0 and row.get("chunk_id") in chunk_id_to_row_id
        ]
        dense_ranked = [int(row_id) for row_id in d_ids if int(row_id) >= 0]
        bm25_rankings.append(bm25_ranked)
        dense_rankings.append(dense_ranked)

    union_rankings = build_union_rankings(bm25_rankings, dense_rankings)
    avg_union = sum(len(x) for x in union_rankings) / len(union_rankings)
    print(f"[STAGE3 3/6] CrossEncoder union scoring: avg candidates={avg_union:.1f}", flush=True)

    reranker = load_reranker()
    score_maps_by_variant = {}
    for variant in INPUT_VARIANTS:
        print(f"  - scoring variant={variant}", flush=True)
        score_maps_by_variant[variant] = score_union(
            reranker,
            questions,
            union_rankings,
            chunks,
            variant=variant,
            batch_size=batch_size,
        )

    configs = []
    for rrf_k in RRF_K_VALUES:
        for bm25_weight in BM25_WEIGHTS:
            for filter_strength in FILTER_STRENGTHS:
                for candidate_k in CANDIDATE_K_VALUES:
                    for prior_alpha in PRIOR_ALPHAS:
                        for input_variant in INPUT_VARIANTS:
                            configs.append({
                                "wide_k": WIDE_K,
                                "rrf_k": rrf_k,
                                "bm25_weight": bm25_weight,
                                "dense_weight": 1.0,
                                "filter_strength": filter_strength,
                                "candidate_k": candidate_k,
                                "prior_alpha": prior_alpha,
                                "reranker_input": input_variant,
                            })

    # Only rows that can appear in the wide BM25/Dense candidate union are needed
    # during grid search. This keeps Windows spawn memory/pickle cost bounded.
    used_row_ids = sorted({
        int(row_id)
        for ranking in union_rankings
        for row_id in ranking
    })
    compact_chunks = {row_id: chunks[row_id] for row_id in used_row_ids}

    print(
        f"[STAGE3 4/6] grid search: {len(configs)} configs | "
        f"workers={args.workers} | task_batch={args.mp_batch_size} | "
        f"candidate_chunks={len(compact_chunks)}",
        flush=True,
    )
    start = time.perf_counter()
    results = []
    config_batches = _batched(configs, args.mp_batch_size)

    if args.workers == 1:
        _init_grid_worker(
            records,
            questions,
            compact_chunks,
            bm25_rankings,
            dense_rankings,
            score_maps_by_variant,
        )
        processed = 0
        for batch in config_batches:
            batch_results = _evaluate_config_batch(batch)
            results.extend(batch_results)
            processed += len(batch_results)
            if processed % 120 < len(batch_results) or processed == len(configs):
                print(f"  evaluated {processed}/{len(configs)}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        processed = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_grid_worker,
            initargs=(
                records,
                questions,
                compact_chunks,
                bm25_rankings,
                dense_rankings,
                score_maps_by_variant,
            ),
        ) as executor:
            futures = [
                executor.submit(_evaluate_config_batch, batch)
                for batch in config_batches
            ]
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)
                processed += len(batch_results)
                if processed % 120 < len(batch_results) or processed == len(configs):
                    print(f"  evaluated {processed}/{len(configs)}", flush=True)

    results.sort(
        key=lambda x: selection_key(x["metrics"], x["config"]),
        reverse=True,
    )
    best = results[0]

    print("[STAGE3 5/6] 5-fold tuning stability estimate", flush=True)
    cv = cross_validation_estimate(results, len(records), folds=5)

    target_count = math.ceil(len(records) * TARGET_PERCENT / 100.0)
    best_failed = [
        row["id"] for row in best["contribs"] if not bool(row["hit5"])
    ]

    top10 = [
        {
            "rank": idx,
            "config": result["config"],
            "metrics": result["metrics"],
            "failed_hit5_ids": [
                row["id"] for row in result["contribs"] if not bool(row["hit5"])
            ],
        }
        for idx, result in enumerate(results[:10], start=1)
    ]

    payload = {
        "schema": "retrieval-stage3-optimization-v1",
        "dataset": str(args.dataset.resolve()),
        "answerable": len(records),
        "stage2_baseline": {
            "hit5_count": 66,
            "hit_at_5_percent": 85.7143,
        },
        "target": {
            "percent": TARGET_PERCENT,
            "minimum_hit5_count": target_count,
        },
        "search_space": {
            "wide_k": WIDE_K,
            "rrf_k": list(RRF_K_VALUES),
            "bm25_weight": list(BM25_WEIGHTS),
            "dense_weight": 1.0,
            "filter_strength": list(FILTER_STRENGTHS),
            "candidate_k": list(CANDIDATE_K_VALUES),
            "prior_alpha": list(PRIOR_ALPHAS),
            "reranker_input": list(INPUT_VARIANTS),
            "total_configs": len(configs),
            "workers": args.workers,
            "mp_batch_size": args.mp_batch_size,
        },
        "best": {
            "config": best["config"],
            "metrics": best["metrics"],
            "failed_hit5_ids": best_failed,
            "target_reached": best["metrics"]["hit5_count"] >= target_count,
        },
        "cross_validation": cv,
        "elapsed_grid_seconds": round(time.perf_counter() - start, 3),
        "note": (
            "This optimizer uses only val_106. test_clean is not read. "
            "The candidate YAML is not a production freeze until its result is reviewed."
        ),
    }

    write_json(RESULT_JSON, payload)
    write_json(TOP_CONFIGS_JSON, top10)

    selected_payload = {
        "version": "stage3-candidate-v1",
        "frozen": False,
        "source_dataset": str(args.dataset.resolve()),
        "target_hit5_percent": TARGET_PERCENT,
        "selected": best["config"],
        "selected_metrics": best["metrics"],
        "cross_validation_heldout_metrics": cv["aggregate_heldout_metrics"],
        "warning": "Candidate only. Do not use in final-test until reviewed and frozen.",
    }
    SELECTED_YAML.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_YAML.write_text(
        yaml.safe_dump(selected_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("[STAGE3 6/6] result", flush=True)
    print(json.dumps({
        "best_config": best["config"],
        "best_metrics": best["metrics"],
        "target_reached": best["metrics"]["hit5_count"] >= target_count,
        "failed_hit5_ids": best_failed,
        "cv_heldout": cv["aggregate_heldout_metrics"],
        "result_json": str(RESULT_JSON),
        "candidate_yaml": str(SELECTED_YAML),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
