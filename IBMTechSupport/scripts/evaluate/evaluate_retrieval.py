# STAGE3_LIGHT_RUNTIME_V1
"""
Dense
BM25
Hybrid
Reranker

Hit@5
Recall@5
MRR@10
Precision@K
nDCG@K
"""

import argparse
import time
from collections import Counter
from pathlib import Path
from document_utils import chunk_filename, filename_key
from .common import check_output, load_jsonl, load_records, report, retrieval_scores


def retrieve(records, method, count):
    questions = [r["question"] for r in records]
    if method == "stage3":
        from stage2_retrieval import retrieve_all_stage2
        documents, _ = retrieve_all_stage2(questions, count, stage3_light=True)
        return [[filename_key(doc["filename"]) for doc in row] for row in documents]
    if method == "bm25":
        from bm25_search import load_index, retrieve as search, normalize_filename
        index, tokenizer = load_index()
        return [[normalize_filename(d["filename"]) if d["score"] > 0 else None
                 for d in search(q, index, tokenizer, count)] for q in questions]
    
    from dense_search import load_retriever, encode_questions
    model, index, chunks = load_retriever()

    if method == "dense":
        _, ids = index.search(encode_questions(model, questions), min(count, index.ntotal))
        rankings = [[int(i) for i in row if i >= 0] for row in ids]
    else:
        from hybrid_search import load_bm25, search_all
        from hybrid_search import BM25_INDEX_DIR

        if not (BM25_INDEX_DIR / "params.index.json").is_file():
            raise FileNotFoundError("Build the Hybrid BM25 index separately before evaluation")
        
        results = search_all(model, index, load_bm25(chunks), questions)
        rankings = [r[2] for r in results]
        if method == "reranker":
            from reranker_search import load_reranker, rerank_candidates
            ranked = rerank_candidates(load_reranker(), questions, rankings, chunks)
            rankings = [[d["row_id"] for d in row] for row in ranked]

    return [[filename_key(chunk_filename(chunks[i])) for i in row[:count]] for row in rankings]


def evaluate(method="hybrid", k=5, dataset="clean", limit=0, predictions=None, output=None):
    if method not in ("bm25", "dense", "hybrid", "reranker", "stage3") or not 1 <= k <= 50:
        raise ValueError("Invalid retriever or K (1..50)")
    
    check_output(output)
    data = load_records(dataset, limit)
    records = [r for r in data if not r["is_impossible"]]
    skipped_no_gold = 0

    if method == "bm25":
        from .common import get_gold_documents
        golds = {r["id"]: get_gold_documents(r) for r in records}
        skipped_no_gold = sum(not golds[r["id"]] for r in records)
        records = [r for r in records if golds[r["id"]]]
    else:
        golds = {r["id"]: {filename_key(c["filename"]) for c in r["contexts"]} for r in records}

    if not records or any(not golds[r["id"]] for r in records):
        raise ValueError("Nonempty records and gold documents required")
    
    start = time.perf_counter()
    if predictions:
        rows = load_jsonl(predictions)
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate prediction IDs")
        
        by_id = {r["id"]: r for r in rows}
        if set(by_id) != {r["id"] for r in data}:
            raise ValueError("Prediction and dataset IDs differ")
        
        rankings = []
        for r in records:
            row = by_id[r["id"]]
            if row["question"] != r["question"]:
                raise ValueError(f"Question mismatch: {r['id']}")
            
            rankings.append([filename_key(chunk_filename(d)) if method != "bm25" or d["score"] > 0 else None
                             for d in row["retrieved_documents"]])
            
        if any(len(row) < max(k, 10) for row in rankings):
            raise ValueError("Saved results have insufficient ranks for K and MRR@10; use live retrieval")
    else:
        rankings = retrieve(records, method, max(k, 10))

    totals = Counter()
    for record, ranked in zip(records, rankings):
        gold = golds[record["id"]]
        legacy = retrieval_scores(gold, ranked, 5)
        added = retrieval_scores(gold, ranked, k)
        for key in ("hit", "recall", "rr"):
            totals[key] += legacy[key]
        for key in ("precision", "ndcg"):
            totals[key] += added[key]

    n = len(records)
    metrics = {"dataset": dataset, "retriever": method, "total": len(data),
               "answerable_evaluated": n, "impossible_skipped": sum(r["is_impossible"] for r in data),
               "skipped_no_gold": skipped_no_gold,
               "hit_at_5_percent": round(totals["hit"] / n * 100, 2),
               "recall_at_5_percent": round(totals["recall"] / n * 100, 2),
               "mrr_at_10": round(totals["rr"] / n, 4),
               f"precision_at_{k}": round(totals["precision"] / n, 4),
               f"ndcg_at_{k}": round(totals["ndcg"] / n, 4),
               "evaluation_seconds": round(time.perf_counter() - start, 3)}
    
    return report(metrics, output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retriever", choices=("bm25", "dense", "hybrid", "reranker", "stage3"), default="hybrid")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dataset", default="clean")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    return evaluate(a.retriever, a.top_k, a.dataset, a.limit, a.predictions, a.output)


if __name__ == "__main__":
    main()
