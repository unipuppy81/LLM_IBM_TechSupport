"""IBM RAG: Compare Reranker candidate K=10/30/50 without modifying Baseline.

From the repository root (PowerShell):
    python IBMTechSupport/scripts/evaluate/rag_candidate_k_experiment.py

Inputs: existing val_clean.json, saved final_e2e/reranker/predictions.jsonl,
        installed Hybrid BM25/FAISS indexes, local embedding & reranker models.
Outputs: results/rag_candidate_k_comparison_v1/{summary.json,cases_90.csv,changes_k30.csv,changes_k50.csv}
No LLM generation, index rebuild, config edits, or training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
K_VALUES = (10, 30, 50)
TOP_K = 5
SEED = 42


def filename_key(s):
    return str(s).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(buf)
    return hasher.hexdigest()


def pct(n, d):
    return round(100 * n / d, 2) if d else None


def percentile_sorted(vals, percentage):
    index = (len(vals) - 1) * percentage
    low = math.floor(index)
    high = math.ceil(index)
    return vals[low] + (vals[high] - vals[low]) * (index - low)


def paired_hit_ci(before, after, draws=10000):
    """Question-cluster paired bootstrap; differences are in percentage points."""
    if len(before) != len(after) or not before:
        raise ValueError("paired bootstrap 입력 길이 오류")
    changes = [int(b) - int(a) for a, b in zip(before, after)]
    rng = random.Random(SEED)
    n = len(changes)
    values = sorted(100 * sum(changes[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return {
        "delta_percentage_points": round(100 * sum(changes) / n, 2),
        "paired_bootstrap_95ci_pp": [round(percentile_sorted(values, .025), 2), round(percentile_sorted(values, .975), 2)],
        "bootstrap_draws": draws,
    }


def rerank_fixed_k(reranker, questions, rankings, chunks, k, batch_size):
    """Reproduce reranker_search.rerank_candidates sorting, but allow K > YAML value."""
    import numpy as np
    from reranker_search import get_chunk_text

    pairs = []
    positions = []
    for query_index, (q, ranking) in enumerate(zip(questions, rankings)):
        for source_rank, row_id in enumerate(ranking[:k]):
            pairs.append((q, get_chunk_text(chunks[row_id])))
            positions.append((query_index, int(row_id), source_rank))
    if not pairs:
        raise ValueError("Reranker 대상 후보가 없습니다")
    start = time.perf_counter()
    scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - start
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(positions) or not np.isfinite(scores).all():
        raise ValueError("Reranker 점수 개수/유효성 오류")
    grouped = [[] for _ in questions]
    for (query_index, row_id, source_rank), score in zip(positions, scores):
        grouped[query_index].append({"row_id": row_id, "score": float(score), "original_rank": source_rank})
    for g in grouped:
        g.sort(key=lambda d: (-d["score"], d["original_rank"]))
    return grouped, elapsed, len(pairs)


def first_gold_rank(ranked_ids, chunks, gold):
    from document_utils import chunk_filename
    return next((r for r, row_id in enumerate(ranked_ids, 1)
                 if filename_key(chunk_filename(chunks[row_id])) in gold), None)


def as_csv(path, rows):
    if not rows:
        return
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="IBM RAG Reranker 후보 K=10/30/50 비교")
    parser.add_argument("--root", type=Path, default=ROOT, help="IBMTechSupport 디렉터리")
    parser.add_argument("--output", type=Path, help="새 결과 디렉터리 (기존 디렉터리에 덮어쓰지 않음)")
    args = parser.parse_args()
    root = args.root.resolve()
    data = root / "data/processed/val_clean.json"
    saved = root / "evaluate_results/rag/final_e2e/reranker/predictions.jsonl"
    yaml_path = root / "configs/rag/retrieval.yaml"
    output = (args.output or root / "results/rag_candidate_k_comparison_v1").resolve()
    for p in (data, saved, yaml_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    for blocked in (root / "data", root / "models"):
        if output == blocked.resolve() or output.is_relative_to(blocked.resolve()):
            raise ValueError("결과는 data/ 또는 models/ 폴더에 저장하면 안 됩니다")
    if output.exists():
        raise FileExistsError(f"기존 결과 보호: {output}\n--output으로 다른 폴더를 지정하세요")

    val_rows = json.loads(data.read_text(encoding="utf-8-sig"))
    original_rows = read_jsonl(saved)
    if len(val_rows) != 135 or len(original_rows) != 135:
        raise ValueError("기존 135개 Baseline 데이터가 아닙니다")
    if len({r["id"] for r in val_rows}) != 135 or len({r["id"] for r in original_rows}) != 135:
        raise ValueError("중복 평가 질문 ID가 있습니다")
    by_id = {r["id"]: r for r in original_rows}
    if set(by_id) != {r["id"] for r in val_rows}:
        raise ValueError("val_clean.json과 저장된 Baseline의 질문 ID가 다릅니다")
    for record in val_rows:
        result = by_id[record["id"]]
        if (record["question"] != result.get("question") or
            record["answer"] != result.get("reference_answer") or
            record["is_impossible"] != result.get("is_impossible")):
            raise ValueError(f"Baseline 데이터 내용 불일치: {record['id']}")

    answerables = [r for r in val_rows if not r["is_impossible"]]
    if len(answerables) != 90:
        raise ValueError("답변 가능한 질문 수가 90개가 아닙니다")

    from retrieval_config import load_retrieval_config
    config = load_retrieval_config()
    batch_size = int(config["reranker"]["batch_size"])
    if int(config["reranker"]["candidate_k"]) != 10 or int(config["reranker"]["top_k"]) != 5:
        raise ValueError("기존 Baseline의 reranker.candidate_k=10, top_k=5가 아닙니다")
    if int(config["hybrid"]["candidate_k"]) < 50:
        raise ValueError("Hybrid 후보 풀 설정 candidate_k가 50보다 작습니다")
    if batch_size < 1:
        raise ValueError("잘못된 batch_size")

    from dense_search import load_retriever
    from hybrid_search import load_bm25, search_all
    from reranker_search import load_reranker, rerank_candidates
    from document_utils import chunk_filename
    from evaluate.common import retrieval_scores

    print("[Reranker 후보 K 비교]", flush=True)
    print(f"전체 135 / 답변 가능 {len(answerables)} / 후보 K={K_VALUES}")
    print("검색 설정은 변경하지 않습니다. 생성 모델도 실행하지 않습니다.")
    embedder, index, chunks = load_retriever()
    bm25 = load_bm25(chunks)
    questions = [r["question"] for r in answerables]
    retrieval_start = time.perf_counter()
    search_results = search_all(embedder, index, bm25, questions)
    retrieval_seconds = round(time.perf_counter() - retrieval_start, 3)
    hybrid_rankings = [r[2] for r in search_results]
    reranker = load_reranker()

    # First reproduce the existing implementation, before changing any experiment variable.
    print("\n[K=10 Baseline 재현 검사]", flush=True)
    rerank10 = rerank_candidates(reranker, questions, hybrid_rankings, chunks, batch_size=batch_size)
    mismatches = []
    for record, ranking in zip(answerables, rerank10):
        expected = [str(d.get("chunk_id")) for d in by_id[record["id"]]["retrieved_documents"][:TOP_K]]
        current = [str(chunks[x["row_id"]].get("chunk_id")) for x in ranking[:TOP_K]]
        if current != expected:
            mismatches.append(record["id"])
    print(f"기존 Top5 재현 성공: {len(answerables)-len(mismatches)}/{len(answerables)}")
    if mismatches:
        raise ValueError(
            f"기존 결과 재현 실패 ({len(mismatches)}개): {mismatches[:10]}\n"
            "현재 검색 설정/인덱스/모델과 저장된 Baseline이 일치하지 않습니다. 비교 실험을 중단합니다."
        )

    reranked = {10: rerank10}
    timings = {}
    for k in (30, 50):
        print(f"\n[K={k} 후보 재정렬]", flush=True)
        reranked[k], elapsed, scored = rerank_fixed_k(
            reranker, questions, hybrid_rankings, chunks, k, batch_size
        )
        timings[str(k)] = {"rerank_seconds": round(elapsed, 3), "pairs_scored": scored}

    records = []
    hit_by_k = {k: [] for k in K_VALUES}
    metrics = {}
    for i, record in enumerate(answerables):
        gold = {filename_key(c["filename"]) for c in record["contexts"]}
        if not gold:
            raise ValueError(f"정답 출처 문서 없음: {record['id']}")
        row = {
            "id": record["id"],
            "question": record["question"],
            "gold_filenames": ";".join(sorted(gold)),
            "bm25_first_gold_rank": first_gold_rank(search_results[i][0], chunks, gold),
            "dense_first_gold_rank": first_gold_rank(search_results[i][1], chunks, gold),
            "hybrid_first_gold_rank": first_gold_rank(hybrid_rankings[i], chunks, gold),
        }
        for k in K_VALUES:
            entries = reranked[k][i]
            ids = [e["row_id"] for e in entries]
            names = [filename_key(chunk_filename(chunks[j])) for j in ids]
            hit = any(name in gold for name in names[:TOP_K])
            hit_by_k[k].append(hit)
            row[f"k{k}_hit5"] = int(hit)
            row[f"k{k}_gold_first_rank"] = first_gold_rank(ids, chunks, gold)
            row[f"k{k}_top5_filenames"] = ";".join(names[:TOP_K])
            row[f"k{k}_top5_chunk_ids"] = ";".join(str(chunks[j].get("chunk_id")) for j in ids[:TOP_K])
            row[f"k{k}_top5_reranker_scores"] = ";".join(f"{e['score']:.5f}" for e in entries[:TOP_K])
        records.append(row)

    for k in K_VALUES:
        total = sum(hit_by_k[k])
        # project-standard MRR@10 and nDCG@5 are computed from full reranked candidate lists.
        mrr = 0.0
        ndcg = 0.0
        for i, record in enumerate(answerables):
            gold = {filename_key(c["filename"]) for c in record["contexts"]}
            entries = reranked[k][i]
            names = [filename_key(chunk_filename(chunks[e["row_id"]])) for e in entries]
            scores = retrieval_scores(gold, names, k=5)
            mrr += scores["rr"]
            ndcg += scores["ndcg"]
        metrics[str(k)] = {
            "hit_at_5_percent": pct(total, len(answerables)),
            "hit_at_5_questions": total,
            "mrr_at_10": round(mrr / len(answerables), 4),
            "ndcg_at_5": round(ndcg / len(answerables), 4),
        }

    comparisons = {}
    changed_rows = {}
    baseline_hit = hit_by_k[10]
    for k in (30, 50):
        new_hit_ids = [r["id"] for r, before, after in zip(records, baseline_hit, hit_by_k[k]) if not before and after]
        lost_ids = [r["id"] for r, before, after in zip(records, baseline_hit, hit_by_k[k]) if before and not after]
        comparisons[str(k)] = {
            "new_success_count": len(new_hit_ids),
            "new_success_ids": new_hit_ids,
            "lost_success_count": len(lost_ids),
            "lost_success_ids": lost_ids,
            "baseline_30_failures_recovered": len(new_hit_ids),
            **paired_hit_ci(baseline_hit, hit_by_k[k]),
        }
        changed_rows[k] = [{
            "id": row["id"],
            "change": "NEW_HIT" if row["id"] in new_hit_ids else "LOST_HIT",
            "question": row["question"],
            "gold_filenames": row["gold_filenames"],
            "hybrid_first_gold_rank": row["hybrid_first_gold_rank"],
            "k10_gold_first_rank": row["k10_gold_first_rank"],
            f"k{k}_gold_first_rank": row[f"k{k}_gold_first_rank"],
            "k10_top5_filenames": row["k10_top5_filenames"],
            f"k{k}_top5_filenames": row[f"k{k}_top5_filenames"],
        } for row in records if row["id"] in set(new_hit_ids + lost_ids)]

    payload = {
        "experiment": "Reranker candidate_k 10 vs 30 vs 50",
        "note": "Gold source filename match is not proof of sufficient answer evidence in any chunk.",
        "dataset_count": len(val_rows),
        "answerable_count": len(answerables),
        "baseline_top5_reproduced": len(answerables),
        "retrieval_seconds": retrieval_seconds,
        "config": {"reranker_model": config["reranker"]["model"], "reranker_batch_size": batch_size,
                   "hybrid": config["hybrid"], "reranker": config["reranker"]},
        "input_sha256": {"val_clean": sha256_file(data), "saved_baseline_predictions": sha256_file(saved),
                         "retrieval_config": sha256_file(yaml_path)},
        "metrics": metrics,
        "versus_k10": comparisons,
        "timings": timings,
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    as_csv(output / "cases_90.csv", records)
    for k in (30, 50):
        if changed_rows[k]:
            as_csv(output / f"changes_k{k}.csv", changed_rows[k])
        else:
            (output / f"changes_k{k}.csv").write_text("id,change\n", encoding="utf-8-sig")

    print("\n[최종 결과]")
    for k in K_VALUES:
        m = metrics[str(k)]
        print(f"K={k:2d}: Hit@5={m['hit_at_5_questions']}/90 ({m['hit_at_5_percent']:.2f}%), "
              f"MRR@10={m['mrr_at_10']:.4f}, nDCG@5={m['ndcg_at_5']:.4f}")
        if k != 10:
            cmp = comparisons[str(k)]
            print(f"      신규 성공 {cmp['new_success_count']} / 기존 성공 손실 {cmp['lost_success_count']} / "
                  f"Hit@5 차이 {cmp['delta_percentage_points']:+.2f}%p "
                  f"(paired CI {cmp['paired_bootstrap_95ci_pp']})")
    print("\n저장:", output)
    print("주의: 파일명 기준 Hit@5이며 정답 근거 청크의 실제 존재 여부는 별도 검증 필요")
    print("Baseline 설정·데이터·모델·인덱스는 변경하지 않았습니다")


if __name__ == "__main__":
    main()
