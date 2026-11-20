"""Step 3 — Expand existing Reranker Top-5 chunks with adjacent source chunks.

This is an independent evaluation. It does NOT change the production RAG,
FAISS/BM25 embeddings, retrieval ranking, corpus, or model weights.

PowerShell (run from repository root):
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_chunk_expansion.py --mode inspect --radius 1
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_chunk_expansion.py --radius 1 --limit 3
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_chunk_expansion.py --radius 1
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_chunk_expansion.py --radius 2
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_chunk_expansion.py --mode report

Input: saved original reranker Top-5 for all 135 questions and the SAME corpus
chunks JSONL used to build the FAISS index. Radius 1: center + previous/next
chunk; radius 2: center + 2 previous/2 next chunks (within same document).
The 150-character overlap is removed using exact source offsets.

Each expanded document is judged independently by the SAME baseline Qwen2.5-
1.5B Evidence Judge prompt; selected documents go to the SAME RAG generation
function/model. For radius=0, input text is unchanged.

Original QA labels and source filenames are evaluation PROXIES, not human-
verified per-chunk evidence sufficiency. Validate on an untouched test set
before claiming real-world gains.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
# Works even if PYTHONPATH is not set in the Windows terminal.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

BASELINE_PATH = PROJECT_DIR / "evaluate_results/rag/final_e2e/reranker/predictions.jsonl"
CORPUS_PATH = PROJECT_DIR / "data/processed/corpus_chunks.jsonl"
CONFIG_PATH = PROJECT_DIR / "configs/rag/v1.yaml"
OUTPUT_ROOT = PROJECT_DIR / "evaluate_results/rag/chunk_expansion"
VERSION = 1
ABSTAIN = "INSUFFICIENT_INFORMATION"


def parse_args():
    p = argparse.ArgumentParser(description="원본 청크 ±N개 확장 후 Judge + Generation A/B 평가")
    p.add_argument("--mode", choices=("run", "inspect", "report"), default="run")
    p.add_argument("--radius", type=int, choices=(0, 1, 2), default=1)
    p.add_argument("--limit", type=int, default=0, help="0=135개, 양수=처음 N개 스모크 평가")
    p.add_argument("--overwrite", action="store_true", help="해당 radius/limit 평가 결과만 지우고 재실행")
    p.add_argument("--verbose", action="store_true", help="Judge의 YES/NO 원문 출력")
    args = p.parse_args()
    if not 0 <= args.limit <= 135:
        p.error("--limit은 0~135만 지원합니다")
    if args.mode != "run" and (args.overwrite or args.verbose):
        p.error("--overwrite/--verbose는 run 모드에서만 사용합니다")
    return args


def read_jsonl(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"필요한 파일이 없습니다: {path}")
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def filename_key(value):
    return str(value).replace("\\", "/").split("/")[-1].strip().casefold()


def load_baseline(limit):
    rows = read_jsonl(BASELINE_PATH)
    if len(rows) != 135 or len({r["id"] for r in rows}) != 135:
        raise ValueError("기존 Judge ON 평가 결과는 중복 없는 135건이어야 합니다")
    for row in rows:
        docs = row["retrieved_documents"]
        if len(docs) != 5 or len({d.get("row_id") for d in docs}) != 5:
            raise ValueError(f"Top-5 청크 누락/중복: {row['id']}")
        for doc in docs:
            if not isinstance(doc.get("row_id"), int) or not isinstance(doc.get("text"), str):
                raise ValueError(f"청크 row_id/text 오류: {row['id']}")
    return rows[:limit] if limit else rows


def load_corpus():
    """Only read existing chunks, never touch FAISS/PCA/BM25 or embedding models."""
    chunks, by_doc = [], {}
    with CORPUS_PATH.open(encoding="utf-8-sig") as file:
        for row_id, line in enumerate(file):
            if not line.strip():
                raise ValueError(f"corpus_chunks.jsonl 빈 줄: index={row_id}; row_id 불일치 가능")
            chunk = json.loads(line)
            doc_id, idx = chunk.get("document_id"), chunk.get("chunk_index")
            if not isinstance(doc_id, str) or not isinstance(idx, int):
                raise ValueError(f"document_id 또는 chunk_index 누락: row_id={row_id}")
            key = (doc_id, idx)
            if key in by_doc:
                raise ValueError(f"중복 문서/청크 인덱스: {key}")
            if not isinstance(chunk.get("text"), str):
                raise ValueError(f"청크 text 누락: row_id={row_id}")
            chunks.append(chunk)
            by_doc[key] = row_id
    return chunks, by_doc


def validate_search_snapshot(rows, chunks):
    """Refuse to compare if the local corpus doesn't match saved retrieval."""
    for row in rows:
        for doc in row["retrieved_documents"]:
            rid = doc["row_id"]
            if not 0 <= rid < len(chunks):
                raise ValueError(f"원본 row_id가 청크 범위를 벗어남: {row['id']} {rid}")
            chunk = chunks[rid]
            if (chunk.get("chunk_id") != doc.get("chunk_id")
                    or filename_key(chunk.get("filename")) != filename_key(doc.get("filename"))
                    or chunk.get("text") != doc["text"]):
                raise ValueError(
                    f"검색 시점의 청크와 현재 corpus가 다릅니다: {row['id']} row_id={rid}. "
                    "기존 인덱스의 corpus_chunks.jsonl을 사용해야 합니다."
                )


def join_source_span(items):
    """Reconstruct the exact contiguous original text using start/end offsets."""
    if not items:
        raise ValueError("확장할 청크가 없습니다")
    first = items[0]
    result = first["text"]
    prev_end = first["end_char"]
    if len(result) != prev_end - first["start_char"]:
        raise ValueError("start_char/end_char 범위와 청크 길이가 다릅니다")
    for chunk in items[1:]:
        if len(chunk["text"]) != chunk["end_char"] - chunk["start_char"]:
            raise ValueError("청크 길이와 위치 불일치")
        overlap = prev_end - chunk["start_char"]
        if overlap < 0 or overlap >= len(chunk["text"]):
            raise ValueError(f"청크 사이가 비었거나 잘못 겹침: {chunk['chunk_id']}")
        if overlap and result[-overlap:] != chunk["text"][:overlap]:
            raise ValueError(f"청크 중첩 문자열이 일치하지 않음: {chunk['chunk_id']}")
        result += chunk["text"][overlap:]
        prev_end = chunk["end_char"]
    return result


def expand_document(doc, chunks, by_doc, radius):
    center = chunks[doc["row_id"]]
    document_id = center["document_id"]
    center_idx = center["chunk_index"]
    indexes = [center_idx]
    for idx in range(center_idx - 1, center_idx - radius - 1, -1):
        if (document_id, idx) in by_doc:
            indexes.append(idx)
        else:
            break
    for idx in range(center_idx + 1, center_idx + radius + 1):
        if (document_id, idx) in by_doc:
            indexes.append(idx)
        else:
            break
    indices = sorted(indexes)
    source_chunks = [chunks[by_doc[(document_id, i)]] for i in indices]
    if len({c.get("filename") for c in source_chunks}) != 1:
        raise ValueError(f"동일 document_id인데 파일명이 다름: {document_id}")
    expanded_text = join_source_span(source_chunks)
    return {
        **doc,
        "text": expanded_text,
        "center_chunk_id": center["chunk_id"],
        "included_chunk_ids": [c["chunk_id"] for c in source_chunks],
        "included_chunk_indexes": indices,
        "source_start_char": source_chunks[0]["start_char"],
        "source_end_char": source_chunks[-1]["end_char"],
    }


def expanded_for_row(row, chunks, by_doc, radius):
    return [expand_document(doc, chunks, by_doc, radius)
            for doc in row["retrieved_documents"]]


def load_config():
    import yaml
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_records(rows):
    from evaluate.common import load_records
    records = load_records("clean")
    if len(records) != 135:
        raise ValueError(f"val_clean.json에 135건이 필요합니다: {len(records)}")
    for src, record in zip(rows, records):
        if (src["id"], src["question"], src["reference_answer"], src["is_impossible"]) != (
            record["id"], record["question"], record["answer"], record["is_impossible"]
        ):
            raise ValueError(f"기존 평가 데이터와 val_clean.json 불일치: {src['id']}")
    return records[:len(rows)]


def output_paths(radius, limit):
    prefix = f"smoke_{limit}_" if limit else ""
    folder = OUTPUT_ROOT / f"radius_{radius}"
    return {
        "predictions": folder / f"{prefix}predictions.jsonl",
        "metrics": folder / f"{prefix}metrics.json",
        "cases": folder / f"{prefix}comparison_cases.csv",
    }


def experiment_fingerprint(rows, radius, config):
    return stable_hash({
        "version": VERSION,
        "radius": radius,
        "question_ids": [row["id"] for row in rows],
        "baseline_sha256": file_hash(BASELINE_PATH),
        "corpus_sha256": file_hash(CORPUS_PATH),
        "config": config,
        "judge_logic_sha256": file_hash(SCRIPTS_DIR / "answerability_detection.py"),
        "generation_logic_sha256": file_hash(SCRIPTS_DIR / "rag_pipeline.py"),
        "script_sha256": file_hash(Path(__file__)),
    })


def save_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_cases(path, records, baseline, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "is_impossible", "baseline_abstained", "expanded_abstained",
              "baseline_f1", "expanded_f1", "baseline_evidence_count", "expanded_evidence_count",
              "baseline_evidence_ranks", "expanded_evidence_ranks", "retrieved_gold_source",
              "baseline_gold_source_selected", "expanded_gold_source_selected", "question"]
    from evaluate.common import token_f1
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for gold, base, row in zip(records, baseline, predictions):
            gold_sources = {filename_key(c["filename"]) for c in gold.get("contexts", [])
                            if isinstance(c, dict) and c.get("filename")}
            source_in_top5 = any(filename_key(d["filename"]) in gold_sources
                                 for d in base["retrieved_documents"])
            base_selected_ids = {d["row_id"] for d in base["evidence_documents"]}
            base_ranks = [i + 1 for i, doc in enumerate(base["retrieved_documents"])
                          if doc["row_id"] in base_selected_ids]
            base_gold = any(filename_key(d["filename"]) in gold_sources
                            for d in base["evidence_documents"])
            new_gold = any(filename_key(base["retrieved_documents"][rank - 1]["filename"]) in gold_sources
                           for rank in row["evidence_ranks"])
            writer.writerow({
                "id": gold["id"], "is_impossible": gold["is_impossible"],
                "baseline_abstained": base["prediction"] == ABSTAIN,
                "expanded_abstained": row["prediction"] == ABSTAIN,
                "baseline_f1": round(token_f1(base["prediction"], gold["answer"]), 4) if not gold["is_impossible"] else "",
                "expanded_f1": round(token_f1(row["prediction"], gold["answer"]), 4) if not gold["is_impossible"] else "",
                "baseline_evidence_count": len(base["evidence_documents"]),
                "expanded_evidence_count": row["evidence_count"],
                "baseline_evidence_ranks": base_ranks,
                "expanded_evidence_ranks": row["evidence_ranks"],
                "retrieved_gold_source": source_in_top5 if not gold["is_impossible"] else "",
                "baseline_gold_source_selected": base_gold if not gold["is_impossible"] else "",
                "expanded_gold_source_selected": new_gold if not gold["is_impossible"] else "",
                "question": gold["question"],
            })


def inspect(rows, chunks, by_doc, radius):
    total_docs, changed_docs, included_extra = 0, 0, 0
    original_chars, expanded_chars = 0, 0
    changed_questions = 0
    for row in rows:
        changed_here = False
        for original, new in zip(row["retrieved_documents"], expanded_for_row(row, chunks, by_doc, radius)):
            total_docs += 1
            changed = len(new["included_chunk_ids"]) > 1
            changed_docs += changed
            changed_here |= changed
            included_extra += len(new["included_chunk_ids"]) - 1
            original_chars += len(original["text"])
            expanded_chars += len(new["text"])
        changed_questions += changed_here
    print(f"Top-5 질문: {len(rows)} | 중심 청크: {total_docs}")
    print(f"Radius: ±{radius} | 확장된 질문: {changed_questions}/{len(rows)}")
    print(f"확장된 문서 청크: {changed_docs}/{total_docs} | 추가 이웃 청크: {included_extra}")
    print(f"평균 원래 길이: {original_chars / total_docs:.1f} chars")
    print(f"평균 확장 길이: {expanded_chars / total_docs:.1f} chars")
    print("중복 문자 제거 및 원본 문서 경계 검증: 통과")
    special = next((r for r in rows if r["id"] == "TRAIN_Q037"), None)
    if special:
        print("\nTRAIN_Q037 example (each Reranker-ranked source):")
        for rank, d in enumerate(expanded_for_row(special, chunks, by_doc, radius), 1):
            print(f"  rank={rank} {d['filename']} center={d['center_chunk_id']} included={d['included_chunk_ids']}")


def validate_resume(existing, baseline, fingerprint):
    if len(existing) > len(baseline):
        raise ValueError("저장된 예측 수가 전체 질문 수보다 많습니다")
    for i, row in enumerate(existing):
        base = baseline[i]
        if row.get("id") != base["id"] or row.get("fingerprint") != fingerprint:
            raise ValueError("저장 결과의 질문/설정/입력 청크가 현재와 다릅니다. --overwrite로 재실행하세요")
        if (row.get("question"), row.get("reference_answer"), row.get("is_impossible")) != (
            base["question"], base["reference_answer"], base["is_impossible"]
        ):
            raise ValueError(f"저장 결과 질문/정답 정보 불일치: {base['id']}")
    return len(existing)


def score_run(paths, records, baseline, rows, radius):
    from evaluate.common import score_predictions, validate_prediction_rows
    validate_prediction_rows(records, rows)
    baseline_metrics = score_predictions(records, [dict(b) for b in baseline], dataset="clean", batch_size=1)
    metrics = score_predictions(records, rows, dataset="clean", batch_size=1)
    source_gold_count = 0
    for gold, base, new in zip(records, baseline, rows):
        if gold["is_impossible"]:
            continue
        keys = {filename_key(c["filename"]) for c in gold.get("contexts", [])
                if isinstance(c, dict) and c.get("filename")}
        source_gold_count += any(filename_key(base["retrieved_documents"][r - 1]["filename"]) in keys
                                 for r in new["evidence_ranks"])
    metrics.update({
        "radius": radius,
        "top_k": 5,
        "reranker_rerun": False,
        "judge_model": "Qwen2.5-1.5B-Instruct (unchanged baseline)",
        "judge_prompt": "answerability_detection.py:detect_answerability (unchanged)",
        "selected_questions": sum(bool(r["evidence_count"]) for r in rows),
        "selected_chunks": sum(r["evidence_count"] for r in rows),
        "gold_source_selected_questions": source_gold_count,
        "total_judge_seconds": round(sum(r["judge_seconds"] for r in rows), 3),
        "total_generation_seconds": round(sum(r["generation_seconds"] for r in rows), 3),
        "avg_generation_input_tokens": round(sum(r["input_tokens"] for r in rows) / len(rows), 1),
        "baseline": {k: baseline_metrics[k] for k in (
            "token_f1_percent", "exact_match_percent", "abstention_rate_percent",
            "false_refusal_rate_percent")},
        "note": "QA labels and matching filenames are proxies; gold-source matching is not chunk-level evidence validation.",
    })
    save_json_atomic(paths["metrics"], metrics)
    save_cases(paths["cases"], records, baseline, rows)
    print("\n[Original Judge vs Adjacent Chunks ±%d]" % radius)
    for key in ("token_f1_percent", "exact_match_percent", "abstention_rate_percent", "false_refusal_rate_percent"):
        print(f"  {key}: baseline={baseline_metrics[key]}  expanded={metrics[key]}")
    print("  gold_source_selected_questions:", source_gold_count)
    print("  selected_questions:", metrics["selected_questions"])
    print("Saved:", paths["predictions"])
    print("Saved:", paths["metrics"])
    print("Saved:", paths["cases"])


def run(args, baseline, chunks, by_doc):
    records = read_records(baseline)
    config = load_config()
    paths = output_paths(args.radius, args.limit)
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    fingerprint = experiment_fingerprint(baseline, args.radius, config)
    if args.overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    existing = read_jsonl(paths["predictions"]) if paths["predictions"].exists() else []
    completed = validate_resume(existing, baseline, fingerprint)
    print(f"Expanded radius=±{args.radius} | count={len(baseline)} | completed={completed} | remaining={len(baseline) - completed}")
    print("Original Top 5 fixed; same baseline Judge + generation; no retrieval re-run")

    if completed < len(baseline):
        import torch
        from answerability_detection import select_evidence_documents
        from load_base_model import load_model
        from rag_pipeline import generate_rag_answer
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen 모델 실행을 위한 NVIDIA CUDA GPU가 필요합니다")
        model, tokenizer, generation_config = load_model(config)
        torch.cuda.reset_peak_memory_stats()
        with paths["predictions"].open("a", encoding="utf-8") as output:
            for index in range(completed, len(baseline)):
                base = baseline[index]
                documents = expanded_for_row(base, chunks, by_doc, args.radius)
                torch.cuda.synchronize()
                start = time.perf_counter()
                logger = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
                with logger:
                    evidence_documents = select_evidence_documents(model, tokenizer, base["question"], documents)
                torch.cuda.synchronize()
                judge_seconds = time.perf_counter() - start
                chosen = {d["row_id"] for d in evidence_documents}
                ranks = [r for r, d in enumerate(documents, 1) if d["row_id"] in chosen]
                if evidence_documents:
                    prediction, seconds, input_tokens = generate_rag_answer(
                        model, tokenizer, generation_config, base["question"],
                        evidence_documents, config["prompt"]
                    )
                else:
                    prediction, seconds, input_tokens = ABSTAIN, 0.0, 0
                result = {
                    "id": base["id"], "question": base["question"],
                    "reference_answer": base["reference_answer"],
                    "is_impossible": base["is_impossible"], "prediction": prediction,
                    "retrieved_count": len(documents), "evidence_count": len(ranks),
                    "evidence_ranks": ranks,
                    "expansion": [{
                        "rank": rank,
                        "filename": d["filename"],
                        "center_chunk_id": d["center_chunk_id"],
                        "included_chunk_ids": d["included_chunk_ids"],
                        "expanded_chars": len(d["text"]),
                    } for rank, d in enumerate(documents, 1)],
                    "judge_seconds": round(judge_seconds, 5),
                    "generation_seconds": round(seconds, 5),
                    "input_tokens": int(input_tokens),
                    "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
                    "fingerprint": fingerprint,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{index + 1:>3}/{len(baseline)}] {base['id']} evidence={len(ranks)} "
                      f"judge={judge_seconds:.2f}s gen={seconds:.2f}s "
                      f"answer={prediction[:75].replace(chr(10), ' ')!r}", flush=True)
        del model, tokenizer, generation_config

    rows = read_jsonl(paths["predictions"])
    validate_resume(rows, baseline, fingerprint)
    if len(rows) != len(baseline):
        raise ValueError("모든 질문에 대한 결과가 완료되지 않았습니다")
    score_run(paths, records, baseline, rows, args.radius)


def report():
    from evaluate.common import score_predictions, load_records
    baseline = load_baseline(0)
    gold = load_records("clean")
    print("[Chunk Expansion A/B: original fixed Top-5]")
    base_metrics = score_predictions(gold, [dict(r) for r in baseline], dataset="clean", batch_size=1)
    print("  baseline(radius 0 saved):",
          "F1=", base_metrics["token_f1_percent"],
          "correct_abstention=", base_metrics["abstention_rate_percent"],
          "false_refusal=", base_metrics["false_refusal_rate_percent"])
    for radius in (1, 2):
        metrics_path = output_paths(radius, 0)["metrics"]
        if not metrics_path.is_file():
            print(f"  radius {radius}: not evaluated")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("total") != 135:
            raise ValueError(f"비교 불가: radius {radius} 평가가 전체 135건이 아닙니다")
        print(f"  radius {radius}: F1={metrics['token_f1_percent']} "
              f"correct_abstention={metrics['abstention_rate_percent']} "
              f"false_refusal={metrics['false_refusal_rate_percent']} "
              f"gold_source_selected={metrics['gold_source_selected_questions']} "
              f"judge_s={metrics['total_judge_seconds']} "
              f"generation_s={metrics['total_generation_seconds']}")
    print("주의: 정답 출처 파일명은 청크 수준의 실제 근거 정답이 아닙니다.")


def main():
    args = parse_args()
    if args.mode == "report":
        report()
        return
    rows = load_baseline(args.limit)
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(f"원본 청크 파일이 없습니다: {CORPUS_PATH}")
    chunks, by_doc = load_corpus()
    validate_search_snapshot(rows, chunks)
    if args.mode == "inspect":
        inspect(rows, chunks, by_doc, args.radius)
        return
    run(args, rows, chunks, by_doc)


if __name__ == "__main__":
    main()
