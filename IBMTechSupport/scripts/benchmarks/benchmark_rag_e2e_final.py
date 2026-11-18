"""End-to-end IBM RAG validation: retrieval -> evidence judge -> Qwen answer.

Unlike evaluate/evaluate_rag.py, this follows the production run_rag() path
by calling select_evidence_documents() before generation. Does not change
retrieval.yaml, model weights, or search code.

From the repository root (PowerShell):
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_e2e_final.py --limit 3
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_e2e_final.py
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_e2e_final.py --resume

Optional identical-pipeline comparison:
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_e2e_final.py --method hybrid

Output:
    IBMTechSupport/evaluate_results/rag/final_e2e/<method>/predictions.jsonl
    IBMTechSupport/evaluate_results/rag/final_e2e/<method>/metrics.json
    (smoke runs use separate smoke_<limit>_ prefixed filenames)
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch
import yaml

from answerability_detection import select_evidence_documents
from evaluate.common import (
    ABSTAIN,
    load_jsonl,
    load_records,
    score_predictions,
    validate_prediction_rows,
)
from load_base_model import load_model
from rag_pipeline import generate_rag_answer, retrieve_all
from retrieval_config import CONFIG_PATH as RETRIEVAL_CONFIG_PATH


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAG_CONFIG_PATH = PROJECT_DIR / "configs" / "rag" / "v1.yaml"
OUTPUT_ROOT = PROJECT_DIR / "evaluate_results" / "rag" / "final_e2e"
VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description="실제 서비스 경로를 사용하는 최종 RAG 답변 평가")
    parser.add_argument("--method", choices=("reranker", "hybrid"), default="reranker")
    parser.add_argument("--limit", type=int, default=0, help="0=135개 전부, 양수=스모크 테스트")
    parser.add_argument("--resume", action="store_true", help="중단된 JSONL에서 이어서 실행")
    parser.add_argument("--overwrite", action="store_true", help="해당 실험의 기존 결과만 삭제하고 재시작")
    parser.add_argument("--verbose", action="store_true", help="문서별 YES/NO 원문 출력")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit은 0 이상이어야 합니다.")
    if args.resume and args.overwrite:
        parser.error("--resume과 --overwrite는 함께 사용할 수 없습니다.")
    return args


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"잘못된 YAML 설정: {path}")
    return config


def fingerprint(records, rag_config, retrieval_config, method, top_k):
    """Resume 시 입력 데이터, 설정, 검색/생성 로직 변경을 감지."""
    scripts_dir = PROJECT_DIR / "scripts"
    source_names = (
        "rag_pipeline.py", "answerability_detection.py", "load_base_model.py",
        "hybrid_search.py", "bm25_search.py", "dense_search.py",
        "reranker_search.py", "document_utils.py",
    )
    source_hashes = {}
    for name in source_names:
        path = scripts_dir / name
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {
        "evaluation_version": VERSION,
        "records": records,
        "rag_config": rag_config,
        "retrieval_config": retrieval_config,
        "method": method,
        "top_k": top_k,
        "sources": source_hashes,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def output_paths(method, limit):
    directory = OUTPUT_ROOT / method
    prefix = f"smoke_{limit}_" if limit else ""
    return directory / f"{prefix}predictions.jsonl", directory / f"{prefix}metrics.json"


def validate_resumed_rows(existing, records, expected_fingerprint):
    """이미 평가한 레코드는 순서와 내용이 모두 일치해야 재사용."""
    if len(existing) > len(records):
        raise ValueError("저장된 행 수가 평가 데이터보다 많습니다.")
    for offset, row in enumerate(existing):
        record = records[offset]
        if row.get("fingerprint") != expected_fingerprint:
            raise ValueError("설정/데이터/코드가 변경되었습니다. --resume 불가; 새 평가를 수행하세요.")
        expected = (record["id"], record["question"], record["answer"], record["is_impossible"])
        actual = (row.get("id"), row.get("question"), row.get("reference_answer"), row.get("is_impossible"))
        if actual != expected or not isinstance(row.get("prediction"), str):
            raise ValueError(f"중단된 결과 파일의 {offset + 1}번째 레코드가 현재 데이터와 다릅니다.")
    return len(existing)


def save_metrics_atomic(path, metrics):
    """기존 결과 파일을 훼손하지 않고 기록."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def print_progress(index, total, record, row):
    label = "impossible" if record["is_impossible"] else "answerable"
    answer = row["prediction"].replace("\n", " ")[:90]
    print(
        f"[{index:>3}/{total}] {record['id']} ({label}) "
        f"evidence={row['evidence_count']}/{row['retrieved_count']} "
        f"judge={row['evidence_seconds']:.2f}s "
        f"generate={row['generation_seconds']:.2f}s "
        f"answer={answer!r}",
        flush=True,
    )


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("load_base_model.load_model은 CUDA GPU가 필요합니다.")

    rag_config = read_yaml(RAG_CONFIG_PATH)
    retrieval_config = read_yaml(RETRIEVAL_CONFIG_PATH)
    top_k = int(rag_config["retrieval"]["top_k"])
    if not 1 <= top_k <= 5:
        raise ValueError("이번 벤치마크는 운영 설정 retrieval.top_k=5 이하를 전제로 합니다.")

    records = load_records("clean")
    expected = int(rag_config["dataset"]["expected_count"])
    if len(records) != expected or expected != 135:
        raise ValueError(f"Validation 데이터 개수 불일치: expected=135, actual={len(records)}")

    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("평가할 데이터가 없습니다.")

    pred_path, metrics_path = output_paths(args.method, args.limit)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    tag = fingerprint(records, rag_config, retrieval_config, args.method, top_k)

    if args.overwrite:
        pred_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)
    elif not args.resume and (pred_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"기존 결과 보호: {pred_path} 또는 {metrics_path}. "
            "중단 후 재개는 --resume, 처음부터 재실행은 --overwrite."
        )
    elif args.resume and not pred_path.exists():
        raise FileNotFoundError(f"--resume 대상이 없습니다: {pred_path}")

    existing = load_jsonl(pred_path) if args.resume else []
    done = validate_resumed_rows(existing, records, tag)
    missing = records[done:]

    print(f"Method: {args.method} | Top K: {top_k}")
    print(f"Validation: {len(records)} | completed={done} | remaining={len(missing)}")
    print("Pipeline: Hybrid -> (Reranker) -> per-document evidence judge -> Qwen generation")
    print("Model:", rag_config["model"]["name"])
    print("Hybrid config:", retrieval_config["hybrid"])
    print("Reranker config:", retrieval_config["reranker"])
    print("Predictions:", pred_path)

    if missing:
        questions = [row["question"] for row in missing]
        # 서비스 코드와 동일한 검색 함수: top_k 5 문서를 반환.
        documents_by_question, retrieval_seconds = retrieve_all(
            questions=questions, method=args.method, top_k=top_k
        )
        if len(documents_by_question) != len(missing):
            raise RuntimeError("검색 결과와 질문 수가 다릅니다.")
        print(f"Shared retrieval: {retrieval_seconds:.3f}s", flush=True)

        # 서비스 코드와 동일한 모델 로더 및 생성 설정.
        model, tokenizer, generation_config = load_model(rag_config)
        torch.cuda.reset_peak_memory_stats()
        per_query_retrieval = retrieval_seconds / len(missing)

        with pred_path.open("a" if args.resume else "x", encoding="utf-8") as out:
            for index, (record, documents) in enumerate(
                zip(missing, documents_by_question), start=done + 1
            ):
                question = record["question"]

                torch.cuda.synchronize()
                started = time.perf_counter()
                # 서비스의 run_rag()와 동일하게 문서 1개씩 YES/NO 판별.
                log_context = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
                with log_context:
                    evidence_documents = select_evidence_documents(
                        model, tokenizer, question, documents
                    )
                torch.cuda.synchronize()
                evidence_seconds = time.perf_counter() - started

                # 서비스의 실제 분기: 선택된 근거 문서가 없으면 응답 거부.
                if evidence_documents:
                    prediction, generation_seconds, input_tokens = generate_rag_answer(
                        model, tokenizer, generation_config,
                        question, evidence_documents, rag_config["prompt"],
                    )
                else:
                    prediction, generation_seconds, input_tokens = ABSTAIN, 0.0, 0

                row = {
                    "id": record["id"],
                    "question": question,
                    "reference_answer": record["answer"],
                    "is_impossible": record["is_impossible"],
                    "prediction": prediction,
                    "retrieved_documents": documents,
                    "evidence_documents": evidence_documents,
                    "retrieved_count": len(documents),
                    "evidence_count": len(evidence_documents),
                    "input_tokens": int(input_tokens),
                    "retrieval_seconds_allocated": round(per_query_retrieval, 7),
                    "evidence_seconds": round(evidence_seconds, 5),
                    "generation_seconds": round(generation_seconds, 5),
                    "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
                    "fingerprint": tag,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()  # 오류/중단 시 이전 질문 결과 보존.
                print_progress(index, len(records), record, row)

        del model, tokenizer, generation_config

    rows = load_jsonl(pred_path)
    validate_prediction_rows(records, rows)
    metrics = score_predictions(
        records, rows, dataset="clean", batch_size=int(rag_config["inference"]["batch_size"])
    )
    sums = {
        "retrieval": sum(r["retrieval_seconds_allocated"] for r in rows),
        "evidence": sum(r["evidence_seconds"] for r in rows),
        "generation": sum(r["generation_seconds"] for r in rows),
    }
    processing_seconds = sum(sums.values())
    metrics.update({
        "retriever": args.method,
        "top_k": top_k,
        "evaluated_count": len(rows),
        "smoke_test": bool(args.limit),
        "evidence_selected_questions": sum(bool(r["evidence_count"]) for r in rows),
        "correct_abstentions": sum(r["is_impossible"] and r["prediction"] == ABSTAIN for r in rows),
        "false_refusals": sum(not r["is_impossible"] and r["prediction"] == ABSTAIN for r in rows),
        "retrieval_seconds": round(sums["retrieval"], 3),
        "evidence_seconds": round(sums["evidence"], 3),
        "generation_seconds": round(sums["generation"], 3),
        "pipeline_seconds": round(processing_seconds, 3),
        "avg_pipeline_seconds_per_question": round(processing_seconds / len(rows), 3),
        "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / len(rows), 1),
        "peak_gpu_memory_gib": round(max(r["peak_gpu_memory_gib"] for r in rows), 3),
        "config_fingerprint": tag,
        "note": "Pipeline time excludes model loading and file IO; shared retrieval amortized across the evaluated questions.",
    })
    save_metrics_atomic(metrics_path, metrics)
    print("\nFinal RAG metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved:", pred_path)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    run(parse_args())
