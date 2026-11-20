"""Evidence Judge A/B 실험: 동일한 Reranker Top 5에서 Judge만 제거.

기준 A: final_e2e/reranker/predictions.jsonl (Judge 사용, 기존 135개)
실험 B: 위 파일의 retrieved_documents를 그대로 사용해 Judge 없이 Qwen 생성

프로젝트 루트 / PowerShell:
    $env:PYTHONPATH = "IBMTechSupport/scripts"
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_judge_ablation.py --limit 3
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_judge_ablation.py
    python IBMTechSupport/scripts/benchmarks/benchmark_rag_judge_ablation.py --resume

기존 최종 평가 결과와 설정/코드는 변경하지 않음.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

from benchmark_rag_e2e_final import fingerprint as baseline_fingerprint
from benchmark_rag_e2e_final import read_yaml
from evaluate.common import (
    ABSTAIN,
    load_jsonl,
    load_records,
    score_predictions,
    validate_prediction_rows,
)
from load_base_model import load_model
from rag_pipeline import generate_rag_answer
from retrieval_config import CONFIG_PATH as RETRIEVAL_CONFIG_PATH


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAG_CONFIG_PATH = PROJECT_DIR / "configs" / "rag" / "v1.yaml"
BASELINE_PATH = PROJECT_DIR / "evaluate_results" / "rag" / "final_e2e" / "reranker" / "predictions.jsonl"
OUTPUT_DIR = PROJECT_DIR / "evaluate_results" / "rag" / "judge_ablation" / "reranker"
EXPERIMENT_VERSION = 1


def arguments():
    parser = argparse.ArgumentParser(description="동일 Top 5 검색 결과에서 Evidence Judge 사용/미사용 비교")
    parser.add_argument("--limit", type=int, default=0, help="0=전체 135개, 양수=앞에서 N개 테스트")
    parser.add_argument("--resume", action="store_true", help="중간 저장 파일부터 재개")
    parser.add_argument("--overwrite", action="store_true", help="현재 실험 출력 파일만 삭제하고 재실행")
    args = parser.parse_args()
    if not 0 <= args.limit <= 135:
        parser.error("--limit은 0~135여야 합니다.")
    if args.resume and args.overwrite:
        parser.error("--resume과 --overwrite를 함께 사용할 수 없습니다.")
    return args


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def experiment_tag(records, config):
    """중단 재개 시 근거 데이터와 생성 코드 변경 방지."""
    source_paths = [
        PROJECT_DIR / "scripts" / "rag_pipeline.py",
        PROJECT_DIR / "scripts" / "load_base_model.py",
    ]
    payload = {
        "version": EXPERIMENT_VERSION,
        "records": records,
        "rag_config": config,
        "baseline_sha256": file_hash(BASELINE_PATH),
        "generation_sources": {path.name: file_hash(path) for path in source_paths},
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def output_paths(limit):
    prefix = f"smoke_{limit}_" if limit else ""
    return {
        "predictions": OUTPUT_DIR / f"{prefix}predictions_no_judge.jsonl",
        "metrics": OUTPUT_DIR / f"{prefix}metrics_no_judge.json",
        "comparison": OUTPUT_DIR / f"{prefix}comparison.json",
        "cases": OUTPUT_DIR / f"{prefix}comparison_cases.csv",
    }


def save_json_atomic(path, value):
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def save_csv_atomic(path, rows):
    fieldnames = (
        "id", "is_impossible", "retrieved_count", "judge_evidence_count",
        "judge_abstained", "no_judge_abstained", "judge_token_f1",
        "no_judge_token_f1", "f1_delta", "judge_prediction", "no_judge_prediction",
    )
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verify_baseline(records, baseline, rag_config, retrieval_config):
    """원본 Judge-on 평가가 현재 코드/설정과 동일한지 검증."""
    if len(records) != 135 or len(baseline) != 135:
        raise ValueError("이번 A/B 테스트에는 전체 135개 Judge-on 기준 예측이 필요합니다.")
    validate_prediction_rows(records, baseline)
    if rag_config["retrieval"]["method"] != "reranker":
        raise ValueError("기준 평가의 retrieval.method는 reranker여야 합니다.")
    top_k = int(rag_config["retrieval"]["top_k"])
    if top_k != 5:
        raise ValueError("기준 평가의 retrieval.top_k=5가 필요합니다.")

    # 기준 평가 스크립트에서 사용하는 동일한 fingerprint를 재계산.
    expected = baseline_fingerprint(records, rag_config, retrieval_config, "reranker", top_k)
    tags = {row.get("fingerprint") for row in baseline}
    if tags != {expected}:
        raise ValueError(
            "기존 Judge-on 평가 이후 retrieval.yaml / RAG YAML / 검색·생성 코드 또는 "
            "데이터가 변경되어 공정한 A/B 비교를 보장할 수 없습니다. "
            "기존 기준 실험과 현재 설정을 확인하세요."
        )

    for row in baseline:
        docs = row.get("retrieved_documents")
        evidence = row.get("evidence_documents")
        if not isinstance(docs, list) or len(docs) != top_k:
            raise ValueError(f"기준 Top 5 검색 결과가 없습니다: {row['id']}")
        if not isinstance(evidence, list):
            raise ValueError(f"기준 Judge 선택 문서가 없습니다: {row['id']}")
        if any(not isinstance(d, dict) or not isinstance(d.get("text"), str) for d in docs):
            raise ValueError(f"검색 문서에 text가 없습니다: {row['id']}")
    return top_k


def verify_resume(rows, records, tag):
    if len(rows) > len(records):
        raise ValueError("재개 파일의 데이터 수가 현재 실험보다 많습니다.")
    for i, row in enumerate(rows):
        record = records[i]
        if row.get("experiment_tag") != tag:
            raise ValueError("데이터/설정/생성 코드가 변경되어 재개할 수 없습니다.")
        target = (record["id"], record["question"], record["answer"], record["is_impossible"])
        got = (row.get("id"), row.get("question"), row.get("reference_answer"), row.get("is_impossible"))
        if target != got or not isinstance(row.get("prediction"), str):
            raise ValueError(f"재개 파일의 {i + 1}번째 질문 내용이 불일치합니다.")
    return len(rows)


def score_side(records, rows, config, top_k, no_judge):
    """같은 평가 함수 사용: EM/F1은 90개, 거부율은 해당 90/45개 분모."""
    metrics = score_predictions(records, rows, "clean", int(config["inference"]["batch_size"]))
    metrics.update({
        "retriever": "reranker",
        "top_k": top_k,
        "judge_enabled": not no_judge,
        "false_refusals": sum(not r["is_impossible"] and r["prediction"] == ABSTAIN for r in rows),
        "correct_abstentions": sum(r["is_impossible"] and r["prediction"] == ABSTAIN for r in rows),
        "incorrect_answers_to_impossible": sum(r["is_impossible"] and r["prediction"] != ABSTAIN for r in rows),
        "generation_seconds": round(sum(float(r.get("generation_seconds", 0)) for r in rows), 3),
        "avg_input_tokens": round(sum(int(r.get("input_tokens", 0)) for r in rows) / len(rows), 1),
        "peak_gpu_memory_gib": round(max(float(r.get("peak_gpu_memory_gib", 0)) for r in rows), 3),
    })
    # 동일한 검색 문서를 사용했기 때문에 기존 Retrieval 시간은 양쪽에 공통으로 배분.
    common_retrieval = sum(float(r.get("retrieval_seconds_allocated", 0)) for r in rows)
    judge_seconds = 0 if no_judge else sum(float(r.get("evidence_seconds", 0)) for r in rows)
    metrics["retrieval_seconds_shared"] = round(common_retrieval, 3)
    metrics["evidence_judge_seconds"] = round(judge_seconds, 3)
    metrics["pipeline_seconds_estimated"] = round(
        common_retrieval + judge_seconds + metrics["generation_seconds"], 3
    )
    metrics["avg_pipeline_seconds_per_question_estimated"] = round(
        metrics["pipeline_seconds_estimated"] / len(rows), 3
    )
    return metrics


def build_comparison(records, judge_rows, nojudge_rows, on, off):
    cases = []
    changes = {
        "answerable_refusals_resolved": 0,
        "answerable_new_refusals": 0,
        "impossible_correct_refusals_lost": 0,
        "impossible_new_correct_refusals": 0,
        "answerable_f1_improved": 0,
        "answerable_f1_declined": 0,
    }
    for record, before, after in zip(records, judge_rows, nojudge_rows):
        impossible = record["is_impossible"]
        was_refused = before["prediction"] == ABSTAIN
        now_refused = after["prediction"] == ABSTAIN
        f1_before = before.get("token_f1")
        f1_after = after.get("token_f1")
        delta = round(f1_after - f1_before, 4) if f1_before is not None else None
        if not impossible:
            changes["answerable_refusals_resolved"] += int(was_refused and not now_refused)
            changes["answerable_new_refusals"] += int(not was_refused and now_refused)
            changes["answerable_f1_improved"] += int(delta is not None and delta > 0)
            changes["answerable_f1_declined"] += int(delta is not None and delta < 0)
        else:
            changes["impossible_correct_refusals_lost"] += int(was_refused and not now_refused)
            changes["impossible_new_correct_refusals"] += int(not was_refused and now_refused)
        cases.append({
            "id": record["id"],
            "is_impossible": impossible,
            "retrieved_count": len(before["retrieved_documents"]),
            "judge_evidence_count": len(before["evidence_documents"]),
            "judge_abstained": was_refused,
            "no_judge_abstained": now_refused,
            "judge_token_f1": f1_before,
            "no_judge_token_f1": f1_after,
            "f1_delta": delta,
            "judge_prediction": before["prediction"],
            "no_judge_prediction": after["prediction"],
        })

    comparison = {
        "design": "Identical saved top-5 reranker documents; only evidence filtering removed",
        "records": len(records),
        "judge_on": on,
        "judge_off": off,
        "delta_off_minus_on": {
            key: round(off[key] - on[key], 2)
            for key in ("exact_match_percent", "token_f1_percent", "abstention_rate_percent", "false_refusal_rate_percent")
        },
        "case_changes": changes,
        "caution": (
            "Abstention rates use original dataset labels. A query marked impossible "
            "may still be answerable from newly retrieved documents. Human review is needed."
        ),
    }
    return comparison, cases


def main():
    args = arguments()
    config = read_yaml(RAG_CONFIG_PATH)
    retrieval_config = read_yaml(RETRIEVAL_CONFIG_PATH)
    records_all = load_records("clean")
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(f"Judge 사용 기준 예측 결과가 없습니다: {BASELINE_PATH}")
    baseline_all = load_jsonl(BASELINE_PATH)
    top_k = verify_baseline(records_all, baseline_all, config, retrieval_config)

    records = records_all[:args.limit] if args.limit else records_all
    judge_rows = baseline_all[:len(records)]
    tag = experiment_tag(records, config)
    paths = output_paths(args.limit)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in paths.values():
            path.unlink(missing_ok=True)
    elif not args.resume and any(p.exists() for p in paths.values()):
        raise FileExistsError("Judge 미사용 실험 결과가 이미 존재합니다. --resume 또는 --overwrite 사용하세요.")
    elif args.resume and not paths["predictions"].exists():
        raise FileNotFoundError(f"재개 파일이 없습니다: {paths['predictions']}")

    existing = load_jsonl(paths["predictions"]) if args.resume else []
    completed = verify_resume(existing, records, tag)
    print(f"A/B test: Evidence Judge ON vs OFF | same saved Top {top_k}")
    print(f"Data: {len(records)} | completed={completed} | remaining={len(records)-completed}")
    print("Baseline:", BASELINE_PATH)
    print("No-Judge predictions:", paths["predictions"])
    print("Hybrid/Reranker is NOT re-run. Original retrieval, generation config are fixed.")

    if completed < len(records):
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen 모델 실행에 CUDA가 필요합니다.")
        model, tokenizer, gen_config = load_model(config)
        torch.cuda.reset_peak_memory_stats()
        with paths["predictions"].open("a" if args.resume else "x", encoding="utf-8") as stream:
            for i in range(completed, len(records)):
                record = records[i]
                saved = judge_rows[i]
                # 재검색하지 않고 Judge-on에서 저장한 원래 Top 5를 그대로 전달.
                documents = saved["retrieved_documents"]
                answer, seconds, input_tokens = generate_rag_answer(
                    model, tokenizer, gen_config,
                    record["question"], documents, config["prompt"],
                )
                row = {
                    "id": record["id"],
                    "question": record["question"],
                    "reference_answer": record["answer"],
                    "is_impossible": record["is_impossible"],
                    "prediction": answer,
                    "retrieved_documents": documents,
                    "input_tokens": int(input_tokens),
                    "retrieval_seconds_allocated": float(saved.get("retrieval_seconds_allocated", 0)),
                    "evidence_seconds": 0.0,
                    "generation_seconds": round(seconds, 5),
                    "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
                    "experiment_tag": tag,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                short = answer.replace("\n", " ")[:90]
                print(f"[{i+1:>3}/{len(records)}] {record['id']} "
                      f"gen={seconds:.2f}s no_judge={short!r}", flush=True)
        del model, tokenizer, gen_config

    nojudge_rows = load_jsonl(paths["predictions"])
    validate_prediction_rows(records, nojudge_rows)
    judge_metrics = score_side(records, judge_rows, config, top_k, no_judge=False)
    nojudge_metrics = score_side(records, nojudge_rows, config, top_k, no_judge=True)
    comparison, cases = build_comparison(records, judge_rows, nojudge_rows, judge_metrics, nojudge_metrics)

    # 기존 기준 예측/지표를 절대로 덮어쓰지 않고 별도 폴더에 저장.
    save_json_atomic(paths["metrics"], nojudge_metrics)
    save_json_atomic(paths["comparison"], comparison)
    save_csv_atomic(paths["cases"], cases)

    print("\n[Judge ON vs OFF]")
    print("                 Judge ON     Judge OFF      Difference(OFF-ON)")
    for key in ("token_f1_percent", "exact_match_percent", "abstention_rate_percent", "false_refusal_rate_percent"):
        print(f"{key:<26}{judge_metrics[key]:>9.2f} {nojudge_metrics[key]:>12.2f} "
              f"{nojudge_metrics[key]-judge_metrics[key]:>15.2f}")
    print("Case changes:", json.dumps(comparison["case_changes"], ensure_ascii=False))
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
