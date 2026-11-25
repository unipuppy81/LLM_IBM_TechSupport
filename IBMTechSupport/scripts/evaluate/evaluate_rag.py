import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import yaml

from load_base_model import load_model
from .common import score_predictions, save_jsonl, report
from rag_pipeline import generate_rag_answer, retrieve_all
from rag_guardrails import answer_with_guardrails
from rag_guardrails_per_document import choose_guardrail_runner

PROJECT_DIR = Path(__file__).resolve().parents[2]


# YAML 설정 로드
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Validation 데이터 전체 답변 생성
def create_predictions(records, config):
    method = config["retrieval"]["method"]
    top_k = int(config["retrieval"]["top_k"])
    guardrails = config.get("guardrails", {})
    guarded = bool(guardrails.get("enabled", False))
    retry_top_k = int(guardrails.get("retry_top_k", top_k)) if guarded else top_k
    if retry_top_k < top_k:
        raise ValueError("guardrails.retry_top_k must be >= retrieval.top_k")

    guardrail_runner = choose_guardrail_runner(config) if guarded else None

    questions = [record["question"] for record in records]
    # Fetch the full fallback window once, but initially judge only top_k.
    documents_by_question, retrieval_seconds = retrieve_all(questions, method, retry_top_k)

    if len(documents_by_question) != len(records):
        raise RuntimeError("검색 결과 개수가 질문 개수와 다릅니다.")

    model, tokenizer, generation_config = load_model(config)
    torch.cuda.reset_peak_memory_stats()

    rows = []
    generation_seconds = 0.0
    judge_seconds = 0.0
    verification_seconds = 0.0
    total_input_tokens = 0
    guardrail_counts = Counter()

    for index, (record, documents) in enumerate(zip(records, documents_by_question), start=1):
        if guarded:
            guarded_result = guardrail_runner(
                question=record["question"],
                retrieved_documents=documents,
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                prompt_config=config["prompt"],
                generate_answer=generate_rag_answer,
                top_k=top_k,
                retry_top_k=retry_top_k,
            )
            answer = guarded_result["answer"]
            seconds = guarded_result["generation_seconds"]
            input_tokens = guarded_result["input_tokens"]
            judge_seconds += guarded_result["judge_seconds"]
            verification_seconds += guarded_result["verification_seconds"]
            guardrail_counts[guarded_result["status"]] += 1
            guardrail_data = {
                "status": guarded_result["status"],
                "judge_history": guarded_result["judge_history"],
                "verification": guarded_result["verification"],
                "used_chunk_ids": [d.get("chunk_id") for d in guarded_result["used_documents"]],
                "candidate_answer": guarded_result["candidate_answer"],
            }
        else:
            answer, seconds, input_tokens = generate_rag_answer(
                model, tokenizer, generation_config,
                record["question"], documents, config["prompt"],
            )
            guardrail_data = None

        generation_seconds += seconds
        total_input_tokens += input_tokens

        rows.append({
            "id": record["id"],
            "question": record["question"],
            "reference_answer": record["answer"],
            "is_impossible": record["is_impossible"],
            "prediction": answer,
            "retrieved_documents": documents,
            "input_tokens": input_tokens,
            **({"guardrails": guardrail_data} if guarded else {}),
        })

        status = f" | {guardrail_data['status']}" if guarded else ""
        print(f"Generated: {index}/{len(records)}{status}", flush=True)

    stats = {
        "retrieval_seconds": round(retrieval_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "avg_input_tokens": round(total_input_tokens / len(records), 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
    }
    if guarded:
        stats.update({
            "guardrails_enabled": True,
            "judge_mode": guardrails.get("judge_mode", "strict"),
            **({"answerability_prompt_sha256": guardrails["answerability_prompt_sha256"],
                "retrieval_config_sha256": guardrails["retrieval_config_sha256"]}
               if guardrails.get("judge_mode") == "per_document_v3" else {}),
            "judge_seconds": round(judge_seconds, 3),
            "verification_seconds": round(verification_seconds, 3),
            "guardrail_status_counts": dict(guardrail_counts),
        })

    return rows, stats


# RAG 평가 실행
def run(args):
    config = load_config(args.config)
    dataset_path = PROJECT_DIR / config["dataset"]["path"]

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    expected_count = config["dataset"]["expected_count"]

    if len(records) != expected_count:
        raise ValueError(
            f"Validation 데이터 개수 불일치: "
            f"expected={expected_count}, actual={len(records)}"
        )

    output_dir = PROJECT_DIR / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, performance = create_predictions(records, config)

    metrics = score_predictions(records, rows, "clean", config["inference"]["batch_size"])
    metrics.update({
        "retriever": config["retrieval"]["method"],
        "top_k": config["retrieval"]["top_k"],
        **performance,
    })

    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    save_jsonl(predictions_path, rows)

    return report(metrics, metrics_path)


# CLI 실행
def main(argv=None):
    parser = argparse.ArgumentParser(description="RAG 성능 평가")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/rag/v1.yaml")

    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()