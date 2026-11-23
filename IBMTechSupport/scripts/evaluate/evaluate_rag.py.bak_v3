import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from load_base_model import load_model
from .common import score_predictions, save_jsonl, report
from rag_pipeline import generate_rag_answer, retrieve_all

PROJECT_DIR = Path(__file__).resolve().parents[2]


# YAML 설정 로드
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Validation 데이터 전체 답변 생성
def create_predictions(records, config):
    method = config["retrieval"]["method"]
    top_k = config["retrieval"]["top_k"]

    questions = [record["question"] for record in records]
    documents_by_question, retrieval_seconds = retrieve_all(questions, method, top_k)

    if len(documents_by_question) != len(records):
        raise RuntimeError("검색 결과 개수가 질문 개수와 다릅니다.")

    model, tokenizer, generation_config = load_model(config)

    torch.cuda.reset_peak_memory_stats()

    rows = []
    generation_seconds = 0.0
    total_input_tokens = 0

    for index, (record, documents) in enumerate(zip(records, documents_by_question), start=1):
        answer, seconds, input_tokens = generate_rag_answer(
            model,
            tokenizer,
            generation_config,
            record["question"],
            documents,
            config["prompt"],
        )

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
        })

        print(f"Generated: {index}/{len(records)}", flush=True)

    stats = {
        "retrieval_seconds": round(retrieval_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "avg_input_tokens": round(total_input_tokens / len(records), 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
    }

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