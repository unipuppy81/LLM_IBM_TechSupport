import argparse
import json
import time

from pathlib import Path


import torch
import yaml

from load_base_model import load_model
from .common import score_predictions, save_jsonl, report

PROJECT_DIR = Path(__file__).resolve().parents[2]

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run(args):
    # 1. YAML 설정 로드
    config = load_config(args.config)

    # 2. Validation 데이터 로드
    dataset_path = PROJECT_DIR / config["dataset"]["path"]

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 3. 전체 데이터 개수 검증
    expected_count = config["dataset"]["expected_count"]

    if len(records) != expected_count:
        raise ValueError(
            f"Validation 데이터 개수 불일치: "
            f"expected={expected_count}, actual={len(records)}"
        )

    # 일부 데이터만 테스트할 때 사용
    if args.limit < 0:
        raise ValueError("limit must be >= 0")

    if args.limit:
        records = records[:args.limit]

    # 4. 추론 설정
    batch_size = config["inference"]["batch_size"]

    # 5. 결과 저장 폴더
    output_dir = PROJECT_DIR / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. 모델 및 토크나이저 로드
    model, tokenizer, generation_config = load_model(config)

    # 7. GPU 추론
    rows, seconds, memory = make_prediction_rows(
        model,
        tokenizer,
        generation_config,
        records,
        batch_size,
        config["prompt"],
    )

    # 8. 성능 평가
    metrics = score_predictions(records, rows, "clean", batch_size)

    metrics.update({
        "generation_seconds": seconds,
        "questions_per_second": (
            round(len(records) / seconds, 3)
            if seconds > 0 else None
        ),
        "peak_gpu_memory_gb": memory,
    })

    # 9. 결과 저장
    prefix = "smoke_" if args.limit else ""
    predictions_path = output_dir / f"{prefix}predictions.jsonl"
    metrics_path = output_dir / f"{prefix}metrics.json"

    save_jsonl(predictions_path, rows)

    return report(metrics, metrics_path)



def generate_batch(model, tokenizer, generation_config, questions, prompt_config):
    conversations = [
        [
            {"role": "system", "content": prompt_config["system"]},
            {"role": "user", "content": prompt_config["user_template"].format(question=question)},
        ]
        for question in questions
    ]

    inputs = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation_config)

    input_width = inputs["input_ids"].shape[1]

    return [
        answer.strip()
        for answer in tokenizer.batch_decode(
            outputs[:, input_width:],
            skip_special_tokens=True,
        )
    ]


def make_prediction_rows(model, tokenizer, generation_config, records, batch_size, prompt_config):
    rows = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        answers = generate_batch(model, tokenizer, generation_config, [record["question"] for record in batch], prompt_config)

        if len(answers) != len(batch):
            raise RuntimeError("Batch 응답 개수가 입력과 다릅니다.")

        for record, answer in zip(batch, answers):
            rows.append({
                "id": record["id"],
                "question": record["question"],
                "reference_answer": record["answer"],
                "is_impossible": record["is_impossible"],
                "prediction": answer,
            })
        print(f"Generated: {len(rows)}/{len(records)}", flush=True)

    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return rows, seconds, peak_gb


def main(argv=None):
    parser = argparse.ArgumentParser(description="Base LLM 평가")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/base/v1.yaml")
    parser.add_argument("--limit", type=int, default=0)

    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
