import argparse
import copy
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "Qwen2.5-1.5B-Instruct"
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

MODEL_NAME = "Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 160
ABSTAIN = "INSUFFICIENT_INFORMATION"
SYSTEM_PROMPT = (
    "You are an IBM technical support assistant. "
    "Answer technical questions using only your existing knowledge. "
    "If you cannot provide a reliable answer, respond with exactly "
    "INSUFFICIENT_INFORMATION. "
    "Otherwise, provide a concise technical answer in English."
)


def tokenize_for_metrics(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+|[^\w\s]", text)


def exact_match(prediction, reference):
    return int(tokenize_for_metrics(prediction) == tokenize_for_metrics(reference))


def token_f1(prediction, reference):
    predicted = tokenize_for_metrics(prediction)
    correct = tokenize_for_metrics(reference)
    if not predicted or not correct:
        return float(predicted == correct)

    common = Counter(predicted) & Counter(correct)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(correct)
    return 2 * precision * recall / (precision + recall)


def percentage(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else None


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer.padding_side = "left"  # Decoder-only 모델 Batch 생성에 필요
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    if {p.device.type for p in model.parameters()} != {"cuda"}:
        raise RuntimeError("모델 전체가 GPU에 배치되지 않았습니다.")

    # Greedy 생성 시 샘플링 파라미터 경고 방지
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None
    generation_config.top_k = None
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.pad_token_id = tokenizer.pad_token_id

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)
    print("Model Load: OK")
    return model, tokenizer, generation_config


def generate_batch(model, tokenizer, generation_config, questions):
    conversations = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for question in questions
    ]

    # 여러 대화를 동시에 토큰화하고 왼쪽 패딩
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
            outputs[:, input_width:], skip_special_tokens=True
        )
    ]


def make_prediction_rows(model, tokenizer, generation_config, records, batch_size):
    rows = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    for offset in range(0, len(records), batch_size):
        batch = records[offset : offset + batch_size]
        answers = generate_batch(
            model, tokenizer, generation_config,
            [record["question"] for record in batch],
        )

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


def validate_prediction_rows(records, predictions):
    if len(records) != len(predictions):
        raise ValueError("예측 파일 레코드 수가 평가 데이터와 다릅니다.")

    for record, result in zip(records, predictions):
        expected = (
            record["id"], record["question"],
            record["answer"], record["is_impossible"],
        )
        actual = (
            result["id"], result["question"],
            result["reference_answer"], result["is_impossible"],
        )
        if actual != expected or not isinstance(result.get("prediction"), str):
            raise ValueError(f"평가 데이터와 예측 결과가 불일치: {record['id']}")


def score_predictions(records, predictions, dataset, batch_size):
    validate_prediction_rows(records, predictions)
    stats = Counter()
    total_f1 = 0.0

    for record, row in zip(records, predictions):
        prediction = row["prediction"]
        abstained = prediction == ABSTAIN
        row["abstained"] = abstained

        if record["is_impossible"]:
            stats["impossible"] += 1
            stats["correct_abstentions"] += int(abstained)
            row["exact_match"] = None
            row["token_f1"] = None
        else:
            stats["answerable"] += 1
            em = exact_match(prediction, record["answer"])
            f1 = token_f1(prediction, record["answer"])
            stats["exact_matches"] += em
            stats["false_refusals"] += int(abstained)
            total_f1 += f1
            row["exact_match"] = em
            row["token_f1"] = round(f1, 4)

    return {
        "model": MODEL_NAME,
        "dataset": dataset,
        "total": len(records),
        "batch_size": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "answerable": stats["answerable"],
        "impossible": stats["impossible"],
        "exact_match_percent": percentage(stats["exact_matches"], stats["answerable"]),
        "token_f1_percent": percentage(total_f1, stats["answerable"]),
        "abstention_rate_percent": percentage(
            stats["correct_abstentions"], stats["impossible"]
        ),
        "false_refusal_rate_percent": percentage(
            stats["false_refusals"], stats["answerable"]
        ),
    }


def save_jsonl(path, records):
    temp_path = path.with_suffix(path.suffix + ".part")
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("clean", "original"), default="clean")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--predictions", type=Path, default=None,
        help="--reuse 시 기존 JSONL 파일 경로 (미지정 시 현재 Batch 출력 사용)",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size >= 1, --limit >= 0 이어야 합니다.")
    if args.predictions is not None and not args.reuse:
        parser.error("--predictions는 --reuse와 함께 사용해야 합니다.")

    data_name = "val_clean.json" if args.dataset == "clean" else "val.json"
    with open(DATA_DIR / data_name, encoding="utf-8") as file:
        records = json.load(file)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError("평가 데이터가 비어 있습니다.")

    prefix = f"base_val_{args.dataset}_b{args.batch_size}"
    if args.limit:
        prefix += f"_smoke{args.limit}"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = RESULT_DIR / f"{prefix}_predictions.jsonl"
    metrics_path = RESULT_DIR / f"{prefix}_metrics.json"
    inference_seconds = None
    peak_memory_gb = None

    print("Evaluation Samples:", len(records))
    print("Dataset:", data_name)
    print("Batch Size:", args.batch_size)

    if args.reuse:
        source_path = args.predictions or prediction_path
        predictions = read_jsonl(source_path)
        print("Reuse Predictions:", source_path)
        # 원본 생성 당시 저장된 속도 측정값이 있다면 유지
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as file:
                old_metrics = json.load(file)
            if (
                old_metrics.get("dataset") == args.dataset
                and old_metrics.get("total") == len(records)
                and old_metrics.get("batch_size") == args.batch_size
            ):
                inference_seconds = old_metrics.get("generation_seconds")
                peak_memory_gb = old_metrics.get("peak_gpu_memory_gb")
    else:
        model, tokenizer, generation_config = load_model()
        predictions, inference_seconds, peak_memory_gb = make_prediction_rows(
            model, tokenizer, generation_config, records, args.batch_size
        )

    metrics = score_predictions(records, predictions, args.dataset, args.batch_size)
    metrics["generation_seconds"] = inference_seconds
    metrics["questions_per_second"] = (
        round(len(records) / inference_seconds, 3)
        if inference_seconds is not None and inference_seconds > 0 else None
    )
    metrics["peak_gpu_memory_gb"] = peak_memory_gb

    # 재평가 시 원본 예측 파일은 변경하지 않고 지표만 다시 저장
    if not args.reuse:
        save_jsonl(prediction_path, predictions)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("\n[Base Model Evaluation]")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Metrics:", metrics_path)
    print("Base Model Evaluation: OK")


if __name__ == "__main__":
    main()
