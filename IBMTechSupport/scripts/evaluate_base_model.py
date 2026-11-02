
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 프로젝트 경로
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "Qwen2.5-1.5B-Instruct"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "val.json"
RESULT_DIR = PROJECT_DIR / "results"

# 평가 설정
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
    """평가를 위한 Unicode 정규화 및 토큰 분리."""
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    return re.findall(r"\w+|[^\w\s]", text)


def exact_match(prediction, reference):
    """정답과 예측이 정규화 후 동일한지 검사."""
    return int(
        tokenize_for_metrics(prediction)
        == tokenize_for_metrics(reference)
    )


def token_f1(prediction, reference):
    """정답과 예측의 토큰 Precision / Recall / F1."""
    pred_tokens = tokenize_for_metrics(prediction)
    ref_tokens = tokenize_for_metrics(reference)

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def load_model():
    """기존 Qwen 모델과 토크나이저 로드."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    model.eval()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)
    print("Model Load: OK")

    return model, tokenizer


def generate_answer(model, tokenizer, question):
    """Base Model로 기술지원 질문에 답변 생성."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    # Qwen Chat Template 적용
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # 학습 없이 추론만 수행
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 입력 부분 제외하고 생성 토큰만 추출
    input_length = inputs["input_ids"].shape[-1]
    generated_ids = outputs[0][input_length:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def percentage(numerator, denominator):
    """백분율 계산. 데이터가 없으면 None."""
    if denominator == 0:
        return None

    return round(numerator / denominator * 100, 2)


def evaluate(model, tokenizer, records, output_path):
    """Validation 데이터 평가 및 예측 저장."""

    stats = Counter()
    total_f1 = 0.0

    with open(output_path, "w", encoding="utf-8") as file:

        for index, record in enumerate(records, start=1):
            question = record["question"]
            reference = record["answer"]
            impossible = record["is_impossible"]

            # 모델 답변 생성
            prediction = generate_answer(model, tokenizer, question)

            # 정확히 지정된 문자열일 때만 답변 거부 인정
            abstained = prediction == ABSTAIN

            em = None
            f1 = None

            if impossible:
                stats["impossible"] += 1

                if abstained:
                    stats["correct_abstentions"] += 1

            else:
                stats["answerable"] += 1

                em = exact_match(prediction, reference)
                f1 = token_f1(prediction, reference)

                stats["exact_matches"] += em
                total_f1 += f1

                if abstained:
                    stats["false_refusals"] += 1

            # 개별 평가 결과
            result = {
                "id": record["id"],
                "question": question,
                "reference_answer": reference,
                "is_impossible": impossible,
                "prediction": prediction,
                "abstained": abstained,
                "exact_match": em,
                "token_f1": round(f1, 4) if f1 is not None else None
            }

            # JSONL 형태로 즉시 저장
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

            # 진행 상태 출력
            print(
                f"[{index}/{len(records)}] "
                f"{record['id']} | "
                f"F1={f1:.4f} " if f1 is not None
                else
                f"[{index}/{len(records)}] "
                f"{record['id']} | Impossible"
            )

    # 최종 평가 지표
    metrics = {
        "model": MODEL_NAME,
        "dataset": "validation",
        "total": len(records),
        "max_new_tokens": MAX_NEW_TOKENS,
        "answerable": stats["answerable"],
        "impossible": stats["impossible"],
        "exact_match_percent": percentage(
            stats["exact_matches"],
            stats["answerable"]
        ),
        "token_f1_percent": percentage(
            total_f1,
            stats["answerable"]
        ),
        "abstention_rate_percent": percentage(
            stats["correct_abstentions"],
            stats["impossible"]
        ),
        "false_refusal_rate_percent": percentage(
            stats["false_refusals"],
            stats["answerable"]
        )
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()

    # 0이면 Validation 전체 평가
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="평가할 질문 수. 0이면 전체 평가"
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Validation 데이터 로드
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    assert records, "Validation dataset is empty"

    # 일부 샘플만 검증 가능
    if args.limit > 0:
        records = records[:args.limit]
        prefix = "base_val_smoke"
    else:
        prefix = "base_val"

    prediction_path = RESULT_DIR / f"{prefix}_predictions.jsonl"
    metrics_path = RESULT_DIR / f"{prefix}_metrics.json"

    print("Evaluation Samples:", len(records))

    # Base Model 로드
    model, tokenizer = load_model()

    # 성능 평가
    metrics = evaluate(model, tokenizer, records, prediction_path)

    # 평가 지표 저장
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("\n[Base Model Evaluation]")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nPredictions:", prediction_path)
    print("Metrics:", metrics_path)

    print("Base Model Evaluation: OK")


if __name__ == "__main__":
    main()
