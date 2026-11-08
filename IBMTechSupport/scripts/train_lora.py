import argparse
import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_base_model import MODEL_DIR, DATA_DIR, RESULT_DIR, SYSTEM_PROMPT, ABSTAIN


TRAIN_PATH = DATA_DIR / "train_clean.json"
VAL_PATH = DATA_DIR / "val_clean.json"

LORA_DIR = MODEL_DIR.parent / "Qwen2.5-1.5B-Instruct-LoRA"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
MAX_ANSWER_TOKENS = 160
MAX_QUESTION_TOKENS = 192



# region 데이터 로드

def load_records(path, limit=0):
    if not path.is_file():
        raise FileNotFoundError(f"데이터 파일 없음: {path}")

    with path.open(encoding="utf-8") as file:
        records = json.load(file)

    if limit:
        records = records[:limit]

    if not records:
        raise ValueError(f"데이터가 비어 있습니다: {path}")

    for record in records:
        for key in ("id", "question", "answer", "is_impossible"):
            if key not in record:
                raise ValueError(f"필수 필드 없음: {key}")

        if not isinstance(record["question"], str):
            raise ValueError(f"질문 형식 오류: {record['id']}")

        if not isinstance(record["is_impossible"], bool):
            raise ValueError(f"답변 가능 여부 형식 오류: {record['id']}")

        if (
            not record["is_impossible"]
            and (
                not isinstance(record["answer"], str)
                or not record["answer"].strip()
            )
        ):
            raise ValueError(f"정답 형식 오류: {record['id']}")

    return records

# endregion


# region 학습 데이터 토큰화

class QADataset(Dataset):

    def __init__(self, records, tokenizer, max_length):
        self.samples = []

        for record in records:
            question = record["question"]

            # 긴 질문은 토큰 단위로 제한
            question_tokens = tokenizer.encode(question, add_special_tokens=False)[:MAX_QUESTION_TOKENS]
            question = tokenizer.decode(question_tokens, skip_special_tokens=True)

            # Base Model 평가와 동일한 시스템 프롬프트
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]

            # 모델에 입력할 질문 부분
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

            # 정답이 없으면 답변 거부 문자열을 학습
            if record["is_impossible"]:
                answer = ABSTAIN
            else:
                answer = record["answer"]

            answer_ids = tokenizer.encode(answer, add_special_tokens=False)

            # 답변 토큰 길이 제한
            remaining = max_length - len(prompt_ids) - 1

            if remaining < 1:
                raise ValueError(f"학습 가능한 답변 공간이 없습니다: {record['id']}")

            answer_ids = answer_ids[:min(MAX_ANSWER_TOKENS, remaining)]

            # 답변 종료 토큰 추가
            target_ids = answer_ids + [tokenizer.eos_token_id]

            input_ids = prompt_ids + target_ids

            # 질문 부분은 Loss 계산에서 제외
            # -100은 CrossEntropyLoss의 ignore_index
            labels = ([-100] * len(prompt_ids) + target_ids)

            self.samples.append({
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
                "labels": torch.tensor([labels], dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

# endregion


# region Base Model 및 LoRA 설정

def load_lora_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, device_map={"": 0}, local_files_only=True)

    # 학습 시 KV Cache 사용 안 함
    model.config.use_cache = False

    # GPU 메모리 절약
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False })

    # 입력 임베딩의 Gradient 흐름 활성화
    model.enable_input_require_grads()

    # 기존 모델 가중치는 고정하고 LoRA만 학습
    lora_config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM")

    model = get_peft_model(model, lora_config)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Base Model:", MODEL_DIR.name)
    print("LoRA Rank:", LORA_R)

    model.print_trainable_parameters()

    return model, tokenizer

# endregion


# region Validation Loss 계산

def evaluate_loss(model, loader, device):
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.inference_mode():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss

            total_loss += loss.item()
            count += 1

    model.train()

    return total_loss / count

# endregion


# region LoRA 학습

def train_lora(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1. 데이터 로드
    train_records = load_records(TRAIN_PATH, args.limit)
    val_records = load_records(VAL_PATH, args.limit)

    print("Train Samples:", len(train_records))
    print("Validation Samples:", len(val_records))

    # 2. 모델 로드 및 LoRA 적용
    model, tokenizer = load_lora_model()

    train_dataset = QADataset(train_records, tokenizer, args.max_length)
    val_dataset = QADataset(val_records, tokenizer, args.max_length)

    # Batch Size 1 고정 (8GB GPU 기준)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    # 3. LoRA 가중치만 Optimizer에 등록
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    # 4. 결과 저장 경로
    smoke = args.limit > 0

    save_dir = (
        LORA_DIR.parent / (LORA_DIR.name + "-smoke") 
        if smoke 
        else LORA_DIR
    )

    metrics_path = RESULT_DIR / (
        "lora_train_smoke_metrics.json"
        if smoke
        else "lora_train_metrics.json"
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 5. 학습
    model.train()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0.0
        optimizer_steps = 0
        loader_length = len(train_loader)

        print(f"\n[Epoch {epoch}/{args.epochs}]")

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to("cuda")
                for key, value in batch.items()
            }

            # 마지막 누적 구간의 샘플 수도 반영
            window_start = (((step - 1) // args.grad_accum) * args.grad_accum)
            accumulation_count = min(args.grad_accum, loader_length - window_start)

            # Forward
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Loss가 유효하지 않습니다: {loss.item()}")

            total_train_loss += loss.item()

            # Backward
            scaled_loss = loss / accumulation_count
            scaler.scale(scaled_loss).backward()

            # Gradient Accumulation 이후 가중치 갱신
            should_update = (step % args.grad_accum == 0 or step == loader_length)

            if should_update:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

                if (optimizer_steps % 20 == 0 or step == loader_length):
                    print(
                        f"Step: {step}/{loader_length} "
                        f"| Loss: {loss.item():.4f}",
                        flush=True,
                    )

        # 6. Epoch 종료 후 Validation
        train_loss = total_train_loss / loader_length
        val_loss = evaluate_loss(model, val_loader, "cuda")
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "optimizer_steps": optimizer_steps,
        })

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Validation Loss가 가장 낮은 어댑터 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            model.save_pretrained(save_dir, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)

            print("Best LoRA Adapter: Saved")

    torch.cuda.synchronize()

    training_seconds = time.perf_counter() - start
    peak_gpu_memory_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3))

    # 7. 학습 결과 저장
    metrics = {
        "model": MODEL_DIR.name,
        "method": "LoRA",
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "grad_accum": args.grad_accum,
        "lora_rank": LORA_R,
        "best_val_loss": round(best_val_loss, 4),
        "training_seconds": round(training_seconds, 3),
        "peak_gpu_memory_gb": round(peak_gpu_memory_gb, 3),
        "history": history,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("\n[LoRA Training Result]")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Adapter Path:", save_dir)
    print("Metrics Path:", metrics_path)
    print("LoRA Training: OK")

# endregion



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum",type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (args.epochs < 1 or args.max_length < 256 or args.grad_accum < 1 or args.learning_rate <= 0 or args.limit < 0):
        parser.error("학습 설정값을 확인하세요.")

    train_lora(args)


if __name__ == "__main__":
    main()

