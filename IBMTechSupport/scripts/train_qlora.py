import argparse
import json
import random
import time

import bitsandbytes as bnb
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from evaluate_base_model import MODEL_DIR, RESULT_DIR
from train_lora import (
    QADataset,
    load_records,
    evaluate_loss,
    TRAIN_PATH,
    VAL_PATH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LEARNING_RATE,
)


# region QLoRA 설정

QLORA_DIR = MODEL_DIR.parent / "Qwen2.5-1.5B-Instruct-QLoRA"

QUANT_TYPE = "nf4"
COMPUTE_DTYPE = torch.float16

# endregion


# region 4비트 모델 로드 및 QLoRA 설정

def load_qlora_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    # 1. 기존 Qwen Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    # 2. 4비트 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )

    # 3. Base Model을 4비트로 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map={"": 0},
        local_files_only=True,
    )

    # 4. 실제 4비트 계층이 존재하는지 검증
    quantized_layers = sum(
        isinstance(module, bnb.nn.Linear4bit)
        for module in model.modules()
    )

    if quantized_layers == 0:
        raise RuntimeError("4비트 양자화가 적용되지 않았습니다.")

    # 학습 시 KV Cache 비활성화
    model.config.use_cache = False

    # 5. 양자화 모델을 LoRA 학습용으로 준비
    # Base 가중치 고정, Gradient Checkpointing 적용
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
    )

    # 6. 기존 LoRA와 같은 Rank / Alpha / 대상 계층
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 7. 4비트 Base Model에 LoRA 어댑터 적용
    model = get_peft_model(model, lora_config)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Base Model:", MODEL_DIR.name)
    print("Quantization:", "4-bit NF4")
    print("Double Quantization:", True)
    print("4-bit Layers:", quantized_layers)
    print("LoRA Rank:", LORA_R)

    model.print_trainable_parameters()

    return model, tokenizer

# endregion


# region QLoRA 학습

def train_qlora(args):
    # 1. 재현성을 위한 Seed 설정
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 2. 기존 LoRA와 동일한 데이터 사용
    train_records = load_records(TRAIN_PATH, args.limit)
    val_records = load_records(VAL_PATH, args.limit)

    print("Train Samples:", len(train_records))
    print("Validation Samples:", len(val_records))

    # 3. 4비트 QLoRA 모델 로드
    model, tokenizer = load_qlora_model()

    # 4. 기존 LoRA와 같은 전처리 방식
    train_dataset = QADataset(train_records, tokenizer, args.max_length)
    val_dataset = QADataset(val_records, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    # 5. 학습 가능한 LoRA 파라미터만 선택
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("학습 가능한 파라미터가 없습니다.")

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")

    # 6. 저장 경로
    smoke = args.limit > 0

    save_dir = (
        QLORA_DIR.parent / (QLORA_DIR.name + "-smoke")
        if smoke
        else QLORA_DIR
    )

    metrics_path = RESULT_DIR / (
        "qlora_train_smoke_metrics.json"
        if smoke
        else "qlora_train_metrics.json"
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 7. GPU 메모리 및 학습 시간 측정
    model.train()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()

    best_val_loss = float("inf")
    history = []

    # 8. Epoch 반복
    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0.0
        optimizer_steps = 0
        loader_length = len(train_loader)

        print(f"\n[Epoch {epoch}/{args.epochs}]")

        # 9. Batch 단위 학습
        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to("cuda")
                for key, value in batch.items()
            }

            window_start = (((step - 1) // args.grad_accum) * args.grad_accum)

            accumulation_count = min(args.grad_accum, loader_length - window_start)

            # Forward: 4비트 Base Model + LoRA
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Loss 오류: {loss.item()}")

            total_train_loss += loss.item()

            # Backward: LoRA 파라미터의 Gradient 계산
            scaled_loss = loss / accumulation_count
            scaler.scale(scaled_loss).backward()

            should_update = (step % args.grad_accum == 0 or step == loader_length)

            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)

                # LoRA 가중치 갱신
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

                if (optimizer_steps % 20 == 0 or step == loader_length):
                    print(
                        f"Step: {step}/{loader_length}"
                        f" | Loss: {loss.item():.4f}",
                        flush=True,
                    )

        # 10. Epoch 종료 후 Validation
        train_loss = total_train_loss / loader_length
        val_loss = evaluate_loss(model, val_loader, "cuda")
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "optimizer_steps": optimizer_steps,
        })

        print(
            f"Epoch {epoch}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
        )

        # 11. Validation Loss가 가장 낮으면 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # 양자화된 Base Model 전체가 아니라
            # LoRA 어댑터 가중치만 저장
            model.save_pretrained(save_dir, safe_serialization=True)
            tokenizer.save_pretrained(save_dir)
            print("Best QLoRA Adapter: Saved")

    # 12. 실행 성능 기록
    torch.cuda.synchronize()

    training_seconds = time.perf_counter() - start
    peak_gpu_memory_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3))

    # 13. 학습 지표 저장
    metrics = {
        "model": MODEL_DIR.name,
        "method": "QLoRA",
        "quantization": "4-bit NF4",
        "double_quantization": True,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "grad_accum": args.grad_accum,
        "lora_rank": LORA_R,
        "best_val_loss": round(best_val_loss, 4),
        "training_seconds": round(training_seconds, 3),
        "peak_gpu_memory_gb": round(
            peak_gpu_memory_gb, 3
        ),
        "history": history,
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("\n[QLoRA Training Result]")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Adapter Path:", save_dir)
    print("Metrics Path:", metrics_path)
    print("QLoRA Training: OK")

# endregion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if (args.epochs < 1 or args.max_length < 256 or args.grad_accum < 1 or args.learning_rate <= 0 or args.limit < 0):
        parser.error("학습 설정값을 확인하세요.")

    train_qlora(args)


if __name__ == "__main__":
    main()

