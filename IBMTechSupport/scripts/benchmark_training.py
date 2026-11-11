import argparse
import json
import random
import time

import torch

from evaluate_base_model import RESULT_DIR
from train_lora import (
    LEARNING_RATE,
    TRAIN_PATH,
    QADataset,
    load_lora_model,
    load_records,
)


# region 벤치마크 설정

def result_path(model_type, optimizer_type, checkpoint):
    name = (
        f"train_benchmark_{model_type}_"
        f"{optimizer_type}_ckpt_{checkpoint}.json"
    )
    return RESULT_DIR / name

# endregion


# region 모델 및 Optimizer 준비

def load_training_model(model_type, checkpoint):
    if model_type == "lora":
        model, tokenizer = load_lora_model()
    else:
        from train_qlora import load_qlora_model
        model, tokenizer = load_qlora_model()

    # 기존 학습에서는 Gradient Checkpointing이 켜져 있음
    if checkpoint == "off":
        model.gradient_checkpointing_disable()

    model.config.use_cache = False
    model.train()

    return model, tokenizer


def make_optimizer(model, optimizer_type):
    parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    options = {"lr": LEARNING_RATE}

    # Fused AdamW 활성화
    if optimizer_type == "fused":
        options["fused"] = True

    optimizer = torch.optim.AdamW(
        parameters,
        **options,
    )

    return optimizer, parameters

# endregion


# region 실제 학습 구간 실행

def train_segment(
    dataset,
    start,
    end,
    model,
    optimizer,
    parameters,
    scaler,
    accum,
):
    total = end - start
    total_loss = 0.0
    updates = 0

    optimizer.zero_grad(set_to_none=True)

    for step, index in enumerate(
        range(start, end),
        start=1,
    ):
        batch = {
            key: value.to("cuda")
            for key, value in dataset[index].items()
        }

        # Forward: FP16 Mixed Precision
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            loss = model(**batch).loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"유효하지 않은 Loss: {loss.item()}"
            )

        total_loss += loss.item()

        # Gradient Accumulation
        window_start = (
            ((step - 1) // accum) * accum
        )

        current_window = min(
            accum,
            total - window_start,
        )

        # Backward
        scaler.scale(
            loss / current_window
        ).backward()

        # Weight Update
        if step % accum == 0 or step == total:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            updates += 1

    return total_loss / total, updates

# endregion


# region 04. 학습 속도 및 GPU 메모리 측정

def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 1. 동일한 Train 데이터 사용
    records = load_records(TRAIN_PATH)

    required = (
        args.warmup_samples
        + args.timed_samples
    )

    if required > len(records):
        raise ValueError(
            f"필요 샘플 수 {required} > "
            f"Train 수 {len(records)}"
        )

    random.Random(args.seed).shuffle(records)
    records = records[:required]

    # 2. 모델 및 Dataset 준비
    model, tokenizer = load_training_model(
        args.model,
        args.checkpoint,
    )

    dataset = QADataset(
        records,
        tokenizer,
        args.max_length,
    )

    optimizer, parameters = make_optimizer(
        model,
        args.optimizer,
    )

    scaler = torch.amp.GradScaler("cuda")

    print("Model:", args.model)
    print("Optimizer:", args.optimizer)
    print("Gradient Checkpointing:", args.checkpoint)
    print("Warmup Samples:", args.warmup_samples)
    print("Measured Samples:", args.timed_samples)

    # 3. 워밍업: 초기 CUDA 커널 및 옵티마이저 상태 준비
    train_segment(
        dataset,
        0,
        args.warmup_samples,
        model,
        optimizer,
        parameters,
        scaler,
        args.grad_accum,
    )

    # 4. 측정 시작
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    loss, updates = train_segment(
        dataset,
        args.warmup_samples,
        required,
        model,
        optimizer,
        parameters,
        scaler,
        args.grad_accum,
    )

    torch.cuda.synchronize()

    seconds = time.perf_counter() - start

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    # 5. 결과 정리
    result = {
        "model": args.model,
        "optimizer": args.optimizer,
        "checkpoint": args.checkpoint,
        "warmup_samples": args.warmup_samples,
        "timed_samples": args.timed_samples,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "optimizer_steps": updates,
        "mean_loss": round(loss, 4),
        "seconds": round(seconds, 3),
        "samples_per_second": round(
            args.timed_samples / seconds, 3
        ),
        "peak_gpu_memory_gb": round(
            peak_memory, 3
        ),
    }

    # 6. 결과 저장
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = result_path(
        args.model,
        args.optimizer,
        args.checkpoint,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[Training Optimization Benchmark]")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("Metrics:", path)
    print("Benchmark: OK")

# endregion


# region 벤치마크 결과 비교

def compare():
    print("\n[Training Optimization Comparison]")

    print(
        f"{'Model':<8} "
        f"{'Checkpoint':<11} "
        f"{'Optimizer':<10} "
        f"{'Seconds':>9} "
        f"{'Samples/s':>11} "
        f"{'GPU GB':>9}"
    )

    configurations = set()
    found = 0

    for model_type in ("lora", "qlora"):
        for checkpoint in ("on", "off"):
            for optimizer_type in ("default", "fused"):
                path = result_path(
                    model_type,
                    optimizer_type,
                    checkpoint,
                )

                if not path.is_file():
                    continue

                with path.open(
                    encoding="utf-8",
                ) as file:
                    metrics = json.load(file)

                configurations.add((
                    metrics["warmup_samples"],
                    metrics["timed_samples"],
                    metrics["grad_accum"],
                    metrics["max_length"],
                ))

                found += 1

                print(
                    f"{model_type:<8} "
                    f"{checkpoint:<11} "
                    f"{optimizer_type:<10} "
                    f"{metrics['seconds']:>9.3f} "
                    f"{metrics['samples_per_second']:>11.3f} "
                    f"{metrics['peak_gpu_memory_gb']:>9.3f}"
                )

    if found == 0:
        print("평가 결과 파일이 없습니다.")

    if len(configurations) > 1:
        print(
            "주의: 실험 설정이 서로 달라 "
            "직접 비교할 수 없습니다."
        )

# endregion

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=("lora", "qlora"),
        default="lora",
    )

    parser.add_argument(
        "--optimizer",
        choices=("default", "fused"),
        default="default",
    )

    parser.add_argument(
        "--checkpoint",
        choices=("on", "off"),
        default="on",
    )

    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--timed-samples",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--compare",
        action="store_true",
    )

    args = parser.parse_args()

    if args.compare:
        compare()
        return

    if (
        args.warmup_samples < 1
        or args.timed_samples < 1
        or args.grad_accum < 1
        or args.max_length < 256
    ):
        parser.error("학습 설정값을 확인하세요.")

    benchmark(args)


if __name__ == "__main__":
    main()


