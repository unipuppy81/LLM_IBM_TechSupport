import math
from pathlib import Path

import torch

from tokenizer import CharTokenizer
from gpt_model import GPT
from loss import NextTokenLoss

ROOT = Path(__file__).resolve().parent.parent

assert torch.cuda.is_available(), "CUDA GPU X"
device = torch.device("cuda")

checkpoint_path = ROOT / "checkpoints" / "latest.pt"
batch_size = 32

tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")

# 학습된 체크포인트 로드
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

config = checkpoint["config"]
model = GPT(**config).to(device)

# 학습된 가중치 복원
model.load_state_dict(checkpoint["model_state_dict"])

# 평가 모드로 전환
model.eval()

criterion = NextTokenLoss()


# Batch 평가 함수
def score_batch(x, targets):
    x = x.to(device)
    targets = targets.to(device)

    # 순전파
    logits = model(x)
    loss = criterion(logits, targets)

    # Greedy 예측
    predictions = torch.argmax(logits, dim=-1)

    # 정답 수
    correct = (predictions == targets).sum().item()
    count = targets.numel()

    return loss.item() * count, correct, count



# validation / Test 평가
@torch.inference_mode()
def evaluate(split):
    # Data Load
    text = (ROOT / "data" / "processed" / f"{split}.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    data = torch.tensor(token_ids, dtype=torch.long)
    block_size = config["max_seq_len"]

    # 입력-정답 토큰 쌍 개수
    total_pairs = len(data) -1
    assert total_pairs > 0

    # 완전한 Block 수
    full_blocks = total_pairs // block_size
    full_tokens = full_blocks * block_size

    x_all = data[:full_tokens].reshape(full_blocks, block_size)
    y_all = data[1:full_tokens+1].reshape(full_blocks, block_size)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # Batch 단위 평가
    for start in range(0, full_blocks, batch_size):
        x = x_all[start:start + batch_size]
        targets = y_all[start:start + batch_size]

        loss_sum, correct, count = score_batch(x, targets)

        total_loss += loss_sum
        total_correct += correct
        total_count += count

    # 마지막 남은 토큰도 평가
    if full_tokens < total_pairs:
        x = data[full_tokens:-1].unsqueeze(0)
        targets = data[full_tokens + 1:].unsqueeze(0)

        loss_sum, correct, count = score_batch(x, targets)

        total_loss += loss_sum
        total_correct += correct
        total_count += count

    # 평가 지표 계산
    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)
    accuracy = total_correct / total_count * 100

    print(f"\n========== {split.upper()} ==========")
    print("Evaluated Tokens:", total_count)
    print("Loss:", round(avg_loss, 4))
    print("Perplexity:", round(perplexity, 4))
    print("Accuracy:", f"{accuracy:.2f}%")

    return avg_loss, perplexity, accuracy


print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Checkpoint Step:", checkpoint["step"])
print("Model Parameters:", sum(p.numel() for p in model.parameters()))

val_result = evaluate("val")
test_result = evaluate("test")

print("\nValidation / Test: OK")