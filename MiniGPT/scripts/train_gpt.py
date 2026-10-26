
from pathlib import Path

import torch

from tokenizer import CharTokenizer
from gpt_model import GPT
from loss import NextTokenLoss


# 기본 설정
torch.manual_seed(42)

ROOT = Path(__file__).resolve().parent.parent

assert torch.cuda.is_available(), "CUDA GPU X"
device = torch.device("cuda")

batch_size = 32
block_size = 64
max_steps = 1000
learning_rate = 3e-4

# GPT 모델 설정
config = {
    "embedding_dim": 16,
    "num_heads": 4,
    "num_layers": 2,
    "max_seq_len": block_size
}


# Tokenizer 로드
tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")

config["vocab_size"] = tokenizer.vocab_size


# Train 데이터 로드
text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")

token_ids = tokenizer.encode(text)

data = torch.tensor(token_ids, dtype=torch.long, device=device)

assert len(data) > block_size


# Mini-Batch 생성 함수
def get_batch():

    # 무작위 시작 위치 32개 선택
    starts = torch.randint(0, len(data) - block_size, (batch_size,), device=device)

    # 각 시작 위치에서 65개 연속 토큰 추출
    offsets = torch.arange(block_size + 1, device=device)

    sequences = data[starts[:, None] + offsets[None, :]]

    # 입력: 앞 64개
    x = sequences[:, :-1]

    # 정답: 한 칸 이동한 64개
    targets = sequences[:, 1:]

    return x, targets


# GPT 모델 생성
model = GPT(**config).to(device)

# Loss 함수
criterion = NextTokenLoss()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# 학습 정보 출력
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Train Tokens:", len(data))
print("Batch Size:", batch_size)
print("Block Size:", block_size)
print("Max Steps:", max_steps)

print("Total Parameters:", sum(p.numel() for p in model.parameters()))


# 학습 시작
model.train()
loss_history = []

for step in range(1, max_steps + 1):
    # Mini-Batch 추출
    x, targets = get_batch()

    # Gradient 초기화
    optimizer.zero_grad(set_to_none=True)

    # 순전파
    logits = model(x)

    # Loss 계산
    loss = criterion(logits, targets)

    if not torch.isfinite(loss):
        raise RuntimeError("Loss is NaN or Inf")

    # 역전파
    loss.backward()

    # Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 가중치 업데이트
    optimizer.step()
    loss_history.append(loss.item())

    # 학습 결과 출력
    if step == 1 or step % 100 == 0:
        avg_loss = sum(loss_history[-100:]) / len(loss_history[-100:])

        print(
            f"Step {step:4d}/{max_steps} | "
            f"Loss: {loss.item():.4f} | "
            f"Avg Loss: {avg_loss:.4f}"
            )


# 학습된 가중치 저장
# 16단계 Validation에서 사용할 체크포인트
save_path = ROOT / "checkpoints" / "latest.pt"
save_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
    "step": max_steps
}, save_path)


# 학습 완료
print("Training Complete!")
print("Checkpoint Saved:", save_path)
