from pathlib import Path

import torch
import torch.nn as nn

from tokenizer import CharTokenizer
from embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock
from output_projection import OutputProjection
from loss import NextTokenLoss


torch.manual_seed(42)
ROOT = Path(__file__).resolve().parent.parent

assert torch.cuda.is_available(), "CUDA GPU X"
device = torch.device("cuda")

embedding_dim = 16
num_heads = 4
max_seq_len = 64
learning_rate = 3e-4

tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")
token_ids = tokenizer.encode(text[:51])

x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
targets = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

# 모델 구성
token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim).to(device)
position_embedding = PositionalEmbedding(max_seq_len, embedding_dim).to(device)
block = TransformerBlock(embedding_dim, num_heads).to(device)
final_norm = nn.LayerNorm(embedding_dim).to(device)

projection = OutputProjection(embedding_dim, tokenizer.vocab_size).to(device)
criterion = NextTokenLoss().to(device)


# 학습 모듈 등록
model = nn.ModuleList([token_embedding, position_embedding, block, final_norm, projection])

# AdamW Optimizer 생성
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 가중치 업데이트 전 값 저장
weight_before = projection.linear.weight.detach().clone()

# 이전 gradient 초기화
optimizer.zero_grad(set_to_none=True)

# 순전파
hidden_states = (token_embedding(x) + position_embedding(x))

hidden_states = block(hidden_states)
hidden_states = final_norm(hidden_states)
logits = projection(hidden_states)

# Loss 계산
loss = criterion(logits, targets)

# Backpropagation
loss.backward()

# Gradient 계산 검증
assert projection.linear.weight.grad is not None
grad_norm = projection.linear.weight.grad.norm().item()

# Optimizer Step
# Gradient 이용하여 가중치 업데이트
optimizer.step()

weight_after = projection.linear.weight.detach().clone()
weight_changed = not torch.equal(weight_before, weight_after)

# 업데이트 후 Loss 계산
with torch.no_grad():

    hidden_states = (token_embedding(x) + position_embedding(x))
    hidden_states = block(hidden_states)
    hidden_states = final_norm(hidden_states)

    logits = projection(hidden_states)

    loss_after = criterion(logits, targets)


print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))

print("Optimizer:", type(optimizer).__name__)
print("Learning Rate:", learning_rate)

print("Loss Before:", loss.item())
print("Loss After:", loss_after.item())

print("Gradient Norm:", grad_norm)

print("Weight Before:", weight_before[0, :5])
print("Weight After:", weight_after[0, :5])

print("Weights Changed:", weight_changed)


assert weight_changed, "가중치가 변경되지 않았습니다."
assert torch.isfinite(loss_after)

print("Optimizer: OK")
