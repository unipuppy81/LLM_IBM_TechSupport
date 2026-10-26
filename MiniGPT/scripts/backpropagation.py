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
tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")
token_ids = tokenizer.encode(text[:51])

# 입력 및 정답 생성
x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
targets = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

# 모델 구성
token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim).to(device)
position_embedding = PositionalEmbedding(max_seq_len, embedding_dim).to(device)
block = TransformerBlock(embedding_dim, num_heads).to(device)
final_norm = nn.LayerNorm(embedding_dim).to(device)
projection = OutputProjection(embedding_dim, tokenizer.vocab_size).to(device)
criterion = NextTokenLoss().to(device)


# 순전파
hidden_stats = (token_embedding(x) + position_embedding(x))
hidden_stats = block(hidden_stats)
hidden_stats = final_norm(hidden_stats)

logits = projection(hidden_stats)


# Loss 계산
loss = criterion(logits, targets)

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Loss:", loss.item())

# 역전파 전 가중치 저장
weight_before = projection.linear.weight.detach().clone()
print("Gradient Before:", projection.linear.weight.grad)

# 역전파
loss.backward()

# 주요 가중치 Gradient 확인
parameters = {
    "Token Embedding": token_embedding.embedding.weight,
    "Positional Embedding": position_embedding.embedding.weight,
    "Attention W_q": block.attention.W_q.weight,
    "Attention W_k": block.attention.W_k.weight,
    "Attention W_v": block.attention.W_v.weight,
    "Attention W_o": block.attention.W_o.weight,
    "FFN Linear 1": block.ffn.net[0].weight,
    "FFN Linear 2": block.ffn.net[2].weight,
    "LayerNorm 1": block.ln1.weight,
    "LayerNorm 2": block.ln2.weight,
    "Final LayerNorm": final_norm.weight,
    "Output Projection": projection.linear.weight,
}

for name, parameters, in parameters.items():
    # Gradient 계산 검증
    assert parameters.grad is not None, f"{name}: No Gradient"

    # NaN / Inf 검증
    assert torch.isfinite(parameters.grad).all()

    # Gradient 크기 확인
    grad_norm = parameters.grad.norm().item()
    print(f"{name}: Gradient Norm = {grad_norm:.6f}")


# 역전파 후 가중치 변경 여부 확인
weight_after = projection.linear.weight.detach()

assert torch.equal(weight_before, weight_after)

print("Weights Unchanged: OK")
print("Backpropagation: OK")