
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import CharTokenizer
from embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock
from output_projection import OutputProjection


# Cross Entropy Loss Class
class NextTokenLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):

        # Logits: (B, T, V)
        # Targets: (B, T)

        B, T, V = logits.shape

        # (B, T, V) -> (B*T, V)
        logits = logits.reshape(B * T, V)

        # (B, T) -> (B*T)
        targets = targets.reshape(B * T)

        # Cross Entropy Loss 계산
        loss = F.cross_entropy(logits, targets)

        return loss


if __name__ == "__main__":
    torch.manual_seed(42)
    ROOT = Path(__file__).resolve().parent.parent

    assert torch.cuda.is_available(), "CUDA GPU 사용 X"
    device = torch.device("cuda")

    embedding_dim = 16
    num_heads = 4
    max_seq_len = 64

    tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
    vocab_size = tokenizer.vocab_size

    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")

    # 입력 50개 + 마지막 정답 1개
    sample = text[:51]
    token_ids = tokenizer.encode(sample)


    # Input / Target 생성
    # 입력은 마지막 토큰 제외
    # 정답은 첫 번째 토큰 제외
    x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)


    # Token Embedding
    token_embedding = TokenEmbedding(vocab_size, embedding_dim).to(device)

    # Positional Embedding
    position_embedding = PositionalEmbedding(max_seq_len, embedding_dim).to(device)

    # Transformer Block
    block = TransformerBlock(embedding_dim, num_heads).to(device)

    # Final LayerNorm
    final_norm = nn.LayerNorm(embedding_dim).to(device)

    # Output Projection
    projection = OutputProjection(embedding_dim, vocab_size).to(device)

    # Loss 함수
    criterion = NextTokenLoss().to(device)


    # 전체 순전파
    hidden_states = (token_embedding(x) + position_embedding(x))
    hidden_states = block(hidden_states)
    hidden_states = final_norm(hidden_states)
    logits = projection(hidden_states)

    # Loss 계산
    loss = criterion(logits, targets)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    print("Input Shape:", x.shape)
    print("Target Shape:", targets.shape)
    print("Logits Shape:", logits.shape)

    print("Input Text:", repr(sample[:-1]))
    print("Target Text:", repr(sample[1:]))

    print("Loss:", loss.item())


    # 검증
    assert x.shape == (1, 50)
    assert targets.shape == (1, 50)
    assert logits.shape == (1, 50, vocab_size)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.is_cuda

    # 역전파를 위한 계산 그래프가 연결되어 있는지
    assert loss.requires_grad

    print("Cross Entropy Loss: OK")
