from pathlib import Path

import torch
import torch.nn as nn

from tokenizer import CharTokenizer
from embedding import TokenEmbedding


# Positional Embedding Class
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super().__init__()
        
        self.max_seq_len = max_seq_len

        # 위치마다 학습 가능한 벡터 생성
        self.embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=embedding_dim)

    def forward(self, x):
        # x Shape: (Batch, Sequence)
        B, T = x.shape

        if T > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")

        # Create Position ID
        positions = torch.arange(T, device=x.device)

        # Position ID to Position Vector
        return self.embedding(positions)


if __name__ == "__main__":

    torch.manual_seed(42)

    ROOT = Path(__file__).resolve().parent.parent

    assert torch.cuda.is_available(), "CUDA GPU 사용할 수 없음"
    device = torch.device("cuda")

    embedding_dim = 16
    max_seq_len = 64

    tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")

    sample = text[:50]
    token_ids = tokenizer.encode(sample)

    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Create Token Embedding
    token_embedding = TokenEmbedding(vocab_size=tokenizer.vocab_size, embedding_dim=embedding_dim).to(device)
    token_vectors = token_embedding(x)

    # Create Positional Embedding
    positional_embedding = PositionalEmbedding(max_seq_len=max_seq_len, embedding_dim=embedding_dim).to(device)
    position_vectors = positional_embedding(x)

    # Token + Position Embedding
    # (1, 50, 16) + (50, 16)
    # Broadcasting을 통해 벡터가 배치에 적용
    output = token_vectors + position_vectors


    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    print("Token IDs:", token_ids)
    print("Position IDs:", list(range(x.shape[1])))

    print("Token Shape:", token_vectors.shape)
    print("Position Shape:", position_vectors.shape)
    print("Output Shape:", output.shape)

    print("First Token Vector:", token_vectors[0, 0])
    print("First Position Vector:", position_vectors[0])
    print("Combined Vector:", output[0, 0])


    # 검증
    assert token_vectors.shape == (1, 50, 16)
    assert position_vectors.shape == (50, 16)
    assert output.shape == (1, 50, 16)
    assert output.is_cuda

    assert torch.allclose(output[0, 0], token_vectors[0, 0] + position_vectors[0])

    print("Positional Embedding: OK")