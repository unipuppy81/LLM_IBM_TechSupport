import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Q, K, V Train Weights
        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Head 결합 후 출력 가중치
        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias=False)


    def forward(self, x):
        # B: Batch, T: Token Count, C: Embedding dimension
        B, T, C = x.shape

        # Create Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Multi-Head
        # (B, T, C) -> (B, H, T, D)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Score
        scores = Q @ K.transpose(-2, -1)

        # Scaling
        scores = scores / math.sqrt(self.head_dim)

        # Casual Mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

        # softmax
        weights = F.softmax(scores, dim=-1)

        # Attention Weight x V
        head_outputs = weights @ V

        # Head Concatenate
        # (B, H, T, D) -> (B, T, C)
        concatenated = head_outputs.transpose(1, 2).contiguous()
        concatenated = concatenated.view(B, T, C)

        # Output Projection
        output= self.W_o(concatenated)

        return output, weights


if __name__ == "__main__":
    from pathlib import Path
    from tokenizer import CharTokenizer
    from embedding import TokenEmbedding
    from positional_embedding import PositionalEmbedding

    torch.manual_seed(42)

    ROOT = Path(__file__).resolve().parent.parent

    assert torch.cuda.is_available(), "CUDA GPU를 사용할 수 없습니다."
    device = torch.device("cuda")

    embedding_dim = 16
    num_heads = 4
    max_seq_len = 64

    # Tokenizer
    tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text[:50])

    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Embedding
    token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim).to(device)
    position_embedding = PositionalEmbedding(max_seq_len, embedding_dim).to(device)
    input_vectors = (token_embedding(x) + position_embedding(x))

    # Multi-Head Attention
    attention = MultiHeadAttention(embedding_dim, num_heads).to(device)
    output, weights = attention(input_vectors)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Number of Heads:", num_heads)
    print("Head Dimension:", attention.head_dim)

    print("Input Shape:", input_vectors.shape)
    print("Attention Weights Shape:", weights.shape)
    print("Output Shape:", output.shape)

    print("Head 1 Attention Weights:")
    print(weights[0, 0].detach().cpu().round(decimals=3))

    # 검증
    assert output.shape == input_vectors.shape
    assert weights.shape == (1, 4, 50, 50)
    assert output.is_cuda

    # 미래 Token에 대한 Attention은 0이어야 함
    assert torch.all(torch.triu(weights, diagonal=1) == 0)

    # 각 Query의 Attention 확률 합은 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))

    print("Multi-Head Attention: OK")