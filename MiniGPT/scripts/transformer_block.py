
from pathlib import Path

import torch
import torch.nn as nn

from tokenizer import CharTokenizer
from embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

# Transformer Block Class
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        # Embedding 차원이 Head 로 나눠지는지 확인
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads


        # Attention 이전 LayerNorm
        self.ln1 = nn.LayerNorm(embedding_dim)

        # Multi-Head Attention
        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        # FFN 이전 LayerNorm
        self.ln2 = nn.LayerNorm(embedding_dim)

        # Feed Forward Network
        self.ffn = FeedForward(embedding_dim)

    def forward(self, x):
        # LayerNorm -> Attention -> Residual
        normalized_x = self.ln1(x)
        attn_output, _ = self.attention(normalized_x)

        # 기존 입력 + Attention 결과
        x = x + attn_output

        # LayerNorm -> FFN -> Residual
        normalized_x = self.ln2(x)
        ffn_output = self.ffn(normalized_x)

        # 기존 입력 + FFN 결과
        x = x + ffn_output
        return x

if __name__ == "__main__":
    torch.manual_seed(42)

    ROOT = Path(__file__).resolve().parent.parent

    assert torch.cuda.is_available(), "CUDA GPU 사용할 수 없음"
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

    # Create Transformer Block
    block = TransformerBlock(embedding_dim=embedding_dim, num_heads=num_heads).to(device)

    # Forward
    output = block(input_vectors)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    print("Embedding Dimension:", embedding_dim)
    print("Number of Heads:", num_heads)

    print("Input Shape:", input_vectors.shape)
    print("Output Shape:", output.shape)


    # 검증
    assert output.shape == input_vectors.shape
    assert output.is_cuda

    # 실제 Transformer 연산으로 출력이 달라졌는지 확인
    assert not torch.allclose(output, input_vectors)

    # LayerNorm 파라미터 확인
    assert block.ln1.weight.shape == (embedding_dim,)
    assert block.ln2.weight.shape == (embedding_dim,)

    # Transformer Block 전체 파라미터
    total_params = sum(p.numel() for p in block.parameters())

    print("LayerNorm 1:", block.ln1)
    print("LayerNorm 2:", block.ln2)
    print("Trainable Parameters:", total_params)

    print("Residual Connection / LayerNorm: OK")
