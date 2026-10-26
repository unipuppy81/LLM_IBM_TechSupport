
from pathlib import Path

import torch
import torch.nn as nn

from tokenizer import CharTokenizer
from embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock


# Output Projection Class
class OutputProjection(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

        # Embedding Dimension -> Vocabulary Size
        self.linear = nn.Linear(in_features=embedding_dim, out_features=vocab_size, bias=False)

    def forward(self, x):
        # (B, T, 16) -> (B, T, 65)
        return self.linear(x)


if __name__ == "__main__":

    torch.manual_seed(42)

    ROOT = Path(__file__).resolve().parent.parent

    assert torch.cuda.is_available(), "CUDA GPU를 사용할 수 없습니다."
    device = torch.device("cuda")

    embedding_dim = 16
    num_heads = 4
    max_seq_len = 64

    # Tokenizer
    tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
    vocab_size = tokenizer.vocab_size
    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")

    token_ids = tokenizer.encode(text[:50])
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Token Embedding
    token_embedding = TokenEmbedding(vocab_size, embedding_dim).to(device)

    # Positional Embedding
    position_embedding = PositionalEmbedding( max_seq_len, embedding_dim).to(device)
    input_vectors = (token_embedding(x) + position_embedding(x))

    # Transformer Block
    block = TransformerBlock(embedding_dim, num_heads).to(device)
    hidden_states = block(input_vectors)

    # Final LayerNorm
    final_norm = nn.LayerNorm(embedding_dim).to(device)
    hidden_states = final_norm(hidden_states)

    # Output Projection
    projection = OutputProjection(embedding_dim, vocab_size).to(device)

    logits = projection(hidden_states)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    print("Vocabulary Size:", vocab_size)
    print("Embedding Dimension:", embedding_dim)

    print("Input Shape:", input_vectors.shape)
    print("Transformer Output:", hidden_states.shape)
    print("Logits Shape:", logits.shape)

    print("First Token Logits:", logits[0, 0])

    print("Projection Parameters:", sum(p.numel() for p in projection.parameters()))

    # 검증
    assert hidden_states.shape == (1, 50, 16)
    assert logits.shape == (1, 50, vocab_size)
    assert logits.is_cuda

    assert projection.linear.weight.shape == (vocab_size, embedding_dim)

    print("Output Projection: OK")
