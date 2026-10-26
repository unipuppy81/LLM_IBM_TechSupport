import torch
import torch.nn as nn

from embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock
from output_projection import OutputProjection

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, num_heads=4, num_layers=2, max_seq_len=64):
        super().__init__()

        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)

        # Positional Embedding
        self.position_embedding = PositionalEmbedding(max_seq_len, embedding_dim)

        # Transformer Block 여러 개 연결
        self.blocks = nn.ModuleList([TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)])

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Output Projection
        self.output_projection = OutputProjection(embedding_dim, vocab_size)

    def forward(self, x):
        # x: (Batch, Sequence)
        B, T = x.shape

        if T > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len")

        # Embedding
        hidden_states = (self.token_embedding(x) + self.position_embedding(x))

        # Transformer Block
        for block in self.blocks:
            hidden_states = block(hidden_states)

        # Final LayerNorm
        hidden_states = self.final_norm(hidden_states)

        # Output Projection
        logits = self.output_projection(hidden_states)

        return logits