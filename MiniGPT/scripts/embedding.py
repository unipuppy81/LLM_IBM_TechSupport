from pathlib import Path

import torch
import torch.nn as nn

from tokenizer import CharTokenizer


# Token Embedding Class
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, x):
        # Token ID -> Embedding Vector
        return self.embedding(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    ROOT = Path(__file__).resolve().parent.parent

    # CUDA GPU Check
    assert torch.cuda.is_available(), "CUDA GPU 사용 불가"
    device = torch.device("cuda")

    embedding_dim = 16

    # Load Tokenizer
    tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
    vocab_size = len(tokenizer.stoi)

    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")
    sample = text[:50]

    # Text To Token ID
    token_ids = tokenizer.encode(sample)
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Create Token Embedding
    token_embedding = TokenEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)

    # Token ID to Embedding Vector
    output = token_embedding(x)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    print("Vocabulary Size:", vocab_size)
    print("Embedding Dimension:", embedding_dim)

    print("Original:", repr(sample))
    print("Token IDs:", token_ids)

    print("Input Shape:", x.shape)
    print("Embedding Shape:", output.shape)

    print("First Token:", sample[0])
    print("First Token ID:", token_ids[0])
    print("First Token Vector:", output[0, 0])

    print("Trainable Parameters:", sum(p.numel() for p in token_embedding.parameters()))


    # 검증
    assert output.shape == (1, len(token_ids), embedding_dim)
    assert output.is_cuda
    assert torch.allclose(output[0,0], token_embedding.embedding.weight[token_ids[0]])

    print("Token Embedding OK")
