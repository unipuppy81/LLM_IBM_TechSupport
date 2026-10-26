import torch
import torch.nn as nn

from pathlib import Path
from tokenizer import CharTokenizer
from embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding


# Feed Forward Network Class
class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        hidden_dim = embedding_dim * 4

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), # 16 -> 64
            nn.GELU(), # 비선형 활성화 함수
            nn.Linear(hidden_dim, embedding_dim) # 64 -> 16
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":

    torch.manual_seed(42)

    ROOT = Path(__file__).resolve().parent.parent

    assert torch.cuda.is_available(), "CUDA GPU를 사용할 수 없습니다."
    device = torch.device("cuda")

    embedding_dim = 16
    max_seq_len = 64

    tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")

    # 학습 데이터 불러오기
    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")

    token_ids = tokenizer.encode(text[:50])
    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Embedding 생성
    token_embedding = TokenEmbedding(tokenizer.vocab_size, embedding_dim).to(device)
    position_embedding = PositionalEmbedding(max_seq_len, embedding_dim).to(device)
    input_vectors = (token_embedding(x) + position_embedding(x))

    # Feed Forward Network 생성
    ffn = FeedForward(embedding_dim=embedding_dim).to(device)

    # Forward
    output = ffn(input_vectors)

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    print("Embedding Dimension:", embedding_dim)
    print("Hidden Dimension:", embedding_dim * 4)

    print("Input Shape:", input_vectors.shape)
    print("Output Shape:", output.shape)

    print("First Input Vector:", input_vectors[0, 0])
    print("First Output Vector:", output[0, 0])

    print("Trainable Parameters:", sum(p.numel() for p in ffn.parameters()))


    assert output.shape == input_vectors.shape
    assert output.is_cuda

    assert not torch.allclose(output, input_vectors)

    print("Feed Forward Network: OK")