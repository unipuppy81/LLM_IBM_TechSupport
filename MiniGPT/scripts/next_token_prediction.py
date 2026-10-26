
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import CharTokenizer
from embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding
from transformer_block import TransformerBlock
from output_projection import OutputProjection


# Next Token Prediction
class NextTokenPrediction(nn.Module):
    def forward(self, logits):

        # Logits -> 확률
        probabilities = F.softmax(logits, dim=-1)

        # 마지막 위치의 다음 토큰 확률
        next_probs = probabilities[:, -1, :]

        # 가장 확률이 높은 토큰 ID 선택
        next_token_id = torch.argmax(next_probs, dim=-1)

        return probabilities, next_probs, next_token_id


# 단독 실행 테스트
if __name__ == "__main__":

    torch.manual_seed(42)

    ROOT = Path(__file__).resolve().parent.parent

    assert torch.cuda.is_available(), "CUDA GPU 사용 못 함"
    device = torch.device("cuda")

    embedding_dim = 16
    num_heads = 4
    max_seq_len = 64


    # Tokenizer
    tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")
    vocab_size = tokenizer.vocab_size
    text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")

    sample = text[:50]
    token_ids = tokenizer.encode(sample)

    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    
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

    # Next Token Prediction
    predictor = NextTokenPrediction().to(device)

    # 전체 순전파
    with torch.no_grad():
        hidden_states = (token_embedding(x) + position_embedding(x))
        hidden_states = block(hidden_states)
        hidden_states = final_norm(hidden_states)

        logits = projection(hidden_states)

        probabilities, next_probs, next_token_id = (predictor(logits))


    # Token ID -> 문자
    predicted_id = next_token_id.item()
    predicted_char = tokenizer.decode([predicted_id])

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    print("Input:", repr(sample))

    print("Logits Shape:", logits.shape)
    print("Probabilities Shape:", probabilities.shape)
    print("Next Token Probabilities Shape:", next_probs.shape)

    print("Predicted Token ID:", predicted_id)
    print("Predicted Character:", repr(predicted_char))

    print("Prediction Probability:", next_probs[0, predicted_id].item())


    # 검증
    assert logits.shape == (1, 50, vocab_size)
    assert probabilities.shape == (1, 50, vocab_size)
    assert next_probs.shape == (1, vocab_size)

    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1)), atol=1e-6)

    assert 0 <= predicted_id < vocab_size
    assert next_probs.is_cuda

    print("Next Token Prediction: OK")
