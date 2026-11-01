from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizer import CharTokenizer
from gpt_model import GPT


ROOT = Path(__file__).resolve().parent.parent

assert torch.cuda.is_available(), "CUDA GPU X"
device = torch.device("cuda")

checkpoint_path = ROOT / "checkpoints" / "latest.pt"

prompt = "ROMEO:"
max_new_tokens = 200

# Sampling 설정
temperature = 0.8
top_k = 20
top_p = 0.9

# 재현 가능한 결과
torch.manual_seed(42)

tokenizer = CharTokenizer.load(
    ROOT / "data" / "tokenizer" / "vocab.json"
)


checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

model = GPT(**checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, greedy=False):
    # Greedy Decoding
    if greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Temperature 검증 및 적용
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    logits = logits / temperature

     # Top-K Filtering
    if top_k is not None:

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        k = min(top_k, logits.size(-1))

        top_values, _ = torch.topk(logits, k, dim=-1)

        threshold = top_values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-P (Nucleus) Filtering
    if top_p is not None:

        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 누적 확률이 P를 넘는 후보 제거
        sorted_mask = cumulative_probs > top_p

        # 임계값을 넘긴 첫 토큰은 유지
        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[:, 0] = False

        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

        # 원래 토큰 ID 순서로 복원
        logits = torch.empty_like(logits).scatter(dim=-1, index=sorted_indices, src=sorted_logits)

    # Softmax로 확률 계산
    probabilities = F.softmax(logits, dim=-1)

    # 확률에 따라 무작위 토큰 선택
    next_token = torch.multinomial(probabilities, num_samples=1)

    return next_token


# 텍스트 생성
@torch.inference_mode()
def generate(model, x, max_new_tokens, temperature=1.0, top_k=None, top_p=None, greedy=False):
    for _ in range(max_new_tokens):

        # 최대 Context 길이 제한
        x_context = x[:, -model.max_seq_len:]

        # 순전파
        logits = model(x_context)

        # 마지막 위치의 Logits
        next_token_logits = logits[:, -1, :]

        # Sampling 알고리즘
        next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k, top_p=top_p, greedy=greedy)

        # 새로운 토큰 연결
        x = torch.cat([x, next_token], dim=1)

    return x

# 입력 준비
token_ids = tokenizer.encode(prompt)

x = torch.tensor([token_ids], dtype=torch.long, device=device)

# Sampling
generated_ids = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)

# Decode
generated_text = tokenizer.decode(generated_ids[0].tolist())


print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Checkpoint Step:", checkpoint["step"])

print("Temperature:", temperature)
print("Top-K:", top_k)
print("Top-P:", top_p)

print("\n========== GENERATED TEXT ==========")
print(generated_text)

print("\nSampling: OK")