from pathlib import Path

import torch

from tokenizer import CharTokenizer
from gpt_model import GPT

ROOT = Path(__file__).resolve().parent.parent

assert torch.cuda.is_available(), "CUDA GPU X"
device = torch.device("cuda")

checkpoint_path = ROOT / "checkpoints" / "latest.pt"

# 시작 문장
prompt = "ROMEO:"

# 새로 생성할 문자 토큰 개수
max_new_tokens = 200

tokenizer = CharTokenizer.load(ROOT / "data" / "tokenizer" / "vocab.json")

# 학습된 모델 로드
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

config = checkpoint["config"]
model = GPT(**config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# 평가 모드 (학습 비활성화)
model.eval()

# 사용자 입력을 Token ID로 변환
token_ids = tokenizer.encode(prompt)
x = torch.tensor([token_ids], dtype=torch.long, device=device)

# 다음 토큰 반복 생성
@torch.inference_mode()
def generate(model, x, max_new_tokens):

    for _ in range(max_new_tokens):
        # 모델이 처리 가능한 최대 문맥 길이 유지
        x_context = x[:, -model.max_seq_len:]

        # 순전파: (B, T) -> (B, T, V)
        logits = model(x_context)

        # 마지막 위치의 다음 토큰 예측 점수
        next_token_logits = logits[:, -1, :]

        # Greedy
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 선택한 토큰을 입력 뒤에 추가
        x = torch.cat([x, next_token], dim=1)

    return x


# 추론
generated_ids = generate(model, x, max_new_tokens)

# token id covert to string
generated_text = tokenizer.decode(generated_ids[0].tolist())

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0))
print("Checkpoint Step:", checkpoint["step"])
print("Prompt:", repr(prompt))
print("Generated Tokens:", max_new_tokens)

print("\n========== GENERATED TEXT ==========")
print(generated_text)

print("\nInference: OK")
