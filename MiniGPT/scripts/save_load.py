from pathlib import Path
import shutil

import torch

from tokenizer import CharTokenizer
from gpt_model import GPT



ROOT = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = ROOT / "checkpoints" / "latest.pt"
vocab_path = ROOT / "data" / "tokenizer" / "vocab.json"

export_dir = ROOT / "export" / "minigpt"


# 모델 저장 함수
def save_model(model, config, step, export_dir):

    export_dir.mkdir(parents=True, exist_ok=True)
    model_path = export_dir / "model.pt"

    # 모델의 가중치, 구조 설정 및 학습 단계 저장
    torch.save({ "model_state_dict": model.state_dict(), "config": config, "step": step}, model_path)

    # Tokenizer 어휘도 함께 저장
    shutil.copy2(vocab_path, export_dir / "vocab.json")
    print("Model Saved:", model_path)
    return model_path


# 모델 로드 함수
def load_model(export_dir, device):

    model_path = export_dir / "model.pt"

    # 저장된 파일 읽기
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    config = checkpoint["config"]

    # 동일한 구조의 새 GPT 생성
    model = GPT(**config).to(device)

    # 학습된 가중치 복원
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 저장된 Tokenizer 로드
    tokenizer = CharTokenizer.load(export_dir / "vocab.json")

    return model, tokenizer, checkpoint["step"]


# 기존 학습 체크포인트 로드
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
config = checkpoint["config"]

original_model = GPT(**config).to(device)
original_model.load_state_dict(checkpoint["model_state_dict"])
original_model.eval()


# 모델 내보내기
save_model(original_model, config, checkpoint["step"], export_dir)

# 저장된 모델을 새 객체로 다시 로드
loaded_model, tokenizer, step = load_model(export_dir, device)


# 가중치 동일성 검증
original_state = original_model.state_dict()
loaded_state = loaded_model.state_dict()

for name in original_state:
    assert torch.equal(original_state[name], loaded_state[name]), f"Weight Mismatch: {name}"

print("Weights Match: OK")


# 동일한 입력으로 추론 검증
prompt = "ROMEO:"

token_ids = tokenizer.encode(prompt)
x = torch.tensor([token_ids], dtype=torch.long, device=device)

with torch.inference_mode():
    original_logits = original_model(x)
    loaded_logits = loaded_model(x)

# 출력 Logits 동일성 검증
assert torch.allclose(original_logits, loaded_logits, atol=1e-6, rtol=1e-5)

print("Inference Match: OK")

print("Device:", device)
print("Checkpoint Step:", step)
print("Model Parameters:", sum(p.numel() for p in loaded_model.parameters()))
print("Prompt:", repr(prompt))
print("Model Save / Load: OK")
