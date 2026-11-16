from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# 프로젝트 경로
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "Qwen2.5-1.5B-Instruct"

def load_model(config):
    model_config = config["model"]
    inference_config = config["inference"]
    model_path = PROJECT_DIR / model_config["path"]
    dtype = getattr(torch, model_config["dtype"])

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, local_files_only=True).to("cuda")
    model.eval()
    generation_config = GenerationConfig(
        max_new_tokens=inference_config["max_new_tokens"],
        do_sample=inference_config["do_sample"],
        num_beams=inference_config["num_beams"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return model, tokenizer, generation_config

def main():
    # GPU 확인
    assert torch.cuda.is_available(), "CUDA GPU를 사용할 수 없습니다."

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)

    # 다운로드한 모델 폴더 확인
    assert (MODEL_DIR / "config.json").exists(), ("Model configuration not found")

    # 1. 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    print("Tokenizer: OK")
    print("Vocabulary Size:", len(tokenizer))

    # 2. Qwen 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    model.eval()

    # GPU에 모델이 배치되었는지 확인
    devices = {p.device.type for p in model.parameters()}
    assert devices == {"cuda"}, f"Unexpected devices: {devices}"

    print("Model: Qwen2.5-1.5B-Instruct")
    print("Model Device:", model.device)
    print("Model Parameters:", sum(p.numel() for p in model.parameters()))

    # 3. 테스트 질문
    messages = [
        {
            "role": "system",
            "content": (
                "You are an IBM technical support assistant. "
                "Give concise, factual answers."
            )
        },
        {
            "role": "user",
            "content": "What is IBM WebSphere Application Server?"
        }
    ]

    # 4. Chat Template 적용 및 토큰화
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    print("Input Tokens:", inputs["input_ids"].shape[-1])

    # 5. 추론
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 6. 새로 생성한 토큰만 디코딩
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n[Base Model Output]")
    print("Question:", messages[-1]["content"])
    print("Answer:", answer)

    assert answer, "Model generated an empty response"

    # GPU 메모리 확인
    used_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak GPU Memory: {used_memory:.2f} GB")
    print("Base Model Load: OK")


if __name__ == "__main__":
    main()
