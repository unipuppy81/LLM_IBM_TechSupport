import json
import torch
import yaml

from load_base_model import load_model
from .common import PROJECT_DIR


# region Answerability Judge

JUDGE_SYSTEM_PROMPT = (
    "You are a strict answerability judge for IBM technical support. "

    "Your task is to determine whether the retrieved documents "
    "contain sufficient evidence to answer the EXACT user question. "

    "Follow these rules: "
    "1. Identify what the user is specifically asking. "
    "2. Relevant keywords or similar problems do NOT prove answerability. "
    "3. If the question asks for a specific configuration or command, "
    "the documents must explicitly support that configuration or command. "
    "4. If the question asks WHY a problem occurred, "
    "instructions describing HOW to perform an operation are insufficient. "
    "5. Do not assume that a solution for a different incident "
    "applies to the user's environment. "
    "6. If the documents do not establish the requested answer, choose NO. "
    "7. Use only the retrieved documents, not your own knowledge. "

    "Output exactly one word: YES or NO. "
    "Do not provide explanations, document names, or other text."
)

def judge_answerability(model, tokenizer, generation_config, question, documents):

    # 1. 검색 문서 5개 구성
    contexts = []

    for rank, doc in enumerate(documents, start=1):
        contexts.append(
            f"[Document {rank}]\n"
            f"Filename: {doc['filename']}\n"
            f"Content:\n{doc['text']}"
        )

    # 2. Judge 입력 구성
    messages = [
        {
            "role": "system",
            "content": JUDGE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved Documents:\n{chr(10).join(contexts)}\n\n"
                "Is there sufficient evidence to answer the question?"
            ),
        },
    ]

    # 3. 토큰화
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    # 4. Judge 추론
    # 기존 generation_config를 변경하지 않도록 복사
    from copy import deepcopy

    judge_config = deepcopy(generation_config)
    judge_config.do_sample = False
    judge_config.num_beams = 1
    judge_config.max_new_tokens = 12

    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=judge_config)

    # 5. 원본 출력 보존
    raw_output = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

    # 6. Judge 출력 검증
    try:
        decision = raw_output.strip()

        if decision not in ("YES", "NO"):
            raise ValueError(f"올바르지 않은 Judge 출력: {decision}")

        result = { "answerable": decision == "YES" }

        return raw_output, result, None

    except ValueError as error:
        return raw_output, None, str(error)


def test_judge_cases(case_ids):

    # 기존 Hybrid RAG 평가 결과 사용
    path = PROJECT_DIR / "evaluate_results" / "rag" / "v1" / "predictions.jsonl"

    with open(path, encoding="utf-8") as f:
        rows = {
            row["id"]: row
            for line in f
            if (row := json.loads(line.strip()))
        }

    # 모델은 한 번만 로드
    config = yaml.safe_load(
    (PROJECT_DIR / "configs/rag/v1.yaml").read_text(encoding="utf-8"))
    model, tokenizer, generation_config = load_model(config)

    for case_id in case_ids:
        row = rows[case_id]
        documents = row["retrieved_documents"]

        assert len(documents) == 5

        raw, result, error = judge_answerability(
            model,
            tokenizer,
            generation_config,
            row["question"],
            documents,
        )

        print("\n" + "=" * 60)
        print("ID:", case_id)
        print("Original is_impossible:", row["is_impossible"])
        print("Top-K:", len(documents))
        print("\n[RAW JUDGE OUTPUT]")
        print(raw)

        print("\n[PARSED JUDGE RESULT]")
        print(result)

        if error:
            print("Parse Error:", error)

        if result is not None:
            for rank in result.get("evidence_ranks", []):
                print("Evidence:", documents[rank - 1]["chunk_id"])

# endregion
