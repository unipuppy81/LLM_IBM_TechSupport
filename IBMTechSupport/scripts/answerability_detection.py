import re

import torch

# region 검색 문서별 근거 적합성 검사

def select_evidence_documents(model, tokenizer, question, documents):
    evidence_documents = []

    for rank, document in enumerate(documents, start=1):
        # 각 문서를 독립적으로 평가
        answerable = detect_answerability(model, tokenizer, question, [document])
        print(f"[Document {rank}] Answerable: {answerable}")

        if answerable:
            evidence_documents.append(document)

    return evidence_documents

# endregion

def detect_answerability(model, tokenizer, question, documents):
    # 1. 검색된 문서 없으면 답변 불가
    if not documents:
        return False

    # test
    # documents = documents[:1]

    # 2. 검색 문서를 하나의 Context로 구성
    contexts = "\n\n".join(
        f"[Document {i}]\n{doc['text']}"
        for i, doc in enumerate(documents, start=1)
    )

    system_prompt = (
        "You are an IBM technical support evidence evaluator. "
        "Determine whether the retrieved documents contain sufficient "
        "evidence to answer the user's specific question. "

        "Evaluate what the question actually asks. "
        "The answer may be a cause, solution, configuration, "
        "explanation, or document location. "

        "Similar products, errors, or symptoms alone are not sufficient. "
        "Check whether the evidence applies to the specific error, "
        "product, version, and conditions stated in the question. "

        "Do not assume that a solution for a similar problem "
        "also solves the user's problem. "

        "If the documents provide sufficient evidence, output YES. "
        "Otherwise, output NO. "

        "Output exactly YES or NO without explanation."

        "Evaluate each retrieved document independently. "
        "If at least one document directly and sufficiently answers "
        "the specific question, output YES, even if other documents "
        "are irrelevant. "
        "Do not combine evidence from unrelated problems. "

        "User input may be a question, an error message, or a problem description. "
        "Do not require an explicit question mark or a direct question. "
        
        "If a document describes the same specific error and environment "
        "and provides its cause, resolution, or confirms that the error "
        "can be safely ignored, consider the evidence sufficient. "
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved Documents:\n{contexts}"
            ),
        },
    ]

    # 3. Chat Template 적용
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    # 4. Qwen 모델로 답변 가능 여부 판별
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 5. 생성된 토큰을 문자열로 변환
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]

    # 6. 생성된 토큰을 문자열로 반환
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print("[Answerability Raw Response]", repr(response))
    match = re.match(r"^(YES|NO)\b", response, re.IGNORECASE)

    if not match:
        return False

    return match.group(1).upper() == "YES"