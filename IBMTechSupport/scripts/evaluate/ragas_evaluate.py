
import json
from pathlib import Path

import asyncio

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall



PROJECT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_DIR / "evaluate_results" / "rag" / "v1" / "predictions.jsonl"


# region 기존 RAG 평가 결과 불러오기

def load_evaluation_data(question_id):

    with PREDICTIONS_PATH.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["id"] != question_id:
                continue

            # 기존 RAG가 생성한 답변
            answer = record["prediction"]

            # 기존 RAG가 검색했던 문서 내용
            contexts = [
                doc["text"]
                for doc in record["retrieved_documents"]
            ]

            return record["question"], answer, contexts, record.get("reference_answer"), record.get("is_impossible", False)

    raise ValueError(f"질문을 찾을 수 없습니다: {question_id}")

# endregion


# region Faithfulness 평가

def run_evaluation(question_id="TRAIN_Q169"):

    # 1. 질문, 생성 답변, 검색 문서 로드
    question, answer, contexts, reference, is_impossible = load_evaluation_data(question_id)

    # 답변 거부 사례는 충실도 평가에서 제외
    if answer.strip() == "INSUFFICIENT_INFORMATION":
        print("Faithfulness: N/A (Abstained)")
        return

    # 2. 로컬 평가 모델 연결
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    evaluator_llm = llm_factory("qwen2.5:7b", provider="openai", client=client)

    # 3. Faithfulness 평가기 생성
    scorer = Faithfulness(llm=evaluator_llm)

    # Answer Relevancy에 사용할 로컬 임베딩 모델
    embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # Answer Relevancy
    relevancy_scorer = AnswerRelevancy(llm=evaluator_llm, embeddings=embeddings)
    # Context Precision
    precision_scorer = ContextPrecision(llm=evaluator_llm)
    # Context Recall
    recall_scorer = ContextRecall(llm=evaluator_llm)

    # 평가 결과를 한국어로 번역 (Ollama 로컬 모델)
    async def translate_to_korean(text):
        response = await client.chat.completions.create(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the following text into Korean. "
                        "Do not explain or summarize. "
                        "Preserve technical terms, file paths, "
                        "commands, and numbers exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()


    # 4. 주장 분해 + 근거 판별 + 점수 계산

    async def evaluate_details():
        # 1. 생성된 답변을 개별 주장으로 분해
        statements = await scorer._create_statements(question, answer)

        print("\n[1. 주장 분해]")

        for index, statement in enumerate(statements, 1):
            print(f"{index}. {statement}")

        if not statements:
            print("추출된 주장이 없습니다.")
            return (float("nan"), None, None, None)

        # 2. 검색 문서를 하나의 컨텍스트로 결합
        context = "\n".join(contexts)
        # context = contexts[0]

        # 3. 각 주장이 컨텍스트에 근거하는지 판정
        verdicts = await scorer._create_verdicts(statements, context)

        print("\n[디버깅]")
        print("Statements: ", statements)
        print("Verdicts: ", verdicts)
        print("Context Length: ", len(context))



        print("\n[2. 주장별 판정]")

        for index, item in enumerate(verdicts.statements, 1):
            # 영어 평가 결과를 한국어로 번역
            statement_ko = await translate_to_korean(item.statement)
            reason_ko = await translate_to_korean(item.reason)

            print(f"\n주장 {index}: {statement_ko}")
            print(f"판정: {item.verdict}")
            print(f"이유: {reason_ko}")

        
        # 1. Faithfulness 점수
        faithfulness_score = scorer._compute_score(verdicts)

        # 2. Answer Relevancy
        # 질문과 생성 답변의 관련성을 평가
        relevancy_result = await relevancy_scorer.ascore(user_input=question, response=answer)

        # 3, 4번은 모범 답변이 필요한 지표
        precision_score = None
        recall_score = None

        has_reference = (
            is_impossible is False
            and isinstance(reference, str)
            and bool(reference.strip())
            and reference.strip() != "INSUFFICIENT_INFORMATION"
        )

        if has_reference:
            # 3. Context Precision
            # 관련 문서가 상위에 검색되었는지 평가
            precision_result = await precision_scorer.ascore(user_input=question, reference=reference, retrieved_contexts=contexts)
            precision_score = precision_result.value

            # 4. Context Recall
            # 정답에 필요한 근거가 검색되었는지 평가
            recall_result = await recall_scorer.ascore(user_input=question, reference=reference, retrieved_contexts=contexts)
            recall_score = recall_result.value

        return faithfulness_score, relevancy_result.value, precision_score, recall_score

    async def evaluate_try_finally():
        try:
            return await evaluate_details()
        finally:
            await client.close()

    scores = asyncio.run(evaluate_details())
    faithfulness, relevancy, precision, recall = scores

    def format_score(value):
        return "N/A" if value is None else f"{value:.4f}"

    print("\n[RAGAS Evaluation]")
    print("Question ID:", question_id)
    print("Retrieved Documents:", len(contexts))
    print("Faithfulness:", format_score(faithfulness))
    print("Answer Relevancy:", format_score(relevancy))
    print("Context Precision:", format_score(precision))
    print("Context Recall:", format_score(recall))

# endregion


if __name__ == "__main__":
    run_evaluation()
