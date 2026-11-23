import argparse
import time
import json
import re

from pathlib import Path

import torch
import yaml
from load_base_model import load_model

from answerability_detection import detect_answerability, select_evidence_documents

PROJECT_DIR = Path(__file__).resolve().parents[1]
ABSTAIN = "INSUFFICIENT_INFORMATION"
TOP_K = 5

# region 검색 시스템 로드

def load_search_system():
    from dense_search import load_retriever
    from hybrid_search import load_bm25

    # Dense Retrieval: 임베딩 모델 + FAISS + 청크
    embedding_model, faiss_index, chunks = load_retriever()

    # BM25: 키워드 검색
    bm25_index = load_bm25(chunks)

    return embedding_model, faiss_index, bm25_index, chunks

def retrieve_all(questions, method, top_k, search_system=None):
    from dense_search import chunk_filename
    from hybrid_search import search_all, BM25_INDEX_DIR
    from reranker_search import load_reranker, rerank_candidates, get_chunk_text

    if method not in ("hybrid", "reranker"):
        raise ValueError("지원하지 않는 검색 방식입니다.")

    if search_system is None:
        if not (BM25_INDEX_DIR / "params.index.json").is_file():
            raise FileNotFoundError("Hybrid BM25 인덱스가 없습니다.")

        search_system = load_search_system()

    embedding_model, faiss_index, bm25_index, chunks = search_system
    reranker = load_reranker() if method == "reranker" else None
    start = time.perf_counter()
    results = search_all(embedding_model, faiss_index, bm25_index, questions)

    hybrid_rankings = [result[2] for result in results]

    if reranker is None:
        selected = [
            [
                {
                    "row_id": row_id,
                    "score": result[3][row_id],
                }
                for row_id in result[2][:top_k]
            ]
            for result in results
        ]
    else:
        reranked = rerank_candidates(reranker, questions, hybrid_rankings, chunks)
        selected = [items[:top_k] for items in reranked]

    documents_by_question = []

    for items in selected:
        documents = []

        for item in items:
            row_id = item["row_id"]
            chunk = chunks[row_id]

            documents.append({
                "row_id": row_id,
                "chunk_id": chunk.get("chunk_id"),
                "filename": chunk_filename(chunk),
                "text": get_chunk_text(chunk),
                "score": float(item["score"]),
            })

        documents_by_question.append(documents)

    elapsed = time.perf_counter() - start

    return documents_by_question, elapsed

# endregion


# region 관련 문서 검색
def retrieve_documents(question, embedding_model, faiss_index, bm25_index, chunks, method, top_k):
    search_system = (embedding_model, faiss_index, bm25_index, chunks)

    documents_by_question, _ = retrieve_all(
        questions=[question],
        method=method,
        top_k=top_k,
        search_system=search_system,
    )

    return documents_by_question[0]
# endregion


# region RAG 프롬프트 생성
def build_rag_prompt(question, documents, prompt_config):
    contexts = "\n\n".join(
        f"Document {index}\n"
        f"Source: {document['filename']}\n"
        f"Content:\n{document['text']}"
        for index, document in enumerate(documents, start=1)
    )

    user_prompt = prompt_config["user_template"].format(question=question, contexts=contexts)

    return [
        {
            "role": "system",
            "content": prompt_config["system"],
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

# endregion


# region Qwen RAG 답변 생성

def generate_rag_answer(model, tokenizer, generation_config, question, documents, prompt_config):

    # 1. 질문 + 검색 문서로 프롬프트 생성
    messages = build_rag_prompt(question, documents, prompt_config)

    # 2. Chat Template 적용 및 토큰화
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[1]

    # 3. 모델 추론
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(**inputs, generation_config=generation_config)

    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start

    # 4. 입력 프롬프트를 제외한 생성 토큰만 디코딩
    generated_tokens = outputs[0, input_tokens:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer, generation_seconds, input_tokens

# endregion

# region RAG 실행

def run_rag(question, method, top_k):
    config_path = PROJECT_DIR / "configs/rag/v1.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 검색 시스템 로드
    embedding_model, faiss_index, bm25_index, chunks = load_search_system()

    # 질문과 관련된 문서 검색
    search_start = time.perf_counter()
    documents = retrieve_documents(question, embedding_model, faiss_index, bm25_index, chunks, method, top_k)
    search_seconds = time.perf_counter() - search_start

    # 검색 결과 출력
    print("\n[Retrieved Documents]")

    for rank, document in enumerate(documents, start=1):
        print(f"\n[Rank {rank}]")
        print("Filename:", document["filename"])
        print("Chunk ID:", document["chunk_id"])
        print("Score:", round(document["score"], 4))
        print("Text:", document["text"][:300])

    # Qwen 모델 로드
    model, tokenizer, generation_config = load_model(config)

    # 문서별 근거 적합성 검사
    evidence_documents = select_evidence_documents(model, tokenizer, question, documents)

    # 선택된 문서들로 최종 답변 가능 여부 검사
    # answerable = bool(evidence_documents) and detect_answerability(model, tokenizer, question, evidence_documents)
    answerable = bool(evidence_documents)
    
    # 답변 가능 여부에 따라 처리
    if answerable:
        answer, generation_seconds, input_tokens = generate_rag_answer(model, tokenizer, generation_config, question, evidence_documents, config["prompt"])
    else:
        answer = ABSTAIN
        generation_seconds = 0.0
        input_tokens = 0

    print("\n[RAG Result]")
    print("Question:", question)
    print("Retriever:", method)
    print("Retrieved Documents:", len(documents))
    print("Input Tokens:", input_tokens)

    print("\nAnswer:")
    print(answer)

    print("\n[Performance]")
    print("Search Seconds:", round(search_seconds, 3))
    print("Generation Seconds:", round(generation_seconds, 3))
    print("Abstained:", answer == ABSTAIN)

    print("\nRAG Pipeline: OK")

# endregion



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--retriever", choices=("hybrid", "reranker"), default="reranker")
    parser.add_argument("--top-k", type=int, default=TOP_K)

    args = parser.parse_args()

    if not 1 <= args.top_k <= 50:
        parser.error("--top-k는 1~50 사이여야 합니다.")

    run_rag(question=args.question, method=args.retriever, top_k=args.top_k)


if __name__ == "__main__":
    main()

