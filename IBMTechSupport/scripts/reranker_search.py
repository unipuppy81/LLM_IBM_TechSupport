import argparse
import json
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from dense_search import chunk_filename, load_retriever
from hybrid_search import load_bm25, search_all


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
CANDIDATE_K = 50
TOP_K = 5
BATCH_SIZE = 16

# region Reranker 모델 로드

def load_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CrossEncoder(
        RERANKER_MODEL,
        device=device,
        max_length=512, 
    )

    print("Reranker Device:", device)
    print("Reranker Model:", RERANKER_MODEL)
    print("Reranker Load: OK")

    return model

# endregion


# region 청크 텍스트 추출

def get_chunk_text(chunk):
    text = (chunk.get("text") or chunk.get("content") or chunk.get("chunk_text"))

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"청크 텍스트가 없습니다: {list(chunk.keys())}")

    return text

# endregion


# region Reranking

def rerank_candidates(reranker, questions, hybrid_rankings, chunks, batch_size=BATCH_SIZE,):
    """
    1. Hybrid 상위 50개 청크 선택
    2. (질문, 청크 텍스트) 쌍 생성
    3. Cross-Encoder 관련성 점수 계산
    4. 점수 기준 내림차순 재정렬
    """

    if len(questions) != len(hybrid_rankings):
        raise ValueError("질문 수와 검색 결과 수가 다릅니다.")

    pairs = []
    positions = []

    for question_index, (question, ranking) in enumerate(zip(questions, hybrid_rankings)):
        for original_rank, row_id in enumerate(ranking[:CANDIDATE_K]):
            pairs.append((question, get_chunk_text(chunks[row_id]),))
            positions.append((question_index, row_id, original_rank,))

    if not pairs:
        raise ValueError("재정렬할 검색 결과가 없습니다.")

    # 질문-문서 쌍을 Batch 단위로 추론
    scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,)
    scores = np.asarray(scores).reshape(-1)

    if len(scores) != len(positions):
        raise RuntimeError("Reranker 점수 개수가 일치하지 않습니다.")

    if not np.isfinite(scores).all():
        raise ValueError("Reranker 점수에 NaN/Inf가 있습니다.")

    grouped = [[] for _ in questions]

    for (question_index, row_id, original_rank), score in zip(positions, scores):
        grouped[question_index].append(
            {
                "row_id": row_id,
                "score": float(score),
                "original_rank": original_rank,
            }
        )

    # 관련성 점수 높은 순으로 재정렬
    for results in grouped:
        results.sort(key=lambda item: (-item["score"], item["original_rank"],))

    return grouped

# endregion

# region 질문 검색 테스트

def search_question(model, faiss_index, bm25_index, reranker, chunks, question, top_k, batch_size, ):
    # Hybrid Search 결과 가져오기
    search_results = search_all(model, faiss_index, bm25_index, [question], )

    hybrid_rankings = [search_results[0][2]]

    # Reranker 적용
    reranked = rerank_candidates(reranker, [question], hybrid_rankings, chunks, batch_size, )[0]

    for rank, result in enumerate(reranked[:top_k], 1):
        row_id = result["row_id"]
        chunk = chunks[row_id]

        print(f"\n[Rank {rank}]")
        print("Filename:", chunk_filename(chunk))
        print("Reranker Score:", round(result["score"], 4))
        print("Original Hybrid Rank:", result["original_rank"] + 1)
        print("Text:", get_chunk_text(chunk)[:400])

    print("\nReranker Search: OK")

# endregion

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    if not 1 <= args.top_k <= CANDIDATE_K:
        parser.error(f"--top-k는 1~{CANDIDATE_K} 사이여야 합니다.")

    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")

    model, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    reranker = load_reranker()

    search_question(model, faiss_index, bm25_index, reranker, chunks, args.question, args.top_k, args.batch_size)


if __name__ == "__main__":
    main()
