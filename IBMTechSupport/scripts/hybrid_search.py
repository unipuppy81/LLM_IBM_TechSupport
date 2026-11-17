import argparse

from bm25_search import (
    INDEX_DIR as BM25_INDEX_DIR,
    load_index as load_bm25_index,
    retrieve as bm25_retrieve
)

from dense_search import chunk_filename, encode_questions, load_retriever
from retrieval_config import load_retrieval_config

# YAML에서 Hybrid 설정 불러오기
HYBRID_CONFIG = load_retrieval_config()["hybrid"]

CANDIDATE_K = HYBRID_CONFIG["candidate_k"]
RRF_K = HYBRID_CONFIG["rrf_k"]

BM25_WEIGHT = HYBRID_CONFIG["bm25_weight"]
DENSE_WEIGHT = HYBRID_CONFIG["dense_weight"]

TOP_K = 10



# BM25 인덱스 로딩 및 최초 구축
def load_bm25(chunks):
    retriever, tokenizer = load_bm25_index()

    # BM25와 FAISS의 문서 개수 일치 여부 확인
    if int(retriever.scores["num_docs"]) != len(chunks):
        raise ValueError("BM25와 FAISS의 청크 개수가 다릅니다.")

    # BM25 검색 결과의 chunk_id를 FAISS row_id로 변환하기 위한 매핑
    chunk_id_to_row_id = {
        chunk["chunk_id"]: row_id
        for row_id, chunk in enumerate(chunks)
    }

    if len(chunk_id_to_row_id) != len(chunks):
        raise ValueError("중복된 chunk_id가 존재합니다.")

    return retriever, tokenizer, chunk_id_to_row_id


# BM25 / Dense 검색 및 RRF 결합
def search_all(model, faiss_index, bm25_index, questions):
    k = min(CANDIDATE_K, faiss_index.ntotal)

    retriever, tokenizer, chunk_id_to_row_id = bm25_index

    # 1. Dense 검색
    vectors = encode_questions(model, questions)
    _, dense_ids = faiss_index.search(vectors, k)

    all_results = []

    for question, d_ids in zip(questions, dense_ids):

        # 2. bm25_search.py의 검색 함수 사용
        bm25_results = bm25_retrieve(
            question,
            retriever,
            tokenizer,
            k
        )

        # 3. BM25 검색 결과를 FAISS row_id로 변환
        bm25_ranked = [
            chunk_id_to_row_id[doc["chunk_id"]]
            for doc in bm25_results
            if doc["score"] > 0
        ]

        # 4. Dense 검색 결과 정리
        dense_ranked = [
            int(row_id)
            for row_id in d_ids
            if row_id >= 0
        ]

        # 5. Weighted RRF 점수 계산
        rrf_scores = {}

        for ranked, weight in ((bm25_ranked, BM25_WEIGHT), (dense_ranked, DENSE_WEIGHT)):
            for rank, row_id in enumerate(ranked, start=1):
                rrf_scores[row_id] = (rrf_scores.get(row_id, 0.0) + weight / (RRF_K + rank))

        # 6. RRF 점수 내림차순 정렬
        hybrid_ranked = sorted(rrf_scores, key=lambda row_id: (-rrf_scores[row_id], row_id))

        all_results.append((bm25_ranked, dense_ranked, hybrid_ranked, rrf_scores))

    return all_results

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--top-k", type=int, default=TOP_K)

    args = parser.parse_args()

    if not 1 <= args.top_k <= CANDIDATE_K:
        parser.error(f"--top-k는 1~{CANDIDATE_K} 사이여야 합니다.")

    model, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)

    results = search_all(model, faiss_index, bm25_index, [args.question])[0]

    rankings = results[2]
    scores = results[3]

    for rank, row_id in enumerate(rankings[:args.top_k], start=1):
        chunk = chunks[row_id]

        content = (chunk.get("text") or chunk.get("content") or chunk.get("chunk_text", ""))

        print(
            f"\n[{rank}] {chunk_filename(chunk)} "
            f"| RRF={scores[row_id]:.5f}"
        )
        print(str(content)[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
