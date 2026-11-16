import argparse
import json

import bm25s

from dense_search import DATA_DIR, chunk_filename, encode_questions, load_retriever

BM25_INDEX_DIR = DATA_DIR / "hybrid_bm25_index"
CANDIDATE_K = 50
RRF_K = 60
TOP_K = 10



# region BM25 인덱스 로딩 및 최초 구축

def load_bm25(chunks):
    marker = BM25_INDEX_DIR / "params.index.json"

    if marker.is_file():
        retriever = bm25s.BM25.load(
            str(BM25_INDEX_DIR),
            load_corpus=False,
            mmap=True,
        )
        print("BM25 Index: Loaded", BM25_INDEX_DIR)

    else:
        texts = []

        for row_id, chunk in enumerate(chunks):
            value = (chunk.get("text") or chunk.get("content") or chunk.get("chunk_text"))

            if not isinstance(value, str):
                raise ValueError(f"청크 텍스트를 찾을 수 없습니다: row={row_id}")

            texts.append(value)

        retriever = bm25s.BM25()
        tokens = bm25s.tokenize(
            texts,
            stopwords="en",
            show_progress=True,
        )

        retriever.index(tokens)

        BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        retriever.save(str(BM25_INDEX_DIR))

        print("BM25 Index: Built", BM25_INDEX_DIR)

    if int(retriever.scores["num_docs"]) != len(chunks):
        raise ValueError("BM25 인덱스와 FAISS 인덱스의 청크 개수가 다릅니다.")

    return retriever

# endregion


# region BM25 / Dense 검색 및 RRF 결합

def search_all(model, faiss_index, bm25_index, questions):
    k = min(CANDIDATE_K, faiss_index.ntotal)

    # Dense Retrieval: 질문 임베딩 -> FAISS 검색
    vectors = encode_questions(model, questions)
    dense_scores, dense_ids = faiss_index.search(vectors, k)

    # BM25: 질문 토큰화 -> 키워드 검색
    tokens = bm25s.tokenize(
        questions,
        stopwords="en",
        show_progress=False,
    )

    bm25_ids, bm25_scores = bm25_index.retrieve(
        tokens,
        k=k,
        show_progress=False,
    )

    all_results = []

    for d_ids, b_ids, b_scores in zip(dense_ids, bm25_ids, bm25_scores):
        dense_ranked = [
            int(i)
            for i in d_ids
            if i >= 0
        ]

        bm25_ranked = [
            int(i)
            for i, score in zip(b_ids, b_scores)
            if i >= 0 and score > 0
        ]

        # RRF: 각 검색기의 순위 점수를 합산
        rrf_scores = {}

        for ranked in (bm25_ranked, dense_ranked):
            for rank, row_id in enumerate(ranked, start=1):
                rrf_scores[row_id] = (rrf_scores.get(row_id, 0.0) + 1 / (RRF_K + rank))

        # RRF 점수가 높은 순서대로 정렬
        hybrid_ranked = sorted(rrf_scores, key=lambda row_id: (-rrf_scores[row_id], row_id, ), )

        all_results.append((bm25_ranked, dense_ranked, hybrid_ranked, rrf_scores, ))

    return all_results

# endregion


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
