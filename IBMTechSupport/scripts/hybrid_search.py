"""BM25 + Dense weighted RRF 검색.

Hybrid 최적화 통합 설정(retrieval.yaml):
    candidate_k: 50
    rrf_k: 10
    bm25_weight: 0.5
    dense_weight: 1.0
    max_chunks_per_document: 1

동일 원본 문서의 청크 수 제한은 RRF 순위 산출 후, Reranker에
입력되기 전에 적용한다. 기존 search_all 반환 구조는 유지한다.
"""

import argparse
from collections import Counter

from bm25_search import (
    INDEX_DIR as BM25_INDEX_DIR,
    load_index as load_bm25_index,
    retrieve as bm25_retrieve,
)
from dense_search import chunk_filename, encode_questions, load_retriever
from document_utils import filename_key
from retrieval_config import load_retrieval_config


HYBRID_CONFIG = load_retrieval_config()["hybrid"]

CANDIDATE_K = int(HYBRID_CONFIG["candidate_k"])
RRF_K = int(HYBRID_CONFIG["rrf_k"])
BM25_WEIGHT = float(HYBRID_CONFIG["bm25_weight"])
DENSE_WEIGHT = float(HYBRID_CONFIG["dense_weight"])

# 설정이 없는 기존 YAML에서는 중복 제한을 적용하지 않음.
MAX_CHUNKS_PER_DOCUMENT = HYBRID_CONFIG.get("max_chunks_per_document")
if MAX_CHUNKS_PER_DOCUMENT is not None:
    if (isinstance(MAX_CHUNKS_PER_DOCUMENT, bool)
            or not isinstance(MAX_CHUNKS_PER_DOCUMENT, int)
            or MAX_CHUNKS_PER_DOCUMENT < 1):
        raise ValueError("hybrid.max_chunks_per_document는 양의 정수 또는 null이어야 합니다.")

if CANDIDATE_K < 1 or RRF_K < 0:
    raise ValueError("hybrid.candidate_k는 1 이상, rrf_k는 0 이상이어야 합니다.")
if BM25_WEIGHT < 0 or DENSE_WEIGHT < 0 or BM25_WEIGHT + DENSE_WEIGHT <= 0:
    raise ValueError("Hybrid 가중치는 0 이상이어야 하고, 둘 다 0일 수 없습니다.")

TOP_K = 10


class ChunkRowMap(dict):
    """기존 dict API를 유지하며 문서 파일명 키를 함께 보관."""

    def __init__(self, values, source_keys):
        super().__init__(values)
        self.source_keys = source_keys


def load_bm25(chunks):
    """BM25 인덱스 로드 및 chunk_id -> FAISS row_id 매핑."""
    retriever, tokenizer = load_bm25_index()

    if int(retriever.scores["num_docs"]) != len(chunks):
        raise ValueError("BM25와 FAISS의 청크 개수가 다릅니다.")

    # 결과는 기존과 같은 3-tuple로 반환하며, 매핑은 dict처럼 사용된다.
    # 검색 중복 제한을 위해 row_id별 원본 문서 파일명 키를 함께 보관한다.
    chunk_id_to_row_id = ChunkRowMap(
        {
            chunk["chunk_id"]: row_id
            for row_id, chunk in enumerate(chunks)
        },
        [filename_key(chunk_filename(chunk)) for chunk in chunks],
    )

    if len(chunk_id_to_row_id) != len(chunks):
        raise ValueError("중복된 chunk_id가 존재합니다.")

    return retriever, tokenizer, chunk_id_to_row_id


def limit_chunks_per_document(ranking, source_keys, max_chunks):
    """기존 RRF 순서를 유지하면서 원본 문서별 청크 수를 제한.

    전체 후보를 순회하므로 앞에서 중복 제거된 청크의 빈자리를
    아래 순위의 서로 다른 문서 청크로 채울 수 있다.
    benchmark_hybrid_dedup.py와 동일한 filename_key 기준.
    """
    if max_chunks is None:
        return list(ranking)

    counts = Counter()
    filtered = []

    for row_id in ranking:
        source_key = source_keys[row_id]
        if counts[source_key] >= max_chunks:
            continue
        counts[source_key] += 1
        filtered.append(row_id)

    return filtered


def search_all(model, faiss_index, bm25_index, questions):
    """각 질문에 대해 (BM25 순위, Dense 순위, Hybrid 순위, RRF 점수) 반환."""
    k = min(CANDIDATE_K, faiss_index.ntotal)
    retriever, tokenizer, chunk_id_to_row_id = bm25_index

    # 1. Dense 검색 (질문을 일괄 인코딩)
    vectors = encode_questions(model, questions)
    _, dense_ids = faiss_index.search(vectors, k)

    all_results = []

    for question, d_ids in zip(questions, dense_ids):
        # 2. BM25 검색
        bm25_results = bm25_retrieve(question, retriever, tokenizer, k)
        bm25_ranked = [
            chunk_id_to_row_id[doc["chunk_id"]]
            for doc in bm25_results
            if doc["score"] > 0
        ]

        dense_ranked = [int(row_id) for row_id in d_ids if row_id >= 0]

        # 3. Weighted RRF (실험 코드와 동일한 계산 및 동점 규칙)
        rrf_scores = {}
        for ranked, weight in (
            (bm25_ranked, BM25_WEIGHT),
            (dense_ranked, DENSE_WEIGHT),
        ):
            for rank, row_id in enumerate(ranked, start=1):
                rrf_scores[row_id] = (
                    rrf_scores.get(row_id, 0.0)
                    + weight / (RRF_K + rank)
                )

        hybrid_ranked = sorted(
            rrf_scores,
            key=lambda row_id: (-rrf_scores[row_id], row_id),
        )

        # 4. Reranker 입력 전에 중복 청크 제한
        #    상위 10개만 확인하지 않고 RRF 후보 전체에서 보충한다.
        hybrid_ranked = limit_chunks_per_document(
            hybrid_ranked,
            chunk_id_to_row_id.source_keys,
            MAX_CHUNKS_PER_DOCUMENT,
        )

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
    rankings, scores = results[2], results[3]

    for rank, row_id in enumerate(rankings[:args.top_k], start=1):
        chunk = chunks[row_id]
        content = chunk.get("text") or chunk.get("content") or chunk.get("chunk_text", "")
        print(f"\n[{rank}] {chunk_filename(chunk)} | RRF={scores[row_id]:.5f}")
        print(str(content)[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
