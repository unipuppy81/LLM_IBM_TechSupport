import argparse
import json
import time

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from dense_search import (
    RESULT_DIR,
    VAL_PATH,
    chunk_filename,
    load_retriever,
)

from hybrid_search import (
    load_bm25,
    measure,
    search_all,
)


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


# region 04. Reranking

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


# region Reranker 적용 전후 성능 평가

def evaluate(model, faiss_index, bm25_index, reranker, chunks, batch_size, limit,):
    with VAL_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    # 답변 가능한 질문만 평가
    records = [
        record
        for record in data
        if not record["is_impossible"]
    ]

    if limit:
        records = records[:limit]

    if not records:
        raise ValueError("평가할 질문이 없습니다.")

    questions = [
        record["question"]
        for record in records
    ]

    # 1. Hybrid Search
    search_start = time.perf_counter()
    search_results = search_all(model, faiss_index, bm25_index, questions,)
    search_seconds = time.perf_counter() - search_start

    hybrid_rankings = [
        result[2]
        for result in search_results
    ]

    # 2. Reranker
    rerank_start = time.perf_counter()
    reranked = rerank_candidates(reranker, questions, hybrid_rankings, chunks, batch_size, )
    rerank_seconds = time.perf_counter() - rerank_start

    reranked_ids = [
        [item["row_id"] for item in results]
        for results in reranked
    ]

    # 같은 평가 기준으로 성능 비교
    hybrid_metrics = measure(hybrid_rankings, records, chunks, )
    reranker_metrics = measure(reranked_ids, records, chunks, )

    metrics = {
        "dataset": "val_clean.json",
        "total_evaluated": len(records),
        "reranker_model": RERANKER_MODEL,
        "candidate_k": CANDIDATE_K,
        "batch_size": batch_size,
        "hybrid": hybrid_metrics,
        "reranker": reranker_metrics,
        "search_seconds": round(search_seconds, 3),
        "rerank_seconds": round(rerank_seconds, 3),
        "avg_rerank_ms_per_question": round(rerank_seconds / len(records) * 1000, 3)
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"_smoke{limit}"
        if limit
        else ""
    )

    output_path = (RESULT_DIR / f"reranker_retrieval_val_clean{suffix}.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, )

    print("\n[Hybrid vs Reranker Evaluation]")
    print("Evaluated:", len(records))

    print("\nHybrid:")
    for key, value in hybrid_metrics.items():
        print(f"  {key}: {value}")

    print("\nReranker:")
    for key, value in reranker_metrics.items():
        print(f"  {key}: {value}")

    print("\nSearch Seconds:", metrics["search_seconds"])
    print("Rerank Seconds:", metrics["rerank_seconds"])
    print("Metrics:", output_path)
    print("Reranker Evaluation: OK")

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

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--question", type=str)
    group.add_argument("--evaluate", action="store_true")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    if args.top_k < 1 or args.top_k > CANDIDATE_K:
        parser.error(f"--top-k는 1~{CANDIDATE_K} 사이여야 합니다.")

    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size >= 1, --limit >= 0 이어야 합니다.")

    # 기존 검색 시스템 로드
    model, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)

    # Cross-Encoder 로드
    reranker = load_reranker()

    if args.evaluate:
        evaluate(model, faiss_index, bm25_index, reranker, chunks, args.batch_size, args.limit)

    else:
        search_question(model, faiss_index, bm25_index, reranker, chunks, args.question, args.top_k, args.batch_size)


if __name__ == "__main__":
    main()
