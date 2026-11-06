import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

INDEX_PATH = DATA_DIR / "corpus_faiss.index"
CHUNKS_PATH = DATA_DIR / "corpus_chunks.jsonl"
VAL_PATH = DATA_DIR / "val_clean.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# region 검색 데이터 및 임베딩 모델 로드

def load_retriever():
    if not INDEX_PATH.is_file() or not CHUNKS_PATH.is_file():
        raise FileNotFoundError("FAISS 인덱스 또는 청크 파일이 없습니다.")

    index = faiss.read_index(str(INDEX_PATH))

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if index.ntotal != len(chunks):
        raise ValueError("인덱스 벡터 수와 청크 수가 일치하지 않습니다.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("Device:", device)
    print("Embedding Model:", EMBEDDING_MODEL)
    print("Index Vectors:", index.ntotal)

    return model, index, chunks


def chunk_filename(chunk):
    metadata = chunk.get("metadata")

    candidates = (
        [chunk, metadata]
        if isinstance(metadata, dict)
        else [chunk]
    )

    for data in candidates:
        for key in (
            "filename",
            "source_file",
            "source",
            "document_id",
            "doc_id",
        ):
            value = data.get(key)

            if isinstance(value, str) and value.strip():
                return value

    raise ValueError(f"청크에 문서 파일명 정보가 없습니다: {list(chunk)}")


def filename_key(filename):
    return str(filename).strip().replace("\\", "/").split("/")[-1].casefold()

# endregion


# region 질문 임베딩

def encode_questions(model, questions):
    # 문서 임베딩에 사용한 것과 동일한 모델 사용
    vectors = model.encode(
        questions,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.ascontiguousarray(vectors, dtype=np.float32,)

# endregion


# region FAISS Dense Search

def dense_search(model, index, chunks, question, top_k):
    # 1. 사용자 질문을 384차원 벡터로 변환
    query_vector = encode_questions(model, [question])

    # 2. FAISS에서 코사인 유사도가 높은 청크 검색
    scores, indices = index.search(query_vector, top_k)

    # 3. 검색된 벡터의 행 번호로 원본 청크 조회
    results = []

    for score, row_id in zip(scores[0], indices[0]):
        if row_id < 0:
            continue

        chunk = chunks[int(row_id)]

        results.append({
            **chunk,
            "row_id": int(row_id),
            "score": float(score),
        })

    return results

# endregion


# region Dense Retrieval 성능 평가

def evaluate(model, index, chunks):
    if not VAL_PATH.is_file():
        raise FileNotFoundError(f"검증 데이터가 없습니다: {VAL_PATH}")

    with VAL_PATH.open(encoding="utf-8") as file:
        records = json.load(file)

    # 정답 문서가 있는 질문만 검색 성능 평가
    answerable = [
        record
        for record in records
        if not record["is_impossible"]
    ]

    questions = [
        record["question"]
        for record in answerable
    ]

    if not questions:
        raise ValueError("평가 가능한 질문이 없습니다.")

    # 질문 전체 임베딩
    start = time.perf_counter()
    query_vectors = encode_questions(model, questions)
    encoding_seconds = time.perf_counter() - start

    # 각 질문에서 상위 10개 청크 검색
    start = time.perf_counter()
    scores, indices = index.search(query_vectors, 10)
    search_seconds = time.perf_counter() - start

    hits = 0
    total_recall = 0.0
    total_rr = 0.0

    for record, row_ids in zip(answerable, indices):
        # 데이터셋의 실제 정답 문서명
        gold = {
            filename_key(context["filename"])
            for context in record["contexts"]
        }

        if not gold:
            raise ValueError(f"정답 문서 정보가 없습니다: {record['id']}")

        # 검색된 청크가 속한 문서명
        retrieved = [
            filename_key(chunk_filename(chunks[int(row_id)]))
            for row_id in row_ids
            if row_id >= 0
        ]

        # Hit@5
        hits += int(bool(gold.intersection(retrieved[:5])))

        # Recall@5
        total_recall += (len(gold.intersection(retrieved[:5])) / len(gold))

        # MRR@10
        total_rr += next(
            (
                1 / rank
                for rank, filename in enumerate(retrieved[:10], 1)
                if filename in gold
            ),
            0.0,
        )

    count = len(answerable)

    metrics = {
        "dataset": "val_clean.json",
        "retriever": "Dense / FAISS IndexFlatIP",
        "embedding_model": EMBEDDING_MODEL,
        "total": len(records),
        "answerable_evaluated": count,
        "impossible_skipped": len(records) - count,
        "hit_at_5_percent": round(hits / count * 100, 2),
        "recall_at_5_percent": round(
            total_recall / count * 100, 2
        ),
        "mrr_at_10": round(total_rr / count, 4),
        "query_encoding_seconds": round(encoding_seconds, 3),
        "faiss_search_seconds": round(search_seconds, 3),
        "avg_faiss_search_ms_per_question": round(
            search_seconds / count * 1000, 3
        ),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (RESULT_DIR / "dense_retrieval_val_clean.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2,)

    print("\n[Dense Retrieval Evaluation]")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Metrics:", output_path)
    print("Dense Retrieval: OK")

# endregion


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--question",
        type=str,
        help="사용자 질문으로 문서 검색",
    )

    group.add_argument(
        "--evaluate",
        action="store_true",
        help="val_clean 전체 검색 성능 평가",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k는 1 이상이어야 합니다.")

    model, index, chunks = load_retriever()

    if args.evaluate:
        evaluate(model, index, chunks)

    else:
        results = dense_search(
            model,
            index,
            chunks,
            args.question,
            args.top_k,
        )

        for rank, hit in enumerate(results, 1):
            filename = chunk_filename(hit)

            content = hit.get("text", hit.get("content", hit.get("chunk_text", ""), ), )

            print(
                f"\n[{rank}] {filename} "
                f"| cosine={hit['score']:.4f}"
            )

            print(str(content)[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
