import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from document_utils import chunk_filename

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

INDEX_PATH = DATA_DIR / "corpus_faiss.index"
CHUNKS_PATH = DATA_DIR / "corpus_chunks.jsonl"

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

    return np.ascontiguousarray(vectors, dtype=np.float32)

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
        results.append({**chunk, "row_id": int(row_id), "score": float(score) })

    return results

# endregion


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--question", required=True, type=str)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k는 1 이상이어야 합니다.")

    model, index, chunks = load_retriever()
    results = dense_search(model, index, chunks, args.question, args.top_k)

    for rank, hit in enumerate(results, start=1):
        filename = chunk_filename(hit)

        content = (hit.get("text") or hit.get("content") or hit.get("chunk_text", ""))

        print(f"\n[{rank}] {filename} | cosine={hit['score']:.4f}")
        print(str(content)[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
