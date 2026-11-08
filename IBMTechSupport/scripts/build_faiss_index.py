from pathlib import Path

import faiss
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
EMBEDDINGS_PATH = DATA_DIR / "corpus_embeddings.npy"
CHUNKS_PATH = DATA_DIR / "corpus_chunks.jsonl"
INDEX_PATH = DATA_DIR / "corpus_faiss.index"
BATCH_SIZE = 8192

# region 원본 데이터 검증
def load_embeddings():
    if not EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"임베딩 파일이 없습니다: {EMBEDDINGS_PATH}")

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r", allow_pickle=False)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"임베딩 배열의 형태가 올바르지 않습니다: {embeddings.shape}")
    
    if embeddings.dtype != np.float32:
        raise ValueError(f"임베딩 자료형이 float32가 아닙니다: {embeddings.dtype}")
    
    return embeddings


def validate_chunk_count(expected_count):
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"청크 파일이 없습니다: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        actual_count = sum(1 for line in file if line.strip())

    if actual_count != expected_count:
        raise ValueError(f"청크 수({actual_count})와 임베딩 수({expected_count})가 다릅니다.")
    
    return actual_count
# endregion


# region FAISS 인덱스 생성
def build_index(embeddings):
    count, dimension = embeddings.shape

    # L2 정규화한 벡터의 내적 = 코사인 유사도
    # IndexFlatIP: 근사 검색이 아닌 정확한 내적 검색
    index = faiss.IndexFlatIP(dimension)

    for start in range(0, count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, count)
        vectors = np.array(embeddings[start:end], dtype=np.float32, order="C", copy=True)

        if not np.isfinite(vectors).all():
            raise ValueError(f"NaN/Inf 임베딩 발견: {start}:{end}")
        
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError(f"영벡터 임베딩 발견: {start}:{end}")

        faiss.normalize_L2(vectors)
        index.add(vectors)
        print(f"Indexed: {index.ntotal}/{count}", flush=True)

    return index
# endregion


# region 저장 및 검색 검증
def verify_index(index, embeddings):
    count, dimension = embeddings.shape
    if index.ntotal != count or index.d != dimension:
        raise ValueError("FAISS 인덱스의 벡터 수 또는 차원이 일치하지 않습니다.")

    sample_ids = [0, count // 2, count - 1]
    queries = np.array(embeddings[sample_ids], dtype=np.float32, order="C", copy=True)
    faiss.normalize_L2(queries)

    scores, ids = index.search(queries, 1)

    for sample_id, score, hit_id in zip(sample_ids, scores[:, 0], ids[:, 0]):
        # 동일 벡터가 여러 개면 자기 자신이 아닌 다른 인덱스가 1위일 수 있음
        if not 0 <= hit_id < count or score < 0.999:
            raise ValueError(f"FAISS 검색 검증 실패: {sample_id}, {hit_id}, {score}")
        
        print(
            f"Search Test: query_row={sample_id}, "
            f"hit_row={hit_id}, score={score:.4f}"
        )


def save_and_verify(index, embeddings):
    temporary_path = INDEX_PATH.with_suffix(INDEX_PATH.suffix + ".tmp")
    try:
        faiss.write_index(index, str(temporary_path))
        reloaded_index = faiss.read_index(str(temporary_path))
        verify_index(reloaded_index, embeddings)
        temporary_path.replace(INDEX_PATH)

    finally:
        temporary_path.unlink(missing_ok=True)
# endregion

def main():
    embeddings = load_embeddings()
    chunk_count = validate_chunk_count(embeddings.shape[0])

    print("Embedding Shape:", embeddings.shape)
    print("Chunk Count:", chunk_count)
    print("Index Type: IndexFlatIP (cosine similarity with L2 normalization)")

    index = build_index(embeddings)
    save_and_verify(index, embeddings)

    print("Index Path:", INDEX_PATH)
    print("Index Vectors:", index.ntotal)
    print("Index Dimension:", index.d)
    print("FAISS Index: OK")


if __name__ == "__main__":
    main()

