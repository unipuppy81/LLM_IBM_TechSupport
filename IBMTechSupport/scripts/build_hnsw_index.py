"""
PCA 192차원 임베딩으로 HNSW 인덱스를 별도로 구축.
"""

import time
from pathlib import Path

import faiss
import numpy as np

from retrieval_config import load_retrieval_config


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
VECTORS_PATH = DATA_DIR / "corpus_embeddings_pca192.npy"
FLAT_PATH = DATA_DIR / "corpus_faiss_pca192.index"
CHUNKS_PATH = DATA_DIR / "corpus_chunks.jsonl"

CONFIG = load_retrieval_config()
DENSE = CONFIG["dense"]
PCA = DENSE.get("pca", {})
FAISS_CONFIG = DENSE.get("faiss", {})

DIMENSION = int(PCA.get("dimension", 192))
M = int(FAISS_CONFIG.get("m", 32))
EF_CONSTRUCTION = int(FAISS_CONFIG.get("ef_construction", 80))
EF_SEARCH = int(FAISS_CONFIG.get("ef_search", 128))
THREADS = int(FAISS_CONFIG.get("threads", 4))
BATCH_SIZE = 8192

OUTPUT_PATH = DATA_DIR / f"corpus_faiss_pca{DIMENSION}_hnsw_m{M}.index"


def main():
    if not PCA.get("enabled", False) or DIMENSION != 192:
        raise ValueError("이 스크립트는 PCA 192차원 활성화 상태에서 사용하세요.")
    
    if FAISS_CONFIG.get("index_type", "flat") != "hnsw":
        raise ValueError("dense.faiss.index_type을 hnsw로 설정하세요.")
    
    if M < 2 or EF_CONSTRUCTION < M or EF_SEARCH < 1 or THREADS < 1:
        raise ValueError("HNSW 파라미터가 올바르지 않습니다.")
    
    for path in (VECTORS_PATH, FLAT_PATH, CHUNKS_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
        
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"기존 HNSW 인덱스를 보호합니다: {OUTPUT_PATH}")

    vectors = np.load(VECTORS_PATH, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or vectors.shape[1] != DIMENSION:
        raise ValueError(f"192차원 float32 벡터가 아닙니다: {vectors.shape}")

    flat = faiss.read_index(str(FLAT_PATH))
    if flat.d != DIMENSION or flat.ntotal != len(vectors):
        raise ValueError("FlatIP와 PCA 임베딩의 차원 또는 개수가 일치하지 않습니다.")
    
    with CHUNKS_PATH.open(encoding="utf-8") as source:
        count = sum(bool(line.strip()) for line in source)

    if count != len(vectors):
        raise ValueError("Chunk와 PCA 벡터 개수가 일치하지 않습니다.")

    # 행 ID 순서와 정규화 여부가 기준 FlatIP와 같은지 확인
    for row_id in (0, len(vectors) // 2, len(vectors) - 1):
        vec = np.array(vectors[row_id], dtype=np.float32, copy=True)
        if abs(float(np.linalg.norm(vec)) - 1.0) > 1e-3:
            raise ValueError("PCA 벡터가 L2 정규화되어 있지 않습니다.")
        
        if not np.allclose(flat.reconstruct(row_id), vec, atol=1e-5):
            raise ValueError("PCA 임베딩과 FlatIP의 행 ID가 불일치합니다.")
    del flat

    faiss.omp_set_num_threads(THREADS)
    index = faiss.IndexHNSWFlat(DIMENSION, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH

    print(f"HNSW Build: vectors={len(vectors):,}, dim={DIMENSION}")
    print(f"M={M}, efConstruction={EF_CONSTRUCTION}, efSearch={EF_SEARCH}, threads={THREADS}")
    start = time.perf_counter()
    for offset in range(0, len(vectors), BATCH_SIZE):
        end = min(offset + BATCH_SIZE, len(vectors))
        batch = np.ascontiguousarray(vectors[offset:end], dtype=np.float32)
        index.add(batch)
        print(f"Indexed: {end:,}/{len(vectors):,}", flush=True)

    if index.ntotal != len(vectors) or index.d != DIMENSION:
        raise RuntimeError("HNSW 인덱스의 벡터 수 또는 차원이 일치하지 않습니다.")

    # 재구축 결과 검증: 각 ID에 들어간 벡터가 원본 PCA 벡터와 동일해야 함
    for row_id in (0, len(vectors) // 2, len(vectors) - 1):
        stored = index.reconstruct(row_id)
        expected = np.array(vectors[row_id], dtype=np.float32, copy=True)

        if not np.allclose(stored, expected, atol=1e-5):
            raise RuntimeError(f"HNSW 저장된 벡터 불일치: row={row_id}")

    temporary = Path(str(OUTPUT_PATH) + ".part")
    if temporary.exists():
        raise FileExistsError(f"이전 생성 실패의 임시 파일이 있습니다: {temporary}")
    
    try:
        faiss.write_index(index, str(temporary))
        verified = faiss.read_index(str(temporary))
        if verified.d != DIMENSION or verified.ntotal != len(vectors):
            raise RuntimeError("HNSW 인덱스 재로딩 검증 실패")
        
        if not hasattr(verified, "hnsw") or verified.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise RuntimeError("HNSW 인덱스 타입/거리 계산 방식 검증 실패")
        
        verified.hnsw.efSearch = EF_SEARCH
        scores, ids = verified.search(np.ascontiguousarray(vectors[0:1], dtype=np.float32), 10)
        
        if not np.isfinite(scores[0, 0]) or ids[0, 0] < 0:
            raise RuntimeError("HNSW 검색 검증 실패")
        
        del verified
        temporary.replace(OUTPUT_PATH)
    finally:
        temporary.unlink(missing_ok=True)

    print("\nHNSW build: OK")
    print("Index:", OUTPUT_PATH)
    print("Build seconds:", round(time.perf_counter() - start, 3))
    print("Original PCA FlatIP index and vectors were not modified.")


if __name__ == "__main__":
    main()
