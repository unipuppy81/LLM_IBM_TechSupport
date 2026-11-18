"""
BGE Small 384차원 문서 임베딩을 PCA 192차원으로 축소하고 별도 FAISS 인덱스 구축
"""

import json
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.decomposition import PCA

from retrieval_config import load_retrieval_config


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
INPUT_EMBEDDINGS = DATA_DIR / "corpus_embeddings.npy"
INPUT_CHUNKS = DATA_DIR / "corpus_chunks.jsonl"
INPUT_INDEX = DATA_DIR / "corpus_faiss.index"

OUTPUT_DIM = 192
FITTED_COMPONENTS = 256  # 비교 실험의 PCA에서 192차원 슬라이스를 얻었음
FIT_SAMPLE = 20_000
RANDOM_STATE = 42
BATCH_SIZE = 4096

OUTPUT_PCA = DATA_DIR / "pca_192.joblib"
OUTPUT_EMBEDDINGS = DATA_DIR / "corpus_embeddings_pca192.npy"
OUTPUT_INDEX = DATA_DIR / "corpus_faiss_pca192.index"
OUTPUT_INFO = DATA_DIR / "pca_192_info.json"


def validate_source():
    config = load_retrieval_config()
    dense = config["dense"]
    chunking = config["chunking"]

    if dense["model"] != "BAAI/bge-small-en-v1.5":
        raise ValueError("BGE Small v1.5 모델만 대상으로 합니다.")
    
    if int(dense["max_seq_length"]) != 512 or dense.get("query_prefix", ""):
        raise ValueError("실험 기준을 재현하려면 max_seq_length=512, query_prefix=''이어야 합니다.")
    
    if not dense["normalize_embeddings"]:
        raise ValueError("실험 기준: normalize_embeddings=true")
    
    if chunking.get("mode", "char") != "char" or chunking["size"] != 1250 or chunking["overlap"] != 150:
        raise ValueError("실험 기준: 문자 청크 1250/150")

    for path in (INPUT_EMBEDDINGS, INPUT_CHUNKS, INPUT_INDEX):
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (OUTPUT_PCA, OUTPUT_EMBEDDINGS, OUTPUT_INDEX, OUTPUT_INFO):
        if path.exists():
            raise FileExistsError(f"기존 축소 결과를 보호합니다: {path}")

    vectors = np.load(INPUT_EMBEDDINGS, mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or vectors.shape[1] != 384:
        raise ValueError(f"원본 벡터는 float32 / 384차원이어야 합니다: {vectors.shape}")

    with INPUT_CHUNKS.open("r", encoding="utf-8") as file:
        chunk_count = sum(bool(line.strip()) for line in file)
    source_index = faiss.read_index(str(INPUT_INDEX))
    
    if chunk_count != vectors.shape[0] or source_index.ntotal != vectors.shape[0] or source_index.d != 384:
        raise ValueError("원본 청크 / 임베딩 / FAISS 개수 또는 차원이 불일치합니다.")
    
    if vectors.shape[0] <= FITTED_COMPONENTS:
        raise ValueError("PCA 학습에 필요한 문서 벡터 수가 부족합니다.")

    return vectors


def main():
    vectors = validate_source()
    n = vectors.shape[0]
    sample_n = min(FIT_SAMPLE, n)

    print(f"Source: {n:,} vectors / 384 dimensions")
    print(f"PCA: fit {FITTED_COMPONENTS} components on {sample_n:,} sampled vectors; retain {OUTPUT_DIM}")

    rng = np.random.default_rng(RANDOM_STATE)
    sampled_rows = rng.choice(n, size=sample_n, replace=False)
    sample = np.asarray(vectors[sampled_rows], dtype=np.float32)
    pca = PCA(n_components=FITTED_COMPONENTS, svd_solver="randomized", random_state=RANDOM_STATE)
    start = time.perf_counter()
    pca.fit(sample)
    print(f"PCA fitted in {time.perf_counter() - start:.2f}s")

    index = faiss.IndexFlatIP(OUTPUT_DIM)
    temporary = {
        OUTPUT_PCA: Path(str(OUTPUT_PCA) + ".part"),
        OUTPUT_EMBEDDINGS: Path(str(OUTPUT_EMBEDDINGS) + ".part"),
        OUTPUT_INDEX: Path(str(OUTPUT_INDEX) + ".part"),
        OUTPUT_INFO: Path(str(OUTPUT_INFO) + ".part"),
    }
    for temp in temporary.values():
        if temp.exists():
            raise FileExistsError(f"이전 실패에서 남은 임시 파일을 확인하세요: {temp}")

    output = None
    try:
        output = np.lib.format.open_memmap(temporary[OUTPUT_EMBEDDINGS], mode="w+", dtype=np.float32, shape=(n, OUTPUT_DIM))
        for offset in range(0, n, BATCH_SIZE):
            end = min(offset + BATCH_SIZE, n)
            batch = np.asarray(vectors[offset:end], dtype=np.float32)
            reduced = np.ascontiguousarray(pca.transform(batch)[:, :OUTPUT_DIM], dtype=np.float32)
            
            if not np.isfinite(reduced).all() or np.any(np.linalg.norm(reduced, axis=1) < 1e-12):
                raise ValueError(f"잘못된 PCA 벡터: {offset}:{end}")
            
            faiss.normalize_L2(reduced)
            output[offset:end] = reduced
            index.add(reduced)
            print(f"Projected: {end:,}/{n:,}", flush=True)

        output.flush()
        del output
        output = None

        if index.ntotal != n or index.d != OUTPUT_DIM:
            raise RuntimeError("PCA 인덱스 크기 불일치")

        # PCA 결과의 단위 벡터 검증
        check = np.load(temporary[OUTPUT_EMBEDDINGS], mmap_mode="r", allow_pickle=False)
        for row_id in (0, n // 2, n - 1):
            row = np.array(check[row_id:row_id + 1], dtype=np.float32, copy=True)
            scores, ids = index.search(np.ascontiguousarray(row), 1)
            if ids[0, 0] < 0 or scores[0, 0] < 0.999:
                raise RuntimeError(f"검색 검증 실패: {row_id}, {scores[0, 0]}")
            
        del row
        del check

        joblib.dump(pca, temporary[OUTPUT_PCA])
        faiss.write_index(index, str(temporary[OUTPUT_INDEX]))
        checked_index = faiss.read_index(str(temporary[OUTPUT_INDEX]))
        
        if checked_index.ntotal != n or checked_index.d != OUTPUT_DIM:
            raise RuntimeError("저장한 PCA 인덱스 검증 실패")

        info = {
            "source_model": "BAAI/bge-small-en-v1.5",
            "original_dimensions": 384,
            "pca_fit_components": FITTED_COMPONENTS,
            "output_dimensions": OUTPUT_DIM,
            "pca_fit_sample": sample_n,
            "random_state": RANDOM_STATE,
            "source_vectors": n,
            "pca_explained_variance_ratio": round(float(pca.explained_variance_ratio_[:OUTPUT_DIM].sum()), 6),
            "normalized": True,
            "faiss_index": "IndexFlatIP",
            "note": "PCA fitted with 256 components; first 192 are used to reproduce dimension-comparison experiment",
        }
        temporary[OUTPUT_INFO].write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        for dest, temp in temporary.items():
            temp.replace(dest)

    finally:
        if output is not None:
            output.flush()
            del output
            
        for temp in temporary.values():
            temp.unlink(missing_ok=True)

    print("\nPCA 192 build: OK")
    print("PCA model:", OUTPUT_PCA)
    print("Reduced vectors:", OUTPUT_EMBEDDINGS)
    print("FAISS index:", OUTPUT_INDEX)
    print("Original 384-dimensional files were not modified.")


if __name__ == "__main__":
    main()
