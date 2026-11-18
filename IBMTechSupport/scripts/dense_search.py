import argparse
import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from document_utils import chunk_filename
from retrieval_config import load_retrieval_config


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
CHUNKS_PATH = DATA_DIR / "corpus_chunks.jsonl"

DENSE_CONFIG = load_retrieval_config()["dense"]
EMBEDDING_MODEL = DENSE_CONFIG["model"]
MODEL_DIR = PROJECT_DIR / "models" / EMBEDDING_MODEL.split("/")[-1]
BATCH_SIZE = DENSE_CONFIG["batch_size"]
NORMALIZE_EMBEDDINGS = DENSE_CONFIG["normalize_embeddings"]
MAX_SEQ_LENGTH = DENSE_CONFIG["max_seq_length"]
QUERY_PREFIX = DENSE_CONFIG.get("query_prefix", "")

PCA_CONFIG = DENSE_CONFIG.get("pca", {})
PCA_ENABLED = bool(PCA_CONFIG.get("enabled", False))
PCA_DIMENSION = int(PCA_CONFIG.get("dimension", 192))

FAISS_CONFIG = DENSE_CONFIG.get("faiss", {})
FAISS_INDEX_TYPE = FAISS_CONFIG.get("index_type", "flat")
HNSW_M = int(FAISS_CONFIG.get("m", 32))
HNSW_EF_SEARCH = int(FAISS_CONFIG.get("ef_search", 128))
FAISS_THREADS = int(FAISS_CONFIG.get("threads", 4))

if FAISS_INDEX_TYPE not in ("flat", "hnsw"):
    raise ValueError("dense.faiss.index_type은 flat 또는 hnsw여야 합니다.")

if FAISS_INDEX_TYPE == "hnsw" and not PCA_ENABLED:
    raise ValueError("현재 HNSW 빌드는 PCA 활성화 상태만 지원합니다.")

if HNSW_M < 2 or HNSW_EF_SEARCH < 1 or FAISS_THREADS < 1:
    raise ValueError("dense.faiss 설정이 올바르지 않습니다.")

if FAISS_INDEX_TYPE == "hnsw":
    index_name = f"corpus_faiss_pca{PCA_DIMENSION}_hnsw_m{HNSW_M}.index"
elif PCA_ENABLED:
    index_name = f"corpus_faiss_pca{PCA_DIMENSION}.index"
else:
    index_name = "corpus_faiss.index"

INDEX_PATH = DATA_DIR / index_name
PCA_MODEL_PATH = DATA_DIR / f"pca_{PCA_DIMENSION}.joblib"


@lru_cache(maxsize=1)
def load_pca():
    """문서 인덱스 구축에 사용한 것과 동일한 PCA 변환을 불러옴."""
    if not PCA_MODEL_PATH.is_file():
        raise FileNotFoundError(f"PCA 변환 파일이 없습니다: {PCA_MODEL_PATH}. build_pca_index.py부터 실행하세요.")

    import joblib
    pca = joblib.load(PCA_MODEL_PATH)  # 로컬에서 직접 생성한 신뢰할 수 있는 파일만 로드
    if pca.n_components_ < PCA_DIMENSION:
        raise ValueError("PCA 모델의 주성분 수가 설정한 차원보다 작습니다.")
    
    return pca


def load_retriever():
    """FAISS 인덱스, 청크, 원본 임베딩 모델 로드."""
    if not INDEX_PATH.is_file() or not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"FAISS 인덱스 또는 청크 파일이 없습니다: {INDEX_PATH}")

    index = faiss.read_index(str(INDEX_PATH))

    if FAISS_INDEX_TYPE == "hnsw":
        if not hasattr(index, "hnsw"):
            raise ValueError("설정은 HNSW인데 인덱스 파일이 HNSW가 아닙니다.")
        
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("HNSW 인덱스는 INNER_PRODUCT 방식이어야 합니다.")
        
        faiss.omp_set_num_threads(FAISS_THREADS)
        index.hnsw.efSearch = HNSW_EF_SEARCH
    elif index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FlatIP 인덱스는 INNER_PRODUCT 방식이어야 합니다.")

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if index.ntotal != len(chunks):
        raise ValueError("인덱스 벡터 수와 청크 수가 일치하지 않습니다.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(MODEL_DIR), device=device, local_files_only=True)
    model.max_seq_length = MAX_SEQ_LENGTH

    if hasattr(model, "get_embedding_dimension"):
        native_dim = model.get_embedding_dimension()
    else:
        native_dim = model.get_sentence_embedding_dimension()

    if PCA_ENABLED:
        if not NORMALIZE_EMBEDDINGS:
            raise ValueError("PCA 비교 실험을 재현하려면 normalize_embeddings: true여야 합니다.")
        
        if index.d != PCA_DIMENSION:
            raise ValueError("FAISS 인덱스와 PCA 출력 차원이 다릅니다.")

        if load_pca().n_features_in_ != native_dim:
            raise ValueError("PCA 모델의 입력 차원과 원본 Embedding 차원이 다릅니다.")
        
    elif native_dim != index.d:
        raise ValueError("FAISS 인덱스와 임베딩 모델 차원이 다릅니다.")

    print("Device:", device)
    print("Embedding Model:", EMBEDDING_MODEL)
    print("Index Vectors:", index.ntotal)
    print("Index Dimension:", index.d)
    print("PCA:", f"enabled ({native_dim} -> {PCA_DIMENSION})" if PCA_ENABLED else "disabled")
    print("FAISS Index Type:", FAISS_INDEX_TYPE)

    if FAISS_INDEX_TYPE == "hnsw":
        print("HNSW M:", HNSW_M)
        print("HNSW efSearch:", index.hnsw.efSearch)
        print("FAISS CPU threads:", FAISS_THREADS)

    return model, index, chunks


def encode_questions(model, questions):
    """Query Prefix + 원본 임베딩 후, PCA 사용 시 문서와 동일한 변환을 적용."""
    prepared_questions = [QUERY_PREFIX + question for question in questions]
    vectors = model.encode(prepared_questions, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=NORMALIZE_EMBEDDINGS, show_progress_bar=False)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if PCA_ENABLED:
        # 동일한 PCA 평균과 주성분 적용. 실험과 같이 첫 192개 성분만 사용.
        pca = load_pca()
        vectors = np.ascontiguousarray(pca.transform(vectors)[:, :PCA_DIMENSION], dtype=np.float32)
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
            raise ValueError("PCA Query 벡터에 NaN/Inf 또는 영벡터가 있습니다.")
        
        faiss.normalize_L2(vectors)

    return vectors


def dense_search(model, index, chunks, question, top_k):
    query_vector = encode_questions(model, [question])
    scores, indices = index.search(query_vector, top_k)
    results = []

    for score, row_id in zip(scores[0], indices[0]):
        if row_id < 0:
            continue
        
        chunk = chunks[int(row_id)]
        results.append({**chunk, "row_id": int(row_id), "score": float(score)})

    return results


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
        content = hit.get("text") or hit.get("content") or hit.get("chunk_text", "")
        print(f"\n[{rank}] {filename} | cosine={hit['score']:.4f}")
        print(str(content)[:400].replace("\n", " "))


if __name__ == "__main__":
    main()
