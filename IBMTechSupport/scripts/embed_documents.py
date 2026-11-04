import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "results"

CORPUS_PATH = DATA_DIR / "corpus_chunks.jsonl"
MODEL_DIR = PROJECT_DIR / "models" / "all-MiniLM-L6-v2"

EMBEDDING_PATH = DATA_DIR / "corpus_embeddings.npy"
COMPARISON_PATH = RESULT_DIR / "retrieval_comparison.json"

BATCH_SIZE = 64


def inspect_corpus():
    """전체 Chunk 개수와 원본 파일 해시 확인."""
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(CORPUS_PATH)

    count = 0
    digest = hashlib.sha256()

    with open(CORPUS_PATH, "rb") as file:
        for line in file:
            digest.update(line)

            if line.strip():
                count += 1

    if count == 0:
        raise ValueError("Empty corpus")

    return count, digest.hexdigest()


def encode_batch(model, texts, output, start):
    """Batch 단위 Embedding 생성 및 저장."""
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # 임베딩 차원 검증
    if embeddings.shape != (len(texts), output.shape[1]):
        raise ValueError("Invalid embedding shape")

    # NaN / Infinity 검사
    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values")

    # L2 정규화 여부 검증
    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embedding normalization failed")

    end = start + len(texts)

    # 파일의 해당 위치에 벡터 기록
    output[start:end] = embeddings

    return end


def build_embeddings(batch_size):
    """IBM 기술 문서 전체 Embedding 생성."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(MODEL_DIR)

    total_chunks, corpus_hash = inspect_corpus()

    print("[Embedding Build]")
    print("Total Chunks:", total_chunks)

    model = SentenceTransformer(str(MODEL_DIR), device="cuda", local_files_only=True)

    dimension = model.get_sentence_embedding_dimension()

    if dimension is None:
        raise ValueError("Unknown embedding dimension")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_DIR.name)
    print("Embedding Dimension:", dimension)
    print("Max Sequence Length:", model.max_seq_length)
    print("Batch Size:", batch_size)

    EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = EMBEDDING_PATH.with_suffix(".npy.part")

    output = None
    processed = 0

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    try:
        # 전체 벡터를 RAM에 보관하지 않고 파일에 직접 기록
        output = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=(total_chunks, dimension))
        texts = []

        with open(CORPUS_PATH, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                text = record["text"]

                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Invalid chunk text")

                texts.append(text)

                # Batch가 채워지면 GPU에서 병렬 처리
                if len(texts) == batch_size:
                    processed = encode_batch(model, texts, output, processed)
                    texts.clear()

                    if processed % (batch_size * 100) == 0:
                        print(f"Embedded: {processed}/{total_chunks}", flush=True)

        # 마지막 남은 Batch 처리
        if texts:
            processed = encode_batch(model, texts, output, processed)

        # 전체 Chunk 처리 검증
        if processed != total_chunks:
            raise ValueError("Chunk count mismatch")

        output.flush()
        del output
        output = None

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        peak_memory = (torch.cuda.max_memory_allocated() / 1024**3)

        temp_path.replace(EMBEDDING_PATH)

    finally:
        if output is not None:
            output.flush()
            del output

        temp_path.unlink(missing_ok=True)

    # 저장 파일 검증
    saved = np.load(EMBEDDING_PATH, mmap_mode="r")

    if saved.shape != (total_chunks, dimension):
        raise ValueError("Saved embedding shape mismatch")

    # 성능 기록
    metrics = {
        "model": MODEL_DIR.name,
        "total_chunks": total_chunks,
        "dimension": dimension,
        "batch_size": batch_size,
        "max_sequence_length": model.max_seq_length,
        "corpus_sha256": corpus_hash,
        "embedding_seconds": round(elapsed, 3),
        "chunks_per_second": round(total_chunks / elapsed, 3),
        "peak_gpu_memory_gb": round(peak_memory, 3),
        "embedding_file": EMBEDDING_PATH.name
    }

    # 기존 BM25 평가 결과를 보존
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if COMPARISON_PATH.exists():
        with open(COMPARISON_PATH, encoding="utf-8") as file:
            comparison = json.load(file)
    else:
        comparison = {}

    comparison["embedding_build"] = metrics
    temp_metrics = COMPARISON_PATH.with_suffix(".json.tmp")

    with open(temp_metrics, "w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)

    temp_metrics.replace(COMPARISON_PATH)

    print("\n[Embedding Result]")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Saved:", EMBEDDING_PATH)
    print("Embedding Build: OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    build_embeddings(args.batch_size)


if __name__ == "__main__":
    main()
