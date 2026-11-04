import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean

import bm25s
from bm25s.tokenization import Tokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
CORPUS_PATH = DATA_DIR / "corpus_chunks.jsonl"
INDEX_DIR = DATA_DIR / "bm25_index"
VAL_PATH = DATA_DIR / "val_clean.json"

RESULT_DIR = PROJECT_DIR / "results"
COMPARISON_PATH = RESULT_DIR / "retrieval_comparison.json"

# k1: 단어 빈도 영향력 조절 : 문서에 검색 단어가 반복해서 등장할 때 점수가 얼마나 증가할지 결정
# B : 문서 길이 보정 : 0(문서 길이 보정 X), 1(문서 길이 완전히 반영해 보정)
K1 = 1.5
B = 0.75

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")

def split_tokens(text):
    return TOKEN_PATTERN.findall(text.casefold())


def create_tokenizer():
    """기존 BM25 토크나이저."""
    return Tokenizer(splitter=split_tokens, stopwords=[])


def load_corpus():
    """Chunk 데이터 로드."""
    records = []
    seen_ids = set()

    with open(CORPUS_PATH, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["chunk_id"] in seen_ids:
                raise ValueError("Duplicate chunk ID")

            seen_ids.add(record["chunk_id"])
            records.append(record)

    if not records:
        raise ValueError("Empty corpus")

    return records


def build_index():
    """BM25 인덱스 생성 및 저장."""
    records = load_corpus()

    print("[BM25 Index Build]")
    print("Total Chunks:", len(records))

    tokenizer = create_tokenizer()
    corpus_tokens = tokenizer.tokenize([record["text"] for record in records], return_as="tuple")
    retriever = bm25s.BM25(method="lucene", k1=K1, b=B)

    retriever.index(corpus_tokens)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    retriever.save(str(INDEX_DIR), corpus=records)
    tokenizer.save_vocab(save_dir=str(INDEX_DIR))

    print("BM25 Index: OK")


def load_index():
    """저장된 인덱스와 토크나이저 복원."""
    if not INDEX_DIR.exists():
        raise FileNotFoundError("Build BM25 index first")

    retriever = bm25s.BM25.load(str(INDEX_DIR), load_corpus=True, mmap=True)

    tokenizer = create_tokenizer()
    tokenizer.load_vocab(str(INDEX_DIR))

    return retriever, tokenizer


def retrieve(query, retriever, tokenizer, top_k):
    """질문에 대한 상위 K개 Chunk 검색."""
    query_tokens = tokenizer.tokenize([query], update_vocab=False, return_as="tuple")
    documents, scores = retriever.retrieve(query_tokens, k=top_k)

    return [
        {
            "chunk_id": documents[0, i]["chunk_id"],
            "filename": documents[0, i]["filename"],
            "text": documents[0, i]["text"],
            "score": float(scores[0, i])
        }
        for i in range(top_k)
    ]


def search(query, top_k):
    """단일 질문 검색."""
    retriever, tokenizer = load_index()
    results = retrieve(query, retriever, tokenizer, top_k)

    print("\n[BM25 Search]")
    print("Query:", query)

    for rank, result in enumerate(results, start=1):
        print(f"\n[Rank {rank}]")
        print("Score:", round(result["score"], 4))
        print("Chunk ID:", result["chunk_id"])
        print("Filename:", result["filename"])
        print("Text:", result["text"][:400])

    print("\nBM25 Search: OK")


def normalize_filename(filename):
    """근거 문서와 검색 문서의 파일명 비교."""
    return Path(filename.replace("\\", "/")).name.casefold()


def get_gold_documents(record):
    """정답 근거 문서 파일명 추출."""
    return {
        normalize_filename(context["filename"])
        for context in record["contexts"]
        if isinstance(context, dict)
        and isinstance(context.get("filename"), str)
        and context["filename"].strip()
    }


def evaluate():
    """Clean Validation에서 검색 성능 평가."""
    with open(VAL_PATH, encoding="utf-8") as file:
        records = json.load(file)

    retriever, tokenizer = load_index()

    hit5_total = 0
    recall5_total = 0.0
    mrr10_total = 0.0

    skipped_impossible = 0
    skipped_no_gold = 0

    latencies = []
    details = []

    print("[BM25 Retrieval Evaluation]")
    print("Validation Records:", len(records))

    for record in records:
        if record["is_impossible"]:
            skipped_impossible += 1
            continue

        gold = get_gold_documents(record)

        if not gold:
            skipped_no_gold += 1
            continue

        # 토큰화 + BM25 검색 시간 측정
        start = time.perf_counter()
        results = retrieve(record["question"], retriever, tokenizer, top_k=10)
        latency = (time.perf_counter() - start) * 1000

        latencies.append(latency)

        # 점수가 0보다 큰 실제 검색 결과만 사용
        ranked_files = [
            normalize_filename(result["filename"])
            if result["score"] > 0 else None
            for result in results
        ]

        # Hit@5
        found5 = {
            filename
            for filename in ranked_files[:5]
            if filename in gold
        }

        hit5 = int(bool(found5))

        # Recall@5: 정답 근거 문서 단위로 계산
        recall5 = len(found5) / len(gold)

        # MRR@10: 최초 정답 근거의 역순위
        reciprocal_rank = 0.0

        for rank, filename in enumerate(ranked_files[:10], start=1):
            if filename in gold:
                reciprocal_rank = 1.0 / rank
                break

        hit5_total += hit5
        recall5_total += recall5
        mrr10_total += reciprocal_rank

        details.append({
            "id": record["id"],
            "gold_filenames": sorted(gold),
            "retrieved": [
                {
                    "rank": i + 1,
                    "chunk_id": result["chunk_id"],
                    "filename": result["filename"],
                    "score": round(result["score"], 4)
                }
                for i, result in enumerate(results)
            ],
            "hit_at_5": hit5,
            "recall_at_5": round(recall5, 4),
            "reciprocal_rank_at_10": round(
                reciprocal_rank, 4
            ),
            "latency_ms": round(latency, 3)
        })

    total = len(details)

    if total == 0:
        raise ValueError("No evaluable questions")

    # 95백분위 지연 시간 계산
    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(0.95 * total + 0.999999) - 1)

    metrics = {
        "dataset": "val_clean.json",
        "total_records": len(records),
        "evaluated_questions": total,
        "skipped_impossible": skipped_impossible,
        "skipped_no_gold": skipped_no_gold,
        "hit_at_5": round(hit5_total / total * 100, 2),
        "recall_at_5": round(recall5_total / total * 100, 2),
        "mrr_at_10": round(mrr10_total / total, 4),
        "avg_latency_ms": round(mean(latencies), 3),
        "p95_latency_ms": round(sorted_latencies[p95_index], 3)
    }

    # 이후 검색 방식과 비교할 수 있도록 누적 저장
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if COMPARISON_PATH.exists():
        with open(COMPARISON_PATH, encoding="utf-8") as file:
            comparison = json.load(file)
    else:
        comparison = {}

    comparison["bm25"] = {
        "metrics": metrics,
        "details": details
    }

    temp_path = COMPARISON_PATH.with_suffix(".json.tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)

    temp_path.replace(COMPARISON_PATH)
    print("\n[BM25 Evaluation Result]")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Saved:", COMPARISON_PATH)
    print("BM25 Evaluation: OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "search", "evaluate"])
    parser.add_argument("--query", type=str)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "build":
        build_index()

    elif args.command == "search":
        if not args.query:
            parser.error("--query is required")

        if args.top_k < 1:
            parser.error("--top-k must be >= 1")

        search(args.query, args.top_k)

    elif args.command == "evaluate":
        evaluate()


if __name__ == "__main__":
    main()
