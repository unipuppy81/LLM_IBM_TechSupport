import argparse
import json
import re

from pathlib import Path
from statistics import mean
from document_utils import normalize_filename
from retrieval_config import load_retrieval_config

import bm25s
from bm25s.tokenization import Tokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
CORPUS_PATH = DATA_DIR / "corpus_chunks.jsonl"
INDEX_DIR = DATA_DIR / "bm25_index"

# k1: 단어 빈도 영향력 조절 : 문서에 검색 단어가 반복해서 등장할 때 점수가 얼마나 증가할지 결정
# B : 문서 길이 보정 : 0(문서 길이 보정 X), 1(문서 길이 완전히 반영해 보정)
# YAML에서 BM25 설정 불러오기
BM25_CONFIG = load_retrieval_config()["bm25"]

METHOD = BM25_CONFIG["method"]
K1 = BM25_CONFIG["k1"]
B = BM25_CONFIG["b"]

TOKEN_PATTERN = re.compile(BM25_CONFIG["token_pattern"])

def split_tokens(text):
    # 1. 기존 방식으로 토큰 추출
    tokens = TOKEN_PATTERN.findall(text.casefold())

    expanded_tokens = []

    for token in tokens:
        # 2. 원본 토큰 유지
        expanded_tokens.append(token)

        # 3. 복합 토큰을 구분자로 분리
        if any(separator in token for separator in "._-"):
            parts = re.split(r"[._-]+", token)

            for part in parts:
                # 길이 3 이상이며 영문자가 포함된 토큰만 추가
                if len(part) >= 3 and any(c.isalpha() for c in part):
                    expanded_tokens.append(part)

    return expanded_tokens


def create_tokenizer():
  return Tokenizer(splitter=split_tokens, stopwords=BM25_CONFIG["stopwords"])


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
    retriever = bm25s.BM25(method=METHOD, k1=K1, b=B)

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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "search"))
    parser.add_argument("--query", type=str)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "build":
        build_index()

    elif args.command == "search":
        if not args.query:
            parser.error("--query가 필요합니다.")

        if args.top_k < 1:
            parser.error("--top-k는 1 이상이어야 합니다.")

        search(args.query, args.top_k)



if __name__ == "__main__":
    main()
