import json
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath
from retrieval_config import load_retrieval_config
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "raw" / "corpus.zip"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "corpus_chunks.jsonl"

CONFIG = load_retrieval_config()

CHUNK_CONFIG = CONFIG["chunking"]
DENSE_CONFIG = CONFIG["dense"]

CHUNK_MODE = CHUNK_CONFIG.get("mode", "char")
CHUNK_SIZE = CHUNK_CONFIG["size"]
CHUNK_OVERLAP = CHUNK_CONFIG["overlap"]

if CHUNK_MODE not in ("char", "token"):
    raise ValueError("지원하지 않는 Chunking 방식입니다.")

if CHUNK_SIZE <= 0 or not 0 <= CHUNK_OVERLAP < int(CHUNK_SIZE * 0.75):
    raise ValueError("Chunk Size 또는 Overlap 설정이 올바르지 않습니다.")


"""
# 1. 청크 재생성
python IBMTechSupport/scripts/chunk_documents.py

# 2. 문서 임베딩 재생성
python IBMTechSupport/scripts/embed_documents.py --batch-size 16

# 3. FAISS 인덱스 재생성
python IBMTechSupport/scripts/build_faiss_index.py

# 4. BM25 인덱스 재생성
python IBMTechSupport/scripts/bm25_search.py build

# 평가
python -m evaluate.evaluate_retrieval --retriever bm25 --dataset clean --top-k 5
python -m evaluate.evaluate_retrieval --retriever dense --dataset clean --top-k 5
python -m evaluate.evaluate_retrieval --retriever hybrid --dataset clean --top-k 5
python -m evaluate.evaluate_retrieval --retriever reranker --dataset clean --top-k 5
"""


def normalize_text(text):
    """Unicode 및 줄바꿈 정규화."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def split_document(text):
    """문서를 Overlap이 있는 Chunk로 분할."""
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # 가능하면 문단 / 문장 / 단어 경계에서 분할
        if end < len(text):
            min_end = start + int(CHUNK_SIZE * 0.75)

            for separator in ("\n\n", "\n", ". ", " "):
                position = text.rfind(separator, min_end, end)

                if position != -1:
                    end = position + len(separator)
                    break

        chunk = text[start:end]

        if chunk.strip():
            yield start, end, chunk

        if end == len(text):
            break

        # 이전 Chunk의 마지막 150자를 중복 포함
        start = end - CHUNK_OVERLAP

def split_document_tokens(text, tokenizer):
    """Tokenizer를 이용한 토큰 기반 Chunking."""

    # 원본 문서를 토큰화하고 각 토큰의 문자 위치를 가져옴
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)

    offsets = encoded["offset_mapping"]
    total_tokens = len(offsets)

    token_start = 0

    while token_start < total_tokens:
        token_end = min(token_start + CHUNK_SIZE, total_tokens)

        # 원본 문서에서 시작 문자 위치 계산
        start = (
            0 
            if token_start == 0
            else offsets[token_start][0]
        )

        while True:
            # 다음 청크 시작 위치 또는 문서 끝을 경계로 사용
            end = ( 
                len(text)  
                if token_end == total_tokens 
                else offsets[token_end][0] 
                )

            chunk = text[start:end]

            # 잘라낸 문자열을 다시 토큰화해 실제 입력 길이 검증
            content_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            input_tokens = len(tokenizer.encode(chunk, add_special_tokens=True))

            if (content_tokens <= CHUNK_SIZE and input_tokens <= DENSE_CONFIG["max_seq_length"]):
                break

            # 재토큰화로 길이가 초과하면 청크 끝을 축소
            token_end -= 1

            if token_end <= token_start:
                raise ValueError("토큰 길이에 맞는 Chunk 생성 실패")

        if chunk.strip():
            yield start, end, chunk

        if token_end == total_tokens:
            break

        # 이전 청크의 마지막 토큰들을 중복 포함
        token_start = max(token_start + 1, token_end - CHUNK_OVERLAP)

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    tokenizer = None

    if CHUNK_MODE == "token":
        model_name = DENSE_CONFIG["model"].split("/")[-1]
        model_dir = PROJECT_DIR / "models" / model_name

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True)

        if not tokenizer.is_fast:
            raise ValueError("Fast Tokenizer가 필요합니다.")

        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

        if CHUNK_SIZE + special_tokens > DENSE_CONFIG["max_seq_length"]:
            raise ValueError("Chunk Size가 모델 입력 길이를 초과합니다.")


    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = OUTPUT_PATH.with_suffix(".jsonl.part")

    document_count = 0
    chunk_count = 0
    fallback_count = 0
    total_length = 0
    min_length = float("inf")
    max_length = 0

    try:
        with zipfile.ZipFile(INPUT_PATH) as archive:
            documents = sorted(
                [
                    info for info in archive.infolist()
                    if not info.is_dir()
                    and info.filename.lower().endswith(".txt")
                ],
                key=lambda info: info.filename
            )

            print("Source Documents:", len(documents))

            with open(temp_path, "w", encoding="utf-8") as output:

                for doc_index, info in enumerate(documents):
                    raw = archive.read(info)

                    # 텍스트 디코딩
                    try:
                        text = raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        text = raw.decode("cp1252", errors="replace")
                        fallback_count += 1

                    text = normalize_text(text)

                    if not text:
                        continue

                    document_id = f"doc_{doc_index:05d}"
                    filename = PurePosixPath(info.filename.replace("\\", "/")).name

                    previous_end = None

                    chunks = (
                         split_document_tokens(text, tokenizer)
                         if CHUNK_MODE == "token"
                         else split_document(text)
                         )

                    for index, (start, end, chunk) in enumerate(chunks):
                        # Chunk 길이 및 위치 검증
                        assert chunk == text[start:end]

                        if CHUNK_MODE == "char":
                            assert len(chunk) <= CHUNK_SIZE

                            if previous_end is not None:
                                assert start == previous_end - CHUNK_OVERLAP

                        else:
                            assert len(tokenizer.encode(chunk, add_special_tokens=True)) <= DENSE_CONFIG["max_seq_length"]

                            if previous_end is not None:
                                assert start <= previous_end

                        record = {
                            "chunk_id": (f"{document_id}_chunk_{index:04d}"),
                            "document_id": document_id,
                            "filename": filename,
                            "source_path": info.filename,
                            "chunk_index": index,
                            "start_char": start,
                            "end_char": end,
                            "text": chunk
                        }

                        output.write(json.dumps(record, ensure_ascii=False) + "\n")

                        chunk_count += 1
                        total_length += len(chunk)
                        min_length = min(min_length, len(chunk))
                        max_length = max(max_length, len(chunk))
                        previous_end = end

                    assert previous_end == len(text)

                    document_count += 1

                    if document_count % 5000 == 0:
                        print(
                            f"Processed: {document_count}"
                            f"/{len(documents)}"
                        )

        if chunk_count == 0:
            raise ValueError("No chunks created")

        temp_path.replace(OUTPUT_PATH)

    finally:
        temp_path.unlink(missing_ok=True)

    print("\n[Chunking Result]")
    print("Processed Documents:", document_count)
    print("Total Chunks:", chunk_count)
    print("Avg Chunk Length:", round(total_length / chunk_count, 2))
    print("Min Chunk Length:", min_length)
    print("Max Chunk Length:", max_length)
    print("Fallback Decoded:", fallback_count)

    print("Saved:", OUTPUT_PATH)
    print("Document Chunking: OK")


if __name__ == "__main__":
    main()
