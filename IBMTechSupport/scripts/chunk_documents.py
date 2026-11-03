
import json
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "raw" / "corpus.zip"
OUTPUT_PATH = (
    PROJECT_DIR / "data" / "processed" / "corpus_chunks.jsonl"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


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
                position = text.rfind(
                    separator, min_end, end
                )

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


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = OUTPUT_PATH.with_suffix(".jsonl.part")

    document_count = 0
    chunk_count = 0
    fallback_count = 0
    total_length = 0
    min_length = CHUNK_SIZE
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

            with open(
                temp_path, "w", encoding="utf-8"
            ) as output:

                for doc_index, info in enumerate(documents):
                    raw = archive.read(info)

                    # 텍스트 디코딩
                    try:
                        text = raw.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        text = raw.decode(
                            "cp1252", errors="replace"
                        )
                        fallback_count += 1

                    text = normalize_text(text)

                    if not text:
                        continue

                    document_id = f"doc_{doc_index:05d}"
                    filename = PurePosixPath(
                        info.filename.replace("\\", "/")
                    ).name

                    previous_end = None

                    for index, (start, end, chunk) in enumerate(
                        split_document(text)
                    ):
                        # Chunk 길이 및 위치 검증
                        assert len(chunk) <= CHUNK_SIZE
                        assert chunk == text[start:end]

                        if previous_end is not None:
                            assert start == (
                                previous_end - CHUNK_OVERLAP
                            )

                        record = {
                            "chunk_id": (
                                f"{document_id}_chunk_{index:04d}"
                            ),
                            "document_id": document_id,
                            "filename": filename,
                            "source_path": info.filename,
                            "chunk_index": index,
                            "start_char": start,
                            "end_char": end,
                            "text": chunk
                        }

                        output.write(
                            json.dumps(
                                record, ensure_ascii=False
                            ) + "\n"
                        )

                        chunk_count += 1
                        total_length += len(chunk)
                        min_length = min(
                            min_length, len(chunk)
                        )
                        max_length = max(
                            max_length, len(chunk)
                        )
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
    print(
        "Avg Chunk Length:",
        round(total_length / chunk_count, 2)
    )
    print("Min Chunk Length:", min_length)
    print("Max Chunk Length:", max_length)
    print("Fallback Decoded:", fallback_count)

    print("Saved:", OUTPUT_PATH)
    print("Document Chunking: OK")


if __name__ == "__main__":
    main()
