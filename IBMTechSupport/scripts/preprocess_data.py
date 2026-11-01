
import json
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

# 경로 설정
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"


def normalize_text(text):
    """Unicode 및 줄바꿈 정규화. 코드와 명령어의 공백은 보존."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def preprocess_qa(records):
    """질문/정답 데이터 검증 및 정제."""
    cleaned = []
    stats = Counter()
    seen_ids = set()
    seen_questions = set()

    for record in records:
        stats["total"] += 1

        if not isinstance(record, dict):
            stats["invalid"] += 1
            continue

        # 필수 필드 확인
        record_id = record.get("id")
        question = normalize_text(record.get("question"))
        answer = normalize_text(record.get("answer"))
        impossible = record.get("is_impossible")
        contexts = record.get("contexts")

        if (
            not isinstance(record_id, str)
            or not record_id.strip()
            or not question
            or not isinstance(impossible, bool)
            or not isinstance(contexts, list)
            or (not answer and not impossible)
        ):
            stats["invalid"] += 1
            continue

        record_id = record_id.strip()

        # 중복 ID는 제거
        if record_id in seen_ids:
            stats["duplicate_ids"] += 1
            continue

        seen_ids.add(record_id)

        # 중복 질문은 기록만 하고 유지
        # 4단계에서 Train/Val/Test 누출 방지에 사용
        question_key = " ".join(question.casefold().split())

        if question_key in seen_questions:
            stats["duplicate_questions"] += 1

        seen_questions.add(question_key)

        # 정답 근거 문서 정제
        clean_contexts = []

        for context in contexts:
            if not isinstance(context, dict):
                stats["invalid_contexts"] += 1
                continue

            filename = context.get("filename")
            text = context.get("text")

            if not isinstance(filename, str) or not isinstance(text, str):
                stats["invalid_contexts"] += 1
                continue

            filename = filename.strip()
            text = normalize_text(text)

            if not filename or not text:
                stats["invalid_contexts"] += 1
                continue

            clean_contexts.append({
                "filename": filename,
                "text": text
            })

        cleaned.append({
            "id": record_id,
            "question": question,
            "answer": answer,
            "is_impossible": impossible,
            "contexts": clean_contexts
        })

        if impossible:
            stats["impossible"] += 1
        else:
            stats["answerable"] += 1

    return cleaned, stats

    
def analyze_corpus(records):
    """ZIP 내부 기술 문서 및 정답 근거 참조 분석."""
    corpus_path = RAW_DIR / "corpus.zip"

    with zipfile.ZipFile(corpus_path) as archive:
        documents = [
            info for info in archive.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(".txt")
        ]

        # 실제 ZIP 문서 이름 확인
        corpus_names = {Path(info.filename).name for info in documents }

        # 내용이 비어 있는 문서 확인
        empty_documents = 0

        for info in documents:
            with archive.open(info) as file:
                content = file.read()

            if not content.strip():
                empty_documents += 1

    # 질문의 정답 근거 파일과 실제 문서 비교
    referenced_files = {
        context["filename"]
        for record in records
        for context in record["contexts"]
    }

    missing_files = referenced_files - corpus_names

    print("\n[Corpus Analysis]")
    print("TXT Documents:", len(documents))
    print("Empty Documents:", empty_documents)
    print("Referenced Documents:", len(referenced_files))
    print("Missing References:", len(missing_files))

    if missing_files:
        print("Missing Examples:", sorted(missing_files)[:5])


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 원본 JSON 로드
    input_path = RAW_DIR / "train.json"

    with open(input_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected a JSON list")

    # 2. 전처리
    cleaned, stats = preprocess_qa(records)

    if not cleaned:
        raise ValueError("No valid QA records found")

    # 3. 통계 출력
    print("[QA Analysis]")
    print("Original Records:", stats["total"])
    print("Clean Records:", len(cleaned))
    print("Invalid Records:", stats["invalid"])
    print("Duplicate IDs:", stats["duplicate_ids"])
    print("Duplicate Questions:", stats["duplicate_questions"])
    print("Invalid Contexts:", stats["invalid_contexts"])
    print("Answerable:", stats["answerable"])
    print("Impossible:", stats["impossible"])

    # 4. 텍스트 길이 분석
    question_lengths = [len(r["question"]) for r in cleaned]
    answer_lengths = [len(r["answer"]) for r in cleaned]

    print("\n[Text Length]")
    print("Avg Question Length:", round(sum(question_lengths) / len(question_lengths), 2))
    print("Avg Answer Length:", round(sum(answer_lengths) / len(answer_lengths), 2))

    # 5. 기술 문서 분석
    analyze_corpus(cleaned)

    # 6. 정제 데이터 저장
    output_path = PROCESSED_DIR / "qa_clean.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(cleaned, file, ensure_ascii=False, indent=2)

    print("\nSaved:", output_path)
    print("Preprocessing: OK")


if __name__ == "__main__":
    main()
