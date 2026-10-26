from pathlib import Path

# 1. 파일 경로 설정
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "raw" / "corpus.txt"
OUTPUT_DIR = ROOT / "data" / "processed"

print(ROOT)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성 (이미 존재하면 무시)


# 2. 원본 데이터 불러오기
text = INPUT_PATH.read_text(encoding="utf-8")
print("원본 문자 수:", len(text))


# 3. 데이터 가공
# BOM 제거
text = text.lstrip("\ufeff")

text = text.replace("\r\n", "\n").replace("\r", "\n")
text = text.replace("\x00", "")

assert len(text) > 0, "데이터가 비어있습니다."

print("가공 후 문자 수 :", len(text))
print("중복 제거 고유 문자 수 :", len(set(text)))


# 4. Train / Validation / Test (90% / 5% / 5%)
n = len(text)

train_end = int(n * 0.9)
val_end = int(n * 0.95)

train_text = text[:train_end]
val_text = text[train_end:val_end]
test_text = text[val_end:]


# 5. 데이터 저장
datasets = { "train" : train_text, "val" : val_text, "test" : test_text }

for name, content in datasets.items():
    path = OUTPUT_DIR / f"{name}.txt"
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"{name}: {len(content):,} characters")


# 6. 데이터 검증
assert all(len(content) > 0 for content in datasets.values())
assert train_text + val_text + test_text == text
print("Data Preprocessing: OK")