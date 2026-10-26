import json
from pathlib import Path


# 1. Character Tokenizer
class CharTokenizer:
    def __init__(self, text):
        # 학습 데이터에서 고유 문자 추출
        chars = sorted(set(text))

        # 문자 -> 정수 ID
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # 정수 ID -> 문자
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        chars = [self.itos[i] for i in range(self.vocab_size)]
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(chars, f, ensure_ascii=False, indent=2)

    # Vocabulary Load
    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            chars = json.load(f)

        return cls("".join(chars))


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    train_text = (ROOT / "data" / "processed" / "train.txt").read_text(encoding="utf-8")

    # Vocabulary 는 Train 데이터로만 생성
    tokenizer = CharTokenizer(train_text)

    vocab_path = ROOT / "data" / "tokenizer" / "vocab.json"
    tokenizer.save(vocab_path)

    tokenizer = CharTokenizer.load(vocab_path)
    sample = train_text[:50]

    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Vocabulary Size: ", tokenizer.vocab_size)
    print("Original: ", repr(sample))
    print("Encoded: ", encoded)
    print("Decoded: ", decoded)

    assert decoded == sample

    for split in ["val", "test"]:
        text = (ROOT / "data" / "processed" / f"{split}.txt").read_text(encoding="utf-8")

        unknown = set(text) - set(tokenizer.stoi)
        assert not unknown, f"{split}에 미등록 문자 종재: {unknown}"

        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text

        print(f"{split}: {len(ids):,} tokens")

    print("Tokenizer: OK")    