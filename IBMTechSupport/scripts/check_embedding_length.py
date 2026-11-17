import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from retrieval_config import load_retrieval_config

PROJECT_DIR = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_DIR / "data" / "processed" / "corpus_chunks.jsonl"

DENSE_CONFIG = load_retrieval_config()["dense"]
MODEL_DIR = PROJECT_DIR / "models" / DENSE_CONFIG["model"].split("/")[-1]

model = SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True)
max_length = DENSE_CONFIG["max_seq_length"]
tokenizer = model.tokenizer

total = 0
exceeded = 0
max_tokens = 0

with CORPUS_PATH.open(encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        chunk = json.loads(line)

        tokens = tokenizer.encode(chunk["text"], add_special_tokens=True, truncation=False)
        token_count = len(tokens)

        total += 1
        exceeded += token_count > max_length
        max_tokens = max(max_tokens, token_count)

print("Max Sequence Length:", max_length)
print("Total Chunks:", total)
print("Exceeded Chunks:", exceeded)
print("Exceeded Percent:", round(exceeded / total * 100, 2))
print("Maximum Tokens:", max_tokens)