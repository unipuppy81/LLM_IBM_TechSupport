import json
import math
import re
import unicodedata

from collections import Counter
from pathlib import Path
from document_utils import chunk_filename, filename_key, normalize_filename

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RESULT_DIR = PROJECT_DIR / "evaluate_results"
MODEL_NAME = "Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 160
ABSTAIN = "INSUFFICIENT_INFORMATION"

def tokenize_for_metrics(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+|[^\w\s]", text)


def percentage(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else None


def exact_match(prediction, reference):
    return int(tokenize_for_metrics(prediction) == tokenize_for_metrics(reference))


def token_f1(prediction, reference):
    predicted = tokenize_for_metrics(prediction)
    correct = tokenize_for_metrics(reference)
    if not predicted or not correct:
        return float(predicted == correct)

    common = Counter(predicted) & Counter(correct)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(correct)
    return 2 * precision * recall / (precision + recall)


def validate_prediction_rows(records, predictions):
    if len({r["id"] for r in records}) != len(records) or len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Duplicate evaluation IDs")
    if len(records) != len(predictions):
        raise ValueError("예측 파일 레코드 수가 평가 데이터와 다릅니다.")

    for record, result in zip(records, predictions):
        expected = (
            record["id"],
            record["question"],
            record["answer"],
            record["is_impossible"],
        )
        actual = (
            result["id"],
            result["question"],
            result["reference_answer"],
            result["is_impossible"],
        )

        if actual != expected or not isinstance(result.get("prediction"), str):
            raise ValueError(f"평가 데이터와 예측 결과가 불일치: {record['id']}")


def score_predictions(records, predictions, dataset, batch_size):
    validate_prediction_rows(records, predictions)
    stats = Counter()
    total_f1 = 0.0

    for record, row in zip(records, predictions):
        prediction = row["prediction"]
        abstained = prediction == ABSTAIN
        row["abstained"] = abstained

        if record["is_impossible"]:
            stats["impossible"] += 1
            stats["correct_abstentions"] += int(abstained)
            row["exact_match"] = None
            row["token_f1"] = None
        else:
            stats["answerable"] += 1
            em = exact_match(prediction, record["answer"])
            f1 = token_f1(prediction, record["answer"])
            stats["exact_matches"] += em
            stats["false_refusals"] += int(abstained)
            total_f1 += f1
            row["exact_match"] = em
            row["token_f1"] = round(f1, 4)

    return {
        "model": MODEL_NAME,
        "dataset": dataset,
        "total": len(records),
        "batch_size": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "answerable": stats["answerable"],
        "impossible": stats["impossible"],
        "exact_match_percent": percentage(stats["exact_matches"], stats["answerable"]),
        "token_f1_percent": percentage(total_f1, stats["answerable"]),
        "abstention_rate_percent": percentage(stats["correct_abstentions"], stats["impossible"]),
        "false_refusal_rate_percent": percentage(stats["false_refusals"], stats["answerable"]),
    }


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def measure(rows, label_key):
    counts = Counter()
    for row in rows:
        predicted = row["predicted_answerable"]
        expected = row[label_key]
        if not isinstance(predicted, bool) or not isinstance(expected, bool):
            raise ValueError(f"Boolean 아닌 예측/레이블: {row['id']}")
        if predicted and expected:
            counts["TP"] += 1
        elif predicted and not expected:
            counts["FP"] += 1
        elif not predicted and expected:
            counts["FN"] += 1
        else:
            counts["TN"] += 1
    tp, fp, tn, fn = (counts[k] for k in ("TP", "FP", "TN", "FN"))
    total = tp + fp + tn + fn
    return {
        "total": total,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round((tp + tn) / total, 4) if total else 0,
        "precision": round(tp / (tp + fp), 4) if tp + fp else 0,
        "recall": round(tp / (tp + fn), 4) if tp + fn else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else 0,
    }


def load_records(dataset="clean", limit=0):
    if limit < 0:
        raise ValueError("limit must be >= 0")
    name = {"clean": "val_clean.json", "original": "val.json"}.get(dataset)
    path = DATA_DIR / name if name else Path(dataset)
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("Empty evaluation dataset")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate dataset IDs")
    return records


def check_output(path):
    if path is None:
        return
    path = Path(path).resolve()
    # Evaluation output must never be written into model or dataset directories.
    for root in (PROJECT_DIR / "models", PROJECT_DIR / "data"):
        if path.is_relative_to(root.resolve()):
            raise ValueError("Output cannot be inside models/ or data/")
    if path.exists():
        raise FileExistsError(f"Existing output is protected: {path}")


def save_json(path, value):
    check_output(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def save_jsonl(path, rows):
    check_output(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report(metrics, output=None):
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if output is not None:
        save_json(output, metrics)
    return metrics



def retrieval_scores(gold, ranked, k=5):
    if not gold or k < 1:
        raise ValueError("Gold documents and positive K required")
    found = gold.intersection(ranked[:k])
    seen = set()
    dcg = 0.0
    for rank, name in enumerate(ranked[:k], 1):
        if name in gold and name not in seen:
            dcg += 1 / math.log2(rank + 1)

        seen.add(name)

    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))

    return {"hit": float(bool(found)), "recall": len(found) / len(gold),
            "rr": next((1 / rank for rank, name in enumerate(ranked[:10], 1) if name in gold), 0.0),
            "precision": len(found) / k, "ndcg": dcg / ideal}


def measure_retrieval(rows, records, chunks):
    # Exact legacy Hit@5 / Recall@5 / MRR@10 aggregation.
    if len(rows) != len(records) or not records:
        raise ValueError("Nonempty, aligned retrieval rows required")
    
    totals = Counter()
    for row_ids, record in zip(rows, records):
        gold = {filename_key(c["filename"]) for c in record["contexts"]}
        ranked = [filename_key(chunk_filename(chunks[i])) for i in row_ids[:10]]
        scores = retrieval_scores(gold, ranked)
        for key in ("hit", "recall", "rr"):
            totals[key] += scores[key]

    n = len(records)

    return {"hit_at_5_percent": round(totals["hit"] / n * 100, 2),
            "recall_at_5_percent": round(totals["recall"] / n * 100, 2),
            "mrr_at_10": round(totals["rr"] / n, 4)}

def get_gold_documents(record):
    """정답 근거 문서 파일명 추출."""
    return {
        normalize_filename(context["filename"])
        for context in record["contexts"]
        if isinstance(context, dict)
        and isinstance(context.get("filename"), str)
        and context["filename"].strip()
    }