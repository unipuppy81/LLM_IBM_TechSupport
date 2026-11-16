"""Score saved Base/RAG predictions without loading models."""
import argparse
from pathlib import Path
from .common import check_output, load_jsonl, load_records, report, score_predictions


def evaluate(predictions, dataset="clean", batch_size=1, limit=0, output=None):
    check_output(output)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    records = load_records(dataset, limit)
    rows = load_jsonl(predictions)
    return report(score_predictions(records, rows, dataset, batch_size), output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--dataset", default="clean", help="clean, original, or dataset JSON path")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    return evaluate(a.predictions, a.dataset, a.batch_size, a.limit, a.output)


if __name__ == "__main__":
    main()
