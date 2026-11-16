"""Pure regression checks: run from scripts with python -B -m unittest evaluate.test_evaluation."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from . import common as c


class EvaluationTests(unittest.TestCase):
    def test_normalization_and_multiset_f1(self):
        self.assertEqual(c.exact_match("\uff21, Foo!", "a, foo!"), 1)
        self.assertEqual(c.token_f1("a a b", "a b b"), 2 / 3)
        self.assertEqual(c.token_f1("", ""), 1)
        self.assertEqual(c.token_f1("", "a"), 0)

    def test_generation_denominators_and_strict_abstention(self):
        records = [{"id": str(i), "question": "q", "answer": "yes", "is_impossible": impossible}
                   for i, impossible in enumerate((False, False, True, True))]
        predictions = [{"id": r["id"], "question": "q", "reference_answer": "yes",
                        "is_impossible": r["is_impossible"], "prediction": pred}
                       for r, pred in zip(records, ("yes", c.ABSTAIN, c.ABSTAIN, " " + c.ABSTAIN))]
        metrics = c.score_predictions(records, predictions, "clean", 1)
        for key in ("exact_match_percent", "token_f1_percent", "abstention_rate_percent", "false_refusal_rate_percent"):
            self.assertEqual(metrics[key], 50.0)
        self.assertIsNone(predictions[2]["token_f1"])
        predictions[1]["id"] = predictions[0]["id"]
        with self.assertRaises(ValueError):
            c.score_predictions(records, predictions, "clean", 1)

    def test_duplicate_documents_preserve_chunk_ranks(self):
        scores = c.retrieval_scores({"a", "b"}, ["x", "a", "a", "b"], 5)
        self.assertEqual(scores["rr"], 0.5)
        self.assertEqual(scores["recall"], 1)
        self.assertEqual(scores["precision"], 0.4)
        expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(scores["ndcg"], expected)
        self.assertEqual(c.retrieval_scores({"a"}, [], 5)["ndcg"], 0)
        with self.assertRaises(ValueError):
            c.retrieval_scores(set(), ["a"])

    def test_bm25_zero_score_placeholder_keeps_rank(self):
        scores = c.retrieval_scores({"gold"}, [None, "gold"])
        self.assertEqual(scores["rr"], 0.5)

    def test_confusion_matrix_and_empty_subset(self):
        rows = [{"predicted_answerable": pred, "label": label}
                for pred, label in ((True, True), (True, False), (False, True), (False, False))]
        result = c.measure(rows, "label")
        self.assertEqual([result[k] for k in ("TP", "FP", "FN", "TN")], [1, 1, 1, 1])
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(c.measure([], "label")["accuracy"], 0)

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metrics.json"
            c.save_json(path, {"value": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                c.save_json(path, {"value": 2})
            self.assertEqual(before, path.read_bytes())
        for directory in ("models", "data"):
            with self.assertRaises(ValueError):
                c.check_output(c.PROJECT_DIR / directory / "new_metrics.json")

    def test_saved_metrics_regression(self):
        dataset = c.DATA_DIR / "val_clean.json"
        if not dataset.exists():
            self.skipTest("Local evaluation artifacts unavailable")

        import json
        records = c.load_records()

        for name in ("base/v1", "rag/v2", "rag/v1"):
            pred = c.RESULT_DIR / name / "predictions.jsonl"
            metrics_path = c.RESULT_DIR / name / "metrics.json"

            if not pred.exists() or not metrics_path.exists():
                continue

            expected = json.loads(metrics_path.read_text(encoding="utf-8"))
            actual = c.score_predictions(records, copy.deepcopy(c.load_jsonl(pred)), "clean", expected["batch_size"])
            
            for key, value in actual.items():
                self.assertEqual(value, expected[key], (name, key))


if __name__ == "__main__":
    unittest.main()
