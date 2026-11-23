"""Offline unit tests; no GPU or Hugging Face weights needed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IBMTechSupport/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IBMTechSupport/scripts/debug"))

import rag_guardrails_relaxed as relaxed
import apply_relaxed_judge as installer


class RelaxedJudgeTest(unittest.TestCase):
    def test_documents_missing(self):
        self.assertEqual(relaxed.judge_evidence_relaxed(None, None, "question", [])["decision"], "NO")

    def test_same_input_template_and_model_call(self):
        docs = [{"filename": "a.txt", "text": "IBM solution: change setting X."}]
        with patch.object(relaxed, "_llm_decision", return_value={"decision": "YES", "raw": "YES", "seconds": 0.1}) as spy:
            result = relaxed.judge_evidence_relaxed(None, None, "How to fix?", docs)
        self.assertEqual(result["decision"], "YES")
        args = spy.call_args.args
        self.assertEqual(args[2], relaxed.RELAXED_ANSWERABILITY_SYSTEM)
        self.assertIn("How to fix?", args[3])
        self.assertIn("IBM solution: change setting X.", args[3])
        self.assertNotIn("is_impossible", args[3])

    def test_relaxed_runner_keeps_verifier(self):
        with patch.object(relaxed, "judge_evidence_relaxed", return_value={"decision": "YES", "raw": "YES", "seconds": 0.1}):
            result = relaxed.answer_with_relaxed_guardrails(
                question="How?", retrieved_documents=[{"filename": "doc.txt", "text": "solution"}],
                model=None, tokenizer=None, generation_config=None, prompt_config={},
                generate_answer=lambda *args: ("Do X.", 1.0, 100),
                top_k=1, retry_top_k=1,
                verify_fn=lambda *args: {"decision": "NO", "raw": "NO", "seconds": 0.1},
            )
        self.assertEqual(result["status"], "ANSWER_NO")
        self.assertEqual(result["answer"], relaxed.ABSTAIN)

    def test_prompt_does_not_allow_keyword_only_match(self):
        self.assertIn("Keyword similarity alone", relaxed.RELAXED_ANSWERABILITY_SYSTEM)
        self.assertIn("Do not invent facts", relaxed.RELAXED_ANSWERABILITY_SYSTEM)


class PatchTest(unittest.TestCase):
    def test_eval_patch_routes_by_config(self):
        original = (
            'from rag_guardrails import answer_with_guardrails\n'
            'def create_predictions(records, config):\n'
            '    guardrails = config.get("guardrails", {})\n'
            '    if True:\n'
            '            guarded_result = answer_with_guardrails(\n'
            '                some_value=42)\n'
        )
        patched = installer.patch_evaluate(original)
        self.assertIn("judge_mode", patched)
        self.assertIn("guardrail_runner(", patched)
        compile(patched, "evaluate_rag.py", "exec")

    def test_pipeline_patch_routes_by_config(self):
        original = (
            'from pathlib import Path\n'
            'from rag_guardrails import answer_with_guardrails\n'
            'def run_rag(question=None, method=None, top_k=None, config=None):\n'
            '    if True:\n'
            '        guarded_result = answer_with_guardrails(\n'
            '            question=question)\n'
            'def main():\n'
            '    parser.add_argument("--top-k", type=int, default=TOP_K)\n'
            '    run_rag(question=args.question, method=args.retriever, top_k=args.top_k)\n'
        )
        patched = installer.patch_pipeline(original)
        self.assertIn("judge_mode", patched)
        self.assertIn('"--config"', patched)
        self.assertIn("config_path=args.config", patched)
        compile(patched, "rag_pipeline.py", "exec")


if __name__ == "__main__":
    unittest.main()
