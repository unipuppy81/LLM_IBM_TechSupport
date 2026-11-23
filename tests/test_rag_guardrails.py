import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IBMTechSupport" / "scripts"))
from rag_guardrails import ABSTAIN, _parse_decision, answer_with_guardrails


def judge_sequence(*decisions):
    decisions = list(decisions)
    def fn(*args):
        return {"decision": decisions.pop(0), "raw": "", "seconds": 0.1}
    return fn


def generator(*args):
    return "Run the IBM service restart command.", 2.5, 700


DOCS = [{"filename": f"doc{i}.txt", "text": f"Evidence {i}"} for i in range(10)]
BASE = dict(question="How to fix error?", retrieved_documents=DOCS, model=None,
            tokenizer=None, generation_config=None, prompt_config={},
            generate_answer=generator, top_k=5, retry_top_k=10)


class TestGuardrails(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(_parse_decision("YES\n"), "YES")
        self.assertEqual(_parse_decision("No."), "NO")
        self.assertEqual(_parse_decision("Probably yes"), "UNCERTAIN")

    def test_no_documents(self):
        result = answer_with_guardrails(**{**BASE, "retrieved_documents": []})
        self.assertEqual(result["status"], "NO_DOCUMENTS")
        self.assertEqual(result["answer"], ABSTAIN)

    def test_first_pass_yes(self):
        result = answer_with_guardrails(**BASE, judge_fn=judge_sequence("YES"), verify_fn=judge_sequence("YES"))
        self.assertEqual(result["status"], "ANSWER_VERIFIED")
        self.assertEqual(len(result["judge_history"]), 1)
        self.assertEqual(len(result["used_documents"]), 5)

    def test_retrieval_expansion(self):
        result = answer_with_guardrails(**BASE, judge_fn=judge_sequence("NO", "YES"), verify_fn=judge_sequence("YES"))
        self.assertEqual(result["status"], "ANSWER_VERIFIED")
        self.assertEqual(len(result["judge_history"]), 2)
        self.assertEqual(len(result["used_documents"]), 10)

    def test_no_evidence(self):
        result = answer_with_guardrails(**BASE, judge_fn=judge_sequence("NO", "NO"))
        self.assertEqual(result["answer"], ABSTAIN)
        self.assertEqual(result["status"], "EVIDENCE_NO")

    def test_uncertain_evidence(self):
        result = answer_with_guardrails(**BASE, judge_fn=judge_sequence("UNCERTAIN", "UNCERTAIN"))
        self.assertEqual(result["answer"], ABSTAIN)
        self.assertEqual(result["status"], "EVIDENCE_UNCERTAIN")

    def test_answer_not_grounded(self):
        result = answer_with_guardrails(**BASE, judge_fn=judge_sequence("YES"), verify_fn=judge_sequence("NO"))
        self.assertEqual(result["answer"], ABSTAIN)
        self.assertEqual(result["status"], "ANSWER_NO")

    def test_answer_generator_abstained(self):
        result = answer_with_guardrails(**{**BASE, "generate_answer": lambda *args: (ABSTAIN, 0.2, 700)},
                                        judge_fn=judge_sequence("YES"))
        self.assertEqual(result["status"], "GENERATOR_ABSTAINED")
        self.assertIsNone(result["verification"])


if __name__ == "__main__":
    unittest.main()
