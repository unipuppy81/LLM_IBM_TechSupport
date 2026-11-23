import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "IBMTechSupport" / "scripts"))
from rag_guardrails_per_document import (
    ABSTAIN, answer_with_per_document_guardrails,
    choose_guardrail_runner, judge_documents_individually,
)

DOCS = [
    {"filename": f"doc{i}.txt", "chunk_id": f"chunk_{i}", "text": f"Document {i}"}
    for i in range(1, 11)
]


def fake_judge(decisions, seen):
    def judge(model, tokenizer, question, documents):
        assert len(documents) == 1, "Every Judge call must receive exactly ONE chunk"
        seen.append(documents[0]["filename"])
        decision = decisions[documents[0]["filename"]]
        return {"decision": decision, "raw": decision, "seconds": 0.1}
    return judge


def fake_verify(seen, decision="YES"):
    def verify(model, tokenizer, question, documents, answer):
        seen.append([doc["filename"] for doc in documents])
        return {"decision": decision, "raw": decision, "seconds": 0.2}
    return verify


def fake_generate(seen, answer="IBM fix"):
    def generate(model, tokenizer, gen, question, documents, prompt):
        seen.append([doc["filename"] for doc in documents])
        return answer, 0.5, 123
    return generate


BASE = dict(
    question="How to fix?", retrieved_documents=DOCS, model=None,
    tokenizer=None, generation_config=None, prompt_config={},
    top_k=5, retry_top_k=10,
)


class TestPerDocumentJudge(unittest.TestCase):
    def test_all_top5_judged_independently_and_aggregate_yes(self):
        seen = []
        choices = {d["filename"]: "YES" if i in (1, 4) else "NO" for i, d in enumerate(DOCS, 1)}
        result = judge_documents_individually(**{k: BASE[k] for k in ("model", "tokenizer", "question", "retrieved_documents", "top_k", "retry_top_k")}, judge_fn=fake_judge(choices, seen))
        self.assertEqual(result["decision"], "YES")
        self.assertEqual(seen, [d["filename"] for d in DOCS[:5]])
        self.assertEqual([d["filename"] for d in result["selected_documents"]], ["doc1.txt", "doc4.txt"])
        self.assertEqual(len(result["judge_history"]), 5)
        self.assertAlmostEqual(result["judge_seconds"], 0.5)

    def test_fallback_only_checks_remaining_5(self):
        seen = []
        choices = {d["filename"]: "YES" if i == 7 else "NO" for i, d in enumerate(DOCS, 1)}
        result = judge_documents_individually(**{k: BASE[k] for k in ("model", "tokenizer", "question", "retrieved_documents", "top_k", "retry_top_k")}, judge_fn=fake_judge(choices, seen))
        self.assertEqual(result["decision"], "YES")
        self.assertEqual(seen, [d["filename"] for d in DOCS])
        self.assertEqual([d["filename"] for d in result["selected_documents"]], ["doc7.txt"])
        self.assertEqual([item["rank"] for item in result["judge_history"]], list(range(1, 11)))

    def test_no_evidence_returns_abstain_without_generation(self):
        seen = []
        choices = {d["filename"]: "NO" for d in DOCS}
        result = answer_with_per_document_guardrails(
            **BASE, generate_answer=fake_generate(seen),
            judge_fn=fake_judge(choices, []), verify_fn=fake_verify(seen),
        )
        self.assertEqual(result["status"], "EVIDENCE_NO")
        self.assertEqual(result["answer"], ABSTAIN)
        self.assertEqual(seen, [])

    def test_uncertain_is_not_overridden_to_yes(self):
        choices = {d["filename"]: "UNCERTAIN" if i == 3 else "NO" for i, d in enumerate(DOCS, 1)}
        result = judge_documents_individually(**{k: BASE[k] for k in ("model", "tokenizer", "question", "retrieved_documents", "top_k", "retry_top_k")}, judge_fn=fake_judge(choices, []))
        self.assertEqual(result["decision"], "UNCERTAIN")

    def test_generation_and_verify_see_only_yes_documents(self):
        seen_generated = []
        seen_verified = []
        choices = {d["filename"]: "YES" if i == 1 else "NO" for i, d in enumerate(DOCS, 1)}
        result = answer_with_per_document_guardrails(
            **BASE, generate_answer=fake_generate(seen_generated),
            judge_fn=fake_judge(choices, []), verify_fn=fake_verify(seen_verified),
        )
        self.assertEqual(result["status"], "ANSWER_VERIFIED")
        self.assertEqual(seen_generated, [["doc1.txt"]])
        self.assertEqual(seen_verified, [["doc1.txt"]])
        self.assertEqual(result["answer"], "IBM fix")
        self.assertEqual(len(result["judge_history"]), 5)

    def test_post_generation_rejection_still_enforced(self):
        choices = {d["filename"]: "YES" for d in DOCS}
        result = answer_with_per_document_guardrails(
            **BASE, generate_answer=fake_generate([]),
            judge_fn=fake_judge(choices, []), verify_fn=fake_verify([], "NO"),
        )
        self.assertEqual(result["status"], "ANSWER_NO")
        self.assertEqual(result["answer"], ABSTAIN)
        self.assertEqual(result["candidate_answer"], "IBM fix")

    def test_empty_retrieval_does_not_call_judge(self):
        result = answer_with_per_document_guardrails(
            **{**BASE, "retrieved_documents": []},
            generate_answer=fake_generate([]),
            judge_fn=lambda *args: self.fail("Should not judge empty retrieval"),
            verify_fn=lambda *args: self.fail("Should not verify empty retrieval"),
        )
        self.assertEqual(result["status"], "NO_DOCUMENTS")
        self.assertEqual(result["answer"], ABSTAIN)

    def test_modes_preserve_v3(self):
        from rag_guardrails import answer_with_guardrails
        from rag_guardrails_relaxed import answer_with_relaxed_guardrails
        self.assertIs(choose_guardrail_runner({"guardrails": {"judge_mode": "strict"}}), answer_with_guardrails)
        self.assertIs(choose_guardrail_runner({"guardrails": {"judge_mode": "relaxed"}}), answer_with_relaxed_guardrails)
        self.assertIs(choose_guardrail_runner({"guardrails": {"judge_mode": "per_document"}}), answer_with_per_document_guardrails)
        with self.assertRaises(ValueError):
            choose_guardrail_runner({"guardrails": {"judge_mode": "unknown"}})

    def test_generation_refusal_preserved(self):
        choices = {d["filename"]: "YES" for d in DOCS}
        result = answer_with_per_document_guardrails(
            **BASE, generate_answer=fake_generate([], ABSTAIN),
            judge_fn=fake_judge(choices, []),
            verify_fn=lambda *args: self.fail("verification should not be called"),
        )
        self.assertEqual(result["status"], "GENERATOR_ABSTAINED")


if __name__ == "__main__":
    unittest.main()
