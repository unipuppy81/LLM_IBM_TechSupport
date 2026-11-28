import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IBMTechSupport/scripts"))
from rag_guardrails_per_document import choose_guardrail_runner
from rag_passage_recheck import recheck_passages, resolve_selection, source_passages
from rag_section_recheck import recheck_focused, recheck_sections


class TestPassageReview(unittest.TestCase):
    question = "How can I restart DB2 9.1 on Linux?"
    docs = [{"chunk_id": "a", "text": "DB2 9.1 on Linux.\n\nRun the restart utility."},
            {"chunk_id": "b", "text": "DB2 9.10 on Windows. Use the console."}]

    def selection(self, number=1, extra=()):
        return json.dumps({"decision": "YES", "requirements": [
            {"need": "restart procedure", "passage": number}], "additional_conditions": list(extra)})

    def test_exact_offsets_and_scope(self):
        indexed, views = source_passages(self.question, self.docs)
        self.assertEqual([d["eligible"] for d in views], [True, False])
        reason, result = resolve_selection(self.selection(), self.question, indexed)
        self.assertEqual(reason, "supported")
        quote = result["requirements"][0]
        self.assertEqual(quote["chunk_id"], "a")
        self.assertEqual(quote["quote"], self.docs[0]["text"][quote["start"]:quote["end"]])
        for raw in (self.selection(2), self.selection(9), self.selection(True),
                    self.selection(extra=["AIX"]), self.selection()+" extra", '{"decision":"NO"}'):
            self.assertNotEqual(resolve_selection(raw, self.question, indexed)[0], "supported")

    def test_unmentioned_condition_in_source_and_exclusion(self):
        question = self.question + " with encryption?"
        indexed, _ = source_passages(question, self.docs)
        self.assertEqual(resolve_selection(self.selection(extra=["encryption"]), question, indexed)[0],
                         "incomplete_applicability")
        excluded, _ = source_passages(self.question, [{"chunk_id": "a", "text":
            self.docs[0]["text"] + " This procedure does not apply to Linux."}])
        self.assertEqual(resolve_selection(self.selection(), self.question, excluded)[0], "excluded_applicability")

    def test_no_eligible_source_fails_closed(self):
        with patch("rag_passage_recheck._llm_response", side_effect=AssertionError):
            result = recheck_passages(None, None, self.question, self.docs[1:])
        self.assertEqual((result["decision"], result["reason"]), ("NO", "no_applicable_chunk"))

    def test_route_keeps_top_five_for_generation_and_verification(self):
        docs = self.docs + [{"chunk_id": str(i), "text": "Unrelated."} for i in range(8)]
        run = choose_guardrail_runner({"guardrails": {"top5_recheck": True, "top5_recheck_method": "passages"}})
        gen = Mock(return_value=("Run the restart utility.", 0, 12))
        verify = Mock(return_value={"decision": "YES", "seconds": 0})
        with patch("rag_passage_recheck._llm_response", return_value=(self.selection(), 1)) as llm:
            result = run(question=self.question, retrieved_documents=docs, model=None, tokenizer=None,
                         generation_config=None, prompt_config={}, generate_answer=gen,
                         judge_fn=lambda *args: {"decision": "NO", "seconds": 0}, verify_fn=verify)
        llm.assert_called_once()
        self.assertEqual(result["status"], "ANSWER_VERIFIED")
        self.assertEqual(gen.call_args.args[4], docs[:5])
        self.assertEqual(verify.call_args.args[3], docs[:5])

    def test_section_choice_binds_to_exact_answer_and_rejects_other_outputs(self):
        source = "QUESTION\nHow can I restart DB2 9.1 on Linux?\n\nANSWER\nUse the restart utility.\n\nRELATED INFORMATION\nOther material."
        docs = [{"chunk_id": "kb", "text": source}]
        for raw, expected in (("A", "YES"), ("N", "NO"), ("B", "NO"), ("A because...", "NO")):
            with patch("rag_section_recheck._llm_response", return_value=(raw, 1)):
                result = recheck_sections(None, None, self.question, docs)
            self.assertEqual(result["decision"], expected)
            if expected == "YES":
                item = result["review"]["requirements"][0]
                self.assertEqual(source[item["start"]:item["end"]], "Use the restart utility.")
                self.assertEqual(item["quote"], source[item["start"]:item["end"]])

    def test_section_choice_without_applicable_answer_never_calls_model(self):
        for source in ("QUESTION\nHow can I restart DB2 9.1 on Linux?",
                       "How can I restart DB2 9.10 on Linux?\nANSWER\nUse the restart utility."):
            with patch("rag_section_recheck._llm_response", side_effect=AssertionError):
                result = recheck_sections(None, None, self.question, [{"chunk_id": "a", "text": source}])
            self.assertEqual(result["reason"], "no_matching_answer_section")

    def test_focused_decision_needs_literal_support_and_explicit_yes(self):
        source = "How can I restart DB2 9.1 on Linux?\nANSWER\nUse the documented restart utility."
        docs = [{"chunk_id": "kb", "text": source}]
        for raw, expected in (("YES", "YES"), ("NO", "NO"), ("YES but uncertain", "NO")):
            with patch("rag_section_recheck._llm_response", return_value=(raw, 1)) as llm:
                result = recheck_focused(None, None, self.question, docs)
            llm.assert_called_once()
            self.assertEqual(result["decision"], expected)
            if expected == "YES":
                self.assertEqual(result["review"]["requirements"][0]["quote"], "Use the documented restart utility.")
        with patch("rag_section_recheck._llm_response", side_effect=AssertionError):
            for q in (self.question.replace("Linux", "AIX"), self.question + "\n\nWe are running with Liberty."):
                self.assertEqual(recheck_focused(None, None, q, docs)["decision"], "NO")
