import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IBMTechSupport/scripts"))
from evaluate import strict_recheck as experiment
from evaluate.strict_recheck import transition
from rag_guardrails import ABSTAIN
from rag_guardrails_per_document import choose_guardrail_runner
from rag_top5_recheck import validate_review


QUESTION = "How do I restart DB2 9.1 on Linux?"
TEXT = "On Linux with DB2 9.1, run the restart utility."
DOCS = [{"chunk_id": f"c{i}", "text": TEXT if i == 0 else "Another issue."} for i in range(10)]
REVIEW = {"decision": "YES", "requirements": [
    {"requirement": "restart procedure", "chunk_id": "c0", "quote": TEXT}
], "conditions": [{"condition": "Linux", "chunk_id": "c0", "quote": TEXT}]}


class TestReview(unittest.TestCase):
    def check(self, review, docs=DOCS, question=QUESTION):
        return validate_review(json.dumps(review), question, docs[:5])[0]

    def test_valid_exact_citations(self):
        self.assertEqual(self.check(REVIEW), "supported")
        fenced = "```json\n" + json.dumps(REVIEW) + "\n```"
        self.assertEqual(validate_review(fenced, QUESTION, DOCS)[0], "supported")
        self.assertEqual(validate_review(fenced + "\nProbably.", QUESTION, DOCS)[0], "invalid_json")

    def test_malformed_and_incomplete_outputs(self):
        for raw in ("yes", "```json\n{}\n```", '[]', '{"decision":"YES"}', '{"decision":"NO"}',
                    '{"decision":"YES","requirements":[],"conditions":[]}',
                    json.dumps(REVIEW).replace('"YES"', '"NO","decision":"YES"')):
            with self.subTest(raw=raw):
                self.assertNotEqual(validate_review(raw, QUESTION, DOCS)[0], "supported")

    def test_wrong_chunk_hallucinated_quote_and_scope(self):
        for update in ({"chunk_id": "c5"}, {"chunk_id": "c1"}, {"quote": TEXT.upper()}):
            item = {**REVIEW["requirements"][0], **update}
            self.assertEqual(self.check({**REVIEW, "requirements": [item]}), "quote_not_in_named_chunk")
        for question in (QUESTION.replace("9.1", "9.10"), QUESTION.replace("9.1", "8"),
                         QUESTION.replace("Linux", "AIX"), QUESTION.replace("DB2", "Liberty"),
                         QUESTION.replace("DB2", "DataPower"), QUESTION + " With SQLCODE -911?"):
            with self.subTest(question=question):
                self.assertNotEqual(self.check({**REVIEW, "conditions": []}, question=question), "supported")

    def test_scope_cannot_be_borrowed_from_another_chunk(self):
        item = {"requirement": "restart", "chunk_id": "c1", "quote": "Another issue."}
        self.assertEqual(self.check({**REVIEW, "requirements": [item]}), "incomplete_applicability")
        self.assertEqual(self.check(REVIEW, [{**DOCS[0], "text": TEXT + " Not applicable to Linux."},
                                             *DOCS[1:]]), "excluded_applicability")

    def test_recheck_flow(self):
        for enabled, docs, decisions, allowed in [
            (False, DOCS, ["NO", "NO"], False), (True, [], [], False),
            (True, DOCS[:5], ["NO"], False), (True, DOCS[:8], ["NO", "NO"], False),
            (True, DOCS, ["UNCERTAIN", "NO"], False), (True, DOCS, ["NO", "UNCERTAIN"], False),
            (True, DOCS, ["YES"], False), (True, DOCS, ["NO", "YES"], False),
            (True, DOCS, ["NO", "NO"], True)
        ]:
            with self.subTest(enabled=enabled, decisions=decisions, size=len(docs)):
                judge = Mock(side_effect=[{"decision": d, "seconds": 0} for d in decisions])
                generate = Mock(return_value=("Run the restart utility.", 1, 50))
                verify = Mock(return_value={"decision": "NO", "seconds": 1})
                run = choose_guardrail_runner({"guardrails": {"top5_recheck": enabled}})
                with patch("rag_top5_recheck._llm_response", return_value=(json.dumps(REVIEW), 1)) as llm:
                    result = run(question=QUESTION, retrieved_documents=docs, model=None, tokenizer=None,
                                 generation_config=None, prompt_config={}, generate_answer=generate,
                                 judge_fn=judge, verify_fn=verify)
                self.assertEqual(llm.call_count, int(allowed))
                self.assertEqual(result["answer"], ABSTAIN)
                if allowed:
                    self.assertEqual(result["status"], "ANSWER_NO")
                    self.assertEqual(generate.call_args.args[4], DOCS[:5])
                    self.assertEqual(verify.call_args.args[3], DOCS[:5])
                    self.assertEqual(result["judge_history"][-1]["decision"], "YES")
                elif "YES" not in decisions:
                    generate.assert_not_called()
                    verify.assert_not_called()

    def test_invalid_review_keeps_existing_refusal(self):
        run = choose_guardrail_runner({"guardrails": {"top5_recheck": True}})
        with patch("rag_top5_recheck._llm_response", return_value=("YES", 1)):
            res = run(question=QUESTION, retrieved_documents=DOCS, model=None, tokenizer=None,
                      generation_config=None, prompt_config={}, generate_answer=Mock(side_effect=AssertionError),
                      judge_fn=lambda *args: {"decision": "NO", "seconds": 0})
        self.assertEqual((res["answer"], res["status"]), (ABSTAIN, "EVIDENCE_NO"))
        self.assertEqual(res["judge_history"][-1]["reason"], "invalid_json")

    def test_passed_review_uses_original_generator_and_verifier(self):
        run = choose_guardrail_runner({"guardrails": {"top5_recheck": True}})
        prompt = {"system": "original"}
        config = object()
        for answer, expected in ((ABSTAIN, "GENERATOR_ABSTAINED"), ("Run the restart utility.", "ANSWER_VERIFIED")):
            generate = Mock(return_value=(answer, 1, 10))
            verify = Mock(return_value={"decision": "YES", "seconds": 1})
            with patch("rag_top5_recheck._llm_response", return_value=(json.dumps(REVIEW), 1)):
                result = run(question=QUESTION, retrieved_documents=DOCS, model=None, tokenizer=None,
                             generation_config=config, prompt_config=prompt, generate_answer=generate,
                             judge_fn=lambda *args: {"decision": "NO", "seconds": 0}, verify_fn=verify)
            self.assertEqual((result["answer"], result["status"]), (answer, expected))
            generate.assert_called_once_with(None, None, config, QUESTION, DOCS[:5], prompt)
            if answer == ABSTAIN:
                verify.assert_not_called()
            else:
                verify.assert_called_once_with(None, None, QUESTION, DOCS[:5], answer)

    def test_transitions(self):
        self.assertEqual(transition("REFUSED", "CORRECT"), "refused_to_correct")
        self.assertEqual(transition("REFUSED", "PARTIAL"), "refused_to_noncorrect")
        self.assertEqual(transition("CORRECT", "UNCERTAIN"), "correct_to_noncorrect")
        self.assertEqual(transition("CORRECT", "REFUSED"), "correct_to_refused")
        self.assertIsNone(transition("REFUSED", "REFUSED"))

    def test_paired_report_counts_and_impossible_denominator(self):
        config = {"model": {"path": "synthetic"}}
        # Synthetic verdict words stand in for answers only in this reporting test.
        pairs = [(ABSTAIN, "CORRECT"), (ABSTAIN, "PARTIAL"), ("CORRECT", "INCORRECT"),
                 ("CORRECT", ABSTAIN), (ABSTAIN, ABSTAIN), (ABSTAIN, "unanswerable response")]
        records = [{"id": str(i), "question": "Question", "answer": "Reference",
                    "is_impossible": i == 5} for i in range(len(pairs))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            dataset = folder / "dataset.json"
            experiment.save(dataset, records)
            cache = folder / "retrieval_cache.json"
            experiment.save(cache, [])
            experiment.save(folder / "manifest.json", {"cache_sha256": experiment.digest(cache),
                            "dataset_sha256": experiment.digest(dataset), "config": config})
            for side, name in enumerate(("strict", "recheck")):
                with (folder / f"{name}.jsonl").open("w", encoding="utf-8") as file:
                    for i, pair in enumerate(pairs):
                        status = "EVIDENCE_NO" if pair[side] == ABSTAIN else "ANSWER_VERIFIED"
                        history = []
                        if side and i in (0, 1, 3, 4, 5):
                            history.append({"stage": "top5_recheck", "decision": "YES", "reason": "supported"})
                            if i == 3:
                                status = "GENERATOR_ABSTAINED"
                            if i == 4:
                                status = "ANSWER_UNCERTAIN"
                        file.write(json.dumps({"id": str(i), "answer": pair[side], "status": status,
                                               "judge_history": history}) + "\n")
            with patch.object(experiment, "DATA", dataset), patch("builtins.print"), \
                    patch("evaluate.evaluate_stage2_quality.Evaluator") as evaluator, \
                    patch("importlib.metadata.version", return_value="test"):
                evaluator.return_value.evaluate.side_effect = lambda _, payload, labels: {
                    "verdict": payload["candidate_answer"]}
                experiment.score(folder, records, config)
            report = experiment.read(folder / "comparison.json")
            self.assertEqual(report["counts"]["strict"], {
                "judge_correct": 2, "false_refusals": 3, "impossible_responses": 0})
            self.assertEqual(report["counts"]["recheck"], {
                "judge_correct": 1, "false_refusals": 2, "impossible_responses": 1})
            self.assertEqual(set(report["transitions"].values()), {1})
            self.assertEqual(report["impossible_transitions"], {"refused_to_response": 1, "response_to_refused": 0})
            self.assertEqual(report["rechecks"], {"attempted": 5, "passed": 5,
                                                  "generator_refused": 1, "verifier_refused": 1})
            self.assertFalse(report["success"])
