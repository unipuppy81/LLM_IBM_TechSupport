"""Debug-hook tests; no GPU or model download needed."""
import importlib.util
import json
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / 'IBMTechSupport/scripts/evaluate/judge_quality/judge_quality.py'
spec = importlib.util.spec_from_file_location('judge_quality_debug_under_test', CODE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
POLICY = {"quality_thresholds": {
    "correctness_min": 4, "groundedness_min": 4,
    "completeness_min": 3, "actionability_min": 3,
    "safety_required": "pass", "format_required": "pass",
}}


class JudgeDebugTest(unittest.TestCase):
    def test_debug_hook_gets_exact_model_messages_and_raw_json(self):
        record = {
            "id": "TEST_Q1", "question": "How?", "prediction": "Use option A.",
            "retrieved_documents": [{"filename":"doc.txt", "chunk_id":"chunk1", "text":"IBM says use option A for this error."}],
            "guardrails": {"used_chunk_ids":["chunk1"]},
        }
        responses = [
            '{"correctness":5}',
            json.dumps({"correctness": 5,"groundedness": 5,"completeness": 5,
                        "actionability": 5,"safety":"pass","format":"pass",
                        "evidence_quote":"IBM says use option A for this error.",
                        "failure_reason":"","risk_level":"low"}),
        ]
        trace=[]
        def gen(messages):
            return responses[len(trace)]
        result=module.evaluate_case(record,POLICY,gen,max_attempts=3,debug_hook=trace.append)
        self.assertEqual(result['judge_status'],'OK')
        self.assertTrue(result['score']['auto_pass'])
        self.assertEqual(result['attempt_count'],2)
        self.assertEqual(len(trace),2)
        self.assertIn('IBM says use option A',trace[0]['messages'][1]['content'])
        self.assertIn('previous evaluation is invalid',trace[1]['messages'][-1]['content'])
        self.assertEqual(trace[0]['raw_response'],responses[0])
        self.assertEqual(trace[1]['parsed_rubric']['evidence_quote_verified'],True)

    def test_debug_hook_off_preserves_existing_results(self):
        record = {"id":"TEST_Q2","question":"Question", "prediction":"Answer",
                  "retrieved_documents": [], "guardrails":{"used_chunk_ids":[]}}
        raw=json.dumps({"correctness":2,"groundedness":2,"completeness":2,
                        "actionability":2,"safety":"pass","format":"pass",
                        "evidence_quote":"","failure_reason":"No evidence.","risk_level":"high"})
        result=module.evaluate_case(record,POLICY,lambda _:raw)
        self.assertEqual(result['attempt_count'],1)
        self.assertEqual(result['judge_status'],'OK')
        self.assertFalse(result['score']['auto_pass'])

if __name__=='__main__':
    unittest.main()
