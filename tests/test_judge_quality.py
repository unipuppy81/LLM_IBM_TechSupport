import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[1] / 'IBMTechSupport/scripts/evaluate/judge_quality/judge_quality.py'
spec = importlib.util.spec_from_file_location('judge_quality', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class JudgeQualityTest(unittest.TestCase):
    def test_confusion_counts_and_kappa(self):
        met = mod.confusion([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((met['TP'],met['FP'],met['TN'],met['FN']), (1,1,1,1))
        self.assertEqual(met['kappa'], 0.0)

    def test_exact_citation_required(self):
        docs = [{'text': 'Add fewer than 1000 people at one time.'}]
        scores = {'correctness':5,'groundedness':5,'completeness':5,'actionability':5,
                  'safety':'pass','format':'pass','evidence_quote':'Add fewer than 1000 people',
                  'failure_reason':'','risk_level':'low'}
        policy={'quality_thresholds':{'correctness_min':4,'groundedness_min':4,'completeness_min':3,
                                     'actionability_min':3,'safety_required':'pass','format_required':'pass'}}
        result = mod.parse_rubric(json.dumps(scores), docs, 'Add fewer than 1000 people.', policy)
        self.assertTrue(result['auto_pass'])
        scores['evidence_quote']='Invented IBM paragraph'
        result = mod.parse_rubric(json.dumps(scores), docs, 'Add fewer than 1000 people.', policy)
        self.assertFalse(result['auto_pass'])

    def test_invalid_judge_output_is_not_pass(self):
        with self.assertRaises(ValueError):
            mod.parse_rubric('{"correctness": 7}', [], 'answer', {})

    def test_balanced_sample(self):
        records = []
        for i in range(12):
            records.append({'id':str(i),'is_impossible':i >= 8,
                            'guardrails':{'status':'ANSWER_VERIFIED'}})
        sample=mod.balanced_sample(records,6,42)
        self.assertEqual(len(sample),6)
        self.assertEqual(sum(r['is_impossible'] for r in sample),3)

    def test_refusal_checks_retrieved_docs(self):
        r={'prediction':mod.ABSTAIN,'retrieved_documents':[{'text':'Found answer'}],
           'guardrails':{'used_chunk_ids':[]}}
        self.assertEqual(len(mod.evaluation_docs(r)), 1)

    def test_gate_report_only_without_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(policy=MODULE.parents[3] / 'configs/judge_quality/policy.json',
                                 calibration=Path(d)/'missing.json', output=Path(d)/'gate.json',
                                 enforce=False)
            mod.gate(args)
            decision=json.loads((Path(d) / 'gate.json').read_text(encoding="utf-8"))
            self.assertEqual(decision['decision'],'REPORT_ONLY')
            self.assertFalse(decision['blocking_allowed'])

    def test_gate_enforce_without_calibration_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(policy=MODULE.parents[3] / 'configs/judge_quality/policy.json',
                                 calibration=Path(d)/'missing.json', output=Path(d)/'gate.json',
                                 enforce=True)
            with self.assertRaises(SystemExit) as e:
                mod.gate(args)
            self.assertEqual(e.exception.code,2)


if __name__=='__main__':
    unittest.main()
