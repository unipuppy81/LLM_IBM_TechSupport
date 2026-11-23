import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'IBMTechSupport/scripts/evaluate/judge_quality/judge_quality.py'
spec = importlib.util.spec_from_file_location('judge_quality_v2', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

POLICY = json.loads((MODULE.parents[3] / 'configs/judge_quality/policy.json').read_text(encoding='utf8'))
QUOTE = 'Add fewer than 1000 people at one time.'
DOC = {'chunk_id':'doc_1','filename':'swg27046675.txt','text':'Title: IBM Known limitation\n\nText:\nTo work around the limitation, '+QUOTE}


def score(quote=QUOTE, **kwargs):
    base = {'correctness':5,'groundedness':5,'completeness':5,'actionability':5,
            'safety':'pass','format':'pass','evidence_quote':quote,
            'failure_reason':'','risk_level':'low'}
    base.update(kwargs)
    return json.dumps(base)


def record(answer='To work around the limitation, add fewer than 1000 people.', docs=None):
    docs = [DOC] if docs is None else docs
    return {'id':'TRAIN_Q270','question':'How to add people to an Atlas matter request?',
            'prediction':answer,'retrieved_documents':docs,
            'guardrails':{'used_chunk_ids':[d['chunk_id'] for d in docs]}}


class JudgeQualityV2Test(unittest.TestCase):
    def test_all_fields_strict_even_if_scores_5(self):
        bad = json.loads(score())
        del bad['risk_level']
        with self.assertRaisesRegex(ValueError, 'risk_level'):
            mod.parse_rubric(json.dumps(bad),[DOC],'answer',POLICY)

    def test_quoted_exact_passes(self):
        parsed = mod.parse_rubric(score(),[DOC],'add fewer than 1000 people',POLICY)
        self.assertTrue(parsed['evidence_quote_verified'])
        self.assertTrue(parsed['auto_pass'])

    def test_high_scores_no_quote_fail_closed(self):
        parsed = mod.parse_rubric(score(''),[DOC],'add fewer than 1000 people',POLICY)
        self.assertFalse(parsed['auto_pass'])
        self.assertIn('MISSING_EXACT_SUPPORTING_QUOTE', parsed['review_flags'])

    def test_no_quote_low_scores_valid_rejection(self):
        parsed = mod.parse_rubric(score('',correctness=2,groundedness=1,
                            risk_level='high',failure_reason='No source supports this remedy'),
                            [DOC],'an unsupported answer',POLICY)
        self.assertFalse(parsed['auto_pass'])
        self.assertEqual(parsed['review_flags'],[])

    def test_quote_from_other_source_not_allowed(self):
        parsed = mod.parse_rubric(score('Unrelated document text for another product.'),[DOC],
                                  'add fewer than 1000 people', POLICY)
        self.assertFalse(parsed['auto_pass'])
        self.assertIn('QUOTE_NOT_IN_SELECTED_DOCS',parsed['review_flags'])

    def test_missing_reason_for_low_score_requires_retry(self):
        parsed = mod.parse_rubric(score('',correctness=1,groundedness=1),[DOC],
                                  'unsupported answer', POLICY)
        self.assertIn('MISSING_FAILURE_REASON',parsed['review_flags'])

    def test_retry_missing_field_then_fixed(self):
        incomplete=json.loads(score())
        del incomplete['risk_level']
        responses=[json.dumps(incomplete),score()]
        messages_seen=[]
        def generate(messages):
            messages_seen.append(messages[:])
            return responses.pop(0)
        res=mod.evaluate_case(record(),POLICY,generate,max_attempts=3)
        self.assertEqual(res['judge_status'],'OK')
        self.assertTrue(res['score']['auto_pass'])
        self.assertEqual(res['attempt_count'],2)
        self.assertEqual(len(messages_seen[1]),4)
        self.assertIn('risk_level',messages_seen[1][-1]['content'])

    def test_retry_high_scores_empty_quote_then_valid_rejection(self):
        responses=[score(''),score('',correctness=2,groundedness=1,risk_level='high',
                                   failure_reason='This evidence contains no supported solution')]
        res=mod.evaluate_case(record(),POLICY,lambda messages:responses.pop(0),3)
        self.assertEqual(res['judge_status'],'OK')
        self.assertFalse(res['score']['auto_pass'])
        self.assertEqual(res['attempt_count'],2)

    def test_exhaust_retries_review_required(self):
        res=mod.evaluate_case(record(),POLICY,lambda messages:score(''),max_attempts=3)
        self.assertEqual(res['judge_status'],'REVIEW_REQUIRED')
        self.assertFalse(res['score']['auto_pass'])
        self.assertEqual(res['attempt_count'],3)

    def test_abstention_with_docs_needs_human_review(self):
        res=mod.evaluate_case(record(mod.ABSTAIN),POLICY,lambda messages:score(''),3)
        self.assertEqual(res['judge_status'],'REVIEW_REQUIRED')
        self.assertFalse(res['score']['auto_pass'])
        self.assertEqual(res['attempt_count'],1)

    def test_title_alone_does_not_verify_evidence(self):
        doc = {'chunk_id':'x','text': 'Title: Resolving missing plugin problem\n\nText:\nThis is unrelated.'}
        parsed = mod.parse_rubric(score('Resolving missing plugin problem'), [doc], 'resolving', POLICY)
        self.assertFalse(parsed['auto_pass'])
        self.assertFalse(parsed['evidence_quote_verified'])

    def test_calibrate_excludes_review_required_and_legacy_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            csv_path=root/'human.csv'
            judge_path=root/'judges.jsonl'
            outfile=root/'metric.json'
            mod.write_csv(csv_path,[
                {'id':'A','human_pass':'fail','qa_label_only_for_sampling':'impossible','v4_status':'ANSWER_VERIFIED'},
                {'id':'B','human_pass':'pass','qa_label_only_for_sampling':'answerable','v4_status':'ANSWER_VERIFIED'},
                {'id':'C','human_pass':'fail','qa_label_only_for_sampling':'impossible','v4_status':'ANSWER_VERIFIED'},
            ],['id','human_pass','qa_label_only_for_sampling','v4_status'])
            mod.write_jsonl(judge_path,[
                {'id':'A','judge_schema_version':mod.JUDGE_SCHEMA_VERSION,'judge_status':'REVIEW_REQUIRED','score':{'auto_pass':False}},
                {'id':'B','judge_schema_version':mod.JUDGE_SCHEMA_VERSION,'judge_status':'OK','score':{
                    'auto_pass':True, 'correctness':5,'groundedness':5,'completeness':5,
                    'actionability':5,'safety':'pass','format':'pass','review_flags':[]}},
                {'id':'C','judge_status':'OK','score':{'auto_pass':True}},
            ])
            mod.calibrate(SimpleNamespace(review_csv=csv_path,judge_jsonl=judge_path,
                                          output=outfile,seed=42,bootstrap=10))
            metrics=json.loads(outfile.read_text(encoding='utf8'))
            self.assertEqual(metrics['n'],1)
            self.assertEqual(metrics['excluded_from_pairing'],2)

    def test_actual_uploaded_case_outputs_not_auto_approved(self):
        data=Path('/mnt/data/rubric_judge.jsonl')
        if not data.is_file(): self.skipTest('uploaded rubric_judge.jsonl not available')
        for row in (json.loads(l) for l in data.read_text(encoding='utf8').splitlines()):
            parsed=None
            try:
                parsed=mod.parse_rubric(row['raw_response'],[DOC], 'unsupported answer', POLICY)
            except (ValueError, KeyError): pass
            self.assertFalse(parsed is not None and parsed['auto_pass'],row['id'])

if __name__ == '__main__':
    unittest.main()
