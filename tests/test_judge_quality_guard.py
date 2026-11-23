import unittest
from IBMTechSupport.scripts.evaluate.judge_quality.judge_quality_guard import (
    audit, contradiction_risk, validate_model_verdict,
    EXACT_TEXT_CONTRADICTION, POTENTIAL_FALSE_CONTRADICTION, REASON_RESTATES_CLAIM
)


class JudgeQualityGuardTest(unittest.TestCase):
    def test_identical_quote_contradicted_needs_review(self):
        c = {'verdict': 'CONTRADICTED', 'text': 'Run the command to restart the service.',
             'quote': 'Run the command to restart the service.'}
        self.assertEqual(contradiction_risk(c)[0], EXACT_TEXT_CONTRADICTION)

    def test_clear_negation_not_flagged_by_similarity(self):
        c = {'verdict': 'CONTRADICTED', 'text': 'The command will work on every machine.',
             'quote': 'The command will not work on every machine.'}
        self.assertIsNone(contradiction_risk(c))

    def test_noncontradiction_not_flagged(self):
        c = {'verdict': 'SUPPORTED', 'text': 'Run the command to restart the service.',
             'quote': 'Run the command to restart the service.'}
        self.assertIsNone(contradiction_risk(c))

    def test_missing_quote_not_flagged(self):
        c = {'verdict': 'CONTRADICTED', 'text': 'Run the command to restart the service.',
             'quote': ''}
        self.assertIsNone(contradiction_risk(c))

    def test_high_similarity_is_only_a_review_signal(self):
        c = {'verdict': 'CONTRADICTED',
             'text': 'WebSphere Portal has no built in setting to limit user logins.',
             'quote': 'WebSphere Portal has no built in mechanism or setting to limit user logins.'}
        self.assertEqual(contradiction_risk(c)[0], POTENTIAL_FALSE_CONTRADICTION)

    def test_reason_with_opposite_polarity_not_flagged(self):
        c = {'verdict':'CONTRADICTED',
             'text':'Renaming the existing server is possible during installation.',
             'quote':'Renaming the existing server is not possible during installation.',
             'reason':'The document explicitly states that renaming the existing server is not possible during installation.'}
        self.assertIsNone(contradiction_risk(c))

    def test_same_numeric_condition_suspect(self):
        c = {'verdict':'CONTRADICTED',
             'text': 'It is recommended to add fewer than 1000 people at once.',
             'quote': 'When adding from clipboard, add fewer than 1000 people at one time.'}
        self.assertEqual(contradiction_risk(c)[0], POTENTIAL_FALSE_CONTRADICTION)

    def test_reason_restates_claim_suspect(self):
        c = {'verdict':'CONTRADICTED',
             'text':'Explicit MQDISC calls in the application prevent the problem from occurring.',
             'quote':'If an explicit MQDISC call is used, the problem does not occur.',
             'reason':'The provided evidence explicitly states that explicit MQDISC calls prevent the problem from occurring.'}
        self.assertEqual(contradiction_risk(c)[0], REASON_RESTATES_CLAIM)

    def test_valid_evidence_id(self):
        self.assertEqual(validate_model_verdict(
            {'verdict':'SUPPORTED','evidence_id':'E1','reason':'same technical version'}, {'E1'}),
            (True, 'OK'))

    def test_invalid_evidence_id(self):
        self.assertEqual(validate_model_verdict(
            {'verdict':'SUPPORTED','evidence_id':'','reason':'same technical version'}, {'E1'}),
            (False, 'EVIDENCE_ID_NOT_IN_CANDIDATES'))

    def test_not_found_has_no_evidence_id(self):
        self.assertEqual(validate_model_verdict(
            {'verdict':'NOT_FOUND','evidence_id':'','reason':'no passage states the condition'}, {'E1'}),
            (True, 'OK'))

    def test_not_found_spurious_evidence_id(self):
        self.assertEqual(validate_model_verdict(
            {'verdict':'NOT_FOUND','evidence_id':'E1','reason':'no passage states the condition'}, {'E1'}),
            (False, 'UNEXPECTED_EVIDENCE_ID_FOR_NOT_FOUND'))

    def test_audit_keeps_original_verdict(self):
        record = {'id':'CASE_1','judge_status':'OK', 'score':{
            'auto_pass':False,'claim_verifications':[{'claim_id':1,'verdict':'CONTRADICTED',
              'text':'Run the command to restart the service.',
              'quote':'Run the command to restart the service.'}]}, 'attempts':[]}
        issues, summary = audit([record])
        self.assertEqual(issues[0]['issue'], EXACT_TEXT_CONTRADICTION)
        self.assertEqual(record['score']['claim_verifications'][0]['verdict'], 'CONTRADICTED')
        self.assertFalse(record['score']['auto_pass'])
        self.assertEqual(summary['audit_affected_record_count'], 1)


if __name__ == '__main__':
    unittest.main()
