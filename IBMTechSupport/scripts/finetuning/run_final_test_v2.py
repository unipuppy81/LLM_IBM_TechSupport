"""Final evaluation audit correction; keep the v1 frozen run untouched.
Install beside run_final_test.py and run_final_qlora.py.
No training, model selection, test filtering, or inference-policy changes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import run_final_test as original

V1 = original.DEST
V1_LOCK = original.LOCK
V1_AUDIT = original.AUDIT
V2 = V1.with_name('final_test_v2')


def previous():
    lock = original.read(V1_LOCK)
    audit = original.read(V1_AUDIT)
    original.require(lock['version'] == 'final-test-v1', 'Unexpected original lock')
    original.require(audit['lock_sha256'] == original.sha(V1_LOCK), 'Original audit/lock mismatch')
    original.require(not audit['passed'], 'Expected original failed audit')
    original.require(audit['comparisons']['original_clean']['id_overlap'], 'Expected historical data overlap')
    original.require(not any((V1 / name).exists() for name in (*original.MODELS, 'retrieval_cache.json', 'comparison.json', 'comparison_no_judge.json')), 'Original test generation already exists; review before migration')
    return lock, audit


def activate():
    original.DEST = V2
    original.LOCK = V2 / 'freeze.json'
    original.AUDIT = V2 / 'leakage_audit.json'
    original.CACHE = V2 / 'retrieval_cache.json'


def freeze():
    old, _ = previous()
    # Verify all existing frozen assets before generating a separate lock.
    original.verify_lock()
    activate()
    original.freeze()
    new = original.read(original.LOCK)
    # No changes to any previously locked asset, model or policy are permitted.
    for path, digest in old['assets'].items():
        original.require(new['assets'].get(path) == digest, f'Original frozen asset changed: {path}')
    for k in ('models', 'context_training_prompt', 'retrieval_selected', 'reranker_snapshot', 'environment', 'guardrail', 'prompt_policy', 'refusal_rule'):
        original.require(new[k] == old[k], f'Frozen setting changed: {k}')
    for k in ('closed', 'rag'):
        old_cfg = {key: val for key, val in old['configs'][k].items() if key != 'output'}
        new_cfg = {key: val for key, val in new['configs'][k].items() if key != 'output'}
        original.require(old_cfg == new_cfg, f'Frozen config changed: {k}')
    original.write(V2 / 'migration.json', {
        'previous_lock_sha256': original.sha(V1_LOCK),
        'previous_audit_sha256': original.sha(V1_AUDIT),
        'new_lock_sha256': original.sha(original.LOCK),
        'reason': 'Historical unused original_clean overlap reported separately; additionally audit validation against test.',
        'same_previous_frozen_assets_and_settings': True,
    })


def verify_migration():
    old, prior = previous()
    migration = original.read(V2 / 'migration.json')
    original.require(migration['previous_lock_sha256'] == original.sha(V1_LOCK) and migration['previous_audit_sha256'] == original.sha(V1_AUDIT) and migration['new_lock_sha256'] == original.sha(V2 / 'freeze.json') and migration['same_previous_frozen_assets_and_settings'] is True, 'Migration provenance mismatch')
    return prior


def overlap(rows, train, norm, normalize):
    ids = {r['id'] for r in train}
    nfc = {norm(r['question']) for r in train}
    nfkc = {normalize(r['question']) for r in train}
    return {'train_count': len(train),
            'id_overlap': [r['id'] for r in rows if r['id'] in ids],
            'nfc_question_overlap': [r['id'] for r in rows if norm(r['question']) in nfc],
            'nfkc_question_overlap': [r['id'] for r in rows if normalize(r['question']) in nfkc]}


def has_overlap(c):
    return any(c[k] for k in ('id_overlap', 'nfc_question_overlap', 'nfkc_question_overlap'))


def audit():
    from run_stage0 import validate, norm
    from final_eval_preflight import normalize
    prior = verify_migration()
    original.verify_lock()
    original.require(not original.AUDIT.exists(), f'Existing audit protected: {original.AUDIT}')
    digest = original.sha(original.TEST)
    original.require(digest == prior['test_sha256'], 'Test changed since original audit')
    rows = validate(original.TEST)
    comps = {k: overlap(rows, validate(p), norm, normalize) for k, p in original.TRAIN.items()}
    for k, c in comps.items():
        original.require(c == prior['comparisons'][k], f'Audit result changed: {k}')
    val = overlap(rows, validate(original.matched.VAL), norm, normalize)
    dup = {n: [q for q, c in Counter(f(r['question']) for r in rows).items() if c > 1] for n, f in [('nfc', norm), ('nfkc', normalize)]}
    a = sum(not r['is_impossible'] for r in rows)
    actual_passed = not any(has_overlap(comps[k]) for k in ('closed', 'context'))
    validation_passed = not has_overlap(val)
    passed = len({r['id'] for r in rows}) == len(rows) and not any(dup.values()) and actual_passed and validation_passed
    result = {'lock_sha256': original.sha(original.LOCK), 'test_sha256': digest, 'total': len(rows), 'answerable': a, 'impossible': len(rows)-a, 'comparisons': comps, 'validation_comparison': val, 'test_duplicate_questions': dup, 'actual_training_overlap_passed': actual_passed, 'validation_overlap_passed': validation_passed, 'historical_original_clean_overlap_warning': has_overlap(comps['original_clean']), 'historical_original_clean_overlap_counts': {k: len(v) for k,v in comps['original_clean'].items() if isinstance(v,list)}, 'gating_policy': 'Block actual closed/context training overlap, validation overlap, or duplicate test records. Report unused historical original_clean overlap, without filtering test.', 'passed': passed}
    original.write(original.AUDIT, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'comparisons'}, ensure_ascii=False, indent=2))
    print('Audit:', original.AUDIT)
    original.require(passed, 'Actual training/validation leakage or test duplicates: final evaluation blocked')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze','audit','cache','evaluate','report'))
    p.add_argument('--model', choices=tuple(original.MODELS))
    p.add_argument('--skip-judge', action='store_true')
    a = p.parse_args()
    if a.action == 'freeze':
        freeze()
        return
    activate()
    verify_migration()
    if a.action == 'evaluate':
        if not a.model: p.error('--model required')
        original.evaluate(a.model, a.skip_judge)
    else:
        {'audit':audit, 'cache':original.cache, 'report':original.report}[a.action]()


if __name__ == '__main__':
    main()
