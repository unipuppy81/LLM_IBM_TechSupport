"""Compare saved Stage2 A/B answers without retrieving or regenerating answers.
Place beside evaluate_stage2_quality.py in IBMTechSupport/scripts/evaluate/.
Only val_106.json is used. No gold information goes to any runtime guardrail.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

import evaluate_stage2_quality as quality

ROOT = Path(__file__).resolve().parents[2]
ABSTAIN = quality.ABSTAIN
PARENT = ROOT / 'evaluate_results/106/rag/stage2_prompt_ab'
DEFAULT_RUN = PARENT / 'run_fa9wu45r'


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def normalized(text):
    return ' '.join(text.split()).casefold()


def refused(p):
    return p['prediction'].strip() == ABSTAIN


def status(p):
    return p['guardrails']['status']


def validate(run):
    comparison = quality.load_json(run / 'comparison.json')
    exp = quality.load_json(run / 'experiment.json')
    if (comparison['status'] != 'COMPLETED' or exp['status'] != 'COMPLETED'
            or exp['judge_mode'] != 'per_document'
            or exp['only_changed_variable'] != 'RELAXED_ANSWERABILITY_SYSTEM'
            or exp['same_retrieval_for_both'] is not True
            or exp['retrieval_rerun'] is not False
            or exp['dataset_sha256'] != quality.sha256(quality.DATASET)):
        raise ValueError('Expected completed Stage2 per-document prompt-only experiment')
    if exp['old_prompt'] == exp['new_prompt']:
        raise ValueError('A/B prompts are identical')
    paths = {k: run / k / 'predictions.jsonl' for k in ('baseline', 'new_prompt')}
    records, a, ma = quality.read_inputs(paths['baseline'])
    _, b, mb = quality.read_inputs(paths['new_prompt'])
    if not (ma and mb):
        raise ValueError('Both generation manifests must match the validation dataset')
    for name, path in paths.items():
        manifest = quality.load_json(path.with_name('generation_manifest.json'))
        if (manifest['prompt_variant'] != name
                or manifest['rag_config_sha256'] != exp['rag_config_sha256']
                or manifest['retrieval_config_sha256'] != exp['historical_retrieval_config_sha256']):
            raise ValueError(f'Manifest mismatch: {name}')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from evaluate.common import score_predictions
    for name, predictions in [('baseline', a), ('new_prompt', b)]:
        rows = [predictions[r['id']] for r in records]
        scores = score_predictions(records, rows, 'val_106.json', 1)
        for key in ('token_f1_percent', 'abstention_rate_percent', 'false_refusal_rate_percent', 'exact_match_percent'):
            if scores[key] != comparison[name][key]:
                raise ValueError(f'Saved metric mismatch: {name}: {key}')
        observed = {'answerable_refused': sum(not p['is_impossible'] and refused(p) for p in rows),
                    'impossible_answered': sum(p['is_impossible'] and not refused(p) for p in rows),
                    'statuses': dict(Counter(status(p) for p in rows))}
        if observed != comparison['counts'][name]:
            raise ValueError(f'Saved counts mismatch: {name}')
    for r in records:
        rid = r['id']
        if a[rid]['retrieved_documents'] != b[rid]['retrieved_documents']:
            raise ValueError(f'Retrieval differs between variants: {rid}')
        for p in (a[rid], b[rid]):
            if len(p['retrieved_documents']) != 10:
                raise ValueError(f'Expected ten original excerpts: {rid}')
            if not isinstance(p['guardrails']['judge_history'], list):
                raise ValueError(f'Missing judge history: {rid}')
    return records, a, b, paths, comparison


def diagnostic(record, a, b):
    """Evidence presence is lexical, not a proof of applicability or answerability."""
    reference = normalized(record['answer'])
    gold = {d.get('filename') for d in record.get('contexts', []) if isinstance(d, dict)} - {None}
    docs = []
    for rank, d in enumerate(b['retrieved_documents'], 1):
        docs.append({'rank': rank, 'filename': d.get('filename'), 'chunk_id': d.get('chunk_id'),
                     'gold_filename_exact_match': d.get('filename') in gold,
                     'reference_substring_normalized': bool(reference and reference in normalized(d['text'])),
                     'text': d['text']})
    def details(p):
        g = p['guardrails']
        return {'status': status(p), 'prediction': p['prediction'],
                'candidate_answer': g.get('candidate_answer'),
                'judge_history': g['judge_history'], 'verification': g.get('verification'),
                'used_chunk_ids': g.get('used_chunk_ids')}
    return {'id': record['id'], 'question': record['question'], 'reference_answer': record['answer'],
            'is_impossible': record['is_impossible'], 'baseline': details(a), 'new_prompt': details(b),
            'gold_documents': record.get('contexts', []), 'retrieved_documents': docs,
            'gold_filename_in_top5': any(d['gold_filename_exact_match'] for d in docs[:5]),
            'gold_filename_in_top10': any(d['gold_filename_exact_match'] for d in docs),
            'reference_substring_in_top5': any(d['reference_substring_normalized'] for d in docs[:5]),
            'reference_substring_in_top10': any(d['reference_substring_normalized'] for d in docs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--judge-model', type=Path, default=ROOT / 'models/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--diagnostics-only', action='store_true', help='Extract refusal diagnostics without GPU inference')
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(PARENT.resolve()):
        parser.error('Use a Stage2 A/B run under stage2_prompt_ab')
    records, a, b, paths, comparison = validate(run)
    out = Path(mkdtemp(prefix='quality_review_', dir=run))
    cases = [diagnostic(r, a[r['id']], b[r['id']]) for r in records
             if not r['is_impossible'] and refused(b[r['id']])]
    newly_refused = [c for c in cases if c['baseline']['prediction'].strip() != ABSTAIN]
    transitions = Counter()
    for r in records:
        x, y = a[r['id']], b[r['id']]
        transitions[(str(r['is_impossible']), 'refused' if refused(x) else 'answered',
                     'refused' if refused(y) else 'answered')] += 1
    diagnostic_report = {
        'false_refusals': len(cases), 'newly_refused': len(newly_refused),
        'retained_refusals': len(cases) - len(newly_refused),
        'restored_non_abstentions': sum(not r['is_impossible'] and refused(a[r['id']]) and not refused(b[r['id']]) for r in records),
        'final_status_counts': dict(Counter(c['new_prompt']['status'] for c in cases)),
        'newly_refused_status_counts': dict(Counter(c['new_prompt']['status'] for c in newly_refused)),
        'lexical_evidence_checks': {k: sum(c[k] for c in cases) for k in ('gold_filename_in_top5', 'gold_filename_in_top10', 'reference_substring_in_top5', 'reference_substring_in_top10')},
        'transitions': [{'is_impossible': label == 'True', 'baseline': before, 'new_prompt': after, 'count': n}
                        for (label, before, after), n in sorted(transitions.items())],
        'note': 'Exact filename/substring matches are diagnostic clues, not sufficient proof. Prompt instruction responsible for NO is not observable from one-word judgments. Manual review required.'}
    write_jsonl(out / 'false_refusal_cases.jsonl', cases)
    write_json(out / 'refusal_diagnostics.json', diagnostic_report)
    print('False-refusal diagnostics saved:', out, flush=True)
    if args.diagnostics_only:
        return
    start = time.perf_counter()
    evaluator = quality.Evaluator(args.judge_model.resolve())
    results = {'baseline': [], 'new_prompt': []}
    for i, r in enumerate(records, 1):
        rid = r['id']
        # Same independent prompts, same evaluator weights, same generation settings.
        # Reuse an evaluation only if the actual answer and actual used evidence match.
        first = quality.evaluate_case(r, a[rid], evaluator)
        results['baseline'].append(first)
        if (a[rid]['prediction'] == b[rid]['prediction'] and
                quality.used_excerpts(a[rid]) == quality.used_excerpts(b[rid])):
            second = first.copy()
        else:
            second = quality.evaluate_case(r, b[rid], evaluator)
        results['new_prompt'].append(second)
        if i == 1 or i % 5 == 0 or i == len(records):
            print(f'[Stage2 A/B Quality] {i}/{len(records)}', flush=True)
    metrics = {name: quality.summarize(rows) for name, rows in results.items()}
    for name, rows in results.items():
        write_jsonl(out / f'{name}_quality_cases.jsonl', rows)
    delta = {}
    for k in ('answer_correctness', 'faithfulness'):
        before, after = (metrics[name][k]['strict_percent'] for name in ('baseline', 'new_prompt'))
        delta[k] = round(after - before, 2) if before is not None and after is not None else None
    # Keep the coverage effect visible; strict faithfulness uses different emitted denominators.
    for name, rows in results.items():
        metrics[name]['correct_and_supported_answerable_count'] = sum(
            not r['is_impossible'] and r['correctness']['verdict'] == 'CORRECT'
            and r['faithfulness']['verdict'] == 'SUPPORTED' for r in rows)
        metrics[name]['correct_and_supported_answerable_percent'] = quality.pct(metrics[name]['correct_and_supported_answerable_count'], 77)
    report = {'source_run': str(run), 'dataset_sha256': quality.sha256(quality.DATASET),
              'prediction_sha256': {k: quality.sha256(p) for k, p in paths.items()},
              'evaluator_script_sha256': quality.sha256(Path(quality.__file__)),
              'judge_model': str(args.judge_model.resolve()),
              'correctness_prompt': quality.CORRECT_SYSTEM, 'faithfulness_prompt': quality.FAITH_SYSTEM,
              'baseline': metrics['baseline'], 'new_prompt': metrics['new_prompt'],
              'delta_pp': delta, 'original_ab_metrics': comparison,
              'elapsed_seconds': round(time.perf_counter() - start, 3),
              'note': 'Saved answers only; no retrieval or regeneration. Default judge shares generator base weights; verdicts require human validation. Faithfulness is answer-level and excludes refusals; compare counts and denominators.'}
    write_json(out / 'quality_comparison.json', report)
    for name in results:
        print(name, '| Answer Correctness:', metrics[name]['answer_correctness']['strict_percent'],
              '| Faithfulness:', metrics[name]['faithfulness']['strict_percent'], flush=True)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
