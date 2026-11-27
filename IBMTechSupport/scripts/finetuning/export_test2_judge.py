import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'evaluate_results/106/finetuning/data_composition_test2/run_20261127_000005_052233/experiments/A_expanded/lora/quality_evaluation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Export saved Test2 judge decisions without running inference.')
    parser.add_argument('--source', type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = args.source.resolve()
    cases_path = source / 'cases.jsonl'
    metrics_path = source / 'metrics.json'
    cases = [json.loads(line) for line in cases_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    metrics = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
    answerable = [r for r in cases if not r['is_impossible']]
    counts = dict(Counter(r['correctness']['verdict'] for r in answerable))
    if len(cases) != 106 or len({r['id'] for r in cases}) != 106 or len(answerable) != 77:
        raise ValueError('Unexpected validation case counts')
    if counts != {'CORRECT': 41, 'INCORRECT': 21, 'UNCERTAIN': 15} or counts != metrics['answer_correctness_counts']:
        raise ValueError('Saved cases do not match Test2 reported verdicts')
    if metrics['answer_correctness_denominator'] != 77 or metrics['answer_correctness_percent'] != round(100 * counts['CORRECT'] / 77, 2):
        raise ValueError('Saved metrics do not match case counts')
    invalid = {k: sum(r[k]['format_valid'] is False for r in cases) for k in ('correctness', 'claim_review')}
    if sum(invalid.values()) != metrics['evaluator_invalid_outputs']:
        raise ValueError('Invalid output count mismatch')
    result = {
        'experiment': 'Data Composition Test2: A_expanded (532 training samples)',
        'source_cases_sha256': sha(cases_path),
        'source_metrics_sha256': sha(metrics_path),
        'metrics': metrics,
        'invalid_outputs_by_judge': invalid,
        'note': 'Original automatic verdicts and reasons, not human-verified. NOT_APPLICABLE is not a correctness verdict. No new judging performed.',
        'cases': cases,
    }
    text = ['Data Composition Test2 — original Judge decisions',
            '106 questions; 77 answerable; 29 impossible',
            f'Verdicts: {counts}', f'Invalid outputs: {invalid}',
            'Original automatic judgments; not human-verified.', '']
    for r in cases:
        c, claim = r['correctness'], r['claim_review']
        text.extend(['=' * 78, f"ID: {r['id']} | is_impossible: {r['is_impossible']} | refused: {r['refused']}",
                     'QUESTION:', r['question'], 'REFERENCE ANSWER:', r['reference_answer'],
                     'GENERATED ANSWER:', r['prediction'],
                     f"CORRECTNESS: {c['verdict']} | format_valid: {c['format_valid']}",
                     'REASON:', c['reason'], 'RAW JUDGE OUTPUT:', c['raw'] or '(not evaluated)',
                     f"CLAIM REVIEW: {claim['verdict']} | format_valid: {claim['format_valid']}",
                     'REASON:', claim['reason'], 'RAW JUDGE OUTPUT:', claim['raw'] or '(not evaluated)', ''])
    out = source.parent / 'judge_analysis'
    out.mkdir(exist_ok=True)
    files = {out / 'test2_judge_decisions.json': json.dumps(result, ensure_ascii=False, indent=2) + '\n',
             out / 'test2_judge_decisions.txt': '\n'.join(text)}
    if any(p.exists() for p in files):
        raise FileExistsError('Export already exists; original files were not modified')
    for path, content in files.items():
        with path.open('x', encoding='utf-8') as f:
            f.write(content)
    print(f'[Export] 106 cases | CORRECT 41 | INCORRECT 21 | UNCERTAIN 15')
    print(f'[Export] {out}')


if __name__ == '__main__':
    main()
