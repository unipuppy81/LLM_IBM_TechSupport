from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    ids = [str(row['id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate Judge IDs')
    return {str(row['id']): row for row in rows}


def analyze(human_json: dict, judge_by_id: dict) -> dict:
    reviews = human_json.get('reviews') or {}
    counts = Counter()
    causes = Counter()
    failures = []
    mismatches = []
    for qa_id, review in reviews.items():
        label = review.get('human_pass')
        if label not in ('pass', 'fail'):
            counts['unlabeled'] += 1
            continue
        score = judge_by_id.get(qa_id, {}).get('score') or {}
        if type(score.get('auto_pass')) is not bool:
            mismatches.append(qa_id)
            continue
        predicted = score['auto_pass']
        human = label == 'pass'
        category = ('TP' if human else 'FP') if predicted else ('FN' if human else 'TN')
        counts[category] += 1
        counts['n'] += 1
        causes[review.get('root_cause') or 'unclassified'] += 1
        if category in ('FN', 'FP'):
            failures.append({'id': qa_id, 'type': category,
                             'judge_status': judge_by_id[qa_id].get('judge_status'),
                             'claim_verdicts': [x.get('verdict') for x in score.get('claim_verifications', [])],
                             'review_flags': score.get('review_flags', []),
                             'human_assessment': review.get('final_assessment', '')})
    def ratio(a,b):
        return round(a/b, 4) if b else None
    tp,fp,tn,fn = (counts[x] for x in ('TP','FP','TN','FN'))
    return {
        'schema': 'judge-human-audit-v3.7',
        'source_judge_version': human_json.get('source_judge_version'),
        'counts': {k:int(counts[k]) for k in ('n','TP','FP','TN','FN','unlabeled')},
        'precision':ratio(tp,tp+fp), 'recall':ratio(tp,tp+fn),
        'accuracy':ratio(tp+tn,tp+fp+tn+fn),
        'false_approval_rate':ratio(fp,fp+tn),
        'root_causes':dict(causes), 'mismatched_ids':mismatches,
        'misclassified':failures,
        'note':'20 human-reviewed cases; v3.7 must be regraded before comparing versions.'
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--human',type=Path,required=True)
    p.add_argument('--judge',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(argv)
    report=analyze(json.loads(args.human.read_text(encoding='utf-8-sig')),
                   load_jsonl(args.judge))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('counts','precision','recall','accuracy','false_approval_rate','mismatched_ids')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
