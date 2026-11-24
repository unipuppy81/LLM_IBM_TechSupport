"""Select exactly the human-reviewed IDs from predictions.jsonl (no label leakage)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--human',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    human=json.loads(a.human.read_text(encoding='utf-8-sig'))
    review_ids=list(human['reviews'])
    if len(review_ids)!=len(set(review_ids)):
        raise ValueError('Duplicate human review IDs')
    rows={}
    for line in a.predictions.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            x=json.loads(line)
            if x.get('id') in review_ids:
                if x['id'] in rows: raise ValueError('Duplicate prediction ID '+x['id'])
                rows[x['id']]=x
    missing=[i for i in review_ids if i not in rows]
    if missing: raise ValueError('Missing predictions: '+', '.join(missing))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8') as out:
        for qa_id in review_ids:
            out.write(json.dumps(rows[qa_id],ensure_ascii=False)+'\n')
    print(f'Prepared {len(rows)} matched cases at {a.output}')

if __name__=='__main__':
    main()
