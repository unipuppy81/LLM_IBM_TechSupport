from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / 'evaluate_results/106/finetuning/stage1_quality/run_20261126_114427_318655/experiments/answer_quality/train.json'
EXPECTED_HASH = 'f7d04670ac9e4aaee70fd70802b813ced05d734086ab8c78f7658b1fb5dc2a28'
PRODUCTS = {
    'WebSphere / WAS': r'\b(?:websphere|was\s*v?\d|was\s+(?:server|ND))\b',
    'MQ': r'\b(?:websphere mq|ibm mq|mqseries|queue manager|mqrc_\w+|mq\s*v?\d)\b',
    'DB2': r'\b(?:db2|db2luw)\b',
    'Rational / Rhapsody / RTC / DOORS': r'\b(?:rational|rhapsody|doors|rtc|clearcase|clearquest|team concert)\b',
    'BPM / BAW': r'\b(?:bpm|baw|business process manager|business automation workflow)\b',
    'Cognos': r'\bcognos\b',
    'Maximo': r'\bmaximo\b',
    'SPSS': r'\bspss\b',
    'Informix': r'\binformix\b',
    'DataPower': r'\bdatapower\b',
    'Domino / Notes': r'\b(?:domino|lotus notes)\b',
    'AIX': r'\baix\b',
    'Tivoli / TSM': r'\b(?:tivoli|tsm|spectrum protect)\b',
    'IBM i / AS400': r'\b(?:ibm i|as/400|as400|iseries)\b',
    'z/OS': r'\b(?:z/os|zos)\b',
    'FileNet': r'\bfilenet\b',
    'Sterling': r'\bsterling\b',
    'Cloud / Bluemix': r'\b(?:bluemix|ibm cloud)\b',
}
URL = re.compile(r'https?://[^\s<>]+', re.I)
DOC_REF = re.compile(r'\b(?:this\s+(?:apar|document|article|technote|link|page|fix|defect)|(?:the\s+)?(?:above|below|following)\s+(?:link|document|table|figure|example|section|steps|procedure)|attached\s+(?:file|document|screenshot)|click\s+here|see\s+(?:here|below|above))\b', re.I)
APAR = re.compile(r'\b(?:[A-Z]{2}\d{5}|APAR)\b', re.I)

def norm(s):
    return ' '.join(unicodedata.normalize('NFC', s).casefold().split())

def words(s):
    return re.findall(r'\b[\w]+\b', s.casefold())

def canonical_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def stats(values):
    return {'min': min(values), 'max': max(values), 'mean': round(statistics.mean(values), 2), 'median': statistics.median(values)} if values else {}

def load(path):
    rows = json.loads(path.read_text(encoding='utf-8-sig'))
    assert isinstance(rows, list) and rows, 'Expected nonempty JSON array'
    seen = set()
    for r in rows:
        if not isinstance(r.get('id'), str) or r['id'] in seen or not isinstance(r.get('question'), str) or not r['question'].strip() or not isinstance(r.get('answer'), str) or type(r.get('is_impossible')) is not bool or not isinstance(r.get('contexts'), list):
            raise ValueError(f'Invalid or duplicate record: {r.get("id")}')
        for c in r['contexts']:
            if not isinstance(c, dict) or not isinstance(c.get('text'), str):
                raise ValueError(f'Invalid context: {r["id"]}')
        seen.add(r['id'])
    return rows

def inspect(rows):
    details, groups, docs, answers = [], defaultdict(list), defaultdict(set), defaultdict(list)
    product_counts = defaultdict(Counter)
    for r in rows:
        q, a, neg = r['question'], r['answer'], r['is_impossible']
        contexts = r['contexts']
        cls = 'impossible' if neg else 'answerable'
        flags = []
        products = [name for name, pat in PRODUCTS.items() if re.search(pat, q, re.I)] or ['Unclassified']
        for product in products:
            product_counts[product][cls] += 1
        paragraphs = [norm(p) for p in re.split(r'\n\s*\n', q) if p.strip()]
        if len(paragraphs) != len(set(paragraphs)): flags.append('repeated_question_paragraph')
        filenames = sorted({c.get('filename') or '(missing filename)' for c in contexts})
        for filename in filenames: docs[filename].add(r['id'])
        if neg:
            flags.append('closed_book_label_unverified')
            if contexts or a.strip() != '-': flags.append('unexpected_negative_fields')
        else:
            answers[norm(a)].append(r['id'])
            if not contexts or not a.strip() or a.strip() == '-': flags.append('missing_answer_or_context')
            if not any(norm(a) in norm(c['text']) for c in contexts): flags.append('answer_not_contiguous_in_context')
            if DOC_REF.search(a): flags.append('document_dependent_phrase')
            if APAR.search(a): flags.append('apar_or_identifier_mention')
            if URL.search(a): flags.append('contains_url')
            if URL.search(a) and not re.search(r'[a-z0-9]', URL.sub('', a), re.I): flags.append('url_only')
            if re.fullmatch(r'\s*[^\n]{1,150}\.(?:pdf|zip|docx?|txt|html?)\s*', a, re.I): flags.append('filename_only')
            if len(a.split()) <= 5: flags.append('answer_at_most_5_words')
        d = {'id': r['id'], 'label': cls, 'products': products, 'question_words': len(q.split()), 'answer_words': None if neg else len(a.split()), 'context_count':len(contexts), 'filenames': filenames, 'flags':flags, 'question':q,'answer':a,'contexts':contexts}
        details.append(d); groups[norm(q)].append(d)
    pairs = []
    # Token-set lexical similarity, not semantic equivalence. Ignore very short questions.
    token_sets = [set(words(d['question'])) for d in details]
    for i,j in itertools.combinations(range(len(details)),2):
        x,y=token_sets[i],token_sets[j]
        if min(len(x),len(y)) < 5: continue
        sim = len(x & y)/len(x | y)
        if sim >= .5:
            a,b=details[i],details[j]
            cross=a['label']!=b['label']
            pairs.append({'id_a':a['id'],'id_b':b['id'],'jaccard':round(sim,4),'cross_label':cross,'question_a':a['question'],'question_b':b['question']})
            if cross:
                for d in (a,b):
                    if 'lexically_similar_opposite_label' not in d['flags']: d['flags'].append('lexically_similar_opposite_label')
    pairs.sort(key=lambda p:(not p['cross_label'],-p['jaccard'],p['id_a'],p['id_b']))
    positives=[d for d in details if d['label']=='answerable']; negatives=[d for d in details if d['label']=='impossible']
    dup=[g for g in groups.values() if len(g)>1]
    summary={'canonical_sha256':canonical_hash(rows),'matches_reconstructed_stage1':canonical_hash(rows)==EXPECTED_HASH,'total':len(rows),'answerable':len(positives),'impossible':len(negatives),'groups':{},'product_keyword_counts':dict(product_counts),'distinct_context_filenames':len(docs),'document_frequency':dict(Counter(len(v) for v in docs.values())),'normalized_duplicate_question_groups':len(dup),'normalized_conflicting_label_groups':sum(len({d['label'] for d in g})>1 for g in dup),'duplicate_answer_groups':sum(len(v)>1 for v in answers.values()),'distinct_normalized_positive_answers':len(answers),'lexical_pairs_ge_0_5':len(pairs),'cross_label_lexical_pairs':sum(p['cross_label'] for p in pairs)}
    for name, ds in [('answerable',positives),('impossible',negatives)]:
        summary['groups'][name]={'question_words':stats([d['question_words'] for d in ds]),'answer_words':stats([d['answer_words'] for d in ds if d['answer_words'] is not None]),'with_context':sum(d['context_count']>0 for d in ds),'flags':dict(Counter(f for d in ds for f in d['flags']))}
    return summary,details,pairs

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,default=DEFAULT_TRAIN); p.add_argument('--out',type=Path); args=p.parse_args()
    rows=load(args.train); summary,details,pairs=inspect(rows)
    out=args.out or ROOT/'evaluate_results/106/finetuning/data_audit'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'records.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in details),encoding='utf-8')
    fields=['id','label','products','question_words','answer_words','context_count','filenames','flags','question','answer','evidence_verdict','answer_completeness','standalone_answer','label_verdict','review_notes','proposed_answer']
    with (out/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader()
        for d in details:
            w.writerow({**d,**{k:'; '.join(d[k]) for k in ('products','filenames','flags')}})
    with (out/'similar_questions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['id_a','id_b','jaccard','cross_label','question_a','question_b']); w.writeheader(); w.writerows(pairs)
    print('[Training Data Audit]'); print(json.dumps(summary,ensure_ascii=False,indent=2)); print('Results:',out)
    if not summary['matches_reconstructed_stage1']: print('WARNING: dataset content differs from reconstructed Stage 1 reference; do not compare as the same dataset.')

if __name__=='__main__': main()
