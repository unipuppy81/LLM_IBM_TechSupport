from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
from datetime import datetime
from pathlib import Path

import run_data_composition_test1 as t1

s0, t2, quality, weighted = t1.s0, t1.t2, t1.quality, t1.weighted
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/data_composition_test2'
SCRIPT = Path(__file__).resolve()
BASELINE = t1.RUNS / 'run_20261126_231847_375481'
NAME = 'A_expanded'
PHASES = ('train', 'generate', 'quality')

# Questions ask for different facts, not paraphrases of the original question.
SECTIONS = {
    'CAUSE': 'What causes the issue described as',
    'SYMPTOM': 'What symptoms identify the issue described as',
    'DIAGNOSING THE PROBLEM': 'How can an administrator diagnose the issue described as',
    'RESOLVING THE PROBLEM': 'How can an administrator resolve the issue described as',
    'ERROR DESCRIPTION': 'Under what technical conditions does the issue occur in',
    'LOCAL FIX': 'What local workaround is available for',
    'TEMPORARY FIX': 'What temporary fix is available for',
    'PROBLEM CONCLUSION': 'What permanent correction or fix is described for',
    'AFFECTED PRODUCTS AND VERSIONS': 'Which products and versions are affected by',
    'REMEDIATION/FIXES': 'What remediation is recommended for',
    'WORKAROUNDS AND MITIGATIONS': 'What workarounds or mitigations are available for',
    'ENVIRONMENT': 'What environment is relevant to the issue described as',
    'INSTALLATION INSTRUCTIONS': 'What installation steps are specified for',
    'PREREQUISITES': 'What prerequisites are required for',
    'REQUIREMENTS': 'What requirements apply to',
}
BOUNDARIES = set(SECTIONS) | {
    'ANSWER', 'QUESTION', 'ABSTRACT', 'SUMMARY', 'PROBLEM', 'PROBLEM(ABSTRACT)',
    'PROBLEM SUMMARY', 'RELATED INFORMATION', 'REFERENCES', 'DISCLAIMER',
    'CHANGE HISTORY', 'PRODUCT ALIAS/SYNONYM', 'VULNERABILITY DETAILS',
    'SECURITY BULLETIN', 'CONTENT', 'IMPORTANT NOTE', 'DOWNLOAD DESCRIPTION',
    'DOWNLOAD PACKAGE', 'DOWNLOADABLE FILES', 'HISTORICAL NUMBER', 'TAB NAVIGATION',
    'APAR STATUS', 'APAR INFORMATION', 'FIX INFORMATION', 'FIXES ARE AVAILABLE',
    'COMMENTS', 'SUBSCRIBE', 'APPLICABLE COMPONENT LEVELS', 'CROSS REFERENCE INFORMATION',
    'DOCUMENT INFORMATION', 'PRODUCT INFORMATION', 'MODIFIED DATE', 'COPYRIGHT',
}


def new_root(prefix):
    root = RUNS / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=False)
    return root


def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT / 'models/finetuning/data_composition_test2' / run.parent.parent.name / run.name / method


def words(text):
    return set(re.findall(r'[a-z0-9]+', text.casefold()))


def similarity(a, b):
    return len(a & b) / max(1, len(a | b))


def fingerprint(text):
    return hashlib.sha256(s0.norm(text).encode('utf-8')).hexdigest()


def section_blocks(text):
    current = None
    lines = []
    for line in text.splitlines():
        name = line.strip().upper()
        # Stop also at unknown all-capital metadata headings. Do not cut
        # commands or list items that happen to use uppercase letters.
        is_heading = (name in BOUNDARIES or
                      (line.strip() == name and re.fullmatch(r'[A-Z][A-Z /-]{3,42}', name)
                       and len(name.split()) >= 2))
        if is_heading:
            if current:
                yield current, '\n'.join(lines).strip()
            current, lines = name, []
        elif current:
            if re.match(r'^\s*(?:Cross reference information|Document information|Modified date)\b', line, re.I):
                yield current, '\n'.join(lines).strip()
                current, lines = None, []
            else:
                lines.append(line)
    if current:
        yield current, '\n'.join(lines).strip()


def proposals(text):
    title_match = re.search(r'^Title:\s*(.+)$', text, re.M)
    if not title_match:
        return
    title = re.sub(r'\s*-\s*United States\s*$', '', title_match[1].strip(), flags=re.I)
    if len(title) < 12 or len(title) > 300:
        return
    question = None
    for section, body in section_blocks(text):
        if section == 'QUESTION':
            question = body
        elif section == 'ANSWER' and question:
            # A FAQ may already contain a useful alternative technical question.
            if len(question.split()) <= 100 and '?' in question and not re.search(r'(?im)^\s*Question\s+\d+', body):
                yield 'FAQ', question.strip(), body
            question = None
        elif section in SECTIONS:
            yield section, f'{SECTIONS[section]} "{title}"?', body
            question = None
        else:
            question = None


def generate_candidates(positives, val, tokenizer=None, spec=None, extra_questions=()):
    # Only training-source contexts are harvested. Validation is used solely
    # for exclusion, never as source text or a source of generated answers.
    blocked_names = {Path(c['filename']).name.casefold() for r in val for c in r['contexts']}
    blocked_texts = {fingerprint(c['text']) for r in val for c in r['contexts']}
    docs = {}
    for row in positives:
        for c in row['contexts']:
            key = Path(c['filename']).name.casefold()
            docs[(key, fingerprint(c['text']))] = c
    seen_q = [words(r['question']) for r in positives + val] + [words(q) for q in extra_questions]
    seen_a = [words(r['answer']) for r in positives]
    q_exact = {s0.norm(r['question']) for r in positives + val} | {s0.norm(q) for q in extra_questions}
    a_exact = {s0.norm(r['answer']) for r in positives}
    accepted, stats = [], {'documents':len(docs), 'blocked_documents':0, 'proposals':0,
                           'length_or_format':0, 'similar_question':0, 'similar_answer':0,
                           'accepted':0}
    for (name, digest), c in sorted(docs.items()):
        if name in blocked_names or digest in blocked_texts:
            stats['blocked_documents'] += 1
            continue
        for section, question, answer in proposals(c['text']):
            stats['proposals'] += 1
            answer = answer.strip()
            question = question.strip()
            if (not 8 <= len(answer.split()) <= 115 or not 5 <= len(question.split()) <= 100
                    or len(answer) > 1600 or re.match(r'(?i)^(?:none|n/?a|not applicable|no fix)\.?$', answer)
                    or re.search(r'(?i)\b(?:see below|as above|following figure|click here|refer to the following link)\b', answer)
                    or re.match(r'(?i)^(?:https?://|www\.)', answer)
                    or answer not in c['text']):
                stats['length_or_format'] += 1
                continue
            if tokenizer is not None and (len(tokenizer.encode(question, add_special_tokens=False)) > spec['question_tokens']
                    or len(tokenizer.encode(answer, add_special_tokens=False)) > spec['answer_tokens']):
                stats['length_or_format'] += 1
                continue
            q, a = words(question), words(answer)
            if s0.norm(question) in q_exact or any(similarity(q, old) >= .85 for old in seen_q):
                stats['similar_question'] += 1
                continue
            if s0.norm(answer) in a_exact or any(similarity(a, old) >= .80 or (len(a) >= 6 and a <= old) for old in seen_a):
                stats['similar_answer'] += 1
                continue
            if a <= q:
                stats['similar_answer'] += 1
                continue
            rid = 'SYN_' + hashlib.sha256((name+'\0'+section+'\0'+question+'\0'+answer).encode()).hexdigest()[:20]
            accepted.append({'record':{'id':rid,'question':question,'answer':answer,'is_impossible':False,'contexts':[c]},
                             'section':section,'source_filename':c['filename'],'source_sha256':digest})
            seen_q.append(q); seen_a.append(a)
            q_exact.add(s0.norm(question)); a_exact.add(s0.norm(answer))
    stats['accepted'] = len(accepted)
    return accepted, stats


def load_baseline(path):
    path = path.resolve()
    m = s0.read(path/'manifest.json')
    run = path/'experiments'/t1.NAME
    t1.verify_snapshot(run, m)
    for phase in PHASES:
        weighted.previous.check_done(run, phase+'_lora')
    positives = s0.validate(run/'train.json')
    if len(positives) != 367 or any(r['is_impossible'] for r in positives):
        raise ValueError('Expected completed 367-positive Test1')
    result = quality.metrics(run, 'lora')
    result['trainable_parameters'] = t1.check_adapter(run, m)
    comparison = s0.read(path/'comparison.json')
    if not comparison['complete'] or any(comparison['candidates'][t1.NAME].get(k) != v for k,v in result.items()):
        raise ValueError('Test1 metrics mismatch')
    return m, positives, result


def prepare(baseline):
    m, positives, result = load_baseline(baseline)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(s0.BASE, local_files_only=True)
    original = s0.validate(Path(m['training_source']))
    val = s0.validate(s0.VAL)
    candidates, stats = generate_candidates(positives, val, tok, m['spec'],
                                            [r['question'] for r in original if r['is_impossible']])
    root = new_root('prep_')
    s0.write(root/'candidates.json', candidates)
    with (root/'review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['id','decision','section','source_filename','question','answer']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            r = c['record']
            writer.writerow(dict(id=r['id'],decision='',section=c['section'],source_filename=c['source_filename'],question=r['question'],answer=r['answer']))
    s0.write(root/'manifest.json', {'baseline':str(baseline.resolve()),'baseline_metrics':result,
              'baseline_manifest_sha256':s0.sha(baseline/'manifest.json'),
              'validation_sha256':s0.sha(s0.VAL),'candidates_sha256':s0.sha(root/'candidates.json'),
              'script_sha256':s0.sha(SCRIPT),'stats':stats})
    print('[Data Expansion Preparation]')
    print(f'Original answerable: {len(positives)} | Candidate QA: {len(candidates)} | Maximum before review: {len(positives)+len(candidates)}')
    for key, value in stats.items():
        print(f'{key}: {value}')
    print(f'Results: {root}')
    print('Review: set decision=accept or reject in review.csv. No training has been run.')


def reviewed(prep):
    p = s0.read(prep/'manifest.json')
    if s0.sha(SCRIPT) != p['script_sha256'] or s0.sha(s0.VAL) != p['validation_sha256'] or s0.sha(prep/'candidates.json') != p['candidates_sha256']:
        raise ValueError('Preparation snapshot changed')
    baseline = Path(p['baseline'])
    if s0.sha(baseline/'manifest.json') != p['baseline_manifest_sha256']:
        raise ValueError('Baseline manifest changed')
    m, positives, result = load_baseline(baseline)
    if result != p['baseline_metrics']:
        raise ValueError('Baseline metrics changed')
    candidates = {c['record']['id']:c for c in s0.read(prep/'candidates.json')}
    accepted, seen = [], set()
    with (prep/'review.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            rid = row['id']
            if rid in seen or rid not in candidates:
                raise ValueError('Unknown or duplicate review ID')
            seen.add(rid)
            c = candidates[rid]; r = c['record']
            if any(row[k] != r[k] for k in ('question','answer')) or row['source_filename'] != c['source_filename'] or row['section'] != c['section']:
                raise ValueError(f'Review must only change decision: {rid}')
            decision = row['decision'].strip().casefold()
            if decision not in ('accept','reject'):
                raise ValueError(f'Unreviewed candidate: {rid}')
            if decision == 'accept':
                accepted.append(r)
    if seen != set(candidates):
        raise ValueError('Review CSV must retain every candidate')
    if not accepted:
        raise ValueError('No accepted new QA')
    random.Random(42).shuffle(accepted)
    return m, positives, result, accepted


def verify(run, m):
    for path, digest in m['runner_hashes'].items():
        if s0.sha(Path(path)) != digest:
            raise ValueError(f'Runner changed: {path}')
    for path, digest in ((s0.VAL,m['data']['val_sha256']), (run/'train.json',m['training_sha256']),
                         (Path(m['preparation'])/'review.csv',m['review_sha256'])):
        if s0.sha(path) != digest:
            raise ValueError(f'Snapshot changed: {path}')
    if s0.read(t2.config_for(run)) != m['config']:
        raise ValueError('Config changed')
    rows = s0.validate(run/'train.json')
    if len(rows) != m['candidate_count'] or any(r['is_impossible'] for r in rows):
        raise ValueError('Training composition changed')


def worker(run, phase):
    m = s0.read(run.parent.parent/'manifest.json')
    verify(run,m)
    s0.SETTINGS = dict(m['settings'])
    weighted.adapter = quality.adapter = s0.adapter = adapter
    if phase == 'train':
        t2.configure_trainer(m['spec'])
        quality.worker(run,'train','lora')
    else:
        weighted.worker(run,phase)
    if t1.previous.check_adapter(adapter(run,'lora'),t1.previous.BASE_TARGETS) != m['baseline_metrics']['trainable_parameters']:
        raise ValueError('Unexpected adapter parameter count')


def execute(root):
    run = root/'experiments'/NAME
    m = s0.read(root/'manifest.json')
    verify(run,m)
    print(f'[Data Composition Test2] Run: {root}',flush=True)
    print(f'[Experiment] 1/1 | {m["candidate_count"]} answerable QA',flush=True)
    try:
        for phase in PHASES:
            old_file, old_adapter = weighted.__file__, weighted.adapter
            try:
                weighted.__file__, weighted.adapter = str(SCRIPT), adapter
                weighted.execute_phase(run,phase)
            finally:
                weighted.__file__, weighted.adapter = old_file, old_adapter
        c = quality.metrics(run,'lora')
        c['trainable_parameters'] = t1.previous.check_adapter(adapter(run,'lora'),t1.previous.BASE_TARGETS)
        b = m['baseline_metrics']
        for row in (b,c):
            row['youden_j_percent_points'] = t1.previous.lr.j(row)
        keys = tuple(dict.fromkeys(('train_samples',*t2.KEYS,'youden_j_percent_points','trainable_parameters')))
        s0.write(root/'comparison.json',{'baseline_367':b,'expanded':c,'complete':True,
            'optimizer_updates_per_epoch':{'baseline':46,'expanded':math.ceil(m['candidate_count']/8)},
            'human_review_required':True,'qlora':'NOT RUN','diagnostic_only':True})
        def fmt(v): return 'N/A' if v is None else f'{v:.4f}'.rstrip('0').rstrip('.')
        lines = ['[Data Composition Test2 Comparison]',f'{"Metric":36} {"Baseline_367":>16} {"Expanded":>16}']
        lines += [f'{k:36} {fmt(b.get(k)):>16} {fmt(c.get(k)):>16}' for k in keys]
        lines += ['Quality: COMPLETE | Complete: True','Diagnostic only; no automatic selection.',f'Results: {root}']
        (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('\n'.join(lines),flush=True)
    except BaseException:
        print(f'Resume: python scripts/finetuning/{SCRIPT.name} --resume "{root}"',flush=True)
        raise


def start(prep,size):
    prep = prep.resolve()
    m, positives, result, accepted = reviewed(prep)
    available = len(positives)+len(accepted)
    if size is None:
        size = available
    if not len(positives) < size <= available:
        raise ValueError(f'Requested size {size}; approved maximum {available}. Do not duplicate data to reach a target.')
    root = new_root('run_')
    run = root/'experiments'/NAME
    rows = positives + accepted[:size-len(positives)]
    s0.write(run/'train.json',rows)
    s0.write(t2.config_for(run),m['config'])
    m['runner_hashes'][str(SCRIPT)] = s0.sha(SCRIPT)
    m.update(stage='data_composition',test=2,experiment='expanded_positive',baseline_metrics=result,
             preparation=str(prep),review_sha256=s0.sha(prep/'review.csv'),candidate_count=size,
             training_sha256=s0.sha(run/'train.json'))
    s0.write(root/'manifest.json',m)
    execute(root)


def main():
    p = argparse.ArgumentParser(description='Data Composition Test2: extractive training QA expansion')
    p.add_argument('--baseline',type=Path,default=BASELINE)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--train',type=Path,metavar='PREPARATION')
    modes.add_argument('--resume',type=Path)
    modes.add_argument('--worker',choices=PHASES,help=argparse.SUPPRESS)
    p.add_argument('--size',type=int)
    p.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        if not a.run or a.run.resolve().parent.parent.parent != RUNS.resolve() or a.run.name != NAME:
            p.error('Invalid worker directory')
        worker(a.run.resolve(),a.worker)
    elif a.resume:
        if a.resume.resolve().parent != RUNS.resolve(): p.error('Invalid resume directory')
        execute(a.resume.resolve())
    elif a.train:
        if a.train.resolve().parent != RUNS.resolve(): p.error('Invalid preparation directory')
        start(a.train,a.size)
    else:
        if a.size is not None: p.error('--size requires --train')
        prepare(a.baseline.resolve())


if __name__ == '__main__':
    main()
