"""Inference-only prompt comparison on the frozen Data Composition Test2 LoRA.
Place in IBMTechSupport/scripts/finetuning/. No Judge calls, training or retrieval.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SOURCE_RUN = 'run_20261127_000005_052233'
SOURCE = ROOT / 'evaluate_results/106/finetuning/data_composition_test2' / SOURCE_RUN / 'experiments/A_expanded/lora'
ADAPTER = ROOT / 'models/finetuning/data_composition_test2' / SOURCE_RUN / 'A_expanded/lora'
RUNS = ROOT / 'evaluate_results/106/finetuning/generation_prompt_test1'
SYSTEM = '''You are an IBM technical support assistant. Answer in English using only your existing knowledge.
Answer the exact question directly. Do not repeat the question or merely restate the error. Do not announce a list or procedure without providing it.
For troubleshooting, give the specific cause or actionable resolution requested. For configuration or installation, provide the necessary settings or steps. For compatibility, state the supported versions or limits when known. For security bulletins, identify the relevant vulnerability details or a reliable source, not just the bulletin title.
Preserve applicable product names, versions, error codes, commands, paths and numerical limits. Do not invent technical details. If a condition matters, give a reliable conditional answer when possible.
If you cannot reliably answer the essential request, respond exactly INSUFFICIENT_INFORMATION. Otherwise output only the technical answer. Do not output analysis or introductory filler.'''
METRICS = ('token_f1_percent', 'exact_match_percent', 'abstention_rate_percent',
           'false_refusal_rate_percent', 'generation_seconds',
           'mean_generation_seconds_per_question', 'questions_per_second', 'peak_gpu_memory_gb')
INTRO_ONLY = re.compile(r'^(?:the following (?:steps|information|is a (?:summary|list)|are (?:the )?(?:steps|details))[^\n]*|here (?:are|is)[^\n]*)[:.]?$', re.I)


def normalized(text):
    return ' '.join(re.findall(r'\w+|[^\w\s]', text.casefold()))


def flags(question, answer):
    """Mechanical review flags only. These do not establish technical correctness."""
    if answer == 'INSUFFICIENT_INFORMATION':
        return []
    found = []
    if not answer.strip():
        return ['empty_answer']
    q, a = normalized(question), normalized(answer)
    if a == q or (len(a.split()) >= 8 and a in q):
        found.append('verbatim_question_fragment')
    if '\n' not in answer.strip() and INTRO_ONLY.fullmatch(answer.strip()):
        found.append('possible_intro_only')
    lines = [normalized(re.sub(r'^\s*(?:\d+[.)]|[-*])\s*', '', x)) for x in answer.splitlines()]
    lines = [x for x in lines if len(x.split()) >= 4]
    if len(lines) != len(set(lines)):
        found.append('repeated_line')
    return found


def main():
    from evaluate import evaluate_lora as ev
    from evaluate import evaluate_stage2_quality as q
    from evaluate.common import score_predictions, save_json, save_jsonl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', type=Path, default=ADAPTER, help='Relocated adapter, checked against original SHA-256')
    args = p.parse_args()
    cfg, records = ev.inputs()
    old_manifest = q.load_json(SOURCE / 'generation_manifest.json')
    ev.ADAPTER = args.adapter.resolve()
    expected = {'dataset_sha256': q.sha256(ev.DATASET), 'expected_count': 106,
                'predictions_sha256': q.sha256(SOURCE / 'predictions.jsonl'),
                'base_config_sha256': q.sha256(ev.BASE_CONFIG),
                'adapter_sha256': ev.adapter_hash(), 'inference': cfg['inference']}
    for key, value in expected.items():
        if old_manifest.get(key) != value:
            raise ValueError(f'Original Test2 provenance mismatch: {key}')
    old_prompt = old_manifest['prompt']
    if old_prompt.get('user_template') != '{question}':
        raise ValueError('Expected closed-book question-only template')
    old = q.load_jsonl(SOURCE / 'predictions.jsonl')
    old_scores = q.load_json(SOURCE / 'metrics.json')
    recomputed = score_predictions(records, old, 'val_106.json', 1)
    for key in METRICS[:4]:
        if old_scores[key] != recomputed[key]:
            raise ValueError(f'Original metric mismatch: {key}')
    cfg = copy.deepcopy(cfg)
    cfg['prompt'] = {**old_prompt, 'system': SYSTEM}
    root = RUNS / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir(parents=True, exist_ok=False)
    ev.OUTPUT = root / 'B_direct_answer'
    save_json(root / 'experiment.json', {'source': str(SOURCE), 'adapter': str(ev.ADAPTER),
              'baseline_manifest': old_manifest, 'candidate_prompt': cfg['prompt'],
              'runner_sha256': q.sha256(Path(__file__)), 'new_training': False,
              'retrieval': False, 'judge_evaluation': False,
              'baseline_reused': True, 'evaluation_scope': 'validation only'})
    ev.generate(cfg, records)
    new = q.load_jsonl(ev.OUTPUT / 'predictions.jsonl')
    new_scores = q.load_json(ev.OUTPUT / 'metrics.json')
    score_predictions(records, new, 'val_106.json', 1)
    new_manifest = q.load_json(ev.OUTPUT / 'generation_manifest.json')
    for key in ('dataset_sha256','expected_count','base_config_sha256','adapter_sha256','inference'):
        if new_manifest[key] != old_manifest[key]:
            raise ValueError(f'Candidate provenance mismatch: {key}')
    pairs, blinded, key_rows = [], [], []
    rng = random.Random(42)
    for r, a, b in zip(records, old, new):
        af, bf = flags(r['question'], a['prediction']), flags(r['question'], b['prediction'])
        pair = {'id': r['id'], 'question': r['question'], 'reference_answer': r['answer'],
                'is_impossible': r['is_impossible'], 'reference_documents': [c['filename'] for c in r['contexts']],
                'baseline': {'prediction': a['prediction'], 'token_f1': a['token_f1'], 'flags': af},
                'candidate': {'prediction': b['prediction'], 'token_f1': b['token_f1'], 'flags': bf}}
        pairs.append(pair)
        if not r['is_impossible']:
            swap = rng.choice([True, False])
            x,y = (b,a) if swap else (a,b)
            blinded.append({'id':r['id'],'question':r['question'],'reference_answer':r['answer'],
                            'reference_documents': '\n'.join(c['filename'] for c in r['contexts']),
                            'reference_evidence': '\n\n'.join(c['text'] for c in r['contexts']),
                            'answer_A':x['prediction'],'answer_B':y['prediction'],
                            'verdict_A':'','verdict_B':'','reason_A':'','reason_B':''})
            key_rows.append({'id':r['id'],'A':'candidate' if swap else 'baseline','B':'baseline' if swap else 'candidate'})
    save_jsonl(root / 'answer_comparison.jsonl', pairs)
    save_json(root / 'review_key.json', key_rows)
    with (root / 'blind_review.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(blinded[0])); w.writeheader(); w.writerows(blinded)
    from collections import Counter
    def summarize(side):
        rows = [x for x in pairs if not x['is_impossible']]
        c = Counter(flag for x in rows for flag in x[side]['flags'])
        return {'flag_counts':dict(c),'flagged_answerable_count':sum(bool(x[side]['flags']) for x in rows)}
    counts = Counter('improved' if x['candidate']['token_f1'] > x['baseline']['token_f1'] else
                     'declined' if x['candidate']['token_f1'] < x['baseline']['token_f1'] else 'tied'
                     for x in pairs if not x['is_impossible'])
    report = {'baseline':{k:old_scores.get(k) for k in METRICS},
              'candidate':{k:new_scores.get(k) for k in METRICS},
              'baseline_diagnostics':summarize('baseline'),'candidate_diagnostics':summarize('candidate'),
              'per_answer_token_f1_change_rounded':dict(counts),
              'answer_changed_count':sum(x['baseline']['prediction'] != x['candidate']['prediction'] for x in pairs),
              'answerable_count':77,'impossible_count':29,'correctness_judge':'NOT RUN',
              'technical_accuracy':'UNVERIFIED; blind review required',
              'limitations':['F1/EM measure lexical overlap, not technical correctness.',
                             'Mechanical flags are heuristic and can miss defects or flag valid answers.',
                             'Baseline latency is historical and may not be comparable across machines.',
                             'Validation prompt tuning is not independent test evaluation.',
                             'Base weights are not fully hashed by the original generation manifest.'],
              'complete':True}
    save_json(root / 'comparison.json', report)
    lines = ['[Generation Prompt Test1]', f'{"Metric":42} {"Baseline":>14} {"B_direct":>14}']
    for k in METRICS:
        lines.append(f'{k:42} {str(old_scores.get(k)):>14} {str(new_scores.get(k)):>14}')
    for k in ('baseline','candidate'):
        lines.append(f'{k} diagnostics: {report[k+"_diagnostics"]}')
    lines.extend([f'Per-answer F1 changes: {dict(counts)}',f'Changed answers: {report["answer_changed_count"]}/106',
                  'Correctness Judge: NOT RUN | Training: NONE | RAG: NONE',
                  'Technical review: blind_review.csv (77 pairs), review_key.json (unblind after scoring)',
                  f'Results: {root}'])
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
