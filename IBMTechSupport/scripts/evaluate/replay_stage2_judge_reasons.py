"""Replay V3 failed per-document judgments with an explicit reason output.
Place beside evaluate_stage2_prompt.py and analyze_stage2_judge_failures.py.
No retrieval, generator, verifier, historical cache or production source changes.
The generated rationale is a diagnostic self-report, not hidden model reasoning.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

import evaluate_stage2_prompt as ab
import analyze_stage2_judge_failures as diag

ORIGINAL_END = 'Respond with exactly one word: YES, NO, or UNCERTAIN.'
REASON_END = '''Respond with YES, NO, or UNCERTAIN alone on the first line.
On the second line, briefly state the decisive evidence or missing requirement for your decision. Identify the relevant technical condition or requested operation and cite a concrete detail from the supplied excerpt when possible. Use at most two sentences. Do not invent evidence. No markdown.'''
MAX_NEW_TOKENS = 128


def reason_prompt(prompt):
    if not prompt.endswith(ORIGINAL_END) or prompt.count(ORIGINAL_END) != 1:
        raise ValueError('Unexpected original output instruction; do not silently alter the evaluation criteria')
    return prompt[:-len(ORIGINAL_END)] + REASON_END


def parse(raw):
    lines = raw.strip().splitlines()
    first = lines[0].strip().upper() if lines else ''
    decision = first if first in diag.DECISIONS else None
    reason = '\n'.join(lines[1:]).strip()
    return decision, reason, bool(decision and reason)


def infer(model, tokenizer, system, user):
    import torch
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt').to(model.device)
    width = inputs['input_ids'].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    generated = output[0, width:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    decision, reason, valid = parse(raw)
    return {'decision': decision, 'reason': reason, 'format_valid': valid,
            'raw': raw, 'generated_tokens': int(generated.numel()),
            'token_limit_reached': int(generated.numel()) >= MAX_NEW_TOKENS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(diag.PARENT.resolve()):
        parser.error('Use an existing Stage2 validation run')
    exp = diag.read(run/'experiment.json')
    comparison = diag.read(run/'comparison.json')
    if comparison.get('status') != 'COMPLETED' or comparison.get('mode') != 'judge_all' or exp.get('mode') != 'judge_all':
        raise ValueError('Expected a completed judge_all run')
    prompt = (run/'candidate_prompt.txt').read_text(encoding='utf-8-sig').strip()
    if prompt != exp['candidate_prompt'].strip():
        raise ValueError('Original candidate prompt mismatch')
    system = reason_prompt(prompt)
    config_path, config, records, predictions, manifest, verified = ab.load_inputs(args)
    if config['guardrails'].get('judge_mode') != 'per_document':
        raise ValueError('Expected per_document mode')
    if ab.digest(config_path) != exp['config_sha256'] or ab.digest(ab.DATA) != exp['dataset_sha256']:
        raise ValueError('Original config/dataset mismatch')
    if not verified:
        raise ValueError('Frozen original predictions manifest is required')
    for filename, sha in exp['source_hashes'].items():
        if ab.digest(ab.SCRIPTS/filename) != sha:
            raise ValueError(f'Source changed since V3 run: {filename}')
    logs = diag.index(diag.rows(run/'judge_cases.jsonl'))
    by_id = diag.index(predictions)
    ids = {r['id'] for r in records}
    if set(logs) != ids or set(exp['sample_ids']) != ids or len(exp['sample_ids']) != 106:
        raise ValueError('Validation ID mismatch')
    failures = []
    for r in records:
        c = diag.make_case(r, by_id[r['id']], logs[r['id']])
        if c:
            failures.append(c)
    counts = Counter(c['category'] for c in failures)
    expected = comparison['judge_only_proxy']['candidate']
    if expected['total'] != 106 or counts['ANSWERABLE_BLOCKED'] != expected['answerable_blocked'] or counts['IMPOSSIBLE_ALLOWED'] != expected['impossible'] - expected['impossible_blocked']:
        raise ValueError('Failure count mismatch')
    sys.path.insert(0, str(ab.SCRIPTS))
    import rag_guardrails as gates
    from load_base_model import load_model
    from evaluate_stage2_prompt_fast import model_identity
    import torch
    import transformers
    identity = model_identity(ab.ROOT/config['model']['path'])
    if identity != exp['model_identity_file_metadata']:
        raise ValueError('Model file metadata changed since original run; restore original local model before replay')
    model, tokenizer, _ = load_model(config)
    model.eval()
    out = Path(mkdtemp(prefix='judge_reason_replay_', dir=run))
    total = sum(c['candidate_evaluated_count'] for c in failures)
    metadata = {'status':'RUNNING', 'source_run':str(run), 'cases':len(failures), 'chunks':total,
        'original_prompt':prompt, 'reason_prompt':system, 'max_new_tokens':MAX_NEW_TOKENS,
        'original_max_new_tokens':12, 'do_sample':False, 'num_beams':1,
        'source_judge_cases_sha256':ab.digest(run/'judge_cases.jsonl'),
        'model_metadata_matched':True, 'torch':str(torch.__version__), 'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'note':'Reason output is a new generation, not the original hidden reasoning. No gold answer, dataset label, or prior decision is included in model input. Replayed chunks are fixed to the original evaluated ranks; not a new end-to-end metric.'}
    ab.write_json(out/'summary.json', metadata)
    print('Cases:',len(failures),'| Chunks:',total,'| Output:',out,flush=True)
    stats = Counter()
    transitions = Counter()
    start = time.perf_counter()
    with (out/'judge_reasons.jsonl').open('w',encoding='utf-8') as jf, (out/'judge_reason_report.txt').open('w',encoding='utf-8-sig') as tf:
        tf.write('V3 Judge 판정 근거 재생성 (내부 추론 기록 아님)\n'+metadata['note']+'\n\n')
        for i,c in enumerate(failures,1):
            tf.write('='*72+'\n'+c['id']+' | '+c['category']+'\n질문: '+c['question']+'\n정답: '+str(c['reference_answer'])+'\n기존 질문 판정: '+c['candidate_decision']+'\n')
            decisions = []
            for d in c['retrieved_documents']:
                old = d['candidate_judge']
                if old is None:
                    continue
                # Render original document dictionary, not the diagnostic record.
                original_doc = by_id[c['id']]['retrieved_documents'][d['rank']-1]
                user = f'Question:\n{c["question"]}\n\nRetrieved Documents:\n{gates._render_documents([original_doc])}'
                result = infer(model, tokenizer, system, user)
                same = result['decision'] == old['decision'] if result['decision'] else None
                status = 'INVALID_FORMAT' if not result['format_valid'] else ('MATCH' if same else 'CHANGED')
                row = {'id':c['id'], 'category':c['category'], 'question':c['question'],
                    'rank':d['rank'], 'filename':d['filename'], 'chunk_id':d['chunk_id'],
                    'original_decision':old['decision'], 'original_raw':old['raw'],
                    'diagnostic_decision':result['decision'], 'decision_matches':same,
                    'comparison_status':status, 'reason':result['reason'], 'raw':result['raw'],
                    'format_valid':result['format_valid'], 'generated_tokens':result['generated_tokens'],
                    'token_limit_reached':result['token_limit_reached'], 'excerpt':d['text']}
                jf.write(json.dumps(row,ensure_ascii=False)+'\n'); jf.flush()
                tf.write(f'\nRank {d["rank"]} | {d["filename"]} | {d["chunk_id"]}\n기존: {old["decision"]} / 재평가: {result["decision"]} / {status}\n근거: {result["reason"] or "(출력 없음)"}\n원본 출력: {result["raw"]}\n[검색 청크]\n{d["text"]}\n')
                tf.flush()
                stats[status] += 1
                stats['PROCESSED'] += 1
                if result['token_limit_reached']: stats['TOKEN_LIMIT_REACHED'] += 1
                transitions[f'{old["decision"]}->{result["decision"] or "INVALID"}'] += 1
                decisions.append(result['decision'])
            tf.write('\n재평가 청크 판정: '+str(decisions)+'\n\n')
            print(f'[Judge reason] {i}/{len(failures)} | {stats["PROCESSED"]}/{total} chunks',flush=True)
    metadata.update(status='COMPLETED', counts=dict(stats), transitions=dict(transitions), wall_seconds=round(time.perf_counter()-start,2))
    ab.write_json(out/'summary.json',metadata)
    print('Saved:',out,flush=True)
    print('MATCH:',stats['MATCH'],'CHANGED:',stats['CHANGED'],'INVALID_FORMAT:',stats['INVALID_FORMAT'],flush=True)


if __name__ == '__main__':
    main()
