"""Stage 1: audit the real Stage 0 dataset and QADataset implementation.
Place next to run_stage0.py in IBMTechSupport/scripts/finetuning/.
No GPU training, test data, source edits, or overwriting of the baseline.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import run_stage0 as s0

ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage1'
CANDIDATES = ROOT / 'data/processed/finetuning/stage1'


def progress(n, name):
    print(f'[Experiment] {n}/6 | {name}', flush=True)


def load_baseline(path=None):
    if path is None:
        completed = [p for p in s0.RUNS.glob('run_*') if (p / 'comparison.json').is_file()]
        if not completed:
            raise FileNotFoundError('No completed Stage 0 run')
        path = max(completed, key=lambda p: p.name)
    path = Path(path).resolve()
    if path.parent != s0.RUNS.resolve():
        raise ValueError('Stage 0 run must be inside the Stage 0 results directory')
    for name in ('generate_base', 'quality_base', 'train_lora', 'generate_lora', 'quality_lora',
                 'train_qlora', 'generate_qlora', 'quality_qlora'):
        if not (path / 'completed' / f'{name}.json').is_file():
            raise ValueError(f'Incomplete Stage 0 run: {name}')
    baseline = s0.read(path / 'manifest.json')['provenance']
    for name, file in [('train_sha256', s0.TRAIN), ('val_sha256', s0.VAL)]:
        if baseline['data'][name] != s0.sha(file):
            raise ValueError(f'Dataset changed since Stage 0: {file}')
    for name, digest in baseline['sources'].items():
        if s0.sha(ROOT / name) != digest:
            raise ValueError(f'Source changed since Stage 0: {name}')
    actual_model = {p.name: s0.sha(p) for p in s0.model_files()}
    if baseline['base_model_files'] != actual_model:
        raise ValueError('Base model files changed since Stage 0')
    if baseline['settings'] != s0.SETTINGS or baseline['baseline'] != s0.BASELINE:
        raise ValueError('Baseline settings changed')
    return path, baseline


def duplicate_audit(rows):
    groups = defaultdict(list)
    exact_content_seen = set()
    exact_duplicates = []
    exact_sft_seen = set()
    sft_duplicates = []
    for r in rows:
        groups[s0.norm(r['question'])].append(r)
        key = json.dumps({k: r[k] for k in ('question', 'answer', 'is_impossible', 'contexts')},
                         sort_keys=True, ensure_ascii=False)
        if key in exact_content_seen:
            exact_duplicates.append(r['id'])
        exact_content_seen.add(key)
        key = (r['question'], r['is_impossible'],
               'INSUFFICIENT_INFORMATION' if r['is_impossible'] else r['answer'])
        if key in exact_sft_seen:
            sft_duplicates.append(r['id'])
        exact_sft_seen.add(key)
    conflicts, equivalent, alternative = [], [], []
    for items in groups.values():
        if len(items) < 2:
            continue
        entry = {'ids': [r['id'] for r in items], 'records': items}
        if len({r['is_impossible'] for r in items}) > 1:
            conflicts.append(entry)
        elif len({r['answer'] for r in items}) == 1:
            equivalent.append(entry)
        else:
            alternative.append(entry)
    conflict_ids = {rid for g in conflicts for rid in g['ids']}
    return {'exact_content_duplicate_ids': exact_duplicates,
            'exact_sft_duplicate_ids': sft_duplicates,
            'normalized_duplicate_extra_rows': len(rows) - len(groups),
            'label_conflict_groups': conflicts, 'same_answer_groups': equivalent,
            'different_answer_groups': alternative,
            'conflict_ids': sorted(conflict_ids)}


def counts(rows):
    a = sum(not r['is_impossible'] for r in rows)
    return {'total': len(rows), 'answerable': a, 'impossible': len(rows)-a}


def length_stats(values):
    return {'count': len(values), 'min': min(values) if values else None,
            'max': max(values) if values else None,
            'mean': round(statistics.mean(values), 2) if values else None,
            'median': statistics.median(values) if values else None,
            'total': sum(values)}


def token_audit(rows, tokenizer, train, cfg, split):
    """Compare real QADataset tensors against inference and full chat templates."""
    flags = defaultdict(list)
    lengths = defaultdict(list)
    details = []
    special = set(tokenizer.all_special_ids)
    for i, r in enumerate(rows, 1):
        rid = r['id']
        q = r['question']
        target = train.ABSTAIN if r['is_impossible'] else r['answer']
        qt = tokenizer.encode(q, add_special_tokens=False)
        at = tokenizer.encode(target, add_special_tokens=False)
        trimmed = tokenizer.decode(qt[:train.MAX_QUESTION_TOKENS], skip_special_tokens=True)
        sample = train.QADataset([r], tokenizer, s0.SETTINGS['max_length'])[0]
        ids = sample['input_ids'].reshape(-1).tolist()
        labels = sample['labels'].reshape(-1).tolist()
        mask = sample['attention_mask'].reshape(-1).tolist()
        start = next((j for j, v in enumerate(labels) if v != -100), len(labels))
        prefix, supervised = ids[:start], labels[start:]
        messages = [{'role': 'system', 'content': train.SYSTEM_PROMPT},
                    {'role': 'user', 'content': trimmed}]
        expected_prefix = tokenizer.apply_chat_template(messages, tokenize=True,
                                   add_generation_prompt=True, return_dict=False)
        inference_messages = [{'role': 'system', 'content': cfg['prompt']['system']},
                              {'role': 'user', 'content': cfg['prompt']['user_template'].format(question=q)}]
        inference_prefix = tokenizer.apply_chat_template(inference_messages, tokenize=True,
                                   add_generation_prompt=True, return_dict=False)
        remaining = s0.SETTINGS['max_length'] - len(expected_prefix) - 1
        expected_target = at[:min(train.MAX_ANSWER_TOKENS, remaining)] + [tokenizer.eos_token_id]
        full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': target}],
                           tokenize=True, add_generation_prompt=False, return_dict=False)
        checks = {
            'shape_or_attention_error': len(ids)!=len(labels) or len(ids)!=len(mask) or any(x!=1 for x in mask),
            'label_mask_error': start == 0 or start == len(labels) or labels[:start] != [-100]*start
                                or supervised != ids[start:] or -100 in supervised,
            'prefix_mismatch': prefix != expected_prefix,
            'target_mismatch': supervised != expected_target,
            'eos_error': not supervised or supervised[-1] != tokenizer.eos_token_id,
            'empty_supervised_answer': len(supervised) <= 1,
            'sequence_overflow': len(ids) > s0.SETTINGS['max_length'],
            'question_truncated': len(qt)>train.MAX_QUESTION_TOKENS,
            'question_roundtrip_changed': len(qt)<=train.MAX_QUESTION_TOKENS and q != trimmed,
            'answer_truncated': len(supervised)-1 < len(at),
            'train_inference_prefix_differs': prefix != inference_prefix,
            'embedded_special_token_question': any(x in special for x in qt),
            'embedded_special_token_answer': any(x in special for x in at),
            # Full assistant template may have a newline after EOS; that is not an error.
            'full_assistant_template_prefix_differs': full[:len(prefix)] != prefix,
            'full_assistant_template_target_differs': full[:len(ids)] != ids if len(supervised)-1 == len(at) else False,
        }
        for key, value in checks.items():
            if value:
                flags[key].append(rid)
        cls = 'impossible' if r['is_impossible'] else 'answerable'
        lengths[cls].append(max(0, len(supervised)-1))
        details.append({'id':rid, 'is_impossible':r['is_impossible'], 'question_tokens':len(qt),
                        'raw_target_tokens':len(at), 'supervised_answer_tokens':max(0,len(supervised)-1),
                        'supervised_tokens_with_eos':len(supervised),'prompt_tokens':start,
                        'sequence_tokens':len(ids),'flags':[k for k,v in checks.items() if v]})
        if i==1 or i%50==0 or i==len(rows):
            print(f'[Audit {split}] Sample {i}/{len(rows)}',flush=True)
    hard = ('shape_or_attention_error','label_mask_error','prefix_mismatch','target_mismatch',
            'eos_error','empty_supervised_answer','sequence_overflow')
    return {'samples':len(rows), 'flags':dict(flags), 'flag_counts':{k:len(v) for k,v in flags.items()},
            'implementation_error_rows':sorted({rid for k in hard for rid in flags[k]}),
            'target_lengths':{k:length_stats(v) for k,v in lengths.items()}, 'details':details}


def quality_audit(rows):
    flags = defaultdict(list)
    for i,r in enumerate(rows,1):
        a, q = r['answer'], r['question']
        ctx = r['contexts']
        if r['is_impossible']:
            if ctx: flags['impossible_has_context'].append(r['id'])
            if a.strip() not in ('','-','INSUFFICIENT_INFORMATION'):
                flags['impossible_unexpected_raw_answer'].append(r['id'])
        else:
            if not ctx: flags['answerable_missing_context'].append(r['id'])
            if s0.norm(a) not in s0.norm(' '.join(c['text'] for c in ctx)):
                flags['reference_not_verbatim_in_context'].append(r['id'])
            if a.strip() in ('','-','INSUFFICIENT_INFORMATION'):
                flags['answerable_refusal_or_placeholder'].append(r['id'])
            if 'INSUFFICIENT_INFORMATION' in a:
                flags['answer_contains_refusal_marker'].append(r['id'])
        if any(not c['text'].strip() or not c['filename'].strip() for c in ctx):
            flags['empty_context_field'].append(r['id'])
        if q != q.strip() or a != a.strip():
            flags['outer_whitespace'].append(r['id'])
        if i==1 or i%100==0 or i==len(rows):
            print(f'[Audit quality] Sample {i}/{len(rows)}',flush=True)
    return {'flags':dict(flags),'flag_counts':{k:len(v) for k,v in flags.items()},
            'impossible_raw_answer_values':dict(Counter(r['answer'] for r in rows if r['is_impossible'])),
            'note':'Verbatim mismatch is a review flag, not proof of an incorrect answer. No semantic quality judgement was made.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,help='Completed Stage 0 run; defaults to latest')
    args=parser.parse_args()
    baseline_path, baseline=load_baseline(args.baseline)
    run = RUNS / ('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    print(f'[Stage 1] Baseline: {baseline_path.name}',flush=True)
    progress(1,'Data duplication / leakage')
    data = s0.audit()
    rows,val=s0.validate(s0.TRAIN),s0.validate(s0.VAL)
    dup=duplicate_audit(rows)
    s0.write(run/'duplicate_review.json',dup)
    progress(2,'Chat template')
    from transformers import AutoTokenizer
    import train_lora as train
    from evaluate import evaluate_lora as eval_lora
    cfg,_=eval_lora.inputs()
    tokenizer=AutoTokenizer.from_pretrained(s0.BASE,local_files_only=True)
    template={'same_system_prompt':train.SYSTEM_PROMPT==cfg['prompt']['system'],
              'user_template':cfg['prompt']['user_template'],
              'eos_token':tokenizer.eos_token,'eos_token_id':tokenizer.eos_token_id,
              'pad_token_id':tokenizer.pad_token_id,
              'refusal_target_token_ids':tokenizer.encode(train.ABSTAIN,add_special_tokens=False)}
    # The actual tensors are inspected once, then reused for masking and EOS checks.
    audits={'train':token_audit(rows,tokenizer,train,cfg,'train'),
            'validation':token_audit(val,tokenizer,train,cfg,'validation')}
    s0.write(run/'token_audit.json',{'template':template,**audits})
    progress(3,'Label masking')
    masking={k:len(v['flags'].get('label_mask_error',[])) for k,v in audits.items()}
    progress(4,'EOS handling')
    eos={k:len(v['flags'].get('eos_error',[])) for k,v in audits.items()}
    progress(5,'Data quality')
    quality=quality_audit(rows)
    progress(6,'Answer format / candidate preparation')
    # Preserve original file. Candidate A changes only exact SFT duplicate frequency.
    # Candidate B independently quarantines both sides of contradictory labels, not relabeling them.
    candidates={}
    cand_dir=CANDIDATES/run.name
    for name,removed in [('exact_dedup',set(dup['exact_sft_duplicate_ids'])),
                         ('conflict_quarantine',set(dup['conflict_ids']))]:
        kept=[r for r in rows if r['id'] not in removed]
        path=cand_dir/f'{name}.json'
        s0.write(path,kept)
        candidates[name]={'path':str(path),'sha256':s0.sha(path),'counts':counts(kept),'removed_ids':sorted(removed)}
    review_ids=set(dup['conflict_ids'])|{rid for g in dup['different_answer_groups'] for rid in g['ids']}|{rid for ids in quality['flags'].values() for rid in ids}
    s0.jsonl(run/'data_review.jsonl',[r for r in rows if r['id'] in review_ids])
    summary={'stage':1,'status':'AUDIT_COMPLETE_NO_TRAINING','baseline_run':str(baseline_path),
             'baseline_manifest_sha256':s0.sha(baseline_path/'manifest.json'),
             'data_audit':data,'duplicate_summary':{'exact_content_extra_rows':len(dup['exact_content_duplicate_ids']),
             'exact_sft_extra_rows':len(dup['exact_sft_duplicate_ids']),'normalized_extra_rows':dup['normalized_duplicate_extra_rows'],
             'conflicting_label_groups':len(dup['label_conflict_groups']),'conflicting_label_rows':len(dup['conflict_ids']),
             'same_answer_groups':len(dup['same_answer_groups']),'different_answer_groups':len(dup['different_answer_groups'])},
             'template':template,'label_mask_errors':masking,'eos_errors':eos,
             'token_summary':{k:{x:y for x,y in v.items() if x not in ('flags','details')} for k,v in audits.items()},
             'quality':quality,'candidate_datasets':candidates,
             'review_count':len(review_ids), 'performance_measured':False,
             'notes':['No semantic near-duplicate or technical correctness judgement was performed.',
                      'No changes to original training/evaluation data or code.',
                      'Candidates are separate one-variable changes relative to Stage 0; neither is adopted yet.']}
    s0.write(run/'report.json',summary)
    lines=['[Stage 1 Audit Summary]',f'Train / Validation: {len(rows)} / {len(val)}',
           f'Cross-split ID / normalized overlap: {len(data["cross_split_id_matches"])} / {len(data["cross_split_normalized_question_matches"])}',
           *[f'{k}: {v}' for k,v in summary['duplicate_summary'].items()],
           f'System prompt identical: {template["same_system_prompt"]}',
           f'Label masking errors (train/val): {masking["train"]}/{masking["validation"]}',
           f'EOS errors (train/val): {eos["train"]}/{eos["validation"]}']
    for k,v in audits.items():
        lines.append(f'{k} token flags: '+json.dumps(v['flag_counts'],ensure_ascii=False))
        lines.append(f'{k} target token lengths: '+json.dumps(v['target_lengths'],ensure_ascii=False))
        lines.append(f'{k} implementation error rows: {len(v["implementation_error_rows"])}')
    lines += ['Quality flags: '+json.dumps(quality['flag_counts'],ensure_ascii=False),
              'Raw impossible answer values: '+json.dumps(quality['impossible_raw_answer_values'],ensure_ascii=False),
              *[f'Candidate {n}: {v["counts"]}' for n,v in candidates.items()],
              f'Review rows: {len(review_ids)}','Training: NOT RUN',f'Results: {run}']
    (run/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
