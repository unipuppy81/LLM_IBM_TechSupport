"""Generate train-only DPO preference candidates from the frozen 548-QA SFT adapter.
Place in IBMTechSupport/scripts/finetuning/. Run from repository root.
No validation/test data, judge, RAG, weight updates, or automatic preference labels.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
TRAIN = ROOT / 'data/processed/train_clean_ver2.json'
EXPECTED_TRAIN_HASH = '9372bbf386fa57b18ff67ac8b8e19be2cbad398f3fc33010485dcf953e86e5c7'
SFT_RUN = ROOT / 'evaluate_results/106/finetuning/train_clean_ver2/run_20261127_134955_078761'
ADAPTER = ROOT / 'models/finetuning/train_clean_ver2/run_20261127_134955_078761/lora'
CONFIG = ROOT / 'evaluate_results/106/finetuning/data_composition_test2/run_20261127_000005_052233/experiments/A_expanded/prompt_config.yaml'
SOURCE_MANIFEST = CONFIG.parents[2] / 'manifest.json'
OUT = ROOT / 'evaluate_results/106/finetuning/dpo'
SEED = 42
ABSTAIN = 'INSUFFICIENT_INFORMATION'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm(text):
    # Normalization only detects near-identical surface forms; never labels errors.
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()


def adapter_hash():
    return {name: sha(ADAPTER / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}


def sources():
    import yaml
    train_hash = sha(TRAIN)
    if train_hash != EXPECTED_TRAIN_HASH:
        raise ValueError('train_clean_ver2.json SHA-256 mismatch. Do not substitute another dataset.')
    records = read(TRAIN)
    if len(records) != 548 or len({r['id'] for r in records}) != 548:
        raise ValueError('Expected 548 unique train records')
    for r in records:
        if r['is_impossible'] is not False or not isinstance(r['question'], str) or not r['question'].strip() or not isinstance(r['answer'], str) or not r['answer'].strip():
            raise ValueError(f'Invalid training record: {r["id"]}')
    exp = read(SFT_RUN / 'experiment.json')
    done = read(SFT_RUN / 'training_complete.json')
    if exp['training_sha256'] != train_hash or done['metrics']['train_samples'] != 548:
        raise ValueError('SFT training provenance mismatch')
    if sha(SFT_RUN / 'train.json') != train_hash:
        raise ValueError('Saved SFT training snapshot mismatch')
    digest = adapter_hash()
    if digest != done['adapter_sha256']:
        raise ValueError('SFT adapter differs from the completed 548-QA run')
    adapter_config = read(ADAPTER / 'adapter_config.json')
    if (adapter_config['r'], adapter_config['lora_alpha'], adapter_config['lora_dropout'], sorted(adapter_config['target_modules'])) != (8,16,.05,['q_proj','v_proj']):
        raise ValueError('Unexpected SFT adapter configuration')
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8-sig'))
    source = read(SOURCE_MANIFEST)
    if cfg != source['config'] or cfg['prompt'] != exp['prompt'] or cfg['inference'] != exp['inference']:
        raise ValueError('Original SFT prompt or inference configuration mismatch')
    inf = cfg['inference']
    if cfg['prompt']['user_template'] != '{question}' or (inf['max_new_tokens'], inf['do_sample'], inf['num_beams'], inf['batch_size']) != (160, False, 1, 1):
        raise ValueError('Expected original closed-book greedy inference')
    base = ROOT / cfg['model']['path']
    if base.resolve() != (ROOT / 'models/Qwen2.5-1.5B-Instruct').resolve():
        raise ValueError('Unexpected base model')
    for name, expected in source['model_files'].items():
        if sha(base / name) != expected:
            raise ValueError(f'Base model hash mismatch: {name}')
    return records, cfg, digest


def select_records(records, tokenizer, cfg, count):
    eligible = []
    excluded = Counter()
    for r in records:
        contexts = r.get('contexts')
        if not isinstance(contexts, list) or not any(isinstance(c, dict) and isinstance(c.get('text'), str) and c['text'].strip() for c in contexts):
            excluded['no_train_evidence'] += 1
            continue
        qlen = len(tokenizer.encode(r['question'], add_special_tokens=False))
        alen = len(tokenizer.encode(r['answer'], add_special_tokens=False))
        # Preserve both complete question and chosen answer. No label truncation.
        if qlen > 192 or alen > 160 or alen == 0:
            excluded['exceeds_sft_question_or_answer_limit'] += 1
            continue
        prompt_ids = tokenizer.apply_chat_template([
            {'role':'system','content':cfg['prompt']['system']},
            {'role':'user','content':r['question']},
        ], tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(prompt_ids) + 161 > 2048:
            excluded['sequence_too_long'] += 1
            continue
        eligible.append(r)
    rng = random.Random(SEED)
    rng.shuffle(eligible)
    # Prefer different train source documents in the initial candidate sample.
    # This does not consult any evaluation document list.
    unique, rest, seen_docs, seen_q = [], [], set(), set()
    for r in eligible:
        q = norm(r['question'])
        if q in seen_q:
            excluded['duplicate_train_question'] += 1
            continue
        seen_q.add(q)
        filenames = {str(c.get('filename','')).strip().casefold() for c in r['contexts'] if isinstance(c,dict) and c.get('filename')}
        if filenames and not filenames.intersection(seen_docs):
            unique.append(r)
            seen_docs.update(filenames)
        else:
            rest.append(r)
    selected = (unique + rest)[:count]
    if not selected:
        raise ValueError('No eligible train questions')
    return selected, dict(excluded), len(unique)+len(rest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=120, help='Maximum candidate questions (train only)')
    args = p.parse_args()
    if not 1 <= args.count <= 548:
        p.error('--count must be between 1 and 548')
    import torch
    from peft import PeftModel
    from load_base_model import load_model
    from evaluate.evaluate_base import generate_batch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    records, cfg, digest = sources()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base, tokenizer, gen_cfg = load_model(copy.deepcopy(cfg))
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False, local_files_only=True)
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Candidate model must be completely frozen')
    selected, excluded, eligible = select_records(records, tokenizer, cfg, args.count)
    run = OUT / ('prep_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    manifest = {'phase':'train_only_candidate_generation', 'train_sha256':sha(TRAIN), 'sft_adapter_sha256':digest,
                'sft_run':str(SFT_RUN), 'config_sha256':sha(CONFIG), 'prompt':cfg['prompt'], 'inference':cfg['inference'],
                'seed':SEED, 'requested_count':args.count, 'eligible_train_questions':eligible,
                'selected_ids':[r['id'] for r in selected], 'selection_exclusions':excluded,
                'validation_used':False, 'test_used':False, 'judge_used':False, 'rag_used':False,
                'preference_labels':'UNREVIEWED', 'training_performed':False}
    write(run/'manifest.json', manifest)
    path = run/'preference_candidates.jsonl'
    counts = Counter()
    torch.cuda.synchronize()
    start = time.perf_counter()
    print(f'[DPO candidate generation] {len(selected)} train questions | no validation/test', flush=True)
    with path.open('x', encoding='utf-8') as f:
        for i, r in enumerate(selected, 1):
            candidate = generate_batch(model, tokenizer, gen_cfg, [r['question']], cfg['prompt'])[0]
            if not candidate.strip():
                status = 'empty_output'
            elif norm(candidate) == norm(ABSTAIN):
                status = 'refusal_output'
            elif norm(candidate) == norm(r['answer']):
                status = 'identical_output'
            else:
                status = 'needs_review'
            counts[status] += 1
            row = {'id':r['id'], 'question':r['question'], 'chosen_candidate':r['answer'],
                   'rejected_candidate':candidate, 'contexts':r['contexts'], 'status':status,
                   'decision':'pending' if status=='needs_review' else 'skip', 'reason':'',
                   'source':'train_clean_ver2.json', 'candidate_model':'548-QA SFT LoRA'}
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
            f.flush()
            if i == 1 or i % 10 == 0 or i == len(selected):
                print(f'Generated: {i}/{len(selected)} | review candidates: {counts["needs_review"]}', flush=True)
    torch.cuda.synchronize()
    if adapter_hash() != digest or sha(TRAIN) != EXPECTED_TRAIN_HASH:
        raise RuntimeError('Source weights or dataset changed during preparation')
    summary = {'complete':True, 'selected':len(selected), 'counts':dict(counts),
               'approved_preference_pairs':0, 'seconds':round(time.perf_counter()-start,3),
               'candidates_sha256':sha(path), 'validation_used':False, 'test_used':False,
               'next':'Review both answers against train source evidence; approve only genuinely better chosen answers. Do not train on unreviewed pairs.'}
    write(run/'summary.json', summary)
    print('\n[DPO Preparation Result]', flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Review file:', path, flush=True)
    print('DPO training: NOT RUN', flush=True)


if __name__ == '__main__':
    main()
