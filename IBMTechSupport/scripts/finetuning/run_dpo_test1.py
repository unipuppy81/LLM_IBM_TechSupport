"""DPO on 76 reviewed train-only preference pairs, starting from the 548-QA LoRA.
Place beside prepare_dpo_pairs.py in IBMTechSupport/scripts/finetuning/.
No validation-based checkpoint selection, Judge, RAG or test data.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import random
import shutil
import time
from datetime import datetime

from prepare_dpo_pairs import (
    ROOT, TRAIN, ADAPTER, SFT_RUN, CONFIG, OUT, SEED, sha, read, write,
    sources, adapter_hash,
)

PAIRS = ROOT / 'data/processed/train_dpo_v1.jsonl'
PAIR_HASH = '88a678f092d6b66aaad294186f3c718756cda28e6450bca7b0fe760d80de48cf'
PREP = OUT / 'prep_20261127_141846_344996'
CANDIDATE_HASH = '107ce016453a8db8738d1410c18be447abf74eb1d99fdf6deecfd8cf73d7a9f4'
VAL = ROOT / 'data/processed/val_106.json'
LR, BETA, EPOCHS, ACCUM = 5e-6, 0.1, 1, 8
MAX_LENGTH = 2048


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def validated_pairs(records, digest, cfg):
    if sha(PAIRS) != PAIR_HASH:
        raise ValueError('Approved preference dataset SHA-256 mismatch')
    candidate_file = PREP / 'preference_candidates.jsonl'
    if sha(candidate_file) != CANDIDATE_HASH:
        raise ValueError('Original candidate file SHA-256 mismatch')
    m = read(PREP / 'manifest.json')
    if (m['sft_adapter_sha256'] != digest or m['train_sha256'] != sha(TRAIN)
            or m['prompt'] != cfg['prompt'] or m['inference'] != cfg['inference']
            or m['validation_used'] is not False or m['test_used'] is not False):
        raise ValueError('DPO candidate provenance mismatch')
    rows = load_jsonl(PAIRS)
    candidates = {r['id']: r for r in load_jsonl(candidate_file)}
    train = {r['id']: r for r in records}
    if len(rows) != 76 or len({r['id'] for r in rows}) != 76 or len(candidates) != 120:
        raise ValueError('Expected 76 unique reviewed pairs from 120 candidates')
    for r in rows:
        t, c = train[r['id']], candidates[r['id']]
        if (t['is_impossible'] is not False or r['prompt'] != t['question']
                or r['chosen'] != t['answer'] or r['prompt'] != c['question']
                or r['chosen'] != c['chosen_candidate'] or r['rejected'] != c['rejected_candidate']
                or not r['chosen'].strip() or not r['rejected'].strip()
                or r['chosen'].strip() == r['rejected'].strip()):
            raise ValueError(f'Invalid preference pair: {r["id"]}')
    return rows


def encode_pairs(rows, tokenizer, cfg):
    encoded = []
    for r in rows:
        prefix = tokenizer.apply_chat_template([
            {'role': 'system', 'content': cfg['prompt']['system']},
            {'role': 'user', 'content': r['prompt']},
        ], tokenize=True, add_generation_prompt=True, return_dict=False)
        c = tokenizer.encode(r['chosen'], add_special_tokens=False) + [tokenizer.eos_token_id]
        n = tokenizer.encode(r['rejected'], add_special_tokens=False) + [tokenizer.eos_token_id]
        # Full responses, no truncation: a missing technical step can change the preference.
        if not prefix or min(len(c), len(n)) <= 1 or len(prefix) + max(len(c), len(n)) > MAX_LENGTH:
            raise ValueError(f'Pair exceeds token budget or is empty: {r["id"]}')
        if c == n:
            raise ValueError(f'Token-identical pair: {r["id"]}')
        encoded.append((r['id'], prefix, c, n))
    return encoded


def completion_logp(model, prefix, completion):
    import torch
    import torch.nn.functional as F
    ids = torch.tensor([prefix + completion], dtype=torch.long, device=model.device)
    # Causal shift: logits at the last prefix token predict the first answer token.
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits[0, len(prefix)-1:-1, :].float()
    targets = ids[0, len(prefix):]
    # Negative summed cross entropy = summed conditional completion log probability.
    return -F.cross_entropy(logits, targets, reduction='sum')


def main():
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from load_base_model import load_model
    from evaluate import evaluate_lora as ev
    from evaluate.common import score_predictions

    records, cfg, digest = sources()  # train-only source validation
    rows = validated_pairs(records, digest, cfg)
    baseline_dir = SFT_RUN / 'generation'
    baseline_manifest = read(baseline_dir / 'generation_manifest.json')
    if (baseline_manifest['adapter_sha256'] != digest or baseline_manifest['prompt'] != cfg['prompt']
            or baseline_manifest['inference'] != cfg['inference']
            or baseline_manifest['predictions_sha256'] != sha(baseline_dir / 'predictions.jsonl')):
        raise ValueError('SFT baseline provenance mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    run = OUT / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    adapter = ROOT / 'models/finetuning/dpo' / run.name / 'lora'
    run.mkdir(parents=True, exist_ok=False)
    if adapter.exists():
        raise FileExistsError(adapter)
    shutil.copyfile(PAIRS, run / 'train_dpo_v1.jsonl')
    settings = {'method': 'DPO', 'pairs': 76, 'beta': BETA, 'lr': LR, 'epochs': EPOCHS,
                'gradient_accumulation': ACCUM, 'batch_size_pairs': 1, 'seed': SEED,
                'max_length': MAX_LENGTH, 'dropout_disabled': True, 'weight_decay': 0.0,
                'reference': 'frozen initial SFT adapter; cached completion log probabilities',
                'completion_logp': 'sum including EOS; no truncation',
                'validation_checkpoint_selection': False}
    write(run / 'experiment.json', {'settings': settings, 'pair_sha256': sha(PAIRS),
          'sft_adapter_sha256': digest, 'prompt': cfg['prompt'], 'inference': cfg['inference'],
          'train_sha256': sha(TRAIN), 'validation_used_for_training': False, 'test_used': False})
    base, tokenizer, generation_config = load_model(cfg)
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=True, local_files_only=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != 1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise ValueError('Expected only original q/v LoRA parameters to be trainable')
    model.config.use_cache = False
    encoded = encode_pairs(rows, tokenizer, cfg)
    # Frozen reference is the initial policy, not the bare base model.
    # Cache the log probabilities before any optimizer step; no second GPU model needed.
    model.eval()
    reference = {}
    torch.cuda.synchronize()
    ref_start = time.perf_counter()
    print('[DPO Reference] caching 76 pairs', flush=True)
    with torch.no_grad():
        for i, (rid,p,c,n) in enumerate(encoded, 1):
            reference[rid] = (completion_logp(model,p,c).item(), completion_logp(model,p,n).item())
            if not all(math.isfinite(x) for x in reference[rid]):
                raise RuntimeError(f'Nonfinite reference score: {rid}')
            if i % 10 == 0 or i == len(encoded):
                print(f'Reference: {i}/{len(encoded)}', flush=True)
    torch.cuda.synchronize()
    ref_seconds = time.perf_counter() - ref_start
    write(run / 'reference_logps.json', reference)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    # Standard DPO disables dropout in both policy and reference.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.train()
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda')
    opt.zero_grad(set_to_none=True)
    order = list(encoded)
    random.Random(SEED).shuffle(order)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    logs, updates, skipped_updates = [], 0, 0
    for i,(rid,p,c,n) in enumerate(order,1):
        # Two-pass exact gradient of the DPO objective: calculate detached margin,
        # then differentiate each completion separately to reduce peak memory.
        with torch.no_grad():
            pc, pn = completion_logp(model,p,c).item(), completion_logp(model,p,n).item()
        rc, rn = reference[rid]
        margin = BETA*((pc-pn)-(rc-rn))
        if not math.isfinite(margin):
            raise RuntimeError(f'Nonfinite DPO margin: {rid}')
        z = torch.tensor(margin, dtype=torch.float32)
        loss = F.softplus(-z).item()
        coeff = BETA*torch.sigmoid(-z).item()
        window = min(ACCUM, len(order)-((i-1)//ACCUM)*ACCUM)
        # d loss / d log p(chosen) = -beta sigmoid(-margin), rejected opposite.
        scaler.scale(-coeff*completion_logp(model,p,c)/window).backward()
        scaler.scale(coeff*completion_logp(model,p,n)/window).backward()
        logs.append({'id':rid, 'loss':loss,'policy_chosen_logp':pc,'policy_rejected_logp':pn,
                     'reference_chosen_logp':rc,'reference_rejected_logp':rn,'reward_margin':margin})
        if i % ACCUM == 0 or i == len(order):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            old_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped_updates += 1
            else:
                updates += 1
            opt.zero_grad(set_to_none=True)
            print(f'DPO: {i}/{len(order)} | Loss: {loss:.4f} | Updates: {updates}', flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter()-start
    peak = torch.cuda.max_memory_allocated()/1024**3
    if updates == 0:
        raise RuntimeError('All optimizer steps skipped; no DPO adapter saved')
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    if adapter_hash() != digest or sha(PAIRS) != PAIR_HASH:
        raise RuntimeError('Original adapter or preference dataset changed')
    write(run / 'training_cases.json', logs)
    train_metrics = {'pairs':len(order),'epochs':1,'optimizer_updates':updates,'skipped_updates':skipped_updates,
        'mean_dpo_loss':sum(x['loss'] for x in logs)/len(logs), 'reference_seconds':round(ref_seconds,3),
        'training_seconds':round(train_seconds,3),'peak_gpu_memory_gb':round(peak,3)}
    write(run / 'training_metrics.json', train_metrics)
    del opt, scaler, trainable, model, base, tokenizer, generation_config
    gc.collect()
    torch.cuda.empty_cache()

    # Validation read only after final adapter is saved. No tuning/checkpoint selection.
    if sha(VAL) != baseline_manifest['dataset_sha256']:
        raise ValueError('Validation dataset differs from SFT baseline')
    val = read(VAL)
    if len(val)!=106 or sum(r['is_impossible'] is False for r in val)!=77 or sum(r['is_impossible'] is True for r in val)!=29:
        raise ValueError('Expected frozen 106-QA validation dataset')
    old_rows = load_jsonl(baseline_dir/'predictions.jsonl')
    old = read(baseline_dir/'metrics.json')
    old_scores = score_predictions(val, old_rows, 'val_106.json', 1)
    for k in ('token_f1_percent','exact_match_percent','abstention_rate_percent','false_refusal_rate_percent'):
        if old[k]!=old_scores[k]:
            raise ValueError(f'SFT baseline metric mismatch: {k}')
    ev.ADAPTER, ev.OUTPUT, ev.BASE_CONFIG = adapter, run/'generation', CONFIG
    ev.generate(copy.deepcopy(cfg), val)
    new = read(ev.OUTPUT/'metrics.json')
    report = {'baseline_generation':old,'dpo_generation':new,'dpo_training':train_metrics,
              'judge':'NOT RUN','rag':False,'test_used':False,'complete':True}
    write(run/'comparison.json',report)
    lines = ['[DPO Test1 Comparison]',f'{"Metric":40} {"SFT 548":>15} {"SFT+DPO 76":>15}']
    for k in ('token_f1_percent','exact_match_percent','abstention_rate_percent','false_refusal_rate_percent',
              'generation_seconds','questions_per_second','peak_gpu_memory_gb'):
        lines.append(f'{k:40} {str(old[k]):>15} {str(new[k]):>15}')
    lines += ['DPO Training: '+json.dumps(train_metrics), 'Judge: NOT RUN | RAG: NONE | Test: NOT USED',
              f'Adapter: {adapter}', f'Results: {run}']
    (run/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
