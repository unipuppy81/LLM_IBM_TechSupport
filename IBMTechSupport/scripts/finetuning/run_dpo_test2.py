"""Run two independent DPO candidates, evaluate four models and export one review bundle.
Place beside run_dpo_test1.py and prepare_dpo_pairs.py in scripts/finetuning/.
Run once from repository root. No extra packages, Judge, RAG or test set.
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

import run_dpo_test1 as t
from prepare_dpo_pairs import ROOT, TRAIN, ADAPTER, SFT_RUN, CONFIG, OUT, SEED, sha, read, write, sources, adapter_hash

TEST1 = OUT / 'run_20261127_143501_864689'
TEST1_ADAPTER = ROOT / 'models/finetuning/dpo' / TEST1.name / 'lora'
CANDIDATES = {'A_lr_2e5': {'lr': 2e-5, 'epochs': 1}, 'B_epoch3': {'lr': 5e-6, 'epochs': 3}}
KEYS = ('token_f1_percent', 'exact_match_percent', 'abstention_rate_percent', 'false_refusal_rate_percent',
        'generation_seconds', 'questions_per_second', 'peak_gpu_memory_gb')


def weights(path):
    return {n: sha(path/n) for n in ('adapter_config.json', 'adapter_model.safetensors')}


def verify_generation(folder, adapter, cfg, val, score_predictions):
    manifest = read(folder/'generation_manifest.json')
    expected = {'adapter_sha256': weights(adapter), 'prompt': cfg['prompt'], 'inference': cfg['inference'],
                'dataset_sha256': sha(t.VAL), 'expected_count': len(val),
                'predictions_sha256': sha(folder/'predictions.jsonl')}
    for k,v in expected.items():
        if manifest.get(k) != v:
            raise ValueError(f'{folder}: generation provenance mismatch: {k}')
    rows = t.load_jsonl(folder/'predictions.jsonl')
    metrics = read(folder/'metrics.json')
    scores = score_predictions(val, rows, 'val_106.json', 1)
    if any(metrics[k] != scores[k] for k in KEYS[:4]):
        raise ValueError(f'{folder}: saved generation metrics mismatch')
    return metrics, rows


def setup_model(cfg, training):
    import torch
    from peft import PeftModel
    from load_base_model import load_model
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base, tokenizer, _ = load_model(copy.deepcopy(cfg))
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=training, local_files_only=True)
    model.config.use_cache = False
    model.eval()
    if training:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if sum(p.numel() for p in trainable) != 1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
            raise ValueError('Unexpected trainable parameters')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.train()
    return model, tokenizer


def cache_reference(rows, cfg, run):
    import torch
    model, tokenizer = setup_model(cfg, False)
    encoded = t.encode_pairs(rows, tokenizer, cfg)
    reference = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for rid,p,c,n in encoded:
            pair = [t.completion_logp(model,p,c).item(), t.completion_logp(model,p,n).item()]
            if not all(math.isfinite(x) for x in pair):
                raise RuntimeError(f'Nonfinite reference score: {rid}')
            reference[rid] = pair
    torch.cuda.synchronize()
    elapsed = round(time.perf_counter()-start,3)
    write(run/'reference_logps.json', reference)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return encoded, reference, elapsed


def train_candidate(name, spec, rows, cfg, encoded, reference, run, model_root):
    import torch
    import torch.nn.functional as F
    out = run/name
    out.mkdir(exist_ok=False)
    adapter = model_root/name/'lora'
    if adapter.exists():
        raise FileExistsError(adapter)
    model, tokenizer = setup_model(cfg, True)
    # Verify tokenization is unchanged across the frozen reference and both policies.
    if t.encode_pairs(rows, tokenizer, cfg) != encoded:
        raise RuntimeError('Reference and policy tokenization mismatch')
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=spec['lr'], weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda')
    opt.zero_grad(set_to_none=True)
    rng = random.Random(SEED)
    logs, history = [], []
    updates = skipped = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, spec['epochs']+1):
        order = list(encoded)
        rng.shuffle(order)
        epoch_losses = []
        for i,(rid,p,c,n) in enumerate(order,1):
            with torch.no_grad():
                pc,pn = t.completion_logp(model,p,c).item(), t.completion_logp(model,p,n).item()
            rc,rn = reference[rid]
            margin = t.BETA*((pc-pn)-(rc-rn))
            if not math.isfinite(margin):
                raise RuntimeError(f'Nonfinite margin: {rid}')
            z = torch.tensor(margin,dtype=torch.float32)
            loss = F.softplus(-z).item()
            coeff = t.BETA*torch.sigmoid(-z).item()
            window = min(t.ACCUM,len(order)-((i-1)//t.ACCUM)*t.ACCUM)
            scaler.scale(-coeff*t.completion_logp(model,p,c)/window).backward()
            scaler.scale(coeff*t.completion_logp(model,p,n)/window).backward()
            epoch_losses.append(loss)
            logs.append({'epoch':epoch,'id':rid,'loss':loss,'reward_margin':margin,
                         'policy_chosen_logp':pc,'policy_rejected_logp':pn,
                         'reference_chosen_logp':rc,'reference_rejected_logp':rn})
            if i % t.ACCUM == 0 or i == len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable,1.0)
                scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < scale:
                    skipped += 1
                else:
                    updates += 1
                opt.zero_grad(set_to_none=True)
        history.append({'epoch':epoch,'mean_dpo_loss':sum(epoch_losses)/len(epoch_losses),
                        'cumulative_updates':updates,'cumulative_skipped_updates':skipped})
        print(f'{name} | Epoch {epoch}/{spec["epochs"]} | Loss {history[-1]["mean_dpo_loss"]:.6f} | Updates {updates}',flush=True)
    torch.cuda.synchronize()
    elapsed = round(time.perf_counter()-start,3)
    peak = round(torch.cuda.max_memory_allocated()/1024**3,3)
    expected = math.ceil(len(rows)/t.ACCUM)*spec['epochs']
    if updates == 0 or updates+skipped != expected:
        raise RuntimeError('Unexpected optimizer update count')
    model.save_pretrained(adapter,safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    metrics = {**spec,'pairs':len(rows),'beta':t.BETA,'optimizer_updates':updates,'skipped_updates':skipped,
               'mean_dpo_loss':sum(x['loss'] for x in logs)/len(logs),'training_seconds':elapsed,
               'peak_gpu_memory_gb':peak,'history':history}
    write(out/'training_metrics.json',metrics)
    write(out/'training_cases.json',logs)
    write(out/'training_complete.json',{'adapter_sha256':weights(adapter),'settings':spec})
    del opt,scaler,trainable,model,tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter,metrics


def export_review(run, val, predictions, report):
    # Anonymous per-question labels; retain a key within the same upload bundle.
    # Do not inspect the key when rating individual answers.
    rng = random.Random(SEED)
    by_model = {name:{r['id']:r for r in rows} for name,rows in predictions.items()}
    cases,key = [],{}
    for r in val:
        if r['is_impossible']:
            continue
        order = list(by_model)
        rng.shuffle(order)
        labels = dict(zip('ABCD',order))
        key[r['id']] = labels
        cases.append({'id':r['id'],'question':r['question'],'reference_answer':r['answer'],
                      'contexts':r.get('contexts',[]),
                      'answers':{label:by_model[name][r['id']]['prediction'] for label,name in labels.items()}})
    if len(cases) != 77:
        raise ValueError('Expected 77 answerable review cases')
    path = run/'dpo_test2_review_bundle.json'
    write(path,{'purpose':'Evaluation only. Never use these validation cases for training or preference pairs.',
                'review_instructions':'Rate each anonymous answer independently against question and evidence, then reveal mapping. CORRECT/PARTIAL/INCORRECT/UNCERTAIN. No automatic technical verdict.',
                'cases':cases,'comparison':report,'review_key':key})
    return path


def main():
    import torch
    from evaluate import evaluate_lora as ev
    from evaluate.common import score_predictions
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    records,cfg,digest = sources()
    rows = t.validated_pairs(records,digest,cfg)
    if len(rows) != 76:
        raise ValueError('Expected approved 76 preference pairs')
    old_exp = read(TEST1/'experiment.json')
    if (old_exp['pair_sha256'] != sha(t.PAIRS) or old_exp['sft_adapter_sha256'] != digest
            or old_exp['prompt'] != cfg['prompt'] or old_exp['inference'] != cfg['inference']
            or old_exp['settings']['lr'] != 5e-6 or old_exp['settings']['epochs'] != 1
            or old_exp['settings']['beta'] != t.BETA or old_exp['settings']['gradient_accumulation'] != t.ACCUM):
        raise ValueError('DPO Test1 provenance mismatch')
    # Confirm files exist before launching lengthy training, without reading validation rows.
    for path in (t.VAL, SFT_RUN/'generation/predictions.jsonl', TEST1/'generation/predictions.jsonl',
                 TEST1/'generation/generation_manifest.json', TEST1_ADAPTER/'adapter_model.safetensors'):
        if not path.is_file():
            raise FileNotFoundError(path)
    run = OUT/('test2_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    model_root = ROOT/'models/finetuning/dpo'/run.name
    run.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(t.PAIRS,run/'train_dpo_v1.jsonl')
    write(run/'experiment.json',{'candidates':CANDIDATES,'seed':SEED,'beta':t.BETA,'gradient_accumulation':t.ACCUM,
          'max_length':t.MAX_LENGTH,'pair_sha256':sha(t.PAIRS),'sft_adapter_sha256':digest,'prompt':cfg['prompt'],
          'inference':cfg['inference'],'reference':'initial SFT, one shared cache','validation_used_for_training':False,
          'test_used':False,'judge_used':False,'rag_used':False})
    print('[DPO Test2] Two independent candidates; one shared reference cache',flush=True)
    encoded,reference,ref_seconds = cache_reference(rows,cfg,run)
    adapters = {'SFT_548':ADAPTER,'DPO_Test1':TEST1_ADAPTER}
    training = {'DPO_Test1':read(TEST1/'training_metrics.json')}
    for name,spec in CANDIDATES.items():
        adapters[name],training[name] = train_candidate(name,spec,rows,cfg,encoded,reference,run,model_root)
        if adapter_hash() != digest or sha(t.PAIRS) != t.PAIR_HASH:
            raise RuntimeError('Original weights or dataset changed')
    # No validation read before both final candidates are saved.
    val = read(t.VAL)
    if len(val)!=106 or sum(r['is_impossible'] is False for r in val)!=77 or sum(r['is_impossible'] is True for r in val)!=29:
        raise ValueError('Unexpected validation dataset')
    metrics,predictions = {},{}
    for name,folder in (('SFT_548',SFT_RUN/'generation'),('DPO_Test1',TEST1/'generation')):
        metrics[name],predictions[name] = verify_generation(folder,adapters[name],cfg,val,score_predictions)
    for name in CANDIDATES:
        ev.ADAPTER,ev.OUTPUT,ev.BASE_CONFIG = adapters[name],run/name/'generation',CONFIG
        ev.generate(copy.deepcopy(cfg),val)
        metrics[name],predictions[name] = verify_generation(ev.OUTPUT,adapters[name],cfg,val,score_predictions)
    report = {'complete':True,'generation':metrics,'training':training,'shared_reference_seconds':ref_seconds,
              'adapters':{k:str(v) for k,v in adapters.items()},'validation_sha256':sha(t.VAL),
              'judge':'NOT RUN','rag':False,'test_used':False,'selected_model':None,
              'limitation':'Lexical metrics do not establish technical correctness; review before selection.'}
    write(run/'comparison.json',report)
    bundle=export_review(run,val,predictions,report)
    names=list(metrics)
    lines=['[DPO Test2 Comparison]',f'{"Metric":40}'+''.join(f'{n:>16}' for n in names)]
    for k in KEYS:
        lines.append(f'{k:40}'+''.join(f'{str(metrics[n][k]):>16}' for n in names))
    for name,data in training.items():
        lines.append(f'{name} Training: '+json.dumps(data,ensure_ascii=False))
    lines += ['Judge: NOT RUN | RAG: NONE | Test: NOT USED','No automatic model selection',
              f'Results: {run}',f'Upload only: {bundle}']
    (run/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
