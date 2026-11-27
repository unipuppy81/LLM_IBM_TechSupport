"""Recreate Test2 LoRA and compare generation prompts using the SAME new adapter.
Place in IBMTechSupport/scripts/finetuning/. No Judge, RAG or original-file changes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
SOURCE = ROOT / 'evaluate_results/106/finetuning/data_composition_test2/run_20261127_000005_052233'
EXPERIMENT = SOURCE / 'experiments/A_expanded'
RUNS = ROOT / 'evaluate_results/106/finetuning/generation_prompt_test1_retrained'
MODELS = ROOT / 'models/finetuning/generation_prompt_test1_retrained'
SYSTEM = '''You are an IBM technical support assistant. Answer in English using only your existing knowledge.
Answer the exact question directly. Do not repeat the question or merely restate the error. Do not announce a list or procedure without providing it.
For troubleshooting, give the specific cause or actionable resolution requested. For configuration or installation, provide the necessary settings or steps. For compatibility, state the supported versions or limits when known. For security bulletins, identify the relevant vulnerability details or a reliable source, not just the bulletin title.
Preserve applicable product names, versions, error codes, commands, paths and numerical limits. Do not invent technical details. If a condition matters, give a reliable conditional answer when possible.
If you cannot reliably answer the essential request, respond exactly INSUFFICIENT_INFORMATION. Otherwise output only the technical answer. Do not output analysis or introductory filler.'''
KEYS = ('token_f1_percent','exact_match_percent','abstention_rate_percent',
        'false_refusal_rate_percent','generation_seconds','questions_per_second','peak_gpu_memory_gb')
ADAPTER_FILES = ('adapter_config.json','adapter_model.safetensors')


def sha(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required original file is missing: {path}')
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Original SHA-256 mismatch: {path}')


def adapter_hash(path):
    return {n:sha(path/n) for n in ADAPTER_FILES}


def validate_sources():
    import yaml
    m = read(SOURCE/'manifest.json')
    old = read(EXPERIMENT/'lora/generation_manifest.json')
    train, val = EXPERIMENT/'train.json', ROOT/'data/processed/val_106.json'
    verify(train, m['training_sha256'])
    verify(val, old['dataset_sha256'])
    verify(EXPERIMENT/'prompt_config.yaml',old['base_config_sha256'])
    verify(ROOT/'scripts/train_lora.py',m['sources']['scripts/train_lora.py'])
    cfg = yaml.safe_load((EXPERIMENT/'prompt_config.yaml').read_text(encoding='utf-8-sig'))
    if cfg != m['config'] or cfg['prompt'] != old['prompt'] or cfg['inference'] != old['inference']:
        raise ValueError('Original configuration mismatch')
    if cfg['prompt']['user_template'] != '{question}':
        raise ValueError('Expected closed-book question-only input')
    if m['settings'] != dict(epochs=2,max_length=2048,grad_accum=8,learning_rate=2e-4,limit=0,seed=42):
        raise ValueError('Unexpected training settings')
    if (m['rank'],m['alpha'],m['dropout'],sorted(m['targets'])) != (8,16,.05,['q_proj','v_proj']):
        raise ValueError('Unexpected original adapter settings')
    if (m['spec']['question_tokens'],m['spec']['answer_tokens'],m['spec']['prompt']) != (192,160,cfg['prompt']['system']):
        raise ValueError('Unexpected training prompt or token limits')
    tr, va = read(train), read(val)
    if (len(tr),len(va),sum(r['is_impossible'] for r in tr),sum(r['is_impossible'] for r in va)) != (532,106,0,29):
        raise ValueError('Expected 532 positive training records and 77/29 validation records')
    if len({r['id'] for r in tr}) != 532 or len({r['id'] for r in va}) != 106 or {r['id'] for r in tr}&{r['id'] for r in va}:
        raise ValueError('Duplicate or overlapping IDs')
    base = ROOT/cfg['model']['path']
    for name,digest in m['model_files'].items():
        verify(base/name,digest)
    return m,old,cfg,train,val,va


def train_new(run, adapter, m, cfg, train, val):
    import torch
    import train_lora as trainer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if trainer.MODEL_DIR.resolve() != (ROOT/cfg['model']['path']).resolve():
        raise ValueError('Unexpected trainer base model path')
    if (trainer.LORA_R,trainer.LORA_ALPHA,trainer.LORA_DROPOUT) != (8,16,.05):
        raise ValueError('Original trainer hyperparameters changed')
    if adapter.exists():
        raise FileExistsError(f'Existing adapter protected: {adapter}')
    trainer.TRAIN_PATH, trainer.VAL_PATH = train,val
    trainer.RESULT_DIR, trainer.LORA_DIR = run/'training',adapter
    trainer.SYSTEM_PROMPT = m['spec']['prompt']
    trainer.MAX_QUESTION_TOKENS,trainer.MAX_ANSWER_TOKENS = 192,160
    print('[Re-training original Data Composition Test2 | 532 QA]',flush=True)
    trainer.train_lora(argparse.Namespace(**m['settings']))
    metrics=read(run/'training/lora_train_metrics.json')
    if (metrics['train_samples'],metrics['val_samples']) != (532,106):
        raise ValueError('Training count mismatch')
    if len(metrics['history'])!=2 or any(h['optimizer_steps']!=67 for h in metrics['history']):
        raise ValueError('Unexpected optimizer step count')
    ac=read(adapter/'adapter_config.json')
    if (ac['r'],ac['lora_alpha'],ac['lora_dropout'],sorted(ac['target_modules'])) != (8,16,.05,['q_proj','v_proj']):
        raise ValueError('Saved adapter configuration mismatch')
    digest=adapter_hash(adapter)
    gc.collect(); torch.cuda.empty_cache()
    write(run/'training_complete.json',{'adapter':str(adapter),'adapter_sha256':digest,'metrics':metrics})
    return digest


def flags(q,a):
    if a=='INSUFFICIENT_INFORMATION': return []
    if not a.strip(): return ['empty_answer']
    norm=lambda t:' '.join(re.findall(r'\w+|[^\w\s]',t.casefold()))
    qn,an=norm(q),norm(a)
    result=[]
    if an==qn or (len(an.split())>=8 and an in qn): result.append('verbatim_question_fragment')
    if '\n' not in a.strip() and re.fullmatch(r'(?:the following (?:steps|information|is a (?:summary|list)|are (?:the )?(?:steps|details))[^\n]*|here (?:are|is)[^\n]*)[:.]?',a.strip(),re.I):
        result.append('possible_intro_only')
    lines=[norm(re.sub(r'^\s*(?:\d+[.)]|[-*])\s*','',x)) for x in a.splitlines()]
    lines=[x for x in lines if len(x.split())>=4]
    if len(lines)!=len(set(lines)): result.append('repeated_line')
    return result


def compare(run, adapter, cfg, records, digest, old):
    from evaluate import evaluate_lora as ev
    from evaluate.common import score_predictions, load_jsonl, save_jsonl
    ev.ADAPTER=adapter
    ev.BASE_CONFIG=EXPERIMENT/'prompt_config.yaml'
    variants={'A_original_prompt':cfg['prompt']['system'],'B_direct_answer':SYSTEM}
    predictions,metrics={},{}
    for name,system in variants.items():
        ev.OUTPUT=run/name
        current=copy.deepcopy(cfg); current['prompt']['system']=system
        if adapter_hash(adapter)!=digest: raise ValueError('Adapter changed')
        print(f'[Generation] {name}',flush=True)
        ev.generate(current,records)
        gen=read(ev.OUTPUT/'generation_manifest.json')
        if gen['adapter_sha256']!=digest or gen['prompt']!=current['prompt'] or gen['inference']!=cfg['inference'] or gen['dataset_sha256']!=old['dataset_sha256']:
            raise ValueError('Generation provenance mismatch')
        predictions[name]=load_jsonl(ev.OUTPUT/'predictions.jsonl')
        metrics[name]=read(ev.OUTPUT/'metrics.json')
        recalculated=score_predictions(records,predictions[name],'val_106.json',1)
        for k in KEYS[:4]:
            if metrics[name][k]!=recalculated[k]: raise ValueError(f'Metric mismatch: {k}')
    a,b=variants
    pairs,blind,key=[],[],[]
    rng=random.Random(42)
    for r,x,y in zip(records,predictions[a],predictions[b]):
        row={'id':r['id'],'question':r['question'],'reference_answer':r['answer'],'is_impossible':r['is_impossible']}
        for name,pred in ((a,x),(b,y)):
            row[name]={'prediction':pred['prediction'],'token_f1':pred['token_f1'],'flags':flags(r['question'],pred['prediction'])}
        pairs.append(row)
        if not r['is_impossible']:
            swap=rng.choice([True,False]); left,right=(y,x) if swap else (x,y)
            blind.append({'id':r['id'],'question':r['question'],'reference_answer':r['answer'],
                          'reference_evidence':json.dumps(r['contexts'],ensure_ascii=False),
                          'answer_A':left['prediction'],'answer_B':right['prediction'],
                          'verdict_A':'','verdict_B':'','reason_A':'','reason_B':''})
            key.append({'id':r['id'],'A':b if swap else a,'B':a if swap else b})
    save_jsonl(run/'answer_comparison.jsonl',pairs)
    write(run/'review_key.json',key)
    with (run/'blind_review.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(blind[0]));w.writeheader();w.writerows(blind)
    diagnostics={name:dict(Counter(flag for row in pairs if not row['is_impossible'] for flag in row[name]['flags'])) for name in variants}
    changes=Counter('improved' if row[b]['token_f1']>row[a]['token_f1'] else 'declined' if row[b]['token_f1']<row[a]['token_f1'] else 'tied' for row in pairs if not row['is_impossible'])
    report={'metrics':metrics,'diagnostics':diagnostics,'f1_changes':dict(changes),'adapter_sha256':digest,
            'original_adapter_sha256':old['adapter_sha256'],'same_as_historical_adapter':digest==old['adapter_sha256'],
            'new_training':True,'judge':'NOT RUN','rag':False,'complete':True,
            'technical_accuracy':'Requires independent review; lexical metrics do not establish correctness.',
            'comparison':'Both prompts run on the same newly trained adapter; historical predictions not reused.'}
    write(run/'comparison.json',report)
    lines=['[Generation Prompt Test1 - Retrained]',f'{"Metric":40} {"A_original":>15} {"B_direct":>15}']
    for k in KEYS: lines.append(f'{k:40} {str(metrics[a].get(k)):>15} {str(metrics[b].get(k)):>15}')
    lines.extend([f'Diagnostics: {diagnostics}',f'Per-answer F1 changes: {dict(changes)}', 'Judge: NOT RUN | RAG: NONE',f'Adapter: {adapter}',f'Results: {run}'])
    (run/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-only',action='store_true',help='Recreate weights without inference')
    p.add_argument('--resume',type=Path,help='Continue generation from a completed training run')
    args=p.parse_args()
    m,old,cfg,train,val,records=validate_sources()
    if args.resume:
        if args.train_only: p.error('--resume and --train-only are mutually exclusive')
        previous=args.resume.resolve()
        if previous.parent!=RUNS.resolve(): raise ValueError('Invalid resume run')
        completed=read(previous/'training_complete.json')
        adapter=Path(completed['adapter']).resolve()
        if adapter!=(MODELS/previous.name/'lora').resolve(): raise ValueError('Unexpected resume adapter location')
        digest=adapter_hash(adapter)
        if digest!=completed['adapter_sha256']: raise ValueError('Resume adapter changed')
        saved=read(previous/'experiment.json')
        if saved['source_manifest_sha256']!=sha(SOURCE/'manifest.json') or saved['training_sha256']!=sha(train): raise ValueError('Resume source changed')
    else:
        previous=None
    run=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True,exist_ok=False)
    if not previous: adapter=MODELS/run.name/'lora'
    write(run/'experiment.json',{'source':str(SOURCE),'source_manifest_sha256':sha(SOURCE/'manifest.json'),
          'training_sha256':sha(train),'validation_sha256':sha(val),'settings':m['settings'],'spec':m['spec'],
          'baseline_prompt':cfg['prompt'],'candidate_prompt':{**cfg['prompt'],'system':SYSTEM},'adapter':str(adapter),
          'resumed_from':str(previous) if previous else None,'judge':False,'rag':False})
    if not previous: digest=train_new(run,adapter,m,cfg,train,val)
    print(f'Adapter: {adapter}\nTraining run: {previous or run}',flush=True)
    if args.train_only: return
    compare(run,adapter,cfg,records,digest,old)


if __name__=='__main__': main()
