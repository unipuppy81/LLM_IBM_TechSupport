"""Train-only LoRA DAPT -> SFT. Place in IBMTechSupport/scripts/finetuning/.
Reuse original SFT trainer and evaluation code; no test, RAG or Judge.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
TRAIN = ROOT / 'data/processed/train_clean_ver2.json'
TRAIN_HASH = '9372bbf386fa57b18ff67ac8b8e19be2cbad398f3fc33010485dcf953e86e5c7'
VAL = ROOT / 'data/processed/val_106.json'
SFT = ROOT / 'evaluate_results/106/finetuning/train_clean_ver2/run_20261127_134955_078761'
BASE = ROOT / 'evaluate_results/106/finetuning/stage3_test5/run_20261126_173316_296155/base'
SOURCE = ROOT / 'evaluate_results/106/finetuning/data_composition_test2/run_20261127_000005_052233'
CONFIG = SOURCE / 'experiments/A_expanded/prompt_config.yaml'
OUT = ROOT / 'evaluate_results/106/finetuning/dapt_sft'
MODELS = ROOT / 'models/finetuning/dapt_sft'
SEED, BLOCK, ACCUM, DAPT_LR, DAPT_EPOCHS = 42, 1024, 8, 1e-4, 1
SFT_SETTINGS = dict(epochs=2,max_length=2048,grad_accum=8,learning_rate=2e-4,limit=0,seed=42)
KEYS = ('token_f1_percent','exact_match_percent','abstention_rate_percent','false_refusal_rate_percent','generation_seconds','questions_per_second','peak_gpu_memory_gb')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)


def adapter_hash(path):
    return {n:sha(path/n) for n in ('adapter_config.json','adapter_model.safetensors')}


def documents(records):
    """Deduplicate exact source text, without consulting validation/test documents."""
    docs = {}
    names = {}
    for r in records:
        if r['is_impossible'] is not False:
            raise ValueError('Only answerable Train Ver2 records allowed')
        for c in r['contexts']:
            name,text = c['filename'],c['text']
            if not isinstance(text,str) or not text.strip() or not isinstance(name,str) or not name.strip():
                raise ValueError(f'Invalid source: {r["id"]}')
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if name in names and names[name] != digest:
                raise ValueError(f'Conflicting source text for {name}')
            names[name] = digest
            if digest not in docs:
                docs[digest] = {'sha256':digest,'text':text,'filenames':[], 'train_ids':[]}
            d = docs[digest]
            if name not in d['filenames']:
                d['filenames'].append(name)
            if r['id'] not in d['train_ids']:
                d['train_ids'].append(r['id'])
    if len(docs) != 310:
        raise ValueError(f'Expected 310 unique train documents, got {len(docs)}')
    return list(docs.values())


def token_blocks(docs,tokenizer):
    """One-token overlap preserves each within-document next-token target exactly once.
    No cross-document prediction, no padding, no synthetic EOS at chunk boundaries.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError('EOS token required')
    blocks, stats = [], []
    for doc in docs:
        ids = tokenizer.encode(doc['text'],add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(ids)<2:
            raise ValueError('Empty tokenized document')
        start_index = len(blocks)
        for start in range(0,len(ids)-1,BLOCK-1):
            block = ids[start:start+BLOCK]
            if len(block)>=2:
                blocks.append(block)
        targets=sum(len(b)-1 for b in blocks[start_index:])
        if targets != len(ids)-1:
            raise RuntimeError('Token coverage mismatch')
        stats.append({'sha256':doc['sha256'],'filenames':doc['filenames'],'train_ids':doc['train_ids'],
                      'tokens_including_eos':len(ids),'target_tokens':targets,'blocks':len(blocks)-start_index})
    return blocks,stats


def validate_training_sources():
    import yaml
    import train_lora as trainer
    if sha(TRAIN)!=TRAIN_HASH or sha(SFT/'train.json')!=TRAIN_HASH:
        raise ValueError('Train Ver2 dataset mismatch')
    records=read(TRAIN)
    if len(records)!=548 or len({r['id'] for r in records})!=548:
        raise ValueError('Expected 548 unique training records')
    source=read(SOURCE/'manifest.json')
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8-sig'))
    exp=read(SFT/'experiment.json')
    if cfg!=source['config'] or exp['training_sha256']!=TRAIN_HASH or exp['settings']!=SFT_SETTINGS:
        raise ValueError('Baseline training provenance mismatch')
    if exp['prompt']!=cfg['prompt'] or exp['inference']!=cfg['inference']:
        raise ValueError('Baseline prompt/inference mismatch')
    if (source['rank'],source['alpha'],source['dropout'],sorted(source['targets']))!=(8,16,.05,['q_proj','v_proj']):
        raise ValueError('Source LoRA configuration mismatch')
    if (source['spec']['question_tokens'],source['spec']['answer_tokens'])!=(192,160):
        raise ValueError('Source token limits mismatch')
    if cfg['prompt']['user_template']!='{question}' or (cfg['inference']['batch_size'],cfg['inference']['max_new_tokens'],cfg['inference']['do_sample'],cfg['inference']['num_beams'])!=(1,160,False,1):
        raise ValueError('Unexpected evaluation settings')
    if sha(ROOT/'scripts/train_lora.py')!=source['sources']['scripts/train_lora.py']:
        raise ValueError('Original SFT trainer modified')
    if (trainer.LORA_R,trainer.LORA_ALPHA,trainer.LORA_DROPOUT)!=(8,16,.05):
        raise ValueError('Unexpected SFT adapter configuration')
    base=ROOT/cfg['model']['path']
    if trainer.MODEL_DIR.resolve()!=base.resolve():
        raise ValueError('Base model path mismatch')
    for name,digest in source['model_files'].items():
        if sha(base/name)!=digest:
            raise ValueError(f'Base model hash mismatch: {name}')
    return records,cfg,exp


def check_trainable(model):
    ps=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in ps)!=1089536 or any('lora_' not in n for n,p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Unexpected trainable parameters')
    return ps


def dapt(docs,run,adapter):
    import torch
    import train_lora as trainer
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model,tokenizer=trainer.load_lora_model()
    ps=check_trainable(model)
    blocks,stats=token_blocks(docs,tokenizer)
    opt=torch.optim.AdamW(ps,lr=DAPT_LR)
    scaler=torch.amp.GradScaler('cuda')
    model.train()
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start=time.perf_counter()
    updates=skipped=0
    history=[]
    rng=random.Random(SEED)
    for epoch in range(1,DAPT_EPOCHS+1):
        order=list(range(len(blocks)))
        rng.shuffle(order)
        total_nll,total_tokens=0.0,0
        for offset in range(0,len(order),ACCUM):
            group=[blocks[i] for i in order[offset:offset+ACCUM]]
            window_tokens=sum(len(x)-1 for x in group)
            for block in group:
                ids=torch.tensor([block],dtype=torch.long,device='cuda')
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    loss=model(input_ids=ids,labels=ids,use_cache=False).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError('Non-finite DAPT loss')
                n=len(block)-1
                total_nll+=loss.item()*n
                total_tokens+=n
                scaler.scale(loss*(n/window_tokens)).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ps,1.0)
            old_scale=scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale()<old_scale:
                skipped+=1
            else:
                updates+=1
            opt.zero_grad(set_to_none=True)
            if (updates+skipped)%20==0 or offset+ACCUM>=len(order):
                print(f'DAPT | blocks {min(offset+ACCUM,len(order))}/{len(order)} | token loss {total_nll/total_tokens:.4f}',flush=True)
        history.append({'epoch':epoch,'mean_token_loss':total_nll/total_tokens,'target_tokens':total_tokens})
    torch.cuda.synchronize()
    if updates<1 or updates+skipped!=math.ceil(len(blocks)/ACCUM)*DAPT_EPOCHS:
        raise RuntimeError('Unexpected DAPT update count')
    metrics={'documents':len(docs),'blocks':len(blocks),'input_tokens_per_epoch':sum(map(len,blocks)),
             'target_tokens_per_epoch':sum(len(b)-1 for b in blocks),'epochs':DAPT_EPOCHS,'block_length':BLOCK,
             'lr':DAPT_LR,'grad_accum':ACCUM,'optimizer_updates':updates,'skipped_updates':skipped,
             'training_seconds':round(time.perf_counter()-start,3),'peak_gpu_memory_gb':round(torch.cuda.max_memory_allocated()/1024**3,3),'history':history}
    model.save_pretrained(adapter,safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    write(run/'dapt_document_manifest.json',stats)
    write(run/'dapt_metrics.json',metrics)
    write(run/'dapt_complete.json',{'adapter_sha256':adapter_hash(adapter),'metrics':metrics})
    del model,tokenizer,opt,scaler,ps,loss,ids
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def sft(run,initial,adapter,cfg):
    import torch
    import train_lora as trainer
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    def load_dapt_adapter():
        tokenizer=AutoTokenizer.from_pretrained(trainer.MODEL_DIR,local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained(trainer.MODEL_DIR,dtype=torch.float16,device_map={'':0},local_files_only=True)
        base.config.use_cache=False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        base.enable_input_require_grads()
        model=PeftModel.from_pretrained(base,str(initial),is_trainable=True,local_files_only=True)
        check_trainable(model)
        return model,tokenizer
    old=trainer.load_lora_model
    trainer.TRAIN_PATH,trainer.VAL_PATH=run/'train.json',VAL
    trainer.RESULT_DIR,trainer.LORA_DIR=run/'sft_training',adapter
    trainer.SYSTEM_PROMPT=cfg['prompt']['system']
    trainer.MAX_QUESTION_TOKENS,trainer.MAX_ANSWER_TOKENS=192,160
    # Only change initialization; identical trainer, tokenization, optimizer, schedule and checkpoint rule.
    trainer.load_lora_model=load_dapt_adapter
    try:
        trainer.train_lora(argparse.Namespace(**SFT_SETTINGS))
    finally:
        trainer.load_lora_model=old
    metrics=read(run/'sft_training/lora_train_metrics.json')
    if (metrics['train_samples'],metrics['val_samples'])!=(548,106) or len(metrics['history'])!=2 or any(h['optimizer_steps']!=69 for h in metrics['history']):
        raise ValueError('SFT training configuration mismatch')
    write(run/'sft_complete.json',{'adapter_sha256':adapter_hash(adapter),'initial_adapter_sha256':adapter_hash(initial),'metrics':metrics})
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def verified_generation(folder,cfg,records,expected_adapter=None):
    from evaluate.common import load_jsonl,score_predictions
    manifest=read(folder/'generation_manifest.json')
    for k,v in {'dataset_sha256':sha(VAL),'expected_count':106,'predictions_sha256':sha(folder/'predictions.jsonl'),'prompt':cfg['prompt'],'inference':cfg['inference']}.items():
        if manifest.get(k)!=v:
            raise ValueError(f'Generation provenance mismatch: {folder} {k}')
    if expected_adapter is not None and manifest.get('adapter_sha256')!=expected_adapter:
        raise ValueError('Adapter provenance mismatch')
    rows=load_jsonl(folder/'predictions.jsonl')
    metrics=read(folder/'metrics.json')
    actual=score_predictions(records,rows,'val_106.json',1)
    if any(actual[k]!=metrics[k] for k in KEYS[:4]):
        raise ValueError('Saved metric mismatch')
    return metrics,rows


def base_generation(run,cfg,records):
    """Reuse a compatible Base result, otherwise evaluate the frozen Base once."""
    import torch
    from load_base_model import load_model
    from evaluate.evaluate_base import make_prediction_rows
    from evaluate.common import score_predictions,save_jsonl
    try:
        metrics,rows=verified_generation(BASE,cfg,records)
        return metrics,rows,'reused: '+str(BASE)
    except (FileNotFoundError,ValueError,KeyError) as exc:
        print(f'Compatible saved Base unavailable; running Base evaluation once: {exc}',flush=True)
    out=run/'base'
    out.mkdir(exist_ok=False)
    model,tokenizer,gen=load_model(copy.deepcopy(cfg))
    try:
        rows,seconds,memory=make_prediction_rows(model,tokenizer,gen,records,1,cfg['prompt'])
        metrics=score_predictions(records,rows,'val_106.json',1)
        metrics.update(method='Base',generation_seconds=round(seconds,3),
                       mean_generation_seconds_per_question=round(seconds/len(records),4),
                       questions_per_second=round(len(records)/seconds,3),peak_gpu_memory_gb=round(memory,3),
                       retrieval=None,faithfulness_percent=None)
        save_jsonl(out/'predictions.jsonl',rows)
        write(out/'metrics.json',metrics)
        write(out/'generation_manifest.json',{'dataset_sha256':sha(VAL),'expected_count':106,
            'predictions_sha256':sha(out/'predictions.jsonl'),'prompt':cfg['prompt'],'inference':cfg['inference'],
            'base_config_sha256':sha(CONFIG)})
    finally:
        del model,tokenizer,gen
        gc.collect()
        torch.cuda.empty_cache()
    metrics,rows=verified_generation(out,cfg,records)
    return metrics,rows,'new: '+str(out)


def export_review(run,records,predictions,report):
    rng=random.Random(SEED)
    by_model={n:{r['id']:r for r in rows} for n,rows in predictions.items()}
    cases,key=[],{}
    for r in records:
        if r['is_impossible']:
            continue
        names=list(by_model)
        rng.shuffle(names)
        labels=dict(zip('ABC',names))
        key[r['id']]=labels
        cases.append({'id':r['id'],'question':r['question'],'reference_answer':r['answer'], 'contexts':r['contexts'],
                      'answers':{label:by_model[name][r['id']]['prediction'] for label,name in labels.items()}})
    if len(cases)!=77:
        raise ValueError('Expected 77 answerable review cases')
    path=run/'dapt_sft_review_bundle.json'
    write(path,{'purpose':'Evaluation only; never use validation questions, answers or review results as training examples.',
                'review_instructions':'Review anonymous answers against reference and evidence; CORRECT/PARTIAL/INCORRECT/UNCERTAIN; then reveal review_key.',
                'cases':cases,'comparison':report,'review_key':key})
    return path


def main():
    import torch
    from evaluate import evaluate_lora as ev
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    records,cfg,exp=validate_training_sources()
    docs=documents(records)
    # Check prerequisites without using validation content in DAPT.
    for p in (VAL, SFT/'generation/predictions.jsonl',SFT/'generation/generation_manifest.json',SFT/'generation/metrics.json',SFT/'training_complete.json',SFT/'training/lora_train_metrics.json'):
        if not p.is_file():
            raise FileNotFoundError(f'Required existing baseline file: {p}')
    run=OUT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    model_root=MODELS/run.name
    initial,adapter=model_root/'dapt_lora',model_root/'lora'
    run.mkdir(parents=True,exist_ok=False)
    if model_root.exists():
        raise FileExistsError(model_root)
    shutil.copyfile(TRAIN,run/'train.json')
    write(run/'experiment.json',{'train_sha256':sha(TRAIN),'documents':len(docs),'dapt':{'epochs':DAPT_EPOCHS,'lr':DAPT_LR,'block_length':BLOCK,'grad_accum':ACCUM,'objective':'causal next-token cross entropy, token-weighted accumulation','seed':SEED,'rank':8,'alpha':16,'targets':['q_proj','v_proj']},'sft':SFT_SETTINGS,'prompt':cfg['prompt'],'inference':cfg['inference'],'dapt_validation_used':False,'sft_validation_use':'loss and checkpoint selection only, then evaluation','test_used':False,'rag_used':False,'judge_used':False,'base_model_files':read(SOURCE/'manifest.json')['model_files']})
    print(f'[DAPT] {len(docs)} unique Train documents',flush=True)
    dapt_metrics=dapt(docs,run,initial)
    # Validation is first read here, after all DAPT updates are complete.
    if sha(VAL)!=exp['validation_sha256']:
        raise ValueError('Validation dataset hash mismatch')
    val=read(VAL)
    if len(val)!=106 or sum(r['is_impossible'] for r in val)!=29 or len({r['id'] for r in val})!=106 or {r['id'] for r in val}&{r['id'] for r in records}:
        raise ValueError('Validation count or split mismatch')
    old_metrics,old_rows=verified_generation(SFT/'generation',cfg,val,read(SFT/'training_complete.json')['adapter_sha256'])
    print('[SFT] 548 QA | initialized from DAPT adapter',flush=True)
    sft_metrics=sft(run,initial,adapter,cfg)
    ev.ADAPTER,ev.OUTPUT,ev.BASE_CONFIG=adapter,run/'generation',CONFIG
    ev.generate(copy.deepcopy(cfg),val)
    new_metrics,new_rows=verified_generation(run/'generation',cfg,val,adapter_hash(adapter))
    base_metrics,base_rows,base_origin=base_generation(run,cfg,val)
    if sha(TRAIN)!=TRAIN_HASH or sha(run/'train.json')!=TRAIN_HASH or adapter_hash(initial)!=read(run/'dapt_complete.json')['adapter_sha256']:
        raise RuntimeError('Training source or DAPT checkpoint changed')
    metrics={'Base':base_metrics,'SFT_548':old_metrics,'DAPT_SFT_548':new_metrics}
    report={'complete':True,'generation':metrics,'base_evaluation_source':base_origin,'dapt_training':dapt_metrics,'sft_training':sft_metrics,'baseline_sft_training':read(SFT/'training/lora_train_metrics.json'),'adapter':str(adapter),'dapt_adapter':str(initial),'judge':'NOT RUN','rag':False,'test_used':False,'selection':'NONE; technical review required','limitation':'Small train-only corpus; LoRA adaptation, not full-parameter pretraining. Validation has been used for model development.'}
    write(run/'comparison.json',report)
    bundle=export_review(run,val,{'Base':base_rows,'SFT_548':old_rows,'DAPT_SFT_548':new_rows},report)
    lines=['[DAPT + SFT Test1 Comparison]',f'{"Metric":40} {"Base":>14} {"SFT 548":>14} {"DAPT+SFT":>14}']
    for k in KEYS:
        lines.append(f'{k:40} '+ ' '.join(f'{str(m.get(k)):>14}' for m in metrics.values()))
    lines += ['DAPT: '+json.dumps(dapt_metrics),'SFT: '+json.dumps(sft_metrics),'Judge: NOT RUN | RAG: NONE | Test: NOT USED',f'Adapter: {adapter}',f'Results: {run}',f'Upload only: {bundle}']
    (run/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
