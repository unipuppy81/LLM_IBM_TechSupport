"""Matched QLoRA training and validation; no Test access. Install in scripts/finetuning.
Uses original training loop and context input/weighting code, replacing only model loader.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'evaluate_results/final_lora_qlora/matched_v1'
MODEL_OUT = ROOT / 'models/finetuning/final_lora_qlora/matched_v1'
OLD = ROOT / 'evaluate_results/106/finetuning'
CLOSED = OLD / 'train_clean_ver2/run_20261127_134955_078761'
CONTEXT = OLD / 'data_composition_test3/run_20261127_003107_337442'
ADAPTERS = {
    'closed': ROOT / 'models/finetuning/finetuning/train_clean_ver2/run_20261127_134955_078761/lora',
    'context': ROOT / 'models/finetuning/data_composition_test3/run_20261127_003107_337442/lora',
}
BASE = ROOT / 'models/Qwen2.5-1.5B-Instruct'
VAL = ROOT / 'data/processed/val_106.json'
SETTINGS = dict(epochs=2, max_length=2048, grad_accum=8, learning_rate=2e-4, limit=0, seed=42)
EXPECTED_CLOSED = '9372bbf386fa57b18ff67ac8b8e19be2cbad398f3fc33010485dcf953e86e5c7'
QUANT = dict(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, compute_dtype='float16')

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def verify(p, digest):
    raw = p.read_bytes()

    candidates = (
        raw,
        raw.replace(b'\r\n', b'\n'),
        raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'),
    )

    require(
        any(hashlib.sha256(b).hexdigest() == digest for b in candidates),
        f'SHA256 mismatch: {p}'
    )

def adapter_hash(p):
    return {n: sha(p/n) for n in ('adapter_config.json', 'adapter_model.safetensors')}

def adapter_check(p):
    a = read(p/'adapter_config.json')
    require((a['r'], a['lora_alpha'], a['lora_dropout'], set(a['target_modules']), a['bias'], a['task_type']) == (8,16,.05,{'q_proj','v_proj'},'none','CAUSAL_LM'), f'Adapter settings differ: {p}')
    return adapter_hash(p)

def locate_recorded(name):
    # Historical manifests may have been saved on a different Windows PC.
    parts = PureWindowsPath(str(name)).parts if '\\' in str(name) else Path(name).parts
    ix = [i for i, p in enumerate(parts) if p.casefold() == 'ibmtechsupport']
    return ROOT.joinpath(*parts[ix[-1]+1:]) if ix else ROOT.joinpath(*parts)

def verify_marker(run, phase, names):
    entries = read(run/'completed'/f'{phase}.json')['files']
    for name in names:
        matches = [v for k,v in entries.items() if str(k).replace('\\','/').endswith('/'+name)]
        require(len(matches)==1, f'No unique original completion hash for {name}')
        verify(run/name, matches[0])

def validate_all():
    import yaml
    import run_stage0 as s0
    import run_data_composition_test3 as ctx
    e = read(CLOSED/'experiment.json')
    cm = read(CONTEXT/'manifest.json')
    cfg = read(CONTEXT/'config.json')
    require(e['settings'] == cm['settings'] == SETTINGS, 'Historical training settings differ')
    require(cfg == cm['config'] and cfg['prompt']['system'] == ctx.SYSTEM and cm['negative_weight'] == .175, 'Context prompt/weight mismatch')
    require(e['prompt']['user_template'] == '{question}', 'Closed-book prompt mismatch')
    verify(CLOSED/'train.json', e['training_sha256'])
    verify(CLOSED/'train.json', EXPECTED_CLOSED)
    verify(ROOT/'data/processed/train_clean_ver2.json', EXPECTED_CLOSED)
    verify(CONTEXT/'train_run/train.json', cm['training_sha256'])
    verify(VAL, e['validation_sha256']); verify(VAL, cm['val_sha256'])
    verify_marker(CONTEXT, 'prepare', ['inputs/train.json','inputs/oracle.json','inputs/audit.json'])
    verify_marker(CONTEXT, 'retrieve', ['inputs/retrieved.json','inputs/rag.json','inputs/retrieval_audit.json'])
    # Verify source files that control the historic training computation.
    historic = read(OLD/'data_composition_test2/run_20261127_000005_052233/manifest.json')
    verify(ROOT/'scripts/train_lora.py', historic['sources']['scripts/train_lora.py'])
    for name, digest in cm['active_sources'].items():
        p = locate_recorded(name)
        if p.name in {'train_lora.py','run_data_composition_test3.py','run_stage2_test4.py','run_stage1_quality.py','run_stage0.py','run_stage4_test1.py','load_base_model.py'}:
            verify(p, digest)
    for name, digest in cm['model_files'].items():
        verify(BASE/name, digest)
    if 'model_files' in historic:
        require(historic['model_files'] == cm['model_files'], 'Historic Base weights differ')
    va = s0.validate(VAL)
    require((len(va),sum(r['is_impossible'] for r in va))==(106,29), 'Validation counts differ')
    closed = s0.validate(CLOSED/'train.json')
    context = s0.validate(CONTEXT/'train_run/train.json')
    for rows, n, neg in [(closed,548,0),(context,539,172)]:
        require((len(rows),sum(r['is_impossible'] for r in rows))==(n,neg), 'Training counts differ')
        require(not ({r['id'] for r in rows}&{r['id'] for r in va}), 'Train/val ID overlap')
        require(not ({s0.norm(r['question']) for r in rows}&{s0.norm(r['question']) for r in va}), 'Train/val normalized question overlap')
    for name, rows in [('train',context),('oracle',va),('rag',va)]:
        items=read(CONTEXT/'inputs'/f'{name}.json')
        require(set(items)=={r['id'] for r in rows}, f'Context {name} input IDs differ')
        require(all(0<len(v['input_ids'])<=1887 for v in items.values()), f'Context {name} token budget mismatch')
    ch = adapter_check(ADAPTERS['closed'])
    th = adapter_check(ADAPTERS['context'])
    require(ch == read(CLOSED/'training_complete.json')['adapter_sha256'], 'Closed adapter hash mismatch')
    require(th == read(CONTEXT/'rag/lora/generation_manifest.json')['adapter_sha256'], 'Context adapter hash mismatch')
    require(read(CLOSED/'comparison.json')['complete'] and read(CONTEXT/'comparison.json')['complete'], 'LoRA comparison incomplete')
    closed_cfg = copy.deepcopy(cfg)
    closed_cfg['prompt'], closed_cfg['inference'] = e['prompt'], e['inference']
    for config in (closed_cfg,cfg):
        require(config['model']['dtype']=='float16' and (ROOT/config['model']['path']).resolve()==BASE.resolve(), 'Base model config differs')
        i=config['inference']; require((i['batch_size'],i['max_new_tokens'],i['do_sample'],i['num_beams'])==(1,160,False,1),'Generation settings differ')
    return {'closed':closed_cfg,'context':cfg}, va, {'closed':ch,'context':th}, cm

def train(kind, cfg, cm):
    import torch
    import train_lora as trainer
    import train_qlora as qtrainer
    import run_data_composition_test3 as ctx
    require(torch.cuda.is_available(), 'CUDA GPU required')
    target=MODEL_OUT/kind/'qlora'
    run=OUT/kind
    marker=run/'training_complete.json'
    source=CLOSED/'train.json' if kind=='closed' else CONTEXT/'train_run/train.json'
    spec={'kind':kind,'settings':SETTINGS,'prompt':cfg['prompt'],'quantization':QUANT,'train_sha256':sha(source),'validation_sha256':sha(VAL),
          'original_adapter':adapter_hash(ADAPTERS[kind]),'base_model_files':cm['model_files'],'runner_sha256':sha(Path(__file__))}
    if kind=='context':
        spec['packed_inputs']={n:sha(CONTEXT/'inputs'/n) for n in ('train.json','oracle.json')}
    if marker.is_file():
        done=read(marker)
        require(done['spec']==spec and done['adapter_sha256']==adapter_check(target), 'Existing QLoRA provenance differs')
        print(f'Already completed; reusing {target}'); return
    require(not run.exists() and not target.exists(), f'Partial/existing run protected: {run}; {target}')
    run.mkdir(parents=True, exist_ok=False)
    write(run/'experiment.json',spec)
    require((trainer.LORA_R,trainer.LORA_ALPHA,trainer.LORA_DROPOUT)==(8,16,.05),'Original trainer parameters differ')
    require(qtrainer.QUANT_TYPE=='nf4' and qtrainer.COMPUTE_DTYPE==torch.float16,'Original QLoRA precision differs')
    # Same optimizer, scaler, accumulation, checkpoint selection and dataset as the LoRA experiment.
    trainer.load_lora_model=qtrainer.load_qlora_model
    if kind=='closed':
        trainer.TRAIN_PATH,trainer.VAL_PATH=source,VAL
        trainer.RESULT_DIR,trainer.LORA_DIR=run/'training',target
        trainer.SYSTEM_PROMPT=cfg['prompt']['system']
        trainer.MAX_QUESTION_TOKENS,trainer.MAX_ANSWER_TOKENS=192,160
        trainer.train_lora(argparse.Namespace(**SETTINGS))
        metrics_path=run/'training/lora_train_metrics.json'
    else:
        # Reuse exact original packed inputs, ContextDataset and weighted training wrapper.
        (run/'train_run').mkdir()
        shutil.copyfile(source,run/'train_run/train.json')
        (run/'inputs').mkdir()
        for n in ('train.json','oracle.json'):
            shutil.copyfile(CONTEXT/'inputs'/n,run/'inputs'/n)
        ctx.adapter=lambda unused: target
        ctx.train(run, cm)
        metrics_path=run/'train_run/training/lora_train_metrics.json'
    metrics=read(metrics_path)
    n=548 if kind=='closed' else 539
    require((metrics['train_samples'],metrics['val_samples'])==(n,106),'Training counts mismatch')
    require(len(metrics['history'])==2 and all(h['optimizer_steps']==math.ceil(n/8) for h in metrics['history']),'Update count mismatch')
    write(marker,{'spec':spec,'adapter_sha256':adapter_check(target),'adapter':str(target),'metrics':metrics,'metrics_source':str(metrics_path),
                  'note':'Original LoRA loop names the raw metrics lora; this run used the original 4-bit QLoRA loader.',
                  'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__})
    print(f'Completed: {marker}')

def load_inference(cfg, adapter, quantized):
    import torch
    from peft import PeftModel
    from load_base_model import load_model
    if not quantized:
        model,tok,gen=load_model(cfg)
    else:
        import bitsandbytes as bnb
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
        tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True,padding_side='left')
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16,quantization_config=quant,device_map={'':0},local_files_only=True)
        require(any(isinstance(m,bnb.nn.Linear4bit) for m in model.modules()),'4-bit layers missing')
        gen=GenerationConfig(max_new_tokens=160,do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True)
    model.eval()
    return model,tok,gen

def judge(records, rows, directory):
    from evaluate import evaluate_stage2_quality as q
    from evaluate.common import save_jsonl
    evaluator=q.Evaluator(BASE)
    cases=[]
    for i,(r,p) in enumerate(zip(records,rows),1):
        answer=p['prediction']
        if r['is_impossible']: verdict=q.fixed('NOT_APPLICABLE','No reference answer')
        elif answer==q.ABSTAIN: verdict=q.fixed('REFUSED','No answer on answerable question')
        elif not answer.strip(): verdict=q.fixed('INCORRECT','Empty answer')
        else: verdict=evaluator.evaluate(q.CORRECT_SYSTEM,{'question':r['question'],'candidate_answer':answer,'reference_answer':r['answer'],'reference_documents':q.gold_excerpts(r)},q.CORRECT_LABELS)
        cases.append({'id':r['id'],'is_impossible':r['is_impossible'],'correctness':verdict})
        if i%10==0 or i==len(records): print(f'Judge {i}/{len(records)}',flush=True)
    counts=Counter(c['correctness']['verdict'] for c in cases if not c['is_impossible'])
    n=sum(counts.values())
    save_jsonl(directory/'judge_cases.jsonl',cases)
    return {'correct':counts['CORRECT'],'denominator':n,'percent':round(100*counts['CORRECT']/n,2),'counts':dict(counts)}

def validate_pair(kind, cfg, records, skip_judge):
    import torch
    import time
    from evaluate.evaluate_base import make_prediction_rows
    from evaluate.common import score_predictions,save_jsonl,load_jsonl
    marker=OUT/kind/'training_complete.json'
    qinfo=read(marker)
    qa=MODEL_OUT/kind/'qlora'
    require(qinfo['adapter_sha256']==adapter_check(qa),'QLoRA adapter changed')
    inputs=read(CONTEXT/'inputs/rag.json') if kind=='context' else None
    report={}
    for method,ad in [('lora',ADAPTERS[kind]),('qlora',qa)]:
        dest=OUT/kind/'validation'/method
        spec={'adapter_sha256':adapter_hash(ad),'validation_sha256':sha(VAL),'prompt':cfg['prompt'],'inference':cfg['inference'], 'context_input_sha256':sha(CONTEXT/'inputs/rag.json') if inputs else None,'quantized':method=='qlora'}
        if (dest/'metrics.json').is_file():
            previous=read(dest/'metrics.json'); require(previous['spec']==spec and previous['predictions_sha256']==sha(dest/'predictions.jsonl'),'Saved validation mismatch')
            rows=load_jsonl(dest/'predictions.jsonl')
        else:
            require(not dest.exists(),f'Partial evaluation protected: {dest}')
            dest.mkdir(parents=True,exist_ok=False)
            model,tok,gen=load_inference(cfg,ad,method=='qlora')
            if inputs is None:
                rows,sec,mem=make_prediction_rows(model,tok,gen,records,1,cfg['prompt'])
            else:
                rows=[]; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
                for i,r in enumerate(records,1):
                    ids=torch.tensor([inputs[r['id']]['input_ids']],dtype=torch.long,device=model.device)
                    with torch.inference_mode(): out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=gen)
                    answer=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
                    rows.append({'id':r['id'],'question':r['question'],'reference_answer':r['answer'],'is_impossible':r['is_impossible'],'prediction':answer})
                    if i%10==0 or i==len(records): print(f'Generate {i}/{len(records)}',flush=True)
                torch.cuda.synchronize(); sec=time.perf_counter()-start; mem=torch.cuda.max_memory_allocated()/1024**3
            previous=score_predictions(records,rows,'val_106.json',1)
            save_jsonl(dest/'predictions.jsonl',rows)
            previous.update(spec=spec,predictions_sha256=sha(dest/'predictions.jsonl'),generation_seconds=round(sec,3),peak_gpu_memory_gib=round(mem,3),gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
            write(dest/'metrics.json',previous)
            del model,tok,gen; gc.collect(); torch.cuda.empty_cache()
        jpath=dest/'judge_metrics.json'
        j=None
        if jpath.is_file():
            j=read(jpath); require(j['predictions_sha256']==sha(dest/'predictions.jsonl'),'Judge input changed')
        elif not skip_judge:
            j=judge(records,rows,dest); j['predictions_sha256']=sha(dest/'predictions.jsonl'); write(jpath,j)
            gc.collect(); torch.cuda.empty_cache()
        report[method]={'metrics':previous,'judge':j,'false_refusals':sum(not r['is_impossible'] and p['prediction']=='INSUFFICIENT_INFORMATION' for r,p in zip(records,rows)),
                        'impossible_responses':sum(r['is_impossible'] and p['prediction']!='INSUFFICIENT_INFORMATION' for r,p in zip(records,rows))}
    # Comparison filename differs for judge/no-judge runs; never overwrite.
    report['note']='Context validation is generator-only on exact saved retrieval prompt tokens; no external guardrails. Training time is historical LoRA versus current QLoRA.'
    report['training']={'lora':read((CLOSED/'training/lora_train_metrics.json') if kind=='closed' else (CONTEXT/'train_run/training/lora_train_metrics.json')),'qlora':qinfo['metrics']}
    dst=OUT/kind/'validation'/('comparison_no_judge.json' if skip_judge else 'comparison.json')
    if not dst.exists(): write(dst,report)
    print(f'Validation comparison: {dst}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','train','validate'])
    p.add_argument('--kind',choices=['closed','context'])
    p.add_argument('--skip-judge',action='store_true')
    a=p.parse_args()
    if a.action!='check' and not a.kind: p.error('--kind required')
    configs,records,hashes,cm=validate_all()
    if a.action=='check':
        print('Original LoRA artifacts, input hashes, validation and training settings: OK')
        for k,ad in ADAPTERS.items(): print(f'{k}: {ad}')
    elif a.action=='train': train(a.kind,configs[a.kind],cm)
    else: validate_pair(a.kind,configs[a.kind],records,a.skip_judge)

if __name__=='__main__': main()
