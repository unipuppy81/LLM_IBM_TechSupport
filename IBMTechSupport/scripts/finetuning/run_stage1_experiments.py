"""Stage 1 controlled data experiments. Reuses Stage 0 and Stage 1 audit.
Place alongside run_stage0.py and run_stage1.py; no existing files are modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import run_stage0 as s0
import run_stage1 as s1

ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage1_experiments'
NAMES = ('exact_dedup','conflict_quarantine','combined','answer_alignment','answer_quality','answer_format')
REVIEWED_DATA_SHA = '388359a011a3ce7384d359ee8b9e0eeed4e6afca00bcc8d651488a1a02396c6c'


def canonical_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def format_answer(text):
    # Only reviewed prose labels and duplicated hyperlink displays, never code or commands.
    text = re.sub(r'(https?://[^\s\[\]]+) \[\1\]', r'\1', text)
    text = re.sub(r'(?m)^(CVEID|DESCRIPTION):(?=\S)', r'\1: ', text)
    text = text.replace('Seehttps://','See https://').replace(']for the current score','] for the current score')
    return text


def prepare_candidates(rows):
    if canonical_hash(rows) != REVIEWED_DATA_SHA:
        raise ValueError('Training content differs from the reviewed 551 records. Do not apply reviewed edits blindly.')
    dup = s1.duplicate_audit(rows)
    a, b = set(dup['exact_sft_duplicate_ids']), set(dup['conflict_ids'])
    result = {n: [r for r in rows if r['id'] not in ids] for n,ids in zip(NAMES[:3], (a,b,a|b))}
    byid = {r['id']:r for r in rows}
    # The requested operation is a diagnosis, not only a remedy. Add the exact CAUSE from the same document.
    source, target = byid['DEV_Q225'], byid['TRAIN_Q042']
    assert not source['is_impossible'] and not target['is_impossible']
    assert s0.norm(source['question']) == s0.norm(target['question'])
    assert source['contexts'] == target['contexts']
    assert source['answer'] in target['contexts'][0]['text']
    alignment = source['answer'] + '\n\n' + target['answer']
    # The historical document lists Java 8 for BPM 8.6 and a BPEL restriction.
    target = byid['DEV_Q291']
    evidence = target['contexts'][0]['text']
    assert 'IBM BPM V8.6 Java 8*' in evidence
    assert 'Java snippets, Java conditions, and Java expressions in BPEL processes must use Java 6 constructs instead of Java 7 or Java 8 constructs.' in evidence
    java = target['answer'] + '\n\nIBM BPM V8.6 supports Java 8. Java snippets, Java conditions, and Java expressions in BPEL processes must use Java 6 constructs instead of Java 7 or Java 8 constructs.'
    changes = {'answer_alignment':{'TRAIN_Q042':alignment}, 'answer_quality':{'DEV_Q291':java},
               'answer_format':{rid:format_answer(byid[rid]['answer']) for rid in ('TRAIN_Q047','TRAIN_Q206')}}
    for n, edits in changes.items():
        result[n] = [dict(r,answer=edits[r['id']]) if r['id'] in edits else r for r in rows]
    reviews = {'different_answer_groups': [
        {'ids':['TRAIN_Q489','DEV_Q029'],'decision':'retain','reason':'Same two connection pool settings; link and bullet layout differ.'},
        {'ids':['TRAIN_Q047','DEV_Q156'],'decision':'retain technical content','reason':'Same CVE, description, score and vector; link and label formatting differ.'},
        {'ids':['TRAIN_Q206','DEV_Q129'],'decision':'retain technical content','reason':'Same two CVEs, scores and vectors; hyperlink display differs.'},
        {'ids':['DEV_Q290','TRAIN_Q189'],'decision':'retain','reason':'Same unsupported Windows service conclusion; longer answer explains why.'},
        {'ids':['TRAIN_Q042','DEV_Q225'],'decision':'candidate answer_alignment','reason':'Question requests cause; add documented CAUSE before existing remedy. swg21664629.txt.'},
        {'ids':['DEV_Q155','TRAIN_Q056'],'decision':'retain','reason':'Same wpcollector definition; longer answer adds support workflow detail.'}],
        'nonverbatim_review':[
        {'ids':['DEV_Q156','DEV_Q129','DEV_Q029'],'decision':'retain','reason':'Formatting and link differences, not established technical errors.'},
        {'ids':['DEV_Q291'],'decision':'candidate answer_quality','reason':'Historical source swg21639640.txt lists BPM V8.6 Java 8 and BPEL Java 6 construct restriction. Existing version-specific answer retained and completed.'}],
        'format_review':{'ids':['TRAIN_Q047','TRAIN_Q206'],'decision':'candidate answer_format','reason':'Normalize prose label spacing and redundant URL display while preserving URLs and technical values.'},
        'scope':'Only reviewed training evidence; not a full semantic audit of all 551 rows. No external answers or validation content added.'}
    return result, reviews


def adapter(run, method):
    return ROOT/'models/finetuning/stage1'/run.parent.parent.name/run.name/method


def worker(run, phase, method):
    # Each experiment has its own directory, and the original trainer remains unchanged.
    s0.adapter = adapter
    if phase == 'train':
        m = importlib.import_module('train_'+method)
        if (m.LORA_R,m.LORA_ALPHA,m.LORA_DROPOUT)!=(8,16,0.05):
            raise RuntimeError('Adapter configuration differs from Stage 0')
        m.TRAIN_PATH = run/'train.json'
        m.VAL_PATH = s0.VAL
        m.RESULT_DIR = run/'training'
        setattr(m,method.upper()+'_DIR',adapter(run,method))
        # Emit per-epoch samples from the actual training DataLoader, not simulated progress.
        old_loader = m.DataLoader
        class ProgressLoader:
            def __init__(self, loader):
                self.loader, self.epoch = loader, 0
            def __len__(self): return len(self.loader)
            def __iter__(self):
                self.epoch += 1
                for i, batch in enumerate(self.loader,1):
                    yield batch
                    if i==1 or i%50==0 or i==len(self):
                        print(f'[Training] Epoch {self.epoch}/{s0.SETTINGS["epochs"]} | Sample {i}/{len(self)}',flush=True)
        def loader(*args,**kwargs):
            obj = old_loader(*args,**kwargs)
            return ProgressLoader(obj) if kwargs.get('shuffle') else obj
        m.DataLoader = loader
        getattr(m,'train_'+method)(argparse.Namespace(**s0.SETTINGS))
        t=s0.read(run/'training'/f'{method}_train_metrics.json')
        if t['train_samples']!=len(s0.validate(run/'train.json')) or t['val_samples']!=106:
            raise RuntimeError('Training dataset count mismatch')
        if not (adapter(run,method)/'adapter_model.safetensors').is_file():
            raise RuntimeError('Adapter missing')
    elif phase=='generate':
        s0.generate_worker(run,method)
    elif phase=='quality':
        # Reuse only exact evaluator input matches from the same frozen Stage 0 judge.
        from evaluate import evaluate_stage2_quality as q
        val={r['id']:r for r in s0.validate(s0.VAL)}
        cache={}
        def key(system,payload):
            return json.dumps([system,payload],ensure_ascii=False,sort_keys=True)
        manifest=s0.read(run.parent.parent/'manifest.json')
        baseline=Path(manifest['baseline'])
        dirs=[baseline/m for m in s0.METHODS]
        dirs.extend(p/m for p in run.parent.iterdir() if p.is_dir() for m in ('lora','qlora') if p!=run)
        for d in dirs:
            casefile=d/'quality_evaluation/cases.jsonl'
            metrics=d/'quality_evaluation/metrics.json'
            pred=d/'predictions.jsonl'
            if not all(p.is_file() for p in (casefile,metrics,pred)): continue
            qm=s0.read(metrics)
            if qm.get('dataset_sha256')!=s0.sha(s0.VAL) or qm.get('predictions_sha256')!=s0.sha(pred): continue
            for c in q.load_jsonl(casefile):
                r=val.get(c['id'])
                if not r or r['is_impossible'] or c['refused'] or not c['prediction'].strip(): continue
                payload={'question':r['question'],'candidate_answer':c['prediction'],'reference_answer':r['answer'],'reference_documents':q.gold_excerpts(r)}
                for system,field in ((q.CORRECT_SYSTEM,'correctness'),(s0.CLAIM_SYSTEM,'claim_review')):
                    v=c.get(field,{})
                    if v.get('format_valid') is True: cache[key(system,payload)]=v
        original=q.Evaluator
        class CachedEvaluator:
            def __init__(self,path): self.path,self.inner=path,None
            def evaluate(self,system,payload,allowed):
                k=key(system,payload)
                if k in cache and cache[k]['verdict'] in allowed: return cache[k]
                if self.inner is None: self.inner=original(self.path)
                result=self.inner.evaluate(system,payload,allowed)
                if result.get('format_valid') is True: cache[k]=result
                return result
        q.Evaluator=CachedEvaluator
        s0.quality_worker(run,method)
    else: raise ValueError(phase)


def phase(run, name, method):
    tag=f'{name}_{method}'
    done=run/'completed'/f'{tag}.json'
    if done.is_file():
        for path,digest in s0.read(done)['files'].items():
            if s0.sha(Path(path))!=digest: raise ValueError(f'Completed artifact changed: {path}')
        return
    if name=='train': targets=[adapter(run,method),run/'training'/f'{method}_train_metrics.json']
    elif name=='generate': targets=[run/method]
    else: targets=[run/method/'quality_evaluation']
    for p in targets:
        if p.exists(): p.rename(p.with_name(p.name+'.incomplete-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    log=run/'logs'/f'{tag}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name,method,'--run',str(run)]
    with log.open('a',encoding='utf-8') as f:
        with subprocess.Popen(cmd,cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}) as proc:
            try:
                for line in proc.stdout:
                    f.write(line); f.flush()
                    text=line.strip()
                    if text.startswith('[Training]'): print(text,flush=True)
                    elif (m:=re.fullmatch(r'Generated: (\d+/\d+)',text)): print('[Evaluation] Question '+m[1],flush=True)
                    elif (m:=re.fullmatch(r'\[(?:lora|qlora) quality\] (\d+/\d+)',text)): print('[Quality] Question '+m[1],flush=True)
                code=proc.wait()
            except BaseException:
                proc.terminate(); proc.wait(); raise
    if code: raise RuntimeError(f'{tag} failed: {log}\n'+'\n'.join(log.read_text(encoding='utf-8').splitlines()[-15:]))
    files=[x for p in targets for x in ([p] if p.is_file() else list(p.rglob('*'))) if x.is_file()]
    if not files: raise RuntimeError(f'No artifacts: {tag}')
    s0.write(done,{'files':{str(p):s0.sha(p) for p in files}})


def metrics(run,method):
    g=s0.read(run/method/'metrics.json'); q=s0.read(run/method/'quality_evaluation/metrics.json'); t=s0.read(run/'training'/f'{method}_train_metrics.json')
    return dict(token_f1_percent=g['token_f1_percent'],answer_correctness_percent=q['answer_correctness_percent'],correct_refusal_rate_percent=g['abstention_rate_percent'],false_refusal_rate_percent=g['false_refusal_rate_percent'],exact_match_percent=g['exact_match_percent'],reference_contradiction_percent=q['reference_contradiction_percent'],reference_unsupported_percent=q['reference_unsupported_percent'],best_val_loss=t['best_val_loss'],training_seconds=t['training_seconds'],training_gpu_gib=t['peak_gpu_memory_gb'],inference_gpu_gib=g['peak_gpu_memory_gb'],questions_per_second=g['questions_per_second'],train_samples=t['train_samples'])


def select(results,baseline):
    # Provisional screening criterion, not statistical significance or independent human judgement.
    # Correct-refusal floor is the already proposed first-pass target (50%).
    eligible={n:r for n,r in results.items() if r['correct_refusal_rate_percent']>=50 and r['false_refusal_rate_percent']<baseline['false_refusal_rate_percent'] and r['token_f1_percent']>=baseline['token_f1_percent'] and r['answer_correctness_percent']>=baseline['answer_correctness_percent'] and (r['token_f1_percent']>baseline['token_f1_percent'] or r['answer_correctness_percent']>baseline['answer_correctness_percent'])}
    if not eligible: return 'baseline'
    return max(eligible,key=lambda n:(eligible[n]['answer_correctness_percent'],eligible[n]['token_f1_percent'],-eligible[n]['false_refusal_rate_percent'],eligible[n]['correct_refusal_rate_percent']))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--baseline',type=Path); p.add_argument('--resume',type=Path); p.add_argument('--prepare-only',action='store_true'); p.add_argument('--worker',nargs=2,help=argparse.SUPPRESS); p.add_argument('--run',type=Path,help=argparse.SUPPRESS); a=p.parse_args()
    if a.worker: worker(a.run,*a.worker); return
    if a.resume:
        root=a.resume.resolve()
        if root.parent!=RUNS.resolve(): raise ValueError('Invalid resume directory')
        previous=s0.read(root/'manifest.json'); a.baseline=Path(previous['baseline'])
    baseline,provenance=s1.load_baseline(a.baseline)
    data=s0.audit(); rows=s0.validate(s0.TRAIN)
    candidates,reviews=prepare_candidates(rows)
    manifest={'baseline':str(baseline),'baseline_manifest_sha256':s0.sha(baseline/'manifest.json'),'baseline_comparison_sha256':s0.sha(baseline/'comparison.json'),'data':data,'sources':provenance['sources'],'model_files':provenance['base_model_files'],'runner_sha256':{p.name:s0.sha(p) for p in (Path(__file__),Path(s0.__file__),Path(s1.__file__))},'settings':s0.SETTINGS,'candidate_content_hashes':{n:canonical_hash(r) for n,r in candidates.items()},'selection':'Correct refusal >= 50%, lower false refusal, nondecreasing F1 and correctness with at least one strict improvement; prioritize correctness then F1. Provisional, human review required.'}
    if a.resume:
        if previous!=manifest: raise ValueError('Resume blocked: provenance changed')
    else:
        root=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')); root.mkdir(parents=True,exist_ok=False); s0.write(root/'manifest.json',manifest)
    s0.write(root/'review_decisions.json',reviews)
    info={}; original={r['id']:r for r in rows}; val=s0.validate(s0.VAL); vi={r['id'] for r in val}; vq={s0.norm(r['question']) for r in val}
    for n,records in candidates.items():
        if any(r['id'] in vi or s0.norm(r['question']) in vq for r in records): raise ValueError('Candidate leakage')
        dest=root/'experiments'/n/'train.json'
        if dest.exists() and canonical_hash(s0.validate(dest))!=canonical_hash(records): raise ValueError('Candidate data changed')
        if not dest.exists(): s0.write(dest,records)
        info[n]={'counts':s1.counts(records),'sha256':s0.sha(dest),'removed_ids':sorted(set(original)-{r['id'] for r in records}),'edited_ids':[r['id'] for r in records if r!=original[r['id']]]}
    s0.write(root/'candidates.json',info)
    print(f'[Stage 1] Run: {root}',flush=True)
    if a.prepare_only:
        for n,v in info.items(): print(n,v['counts'],flush=True)
        print('Training: NOT RUN',flush=True); return
    base=s0.read(baseline/'comparison.json'); results={}
    try:
        for i,n in enumerate(NAMES,1):
            print(f'[Experiment] {i}/8 | {n}',flush=True)
            run=root/'experiments'/n
            for name in ('train','generate','quality'): phase(run,name,'lora')
            results[n]=metrics(run,'lora')
        print('[Experiment] 7/8 | Select data',flush=True)
        selected=select(results,base['lora']); s0.write(root/'selection.json',{'selected':selected,'criterion':manifest['selection'],'lora':results})
        chosen=rows if selected=='baseline' else candidates[selected]; s0.write(root/'selected_train.json',chosen)
        print('[Experiment] 8/8 | QLoRA verification',flush=True)
        if selected=='baseline': qr=base['qlora']; verification='REUSED_STAGE0_UNCHANGED_DATA'
        else:
            run=root/'experiments'/selected
            for name in ('train','generate','quality'): phase(run,name,'qlora')
            qr=metrics(run,'qlora'); verification='MEASURED'
        if s0.sha(s0.TRAIN)!=data['train_sha256'] or s0.sha(s0.VAL)!=data['val_sha256']: raise RuntimeError('Original dataset changed')
        report={'baseline':base,'candidates':info,'lora':results,'selected_lora_data':selected,'qlora_verification':verification,'qlora':qr,'qlora_passes_same_screen':select({selected:qr},base['qlora'])!='baseline' if selected!='baseline' else None,'human_review_required':True,'hallucination_rate_percent':None,'note':'Independent changes against Stage 0; no untested combination adopted. Stage 1 uses val_106 only.'}
        s0.write(root/'comparison.json',report)
        lines=['[Stage 1 Experiment Comparison]',f'{"Candidate":24} {"F1":>8} {"Correct":>8} {"CRR":>8} {"FRR":>8} {"Val loss":>10}']
        for n,r in {'baseline':base['lora'],**results,'selected_qlora':qr}.items():
            lines.append(f'{n:24} '+ ' '.join(f'{r[k]:8.2f}' for k in ('token_f1_percent','answer_correctness_percent','correct_refusal_rate_percent','false_refusal_rate_percent'))+f' {r["best_val_loss"]:10.4f}')
        lines.extend([f'Selected LoRA data: {selected}',f'QLoRA: {verification}',f'QLoRA passes screen: {report["qlora_passes_same_screen"]}', 'Automated quality verdicts require human review.',f'Results: {root}'])
        (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines),flush=True)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/run_stage1_experiments.py --resume "{root}"',flush=True); raise

if __name__=='__main__': main()
