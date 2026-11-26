"""Stage 1, first controlled experiment: duplicate + conflict cleanup only.
Place next to run_stage0.py and run_stage1.py in scripts/finetuning/.
No edits to original data, training scripts, models, or baseline results.
"""
from __future__ import annotations

import argparse
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
RUNS = ROOT / 'evaluate_results/106/finetuning/stage1_cleanup'
REVIEWED_DATA_SHA = '388359a011a3ce7384d359ee8b9e0eeed4e6afca00bcc8d651488a1a02396c6c'

def canonical_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def prepare_candidate(rows):
    if canonical_hash(rows) != REVIEWED_DATA_SHA:
        raise ValueError('Training content differs from the reviewed 551 records.')
    audit = s1.duplicate_audit(rows)
    exact = set(audit['exact_sft_duplicate_ids'])
    conflict = set(audit['conflict_ids'])
    if (len(exact), len(conflict), len(exact & conflict)) != (2, 10, 0):
        raise ValueError('Unexpected duplicate or conflict counts')
    candidate = [r for r in rows if r['id'] not in exact | conflict]
    if s1.counts(candidate) != {'total':539,'answerable':367,'impossible':172}:
        raise ValueError('Unexpected candidate counts')
    return candidate, {'exact_duplicate_ids':sorted(exact), 'conflicting_label_ids':sorted(conflict),
                       'removed_ids':sorted(exact | conflict), 'counts':s1.counts(candidate),
                       'changed_answer_ids':[], 'candidate_content_sha256':canonical_hash(candidate)}


def adapter(run, method):
    return ROOT/'models/finetuning/stage1_cleanup'/run.parent.parent.name/run.name/method


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, help='Specific completed Stage 0 directory (default: latest)')
    p.add_argument('--resume', type=Path, help='Resume the same cleanup experiment')
    p.add_argument('--prepare-only', action='store_true', help='Audit and prepare data without training')
    p.add_argument('--worker', nargs=2, help=argparse.SUPPRESS)
    p.add_argument('--run', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        if not a.run or a.worker[0] not in ('train','generate','quality') or a.worker[1] != 'lora':
            p.error('Invalid worker arguments')
        worker(a.run, *a.worker)
        return
    if a.resume:
        root = a.resume.resolve()
        if root.parent != RUNS.resolve():
            raise ValueError('Invalid resume directory')
        previous = s0.read(root/'manifest.json')
        a.baseline = Path(previous['baseline'])
    baseline, provenance = s1.load_baseline(a.baseline)
    data = s0.audit()
    rows = s0.validate(s0.TRAIN)
    candidate, info = prepare_candidate(rows)
    val = s0.validate(s0.VAL)
    vi = {r['id'] for r in val}
    vq = {s0.norm(r['question']) for r in val}
    if any(r['id'] in vi or s0.norm(r['question']) in vq for r in candidate):
        raise ValueError('Candidate leakage')
    if provenance.get('runner_sha256') != s0.sha(Path(s0.__file__)):
        raise ValueError('Stage 0 evaluator runner changed since baseline')
    manifest = {'stage':1, 'experiment':'combined_cleanup', 'baseline':str(baseline),
                'baseline_manifest_sha256':s0.sha(baseline/'manifest.json'),
                'baseline_comparison_sha256':s0.sha(baseline/'comparison.json'),
                'data':data,'sources':provenance['sources'],'model_files':provenance['base_model_files'],
                'runner_sha256':{f.name:s0.sha(f) for f in (Path(__file__),Path(s0.__file__),Path(s1.__file__))},
                'settings':s0.SETTINGS,'candidate':info,
                'selection':'Provisional screen: correct refusal >= 50%; lower false refusal; F1 and correctness nondecreasing, at least one strictly improved. Human review required.',
                'limitation':'Joint cleanup effect only; individual duplicate and conflict removal effects are not isolated.'}
    if a.resume:
        if previous != manifest:
            raise ValueError('Resume blocked: provenance changed')
    else:
        root = RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False)
        s0.write(root/'manifest.json',manifest)
    run = root/'experiments'/'combined_cleanup'
    dest = run/'train.json'
    if dest.exists():
        if canonical_hash(s0.validate(dest)) != canonical_hash(candidate):
            raise ValueError('Candidate changed')
    else:
        s0.write(dest,candidate)
    s0.write(root/'candidate.json',{**info,'file_sha256':s0.sha(dest)})
    print(f'[Stage 1] Run: {root}',flush=True)
    print('[Experiment] 1/1 | Combined cleanup: 551 -> 539',flush=True)
    if a.prepare_only:
        print('Training: NOT RUN',flush=True)
        return
    base = s0.read(baseline/'comparison.json')['lora']
    try:
        for name in ('train','generate','quality'):
            phase(run,name,'lora')
        result = metrics(run,'lora')
        if s0.sha(s0.TRAIN)!=data['train_sha256'] or s0.sha(s0.VAL)!=data['val_sha256']:
            raise RuntimeError('Original dataset changed')
        if canonical_hash(s0.validate(dest)) != canonical_hash(candidate):
            raise RuntimeError('Candidate changed during run')
        selected = select({'combined_cleanup':result},base)
        keys = ('token_f1_percent','answer_correctness_percent','correct_refusal_rate_percent',
                'false_refusal_rate_percent','exact_match_percent','best_val_loss','training_seconds',
                'training_gpu_gib','inference_gpu_gib','questions_per_second',
                'reference_contradiction_percent','reference_unsupported_percent')
        report = {'baseline_run':str(baseline),'baseline_lora':base,'candidate_lora':result,
                  'candidate':info, 'provisional_selection':selected,'selection_rule':manifest['selection'],
                  'delta':{k:round(result[k]-base[k],4) if result.get(k) is not None and base.get(k) is not None else None for k in keys},
                  'quality_edits':'NOT RUN','qlora':'NOT RUN','human_review_required':True,
                  'hallucination_rate_percent':None,'limitation':manifest['limitation']}
        s0.write(root/'comparison.json',report)
        def fmt(x): return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
        lines=['[Stage 1 Cleanup Comparison]',f'{"Metric":36} {"Stage 0":>12} {"Cleanup":>12} {"Delta":>12}']
        for k in keys:
            lines.append(f'{k:36} {fmt(base.get(k)):>12} {fmt(result.get(k)):>12} {fmt(report["delta"][k]):>12}')
        lines += [f'Provisional selection: {selected}', 'Quality edits and QLoRA: NOT RUN',
                  'Joint effect only. Automated quality verdicts require human review.',f'Results: {root}']
        (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('\n'.join(lines),flush=True)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/run_stage1_cleanup.py --resume "{root}"',flush=True)
        raise


if __name__ == '__main__':
    main()
