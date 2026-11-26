"""Stage 2 Test4: lower refusal loss weights, controlled against Test3 B_weight_025.
Place alongside the previous runners in IBMTechSupport/scripts/finetuning.
Only the per-sample training loss changes; validation is unweighted. No test data.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import run_stage2_test2 as t2
import run_stage2_test3 as t3

s0, quality, previous = t2.s0, t2.quality, t2.previous
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage2_test4'
WEIGHTS = {'C_weight_010': 0.1, 'D_weight_005': 0.05}
PHASES = ('train', 'generate', 'quality')


def adapter(run, method):
    if method != 'lora':
        raise ValueError('This experiment is LoRA only')
    return ROOT/'models/finetuning/stage2_test4'/run.parent.parent.name/run.name/method


def source_hashes():
    modules = (s0, quality, quality.s1, quality.cleanup, previous, t2, t3)
    return {f.name: s0.sha(f) for f in (Path(__file__), *(Path(m.__file__) for m in modules))}


def load_parent(path=None):
    if path is None:
        candidates = [p for p in t3.RUNS.glob('run_*') if (p/'comparison.json').is_file()
                      and s0.read(p/'comparison.json').get('complete')]
        if not candidates:
            raise FileNotFoundError('Complete Stage 2 Test3 first')
        path = max(candidates, key=lambda p: p.name)
    path = Path(path).resolve()
    if path.parent != t3.RUNS.resolve():
        raise ValueError('Expected Stage 2 Test3 run directory')
    m = s0.read(path/'manifest.json')
    c = s0.read(path/'comparison.json')
    if (m['stage'],m['test'],c['complete'],c['provisional_selection']) != (2,3,True,'B_weight_025'):
        raise ValueError('Expected completed Test3 with B_weight_025 selected')
    if m['weights'] != t3.WEIGHTS or c['weights'] != t3.WEIGHTS:
        raise ValueError('Test3 weights changed')
    for name,digest in m['runner_sha256'].items():
        if source_hashes().get(name) != digest:
            raise ValueError(f'Previous runner changed: {name}')
    parent,base,parent_manifest,spec,cfg,source,baseline_metrics = t3.load_parent(Path(m['parent_run']))
    if (s0.sha(parent/'manifest.json') != m['parent_manifest_sha256']
            or s0.sha(parent/'comparison.json') != m['parent_comparison_sha256']
            or m['baseline'] != str(base) or m['training_sha256'] != s0.sha(source)
            or m['spec'] != spec or m['config'] != cfg or m['data'] != s0.audit()
            or m['sources'] != parent_manifest['sources'] or m['model_files'] != parent_manifest['model_files']
            or m['settings'] != dict(s0.SETTINGS,max_length=spec['max_length'])
            or c['baseline_lora'] != baseline_metrics):
        raise ValueError('Test3 provenance mismatch')
    for name in t3.WEIGHTS:
        run=path/'experiments'/name
        for phase in PHASES:
            previous.check_done(run,phase+'_lora')
        if s0.sha(run/'train.json') != m['training_sha256'] or s0.read(t2.config_for(run)) != cfg:
            raise ValueError('Test3 data/config mismatch')
        if quality.metrics(run,'lora') != c['candidates'][name]:
            raise ValueError('Test3 candidate metrics mismatch')
    run=path/'experiments/B_weight_025'
    audit=s0.read(run/'training/weight_audit.json')
    if audit['impossible_weight'] != 0.25 or audit['answerable_weight'] != 1.0 or audit['validation_weight'] != 1.0:
        raise ValueError('Expected weight audit for B_weight_025')
    return path,base,m,spec,cfg,run/'train.json',c['candidates']['B_weight_025']


def apply_sample_weight(output, weight, training):
    # Keep the denominator and accumulation behavior of the original trainer.
    # Weight the scalar per-example mean CE before GradScaler/backward.
    if training:
        if weight is None or weight.numel() != 1:
            raise ValueError('Expected exactly one loss weight for batch size 1')
        output.loss = output.loss * weight.reshape(()).to(device=output.loss.device, dtype=output.loss.dtype)
    return output


def install_weighting(module, impossible_weight):
    import torch
    if impossible_weight not in WEIGHTS.values():
        raise ValueError('Unexpected class weight')
    original_dataset = module.QADataset
    original_loader = module.load_lora_model
    stats = {'answerable':0, 'impossible':0, 'validation':0}

    class WeightedDataset(original_dataset):
        def __init__(self, records, tokenizer, max_length):
            super().__init__(records, tokenizer, max_length)
            if len(self) != len(records):
                raise ValueError('Dataset order/length mismatch')
            self.loss_weights = [impossible_weight if r['is_impossible'] else 1.0 for r in records]
            self.impossible = [r['is_impossible'] for r in records]

        def __getitem__(self, index):
            # Do not modify input_ids, attention_mask or labels.
            item = dict(super().__getitem__(index))
            item['sample_loss_weight'] = torch.tensor(self.loss_weights[index], dtype=torch.float32)
            return item

    def weighted_loader():
        model, tokenizer = original_loader()
        original_forward = model.forward

        @functools.wraps(original_forward)
        def forward(*args, **kwargs):
            weight = kwargs.pop('sample_loss_weight', None)
            if weight is None:
                raise ValueError('Missing sample loss weight')
            # Existing evaluate_loss sets eval() then restores train().
            # Never change the validation objective or checkpoint selection.
            if model.training:
                value = float(weight.item())
                if value == 1.0:
                    stats['answerable'] += 1
                elif math.isclose(value, impossible_weight, rel_tol=1e-6, abs_tol=1e-8):
                    stats['impossible'] += 1
                else:
                    raise ValueError(f'Unexpected sample weight: {value}')
            else:
                stats['validation'] += 1
            return apply_sample_weight(original_forward(*args, **kwargs), weight, model.training)

        model.forward = forward
        return model, tokenizer

    module.QADataset = WeightedDataset
    module.load_lora_model = weighted_loader
    return stats


def worker(run, phase):
    m = s0.read(run.parent.parent/'manifest.json')
    spec = m['spec']
    cfg = s0.read(t2.config_for(run))
    if cfg != m['config'] or s0.sha(run/'train.json') != m['training_sha256'] or s0.sha(s0.VAL) != m['data']['val_sha256']:
        raise ValueError('Experiment data/config changed')
    quality.adapter = adapter
    s0.adapter = adapter
    if phase == 'train':
        trainer = importlib.import_module('train_lora')
        if (trainer.SYSTEM_PROMPT,trainer.MAX_QUESTION_TOKENS,trainer.MAX_ANSWER_TOKENS) != (t2.OLD_PROMPT,192,160):
            raise ValueError('Original trainer settings changed')
        t2.configure_trainer(spec)
        s0.SETTINGS = dict(s0.SETTINGS, max_length=spec['max_length'])
        stats = install_weighting(trainer, m['weights'][run.name])
        quality.worker(run,'train','lora')
        expected = {'answerable':367*s0.SETTINGS['epochs'], 'impossible':172*s0.SETTINGS['epochs'], 'validation':106*s0.SETTINGS['epochs']}
        if stats != expected:
            raise ValueError(f'Observed loss-weight application counts {stats} != {expected}')
        s0.write(run/'training/weight_audit.json', {'observed_forward_counts':stats,
            'answerable_weight':1.0,'impossible_weight':m['weights'][run.name],
            'validation_weight':1.0,'normalization':'original sample-count denominator',
            'training_loss_is_weighted':True})
    elif phase == 'generate':
        from evaluate import evaluate_lora, common
        common.MAX_NEW_TOKENS = spec['max_new_tokens']
        evaluate_lora.ADAPTER = adapter(run,'lora')
        evaluate_lora.OUTPUT = run/'lora'
        evaluate_lora.BASE_CONFIG = t2.config_for(run)
        evaluate_lora.generate(cfg, s0.validate(s0.VAL))
    elif phase == 'quality':
        from evaluate import common
        common.MAX_NEW_TOKENS = spec['max_new_tokens']
        generated = s0.read(run/'lora/generation_manifest.json')
        if generated['prompt'] != cfg['prompt'] or generated['inference'] != cfg['inference']:
            raise ValueError('Generation config mismatch')
        s0.CONFIG = t2.config_for(run)
        quality.worker(run,'quality','lora')
    else:
        raise ValueError(phase)


def execute_phase(run, phase):
    tag = phase+'_lora'
    marker = run/'completed'/f'{tag}.json'
    if marker.exists():
        previous.check_done(run, tag)
        return
    targets = ([adapter(run,'lora'), run/'training'] if phase == 'train' else
               [run/'lora'] if phase == 'generate' else [run/'lora/quality_evaluation'])
    for p in targets:
        if p.exists():
            p.rename(p.with_name(p.name+'.incomplete-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    log = run/'logs'/f'{tag}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',phase,'--run',str(run)]
    env = {**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}
    with log.open('a',encoding='utf-8') as f:
        with subprocess.Popen(cmd,cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,env=env) as proc:
            try:
                for line in proc.stdout:
                    f.write(line); f.flush()
                    line = line.strip()
                    if line.startswith('[Training]'):
                        print(line,flush=True)
                    elif match := re.fullmatch(r'Generated: (\d+/\d+)',line):
                        print('[Evaluation] Question '+match[1],flush=True)
                    elif match := re.fullmatch(r'\[lora quality\] (\d+/\d+)',line):
                        print('[Quality] Question '+match[1],flush=True)
                code = proc.wait()
            except BaseException:
                proc.terminate(); proc.wait(); raise
    if code:
        raise RuntimeError(f'{tag} failed: {log}\n'+'\n'.join(log.read_text(encoding='utf-8').splitlines()[-15:]))
    files = [f for p in targets for f in ([p] if p.is_file() else list(p.rglob('*'))) if f.is_file()]
    if not files or not all(p.exists() for p in targets):
        raise RuntimeError(f'Missing artifacts: {tag}')
    s0.write(marker, {'files':{str(f):s0.sha(f) for f in files}})


def report(root, base, results):
    selected = quality.select(results, base)
    delta = {n:{k:round(r[k]-base[k],4) if r.get(k) is not None and base.get(k) is not None else None for k in t2.KEYS} for n,r in results.items()}
    s0.write(root/'comparison.json', {'baseline_lora':base,'candidates':results,'delta_vs_baseline':delta,
        'weights':WEIGHTS,'baseline_weight':0.25,'provisional_selection':selected,'complete':len(results)==len(WEIGHTS),
        'human_review_required':True,'hallucination_rate_percent':None,'qlora':'NOT RUN',
        'notes':['Independent training from base weights for each candidate.','Only training loss is weighted; validation loss is unweighted.',
                 'Absolute gradient scale may also change because the sample-count denominator is unchanged.',
                 'Correct-refusal floor 50%; F1 and correctness must not decline, at least one must improve, and false-refusal must decrease.']})
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    lines=['[Stage 2 Test4 Loss Weighting Comparison]',f'{"Metric":36} {"B_weight_025":>12} '+ ' '.join(f'{n:>14}' for n in results)]
    for k in t2.KEYS:
        lines.append(f'{k:36} {fmt(base.get(k)):>12} '+' '.join(f'{fmt(r.get(k)):>14}' for r in results.values()))
    lines += [f'Provisional selection: {selected}','Validation loss: unweighted. Training loss: weighted.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN', f'Results: {root}']
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,help='Completed Stage 2 Test3 directory (default latest)')
    p.add_argument('--resume',type=Path,help='Resume same run without retraining completed phases')
    p.add_argument('--prepare-only',action='store_true',help='Check provenance and prepare without GPU training')
    p.add_argument('--through',choices=['C','D'],default='D',help='Run only C or both candidates')
    p.add_argument('--worker',choices=PHASES,help=argparse.SUPPRESS)
    p.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        if args.run is None or args.run.resolve().parent.parent.parent!=RUNS.resolve() or args.run.name not in WEIGHTS:
            p.error('Invalid worker directory')
        worker(args.run.resolve(),args.worker);return
    if args.resume:
        root=args.resume.resolve()
        if root.parent!=RUNS.resolve():p.error('Invalid resume directory')
        old=s0.read(root/'manifest.json'); args.parent=Path(old['parent_run'])
    parent,base,m,spec,cfg,source,metrics=load_parent(args.parent)
    manifest={'stage':2,'test':4,'experiment':'loss_weighting_refinement','parent_run':str(parent),'baseline':str(base),
              'parent_manifest_sha256':s0.sha(parent/'manifest.json'),'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
              'training_source':str(source),'training_sha256':s0.sha(source),'data':m['data'],
              'sources':m['sources'],'model_files':m['model_files'],'spec':spec,'config':cfg,
              'settings':dict(s0.SETTINGS,max_length=spec['max_length']),'weights':WEIGHTS,'baseline_weight':0.25,
              'runner_sha256':source_hashes()}
    if args.resume:
        if old!=manifest:raise ValueError('Resume blocked: provenance changed')
    else:
        root=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False);s0.write(root/'manifest.json',manifest)
    print(f'[Stage 2 Test4] Run: {root}',flush=True)
    try:
        results={}
        for i,name in enumerate(WEIGHTS,1):
            if args.through=='C' and i>1:break
            run=root/'experiments'/name;run.mkdir(parents=True,exist_ok=True)
            dest=run/'train.json'
            if not dest.exists():dest.write_bytes(source.read_bytes())
            if s0.sha(dest)!=manifest['training_sha256']:raise ValueError('Training data changed')
            cpath=t2.config_for(run)
            if not cpath.exists():s0.write(cpath,cfg)
            if s0.read(cpath)!=cfg:raise ValueError('Config changed')
            if args.prepare_only:continue
            print(f'[Experiment] {i}/2 | {name}',flush=True)
            for phase in PHASES:execute_phase(run,phase)
            result=quality.metrics(run,'lora')
            if result['train_samples']!=539:raise ValueError('Unexpected training count')
            results[name]=result
            if s0.audit()!=manifest['data'] or s0.sha(source)!=manifest['training_sha256']:
                raise ValueError('Dataset changed during experiment')
            report(root,metrics,results)
        if args.prepare_only:
            print('Preparation: OK (no training)',flush=True)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/run_stage2_test4.py --resume "{root}" --through {args.through}',flush=True)
        raise


if __name__=='__main__':main()
