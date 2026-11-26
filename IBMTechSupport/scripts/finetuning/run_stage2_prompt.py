"""Test3 / Stage 2-1: change only the shared training/inference system prompt.
Place in IBMTechSupport/scripts/finetuning alongside the Stage 0/1 runners.
Uses the completed Stage 1-2 539-row dataset; never edits source files or test data.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import run_stage0 as s0
import run_stage1_quality as quality

ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage2_prompt'
OLD_PROMPT = ('You are an IBM technical support assistant. Answer technical questions '
              'using only your existing knowledge. If you cannot provide a reliable '
              'answer, respond with exactly INSUFFICIENT_INFORMATION. Otherwise, '
              'provide a concise technical answer in English.')
NEW_PROMPT = ('You are an IBM technical support assistant. Answer technical questions '
              'using only your existing knowledge. Provide a concise technical answer '
              'in English whenever you can answer the question reliably. Do not refuse '
              'solely because nonessential details are missing. If essential information '
              'is missing or you cannot provide a reliable answer, respond with exactly '
              'INSUFFICIENT_INFORMATION.')
KEYS = ('token_f1_percent', 'answer_correctness_percent', 'correct_refusal_rate_percent',
        'false_refusal_rate_percent', 'exact_match_percent', 'best_val_loss',
        'training_seconds', 'training_gpu_gib', 'inference_gpu_gib',
        'questions_per_second', 'reference_contradiction_percent', 'reference_unsupported_percent')


def adapter(run, method):
    return ROOT/'models/finetuning/stage2_prompt'/run.parent.parent.name/method


def check_done(run, tag):
    marker = s0.read(run/'completed'/f'{tag}.json')
    files = marker.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError(f'Invalid completion marker: {tag}')
    for filename, digest in files.items():
        if s0.sha(Path(filename)) != digest:
            raise ValueError(f'Completed artifact changed: {filename}')


def load_parent(path=None):
    if path is None:
        paths = [p for p in quality.RUNS.glob('run_*') if (p/'comparison.json').is_file()]
        if not paths:
            raise FileNotFoundError('Completed Stage 1-2 quality run required')
        path = max(paths, key=lambda p: p.name)
    path = Path(path).resolve()
    if path.parent != quality.RUNS.resolve():
        raise ValueError('Expected Stage 1-2 quality run directory')
    m = s0.read(path/'manifest.json')
    if m['experiment'] != 'answer_quality':
        raise ValueError('Unexpected parent experiment')
    cleanup_path, baseline, provenance, cleaned, _ = quality.load_cleanup(
        Path(m['cleanup_run']), Path(m['baseline']))
    expected, info = quality.prepare_candidate(s0.validate(s0.TRAIN), cleaned)
    if (m['settings'] != s0.SETTINGS or m['data'] != s0.audit()
            or m['sources'] != provenance['sources']
            or m['model_files'] != provenance['base_model_files']
            or m['candidate'] != info):
        raise ValueError('Stage 1-2 provenance mismatch')
    for p, key in ((cleanup_path/'manifest.json','cleanup_manifest_sha256'),
                   (cleanup_path/'comparison.json','cleanup_comparison_sha256'),
                   (baseline/'manifest.json','baseline_manifest_sha256'),
                   (baseline/'comparison.json','baseline_comparison_sha256')):
        if s0.sha(p) != m[key]:
            raise ValueError(f'Ancestor artifact changed: {p}')
    for module in (s0, quality.s1, quality.cleanup, quality):
        f = Path(module.__file__)
        if s0.sha(f) != m['runner_sha256'][f.name]:
            raise ValueError(f'Parent runner changed: {f.name}')
    if provenance['runner_sha256'] != s0.sha(Path(s0.__file__)):
        raise ValueError('Stage 0 runner changed')
    run = path/'experiments/answer_quality'
    for tag in ('train_lora','generate_lora','quality_lora'):
        check_done(run, tag)
    source = run/'train.json'
    rows = s0.validate(source)
    if (rows != expected or s0.read(path/'candidate.json')['file_sha256'] != s0.sha(source)
            or quality.s1.counts(rows) != {'total':539,'answerable':367,'impossible':172}):
        raise ValueError('Stage 1-2 dataset mismatch')
    metrics = quality.metrics(run, 'lora')
    if metrics != s0.read(path/'comparison.json')['candidate_lora']:
        raise ValueError('Stage 1-2 metrics mismatch')
    return path, baseline, provenance, source, metrics


def config_for_run(run):
    return run/'prompt_config.yaml'


def make_config():
    import yaml
    config = yaml.safe_load(s0.CONFIG.read_text(encoding='utf-8-sig'))
    if config['prompt']['system'] != OLD_PROMPT:
        raise ValueError('Original inference prompt differs from expected Stage 0 prompt')
    if config['prompt']['user_template'] != '{question}':
        raise ValueError('Unexpected user template')
    config['prompt']['system'] = NEW_PROMPT
    return config


def worker(run, name):
    # New output paths and runtime prompt override only. The original sources are immutable.
    quality.adapter = adapter
    if name == 'train':
        m = importlib.import_module('train_lora')
        if m.SYSTEM_PROMPT != OLD_PROMPT:
            raise ValueError('Original training prompt differs from Stage 0')
        m.SYSTEM_PROMPT = NEW_PROMPT
    elif name == 'generate':
        from evaluate import evaluate_lora
        evaluate_lora.BASE_CONFIG = config_for_run(run)
    elif name == 'quality':
        s0.CONFIG = config_for_run(run)
        generated = s0.read(run/'lora/generation_manifest.json')
        if generated['prompt'] != {'system':NEW_PROMPT,'user_template':'{question}'}:
            raise ValueError('Generated answers used the wrong prompt')
    else:
        raise ValueError(name)
    quality.worker(run, name, 'lora')


def phase(run, name):
    tag = name+'_lora'
    marker = run/'completed'/f'{tag}.json'
    if marker.is_file():
        check_done(run, tag)
        return
    if name == 'train':
        targets = [adapter(run,'lora'), run/'training/lora_train_metrics.json']
    elif name == 'generate':
        targets = [run/'lora']
    else:
        targets = [run/'lora/quality_evaluation']
    for p in targets:
        if p.exists():
            p.rename(p.with_name(p.name+'.incomplete-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    log = run/'logs'/f'{tag}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name,'--run',str(run)]
    with log.open('a', encoding='utf-8') as f:
        with subprocess.Popen(cmd,cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',bufsize=1,
                              env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}) as proc:
            try:
                for line in proc.stdout:
                    f.write(line); f.flush()
                    line = line.strip()
                    if line.startswith('[Training]'):
                        print(line,flush=True)
                    elif (match := re.fullmatch(r'Generated: (\d+/\d+)',line)):
                        print('[Evaluation] Question '+match[1],flush=True)
                    elif (match := re.fullmatch(r'\[lora quality\] (\d+/\d+)',line)):
                        print('[Quality] Question '+match[1],flush=True)
                code = proc.wait()
            except BaseException:
                proc.terminate(); proc.wait(); raise
    if code:
        raise RuntimeError(f'{tag} failed: {log}\n'+'\n'.join(log.read_text(encoding='utf-8').splitlines()[-15:]))
    files = [f for p in targets for f in ([p] if p.is_file() else list(p.rglob('*'))) if f.is_file()]
    if not files or not all(p.exists() for p in targets):
        raise RuntimeError(f'Missing artifacts: {tag}')
    s0.write(marker, {'files':{str(p):s0.sha(p) for p in files}})


def report(root, run, parent_metrics, manifest):
    result = quality.metrics(run,'lora')
    if result['train_samples'] != 539:
        raise ValueError('Wrong training count')
    delta = {k: round(result[k]-parent_metrics[k],4)
             if result.get(k) is not None and parent_metrics.get(k) is not None else None for k in KEYS}
    selected = quality.select({'system_prompt':result},parent_metrics)
    value = {'baseline_run':manifest['parent_run'],'baseline_lora':parent_metrics,
             'candidate_lora':result,'delta':delta,'provisional_selection':selected,
             'selection_rule':manifest['selection_rule'],'prompt_before':OLD_PROMPT,
             'prompt_after':NEW_PROMPT,'human_review_required':True,'qlora':'NOT RUN',
             'hallucination_rate_percent':None,
             'limitation':'Joint training/inference prompt effect; validation loss is conditioned on different prompts.'}
    s0.write(root/'comparison.json',value)
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    lines = ['[Test3 System Prompt Comparison]',f'{"Metric":36} {"Stage 1-2":>12} {"Prompt":>12} {"Delta":>12}']
    for k in KEYS:
        lines.append(f'{k:36} {fmt(parent_metrics.get(k)):>12} {fmt(result.get(k)):>12} {fmt(delta[k]):>12}')
    lines.extend([f'Provisional selection: {selected}', 'QLoRA: NOT RUN',
                  'Validation loss uses different prompts. Automated quality verdicts require human review.',f'Results: {root}'])
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,help='Completed Stage 1-2 run (default: latest)')
    p.add_argument('--resume',type=Path,help='Resume the same Test3 run')
    p.add_argument('--prepare-only',action='store_true',help='Verify and prepare without GPU training')
    p.add_argument('--worker',choices=('train','generate','quality'),help=argparse.SUPPRESS)
    p.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        if args.run is None or args.run.resolve().parent.parent.parent != RUNS.resolve():
            p.error('Invalid worker directory')
        worker(args.run.resolve(),args.worker)
        return
    if args.resume:
        root = args.resume.resolve()
        if root.parent != RUNS.resolve():
            p.error('Invalid resume directory')
        previous = s0.read(root/'manifest.json')
        args.parent = Path(previous['parent_run'])
    parent, baseline, provenance, source, base = load_parent(args.parent)
    cfg = make_config()
    data = s0.audit()
    manifest = {'stage':2,'test':3,'experiment':'system_prompt','parent_run':str(parent),
                'baseline':str(baseline),'parent_manifest_sha256':s0.sha(parent/'manifest.json'),
                'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
                'training_source':str(source),'training_sha256':s0.sha(source),
                'data':data,'settings':s0.SETTINGS,'sources':provenance['sources'],
                'model_files':provenance['base_model_files'],'prompt_before':OLD_PROMPT,'prompt_after':NEW_PROMPT,
                'inference_config':cfg,
                'runner_sha256':{f.name:s0.sha(f) for f in (Path(__file__),Path(s0.__file__),Path(quality.__file__),Path(quality.s1.__file__),Path(quality.cleanup.__file__))},
                'selection_rule':'Provisional screen: correct refusal >=50%, lower false refusal, F1 and correctness nondecreasing with at least one strictly improved. Human review required.'}
    if args.resume:
        if previous != manifest:
            raise ValueError('Resume blocked: provenance changed')
    else:
        root = RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False)
        s0.write(root/'manifest.json',manifest)
    run = root/'experiments/system_prompt'
    run.mkdir(parents=True,exist_ok=True)
    dest = run/'train.json'
    if not dest.exists():
        dest.write_bytes(source.read_bytes())
    if s0.sha(dest) != manifest['training_sha256']:
        raise ValueError('Training data changed')
    # JSON is valid YAML; keeping a separate snapshot gives the existing evaluator an actual config hash.
    cpath = config_for_run(run)
    if not cpath.exists():
        s0.write(cpath,cfg)
    if s0.read(cpath) != cfg:
        raise ValueError('Prompt config changed')
    print(f'[Stage 2] Run: {root}',flush=True)
    print('[Experiment] 1/1 | System Prompt (539 unchanged)',flush=True)
    if args.prepare_only:
        print('Training: NOT RUN',flush=True)
        return
    try:
        for name in ('train','generate','quality'):
            phase(run,name)
        if (s0.sha(dest) != manifest['training_sha256'] or s0.sha(source) != manifest['training_sha256']
                or s0.audit() != data or s0.read(cpath) != cfg
                or s0.sha(parent/'comparison.json') != manifest['parent_comparison_sha256']):
            raise RuntimeError('Inputs changed during experiment')
        report(root,run,base,manifest)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/run_stage2_prompt.py --resume "{root}"',flush=True)
        raise


if __name__ == '__main__':
    main()
