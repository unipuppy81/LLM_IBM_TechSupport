"""Stage 2 Test3 E: refusal loss weight 0.175; one training run.
Place beside run_stage2_test4.py. Reuse A-D; screen on generation metrics
before running the unchanged quality evaluator. No test dataset access.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import run_stage2_test4 as t4

s0, t2, t3, quality, previous = t4.s0, t4.t2, t4.t3, t4.quality, t4.previous
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage2_test3_midpoint'
NAME = 'E_weight_0175'
WEIGHT = 0.175
WEIGHTS = {'A_weight_050': 0.5, 'B_weight_025': 0.25,
           'C_weight_010': 0.1, 'D_weight_005': 0.05, NAME: WEIGHT}
PHASES = ('train', 'generate', 'quality')
SCRIPT = Path(__file__).resolve()


def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT / 'models/finetuning/stage2_test3_midpoint' / run.parent.parent.name / run.name / method


def source_hashes():
    return {**t4.source_hashes(), SCRIPT.name: s0.sha(SCRIPT)}


def load_parent(path=None):
    if path is None:
        candidates = [p for p in t4.RUNS.glob('run_*')
                      if (p/'comparison.json').is_file() and s0.read(p/'comparison.json').get('complete')]
        if not candidates:
            raise FileNotFoundError('Complete the original Stage 2 Test4 (C and D) first')
        path = max(candidates, key=lambda p: p.name)
    path = Path(path).resolve()
    if path.parent != t4.RUNS.resolve():
        raise ValueError('Expected original Stage 2 Test4 directory')
    m = s0.read(path/'manifest.json')
    c = s0.read(path/'comparison.json')
    if (m['stage'], m['test'], c['complete']) != (2, 4, True):
        raise ValueError('Expected complete C/D run')
    if m['weights'] != t4.WEIGHTS or c['weights'] != t4.WEIGHTS or c['baseline_weight'] != 0.25:
        raise ValueError('C/D weights mismatch')
    hashes = source_hashes()
    for name, digest in m['runner_sha256'].items():
        if hashes.get(name) != digest:
            raise ValueError(f'Previous runner changed: {name}')
    p3, base, m3, spec, cfg, source, b_metrics = t4.load_parent(Path(m['parent_run']))
    if (s0.sha(p3/'manifest.json') != m['parent_manifest_sha256']
            or s0.sha(p3/'comparison.json') != m['parent_comparison_sha256']
            or m['baseline'] != str(base) or m['training_sha256'] != s0.sha(source)
            or m['data'] != m3['data'] or m['sources'] != m3['sources']
            or m['model_files'] != m3['model_files'] or m['spec'] != spec or m['config'] != cfg
            or m['settings'] != dict(s0.SETTINGS, max_length=spec['max_length'])
            or c['baseline_lora'] != b_metrics):
        raise ValueError('C/D provenance mismatch')
    c3 = s0.read(p3/'comparison.json')
    prior = dict(c3['candidates'])
    for name, weight in t4.WEIGHTS.items():
        run = path/'experiments'/name
        for phase in PHASES:
            previous.check_done(run, phase+'_lora')
        if s0.sha(run/'train.json') != m['training_sha256'] or s0.read(t2.config_for(run)) != cfg:
            raise ValueError('C/D data/config mismatch')
        audit = s0.read(run/'training/weight_audit.json')
        if (audit['answerable_weight'], audit['impossible_weight'], audit['validation_weight']) != (1.0, weight, 1.0):
            raise ValueError('C/D weight audit mismatch')
        actual = quality.metrics(run, 'lora')
        if actual != c['candidates'][name] or actual['train_samples'] != 539:
            raise ValueError('C/D metrics mismatch')
        prior[name] = actual
    return path, p3, base, m, spec, cfg, source, c3['baseline_lora'], prior


def reuse_worker(run, phase):
    # Reuse the exact successful training and evaluation implementation, changing
    # only the candidate weight and adapter output location in this subprocess.
    t4.WEIGHTS = {NAME: WEIGHT}
    t4.adapter = adapter
    t4.worker(run, phase)


def execute_phase(run, phase):
    # The original runner dispatches workers using its module __file__.
    # Redirect it only during this call; never modify previous files on disk.
    old_file, old_adapter = t4.__file__, t4.adapter
    try:
        t4.__file__ = str(SCRIPT)
        t4.adapter = adapter
        t4.execute_phase(run, phase)
    finally:
        t4.__file__, t4.adapter = old_file, old_adapter


def generation_metrics(run):
    g = s0.read(run/'lora/metrics.json')
    t = s0.read(run/'training/lora_train_metrics.json')
    return dict(token_f1_percent=g['token_f1_percent'],
                answer_correctness_percent=None,
                correct_refusal_rate_percent=g['abstention_rate_percent'],
                false_refusal_rate_percent=g['false_refusal_rate_percent'],
                exact_match_percent=g['exact_match_percent'],
                best_val_loss=t['best_val_loss'], training_seconds=t['training_seconds'],
                training_gpu_gib=t['peak_gpu_memory_gb'], inference_gpu_gib=g['peak_gpu_memory_gb'],
                questions_per_second=g['questions_per_second'], reference_contradiction_percent=None,
                reference_unsupported_percent=None, train_samples=t['train_samples'])


def screen(candidate, baseline):
    reasons = []
    if candidate['correct_refusal_rate_percent'] < 50:
        reasons.append('Correct-refusal rate < 50%')
    if candidate['false_refusal_rate_percent'] >= baseline['false_refusal_rate_percent']:
        reasons.append('False-refusal rate did not decrease')
    if candidate['token_f1_percent'] < baseline['token_f1_percent']:
        reasons.append('Token F1 decreased')
    return reasons


def report(root, base, prior, candidate, status, reasons):
    results = {**prior, NAME: candidate}
    selected = 'B_weight_025'
    if status == 'COMPLETE':
        chosen = quality.select({NAME: candidate}, prior['B_weight_025'])
        if chosen != 'baseline':
            selected = chosen
    s0.write(root/'comparison.json', {
        'baseline_lora':base, 'candidates':results, 'weights':WEIGHTS,
        'baseline_weight':1.0, 'screening_baseline':'B_weight_025',
        'quality_status':status, 'screen_reasons':reasons,
        'provisional_selection':selected, 'complete':status in ('COMPLETE', 'SKIPPED'),
        'human_review_required':True, 'qlora':'NOT RUN', 'hallucination_rate_percent':None,
        'notes':['A-D reused without training or evaluation.',
                 'E trained independently from original base weights.',
                 'Screening uses existing 50% correct-refusal floor and B metrics.',
                 'Missing quality metrics are null, not zero.',
                 'Validation loss is unweighted.']})
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    columns = [base, *results.values()]
    lines = ['[Stage 2 Test3 Combined Loss Weighting Comparison]',
             f'{"Metric":36} ' + ' '.join(f'{n:>12}' for n in ('B_prompt','A_0.5','B_0.25','C_0.1','D_0.05','E_0.175'))]
    for key in t2.KEYS:
        lines.append(f'{key:36} ' + ' '.join(f'{fmt(r.get(key)):>12}' for r in columns))
    lines += [f'E quality: {status}', 'Screen: '+('; '.join(reasons) if reasons else 'PASS'),
              f'Provisional selection: {selected}', 'Validation loss: unweighted. Training loss: weighted.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN', f'Results: {root}']
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, help='Completed original C/D (Test4) directory; latest by default')
    p.add_argument('--resume', type=Path, help='Resume existing E run')
    p.add_argument('--prepare-only', action='store_true', help='Verify provenance without GPU work')
    p.add_argument('--full-quality', action='store_true', help='Run original judge even if screening fails')
    p.add_argument('--worker', choices=PHASES, help=argparse.SUPPRESS)
    p.add_argument('--run', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        if args.run is None or args.run.resolve().parent.parent.parent != RUNS.resolve() or args.run.name != NAME:
            p.error('Invalid worker directory')
        reuse_worker(args.run.resolve(), args.worker)
        return
    if args.resume:
        root = args.resume.resolve()
        if root.parent != RUNS.resolve():
            p.error('Invalid resume directory')
        old = s0.read(root/'manifest.json')
        args.parent = Path(old['parent_run'])
    parent, p3, base, m, spec, cfg, source, baseline, prior = load_parent(args.parent)
    manifest = {'stage':2,'test':3,'experiment':'loss_weighting_midpoint',
                'parent_run':str(parent),'baseline':str(base),'test3_run':str(p3),
                'parent_manifest_sha256':s0.sha(parent/'manifest.json'),
                'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
                'test3_comparison_sha256':s0.sha(p3/'comparison.json'),
                'training_source':str(source),'training_sha256':s0.sha(source),
                'data':m['data'],'sources':m['sources'],'model_files':m['model_files'],
                'spec':spec,'config':cfg,'settings':m['settings'],'weights':{NAME:WEIGHT},
                'runner_sha256':source_hashes()}
    if args.resume:
        if old != manifest:
            raise ValueError('Resume blocked: provenance changed')
    else:
        root = RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False)
        s0.write(root/'manifest.json',manifest)
    run = root/'experiments'/NAME
    run.mkdir(parents=True,exist_ok=True)
    dest = run/'train.json'
    if not dest.exists():
        dest.write_bytes(source.read_bytes())
    if s0.sha(dest) != manifest['training_sha256']:
        raise ValueError('Training snapshot mismatch')
    cpath = t2.config_for(run)
    if not cpath.exists():
        s0.write(cpath,cfg)
    if s0.read(cpath) != cfg:
        raise ValueError('Config snapshot mismatch')
    print(f'[Stage 2 Test3 E] Run: {root}',flush=True)
    if args.prepare_only:
        print('Preparation: OK (no training)',flush=True)
        return
    try:
        print(f'[Experiment] 1/1 | {NAME}',flush=True)
        for phase in ('train','generate'):
            execute_phase(run,phase)
        candidate = generation_metrics(run)
        if candidate['train_samples'] != 539:
            raise ValueError('Unexpected training count')
        reasons = screen(candidate,prior['B_weight_025'])
        print(f'[Screen] F1={candidate["token_f1_percent"]}% | Correct refusal={candidate["correct_refusal_rate_percent"]}% | False refusal={candidate["false_refusal_rate_percent"]}%',flush=True)
        # Save screening results before potentially slow judge evaluation.
        s0.write(root/'screening.json',{'metrics':candidate,'reasons':reasons,'passed':not reasons})
        done = (run/'completed/quality_lora.json').is_file()
        if not reasons or args.full_quality or done:
            execute_phase(run,'quality')
            candidate = quality.metrics(run,'lora')
            status = 'COMPLETE'
        else:
            status = 'SKIPPED'
        if s0.audit() != manifest['data'] or s0.sha(source) != manifest['training_sha256']:
            raise ValueError('Dataset changed during experiment')
        report(root,baseline,prior,candidate,status,reasons)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/{SCRIPT.name} --resume "{root}"',flush=True)
        raise


if __name__ == '__main__':
    main()
