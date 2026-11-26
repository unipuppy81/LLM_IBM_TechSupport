"""Stage 3 Test1: controlled learning-rate search, LoRA only.
Save next to run_stage2_test3_midpoint.py. Reuse E (0.175) as baseline.
No test data, no modifications to previous scripts or artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import run_stage2_test3_midpoint as mid

s0, t2, t4, quality, previous = mid.s0, mid.t2, mid.t4, mid.quality, mid.previous
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage3_test1'
SCRIPT = Path(__file__).resolve()
RATES = {'A_lr_1e4': 1e-4, 'B_lr_5e4': 5e-4}
WEIGHT = 0.175
PHASES = ('train', 'generate', 'quality')
# The completed E run and metrics supplied by the user. A historical launcher
# mismatch may be tolerated only for reuse of this already-completed baseline,
# never for an underlying trainer, evaluator, or earlier-stage runner.
KNOWN_E_RUN = 'run_20261126_153404_972504'
KNOWN_E_METRICS = dict(token_f1_percent=7.03, answer_correctness_percent=16.88,
                       correct_refusal_rate_percent=75.86, false_refusal_rate_percent=63.64,
                       exact_match_percent=0, best_val_loss=1.7973, train_samples=539,
                       training_seconds=535.475, training_gpu_gib=4.87,
                       inference_gpu_gib=3.096, questions_per_second=1.135,
                       reference_contradiction_percent=0, reference_unsupported_percent=14.29)


def parent_runner_verification(recorded):
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError('Invalid Stage 2 runner hash manifest')
    current = mid.source_hashes()
    launcher = Path(mid.__file__).resolve()
    name = launcher.name
    for key, digest in recorded.items():
        if key == name:
            continue
        # Compare the files actually named by the historical manifest, not
        # dictionary shape (which may differ across launcher versions).
        if Path(key).name != key or not key.endswith('.py'):
            raise ValueError(f'Unexpected Stage 2 runner entry: {key}')
        f = SCRIPT.parent / key
        if not f.is_file() or s0.sha(f) != digest:
            actual = s0.sha(f) if f.is_file() else 'MISSING'
            raise ValueError(f'Stage 2 dependency changed: {key} | expected={digest} | actual={actual}')
    expected = recorded.get(name)
    actual = s0.sha(launcher)
    raw = launcher.read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    variants = {hashlib.sha256(b).hexdigest() for b in (lf, lf.replace(b'\n', b'\r\n'))}
    status = ('EXACT' if expected == actual else 'LINE_ENDINGS_ONLY' if expected in variants
              else 'HISTORICAL_LAUNCHER_UNVERIFIED')
    return {'status':status,'file':name,'recorded_sha256':expected,'current_sha256':actual,
            'manifest_only_keys':sorted(set(recorded)-set(current)),
            'current_only_keys':sorted(set(current)-set(recorded))}



def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT/'models/finetuning/stage3_test1'/run.parent.parent.name/run.name/method


def source_hashes():
    return {**mid.source_hashes(), SCRIPT.name: s0.sha(SCRIPT)}


def load_parent(path=None):
    if path is None:
        candidates = [p for p in mid.RUNS.glob('run_*') if (p/'comparison.json').is_file()
                      and s0.read(p/'comparison.json').get('complete')
                      and s0.read(p/'comparison.json').get('provisional_selection') == mid.NAME]
        if not candidates:
            raise FileNotFoundError('Complete Stage 2 Test3 E with E selected first')
        path = max(candidates, key=lambda p: p.name)
    path = Path(path).resolve()
    if path.parent != mid.RUNS.resolve():
        raise ValueError('Expected Stage 2 midpoint run directory')
    m, c = s0.read(path/'manifest.json'), s0.read(path/'comparison.json')
    if (m['stage'], m['test'], c['complete'], c['quality_status'], c['provisional_selection']) != (2,3,True,'COMPLETE',mid.NAME):
        raise ValueError('Expected complete Stage 2 E with quality evaluation')
    if m['weights'] != {mid.NAME: WEIGHT} or c['weights'] != mid.WEIGHTS:
        raise ValueError('Unexpected Stage 2 weights')
    verification = parent_runner_verification(m['runner_sha256'])
    p4, p3, base, ancestor, spec, cfg, source, old_base, prior = mid.load_parent(Path(m['parent_run']))
    if (m['parent_manifest_sha256'] != s0.sha(p4/'manifest.json')
            or m['parent_comparison_sha256'] != s0.sha(p4/'comparison.json')
            or m['test3_comparison_sha256'] != s0.sha(p3/'comparison.json')
            or m['baseline'] != str(base) or m['test3_run'] != str(p3)
            or m['training_source'] != str(source) or m['training_sha256'] != s0.sha(source)
            or m['data'] != s0.audit() or m['sources'] != ancestor['sources']
            or m['model_files'] != ancestor['model_files'] or m['spec'] != spec or m['config'] != cfg
            or m['settings'] != dict(s0.SETTINGS,max_length=spec['max_length'])
            or c['baseline_lora'] != old_base
            or any(c['candidates'].get(n) != v for n,v in prior.items())):
        raise ValueError('Stage 2 ancestry mismatch')
    run = path/'experiments'/mid.NAME
    for phase in PHASES:
        previous.check_done(run, phase+'_lora')
    if s0.sha(run/'train.json') != m['training_sha256'] or s0.read(t2.config_for(run)) != cfg:
        raise ValueError('Stage 2 E training/config mismatch')
    audit = s0.read(run/'training/weight_audit.json')
    if (audit['answerable_weight'],audit['impossible_weight'],audit['validation_weight']) != (1.0,WEIGHT,1.0):
        raise ValueError('Stage 2 E weight mismatch')
    metrics = quality.metrics(run,'lora')
    if metrics != c['candidates'][mid.NAME] or metrics['train_samples'] != 539:
        raise ValueError('Stage 2 E metrics mismatch')
    if verification['status'] == 'HISTORICAL_LAUNCHER_UNVERIFIED':
        if path.name != KNOWN_E_RUN or any(abs(metrics.get(k, float('inf'))-v)>0.00005
                                           for k,v in KNOWN_E_METRICS.items()):
            raise ValueError(f'Unverified Stage 2 launcher for unknown baseline: {verification}')
        # All earlier-source, config, model, data, artifact and metric checks
        # above must pass before this baseline-only compatibility exception.
        print('[Provenance] Historical midpoint launcher SHA differs; verified completed E baseline reused. '
              'Historical launcher source not verified.',flush=True)
    elif verification['status'] == 'LINE_ENDINGS_ONLY':
        print('[Provenance] Stage 2 midpoint launcher differs only in line endings.',flush=True)
    return path, base, m, run/'train.json', metrics, verification


def worker(run, phase):
    m = s0.read(run.parent.parent/'manifest.json')
    if m['learning_rates'] != RATES or m['weights'] != dict.fromkeys(RATES,WEIGHT):
        raise ValueError('Learning-rate experiment manifest mismatch')
    expected = dict(s0.SETTINGS,max_length=m['spec']['max_length'])
    if m['settings'] != expected:
        raise ValueError('Base settings changed')
    # t4.worker preserves SETTINGS other than setting max_length from spec.
    # Each subprocess starts with the original trainer and base-model weights.
    s0.SETTINGS = dict(expected, learning_rate=RATES[run.name])
    t4.WEIGHTS = dict.fromkeys(RATES,WEIGHT)
    t4.adapter = adapter
    t4.worker(run,phase)
    if phase == 'train':
        s0.write(run/'training/lr_audit.json', {'settings':s0.SETTINGS, 'answerable_weight':1.0,
                 'impossible_weight':WEIGHT,'validation_weight':1.0})


def execute_phase(run,phase):
    old_file, old_adapter = t4.__file__, t4.adapter
    try:
        t4.__file__, t4.adapter = str(SCRIPT), adapter
        t4.execute_phase(run,phase)
    finally:
        t4.__file__, t4.adapter = old_file, old_adapter


def j(m):
    return round(m['correct_refusal_rate_percent']-m['false_refusal_rate_percent'], 4)


def screen(m, baseline):
    reasons = []
    if m['correct_refusal_rate_percent'] < 50:
        reasons.append('Correct refusal < 50%')
    if m['false_refusal_rate_percent'] >= baseline['false_refusal_rate_percent']:
        reasons.append('False refusal did not decrease')
    if m['token_f1_percent'] < baseline['token_f1_percent']:
        reasons.append('Token F1 decreased')
    if j(m) < j(baseline):
        reasons.append("Youden's J decreased")
    return reasons


def select(results, baseline):
    eligible = {n:r for n,r in results.items() if not screen(r,baseline)
                and r['answer_correctness_percent'] is not None
                and r['answer_correctness_percent'] >= baseline['answer_correctness_percent']
                and (j(r)>j(baseline) or r['answer_correctness_percent']>baseline['answer_correctness_percent']
                     or r['token_f1_percent']>baseline['token_f1_percent'])}
    if not eligible:
        return 'baseline'
    return max(eligible,key=lambda n:(j(eligible[n]),eligible[n]['answer_correctness_percent'],eligible[n]['token_f1_percent']))


def report(root,baseline,results,statuses):
    chosen=select(results,baseline)
    complete=set(results)==set(RATES)
    s0.write(root/'comparison.json', {'baseline_lora':baseline, 'baseline_learning_rate':2e-4,
             'learning_rates':RATES,'candidates':results,'quality_status':statuses,
             'youden_j_percent_points':{'baseline':j(baseline),**{n:j(r) for n,r in results.items()}},
             'provisional_selection':chosen,'complete':complete,'human_review_required':True,
             'qlora':'NOT RUN','hallucination_rate_percent':None,
             'notes':['Same E data, prompt, loss weighting, seed and training settings except LR.',
                      'Quality skipped for failed generation screens unless explicitly overridden.',
                      'Selection: 50% correct-refusal floor, nondecreasing F1/J/correctness, lower false refusal; rank by J first.',
                      'Validation loss is unweighted.']})
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    cols=[baseline,*results.values()]
    lines=['[Stage 3 Test1 Learning Rate Comparison]',f'{"Metric":36} '+ ' '.join(f'{n:>16}' for n in ['E_lr_2e4',*results])]
    for k in (*t2.KEYS,'youden_j_percent_points'):
        vals=[j(r) if k=='youden_j_percent_points' else r.get(k) for r in cols]
        lines.append(f'{k:36} '+' '.join(f'{fmt(v):>16}' for v in vals))
    lines += [f'Quality: {statuses}',f'Provisional selection: {chosen}',f'Complete: {complete}',
              'Validation loss: unweighted. Training loss: weighted.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN', f'Results: {root}']
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,help='Selected Stage 2 E run, latest by default')
    p.add_argument('--resume',type=Path,help='Resume existing Stage 3 Test1 run')
    p.add_argument('--through',choices=('A','B'),default='B',help='Stop after A, or run both (default)')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--full-quality',action='store_true',help='Evaluate quality even for failed screens')
    p.add_argument('--worker',choices=PHASES,help=argparse.SUPPRESS)
    p.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        if a.run is None or a.run.resolve().parent.parent.parent != RUNS.resolve() or a.run.name not in RATES:
            p.error('Invalid worker directory')
        worker(a.run.resolve(),a.worker)
        return
    if a.resume:
        root=a.resume.resolve()
        if root.parent != RUNS.resolve(): p.error('Invalid resume directory')
        old=s0.read(root/'manifest.json')
        a.parent=Path(old['parent_run'])
    parent,base,m,source,baseline,verification=load_parent(a.parent)
    manifest={'stage':3,'test':1,'experiment':'learning_rate','parent_run':str(parent),
              'baseline':str(base),'parent_manifest_sha256':s0.sha(parent/'manifest.json'),
              'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
              'parent_runner_verification':verification,
              'training_source':str(source),'training_sha256':s0.sha(source),
              'data':m['data'],'sources':m['sources'],'model_files':m['model_files'],
              'spec':m['spec'],'config':m['config'],'settings':m['settings'],
              'learning_rates':RATES,'weights':dict.fromkeys(RATES,WEIGHT),'runner_sha256':source_hashes()}
    if a.resume:
        if old != manifest: raise ValueError('Resume blocked: provenance changed')
    else:
        root=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False)
        s0.write(root/'manifest.json',manifest)
    print(f'[Stage 3 Test1] Run: {root}',flush=True)
    if a.prepare_only:
        print('Preparation: OK (no training)',flush=True)
        return
    results,statuses={},{}
    try:
        for i,name in enumerate(RATES,1):
            run=root/'experiments'/name
            # Do not discard a previously completed B on a restricted resume.
            if i==2 and a.through=='A' and not (run/'completed/generate_lora.json').is_file(): break
            run.mkdir(parents=True,exist_ok=True)
            dest=run/'train.json'
            if not dest.exists(): dest.write_bytes(source.read_bytes())
            if s0.sha(dest)!=manifest['training_sha256']: raise ValueError('Training snapshot mismatch')
            cp=t2.config_for(run)
            if not cp.exists(): s0.write(cp,m['config'])
            if s0.read(cp)!=m['config']: raise ValueError('Config snapshot mismatch')
            print(f'[Experiment] {i}/{len(RATES)} | {name}',flush=True)
            for phase in ('train','generate'): execute_phase(run,phase)
            candidate=mid.generation_metrics(run)
            audit=s0.read(run/'training/lr_audit.json')
            if candidate['train_samples']!=539 or audit['settings']!=dict(m['settings'],learning_rate=RATES[name]):
                raise ValueError('Training count or LR audit mismatch')
            reasons=screen(candidate,baseline)
            s0.write(run/'screening.json',{'metrics':candidate,'youden_j_percent_points':j(candidate),'reasons':reasons,'passed':not reasons})
            print(f'[Screen] F1={candidate["token_f1_percent"]}% | Correct refusal={candidate["correct_refusal_rate_percent"]}% | False refusal={candidate["false_refusal_rate_percent"]}% | J={j(candidate)}%p',flush=True)
            if not reasons or a.full_quality or (run/'completed/quality_lora.json').is_file():
                execute_phase(run,'quality')
                candidate=quality.metrics(run,'lora')
                status='COMPLETE'
            else:
                status='SKIPPED: '+'; '.join(reasons)
            if s0.audit()!=manifest['data'] or s0.sha(source)!=manifest['training_sha256']:
                raise ValueError('Dataset changed during run')
            results[name],statuses[name]=candidate,status
            report(root,baseline,results,statuses)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/{SCRIPT.name} --resume "{root}"',flush=True)
        raise


if __name__=='__main__':
    main()
