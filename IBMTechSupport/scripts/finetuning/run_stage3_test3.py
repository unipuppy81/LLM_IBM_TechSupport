from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import run_stage3_test1 as lr
import run_stage3_test2 as epoch

mid = lr.mid

s0, t2, t4, quality, previous = mid.s0, mid.t2, mid.t4, mid.quality, mid.previous
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage3_test3'
SCRIPT = Path(__file__).resolve()
ACCUM = {'A_accum_4': 4, 'B_accum_16': 16}
WEIGHT = 0.175
PHASES = ('train', 'generate', 'quality')
def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT/'models/finetuning/stage3_test3'/run.parent.parent.name/run.name/method


def source_hashes():
    return {**lr.source_hashes(), Path(epoch.__file__).name: s0.sha(Path(epoch.__file__)), SCRIPT.name: s0.sha(SCRIPT)}


def load_parent(path=None):
    if path is None:
        candidates = [p for p in epoch.RUNS.glob('run_*') if (p/'comparison.json').is_file()
                      and s0.read(p/'comparison.json').get('complete')]
        if not candidates:
            raise FileNotFoundError('Complete Stage 3 Test2 first')
        path = max(candidates, key=lambda p: p.name)
    path = Path(path).resolve()
    if path.parent != epoch.RUNS.resolve():
        raise ValueError('Expected Stage 3 Test2 run directory')
    m, c = s0.read(path/'manifest.json'), s0.read(path/'comparison.json')
    if (m['stage'],m['test'],c['complete'],c['provisional_selection']) != (3,2,True,'baseline'):
        raise ValueError('Expected completed epoch test with baseline retained')
    lr_run,e,base,ancestor,source,baseline,verification = epoch.load_parent(Path(m['parent_run']))
    if (m['parent_manifest_sha256'] != s0.sha(lr_run/'manifest.json')
            or m['parent_comparison_sha256'] != s0.sha(lr_run/'comparison.json')
            or m['stage2_manifest_sha256'] != s0.sha(e/'manifest.json')
            or m['stage2_comparison_sha256'] != s0.sha(e/'comparison.json')
            or m['stage2_run'] != str(e) or m['baseline'] != str(base)
            or m['training_source'] != str(source) or m['training_sha256'] != s0.sha(source)
            or any(m[k] != ancestor[k] for k in ('data','sources','model_files','spec','config','settings'))
            or m['epochs'] != epoch.EPOCHS or c['epochs'] != epoch.EPOCHS
            or m['weights'] != dict.fromkeys(epoch.EPOCHS,WEIGHT)
            or c['baseline_lora'] != baseline or c['baseline_epochs'] != 2
            or m['settings']['grad_accum'] != 8 or m['settings']['epochs'] != 2
            or m['settings']['learning_rate'] != 2e-4):
        raise ValueError('Epoch test ancestry/settings mismatch')
    # Stage3 Test1/2 only provide selection records. The reused training,
    # config, adapter and evaluation artifacts are the verified Stage2 E run.
    return path,base,ancestor,source,baseline,verification


def worker(run, phase):
    m = s0.read(run.parent.parent/'manifest.json')
    if m['accumulation_steps'] != ACCUM or m['weights'] != dict.fromkeys(ACCUM,WEIGHT):
        raise ValueError('Gradient-accumulation experiment manifest mismatch')
    expected = dict(s0.SETTINGS,max_length=m['spec']['max_length'])
    if m['settings'] != expected:
        raise ValueError('Base settings changed')
    # t4.worker preserves SETTINGS other than setting max_length from spec.
    # Each subprocess starts with the original trainer and base-model weights.
    s0.SETTINGS = dict(expected, grad_accum=ACCUM[run.name])
    t4.WEIGHTS = dict.fromkeys(ACCUM,WEIGHT)
    t4.adapter = adapter
    t4.worker(run,phase)
    if phase == 'train':
        s0.write(run/'training/accum_audit.json', {'settings':s0.SETTINGS, 'answerable_weight':1.0,
                 'impossible_weight':WEIGHT,'validation_weight':1.0})


def execute_phase(run,phase):
    old_file, old_adapter = t4.__file__, t4.adapter
    try:
        t4.__file__, t4.adapter = str(SCRIPT), adapter
        t4.execute_phase(run,phase)
    finally:
        t4.__file__, t4.adapter = old_file, old_adapter


j, screen, select = lr.j, lr.screen, lr.select


def report(root,baseline,results,statuses):
    chosen=select(results,baseline)
    complete=set(results)==set(ACCUM)
    s0.write(root/'comparison.json', {'baseline_lora':baseline, 'baseline_grad_accum':8,
             'accumulation_steps':ACCUM,'candidates':results,'quality_status':statuses,
             'youden_j_percent_points':{'baseline':j(baseline),**{n:j(r) for n,r in results.items()}},
             'planned_update_windows_per_epoch':{'baseline':68,**{n:(539+v-1)//v for n,v in ACCUM.items()}},
             'provisional_selection':chosen,'complete':complete,'human_review_required':True,
             'qlora':'NOT RUN','hallucination_rate_percent':None,
             'notes':['Same E data, prompt, loss weighting, seed and training settings except gradient accumulation.',
                      'Quality skipped for failed generation screens unless explicitly overridden.',
                      'Selection: 50% correct-refusal floor, nondecreasing F1/J/correctness, lower false refusal; rank by J first.',
                      'Validation loss is unweighted.', 'Update windows change along with effective batch size; AMP can skip actual updates.']})
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    cols=[baseline,*results.values()]
    lines=['[Stage 3 Test3 Gradient Accumulation Comparison]',f'{"Metric":36} '+ ' '.join(f'{n:>16}' for n in ['Baseline_8',*results])]
    for k in (*t2.KEYS,'youden_j_percent_points'):
        vals=[j(r) if k=='youden_j_percent_points' else r.get(k) for r in cols]
        lines.append(f'{k:36} '+' '.join(f'{fmt(v):>16}' for v in vals))
    lines += ['Update windows per epoch: Baseline=68, A=135, B=34 (AMP may skip updates)', f'Quality: {statuses}',f'Provisional selection: {chosen}',f'Complete: {complete}',
              'Validation loss: unweighted. Training loss: weighted.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN', f'Results: {root}']
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description="Stage 3 Test3: Gradient Accumulation")
    p.add_argument('--parent',type=Path,help='Completed Stage 3 Test2 run, latest by default')
    p.add_argument('--resume',type=Path,help='Resume existing Stage 3 Test3 run')
    p.add_argument('--through',choices=('A','B'),default='B',help='Stop after A, or run both (default)')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--full-quality',action='store_true',help='Evaluate quality even for failed screens')
    p.add_argument('--worker',choices=PHASES,help=argparse.SUPPRESS)
    p.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        if a.run is None or a.run.resolve().parent.parent.parent != RUNS.resolve() or a.run.name not in ACCUM:
            p.error('Invalid worker directory')
        worker(a.run.resolve(),a.worker)
        return
    if a.resume:
        root=a.resume.resolve()
        if root.parent != RUNS.resolve(): p.error('Invalid resume directory')
        old=s0.read(root/'manifest.json')
        a.parent=Path(old['parent_run'])
    parent,base,m,source,baseline,verification=load_parent(a.parent)
    manifest={'stage':3,'test':3,'experiment':'gradient_accumulation','parent_run':str(parent),
              'baseline':str(base),'parent_manifest_sha256':s0.sha(parent/'manifest.json'),
              'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
              'parent_runner_verification':verification,
              'training_source':str(source),'training_sha256':s0.sha(source),
              'data':m['data'],'sources':m['sources'],'model_files':m['model_files'],
              'spec':m['spec'],'config':m['config'],'settings':m['settings'],
              'accumulation_steps':ACCUM,'weights':dict.fromkeys(ACCUM,WEIGHT),'runner_sha256':source_hashes()}
    if a.resume:
        if old != manifest: raise ValueError('Resume blocked: provenance changed')
    else:
        root=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False)
        s0.write(root/'manifest.json',manifest)
    print(f'[Stage 3 Test3] Run: {root}',flush=True)
    if a.prepare_only:
        print('Preparation: OK (no training)',flush=True)
        return
    results,statuses={},{}
    try:
        for i,name in enumerate(ACCUM,1):
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
            print(f'[Experiment] {i}/{len(ACCUM)} | {name}',flush=True)
            for phase in ('train','generate'): execute_phase(run,phase)
            candidate=mid.generation_metrics(run)
            audit=s0.read(run/'training/accum_audit.json')
            training = s0.read(run/'training/lora_train_metrics.json')
            expected_windows = (539 + ACCUM[name] - 1) // ACCUM[name]
            if (candidate['train_samples']!=539 or audit['settings']!=dict(m['settings'],grad_accum=ACCUM[name])
                    or training['grad_accum'] != ACCUM[name] or training['epochs'] != 2
                    or training['learning_rate'] != 2e-4 or len(training['history']) != 2
                    or any(h['optimizer_steps'] != expected_windows for h in training['history'])):
                raise ValueError('Training count or Gradient accumulation audit mismatch')
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
