from __future__ import annotations

import argparse
import functools
import importlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_stage3_test1 as lr

s0, t2, t4, quality, previous, mid = lr.s0, lr.t2, lr.t4, lr.quality, lr.previous, lr.mid
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage3_test2'
SCRIPT = Path(__file__).resolve()
A, B = 'A_epoch_1', 'B_epoch_3'
EPOCHS = {A: 1, B: 3}
WEIGHT = 0.175
PHASES = ('train', 'generate', 'quality')


def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT/'models/finetuning/stage3_test2'/run.parent.parent.name/run.name/method


def load_parent(path=None):
    if path is None:
        candidates = [p for p in lr.RUNS.glob('run_*') if (p/'comparison.json').is_file()
                      and s0.read(p/'comparison.json').get('complete')]
        if not candidates:
            raise FileNotFoundError('Complete Stage 3 Test1 first')
        path = max(candidates, key=lambda p:p.name)
    path = Path(path).resolve()
    if path.parent != lr.RUNS.resolve():
        raise ValueError('Expected Stage 3 Test1 directory')
    m, c = s0.read(path/'manifest.json'), s0.read(path/'comparison.json')
    if (m['stage'],m['test'],c['complete'],c['provisional_selection']) != (3,1,True,'baseline'):
        raise ValueError('Expected completed LR test with 2e-4 baseline retained')
    parent, base, ancestor, source, baseline, verification = lr.load_parent(Path(m['parent_run']))
    if (m['parent_manifest_sha256'] != s0.sha(parent/'manifest.json')
            or m['parent_comparison_sha256'] != s0.sha(parent/'comparison.json')
            or m['training_sha256'] != s0.sha(source)
            or m['data'] != ancestor['data'] or m['sources'] != ancestor['sources']
            or m['model_files'] != ancestor['model_files'] or m['spec'] != ancestor['spec']
            or m['config'] != ancestor['config'] or m['settings'] != ancestor['settings']
            or m['baseline'] != str(base) or c['baseline_lora'] != baseline
            or m['learning_rates'] != lr.RATES or c['learning_rates'] != lr.RATES
            or m['settings']['epochs'] != 2 or m['settings']['learning_rate'] != 2e-4):
        raise ValueError('LR test provenance/settings mismatch')
    # Test1 is not used for training or inference artifacts: only its verified selection.
    # Existing Stage2 dependency checks remain in lr.load_parent.
    return path, parent, base, ancestor, source, baseline, verification


def install_epoch_snapshot(trainer, run):
    import torch
    snapshot = run/'training/epoch1_snapshot'
    state = {'losses': [], 'start': None, 'epoch1': None}
    original_clock = trainer.time.perf_counter
    original_eval = trainer.evaluate_loss
    original_load = trainer.load_lora_model

    def clock():
        now = original_clock()
        if state['start'] is None:
            state['start'] = now
        return now

    # The verified original trainer uses perf_counter only to start/stop the
    # training timer. Replace only its module-local reference, not global time.
    trainer.time = SimpleNamespace(perf_counter=clock)

    @functools.wraps(original_eval)
    def evaluate(*args, **kwargs):
        loss = original_eval(*args, **kwargs)
        state['losses'].append(float(loss))
        return loss

    @functools.wraps(original_load)
    def load(*args, **kwargs):
        model, tokenizer = original_load(*args, **kwargs)
        original_save = tokenizer.save_pretrained

        @functools.wraps(original_save)
        def save(*args, **kwargs):
            result = original_save(*args, **kwargs)
            # Original trainer saves the first epoch unconditionally for finite
            # validation loss. Copy its serialized checkpoint, not model state.
            if len(state['losses']) == 1 and state['epoch1'] is None:
                if state['start'] is None:
                    raise RuntimeError('Training timer not initialized')
                torch.cuda.synchronize()
                state['epoch1'] = {
                    'training_seconds': round(original_clock()-state['start'],3),
                    'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated()/1024**3,3),
                }
                snapshot.mkdir(parents=True, exist_ok=False)
                shutil.copytree(adapter(run,'lora'),snapshot/'adapter')
            return result
        tokenizer.save_pretrained = save
        return model, tokenizer

    trainer.evaluate_loss = evaluate
    trainer.load_lora_model = load
    return state


def finish_snapshot(run, state):
    t = s0.read(run/'training/lora_train_metrics.json')
    if (len(state['losses']) != 3 or state['epoch1'] is None or t['epochs'] != 3
            or len(t['history']) != 3 or t['train_samples'] != 539 or t['val_samples'] != 106
            or any(h['epoch'] != i or h['val_loss'] != round(loss,4)
                   for i,(h,loss) in enumerate(zip(t['history'],state['losses']),1))):
        raise ValueError('Shared training history/snapshot mismatch')
    best_epoch = min(range(1,4),key=lambda i:state['losses'][i-1])
    if t['best_val_loss'] != round(state['losses'][best_epoch-1],4):
        raise ValueError('Best checkpoint mismatch')
    snapshot = run/'training/epoch1_snapshot'
    if not (snapshot/'adapter/adapter_model.safetensors').is_file():
        raise ValueError('Epoch 1 checkpoint missing')
    s0.write(snapshot/'lora_train_metrics.json',dict(t,epochs=1,history=t['history'][:1],
             best_val_loss=round(state['losses'][0],4),**state['epoch1']))
    s0.write(run/'training/epoch_audit.json',{'epochs':3,'best_epoch':best_epoch,
             'validation_losses':state['losses'],'epoch1_snapshot':str(snapshot),
             'checkpoint_selection':'minimum unweighted validation loss; earliest on ties',
             'shared_trajectory':True})


def worker(run, phase):
    m = s0.read(run.parent.parent/'manifest.json')
    if (m['epochs'] != EPOCHS or m['weights'] != dict.fromkeys(EPOCHS,WEIGHT)
            or m['settings'] != dict(s0.SETTINGS,max_length=m['spec']['max_length'])):
        raise ValueError('Experiment manifest/settings mismatch')
    s0.SETTINGS = dict(m['settings'], epochs=EPOCHS[run.name])
    t4.WEIGHTS = dict.fromkeys(EPOCHS,WEIGHT)
    t4.adapter = adapter
    if phase == 'train':
        if run.name != B:
            raise ValueError('Only B executes training; A is a saved prefix checkpoint')
        trainer = importlib.import_module('train_lora')
        state = install_epoch_snapshot(trainer,run)
        t4.worker(run,phase)
        finish_snapshot(run,state)
    else:
        t4.worker(run,phase)


def execute_phase(run,phase):
    old_file, old_adapter = t4.__file__, t4.adapter
    try:
        t4.__file__, t4.adapter = str(SCRIPT), adapter
        t4.execute_phase(run,phase)
    finally:
        t4.__file__, t4.adapter = old_file, old_adapter


def materialize_a(a,b):
    previous.check_done(b,'train_lora')
    marker = a/'completed/train_lora.json'
    if marker.is_file():
        previous.check_done(a,'train_lora')
        return
    snap = b/'training/epoch1_snapshot'
    targets = [adapter(a,'lora'),a/'training']
    for p in targets:
        if p.exists():
            p.rename(p.with_name(p.name+'.incomplete-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    targets[0].parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(snap/'adapter',targets[0])
    targets[1].mkdir(parents=True,exist_ok=True)
    shutil.copy2(snap/'lora_train_metrics.json',targets[1]/'lora_train_metrics.json')
    s0.write(targets[1]/'epoch_audit.json',{'epochs':1,'best_epoch':1,'shared_trajectory':True,
             'source':str(snap),'source_marker_sha256':s0.sha(b/'completed/train_lora.json')})
    s0.write(targets[1]/'weight_audit.json',{'answerable_weight':1.0,'impossible_weight':WEIGHT,
             'validation_weight':1.0,'observed_forward_counts':{'answerable':367,'impossible':172,'validation':106},
             'source':'first epoch of verified three-epoch run'})
    files = [f for p in targets for f in p.rglob('*') if f.is_file()]
    s0.write(marker,{'files':{str(f):s0.sha(f) for f in files}})


def report(root,baseline,results,statuses,prefix,best_epoch):
    chosen = lr.select(results,baseline)
    complete = set(results)==set(EPOCHS)
    s0.write(root/'comparison.json',{'baseline_lora':baseline,'baseline_epochs':2,'epochs':EPOCHS,
             'candidates':results,'quality_status':statuses,'youden_j_percent_points':{'baseline':lr.j(baseline),**{n:lr.j(r) for n,r in results.items()}},
             'provisional_selection':chosen,'complete':complete,'best_checkpoint_epochs':{A:1,B:best_epoch},
             'prefix_history_matches_baseline':prefix,'human_review_required':True,'qlora':'NOT RUN',
             'hallucination_rate_percent':None,'notes':['One 3-epoch training trajectory; A is the saved first-epoch adapter.',
             'B selects minimum unweighted validation loss over epochs 1-3, not necessarily epoch 3.',
             'A training seconds are measured at epoch 1; B seconds are cumulative and include snapshot-copy overhead.',
             'Quality screen/selection unchanged from Stage3 Test1. No independent test evaluation.']})
    def fmt(v): return 'N/A' if v is None else f'{v:.4f}'.rstrip('0').rstrip('.')
    lines=['[Stage 3 Test2 Epoch Comparison]',f'{"Metric":36} '+ ' '.join(f'{n:>16}' for n in ['Baseline_2',*results])]
    for k in (*t2.KEYS,'youden_j_percent_points'):
        lines.append(f'{k:36} '+' '.join(f'{fmt(lr.j(r) if k=="youden_j_percent_points" else r.get(k)):>16}' for r in [baseline,*results.values()]))
    lines += [f'Best checkpoint epochs: A=1, B={best_epoch}',f'First two epoch history matches baseline: {prefix}',
              f'Quality: {statuses}',f'Provisional selection: {chosen}',f'Complete: {complete}',
              'Training times are cumulative, not additive. Validation loss: unweighted.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN',f'Results: {root}']
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description='Stage 3 Test2: shared epoch trajectory')
    p.add_argument('--parent',type=Path,help='Completed Stage3 Test1 directory')
    p.add_argument('--resume',type=Path,help='Resume an existing Test2 run')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--full-quality',action='store_true')
    p.add_argument('--worker',choices=PHASES,help=argparse.SUPPRESS)
    p.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        if args.run is None or args.run.resolve().parent.parent.parent!=RUNS.resolve() or args.run.name not in EPOCHS:
            p.error('Invalid worker directory')
        worker(args.run.resolve(),args.worker); return
    if args.resume:
        root=args.resume.resolve()
        if root.parent!=RUNS.resolve(): p.error('Invalid resume directory')
        old=s0.read(root/'manifest.json'); args.parent=Path(old['parent_run'])
    parent,e,base,m,source,baseline,verification=load_parent(args.parent)
    manifest={'stage':3,'test':2,'experiment':'epoch','parent_run':str(parent),'stage2_run':str(e),'baseline':str(base),
              'parent_manifest_sha256':s0.sha(parent/'manifest.json'),'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
              'stage2_manifest_sha256':s0.sha(e/'manifest.json'),'stage2_comparison_sha256':s0.sha(e/'comparison.json'),
              'parent_runner_verification':verification,'training_source':str(source),'training_sha256':s0.sha(source),
              'data':m['data'],'sources':m['sources'],'model_files':m['model_files'],'spec':m['spec'],'config':m['config'],
              'settings':m['settings'],'epochs':EPOCHS,'weights':dict.fromkeys(EPOCHS,WEIGHT),
              'runner_sha256':{**lr.source_hashes(),SCRIPT.name:s0.sha(SCRIPT)}}
    if args.resume:
        if old!=manifest: raise ValueError('Resume blocked: provenance changed')
    else:
        root=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False); s0.write(root/'manifest.json',manifest)
    print(f'[Stage 3 Test2] Run: {root}',flush=True)
    if args.prepare_only:
        print('Preparation: OK (no training)',flush=True); return
    try:
        for name in EPOCHS:
            run=root/'experiments'/name; run.mkdir(parents=True,exist_ok=True)
            dest=run/'train.json'
            if not dest.exists(): dest.write_bytes(source.read_bytes())
            if s0.sha(dest)!=manifest['training_sha256']: raise ValueError('Training snapshot mismatch')
            cp=t2.config_for(run)
            if not cp.exists(): s0.write(cp,m['config'])
            if s0.read(cp)!=m['config']: raise ValueError('Config snapshot mismatch')
        a,b=(root/'experiments'/n for n in EPOCHS)
        print('[Experiment] Shared training | Epoch 1 and 3',flush=True)
        execute_phase(b,'train'); materialize_a(a,b)
        bt=s0.read(b/'training/lora_train_metrics.json')
        et=s0.read(e/'experiments'/mid.NAME/'training/lora_train_metrics.json')
        prefix=bt['history'][:2]==et['history']
        best_epoch=s0.read(b/'training/epoch_audit.json')['best_epoch']
        results,statuses={},{}
        for i,name in enumerate(EPOCHS,1):
            run=root/'experiments'/name
            print(f'[Experiment] {i}/2 | {name}',flush=True)
            execute_phase(run,'generate')
            candidate=mid.generation_metrics(run)
            reasons=lr.screen(candidate,baseline)
            s0.write(run/'screening.json',{'metrics':candidate,'youden_j_percent_points':lr.j(candidate),'reasons':reasons,'passed':not reasons})
            print(f'[Screen] F1={candidate["token_f1_percent"]}% | Correct refusal={candidate["correct_refusal_rate_percent"]}% | False refusal={candidate["false_refusal_rate_percent"]}% | J={lr.j(candidate)}%p',flush=True)
            if not reasons or args.full_quality or (run/'completed/quality_lora.json').is_file():
                execute_phase(run,'quality'); candidate=quality.metrics(run,'lora'); status='COMPLETE'
            else: status='SKIPPED: '+'; '.join(reasons)
            if s0.audit()!=manifest['data'] or s0.sha(source)!=manifest['training_sha256']:
                raise ValueError('Dataset changed during run')
            results[name],statuses[name]=candidate,status
            report(root,baseline,results,statuses,prefix,best_epoch)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/{SCRIPT.name} --resume "{root}"',flush=True)
        raise


if __name__=='__main__':
    main()
