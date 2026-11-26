from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import re
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import run_stage0 as s0
import run_stage1_quality as quality

ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage3_test4_prompt'
E_ROOT = ROOT / 'evaluate_results/106/finetuning/stage2_test3_midpoint/run_20261126_153404_972504'
E_NAME = 'E_weight_0175'
# Edit these strings to try different inference prompts. No training is performed.
PROMPTS = {
    'A_short_original': (
        'You are an IBM technical support assistant. '
        'Answer technical questions using only your existing knowledge. '
        'If you cannot provide a reliable answer, respond with exactly INSUFFICIENT_INFORMATION. '
        'Otherwise, provide a concise technical answer in English.'
    ),
    'B_concise_answerability': (
        'You are an IBM technical support assistant.\n'
        'Identify the technical information requested and answer it directly in English using your existing knowledge.\n'
        'Respect explicitly stated products, versions, error codes and environments. '
        'Do not assume missing details. If a missing detail is not essential, answer normally. '
        'If the solution depends on an unspecified condition, give a reliable conditional answer when possible.\n'
        'Do not invent technical facts, commands or procedures. '
        'Only if you cannot reliably answer the essential request, output exactly INSUFFICIENT_INFORMATION.\n'
        'Output only the answer or the exact refusal string.'
    ),
}
KEYS = ('token_f1_percent', 'answer_correctness_percent', 'correct_refusal_rate_percent',
        'false_refusal_rate_percent', 'youden_j_percent_points', 'exact_match_percent',
        'generation_seconds', 'questions_per_second', 'inference_gpu_gib',
        'reference_contradiction_percent', 'reference_unsupported_percent')


def j(row):
    return round(row['correct_refusal_rate_percent'] - row['false_refusal_rate_percent'], 4)


def screen(row, baseline):
    reasons = []
    if row['correct_refusal_rate_percent'] < 50:
        reasons.append('Correct refusal < 50%')
    if row['false_refusal_rate_percent'] >= baseline['false_refusal_rate_percent']:
        reasons.append('False refusal did not decrease')
    if row['token_f1_percent'] < baseline['token_f1_percent']:
        reasons.append('Token F1 decreased')
    if j(row) < j(baseline):
        reasons.append("Youden's J decreased")
    return reasons


def rank_key(row):
    return (j(row), row['token_f1_percent'], -row['false_refusal_rate_percent'])


def verify_sha(path, expected, allow_line_endings=False):
    actual = s0.sha(path)
    if actual != expected:
        if allow_line_endings:
            lf = path.read_bytes().replace(b'\r\n', b'\n')
            variants = {hashlib.sha256(b).hexdigest() for b in (lf, lf.replace(b'\n', b'\r\n'))}
            if expected in variants:
                return actual
        raise ValueError(f'Artifact changed: {path} | expected={expected} | actual={actual}')
    return actual


def load_baseline(parent):
    parent = parent.resolve()
    if parent.parent != E_ROOT.parent.resolve():
        raise ValueError('Expected a Stage 2 midpoint run directory')
    m, c = s0.read(parent/'manifest.json'), s0.read(parent/'comparison.json')
    if (m['stage'], m['test'], c['complete'], c['quality_status'], c['provisional_selection']) != (2, 3, True, 'COMPLETE', E_NAME):
        raise ValueError('Expected completed and selected Stage 2 E run')
    if m['weights'] != {E_NAME: 0.175} or m['settings'] != dict(s0.SETTINGS, max_length=2048):
        raise ValueError('Unexpected E training settings')
    cfg = m['config']
    spec = m['spec']
    if (spec['question_tokens'], spec['answer_tokens'], spec['max_new_tokens'], spec['max_length']) != (192,160,160,2048):
        raise ValueError('Unexpected E token limits')
    if (ROOT/cfg['dataset']['path']).resolve() != s0.VAL.resolve() or cfg['dataset']['expected_count'] != 106:
        raise ValueError('Only val_106.json may be evaluated')
    if (ROOT/cfg['model']['path']).resolve() != s0.BASE.resolve():
        raise ValueError('Unexpected base model')
    inf = cfg['inference']
    if (inf['batch_size'], inf['max_new_tokens'], inf['do_sample'], inf['num_beams']) != (1,160,False,1):
        raise ValueError('Expected original greedy inference configuration')
    if cfg['prompt']['user_template'] != '{question}' or cfg['prompt']['system'] != spec['prompt']:
        raise ValueError('Unexpected original prompt')
    run = parent/'experiments'/E_NAME
    adapter = ROOT/'models/finetuning/stage2_test3_midpoint'/parent.name/E_NAME/'lora'
    # Verify completed artifacts, not irrelevant historical training launchers.
    marked = {}
    for phase in ('train', 'generate', 'quality'):
        marker = s0.read(run/'completed'/f'{phase}_lora.json')['files']
        if not marker:
            raise ValueError(f'Empty completion marker: {phase}')
        for filename, digest in marker.items():
            path = Path(filename).resolve()
            if not (path.is_relative_to(run.resolve()) or path.is_relative_to(adapter.resolve())):
                raise ValueError(f'Unexpected completion marker path: {path}')
            verify_sha(path,digest)
            marked[path] = digest
    required = [adapter/'adapter_model.safetensors', adapter/'adapter_config.json',
                run/'training/lora_train_metrics.json', run/'training/weight_audit.json',
                run/'lora/predictions.jsonl', run/'lora/metrics.json',
                run/'lora/generation_manifest.json', run/'lora/quality_evaluation/metrics.json',
                run/'lora/quality_evaluation/cases.jsonl']
    if not all(p.resolve() in marked for p in required):
        raise ValueError('Missing protected baseline artifacts')
    if s0.sha(run/'train.json') != m['training_sha256'] or s0.read(run/'prompt_config.yaml') != cfg:
        raise ValueError('Baseline data or prompt snapshot changed')
    audit = s0.read(run/'training/weight_audit.json')
    if (audit['answerable_weight'], audit['impossible_weight'], audit['validation_weight']) != (1.0,0.175,1.0):
        raise ValueError('Baseline weight audit mismatch')
    records = s0.validate(s0.VAL)
    if (len(records),sum(r['is_impossible'] for r in records)) != (106,29):
        raise ValueError('Expected 77 answerable / 29 impossible questions')
    verify_sha(s0.VAL,m['data']['val_sha256'])
    # The old evaluators and shared judge implementation must be unchanged;
    # tolerate only CRLF/LF differences. Trainer/launcher docstrings are irrelevant here.
    sources = {}
    for name in ('scripts/load_base_model.py','scripts/evaluate/evaluate_base.py',
                 'scripts/evaluate/evaluate_lora.py','scripts/evaluate/evaluate_stage2_quality.py',
                 'scripts/evaluate/common.py','scripts/document_utils.py'):
        sources[name] = verify_sha(ROOT/name,m['sources'][name],True)
    for module in (s0,quality):
        p=Path(module.__file__).resolve()
        sources[p.name]=verify_sha(p,m['runner_sha256'][p.name],True)
    model_hashes = m['model_files']
    if set(model_hashes) != {p.name for p in s0.model_files()}:
        raise ValueError('Base model file set changed')
    for name,digest in model_hashes.items():
        verify_sha(s0.BASE/name,digest)
    base_metrics = quality.metrics(run,'lora')
    if base_metrics != c['candidates'][E_NAME] or base_metrics['train_samples'] != 539:
        raise ValueError('E baseline metrics mismatch')
    gm = s0.read(run/'lora/generation_manifest.json')
    fingerprint = {name:s0.sha(adapter/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    if (gm['adapter_sha256'] != fingerprint or gm['dataset_sha256'] != s0.sha(s0.VAL)
            or gm['predictions_sha256'] != s0.sha(run/'lora/predictions.jsonl')
            or gm['base_config_sha256'] != s0.sha(run/'prompt_config.yaml')
            or gm['prompt'] != cfg['prompt'] or gm['inference'] != inf):
        raise ValueError('E generation provenance mismatch')
    base_metrics.update(generation_seconds=s0.read(run/'lora/metrics.json')['generation_seconds'])
    base_metrics['youden_j_percent_points'] = j(base_metrics)
    provenance = {'parent_run':str(parent), 'baseline':str(run), 'adapter':str(adapter),
                  'parent_manifest_sha256':s0.sha(parent/'manifest.json'),
                  'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
                  'dataset_sha256':s0.sha(s0.VAL),'training_sha256':m['training_sha256'],
                  'adapter_sha256':fingerprint,'model_files':model_hashes,'sources':sources,
                  'config':cfg,'settings':m['settings'],'loss_weight':0.175}
    return records, adapter, cfg, base_metrics, provenance


class ProgressLog:
    def __init__(self, file, terminal):
        self.file,self.terminal,self.pending = file,terminal,''
    def write(self, value):
        self.file.write(value)
        self.file.flush()
        self.pending += value
        while '\n' in self.pending:
            line,self.pending = self.pending.split('\n',1)
            g = re.fullmatch(r'Generated: (\d+/\d+)',line.strip())
            q = re.fullmatch(r'\[lora quality\] (\d+/\d+)',line.strip())
            if g: print('[Evaluation] Question '+g[1],file=self.terminal,flush=True)
            if q: print('[Quality] Question '+q[1],file=self.terminal,flush=True)
        return len(value)
    def flush(self):
        self.file.flush()
    @property
    def encoding(self):
        return self.file.encoding
    def fileno(self):
        return self.file.fileno()
    def isatty(self):
        return False


def config_for(cfg, prompt):
    new = copy.deepcopy(cfg)
    new['prompt']['system'] = prompt
    return new


def generate_pending(root, pending, cfg, records, adapter, fingerprint):
    import torch
    from peft import PeftModel
    from load_base_model import load_model
    from evaluate.evaluate_base import make_prediction_rows
    from evaluate.common import score_predictions
    terminal = sys.stdout
    with (root/'generation.log').open('a',encoding='utf-8') as log:
        with redirect_stdout(ProgressLog(log,terminal)), redirect_stderr(log):
            base, tokenizer, generation_config = load_model(cfg)
            model = PeftModel.from_pretrained(base,str(adapter),is_trainable=False,local_files_only=True)
            model.eval()
            try:
                for i,(name,prompt) in enumerate(pending,1):
                    print(f'[Experiment] {i}/{len(pending)} | {name}',file=terminal,flush=True)
                    run = root/'experiments'/name
                    run.mkdir(parents=True,exist_ok=True)
                    out = run/'lora'
                    if out.exists():
                        out.rename(out.with_name('lora.incomplete-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
                    cp = run/'prompt_config.yaml'
                    new = config_for(cfg,prompt)
                    s0.write(cp,new)
                    rows,seconds,memory = make_prediction_rows(model,tokenizer,generation_config,records,1,new['prompt'])
                    metrics = score_predictions(records,rows,'val_106.json',1)
                    metrics.update(model='Qwen2.5-1.5B-Instruct-LoRA',method='LoRA',
                                   generation_seconds=round(seconds,3),
                                   mean_generation_seconds_per_question=round(seconds/len(records),4),
                                   questions_per_second=round(len(records)/seconds,3),
                                   peak_gpu_memory_gb=round(memory,3),retrieval=None,faithfulness_percent=None)
                    s0.jsonl(out/'predictions.jsonl',rows)
                    s0.write(out/'metrics.json',metrics)
                    s0.write(out/'generation_manifest.json',{'dataset_sha256':s0.sha(s0.VAL),
                             'expected_count':len(records),'predictions_sha256':s0.sha(out/'predictions.jsonl'),
                             'base_config_sha256':s0.sha(cp),'adapter_sha256':fingerprint,
                             'inference':new['inference'],'prompt':new['prompt']})
                    s0.write(run/'generation_complete.json',{'files':{str(p):s0.sha(p) for p in
                             (cp,out/'predictions.jsonl',out/'metrics.json',out/'generation_manifest.json')}})
            finally:
                del model,base,tokenizer,generation_config
                gc.collect()
                torch.cuda.empty_cache()


def checked_generation(run, cfg, records, fingerprint):
    from evaluate.common import score_predictions, load_jsonl
    marker = s0.read(run/'generation_complete.json')['files']
    for name,digest in marker.items(): verify_sha(Path(name),digest)
    cp = run/'prompt_config.yaml'
    out = run/'lora'
    gm = s0.read(out/'generation_manifest.json')
    if (s0.read(cp) != cfg or gm['prompt'] != cfg['prompt'] or gm['inference'] != cfg['inference']
            or gm['adapter_sha256'] != fingerprint or gm['dataset_sha256'] != s0.sha(s0.VAL)
            or gm['predictions_sha256'] != s0.sha(out/'predictions.jsonl')
            or gm['base_config_sha256'] != s0.sha(cp) or gm['expected_count'] != len(records)):
        raise ValueError(f'Generation provenance mismatch: {run}')
    g = s0.read(out/'metrics.json')
    scores=score_predictions(records,load_jsonl(out/'predictions.jsonl'),'val_106.json',1)
    for key in ('token_f1_percent','exact_match_percent','abstention_rate_percent','false_refusal_rate_percent'):
        if g[key] != scores[key]: raise ValueError(f'Metric mismatch: {key}')
    row = {'token_f1_percent':g['token_f1_percent'],'answer_correctness_percent':None,
           'correct_refusal_rate_percent':g['abstention_rate_percent'],'false_refusal_rate_percent':g['false_refusal_rate_percent'],
           'exact_match_percent':g['exact_match_percent'],'generation_seconds':g['generation_seconds'],
           'questions_per_second':g['questions_per_second'],'inference_gpu_gib':g['peak_gpu_memory_gb'],
           'reference_contradiction_percent':None,'reference_unsupported_percent':None}
    row['youden_j_percent_points']=j(row)
    return row


def judge(run, adapter):
    # Original judge and its exact-input cache, using original base weights.
    qpath=run/'lora/quality_evaluation'
    done=run/'quality_complete.json'
    if done.exists():
        for name,digest in s0.read(done)['files'].items(): verify_sha(Path(name),digest)
    else:
        if qpath.exists(): qpath.rename(qpath.with_name('quality_evaluation.incomplete-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
        from evaluate import evaluate_stage2_quality as qmodule
        original_evaluator=qmodule.Evaluator
        original_config,original_adapter,original_qadapter=s0.CONFIG,s0.adapter,quality.adapter
        terminal=sys.stdout
        try:
            s0.CONFIG=run/'prompt_config.yaml'
            quality.adapter=lambda _run,_method: adapter
            with (run/'quality.log').open('a',encoding='utf-8') as log:
                with redirect_stdout(ProgressLog(log,terminal)),redirect_stderr(log):
                    quality.worker(run,'quality','lora')
            files=[p for p in qpath.iterdir() if p.is_file()]
            s0.write(done,{'files':{str(p):s0.sha(p) for p in files}})
        finally:
            s0.CONFIG,s0.adapter,quality.adapter=original_config,original_adapter,original_qadapter
            qmodule.Evaluator=original_evaluator
    q=s0.read(qpath/'metrics.json')
    if q['dataset_sha256'] != s0.sha(s0.VAL) or q['predictions_sha256'] != s0.sha(run/'lora/predictions.jsonl'):
        raise ValueError('Quality provenance mismatch')
    return {k:q[k] for k in ('answer_correctness_percent','reference_contradiction_percent','reference_unsupported_percent')}


def report(root, baseline, results, statuses, selected, complete):
    s0.write(root/'comparison.json',{'baseline_lora':baseline,'candidates':results,
             'quality_status':statuses,'provisional_selection':selected,'complete':complete,
             'training':'NOT RUN','qlora':'NOT RUN','human_review_required':True,
             'notes':['Only inference system prompt changed. Baseline metrics reused.',
                      'Quality evaluation of only the highest-ranked passing candidate by default.',
                      'Missing quality metrics are null, not zero. Repeated validation tuning is not an independent test.']})
    def fmt(x): return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    lines=['[Stage 3 Test4 Inference Prompt Comparison]',f'{"Metric":36} '+' '.join(f'{n:>24}' for n in ['Baseline_E',*results])]
    for key in KEYS:
        lines.append(f'{key:36} '+' '.join(f'{fmt(r.get(key)):>24}' for r in [baseline,*results.values()]))
    lines += [f'Quality: {statuses}',f'Provisional selection: {selected}',f'Complete: {complete}',
              'New training: NONE | LoRA adapter fixed | QLoRA: NOT RUN',
              'Automated quality verdicts require human review.',f'Results: {root}']
    text='\n'.join(lines)+'\n'
    (root/'comparison.txt').write_text(text,encoding='utf-8')
    print(text,flush=True)


def main():
    p=argparse.ArgumentParser(description='Stage 3 Test4: inference-only system prompt comparison')
    p.add_argument('--parent',type=Path,default=E_ROOT)
    p.add_argument('--resume',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--generation-only',action='store_true',help='Do not run the judge yet')
    p.add_argument('--full-quality',action='store_true',help='Evaluate every candidate regardless of screening')
    args=p.parse_args()
    if args.generation_only and args.full_quality: p.error('Choose only one quality mode')
    if args.resume:
        root=args.resume.resolve()
        if root.parent != RUNS.resolve(): p.error('Invalid resume directory')
        old=s0.read(root/'manifest.json')
        args.parent=Path(old['parent_run'])
        prompts=old['prompts']
    else:
        prompts=PROMPTS
    records,adapter,cfg,baseline,manifest=load_baseline(args.parent)
    if not prompts or any(not re.fullmatch(r'[A-Za-z0-9_]+',n) or not isinstance(v,str) or not v.strip() or v==cfg['prompt']['system'] for n,v in prompts.items()) or len(set(prompts.values()))!=len(prompts):
        raise ValueError('Each candidate needs a unique nonempty system prompt and safe name')
    manifest.update(stage=3,test=4,experiment='inference_prompt',prompts=prompts,
                    runner_sha256=s0.sha(Path(__file__).resolve()))
    if args.resume:
        if manifest != old: raise ValueError('Resume blocked: provenance changed')
    else:
        root=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False)
        s0.write(root/'manifest.json',manifest)
    print(f'[Stage 3 Test4] Run: {root}',flush=True)
    if args.prepare_only:
        print('Preparation: OK (no GPU work)',flush=True)
        return
    try:
        pending=[(n,v) for n,v in prompts.items() if not (root/'experiments'/n/'generation_complete.json').exists()]
        if pending: generate_pending(root,pending,cfg,records,adapter,manifest['adapter_sha256'])
        results={n:checked_generation(root/'experiments'/n,config_for(cfg,v),records,manifest['adapter_sha256']) for n,v in prompts.items()}
        reasons={n:screen(r,baseline) for n,r in results.items()}
        eligible=[n for n in results if not reasons[n]]
        top=max(eligible,key=lambda n:rank_key(results[n])) if eligible else None
        targets=set(results) if args.full_quality else ({top} if top is not None else set())
        if args.generation_only: targets=set()
        statuses={}
        for n,r in results.items():
            run=root/'experiments'/n
            print(f'[Screen] {n} | F1={r["token_f1_percent"]}% | Correct refusal={r["correct_refusal_rate_percent"]}% | False refusal={r["false_refusal_rate_percent"]}% | J={j(r)}%p',flush=True)
            if n in targets or (run/'quality_complete.json').exists():
                r.update(judge(run,adapter))
                statuses[n]='COMPLETE'
            else:
                statuses[n]='SKIPPED: '+('; '.join(reasons[n]) if reasons[n] else 'Not selected for quality evaluation' if not args.generation_only else 'Generation only')
        viable=[n for n in eligible if results[n]['answer_correctness_percent'] is not None and results[n]['answer_correctness_percent'] >= baseline['answer_correctness_percent']]
        selected=max(viable,key=lambda n:(j(results[n]),results[n]['answer_correctness_percent'],results[n]['token_f1_percent'])) if viable else 'baseline'
        verify_sha(s0.VAL,manifest['dataset_sha256'])
        for n,digest in manifest['adapter_sha256'].items(): verify_sha(adapter/n,digest)
        report(root,baseline,results,statuses,selected,not args.generation_only)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/{Path(__file__).name} --resume "{root}"',flush=True)
        raise


if __name__=='__main__':
    main()
