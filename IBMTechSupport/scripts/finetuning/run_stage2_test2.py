"""Stage 2 Test2: controlled prompt/length ladder (validation 106 only).
Save next to run_stage2_prompt.py and the existing Stage 0/1 runners.
No edits to original source, datasets, adapters or prior results.
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

import run_stage2_prompt as previous

s0 = previous.s0
quality = previous.quality
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage2_test2'
OLD_PROMPT = previous.OLD_PROMPT
# User-approved detailed prompt. Edit only before starting a new run.
NEW_PROMPT = '''You are an IBM technical support assistant specializing in technical troubleshooting and product support.

# 1. Question Analysis

Break down the user's question into 1 to 5 essential answer requirements.
Identify what information the user is requesting.

Distinguish between the following types of technical questions:
- Error diagnosis
- Troubleshooting
- Configuration
- Installation and upgrade
- Product compatibility
- Technical explanation
- Product functionality

# 2. Technical Condition Extraction

Use the following five technical conditions as basic validation criteria:
- Product
- Version
- Error code
- Environment
- Module

Extract these conditions from the user's input.
Use only the conditions explicitly present in the user's input for subsequent validation.
Skip any conditions that are not present in the user's input.
Do not assume or infer unspecified technical conditions.

The five technical conditions are basic validation criteria, not mandatory input requirements.
If a condition is not specified in the user's input, exclude that condition from the mandatory matching requirements.

An unspecified condition must not, by itself, cause answer refusal.
A question may be answerable even if one or more technical conditions are unspecified.

# 3. Technical Knowledge Validation

Use your existing technical knowledge to determine whether you can answer the question.

Validate your technical knowledge against the explicitly stated technical conditions.

Do not apply a solution that is known to be incompatible with the specified product, version, error code, environment, or module.

Do not confuse missing technical conditions with incompatible technical conditions.

Matching a product name or error code alone is insufficient.
Your answer must address the user's actual requested operation or technical issue.

# 4. Answerability Determination

Determine whether you can provide a technically reliable answer to the essential answer requirements.

A question is answerable when:
- You understand the user's requested information.
- Your technical knowledge is compatible with the explicitly stated conditions.
- You can provide a reliable answer to the requested technical issue.

Do not refuse a question merely because:
- The question is short.
- Some technical conditions are unspecified.
- Nonessential details are missing.
- Multiple valid solutions may exist.
- The question does not include an error code or product version.

If an unspecified condition materially affects the solution, provide a conditional answer when you have sufficient knowledge to explain the applicable conditions.

# 5. Answer Generation

When the question is answerable, provide a direct and technically accurate answer in English.

Address the user's requested information before providing additional explanations.

For troubleshooting questions, provide the relevant cause and resolution when known.

For configuration questions, provide the relevant configuration settings and procedures.

For compatibility questions, identify the applicable products, versions, and limitations when known.

Include commands, configuration parameters, prerequisites, and technical restrictions when necessary.

Do not invent product names, version numbers, error codes, commands, configuration parameters, or technical procedures.

Do not introduce unrelated technical information.

# 6. Answer Refusal

Refuse to answer only when:
- You lack the technical knowledge necessary to address the essential requested information.
- Essential technical information is missing and no reliable conditional answer can be provided.
- You cannot provide a technically reliable answer without inventing information.

Do not use answer refusal as the default response.

When a reliable answer can be provided, generate the technical answer instead of refusing.

# 7. Output Requirements

Perform question analysis, technical validation, and answerability determination internally.

Do not output the analysis process, extracted conditions, or intermediate decisions.

If the question is answerable, provide only the technical answer in English.

If the question is not answerable, respond with exactly:

INSUFFICIENT_INFORMATION'''

KEYS = previous.KEYS
NAMES = ('A_length', 'B_prompt', 'C_question', 'D_answer', 'E_generation')


def variants():
    specs = [('baseline', OLD_PROMPT, 512, 192, 160, 160),
             (NAMES[0], OLD_PROMPT, 2048, 192, 160, 160),
             (NAMES[1], NEW_PROMPT, 2048, 192, 160, 160),
             (NAMES[2], NEW_PROMPT, 2048, 512, 160, 160),
             (NAMES[3], NEW_PROMPT, 2048, 512, 512, 160),
             (NAMES[4], NEW_PROMPT, 2048, 512, 512, 512)]
    return {n: dict(prompt=p, max_length=l, question_tokens=q,
                    answer_tokens=a, max_new_tokens=g) for n,p,l,q,a,g in specs}


def config_for(run):
    return run/'prompt_config.yaml'


def config_for_spec(spec):
    import yaml
    cfg = yaml.safe_load(s0.CONFIG.read_text(encoding='utf-8-sig'))
    if cfg['prompt'] != {'system': OLD_PROMPT, 'user_template':'{question}'}:
        raise ValueError('Original prompt configuration changed')
    if (cfg['inference']['max_new_tokens'],cfg['inference']['do_sample'],cfg['inference']['num_beams']) != (160,False,1):
        raise ValueError('Unexpected original inference settings')
    cfg['prompt']['system'] = spec['prompt']
    cfg['inference']['batch_size'] = 1
    cfg['inference']['max_new_tokens'] = spec['max_new_tokens']
    return cfg


def adapter(run, method):
    if method != 'lora':
        raise ValueError('Test2 supports LoRA only')
    name = NAMES[3] if run.name == NAMES[4] else run.name
    return ROOT/'models/finetuning/stage2_test2'/run.parent.parent.name/name/method


def configure_trainer(spec):
    m = importlib.import_module('train_lora')
    m.SYSTEM_PROMPT = spec['prompt']
    m.MAX_QUESTION_TOKENS = spec['question_tokens']
    m.MAX_ANSWER_TOKENS = spec['answer_tokens']
    return m


def audit_tokens(source, specs):
    # Measure the real tokenizer and actual QADataset, not estimated token counts.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(s0.BASE, local_files_only=True)
    splits = {'train':s0.validate(source),'validation':s0.validate(s0.VAL)}
    raw = {}
    for name, rows in splits.items():
        raw[name] = [(len(tok.encode(r['question'], add_special_tokens=False)),
                      len(tok.encode('INSUFFICIENT_INFORMATION' if r['is_impossible'] else r['answer'],add_special_tokens=False))) for r in rows]
    def distribution(values):
        a = sorted(values)
        return {'min':a[0],'median':a[(len(a)-1)//2], 'p95':a[(95*len(a)+99)//100-1], 'max':a[-1]}
    out = {'raw_lengths':{n:{'question':distribution([x[0] for x in pairs]),
                             'answer':distribution([x[1] for x in pairs])} for n,pairs in raw.items()},'variants':{}}
    for index,(name,spec) in enumerate(specs.items()):
        # E changes only generation; D's training tensors are identical by construction.
        if name == NAMES[4]:
            out['variants'][name] = out['variants'][NAMES[3]]
            continue
        print(f'[Token audit] Configuration {index+1}/5',flush=True)
        m = configure_trainer(spec)
        info = {'system_prompt_tokens':len(tok.encode(spec['prompt'],add_special_tokens=False)),'splits':{}}
        for split, rows in splits.items():
            h = hashlib.sha256()
            qcut = acut = maxseq = 0
            used_lengths = []
            dataset = m.QADataset(rows,tok,spec['max_length'])
            if len(dataset) != len(rows):
                raise ValueError('Unexpected QADataset size')
            for i,(row,sample,(qlen,alen)) in enumerate(zip(rows,dataset,raw[split]),1):
                ids = sample['input_ids'].reshape(-1).tolist()
                labels = sample['labels'].reshape(-1).tolist()
                mask = sample['attention_mask'].reshape(-1).tolist()
                if not(len(ids)==len(labels)==len(mask)) or len(ids)>spec['max_length']:
                    raise ValueError(f'Invalid tensor length: {row["id"]}')
                targets = [v for v in labels if v != -100]
                if len(targets)<2 or targets[-1]!=tok.eos_token_id:
                    raise ValueError(f'No valid answer token space: {row["id"]}')
                if labels != [-100]*(len(ids)-len(targets))+ids[-len(targets):]:
                    raise ValueError(f'Invalid label masking: {row["id"]}')
                h.update(json.dumps([ids,mask,labels],separators=(',',':')).encode('utf-8')+b'\n')
                qcut += qlen > spec['question_tokens']
                acut += alen > len(targets)-1
                maxseq = max(maxseq,len(ids))
                used_lengths.append(len(targets)-1)
                if i==1 or i%100==0 or i==len(rows):
                    print(f'[Token audit {split}] Sample {i}/{len(rows)}',flush=True)
            info['splits'][split] = {'samples':len(rows),'question_truncated':qcut,'answer_truncated':acut,
                                    'max_sequence_tokens':maxseq,'answer_tokens_used':distribution(used_lengths),'tensor_sha256':h.hexdigest()}
        out['variants'][name] = info
    # Restore module globals in the parent after the CPU audit.
    configure_trainer(specs['baseline'])
    return out


def worker(run, name):
    manifest = s0.read(run.parent.parent/'manifest.json')
    spec = manifest['variants'][run.name]
    cfg = s0.read(config_for(run))
    if s0.sha(run/'train.json') != manifest['training_sha256']:
        raise ValueError('Training dataset snapshot changed')
    # Only process-local overrides; no existing code files are written.
    quality.adapter = adapter
    s0.adapter = adapter
    if name=='train':
        if run.name==NAMES[4]:
            raise ValueError('E must not retrain')
        m = importlib.import_module('train_lora')
        if (m.SYSTEM_PROMPT,m.MAX_QUESTION_TOKENS,m.MAX_ANSWER_TOKENS)!=(OLD_PROMPT,192,160):
            raise ValueError('Original trainer settings changed')
        configure_trainer(spec)
        s0.SETTINGS = dict(s0.SETTINGS,max_length=spec['max_length'])
        quality.worker(run,'train','lora')
    elif name=='generate':
        from evaluate import evaluate_lora, common
        # inputs() intentionally enforces the historical 160-token setting;
        # call the existing generate() directly with the verified new config.
        common.MAX_NEW_TOKENS = spec['max_new_tokens']
        evaluate_lora.ADAPTER = adapter(run,'lora')
        evaluate_lora.OUTPUT = run/'lora'
        evaluate_lora.BASE_CONFIG = config_for(run)
        records = s0.validate(s0.VAL)
        evaluate_lora.generate(cfg,records)
    elif name=='quality':
        from evaluate import common
        common.MAX_NEW_TOKENS = spec['max_new_tokens']
        generated = s0.read(run/'lora/generation_manifest.json')
        if generated['prompt'] != cfg['prompt'] or generated['inference'] != cfg['inference']:
            raise ValueError('Generation config mismatch')
        s0.CONFIG = config_for(run)
        quality.worker(run,'quality','lora')
    else:
        raise ValueError(name)


def phase(run,name):
    tag = name+'_lora'
    marker = run/'completed'/f'{tag}.json'
    if marker.is_file():
        previous.check_done(run,tag)
        return
    targets = ([adapter(run,'lora'),run/'training/lora_train_metrics.json'] if name=='train'
               else [run/'lora'] if name=='generate' else [run/'lora/quality_evaluation'])
    for p in targets:
        if p.exists():
            p.rename(p.with_name(p.name+'.incomplete-'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    log = run/'logs'/f'{tag}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',name,'--run',str(run)]
    with log.open('a',encoding='utf-8') as f:
        with subprocess.Popen(cmd,cwd=ROOT.parent,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',bufsize=1,
                              env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONUNBUFFERED':'1'}) as proc:
            try:
                for line in proc.stdout:
                    f.write(line);f.flush()
                    line=line.strip()
                    if line.startswith('[Training]'): print(line,flush=True)
                    elif (match:=re.fullmatch(r'Generated: (\d+/\d+)',line)): print('[Evaluation] Question '+match[1],flush=True)
                    elif (match:=re.fullmatch(r'\[lora quality\] (\d+/\d+)',line)): print('[Quality] Question '+match[1],flush=True)
                code=proc.wait()
            except BaseException:
                proc.terminate();proc.wait();raise
    if code:
        raise RuntimeError(f'{tag} failed: {log}\n'+'\n'.join(log.read_text(encoding='utf-8').splitlines()[-15:]))
    files=[f for p in targets for f in ([p] if p.is_file() else list(p.rglob('*'))) if f.is_file()]
    if not files or not all(p.exists() for p in targets):
        raise RuntimeError(f'Missing artifacts: {tag}')
    s0.write(marker,{'files':{str(f):s0.sha(f) for f in files}})


def get_metrics(run, training_run=None):
    # E uses D's unchanged training metrics, with E's own generation/quality metrics.
    if training_run is None:
        return quality.metrics(run,'lora')
    g=s0.read(run/'lora/metrics.json');q=s0.read(run/'lora/quality_evaluation/metrics.json')
    result=quality.metrics(training_run,'lora')
    result.update(token_f1_percent=g['token_f1_percent'],answer_correctness_percent=q['answer_correctness_percent'],
                  correct_refusal_rate_percent=g['abstention_rate_percent'],false_refusal_rate_percent=g['false_refusal_rate_percent'],
                  exact_match_percent=g['exact_match_percent'],inference_gpu_gib=g['peak_gpu_memory_gb'],
                  questions_per_second=g['questions_per_second'],reference_contradiction_percent=q['reference_contradiction_percent'],
                  reference_unsupported_percent=q['reference_unsupported_percent'])
    return result


def save_comparison(root,base,results,execution):
    deltas={};prior=base
    for n in NAMES:
        if n not in results: break
        r=results[n]
        deltas[n]={k:round(r[k]-prior[k],4) if r.get(k) is not None and prior.get(k) is not None else None for k in KEYS}
        prior=r
    selected=quality.select(results,base)
    value={'baseline_lora':base,'candidates':results,'delta_vs_previous':deltas,'execution':execution,
           'provisional_selection':selected,'complete':len(results)==5,'qlora':'NOT RUN','human_review_required':True,
           'hallucination_rate_percent':None,'notes':['B-D are cumulative controlled comparisons, not independent ablations.','A is reused only if all train/val tensors and inference settings match baseline.','E shares D training metrics; no new training.','Validation losses with different prompt/target lengths are not directly comparable.']}
    s0.write(root/'comparison.json',value)
    def fmt(x): return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    names=[n for n in NAMES if n in results]
    lines=['[Stage 2 Test2 Comparison]',f'{"Metric":36} {"Stage 1-2":>12} '+ ' '.join(f'{n:>12}' for n in names)]
    for k in KEYS:
        lines.append(f'{k:36} {fmt(base.get(k)):>12} '+' '.join(f'{fmt(results[n].get(k)):>12}' for n in names))
    lines += [f'Execution: {execution}',f'Provisional selection: {selected}',
              'Validation losses across different tokenizations are not directly comparable.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN',f'Results: {root}']
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,help='Completed Stage 1-2 quality run; default latest')
    p.add_argument('--resume',type=Path,help='Resume same run')
    p.add_argument('--audit-only',action='store_true',help='Prepare and audit tokens without training')
    p.add_argument('--through',choices=['A','B','C','D','E'],default='E',help='Stop after specified experiment (default E)')
    p.add_argument('--worker',choices=['train','generate','quality'],help=argparse.SUPPRESS)
    p.add_argument('--run',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        if args.run is None or args.run.resolve().parent.parent.parent!=RUNS.resolve() or args.run.name not in NAMES:
            p.error('Invalid worker directory')
        worker(args.run.resolve(),args.worker);return
    if args.resume:
        root=args.resume.resolve()
        if root.parent!=RUNS.resolve():p.error('Invalid resume directory')
        old=s0.read(root/'manifest.json');args.parent=Path(old['parent_run'])
    parent,baseline,provenance,source,base=previous.load_parent(args.parent)
    specs=variants()
    configs={n:config_for_spec(s) for n,s in specs.items()}
    data=s0.audit()
    modules=(s0,quality,quality.s1,quality.cleanup,previous)
    manifest={'stage':2,'test':2,'experiment':'prompt_length_ladder','parent_run':str(parent),'baseline':str(baseline),
              'parent_manifest_sha256':s0.sha(parent/'manifest.json'),'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
              'training_source':str(source),'training_sha256':s0.sha(source),'data':data,'settings':s0.SETTINGS,
              'variants':specs,'configs':configs,'sources':provenance['sources'],'model_files':provenance['base_model_files'],
              'runner_sha256':{f.name:s0.sha(f) for f in (Path(__file__),*(Path(m.__file__) for m in modules))}}
    if args.resume:
        if manifest!=old:raise ValueError('Resume blocked: provenance changed')
    else:
        root=RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True,exist_ok=False);s0.write(root/'manifest.json',manifest)
    print(f'[Stage 2 Test2] Run: {root}',flush=True)
    audit_file=root/'token_audit.json'
    try:
        if audit_file.is_file():
            audit=s0.read(audit_file)
            if audit['manifest_sha256']!=s0.sha(root/'manifest.json'):raise ValueError('Audit provenance mismatch')
        else:
            audit=audit_tokens(source,specs);audit['manifest_sha256']=s0.sha(root/'manifest.json');s0.write(audit_file,audit)
        for n in ('baseline',*NAMES):
            a=audit['variants'][n]
            print(f'[Token audit] {n}: prompt={a["system_prompt_tokens"]}, train question/answer truncated={a["splits"]["train"]["question_truncated"]}/{a["splits"]["train"]["answer_truncated"]}, max sequence={a["splits"]["train"]["max_sequence_tokens"]}',flush=True)
        if args.audit_only:return
        results={};execution={}
        a=audit['variants'];reuse_a=all(a['baseline']['splits'][s]['tensor_sha256']==a[NAMES[0]]['splits'][s]['tensor_sha256'] for s in ('train','validation')) and configs['baseline']==configs[NAMES[0]]
        for i,n in enumerate(NAMES,1):
            if i>'ABCDE'.index(args.through)+1:break
            print(f'[Experiment] {i}/5 | {n}',flush=True)
            if n==NAMES[0] and reuse_a:
                results[n]=dict(base);execution[n]='REUSED Stage 1-2 (identical training tensors and inference config)'
                print('[Experiment] A reused: identical tokenized inputs/labels',flush=True);continue
            run=root/'experiments'/n;run.mkdir(parents=True,exist_ok=True)
            dest=run/'train.json'
            if not dest.exists():dest.write_bytes(source.read_bytes())
            if s0.sha(dest)!=manifest['training_sha256']:raise ValueError('Training data changed')
            cpath=config_for(run)
            if not cpath.exists():s0.write(cpath,configs[n])
            if s0.read(cpath)!=configs[n]:raise ValueError('Config changed')
            if n==NAMES[4]:
                d=root/'experiments'/NAMES[3]
                previous.check_done(d,'train_lora')
                phase(run,'generate');phase(run,'quality')
                results[n]=get_metrics(run,d);execution[n]='EVALUATION ONLY (D adapter)'
            else:
                for name in ('train','generate','quality'):phase(run,name)
                results[n]=get_metrics(run);execution[n]='TRAIN AND EVALUATE'
            if results[n]['train_samples']!=539:raise ValueError('Unexpected training count')
            if s0.sha(source)!=manifest['training_sha256'] or s0.audit()!=data:raise ValueError('Dataset changed during experiment')
        save_comparison(root,base,results,execution)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/run_stage2_test2.py --resume "{root}" --through {args.through}',flush=True)
        raise


if __name__=='__main__':main()
