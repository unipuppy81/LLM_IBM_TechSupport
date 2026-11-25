"""Fast Stage2 prompt development. Place beside evaluate_stage2_prompt.py.
Default: fixed 20-case judge-only screen. --judge-all: 106 judge-only.
--full: 106 end-to-end, --quality: optional saved-input quality evaluation.
Only val_106; no production source or existing results are modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

import evaluate_stage2_prompt as ab

ROOT = ab.ROOT
PARENT = ROOT / 'evaluate_results/106/rag/stage2_prompt_ab'
PREVIOUS = PARENT / 'run_fa9wu45r'
ABSTAIN = 'INSUFFICIENT_INFORMATION'


def key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def save_rows(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def pct(n, d):
    return round(100*n/d, 2) if d else None


class Cache:
    def __init__(self, path, namespace):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS cache (namespace TEXT, kind TEXT, input_key TEXT, value TEXT, PRIMARY KEY(namespace,kind,input_key))')
        self.namespace = namespace
        self.stats = Counter()

    def get(self, kind, inputs):
        result = self.db.execute('SELECT value FROM cache WHERE namespace=? AND kind=? AND input_key=?',
                                 (self.namespace, kind, key(inputs))).fetchone()
        return json.loads(result[0]) if result else None

    def put(self, kind, inputs, value):
        prior = self.get(kind, inputs)
        if prior is not None:
            if prior != value:
                raise ValueError(f'Conflicting cached {kind} result; use a fresh cache / verify model and logs')
            return
        self.db.execute('INSERT INTO cache VALUES (?,?,?,?)', (self.namespace,kind,key(inputs),json.dumps(value,ensure_ascii=False)))
        self.db.commit()

    def call(self, kind, inputs, fn):
        v = self.get(kind, inputs)
        if v is not None:
            self.stats[kind+'_hit'] += 1
            return v, True
        v = fn()
        self.put(kind, inputs, v)
        self.stats[kind+'_miss'] += 1
        return v, False

    def close(self):
        self.db.close()


def model_identity(path):
    # Avoid hashing multi-GB weights on every prompt experiment; changes to file
    # sizes/mtime invalidate this local cache. This is not a cryptographic model hash.
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError('Empty model directory')
    return [{'name': str(p.relative_to(path)), 'size': p.stat().st_size,
             'mtime_ns': p.stat().st_mtime_ns} for p in files]


def judge_key(prompt, question, doc):
    return [prompt, question, doc]


def selected_from_history(p):
    docs = p['retrieved_documents']
    hist = p['guardrails']['judge_history']
    if len(hist) not in (5,10):
        raise ValueError(f'Unexpected per_document history length: {p["id"]}')
    selected = []
    for rank, h in enumerate(hist, 1):
        d = docs[rank-1]
        if (h.get('rank'), h.get('document_count'), h.get('chunk_id'), h.get('filename')) != (rank,1,d.get('chunk_id'),d.get('filename')):
            raise ValueError(f'History/document mismatch: {p["id"]}')
        if h.get('decision') not in ('YES','NO','UNCERTAIN') or not isinstance(h.get('raw'),str):
            raise ValueError(f'Invalid judge history: {p["id"]}')
        if h['decision'] == 'YES':
            selected.append(d)
    if (len(hist)==5) != bool(any(h['decision']=='YES' for h in hist[:5])):
        raise ValueError(f'Unexpected retry behavior: {p["id"]}')
    if [d.get('chunk_id') for d in selected] != p['guardrails']['used_chunk_ids']:
        raise ValueError(f'Selected chunk mismatch: {p["id"]}')
    return selected


def gate_from_history(p):
    h = p['guardrails']['judge_history']
    return 'YES' if any(x['decision']=='YES' for x in h) else ('UNCERTAIN' if any(x['decision']=='UNCERTAIN' for x in h) else 'NO')


def gate_summary(rows):
    a = [r for r in rows if not r['is_impossible']]
    n = [r for r in rows if r['is_impossible']]
    false = sum(r['decision']!='YES' for r in a)
    correct = sum(r['decision']!='YES' for r in n)
    return {'total':len(rows), 'answerable':len(a), 'impossible':len(n),
            'decisions':dict(Counter(r['decision'] for r in rows)),
            'answerable_blocked':false, 'impossible_blocked':correct,
            'answerable_blocked_percent':pct(false,len(a)),
            'impossible_blocked_percent':pct(correct,len(n))}


def screen(records):
    rng = random.Random(42)
    a = sorted((r for r in records if not r['is_impossible']),key=lambda r:r['id'])
    n = sorted((r for r in records if r['is_impossible']),key=lambda r:r['id'])
    ids = {r['id'] for r in rng.sample(a,12)+rng.sample(n,8)}
    return [r for r in records if r['id'] in ids]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--prompt',type=Path,help='UTF-8 plain text candidate prompt; default is previous B')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--judge-all',action='store_true')
    group.add_argument('--full',action='store_true')
    parser.add_argument('--quality',action='store_true',help='With --full, compare correctness and faithfulness')
    parser.add_argument('--previous-run',type=Path,default=PREVIOUS)
    parser.add_argument('--no-historical-cache',action='store_true',help='Disable seeding historical inference; baseline still uses recorded predictions')
    args=parser.parse_args()
    if args.quality and not args.full:
        parser.error('--quality requires --full')
    start=time.perf_counter()
    sys.path.insert(0,str(ab.SCRIPTS))
    import rag_guardrails as gates
    import rag_guardrails_relaxed as relaxed
    from rag_guardrails_per_document import judge_documents_individually, answer_with_per_document_guardrails
    from rag_pipeline import generate_rag_answer
    from load_base_model import load_model
    from evaluate.common import score_predictions
    import torch, transformers
    config_path, config, records, original, manifest, verified=ab.load_inputs(args)
    if config['guardrails'].get('judge_mode')!='per_document':
        raise ValueError('Expected original Stage2 per_document judge mode')
    old_prompt=relaxed.RELAXED_ANSWERABILITY_SYSTEM
    prompt=args.prompt.read_text(encoding='utf-8-sig').strip() if args.prompt else ab.NEW
    if not prompt:
        raise ValueError('Empty candidate prompt')
    baseline={p['id']:p for p in original}
    old_scores=score_predictions(records, original, 'val_106.json',1)
    if any(old_scores[k]!=v for k,v in ab.HISTORICAL.items()):
        raise ValueError('Original Stage2 metric mismatch')
    source_hashes={n:ab.digest(ab.SCRIPTS/n) for n in ('rag_guardrails.py','rag_guardrails_relaxed.py','rag_guardrails_per_document.py','rag_pipeline.py','load_base_model.py')}
    model_path=ROOT/config['model']['path']
    identity=model_identity(model_path)
    namespace=key({'config':ab.digest(config_path),'dataset':ab.digest(ab.DATA),
        'original_predictions':ab.digest(ab.BASELINE/'predictions.jsonl'),'sources':source_hashes,
        'model_path':str(model_path.resolve()),'model_files':identity,'historical_reuse_allowed':not args.no_historical_cache,'torch':torch.__version__,
        'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    PARENT.mkdir(parents=True,exist_ok=True)
    out=Path(mkdtemp(prefix='fast_',dir=PARENT))
    cache=Cache(PARENT/'fast_cache.sqlite3',namespace)
    model=tokenizer=generation_config=None
    def ensure_model():
        nonlocal model,tokenizer,generation_config
        if model is None:
            model,tokenizer,generation_config=load_model(config)

    def gen_key(question, docs):
        return [config['prompt'],question,docs]
    def ver_key(question, docs, answer):
        return [gates.GROUNDING_SYSTEM,question,docs,answer]
    def judgement(model_unused,tokenizer_unused,question,docs):
        if len(docs)!=1:
            raise ValueError('Expected one excerpt')
        def execute():
            ensure_model()
            user=f'Question:\n{question}\n\nRetrieved Documents:\n{gates._render_documents(docs)}'
            r=gates._llm_decision(model,tokenizer,prompt,user)
            return {k:r[k] for k in ('decision','raw')}
        v,hit=cache.call('judge',judge_key(prompt,question,docs[0]),execute)
        return {**v,'seconds':0.0,'cache_hit':hit}
    def generate(_m,_t,_g,question,docs,_p):
        def execute():
            ensure_model()
            ans,_,tokens=generate_rag_answer(model,tokenizer,generation_config,question,docs,config['prompt'])
            return [ans,tokens]
        v,_=cache.call('generation',gen_key(question,docs),execute)
        return v[0],0.0,v[1]
    def verify(_m,_t,question,docs,answer):
        def execute():
            ensure_model()
            v=gates.verify_grounding(model,tokenizer,question,docs,answer)
            return {k:v[k] for k in ('decision','raw')}
        v,hit=cache.call('verification',ver_key(question,docs,answer),execute)
        return {**v,'seconds':0.0,'cache_hit':hit}
    def seed(rows, source_prompt):
        for p in rows:
            if p['id'] not in baseline:
                raise ValueError('Unexpected cached record ID')
            orig=baseline[p['id']]
            if any(p[k]!=orig[k] for k in ('question','reference_answer','is_impossible','retrieved_documents')):
                raise ValueError(f'Historical cache input mismatch: {p["id"]}')
            docs=p['retrieved_documents']; g=p['guardrails']; q=p['question']
            selected=selected_from_history(p)
            for h,d in zip(g['judge_history'],docs):
                cache.put('judge',judge_key(source_prompt,q,d),{k:h[k] for k in ('decision','raw')})
            ans=g.get('candidate_answer')
            if isinstance(ans,str) and selected:
                cache.put('generation',gen_key(q,selected),[ans,p['input_tokens']])
                v=g.get('verification')
                if isinstance(v,dict):
                    cache.put('verification',ver_key(q,selected,ans),{k:v[k] for k in ('decision','raw')})
    seeded=False
    prev=args.previous_run.resolve()
    if not prev.is_relative_to(PARENT.resolve()):
        raise ValueError('Previous run must be inside Stage2 A/B result directory')
    if not args.no_historical_cache and (prev/'experiment.json').is_file():
        exp=read(prev/'experiment.json'); report=read(prev/'comparison.json')
        checks=(exp['status']=='COMPLETED',report['status']=='COMPLETED',
                exp['judge_mode']=='per_document',exp['only_changed_variable']=='RELAXED_ANSWERABILITY_SYSTEM',
                report['historical_stage2_exact_prediction_matches']==106,
                report['historical_stage2_metrics_reproduced'] is True,
                exp['dataset_sha256']==ab.digest(ab.DATA),exp['rag_config_sha256']==ab.digest(config_path),
                exp['original_predictions_sha256']==ab.digest(ab.BASELINE/'predictions.jsonl'),
                exp['old_prompt']==old_prompt,exp['grounding_prompt']==gates.GROUNDING_SYSTEM,
                exp['generator_prompt']==config['prompt'],
                exp['guardrails_source_sha256']==source_hashes['rag_guardrails.py'],
                exp['relaxed_source_sha256']==source_hashes['rag_guardrails_relaxed.py'],
                exp['per_document_source_sha256']==source_hashes['rag_guardrails_per_document.py'])
        if not all(checks):
            raise ValueError('Previous A/B provenance mismatch. Restore the original setup or use --no-historical-cache.')
        for variant,pr in [('baseline',old_prompt),('new_prompt',exp['new_prompt'])]:
            rs=read_rows(prev/variant/'predictions.jsonl')
            if len(rs)!=106 or {r['id'] for r in rs}!=set(baseline):
                raise ValueError('Previous A/B record IDs mismatch')
            scores=score_predictions(records,rs,'val_106.json',1)
            if any(scores[k]!=report[variant][k] for k in (*ab.HISTORICAL,'exact_match_percent')):
                raise ValueError('Previous A/B score mismatch')
            if variant=='baseline' and any(r['prediction']!=baseline[r['id']]['prediction'] for r in rs):
                raise ValueError('Previous baseline differs from original')
            seed(rs,pr)
        seeded=True
    # The historical A/B logs did not record weight hashes or generator source hash.
    # Reuse assumes that those components have not changed since that run.
    selected_records=records if args.full or args.judge_all else screen(records)
    experiment={'mode':'full' if args.full else 'judge_all' if args.judge_all else 'screen',
        'candidate_prompt':prompt,'baseline_prompt':old_prompt,'dataset_sha256':ab.digest(ab.DATA),
        'config_sha256':ab.digest(config_path),'cache_namespace':namespace,'source_hashes':source_hashes,
        'model_identity_file_metadata':identity,'historical_cache_seeded':seeded,
        'historical_cache_assumption':'Model weights and generator implementation unchanged since previous A/B; not provable from old manifest',
        'baseline_prediction_manifest_verified':verified,'sample_ids':[r['id'] for r in selected_records]}
    ab.write_json(out/'experiment.json',experiment)
    (out/'candidate_prompt.txt').write_text(prompt,encoding='utf-8')
    print('Mode:',experiment['mode'],'| Cases:',len(selected_records),'| Output:',out,flush=True)
    a=[]; b=[]; preds=[]
    for i,r in enumerate(selected_records,1):
        p=baseline[r['id']]; docs=p['retrieved_documents']; q=r['question']
        a.append({'id':r['id'],'is_impossible':r['is_impossible'],'decision':gate_from_history(p),'judge_history':p['guardrails']['judge_history']})
        if args.full:
            result=answer_with_per_document_guardrails(question=q,retrieved_documents=docs,model=None,tokenizer=None,generation_config=None,
                prompt_config=config['prompt'],generate_answer=generate,top_k=5,retry_top_k=10,judge_fn=judgement,verify_fn=verify)
            preds.append(ab.row(r,docs,result))
            hist=result['judge_history']; decision='YES' if any(h['decision']=='YES' for h in hist) else ('UNCERTAIN' if any(h['decision']=='UNCERTAIN' for h in hist) else 'NO')
        else:
            result=judge_documents_individually(model=None,tokenizer=None,question=q,retrieved_documents=docs,top_k=5,retry_top_k=10,judge_fn=judgement)
            hist=result['judge_history']; decision=result['decision']
        b.append({'id':r['id'],'is_impossible':r['is_impossible'],'decision':decision,'judge_history':hist})
        if i==1 or i%5==0 or i==len(selected_records):
            print(f'[Fast Stage2] {i}/{len(selected_records)}',flush=True)
    save_rows(out/'judge_cases.jsonl',[{'id':x['id'],'is_impossible':x['is_impossible'],'baseline':x,'candidate':y} for x,y in zip(a,b)])
    comparison={'status':'COMPLETED','mode':experiment['mode'],'judge_only_proxy':{'baseline':gate_summary(a),'candidate':gate_summary(b)},
        'cache_stats':dict(cache.stats),'note':'Judge-only blocked rates use dataset labels as proxies, not actual evidence sufficiency or final refusal rates. Screening is not a generalization test. Cached timing fields are zero; do not use them for latency comparisons.'}
    if args.full:
        new_scores=score_predictions(records,preds,'val_106.json',1)
        comparison.update({'baseline':old_scores,'new_prompt':new_scores,
            'delta_pp':{k:round(new_scores[k]-old_scores[k],2) for k in ab.HISTORICAL},
            'counts':{'baseline':ab.counts(original,ABSTAIN),'new_prompt':ab.counts(preds,ABSTAIN)}})
        save_rows(out/'predictions.jsonl',preds)
        ab.write_json(out/'generation_manifest.json',{'dataset_sha256':ab.digest(ab.DATA),'rag_config_sha256':ab.digest(config_path),
            'retrieval_config_sha256':manifest['retrieval_config_sha256'],'expected_count':106,'prompt_variant':'candidate'})
        # Write generation result before optional quality work so it survives an evaluator failure.
        ab.write_json(out/'comparison.json',comparison)
    if args.quality:
        import evaluate_stage2_quality as quality
        import gc
        model=tokenizer=generation_config=None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        qnamespace=key({'quality_source':ab.digest(Path(quality.__file__)),'model':namespace})
        qcache=Cache(PARENT/'fast_cache.sqlite3',qnamespace)
        evaluator=None
        class CachedEvaluator:
            def evaluate(self,system,payload,allowed):
                nonlocal evaluator
                def execute():
                    nonlocal evaluator
                    if evaluator is None: evaluator=quality.Evaluator(model_path.resolve())
                    return evaluator.evaluate(system,payload,allowed)
                v,_=qcache.call('quality',[system,payload,sorted(allowed)],execute)
                return v
        # Seed previous original quality only when prediction and dataset hashes match.
        # Old quality log does not fingerprint the evaluator code/weights.
        oldq=ab.BASELINE/'quality_evaluation'
        qseed=False
        if not args.no_historical_cache and (oldq/'metrics.json').is_file() and (oldq/'cases.jsonl').is_file():
            m=read(oldq/'metrics.json'); rs=read_rows(oldq/'cases.jsonl'); idx={r['id']:r for r in rs}
            if (m['predictions_sha256']!=ab.digest(ab.BASELINE/'predictions.jsonl') or m['dataset_sha256']!=ab.digest(ab.DATA)
                or Path(m['judge_model']).resolve()!=model_path.resolve() or len(rs)!=106 or set(idx)!=set(baseline)):
                raise ValueError('Original quality provenance mismatch')
            summary=quality.summarize(rs)
            if any(summary[k]!=m[k] for k in ('answer_correctness','faithfulness','policy')):
                raise ValueError('Original quality metrics mismatch')
            for r in records:
                p=baseline[r['id']]; c=idx[r['id']]; ans=p['prediction']; used,_=quality.used_excerpts(p)
                if any(c[k]!=p[k] for k in ('question','reference_answer','is_impossible','prediction')) or c['used_chunk_ids']!=(None if used is None else [d['chunk_id'] for d in used]):
                    raise ValueError('Original quality inputs mismatch')
                common={'question':r['question'],'candidate_answer':ans}
                if not r['is_impossible'] and ans.strip()!=ABSTAIN:
                    qcache.put('quality',[quality.CORRECT_SYSTEM,{**common,'reference_answer':r['answer'],'reference_documents':quality.gold_excerpts(r)},sorted(quality.CORRECT_LABELS)],c['correctness'])
                if ans.strip()!=ABSTAIN and used:
                    qcache.put('quality',[quality.FAITH_SYSTEM,{**common,'generator_excerpts':used},sorted(quality.FAITH_LABELS)],c['faithfulness'])
            qseed=True
        qe=CachedEvaluator(); qa=[]; qb=[]
        for i,(r,p) in enumerate(zip(records,preds),1):
            qa.append(quality.evaluate_case(r,baseline[r['id']],qe))
            qb.append(quality.evaluate_case(r,p,qe))
            if i==1 or i%5==0 or i==len(records): print(f'[Fast Quality] {i}/{len(records)}',flush=True)
        save_rows(out/'baseline_quality_cases.jsonl',qa); save_rows(out/'candidate_quality_cases.jsonl',qb)
        qr={'baseline':quality.summarize(qa),'candidate':quality.summarize(qb),'cache_stats':dict(qcache.stats),
            'historical_quality_seeded':qseed,'note':'Historical quality reuse assumes unchanged evaluator prompts and model weights; same-base automated judge requires human review.'}
        ab.write_json(out/'quality_comparison.json',qr); comparison['quality_file']='quality_comparison.json'; qcache.close()
    comparison['wall_seconds']=round(time.perf_counter()-start,2)
    ab.write_json(out/'comparison.json',comparison)
    cache.close()
    print(json.dumps(comparison,ensure_ascii=False,indent=2),flush=True)
    print('Saved:',out,flush=True)


if __name__=='__main__':
    main()
