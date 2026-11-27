"""Frozen six-way final test. Install beside run_final_qlora.py. No training."""
from __future__ import annotations
import argparse
import copy
import gc
import importlib.metadata
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import run_final_qlora as matched
from run_final_qlora import ROOT, BASE, OUT, MODEL_OUT, ADAPTERS, CLOSED, CONTEXT, read, write, sha, require

DEST = OUT / 'final_test_v1'
LOCK = DEST / 'freeze.json'
AUDIT = DEST / 'leakage_audit.json'
CACHE = DEST / 'retrieval_cache.json'
TEST = ROOT / 'data/processed/test_clean.json'
TRAIN = {'closed': CLOSED/'train.json', 'context': CONTEXT/'train_run/train.json',
         'original_clean': ROOT/'data/processed/train_clean.json'}
MODELS = {
    'base': ('closed', None, False),
    'base_rag': ('rag', None, False),
    'base_lora': ('closed', ADAPTERS['closed'], False),
    'base_qlora': ('closed', MODEL_OUT/'closed/qlora', True),
    'rag_context_lora': ('rag', ADAPTERS['context'], False),
    'rag_context_qlora': ('rag', MODEL_OUT/'context/qlora', True),
}


def environment():
    import torch
    require(torch.cuda.is_available(), 'CUDA required')
    return {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            **{n: importlib.metadata.version(n) for n in ('transformers','peft','bitsandbytes','sentence-transformers','huggingface-hub','bm25s','faiss-cpu') if installed(n)}}


def installed(name):
    try: importlib.metadata.version(name); return True
    except importlib.metadata.PackageNotFoundError: return False


def asset_hashes(paths):
    result = {}
    for path in paths:
        path = Path(path)
        require(path.exists(), f'Missing freeze asset: {path}')
        items = sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts) if path.is_dir() else [path]
        require(bool(items), f'Empty freeze asset: {path}')
        for p in items:
            require(p.name != 'test_clean.json', 'Do not hash Test before freeze')
            result[str(p.absolute())] = sha(p)
    return result


def freeze():
    import yaml
    import dense_search as dense
    import hybrid_search as hybrid
    import reranker_search as rerank
    from huggingface_hub import snapshot_download
    configs, _, hashes, _ = matched.validate_all()
    require(not DEST.exists(), f'Existing final run protected: {DEST}')
    rag = yaml.safe_load((ROOT/'configs/rag/v3.yaml').read_text(encoding='utf-8-sig'))
    # The installed v3 YAML defaults to reranker. Use the existing Stage3
    # runtime explicitly, as in the Context experiment, retaining v3 prompt/strict guardrail.
    rag['retrieval']['method'] = 'stage3'
    rag['guardrails']['judge_mode'] = 'strict'
    require(rag['guardrails']['enabled'] and rag['retrieval']['top_k']==5 and rag['guardrails']['retry_top_k']==10, 'RAG guardrail settings differ')
    closed = configs['closed']
    require(rag['inference']==closed['inference'] and rag['model']==closed['model'], 'Base/generation settings differ')
    selected = yaml.safe_load((ROOT/'configs/rag/retrieval_stage2_selected.yaml').read_text(encoding='utf-8-sig'))
    require(selected['selected']['reranker_enabled'] and not selected['selected']['query_rewrite_enabled'], 'Unexpected selected retrieval settings')
    rmodel = Path(rerank.RERANKER_MODEL)
    if not rmodel.is_dir():
        rmodel = Path(snapshot_download(repo_id=rerank.RERANKER_MODEL, local_files_only=True))
    paths = [ROOT/'scripts',ROOT/'configs',BASE, dense.MODEL_DIR, dense.INDEX_PATH, dense.CHUNKS_PATH, hybrid.BM25_INDEX_DIR, rmodel, matched.VAL, *TRAIN.values(), *ADAPTERS.values()]
    if dense.PCA_ENABLED: paths.append(dense.PCA_MODEL_PATH)
    for k in ('closed','context'):
        qpath = MODEL_OUT/k/'qlora'
        qinfo = read(OUT/k/'training_complete.json')
        require(qinfo['adapter_sha256']==matched.adapter_check(qpath), f'QLoRA adapter mismatch: {k}')
        comp = read(OUT/k/'validation/comparison.json')
        for method, p in [('lora', ADAPTERS[k]), ('qlora', qpath)]:
            spec = comp[method]['metrics']['spec']
            require(spec['adapter_sha256']==matched.adapter_hash(p), f'Validation adapter mismatch: {k}/{method}')
            require(spec['validation_sha256']==sha(matched.VAL) and spec['prompt']==configs[k]['prompt'] and spec['inference']==configs[k]['inference'], 'Validation configuration mismatch')
            require(spec['quantized']==(method=='qlora'), 'Validation precision mismatch')
        paths += [qpath, OUT/k/'training_complete.json', OUT/k/'validation/comparison.json']
    paths += [CLOSED/'experiment.json', CLOSED/'training_complete.json', CONTEXT/'manifest.json', CONTEXT/'config.json']
    for cfg in (closed,rag):
        cfg['dataset']={'path':'data/processed/test_clean.json'}
        cfg['output']={'directory':str(DEST.relative_to(ROOT))}
    obj={'version':'final-test-v1','configs':{'closed':closed,'rag':rag},'context_training_prompt':configs['context']['prompt'],
         'retrieval_selected':selected,'reranker_snapshot':str(rmodel.absolute()),'environment':environment(),
         'assets':asset_hashes(paths), 'models':{k:{'input':v[0],'adapter':str(v[1]) if v[1] else None,'quantized':v[2]} for k,v in MODELS.items()},
         'guardrail':'same original strict functions; unadapted FP16 Base evidence judge and grounding verifier for every RAG configuration',
         'prompt_policy':'shared detailed closed-book prompt for Base/LoRA/QLoRA; shared existing v3 RAG prompt for all RAG generators (different from Context SFT prompt)',
         'refusal_rule':'prediction == INSUFFICIENT_INFORMATION after existing generation strip', 'test_accessed':False}
    write(LOCK,obj)
    print('Frozen:',LOCK)


def verify_lock():
    lock=read(LOCK)
    require(lock['version']=='final-test-v1','Wrong lock version')
    for name,digest in lock['assets'].items():
        require(sha(Path(name))==digest, f'Frozen asset changed: {name}')
    require(lock['environment']==environment(), 'Frozen runtime environment changed')
    return lock


def audit():
    from run_stage0 import validate, norm
    from final_eval_preflight import normalize
    verify_lock()
    require(not AUDIT.exists(), f'Existing audit protected: {AUDIT}')
    rows=validate(TEST)
    ids={r['id'] for r in rows}
    comps={}
    for name,p in TRAIN.items():
        tr=validate(p)
        ti={r['id'] for r in tr}
        tq={norm(r['question']) for r in tr}
        tq2={normalize(r['question']) for r in tr}
        comps[name]={'train_count':len(tr),'id_overlap':[r['id'] for r in rows if r['id'] in ti],
            'nfc_question_overlap':[r['id'] for r in rows if norm(r['question']) in tq],
            'nfkc_question_overlap':[r['id'] for r in rows if normalize(r['question']) in tq2]}
    dup={n:[q for q,c in Counter(f(r['question']) for r in rows).items() if c>1] for n,f in [('nfc',norm),('nfkc',normalize)]}
    passed=len(ids)==len(rows) and not any(dup.values()) and not any(v[k] for v in comps.values() for k in ('id_overlap','nfc_question_overlap','nfkc_question_overlap'))
    a=sum(not r['is_impossible'] for r in rows)
    result={'lock_sha256':sha(LOCK),'test_sha256':sha(TEST),'total':len(rows),'answerable':a,'impossible':len(rows)-a,'comparisons':comps,'test_duplicate_questions':dup,'passed':passed}
    write(AUDIT,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    require(passed,'Leakage/duplicate detected; final evaluation blocked')


def test_records():
    from run_stage0 import validate
    lock=verify_lock()
    au=read(AUDIT)
    require(au['passed'] and au['lock_sha256']==sha(LOCK) and au['test_sha256']==sha(TEST),'Audit invalid or Test changed')
    return lock,validate(TEST)


def clean_gpu():
    import torch
    gc.collect(); torch.cuda.empty_cache()


def cache():
    import reranker_search
    from rag_pipeline import retrieve_all
    lock,rows=test_records()
    require(not CACHE.exists(),f'Existing cache protected: {CACHE}')
    reranker_search.RERANKER_MODEL=lock['reranker_snapshot']
    docs,seconds=retrieve_all([r['question'] for r in rows],'stage3',10)
    require(len(docs)==len(rows),'Retrieval count mismatch')
    for group in docs:
        require(len(group)<=10 and all(isinstance(d['text'],str) and 'chunk_id' in d for d in group),'Invalid cached document')
    write(CACHE,{'lock_sha256':sha(LOCK),'test_sha256':sha(TEST),'retrieval_seconds':seconds,'rows':[{'id':r['id'],'question':r['question'],'documents':d} for r,d in zip(rows,docs)]})
    print('Cached:',CACHE)


def cached(rows):
    obj=read(CACHE)
    require(obj['lock_sha256']==sha(LOCK) and obj['test_sha256']==sha(TEST),'Cache provenance mismatch')
    require([(r['id'],r['question']) for r in rows]==[(r['id'],r['question']) for r in obj['rows']],'Cache order differs')
    return [r['documents'] for r in obj['rows']]


def evaluate(name, skip_judge=False):
    import torch
    from evaluate.evaluate_base import make_prediction_rows
    from evaluate.common import score_predictions, save_jsonl, load_jsonl, validate_prediction_rows
    from load_base_model import load_model
    from rag_pipeline import generate_rag_answer
    from rag_guardrails import answer_with_guardrails, judge_evidence, verify_grounding
    lock,records=test_records()
    kind,adapter,quantized=MODELS[name]
    cfg=lock['configs'][kind]
    docsets=cached(records) if kind=='rag' else None
    target=DEST/name
    spec={'lock_sha256':sha(LOCK),'test_sha256':sha(TEST),'cache_sha256':sha(CACHE) if kind=='rag' else None,'model':name,'quantized':quantized}
    metrics_path=target/'metrics.json'
    if metrics_path.exists():
        metrics=read(metrics_path)
        require(metrics['spec']==spec and metrics['predictions_sha256']==sha(target/'predictions.jsonl'),'Saved predictions differ')
        preds=load_jsonl(target/'predictions.jsonl'); validate_prediction_rows(records,preds)
    else:
        require(not target.exists(),f'Partial run protected: {target}')
        target.mkdir(parents=True,exist_ok=False)
        if adapter is None: model,tok,gen=load_model(cfg)
        else: model,tok,gen=matched.load_inference(cfg,adapter,quantized)
        if kind=='closed':
            preds,sec,mem=make_prediction_rows(model,tok,gen,records,1,cfg['prompt'])
            timings={'inference_seconds':sec,'peak_gpu_memory_gib':mem,'scope':'closed generation loop; model load excluded'}
        else:
            # Same base judge for all RAG runs. For FP16 LoRA use the base weights
            # with adapters disabled, avoiding a duplicate FP16 model allocation.
            guard_model,guard_tok=(model,tok)
            if quantized:
                guard_model,guard_tok,_=load_model(cfg)
            def call_judge(_m,_t,q,docs):
                if adapter is not None and not quantized:
                    with model.disable_adapter(): return judge_evidence(model,tok,q,docs)
                return judge_evidence(guard_model,guard_tok,q,docs)
            def call_verify(_m,_t,q,docs,a):
                if adapter is not None and not quantized:
                    with model.disable_adapter(): return verify_grounding(model,tok,q,docs,a)
                return verify_grounding(guard_model,guard_tok,q,docs,a)
            preds=[]; sums=Counter(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
            for i,(r,docs) in enumerate(zip(records,docsets),1):
                result=answer_with_guardrails(question=r['question'],retrieved_documents=copy.deepcopy(docs),model=model,tokenizer=tok,generation_config=gen,prompt_config=cfg['prompt'],generate_answer=generate_rag_answer,top_k=5,retry_top_k=10,judge_fn=call_judge,verify_fn=call_verify)
                for k in ('generation_seconds','judge_seconds','verification_seconds'): sums[k]+=result[k]
                preds.append({'id':r['id'],'question':r['question'],'reference_answer':r['answer'],'is_impossible':r['is_impossible'],'prediction':result['answer'],'retrieved_documents':docs,'guardrails':result})
                print(f'{name}: {i}/{len(records)} | {result["status"]}',flush=True)
            torch.cuda.synchronize(); sec=time.perf_counter()-start; mem=torch.cuda.max_memory_allocated()/1024**3
            timings={'inference_seconds':round(sec,3),'peak_gpu_memory_gib':round(mem,3),**{k:round(v,3) for k,v in sums.items()},'scope':'RAG guardrail + generation loop; cache retrieval/model load excluded; includes base guardrail model memory'}
            del call_judge,call_verify,guard_model,guard_tok
        metrics=score_predictions(records,preds,'test_clean.json',1)
        save_jsonl(target/'predictions.jsonl',preds)
        metrics.update(spec=spec,predictions_sha256=sha(target/'predictions.jsonl'),**timings)
        write(metrics_path,metrics)
        del model,tok,gen; clean_gpu()
    jpath=target/'judge_metrics.json'
    if jpath.exists():
        j=read(jpath); require(j['predictions_sha256']==sha(target/'predictions.jsonl'),'Judge input changed')
    elif not skip_judge:
        require(not (target/'judge_cases.jsonl').exists(),'Partial Judge output protected')
        j=matched.judge(records,preds,target); j['predictions_sha256']=sha(target/'predictions.jsonl'); write(jpath,j); clean_gpu()
    print('Result:',metrics_path)


def report():
    from evaluate.common import load_jsonl, validate_prediction_rows
    lock,records=test_records()
    cached(records)
    n=sum(not r['is_impossible'] for r in records); m=len(records)-n
    results={}
    for name in MODELS:
        p=DEST/name
        metrics=read(p/'metrics.json'); preds=load_jsonl(p/'predictions.jsonl'); validate_prediction_rows(records,preds)
        expected={'lock_sha256':sha(LOCK),'test_sha256':sha(TEST),'cache_sha256':sha(CACHE) if MODELS[name][0]=='rag' else None,'model':name,'quantized':MODELS[name][2]}
        require(metrics['spec']==expected and metrics['predictions_sha256']==sha(p/'predictions.jsonl'),'Prediction provenance mismatch')
        j=read(p/'judge_metrics.json') if (p/'judge_metrics.json').exists() else None
        if j: require(j['denominator']==n and j['predictions_sha256']==metrics['predictions_sha256'],'Judge denominator/hash mismatch')
        f=sum(not r['is_impossible'] and x['prediction']=='INSUFFICIENT_INFORMATION' for r,x in zip(records,preds))
        a=sum(r['is_impossible'] and x['prediction']!='INSUFFICIENT_INFORMATION' for r,x in zip(records,preds))
        results[name]={'token_f1_percent':metrics['token_f1_percent'],'judge':j,'false_refusals':{'count':f,'denominator':n,'percent':round(f/n*100,2) if n else None},'impossible_responses':{'count':a,'denominator':m,'percent':round(a/m*100,2) if m else None},'efficiency':metrics}
    training={k:read(OUT/k/'validation/comparison.json')['training'] for k in ('closed','context')}
    output=DEST/('comparison.json' if all(v['judge'] is not None for v in results.values()) else 'comparison_no_judge.json')
    write(output,{'test':read(AUDIT),'results':results,'training':training,'settings':lock['configs'],'note':'RAG uses same Stage3 Top10 cache, strict guardrails and shared v3 generator prompt. RAG inference memory includes guardrail model; not directly comparable to generator-only validation.'})
    print(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','audit','cache','evaluate','report'))
    p.add_argument('--model',choices=tuple(MODELS))
    p.add_argument('--skip-judge',action='store_true')
    a=p.parse_args()
    if a.action=='evaluate':
        if not a.model: p.error('--model required')
        evaluate(a.model,a.skip_judge)
    else: {'freeze':freeze,'audit':audit,'cache':cache,'report':report}[a.action]()

if __name__=='__main__': main()
