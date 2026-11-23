#!/usr/bin/env python3
"""Generate a 135-question IBM RAG audit report from the existing local repo outputs.

Run from the repository root:
    python make_rag_case_report.py --repo .

All source files are read-only. The script only creates one TXT report.
Python 3.10+; no third-party dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path

SOURCE = {
    'reranker': 'IBMTechSupport/evaluate_results/rag/v1/predictions.jsonl',
    'hybrid': 'IBMTechSupport/evaluate_results/rag/v2/predictions.jsonl',
    'base': 'IBMTechSupport/evaluate_results/base/v1/predictions.jsonl',
    'judge15': 'IBMTechSupport/results/answerability_evaluation.jsonl',
    'judge7': 'IBMTechSupport/results/evidence_judge_qwen25_7b_v2.jsonl',
    'review': 'IBMTechSupport/results/evidence_ground_truth_135_reviewed.jsonl',
    'retrieval_comparison': 'IBMTechSupport/results/retrieval_comparison.json',
    'original_qa': 'IBMTechSupport/data/processed/val_clean.json',
    'metric_reranker': 'IBMTechSupport/evaluate_results/rag/v1/metrics.json',
    'metric_hybrid': 'IBMTechSupport/evaluate_results/rag/v2/metrics.json',
    'metric_base': 'IBMTechSupport/evaluate_results/base/v1/metrics.json',
    'metric_retrieval': 'IBMTechSupport/evaluate_results/rag/v1/retrieval_metrics.json',
    'train_lora': 'IBMTechSupport/results/lora_train_metrics.json',
    'train_qlora': 'IBMTechSupport/results/qlora_train_metrics.json',
    'e2e_reranker': 'IBMTechSupport/evaluate_results/rag/final_e2e/reranker/predictions.jsonl',
    'e2e_hybrid': 'IBMTechSupport/evaluate_results/rag/final_e2e/hybrid/predictions.jsonl',
    'e2e_metrics': 'IBMTechSupport/evaluate_results/rag/final_e2e/reranker/metrics.json',
}


def load(path: Path, optional=False):
    if not path.is_file():
        if optional:
            return None
        raise FileNotFoundError(f'필수 결과 파일 없음: {path}')
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding='utf-8-sig') as f:
        return json.load(f)


def index(rows, name):
    if rows is None:
        return {}
    if not isinstance(rows, list):
        raise ValueError(f'{name}: JSONL 또는 리스트 자료가 아닙니다')
    table = {}
    for item in rows:
        id_ = item['id']
        if id_ in table:
            raise ValueError(f'{name}: ID 중복 {id_}')
        table[id_] = item
    return table


def namekey(filename):
    return Path(str(filename).replace('\\', '/')).name.casefold()


def gold_ranks(docs, gold_files):
    gold = {namekey(f) for f in gold_files}
    return [i for i, doc in enumerate(docs, 1) if namekey(doc['filename']) in gold]


def conf(actual, predicted):
    if actual:
        return 'TP(정상 승인)' if predicted else 'FN(잘못된 거부)'
    return 'FP(잘못된 승인)' if predicted else 'TN(정상 거부)'


def pct(n, d):
    return f'{100 * n / d:.2f}%' if d else 'N/A'


def txt(value):
    return '[미기록]' if value is None else str(value)


def seq(items):
    if not items:
        return '없음'
    return ', '.join(map(str, items))


def run(repo: Path, output: Path) -> dict:
    repo = repo.resolve()
    raw = {}
    required = {'reranker', 'hybrid', 'base', 'judge15', 'judge7', 'review', 'retrieval_comparison'}
    for key, rel in SOURCE.items():
        raw[key] = load(repo / rel, optional=(key not in required))

    tables = {key: index(raw[key], key) for key in
              ('reranker', 'hybrid', 'base', 'judge15', 'judge7', 'review', 'original_qa', 'e2e_reranker', 'e2e_hybrid')}
    ids = [r['id'] for r in raw['reranker']]
    if len(ids) != 135 or len(set(ids)) != 135:
        raise ValueError(f'기대 평가 건수 135건, 실제 {len(ids)}건')
    for key in ('hybrid', 'base', 'judge15', 'judge7', 'review'):
        if set(tables[key]) != set(ids):
            raise ValueError(f'{key} 결과의 ID 집합이 Reranker 135건과 다릅니다')

    rc = raw['retrieval_comparison']
    gold_index = {r['id']: r['gold_filenames'] for r in rc['bm25']['details']}
    if len(gold_index) != 90:
        raise ValueError(f'원본 gold 파일명 데이터는 90건이어야 합니다. 실제 {len(gold_index)}건')

    records = []
    ranking = Counter()
    ranking_h = Counter()
    qa_matrix, evidence_matrix = Counter(), Counter()
    verdicts, evidence_labels = Counter(), Counter()
    selected7, matched7, support_count = 0, 0, 0
    for id_ in ids:
        r = tables['reranker'][id_]
        h = tables['hybrid'][id_]
        a = tables['judge15'][id_]
        j = tables['judge7'][id_]
        g = tables['review'][id_]
        o = tables['original_qa'].get(id_)
        e = tables['e2e_reranker'].get(id_)
        docs = r['retrieved_documents']
        if len(docs) != 5:
            raise ValueError(f'{id_}: 저장 RAG v1 문서 수가 5가 아닙니다: {len(docs)}')
        if any(other['question'] != r['question'] for other in (a, j, g, h)):
            raise ValueError(f'{id_}: 질문이 다른 평가 결과를 섞으려 했습니다')
        if not all(doc.get('chunk_id') == a['retrieved_documents'][i].get('chunk_id')
                   and doc.get('chunk_id') == j['retrieved_document_meta'][i].get('chunk_id')
                   for i, doc in enumerate(docs)):
            raise ValueError(f'{id_}: Judge가 다른 Top5 검색 결과를 사용했습니다')
        if g.get('source_fingerprint') != j.get('input_fingerprint'):
            raise ValueError(f'{id_}: 근거 검토표와 7B Judge의 입력 fingerprint 불일치')

        is_qa = not r['is_impossible']
        if o is not None:
            if o['question'] != r['question'] or o['is_impossible'] != r['is_impossible']:
                raise ValueError(f'{id_}: val_clean 질문/레이블 불일치')
            files = [c['filename'] for c in o.get('contexts', [])]
            gold_texts = [(c['filename'], c.get('text', '')) for c in o.get('contexts', [])]
        else:
            files = gold_index.get(id_, [])
            gold_texts = []
        if is_qa and not files:
            raise ValueError(f'{id_}: 답변 가능인데 Gold 문서 목록이 없습니다')
        # Comparison JSON is the retriever's original recorded gold document set.
        if is_qa and set(map(namekey, files)) != set(map(namekey, gold_index.get(id_, []))):
            raise ValueError(f'{id_}: 로컬 val_clean과 retrieval_comparison Gold 문서가 다릅니다')

        rr = gold_ranks(docs, files) if is_qa else []
        hr = gold_ranks(h['retrieved_documents'], files) if is_qa else []
        if is_qa:
            ranking[rr[0] if rr else 0] += 1
            ranking_h[hr[0] if hr else 0] += 1
        predicted = bool(a['predicted_answerable'])
        qa_matrix[conf(is_qa, predicted)[:2]] += 1
        label = g['evidence_label']
        evidence_labels[label] += 1
        if label in ('Y', 'N'):
            evidence_matrix[conf(label == 'Y', predicted)[:2]] += 1
        verdicts[j['judge_verdict']] += 1
        support_ranks = set(g.get('evidence_document_ranks') or [])
        support_count += len(support_ranks)
        chosen_j = (j.get('evidence_document_ranks') or []) if j['judge_verdict'] == 'YES' else []
        selected7 += len(chosen_j)
        matched7 += sum(v in support_ranks for v in chosen_j)
        records.append({'id': id_, 'r': r, 'h': h, 'b': tables['base'][id_], 'a': a, 'j': j,
                        'g': g, 'o': o, 'e': e, 'is_qa': is_qa, 'gold': files,
                        'gold_texts': gold_texts, 'ranks': rr, 'ranks_h': hr})

    n_pos = sum(x['is_qa'] for x in records)
    n_neg = len(records) - n_pos
    if (n_pos, n_neg) != (90, 45):
        raise ValueError('평가 레이블 수가 90/45가 아닙니다')
    counts = {'n': len(ids), 'answerable': n_pos, 'impossible': n_neg,
              'reranker_hit5': n_pos - ranking[0], 'hybrid_hit5': n_pos - ranking_h[0],
              'qa': dict(qa_matrix), 'evidence': dict(evidence_matrix)}

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig') as f:
        def emit(*parts):
            for part in parts:
                f.write(str(part) + '\n')

        emit('IBM TechSupport RAG — 135건 질문별 검색·생성·판단 분석',
             f'생성일: {date.today().isoformat()}', f'저장소: {repo}',
             '분석 기준: Reranker RAG v1 검색/생성 + 별도 1.5B Judge + 별도 7B Judge',
             '=' * 110, '1. 종합 지표')
        emit(f'QA: 전체 {len(records)}, 답변 가능 {n_pos}, 답변 불가능 {n_neg}',
             f'Reranker v1 Gold Hit@5: {counts["reranker_hit5"]}/{n_pos} = {pct(counts["reranker_hit5"], n_pos)}',
             f'Reranker v1 Gold 순위: 1위 {ranking[1]} / 2위 {ranking[2]} / 3위 {ranking[3]} / 4위 {ranking[4]} / 5위 {ranking[5]} / Top5 실패 {ranking[0]}',
             f'Hit@1 {pct(ranking[1],n_pos)}, Hit@3 {pct(ranking[1]+ranking[2]+ranking[3],n_pos)}, Hit@5 {pct(counts["reranker_hit5"],n_pos)}',
             f'Hybrid v2 Gold Hit@5: {counts["hybrid_hit5"]}/{n_pos} = {pct(counts["hybrid_hit5"], n_pos)}',
             f'1.5B Judge / QA 기준: TP={qa_matrix["TP"]} FN(잘못된 거부)={qa_matrix["FN"]} FP(잘못된 승인)={qa_matrix["FP"]} TN(정상 거부)={qa_matrix["TN"]}',
             f'1.5B Judge / 검토 근거 기준: TP={evidence_matrix["TP"]} FN={evidence_matrix["FN"]} FP={evidence_matrix["FP"]} TN={evidence_matrix["TN"]}; 보류 {evidence_labels["보류"]}건 제외',
             f'정상 거부율(45건 기준): {pct(qa_matrix["TN"],n_neg)}; 잘못된 거부율(90건 기준): {pct(qa_matrix["FN"],n_pos)}; 잘못된 승인율(45건 기준): {pct(qa_matrix["FP"],n_neg)}',
             f'근거 검토 레이블: Y={evidence_labels["Y"]}, N={evidence_labels["N"]}, 보류={evidence_labels["보류"]}',
             f'7B Judge: YES={verdicts["YES"]}, NO={verdicts["NO"]}, UNCERTAIN={verdicts["UNCERTAIN"]}',
             f'7B Judge YES 선택 rank 수={selected7}, 검토된 충분 근거 rank와 일치={matched7}, 검토된 충분 근거 rank 총수={support_count}')
        for key, title in (('metric_base','Base'),('metric_reranker','Reranker RAG v1'),('metric_hybrid','Hybrid RAG v2')):
            metric=raw[key]
            if metric:
                emit(f'{title}: EM {metric.get("exact_match_percent")}% / Token F1 {metric.get("token_f1_percent")}% / 답변 거부율 {metric.get("abstention_rate_percent")}% / False Refusal {metric.get("false_refusal_rate_percent")}% / 생성시간 {metric.get("generation_seconds")}초 / 최고 GPU {metric.get("peak_gpu_memory_gb")}GiB')
        m=raw['metric_retrieval']
        if m: emit(f'별도 Reranker 검색 평가: Hit@5={m.get("hit_at_5_percent")}% / MRR@10={m.get("mrr_at_10")} / nDCG@5={m.get("ndcg_at_5")}')
        for key,title in (('train_lora','LoRA'),('train_qlora','QLoRA')):
            tr=raw[key]
            if tr: emit(f'{title}: Epoch={tr.get("epochs")} / best_val_loss={tr.get("best_val_loss")} / time={tr.get("training_seconds")}초 / peak GPU={tr.get("peak_gpu_memory_gb")}GiB')
        if raw['e2e_metrics']: emit('실제 E2E 측정값:',json.dumps(raw['e2e_metrics'],ensure_ascii=False,indent=2))
        emit('', '2. 해석 주의사항',
             '- Gold 파일명 일치는 정답 출처 검색 성공이지, 검색된 청크에 정답 문장이 있는 것까지 보장하지 않는다.',
             '- QA 답변가능 여부(is_impossible)와 검색된 Top5의 근거 충분(Y/N/보류)은 다른 레이블이다.',
             '- 1.5B Judge의 기존 JSONL은 질문 단위 승인 여부만 기록했다. 문서별 원시 YES/NO와 선택 이유는 없다.',
             '- 7B Judge의 evidence_document_ranks는 별도 모델의 선택이다. 기존 1.5B Judge 선택으로 오해하면 안 된다.',
             '- 7B judge_reason과 근거 검토 review_rationale은 질문 전체에 대한 설명이며 문서별 독립 설명이 아니다.',
             '- RAG v1/v2 예측값은 Judge 미적용 생성 결과다. Judge가 승인한 뒤 실제 반환된 답변과 다를 수 있다.',
             '- 7B UNCERTAIN의 evidence_document_ranks는 충분 근거 승인 대신 부분 관련 참고로 표시한다.',
             '- 근거 검토 보류 23건은 확정 평가에서 제외한다.',
             '- E2E 결과가 있으면 이를 별도로 표기한다. 없으면 최종 승인 후 생성 결과는 미기록으로 둔다.',
             '', '3. 135개 질문 빠른 인덱스',
             '번호 | ID | QA가능 | Gold 최초순위 | 1.5B Judge | QA판정 | 근거레이블 | 7B판정 | RAGv1 F1')
        for i,c in enumerate(records,1):
            r,a,j,g=c['r'],c['a'],c['j'],c['g']
            grade=conf(c['is_qa'],a['predicted_answerable'])
            rank=(f'{c["ranks"][0]}위' if c['ranks'] else 'Top5 미검색') if c['is_qa'] else 'N/A'
            f1= f'{r["token_f1"]*100:.2f}%' if r.get('token_f1') is not None else 'N/A'
            emit(f'{i:03d} | {c["id"]} | {"YES" if c["is_qa"] else "NO"} | {rank} | {"YES" if a["predicted_answerable"] else "NO"} | {grade} | {g["evidence_label"]} | {j["judge_verdict"]} | {f1}')

        emit('', '4. 질문별 상세 분석', '='*110)
        for i,c in enumerate(records,1):
            id_,r,h,b,a,j,g,o,e=c['id'],c['r'],c['h'],c['b'],c['a'],c['j'],c['g'],c['o'],c['e']
            rr,hr,docs=c['ranks'],c['ranks_h'],r['retrieved_documents']
            goldset={namekey(n) for n in c['gold']}
            reviewed=set(g.get('evidence_document_ranks') or [])
            related=set(g.get('related_document_ranks') or [])
            picked7=set(j.get('evidence_document_ranks') or [])
            selected_e2e={doc['chunk_id'] for doc in (e.get('evidence_documents') or [])} if e else None
            e2e_docs=e.get('retrieved_documents') or [] if e else []
            e2e_same_docs=(len(e2e_docs)==len(docs) and all(
                x.get('chunk_id')==y.get('chunk_id') for x,y in zip(e2e_docs,docs))) if e else False
            selected_e2e_ranks=[k for k,doc in enumerate(e2e_docs,1) if selected_e2e is not None and doc['chunk_id'] in selected_e2e]
            n_verified = sum(1 for rank in selected_e2e_ranks if rank in reviewed) if e2e_same_docs else None
            n_gold_sel = sum(1 for rank in selected_e2e_ranks if namekey(e2e_docs[rank-1]['filename']) in goldset)
            matched_gold={namekey(x) for x in (docs[p-1]['filename'] for p in rr)}
            precision = len(matched_gold)/5 if c['is_qa'] else None
            recall = len(matched_gold)/len(goldset) if c['is_qa'] and goldset else None
            rr_score=1/rr[0] if rr else 0
            seen_gold=set()
            dcg=0.0
            for pos in rr:
                key=namekey(docs[pos-1]['filename'])
                if key not in seen_gold:
                    dcg+=1/math.log2(pos+1)
                    seen_gold.add(key)
            ideal=sum(1/math.log2(pos+1) for pos in range(1,min(5,len(goldset))+1))
            ndcg=dcg/ideal if ideal else None

            emit('', '='*110,f'[{i:03d}/135] {id_}', '='*110,
                 'Question:', txt(r['question']), '',
                 'Answer (정답):', txt(r['reference_answer']) if c['is_qa'] else '[원본 QA 답변 불가능 / 정답 없음]', '',
                 '답변이 가능한가:',
                 f'- QA: {"YES" if c["is_qa"] else "NO"} / Top5 근거 검토: {g["evidence_label"]}',
                 f'- 검토상태: {txt(g.get("review_status"))} / 검토출처: {txt(g.get("review_origin"))} / 확신도: {txt(g.get("confidence"))}',
                 f'- 검토 사유: {txt(g.get("review_rationale"))}', '',
                 '정답 문서와 내용:',
                 f'- QA 정답 출처 문서: {seq(c["gold"]) if c["is_qa"] else "N/A"}',
                 f'- Reranker v1 Gold Hit@5: {"YES" if rr else "NO" if c["is_qa"] else "N/A"}',
                 f'- Gold 검색된 모든 순위: {seq(rr)} / 첫 순위: {rr[0] if rr else "Top5 밖 또는 해당없음"}',
                 f'- 근거 검토의 충분한 문서 rank: {seq(g.get("evidence_document_ranks"))}',
                 f'- 근거 검토의 부분 관련 문서 rank: {seq(g.get("related_document_ranks"))}',
                 f'- 근거 검토 참고 파일: {seq(g.get("reference_document_filenames"))}',
                 f'- 근거 검토 참고 구절: {txt(g.get("reference_excerpt"))}')
            if c['gold_texts']:
                for fname, source_text in c['gold_texts']:
                    emit(f'[Gold 원문] {fname}:',source_text)
            elif c['is_qa']:
                emit('[Gold 원문 미확인: data/processed/val_clean.json을 찾지 못해 정답 원문 문서 내용은 생략. 검색 Top5 내 청크는 아래 포함]')
            emit('', '실제 생성한 답 (Reranker RAG v1, Judge 미적용):', txt(r['prediction']),
                 f'- Token F1={r.get("token_f1")}, EM={r.get("exact_match")}, 답변거부={r.get("abstained")}, 입력토큰={r.get("input_tokens")}',
                 'Base Model 생성답:', txt(b['prediction']),
                 f'- Base F1={b.get("token_f1")}, EM={b.get("exact_match")}',
                 'Hybrid RAG v2 생성답:', txt(h['prediction']),
                 f'- Hybrid F1={h.get("token_f1")}, EM={h.get("exact_match")}, Gold 첫순위={hr[0] if hr else "Top5 밖 또는 해당없음"}')
            if e:
                emit('실제 E2E 최종 답 (Judge 적용):',txt(e['prediction']),
                     f'- E2E 1.5B Judge 선택 순위: {seq(selected_e2e_ranks)} / Top5 중 선택률: {pct(len(selected_e2e_ranks),len(docs))}',
                     f'- 선택된 것 중 Gold 파일명 일치: {n_gold_sel}개 / 검토상 충분근거 rank와 일치: {n_verified if n_verified is not None else "N/A (Top5 검색 결과가 v1과 다름)"}',
                     f'- E2E 처리: evidence_seconds={e.get("evidence_seconds")}, generation_seconds={e.get("generation_seconds")}, GPU={e.get("peak_gpu_memory_gib")}')
                if not e2e_same_docs:
                    emit('※ E2E와 v1 검색 문서가 다르므로 E2E 자체 Top5를 별도 표기:')
                    for erank, edoc in enumerate(e2e_docs,1):
                        emit(f'  E2E Top{erank}: {edoc.get("filename")} / {edoc.get("chunk_id")} / Judge 선택={edoc.get("chunk_id") in selected_e2e}')
            else:
                emit('실제 E2E Judge 적용 후 답:', '[저장된 E2E predictions 파일이 없어 확인 불가]')
            emit('', 'Hit@5 / 정답 문서 위치 / 추가 검색 지표:',
                 f'- 전체 Reranker Gold Hit@5: {counts["reranker_hit5"]}/{n_pos} ({pct(counts["reranker_hit5"],n_pos)})',
                 f'- 이 질문: Hit@5={int(bool(rr)) if c["is_qa"] else "N/A"}; 처음 Gold 순위={rr[0] if rr else "N/A / Top5 미검색"}',
                 f'- 이 질문: Reciprocal Rank@5={rr_score:.4f}; Precision@5={precision}; Recall@5={recall}; nDCG@5={ndcg}')
            emit('', 'Top5 문서별 검색 근거 / 선택 근거 / 전체 청크 내용:')
            for rank,doc in enumerate(docs,1):
                qa_gold = namekey(doc.get('filename')) in goldset if c['is_qa'] else None
                is_7 = rank in picked7
                if j['judge_verdict']=='YES':
                    judge_note = 'YES의 근거로 지정' if is_7 else 'YES이지만 근거 rank에 포함되지 않음'
                elif j['judge_verdict']=='UNCERTAIN':
                    judge_note = '부분 관련으로 표시' if is_7 else 'UNCERTAIN이며 rank 지정 없음'
                else:
                    judge_note = 'NO 판정, 근거 미지정'
                review_note = ('충분한 근거' if rank in reviewed else
                               '관련 있으나 충분하지 않음' if rank in related else
                               '보류/미지정' if g['evidence_label']=='보류' else
                               '충분한 근거 또는 관련 문서로 지정되지 않음')
                if selected_e2e is None:
                    e2e_note='[1.5B 문서별 선택 결과 미기록]'
                elif not e2e_same_docs:
                    e2e_note='[E2E Top5가 RAG v1과 다름: 이 문서의 E2E 선택 여부 직접 대조 금지]'
                else:
                    e2e_note='YES(실제 E2E 선택)' if doc['chunk_id'] in selected_e2e else 'NO(실제 E2E 미선택)'
                emit('-'*100,f'Top {rank}: {doc.get("filename")}',
                     f'chunk_id={doc.get("chunk_id")}; score={doc.get("score")}',
                     f'원본 QA Gold 파일명 일치: {qa_gold if qa_gold is not None else "N/A"}',
                     f'1.5B E2E Judge 선택: {e2e_note}',
                     f'7B Judge rank 평가: {judge_note}',
                     f'근거 검토 분류: {review_note}',
                     f'선택/미선택 이유: 7B Judge 질문 전체 사유 = {txt(j.get("judge_reason"))}',
                     f'선택/미선택 이유: 근거 검토 질문 전체 사유 = {txt(g.get("review_rationale"))}',
                     '※ 위 두 사유는 문서별로 독립적으로 기록된 이유는 아님',
                     '검색 청크 전체 내용:',txt(doc.get('text')))
            qa_outcome=conf(c['is_qa'],bool(a['predicted_answerable']))
            evidence_outcome=conf(g['evidence_label']=='Y',bool(a['predicted_answerable'])) if g['evidence_label'] in ('Y','N') else '보류(미산입)'
            yes_selections=(j.get('evidence_document_ranks') or []) if j['judge_verdict']=='YES' else []
            agreement=[rank for rank in yes_selections if rank in reviewed]
            emit('', 'Evidence Judge / 정상 거부 / 잘못된 거부 / 잘못된 승인:',
                 f'1.5B 질문 단위 predicted_answerable: {a["predicted_answerable"]}',
                 f'1.5B QA 레이블 기준: {qa_outcome}',
                 f'1.5B 근거 검토 기준: {evidence_outcome}',
                 f'독립 Qwen 7B verdict={j["judge_verdict"]}, judge_version={j.get("judge_version")}, model={j.get("judge_model")}',
                 f'7B 판단 사유: {txt(j.get("judge_reason"))}',
                 f'7B 문서 지정 rank: {seq(j.get("evidence_document_ranks"))}',
                 f'7B 인용: {txt(j.get("evidence_quote"))}',
                 f'7B review_flags: {seq(j.get("review_flags"))}',
                 f'7B YES 선택 문서와 검토 충분근거의 rank 일치: {len(agreement)}/{len(yes_selections)}, 겹치는 순위 {seq(agreement)}',
                 '※ 1.5B와 7B는 별도의 판정 실행이다. 위 선택 근거를 1.5B의 설명으로 오인하면 안 된다.',
                 '[CASE END]')
        emit('', '5. 사용한 파일 경로')
        for key,rel in SOURCE.items():
            emit(f'{key}: {rel} | {"있음" if (repo/rel).is_file() else "미확인"}')
        emit('', '[보고서 종료]')
    return {**counts,'path': str(output),'bytes':output.stat().st_size,
            'qa_matrix':dict(qa_matrix),'evidence_matrix':dict(evidence_matrix),
            'review_labels':dict(evidence_labels),'judge7':dict(verdicts)}


def main():
    parser=argparse.ArgumentParser(description='IBM RAG 135건 근거/검색/생성 상세 TXT 보고서')
    parser.add_argument('--repo',type=Path,default=Path.cwd(),help='LLM_IBM_TechSupport 루트 디렉터리')
    parser.add_argument('--output',type=Path,default=None,help='보고서 TXT 생성 위치')
    opts=parser.parse_args()
    output=opts.output or opts.repo/'IBMTechSupport/docs/rag_case_analysis_135_full.txt'
    result=run(opts.repo,output.resolve())
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
