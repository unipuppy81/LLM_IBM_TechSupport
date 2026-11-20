r"""IBM RAG: Judge YES/NO logits as a score, compared with Reranker scores.

Run from the repository root (Windows PowerShell):
    python IBMTechSupport/scripts/evaluate/rag_judge_probability.py --limit 3
    python IBMTechSupport/scripts/evaluate/rag_judge_probability.py --approved "$env:USERPROFILE\Downloads\oracle_approved_12.csv"
    python IBMTechSupport/scripts/evaluate/rag_judge_probability.py --approved "$env:USERPROFILE\Downloads\oracle_approved_12.csv" --resume

Does NOT change retrieval, weights, baseline outputs or configuration.
Uses the *existing* answerability_detection.py system prompt, read via Python AST.
Computes P(YES | first answer token is YES or NO), not calibrated P(correct evidence).
The 135-item QA answerability labels do not prove any individual chunk is sufficient.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

ABSTAIN = "INSUFFICIENT_INFORMATION"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path):
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def name_key(s):
    return str(s).strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def tok(s):
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", s).casefold())


def contains_answer(chunk, answer, min_tokens=8):
    needle, haystack = tok(answer), tok(chunk)
    if len(needle) < min_tokens or len(needle) > len(haystack):
        return False
    first = needle[0]
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1)
               if haystack[i] == first)


def get_system_prompt(code_path):
    """Extract exactly the literal system_prompt assigned in detect_answerability."""
    tree = ast.parse(code_path.read_text(encoding="utf-8-sig"))
    target = next((n for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == "detect_answerability"), None)
    if target is None:
        raise ValueError("answerability_detection.py에 detect_answerability 함수 없음")
    for node in ast.walk(target):
        if isinstance(node, ast.Assign) and any(
            isinstance(x, ast.Name) and x.id == "system_prompt" for x in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise ValueError("Judge 시스템 프롬프트가 비문자열로 변경됨; 원본과 동일한 프롬프트 추출 불가") from exc
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Judge 시스템 프롬프트가 비어 있음")
            return value
    raise ValueError("Judge 시스템 프롬프트를 찾지 못함")


def keyed(rows, source):
    output = {}
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in output:
            raise ValueError(f"{source} ID 중복/오류: {rid!r}")
        output[rid] = row
    return output


def checked_data(val_path, baseline_path, limit):
    val_rows = load_json(val_path)
    pred_rows = load_jsonl(baseline_path)
    if len(val_rows) != 135 or len(pred_rows) != 135:
        raise ValueError(f"Baseline 135개 불일치: val={len(val_rows)} predictions={len(pred_rows)}")
    val, pred = keyed(val_rows, "val"), keyed(pred_rows, "baseline")
    if set(val) != set(pred):
        raise ValueError("Baseline과 검증 데이터의 ID가 다름")
    if sum(not r["is_impossible"] for r in val_rows) != 90:
        raise ValueError("답변 가능한 질문 90개가 아님")
    for rid, row in val.items():
        other = pred[rid]
        for field, predfield in (("question", "question"), ("answer", "reference_answer"),
                                 ("is_impossible", "is_impossible")):
            if row[field] != other.get(predfield):
                raise ValueError(f"{rid} 원본과 예측 {field} 불일치")
        docs = other.get("retrieved_documents")
        if not isinstance(docs, list) or len(docs) != 5:
            raise ValueError(f"{rid}: 저장된 Top5가 5개가 아님")
        for i, doc in enumerate(docs, 1):
            if not isinstance(doc.get("text"), str) or not doc["text"].strip():
                raise ValueError(f"{rid} {i}위: 청크 텍스트 없음")
            if not isinstance(doc.get("score"), (int, float)) or not math.isfinite(doc["score"]):
                raise ValueError(f"{rid} {i}위: Reranker score 없음/오류")
            if not doc.get("chunk_id"):
                raise ValueError(f"{rid} {i}위: chunk_id 없음")
    val_rows = val_rows[:limit] if limit else val_rows
    return val_rows, pred


def approved_map(path):
    if not path:
        return {}
    approved = defaultdict(set)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "chunk_id", "filename", "chunk_text", "verified"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"승인 CSV 컬럼 누락: {required - set(reader.fieldnames or [])}")
        for row in reader:
            if row["verified"].strip().casefold() not in {"yes", "y", "true", "1", "승인", "확인"}:
                continue
            key = (str(row["chunk_id"]), name_key(row["filename"]), row["chunk_text"].strip())
            approved[row["id"]].add(key)
    return dict(approved)


def selected_ids(row):
    return {d.get("chunk_id") for d in row.get("evidence_documents", [])}


def case_items(val_rows, pred, approvals):
    items = []
    counts = Counter()
    for val in val_rows:
        rid = val["id"]
        base = pred[rid]
        golds = {name_key(c["filename"]) for c in val.get("contexts", [])}
        selected = selected_ids(base)
        docs = base["retrieved_documents"]
        for rank, doc in enumerate(docs, 1):
            cid = str(doc["chunk_id"])
            fname = name_key(doc["filename"])
            tup = (cid, fname, doc["text"].strip())
            manually_positive = tup in approvals.get(rid, set())
            full_match = (not val["is_impossible"] and fname in golds and
                          contains_answer(doc["text"], val["answer"]))
            label = ("reviewed_positive" if manually_positive else
                     "weak_negative_impossible" if val["is_impossible"] else
                     "weak_full_reference_positive" if full_match else "unlabeled")
            if manually_positive and val["is_impossible"]:
                raise ValueError(f"{rid} 답변 불가능 질문이 승인된 청크")
            counts[label] += 1
            items.append({
                "id": rid, "question": val["question"], "reference_answer": val["answer"],
                "is_impossible": val["is_impossible"], "rank": rank,
                "chunk_id": cid, "filename": doc["filename"],
                "chunk_text": doc["text"], "reranker_score": float(doc["score"]),
                "baseline_judge_yes": cid in selected,
                "weak_full_reference": full_match,
                "manual_positive": manually_positive,
                "evidence_label": label,
            })
    seen = {(x["id"], x["chunk_id"], name_key(x["filename"]), x["chunk_text"].strip()) for x in items}
    processed_ids = {x["id"] for x in items}
    not_seen = [(rid, k) for rid, keys in approvals.items() if rid in processed_ids for k in keys if (rid, *k) not in seen]
    if not_seen:
        print(f"주의: 수동 승인 청크 {len(not_seen)}개가 현재 평가 Top5에 없음. 제외.", flush=True)
    return items, counts


def sha256file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def input_digest(val_path, baseline_path, judge_source, approved_path):
    names = [val_path, baseline_path, judge_source]
    if approved_path:
        names.append(approved_path)
    return hashlib.sha256(json.dumps([sha256file(p) for p in names]).encode()).hexdigest()


def make_prompt(question, chunk_text, system_prompt, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question:\n{question}\n\nRetrieved Documents:\n[Document 1]\n{chunk_text}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def check_token_ids(tokenizer):
    choices = {}
    for word in ("YES", "NO"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{word!r}가 단일 토큰이 아님: {ids}. 단일 forward 점수 비교 중단")
        choices[word] = ids[0]
    if choices["YES"] == choices["NO"]:
        raise ValueError("YES/NO 토큰 ID 동일")
    return choices


def single_forward(model, tokenizer, system_prompt, item, ids):
    import torch
    prompt = make_prompt(item["question"], item["chunk_text"], system_prompt, tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        # Qwen's causal-LM forward may allocate [sequence_length, vocab_size]
        # logits; project ONLY the final hidden state to avoid excessive GPU RAM.
        if not (hasattr(model, "model") and hasattr(model, "lm_head")):
            raise ValueError("Qwen model.model/lm_head을 찾을 수 없어 안전한 마지막 토큰 계산 불가")
        hidden = model.model(**encoded, use_cache=False, return_dict=True).last_hidden_state[:, -1:, :]
        logits = model.lm_head(hidden)[0, -1, :].float()
        two = torch.stack([logits[ids["YES"]], logits[ids["NO"]]])
        binary = torch.softmax(two, dim=0)
        full_log_denom = torch.logsumexp(logits, dim=0)
        yes_raw = torch.exp(logits[ids["YES"]] - full_log_denom)
        no_raw = torch.exp(logits[ids["NO"]] - full_log_denom)
        greedy = int(torch.argmax(logits).item())
        out = {
            "p_yes_binary": float(binary[0].item()),
            "p_no_binary": float(binary[1].item()),
            "p_yes_vocab": float(yes_raw.item()),
            "p_no_vocab": float(no_raw.item()),
            "greedy_first_token": tokenizer.decode([greedy]),
            "input_tokens": int(encoded["input_ids"].shape[1]),
        }
    del hidden, logits, encoded
    return out


def auc(labels, scores):
    """Rank-based ROC-AUC, ties handled by average ranks."""
    if len(labels) != len(scores):
        raise ValueError("AUC labels/scores length mismatch")
    p = sum(int(x) for x in labels)
    n = len(labels) - p
    if not p or not n:
        return None
    sorted_pairs = sorted(zip(scores, labels), key=lambda s: s[0])
    rank_sum = 0.0
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += sum(int(v) for _, v in sorted_pairs[i:j]) * avg_rank
        i = j
    return (rank_sum - p * (p + 1) / 2) / (p * n)


def auc_paired_bootstrap(qrows, iters=1000, seed=42):
    """Paired, stratified question-level bootstrap for Δ AUC (Judge-Reranker)."""
    rng = random.Random(seed)
    pos = [r for r in qrows if not r["is_impossible"]]
    neg = [r for r in qrows if r["is_impossible"]]
    diffs = []
    for _ in range(iters):
        sampled = rng.choices(pos, k=len(pos)) + rng.choices(neg, k=len(neg))
        labels = [not r["is_impossible"] for r in sampled]
        a = auc(labels, [r["max_p_yes"] for r in sampled])
        b = auc(labels, [r["max_reranker_score"] for r in sampled])
        diffs.append(a - b)
    diffs.sort()
    return [round(diffs[int(.025 * (iters - 1))], 4), round(diffs[int(.975 * (iters - 1))], 4)]


def threshold_curve(qrows, metric):
    values = sorted({r[metric] for r in qrows})
    if not values:
        return []
    # Include the two extremes, and every score boundary (including ties).
    thresholds = [math.nextafter(values[0], -math.inf)] + values + [math.nextafter(values[-1], math.inf)]
    out = []
    for t in thresholds:
        false_refusals = correct_refusals = allowed_answerable = allowed_impossible = 0
        for r in qrows:
            predicted_answerable = r[metric] >= t
            if not r["is_impossible"]:
                allowed_answerable += int(predicted_answerable)
                false_refusals += int(not predicted_answerable)
            else:
                allowed_impossible += int(predicted_answerable)
                correct_refusals += int(not predicted_answerable)
        out.append({
            "method": "judge_p_yes" if metric == "max_p_yes" else "reranker_top1_score",
            "threshold": t,
            "answerable_allowed": allowed_answerable,
            "false_refusals": false_refusals,
            "correct_refusals": correct_refusals,
            "impossible_allowed": allowed_impossible,
            "false_refusal_rate": round(false_refusals / (allowed_answerable + false_refusals), 5),
            "correct_refusal_rate": round(correct_refusals / (correct_refusals + allowed_impossible), 5),
        })
    return out


def write_csv(path, rows, columns):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(items, scored, output_dir, count_labels, expected_digest, include_approved):
    if len(items) != len(scored):
        raise ValueError("점수화 결과 수 불일치")
    q = defaultdict(list)
    raw = []
    for item, row in zip(items, scored):
        if item["id"] != row["id"] or item["rank"] != row["rank"] or item["chunk_id"] != row["chunk_id"]:
            raise ValueError("스코어 순서 불일치")
        item_short = {k: v for k, v in item.items() if k not in {"chunk_text", "question", "reference_answer"}}
        combined = {**item_short, **{k: row[k] for k in ("p_yes_binary", "p_no_binary", "p_yes_vocab", "p_no_vocab", "greedy_first_token", "input_tokens")}}
        raw.append(combined)
        q[item["id"]].append(combined)
    qrows = []
    for rid, docs in q.items():
        qrows.append({
            "id": rid,
            "is_impossible": docs[0]["is_impossible"],
            "max_p_yes": max(d["p_yes_binary"] for d in docs),
            "max_reranker_score": max(d["reranker_score"] for d in docs),
            "judge_threshold_05_allowed": any(d["p_yes_binary"] >= .5 for d in docs),
            "baseline_judge_allowed": any(d["baseline_judge_yes"] for d in docs),
            "n_reviewed_positive": sum(d["manual_positive"] for d in docs),
            "n_weak_positive": sum(d["weak_full_reference"] for d in docs),
        })
    labels = [not row["is_impossible"] for row in qrows]
    judge_auc = auc(labels, [r["max_p_yes"] for r in qrows])
    rr_auc = auc(labels, [r["max_reranker_score"] for r in qrows])
    ci = auc_paired_bootstrap(qrows) if len(qrows) >= 20 and any(labels) and not all(labels) else None
    weak = [d for d in raw if d["evidence_label"] in {"weak_full_reference_positive", "reviewed_positive", "weak_negative_impossible"}]
    wlab = [d["evidence_label"] != "weak_negative_impossible" for d in weak]
    approved = [d for d in raw if d["manual_positive"]]
    original = {
        "questions": len(qrows), "chunks_scored": len(raw),
        "question_label_judge_roc_auc": round(judge_auc, 4) if judge_auc is not None else None,
        "question_label_reranker_roc_auc": round(rr_auc, 4) if rr_auc is not None else None,
        "question_label_judge_minus_reranker_auc": round(judge_auc - rr_auc, 4) if judge_auc is not None and rr_auc is not None else None,
        "question_label_auc_difference_paired_95ci": ci,
        "question_label_warning": "원본 QA 가능/불가능 레이블은 검색된 문서의 실제 근거 충분성을 검증하지 않습니다. AUC는 대리 지표입니다.",
        "weak_labeled_chunk_count": len(weak),
        "weak_chunk_judge_auc": round(auc(wlab, [d["p_yes_binary"] for d in weak]), 4) if auc(wlab, [d["p_yes_binary"] for d in weak]) is not None else None,
        "weak_chunk_reranker_auc": round(auc(wlab, [d["reranker_score"] for d in weak]), 4) if auc(wlab, [d["reranker_score"] for d in weak]) is not None else None,
        "reviewed_positive_chunks_in_top5": len(approved),
        "reviewed_positive_p_yes": [{"id": d["id"], "p_yes": round(d["p_yes_binary"], 5), "baseline_judge_yes": d["baseline_judge_yes"]} for d in approved],
        "labels": dict(count_labels),
        "baseline_judge_questions_allowed": sum(r["baseline_judge_allowed"] for r in qrows),
        "baseline_false_refusals": sum(not r["is_impossible"] and not r["baseline_judge_allowed"] for r in qrows),
        "baseline_correct_refusals": sum(r["is_impossible"] and not r["baseline_judge_allowed"] for r in qrows),
        "original_yes_no_vs_p05_chunk_decision_differences": sum(d["baseline_judge_yes"] != (d["p_yes_binary"] >= .5) for d in raw),
        "p_yes_0_5_questions_allowed": sum(r["judge_threshold_05_allowed"] for r in qrows),
        "p_yes_0_5_false_refusals": sum(not r["is_impossible"] and not r["judge_threshold_05_allowed"] for r in qrows),
        "p_yes_0_5_correct_refusals": sum(r["is_impossible"] and not r["judge_threshold_05_allowed"] for r in qrows),
        "token_ids_warning": "P(YES)는 YES/NO 첫 토큰만을 대상으로 한 조건부 softmax이며 보정된 확률이 아닙니다.",
        "threshold_warning": "135개에서 선택한 최적 임곗값은 학습셋 과적합 가능성이 있으므로 별도 검증 필요. 최종 답변 F1은 새 임곗값으로 다시 생성해야 계산 가능.",
        "input_fingerprint": expected_digest,
        "approved_csv_loaded": bool(include_approved),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "chunk_csv": output_dir / "chunk_scores.csv",
        "question_csv": output_dir / "questions.csv",
        "curve_csv": output_dir / "threshold_curves.csv",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"기존 결과 보호: {path}")
    paths["summary"].write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(paths["chunk_csv"], raw, ["id", "rank", "chunk_id", "filename", "is_impossible", "evidence_label", "manual_positive", "weak_full_reference", "baseline_judge_yes", "reranker_score", "p_yes_binary", "p_no_binary", "p_yes_vocab", "p_no_vocab", "greedy_first_token", "input_tokens"])
    write_csv(paths["question_csv"], qrows, list(qrows[0]))
    curves = threshold_curve(qrows, "max_p_yes") + threshold_curve(qrows, "max_reranker_score")
    write_csv(paths["curve_csv"], curves, list(curves[0]))
    return original


def main(argv=None):
    p = argparse.ArgumentParser(description="IBM RAG Judge P(YES) vs Reranker 점수 진단")
    p.add_argument("--root", type=Path, default=PROJECT)
    p.add_argument("--approved", type=Path, help="수동 검증한 oracle_approved_12.csv (선택)")
    p.add_argument("--limit", type=int, default=0, help="처음 N개 질문만 실행 (0=전체 135개)")
    p.add_argument("--output-dir", type=Path, help="결과 폴더 (기본: results/rag_judge_probability_v1)")
    p.add_argument("--resume", action="store_true", help="중단된 chunk_scores.jsonl 이어서 점수화")
    args = p.parse_args(argv)
    if args.limit < 0 or args.limit > 135:
        p.error("--limit은 0~135여야 합니다")
    root = args.root.resolve()
    val_path = root / "data/processed/val_clean.json"
    baseline_path = root / "evaluate_results/rag/final_e2e/reranker/predictions.jsonl"
    judge_source = root / "scripts/answerability_detection.py"
    rag_cfg = root / "configs/rag/v1.yaml"
    output_dir = (args.output_dir.resolve() if args.output_dir else
                  root / "results" / "rag_judge_probability_v1" / (f"smoke_{args.limit}" if args.limit else "full_135"))
    jsonl_path = output_dir / "chunk_scores.jsonl"
    approvals = approved_map(args.approved)
    val_rows, pred = checked_data(val_path, baseline_path, args.limit)
    items, count_labels = case_items(val_rows, pred, approvals)
    judge_prompt = get_system_prompt(judge_source)
    fingerprint = input_digest(val_path, baseline_path, judge_source, args.approved)
    print(f"[Judge 첫 토큰 점수 분석] 질문={len(val_rows)}, 문서={len(items)}, 승인 청크={count_labels['reviewed_positive']}")
    print("기존 Judge 프롬프트를 원본 소스에서 추출하며, 모델은 generate가 아닌 단일 forward로 평가합니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists() and not args.resume:
        raise FileExistsError(f"기존 스코어 파일 보호: {jsonl_path} (이어 하려면 --resume)")
    existing = load_jsonl(jsonl_path) if jsonl_path.exists() else []
    if len(existing) > len(items):
        raise ValueError("저장된 스코어가 현재 입력보다 많음")
    for pos, (row, item) in enumerate(zip(existing, items)):
        if row.get("fingerprint") != fingerprint or row.get("id") != item["id"] or row.get("rank") != item["rank"] or row.get("chunk_id") != item["chunk_id"]:
            raise ValueError(f"저장된 {pos+1}번째 결과가 현재 입력과 다름")
    missing = items[len(existing):]
    if missing:
        import torch
        import yaml
        from load_base_model import load_model
        with rag_cfg.open(encoding="utf-8-sig") as fp:
            cfg = yaml.safe_load(fp)
        model, tokenizer, _ = load_model(cfg)
        ids = check_token_ids(tokenizer)
        print(f"YES 토큰: {ids['YES']} / NO 토큰: {ids['NO']} / Device: {model.device}")
        mode = "a" if existing else "x"
        with jsonl_path.open(mode, encoding="utf-8") as fp:
            for i, item in enumerate(missing, start=len(existing) + 1):
                sc = single_forward(model, tokenizer, judge_prompt, item, ids)
                row = {"id": item["id"], "rank": item["rank"], "chunk_id": item["chunk_id"], "fingerprint": fingerprint, **sc}
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                fp.flush()
                if i % 25 == 0 or i == len(items):
                    print(f"점수화 {i}/{len(items)}", flush=True)
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("저장된 점수 모두 재사용. 모델 로딩 생략.")
    if args.resume and (output_dir / "summary.json").exists():
        print("이미 분석이 완료된 폴더입니다. 기존 요약을 보존합니다:", output_dir / "summary.json")
        return
    scored = load_jsonl(jsonl_path)
    result = analyze(items, scored, output_dir, count_labels, fingerprint, args.approved)
    print("\n[분석 결과]")
    for key in ("questions", "chunks_scored", "question_label_judge_roc_auc", "question_label_reranker_roc_auc", "question_label_judge_minus_reranker_auc", "question_label_auc_difference_paired_95ci", "weak_chunk_judge_auc", "weak_chunk_reranker_auc", "baseline_false_refusals", "baseline_correct_refusals", "p_yes_0_5_false_refusals", "p_yes_0_5_correct_refusals", "reviewed_positive_chunks_in_top5", "original_yes_no_vs_p05_chunk_decision_differences"):
        print(f"{key}: {result[key]}")
    print("결과 저장:", output_dir)
    print("주의: 질문 레이블 기준 AUC는 대리 지표이며, 실제 청크 근거 판별력으로 단정할 수 없습니다.")


if __name__ == "__main__":
    main()
