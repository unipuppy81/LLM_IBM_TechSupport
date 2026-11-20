"""IBM RAG: validate retrieved *chunks* against reference text, not just source filenames.

No models are loaded; no existing data/results are modified. Python 3.10+ stdlib.
Run from project root:
    python IBMTechSupport/scripts/evaluate/rag_gold_chunk_audit.py

Default inputs:
    data/processed/val_clean.json
    data/processed/corpus_chunks.jsonl
    evaluate_results/rag/final_e2e/reranker/predictions.jsonl
    results/rag_candidate_k_comparison_v1/cases_90.csv (optional)

Writes a NEW folder: results/rag_gold_chunk_audit_v1/

IMPORTANT: full-reference contiguous token matches are weak positive candidates,
NOT validated sufficient evidence. No match is NOT a negative label.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ABSTAIN = "INSUFFICIENT_INFORMATION"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def tokens(s: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", s).casefold())


def normalized_filename(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"유효하지 않은 문서 파일명: {s!r}")
    return s.strip().replace("\\", "/").rsplit("/", 1)[-1].casefold()


def full_ref_match(chunk_text: str, answer: str, min_tokens: int) -> bool:
    """Token-consecutive literal reference match (case/NFKC-insensitive)."""
    needle, haystack = tokens(answer), tokens(chunk_text)
    if len(needle) < min_tokens or len(needle) > len(haystack):
        return False
    first = needle[0]
    return any(haystack[i:i + len(needle)] == needle
               for i, word in enumerate(haystack[:len(haystack) - len(needle) + 1])
               if word == first)


def overlap_scores(chunk_text: str, answer: str) -> tuple[float, float]:
    """Lexical recall and F1 only for selecting manual-review candidates."""
    a, b = Counter(tokens(answer)), Counter(tokens(chunk_text))
    common = sum((a & b).values())
    na, nb = sum(a.values()), sum(b.values())
    recall = common / na if na else 0.0
    f1 = 2 * common / (na + nb) if na + nb else 0.0
    return recall, f1


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def map_by_id(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in result:
            raise ValueError(f"{name}: 누락·중복 ID: {rid!r}")
        result[rid] = row
    return result


def selected_ids(retrieved: list[dict], selected: list[dict], rid: str) -> set[str]:
    """Validate Judge evidence belongs to saved Top5, using chunk ID and content."""
    by_id = {}
    for doc in retrieved:
        cid = doc.get("chunk_id")
        if not isinstance(cid, str) or cid in by_id:
            raise ValueError(f"{rid}: Top5 chunk_id 없음/중복: {cid!r}")
        by_id[cid] = doc
    ids = set()
    for doc in selected:
        cid = doc.get("chunk_id")
        if cid in ids or cid not in by_id:
            raise ValueError(f"{rid}: Judge 선택 청크가 Top5에 없거나 중복: {cid!r}")
        original = by_id[cid]
        if (original.get("text") != doc.get("text") or
                normalized_filename(original["filename"]) != normalized_filename(doc["filename"])):
            raise ValueError(f"{rid}: Judge 선택 청크 내용 불일치: {cid}")
        ids.add(cid)
    return ids


def k_top5_ids(row: dict, k: int, rid: str) -> list[str]:
    field = f"k{k}_top5_chunk_ids"
    if field not in row or not isinstance(row[field], str):
        raise ValueError(f"{rid}: Candidate K 결과 필드 누락: {field}")
    ids = [s.strip() for s in row[field].split(";") if s.strip()]
    if len(ids) != 5 or len(set(ids)) != len(ids):
        raise ValueError(f"{rid}: {field}가 Top5 청크 ID 5개가 아닙니다")
    return ids


def existing_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"필수 입력 파일 없음: {path}")


def main():
    p = argparse.ArgumentParser(description="IBM RAG 정답 청크 일치 여부 진단 / 수동 검증 목록 생성")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="IBMTechSupport 폴더")
    p.add_argument("--val", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--k-cases", type=Path, help="Candidate K 결과 cases_90.csv; 없으면 기본 경로 자동 조회")
    p.add_argument("--output", type=Path, help="새 출력 폴더; 이미 있으면 중단")
    p.add_argument("--top-candidates", type=int, default=3)
    p.add_argument("--min-ref-tokens", type=int, default=8)
    args = p.parse_args()
    if args.top_candidates < 1 or args.min_ref_tokens < 1:
        p.error("--top-candidates, --min-ref-tokens 는 1 이상이어야 합니다")

    root = args.root.resolve()
    val_path = args.val or root / "data/processed/val_clean.json"
    baseline_path = args.baseline or root / "evaluate_results/rag/final_e2e/reranker/predictions.jsonl"
    corpus_path = args.corpus or root / "data/processed/corpus_chunks.jsonl"
    k_path = args.k_cases or root / "results/rag_candidate_k_comparison_v1/cases_90.csv"
    out = (args.output or root / "results/rag_gold_chunk_audit_v1").resolve()
    for path in (val_path, baseline_path, corpus_path):
        existing_file(path)
    for forbidden in (root / "data", root / "models"):
        if out == forbidden.resolve() or out.is_relative_to(forbidden.resolve()):
            raise ValueError("결과를 data/ 또는 models/ 아래 저장할 수 없습니다")
    if out.exists():
        raise FileExistsError(f"기존 결과 보호: {out} (새 --output 경로를 선택하세요)")

    val_list, baseline_list = read_json(val_path), read_jsonl(baseline_path)
    if len(val_list) != 135 or len(baseline_list) != 135:
        raise ValueError(f"135개 평가 자료가 아닙니다: val={len(val_list)}, baseline={len(baseline_list)}")
    val, base = map_by_id(val_list, "VAL"), map_by_id(baseline_list, "Baseline")
    if set(val) != set(base):
        raise ValueError("VAL/Baseline 질문 ID가 서로 다릅니다")
    answerable = [r for r in val_list if not r["is_impossible"]]
    if len(answerable) != 90:
        raise ValueError(f"답변 가능 질문 90개 예상, 실제 {len(answerable)}개")

    retrieved_by_q = {}
    chosen_by_q = {}
    gold_names = {}
    for record in val_list:
        rid, old = record["id"], base[record["id"]]
        if ((record["question"], record["answer"], record["is_impossible"])
                != (old.get("question"), old.get("reference_answer"), old.get("is_impossible"))):
            raise ValueError(f"평가 데이터 불일치: {rid}")
        retrieved = old.get("retrieved_documents")
        chosen = old.get("evidence_documents")
        if not isinstance(retrieved, list) or len(retrieved) != 5 or not isinstance(chosen, list):
            raise ValueError(f"검색 Top5/Judge 선택 청크 누락: {rid}")
        ids = [d.get("chunk_id") for d in retrieved]
        if any(not isinstance(cid, str) for cid in ids) or len(set(ids)) != 5:
            raise ValueError(f"검색 Top5 ID 누락/중복: {rid}")
        if not isinstance(old.get("prediction"), str):
            raise ValueError(f"생성 답변 없음: {rid}")
        retrieved_by_q[rid] = retrieved
        chosen_by_q[rid] = selected_ids(retrieved, chosen, rid)
        gold_names[rid] = {normalized_filename(c["filename"]) for c in record.get("contexts", [])}

    # Candidate K experiment output is optional, but if present must match baseline exactly.
    k_case_map = None
    if k_path.exists():
        k_case_list = read_csv(k_path)
        k_case_map = map_by_id(k_case_list, "CandidateK")
        if set(k_case_map) != {r["id"] for r in answerable}:
            raise ValueError("Candidate K 실험의 90개 질문이 Baseline과 일치하지 않습니다")
        for rec in answerable:
            rid = rec["id"]
            row = k_case_map[rid]
            if rec["question"] != row.get("question"):
                raise ValueError(f"Candidate K 질문 불일치: {rid}")
            expected = [d["chunk_id"] for d in retrieved_by_q[rid]]
            if k_top5_ids(row, 10, rid) != expected:
                raise ValueError(f"Candidate K의 K10 Top5와 Baseline이 다릅니다: {rid}")
    elif args.k_cases:
        raise FileNotFoundError(f"지정한 Candidate K 결과 파일 없음: {k_path}")

    # One-pass corpus scan: retain only validated gold-source chunks and K10/30/50 Top5 chunks.
    relevant_gold = set().union(*(gold_names[r["id"]] for r in answerable))
    watched_ids = {doc["chunk_id"] for docs in retrieved_by_q.values() for doc in docs}
    if k_case_map:
        watched_ids.update(cid for row in k_case_map.values()
                           for k in (10, 30, 50) for cid in k_top5_ids(row, k, row["id"]))
    gold_chunks: dict[str, list[dict]] = defaultdict(list)
    indexed_chunks: dict[str, dict] = {}
    corpus_read = 0
    with corpus_path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            corpus_read += 1
            cid, fn = chunk.get("chunk_id"), normalized_filename(chunk.get("filename"))
            if fn in relevant_gold:
                gold_chunks[fn].append(chunk)
            if cid in watched_ids:
                if cid in indexed_chunks:
                    raise ValueError(f"Corpus에 중복 chunk_id 있음: {cid}")
                indexed_chunks[cid] = chunk
    missing_ids = watched_ids - set(indexed_chunks)
    if missing_ids:
        raise ValueError(f"Baseline/K 실험 청크 ID가 Corpus에 없습니다: {sorted(missing_ids)[:5]}")
    for rid, docs in retrieved_by_q.items():
        for doc in docs:
            original = indexed_chunks[doc["chunk_id"]]
            if normalized_filename(original["filename"]) != normalized_filename(doc["filename"]):
                raise ValueError(f"Corpus/Baseline 파일명 불일치: {rid}, {doc['chunk_id']}")
            if original["text"] != doc["text"]:
                raise ValueError(f"Corpus/Baseline 청크 내용 불일치: {rid}, {doc['chunk_id']}")

    categories = Counter()
    result = []
    reviews = []
    k_details = []
    change_details = []
    manual_columns = ["id", "priority", "question", "reference_answer", "baseline_prediction",
                      "gold_source_filename", "chunk_id", "in_baseline_top5", "baseline_rank",
                      "judge_selected", "k30_in_top5", "k50_in_top5", "full_reference_match_weak",
                      "reference_token_recall", "reference_token_f1", "chunk_text", "sufficient_evidence",
                      "review_note"]
    for record in answerable:
        rid = record["id"]
        gold = gold_names[rid]
        docs = retrieved_by_q[rid]
        selected = chosen_by_q[rid]
        refs = [indexed_chunks[d["chunk_id"]] for d in docs]
        gold_docs = [d for d in refs if normalized_filename(d["filename"]) in gold]
        corpus_gold = [c for fn in gold for c in gold_chunks.get(fn, [])]
        ref_text = record["answer"]
        # These are weak positives only: zero is never a verified negative.
        found_top5 = any(full_ref_match(c["text"], ref_text, args.min_ref_tokens) for c in refs)
        found_gold_top5 = any(full_ref_match(c["text"], ref_text, args.min_ref_tokens) for c in gold_docs)
        found_gold_corpus = any(full_ref_match(c["text"], ref_text, args.min_ref_tokens) for c in corpus_gold)
        found_judge_selected = any(d["chunk_id"] in selected and full_ref_match(d["text"], ref_text, args.min_ref_tokens)
                                   for d in refs)
        hit_source = bool(gold_docs)
        judge_selected_gold = any(d["chunk_id"] in selected for d in gold_docs)
        abstained = base[rid]["prediction"] == ABSTAIN
        if hit_source and found_gold_top5:
            category = "GOLD_SOURCE_AND_REFERENCE_IN_TOP5"
        elif hit_source and found_gold_corpus:
            category = "GOLD_SOURCE_TOP5_BUT_REFERENCE_ONLY_ELSEWHERE"
        elif hit_source:
            category = "GOLD_SOURCE_TOP5_REFERENCE_NOT_VERBATIM"
        elif found_gold_corpus:
            category = "GOLD_SOURCE_MISSED_REFERENCE_IN_CORPUS"
        else:
            category = "GOLD_SOURCE_MISSED_REFERENCE_NOT_VERBATIM"
        categories[category] += 1
        if hit_source and found_gold_top5 and not found_judge_selected:
            priority = "P1_REVIEW_JUDGE"
        elif hit_source and found_gold_corpus and not found_gold_top5:
            priority = "P1_REVIEW_CHUNK_SELECTION"
        elif not hit_source and found_gold_corpus:
            priority = "P1_REVIEW_RETRIEVAL"
        elif not found_gold_corpus:
            priority = "P2_MANUAL_REFERENCE_CHECK"
        else:
            priority = "P2_REVIEW_GENERATION_OR_JUDGE"

        k_ids = {k: set(k_top5_ids(k_case_map[rid], k, rid)) if k_case_map else set()
                 for k in (30, 50)}
        score_candidates = []
        baseline_rank = {d["chunk_id"]: i for i, d in enumerate(docs, 1)}
        for c in corpus_gold:
            cid = c["chunk_id"]
            recall, lexical_f1 = overlap_scores(c["text"], ref_text)
            literal = full_ref_match(c["text"], ref_text, args.min_ref_tokens)
            score_candidates.append({
                "id": rid, "priority": priority, "question": record["question"],
                "reference_answer": ref_text, "baseline_prediction": base[rid]["prediction"],
                "gold_source_filename": c["filename"], "chunk_id": cid,
                "in_baseline_top5": cid in baseline_rank,
                "baseline_rank": baseline_rank.get(cid, ""),
                "judge_selected": cid in selected,
                "k30_in_top5": cid in k_ids[30] if k_case_map else "",
                "k50_in_top5": cid in k_ids[50] if k_case_map else "",
                "full_reference_match_weak": literal,
                "reference_token_recall": round(recall, 4),
                "reference_token_f1": round(lexical_f1, 4),
                "chunk_text": c["text"],
                "sufficient_evidence": "",  # fill YES / NO / UNSURE only after manual reading
                "review_note": "",
            })
        score_candidates.sort(key=lambda x: (not x["full_reference_match_weak"],
                                             not x["in_baseline_top5"],
                                             -x["reference_token_recall"],
                                             -x["reference_token_f1"], x["chunk_id"]))
        # Include all strong literal candidates up to a reasonable per-question limit,
        # plus up to N others so reference mismatch cases also get review options.
        literal_rows = [r for r in score_candidates if r["full_reference_match_weak"]][:8]
        remainder = [r for r in score_candidates if not r["full_reference_match_weak"]][:args.top_candidates]
        reviews.extend(literal_rows + remainder)
        result.append({
            "id": rid,
            "priority": priority,
            "category": category,
            "question": record["question"],
            "reference_answer": ref_text,
            "judge_prediction": base[rid]["prediction"],
            "judge_refused": not selected,
            "prediction_refused": abstained,
            "gold_filenames": ";".join(sorted(gold)),
            "gold_source_in_top5": hit_source,
            "gold_source_selected": judge_selected_gold,
            "gold_corpus_chunk_count": len(corpus_gold),
            "reference_verbatim_in_gold_corpus_weak": found_gold_corpus,
            "reference_verbatim_in_top5_weak": found_top5,
            "reference_verbatim_in_gold_top5_weak": found_gold_top5,
            "reference_verbatim_judge_selected_weak": found_judge_selected,
            "reference_tokens": len(tokens(ref_text)),
            "manual_evidence_status": "",  # user reviews later; never infer sufficiency from literal match
        })
        if k_case_map:
            case = k_case_map[rid]
            row = {"id": rid, "question": record["question"], "priority": priority,
                   "gold_source_in_k10": hit_source, "gold_reference_verbatim_in_k10_weak": found_gold_top5}
            for k in (30, 50):
                ids = k_ids[k]
                candidate_docs = [indexed_chunks[cid] for cid in k_top5_ids(case, k, rid)]
                k_source_hit = any(normalized_filename(d["filename"]) in gold for d in candidate_docs)
                k_full_ref_gold = any(normalized_filename(d["filename"]) in gold and
                                      full_ref_match(d["text"], ref_text, args.min_ref_tokens)
                                      for d in candidate_docs)
                row[f"gold_source_in_k{k}"] = k_source_hit
                row[f"gold_reference_verbatim_in_k{k}_weak"] = k_full_ref_gold
                row[f"k{k}_source_change"] = ("NEW_HIT" if not hit_source and k_source_hit
                                              else "LOST_HIT" if hit_source and not k_source_hit
                                              else "UNCHANGED")
            k_details.append(row)
            if any(row[f"k{k}_source_change"] != "UNCHANGED" for k in (30, 50)):
                change_details.append(row)

    impossible_generated = []
    for rec in val_list:
        if rec["is_impossible"] and base[rec["id"]]["prediction"] != ABSTAIN:
            rid = rec["id"]
            impossible_generated.append({
                "id": rid, "question": rec["question"],
                "prediction": base[rid]["prediction"],
                "retrieved_filenames": ";".join(d["filename"] for d in retrieved_by_q[rid]),
                "judge_selected_chunk_ids": ";".join(sorted(chosen_by_q[rid])),
                "manual_faithfulness": "", "review_note": "",
            })

    # Output only after all validations and computations have succeeded.
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "cases_90.csv", result, list(result[0]))
    write_csv(out / "gold_chunk_manual_review.csv", reviews, manual_columns)
    write_csv(out / "impossible_generated_review.csv", impossible_generated,
              ["id", "question", "prediction", "retrieved_filenames", "judge_selected_chunk_ids", "manual_faithfulness", "review_note"])
    if k_case_map:
        write_csv(out / "k_chunk_comparison_90.csv", k_details, list(k_details[0]))
        write_csv(out / "k_changed_review.csv", change_details, list(k_details[0]))
    weak_stats = {
        "questions_gold_source_in_top5": sum(r["gold_source_in_top5"] for r in result),
        "questions_gold_source_missing_top5": sum(not r["gold_source_in_top5"] for r in result),
        "questions_full_reference_in_gold_corpus": sum(r["reference_verbatim_in_gold_corpus_weak"] for r in result),
        "questions_full_reference_in_any_top5": sum(r["reference_verbatim_in_top5_weak"] for r in result),
        "questions_full_reference_in_gold_top5": sum(r["reference_verbatim_in_gold_top5_weak"] for r in result),
        "questions_full_reference_selected_by_judge": sum(r["reference_verbatim_judge_selected_weak"] for r in result),
        "questions_full_reference_in_gold_top5_but_not_judge_selected": sum(
            r["reference_verbatim_in_gold_top5_weak"] and not r["reference_verbatim_judge_selected_weak"]
            for r in result),
    }
    report = {
        "evaluation_total": len(val_list), "answerable": len(answerable),
        "corpus_chunks_scanned": corpus_read,
        "source_file_matching_is_not_chunk_evidence": True,
        "reference_match_is_weak_label_not_verified_evidence": True,
        "reference_no_match_is_not_negative": True,
        "reference_minimum_token_count": args.min_ref_tokens,
        "k_comparison_loaded": bool(k_case_map),
        "counts": dict(categories),
        "weak_diagnostics": weak_stats,
        "manual_review_candidates": len(reviews),
        "impossible_generated": len(impossible_generated),
        "k_changed_question_count": len(change_details),
        "notes": [
            "완전한 참조 정답이 청크에 연속 등장해야 weak match입니다. 부분 정답·의미 동등 표현은 검색되지 않을 수 있습니다.",
            "정답 파일명은 정답 청크를 의미하지 않습니다. 충분 근거 확인은 사람이 질문과 청크를 함께 읽어야 합니다.",
            "참조 정답을 활용한 청크 후보 랭킹은 평가/검토 편의를 위한 것으로 실서비스 Retriever 성능이 아닙니다.",
            "기존 모델과 인덱스, Baseline 결과는 변경하지 않습니다.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n[정답 근거 청크 진단]")
    print(f"검증셋 {len(val_list)}개 / 답변 가능 {len(answerable)}개 / Corpus 청크 {corpus_read}개")
    for label, num in categories.most_common():
        print(f"  {label}: {num}")
    print("정답 출처 파일명 Top5 성공:", weak_stats["questions_gold_source_in_top5"])
    print("정답 전문 연속 일치 (전체 정답 출처 코퍼스):", weak_stats["questions_full_reference_in_gold_corpus"])
    print("정답 전문 연속 일치 (최종 Top5):", weak_stats["questions_full_reference_in_any_top5"])
    print("정답 전문 연속 일치 (정답 출처 Top5):", weak_stats["questions_full_reference_in_gold_top5"])
    print("정답 전문 일치 청크 Judge 선택:", weak_stats["questions_full_reference_selected_by_judge"])
    print("정답 전문 일치 출처 청크 있으나 Judge 미선택:",
          weak_stats["questions_full_reference_in_gold_top5_but_not_judge_selected"])
    print("수동 검토 후보 청크:", len(reviews), "/ 답변 불가능 질문 중 생성:", len(impossible_generated))
    if k_case_map:
        print("K=30/50 변화 검토 질문:", len(change_details))
    print("저장:", out)
    print("주의: 문자 일치는 약한 양성 후보일 뿐, 실제 근거 충분성 판정이 아닙니다.")


if __name__ == "__main__":
    main()
