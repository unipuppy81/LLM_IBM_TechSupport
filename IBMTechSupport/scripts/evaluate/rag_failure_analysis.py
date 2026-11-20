from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

ABSTAIN = "INSUFFICIENT_INFORMATION"


def tok(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", unicodedata.normalize("NFKC", text).casefold())


def f1(prediction: str, answer: str) -> float:
    p, a = tok(prediction), tok(answer)
    if not p or not a:
        return float(p == a)
    overlap = sum((Counter(p) & Counter(a)).values())
    return 2 * overlap / (len(p) + len(a)) if overlap else 0.0


def fname(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"파일명 정보 오류: {s!r}")
    return s.strip().replace("\\", "/").split("/")[-1].casefold()


def includes_tokens(haystack: str, needle: str) -> bool:
    a, b = tok(haystack), tok(needle)
    if not a or not b or len(b) > len(a):
        return False
    # token-boundary contiguous match, not a semantic sufficiency label
    return any(a[i:i + len(b)] == b for i in range(len(a) - len(b) + 1))


def load_json(path: Path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def load_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def keyed(rows: list[dict], origin: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in result:
            raise ValueError(f"{origin}: 중복 또는 잘못된 ID: {rid!r}")
        result[rid] = row
    return result


def evidence_rank(documents: list[dict], selected: list[dict], rid: str) -> list[int]:
    """Match selected evidence using all fields provided; fail on ambiguous joins."""
    ranks = []
    for doc in selected:
        possible = list(range(len(documents)))
        for key in ("chunk_id", "row_id", "filename", "text"):
            if key not in doc or doc[key] is None:
                continue
            if key == "filename":
                possible = [i for i in possible if fname(documents[i].get(key)) == fname(doc[key])]
            else:
                possible = [i for i in possible if documents[i].get(key) == doc[key]]
        if len(possible) != 1:
            raise ValueError(f"{rid}: Judge 선택 문서 연결 실패/모호: candidates={possible}")
        ranks.append(possible[0] + 1)
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{rid}: Judge 중복 선택")
    return sorted(ranks)


def require_same_retrieval(on: dict, off: dict, rid: str):
    def signature(doc):
        return (doc.get("row_id"), doc.get("chunk_id"), fname(doc["filename"]), doc.get("text"))
    if list(map(signature, on)) != list(map(signature, off)):
        raise ValueError(f"{rid}: ON/OFF 검색 Top5가 같지 않아 직접 비교 불가능")


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    index = p * (len(sorted_values) - 1)
    floor = int(index)
    ceiling = min(len(sorted_values) - 1, floor + 1)
    return sorted_values[floor] * (ceiling - index) + sorted_values[ceiling] * (index - floor)


def paired_ci(values: list[float], seed=42, iterations=4000):
    """values = paired differences (percentage points). Resample QA pairs, not docs."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    distribution = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return [round(percentile(distribution, 0.025), 3), round(percentile(distribution, 0.975), 3)]


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("x", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scan_corpus_chunks(path: Path, val: dict[str, dict], out: Path):
    """Suggest *weak* oracle candidates via full-answer verbatim match in a gold source file."""
    wanted_files = set()
    for record in val.values():
        if not record["is_impossible"]:
            wanted_files.update(fname(c["filename"]) for c in record.get("contexts", []))
    grouped: dict[str, list[dict]] = {}
    scanned = 0
    with path.open(encoding="utf-8-sig") as fp:
        for line in fp:
            if not line.strip():
                continue
            chunk = json.loads(line)
            scanned += 1
            name = fname(chunk["filename"])
            if name in wanted_files:
                grouped.setdefault(name, []).append(chunk)
    candidates = []
    matched_ids = set()
    for rid, record in val.items():
        if record["is_impossible"]:
            continue
        answer = record["answer"]
        for context in record.get("contexts", []):
            for chunk in grouped.get(fname(context["filename"]), []):
                if len(tok(answer)) >= 8 and includes_tokens(chunk["text"], answer):
                    candidates.append({
                        "id": rid,
                        "question": record["question"],
                        "reference_answer": answer,
                        "filename": chunk["filename"],
                        "chunk_id": chunk["chunk_id"],
                        "chunk_text": chunk["text"],
                        "verified": "",  # reviewer writes YES only after reading question, answer, context
                        "review_note": "",
                    })
                    matched_ids.add(rid)
    colnames = ["id", "question", "reference_answer", "filename", "chunk_id", "chunk_text", "verified", "review_note"]
    write_csv(out / "oracle_candidates_review.csv", candidates, colnames)
    remaining = []
    for rid, record in val.items():
        if record["is_impossible"] or rid in matched_ids:
            continue
        remaining.append({
            "id": rid,
            "question": record["question"],
            "reference_answer": record["answer"],
            "gold_filenames": "; ".join(c["filename"] for c in record.get("contexts", [])),
            "reason": "검증된 후보 없음: 직접 청크를 찾아 추가 검증 필요",
        })
    write_csv(out / "oracle_needs_manual_search.csv", remaining,
              ["id", "question", "reference_answer", "gold_filenames", "reason"])
    return {"corpus_chunks_scanned": scanned, "weak_oracle_candidate_rows": len(candidates),
            "weak_oracle_candidate_questions": len(matched_ids),
            "remaining_answerable_questions": len(remaining)}


def main():
    parser = argparse.ArgumentParser(description="RAG 135개 평가 실패 분석: 기존 결과 재사용 / 모델 재실행 없음")
    base = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=base, help="IBMTechSupport 폴더")
    parser.add_argument("--val", type=Path, help="val_clean.json 경로")
    parser.add_argument("--on", type=Path, help="Judge ON predictions.jsonl 경로")
    parser.add_argument("--off", type=Path, help="선택: Judge OFF predictions_no_judge.jsonl")
    parser.add_argument("--out", type=Path, help="출력 폴더; 새 폴더만 허용")
    parser.add_argument("--scan-corpus", action="store_true", help="원본 청크에서 Oracle 후보 문장 찾기")
    parser.add_argument("--chunks", type=Path, help="corpus_chunks.jsonl 경로")
    parser.add_argument("--expect-count", type=int, default=135)
    args = parser.parse_args()
    root = args.root.resolve()
    val_path = args.val or root / "data/processed/val_clean.json"
    on_path = args.on or root / "evaluate_results/rag/final_e2e/reranker/predictions.jsonl"
    off_path = args.off or root / "evaluate_results/rag/judge_ablation/reranker/predictions_no_judge.jsonl"
    out = args.out or root / "results/rag_failure_analysis_v1"
    for required in (val_path, on_path):
        if not required.is_file():
            raise FileNotFoundError(f"필수 파일이 없습니다: {required}")
    if off_path.exists() and not off_path.is_file():
        raise ValueError(f"OFF 경로가 파일이 아닙니다: {off_path}")
    if args.scan_corpus:
        scan_path = args.chunks or root / "data/processed/corpus_chunks.jsonl"
        if not scan_path.is_file():
            raise FileNotFoundError(f"청크 파일이 없습니다: {scan_path}")
    if out.exists():
        raise FileExistsError(f"기존 결과 보호: {out} (새 --out 경로 지정)")
    if out.resolve().is_relative_to((root / "models").resolve()) or out.resolve().is_relative_to((root / "data").resolve()):
        raise ValueError("출력 폴더는 models/나 data/ 안에 둘 수 없습니다")

    val = keyed(load_json(val_path), "VAL")
    on = keyed(load_jsonl(on_path), "ON")
    off = keyed(load_jsonl(off_path), "OFF") if off_path.is_file() else None
    expected = args.expect_count
    if len(val) != expected or len(on) != expected:
        raise ValueError(f"135개 검증 실패: VAL={len(val)}, ON={len(on)} (기대 {expected})")
    if set(val) != set(on) or (off is not None and set(off) != set(on)):
        raise ValueError("VAL, ON, OFF 질문 ID가 다릅니다")
    cases, chunks, review = [], [], []
    for rid, record in val.items():
        on_row = on[rid]
        for label, row in (("ON", on_row), ("OFF", off[rid] if off else None)):
            if row is None:
                continue
            if ((record["question"], record["answer"], record["is_impossible"])
                    != (row.get("question"), row.get("reference_answer"), row.get("is_impossible"))):
                raise ValueError(f"{rid}: {label} 질문/정답/라벨이 val_clean과 다릅니다")
            if not isinstance(row.get("prediction"), str):
                raise ValueError(f"{rid}: {label} prediction 없음")
        docs = on_row.get("retrieved_documents")
        chosen = on_row.get("evidence_documents")
        if not isinstance(docs, list) or not docs or not isinstance(chosen, list):
            raise ValueError(f"{rid}: ON 검색/선택 문서 누락")
        if off:
            require_same_retrieval(docs, off[rid].get("retrieved_documents", []), rid)
        ranks = evidence_rank(docs, chosen, rid)
        gold = {fname(c["filename"]) for c in record.get("contexts", [])}
        ref = record["answer"] if not record["is_impossible"] else ""
        has_source = any(fname(doc["filename"]) in gold for doc in docs) if gold else False
        selected_source = any(fname(docs[i - 1]["filename"]) in gold for i in ranks) if gold else False
        exact_ranks = []
        for rank, doc in enumerate(docs, 1):
            if not isinstance(doc.get("text"), str):
                raise ValueError(f"{rid}: 검색 청크 텍스트 누락")
            is_weak = bool(ref) and len(tok(ref)) >= 8 and includes_tokens(doc["text"], ref)
            if is_weak:
                exact_ranks.append(rank)
            chunks.append({
                "id": rid, "is_impossible": record["is_impossible"], "rank": rank,
                "chunk_id": doc.get("chunk_id"), "filename": doc["filename"],
                "reranker_score": doc.get("score"), "gold_source_filename_match": bool(gold) and fname(doc["filename"]) in gold,
                "judge_selected": rank in ranks, "weak_full_reference_match": is_weak,
                "text": doc["text"],
            })
        abstained = on_row["prediction"] == ABSTAIN
        if record["is_impossible"]:
            category = "impossible_correct_refusal" if abstained else "impossible_generated"
        elif not has_source:
            category = "source_miss_refusal" if abstained else "source_miss_generated"
        elif not ranks:
            category = "source_hit_judge_rejected_all"
        elif not selected_source:
            category = "source_hit_judge_selected_other"
        else:
            category = "source_hit_judge_selected_gold"
        answer_f1 = None if record["is_impossible"] else f1(on_row["prediction"], record["answer"])
        off_f1 = None if off is None or record["is_impossible"] else f1(off[rid]["prediction"], record["answer"])
        cases.append({
            "id": rid, "category": category, "is_impossible": record["is_impossible"],
            "question": record["question"], "reference_answer": record["answer"],
            "prediction": on_row["prediction"], "off_prediction": off[rid]["prediction"] if off else "",
            "gold_filenames": "; ".join(sorted(gold)), "retrieved_filenames": "; ".join(d["filename"] for d in docs),
            "gold_source_in_top5": has_source, "selected_gold_source": selected_source,
            "judge_selected_ranks": ";".join(str(i) for i in ranks),
            "weak_full_reference_ranks": ";".join(str(i) for i in exact_ranks),
            "judge_refused": not ranks, "prediction_refused": abstained,
            "token_f1": answer_f1, "off_token_f1": off_f1,
        })
        if category not in {"source_hit_judge_selected_gold", "impossible_correct_refusal"}:
            reason = {
                "source_miss_refusal": "정답 출처 Top5 없음 + 답변 거부",
                "source_miss_generated": "정답 출처 Top5 없음 + 답변 생성",
                "source_hit_judge_rejected_all": "정답 출처 Top5 있음 + Judge 모두 NO (실제 근거 확인 필요)",
                "source_hit_judge_selected_other": "정답 출처 Top5 있음 + 다른 출처만 선택 (실제 근거 확인 필요)",
                "impossible_generated": "원본 정답 불가능 라벨인데 답변 생성 (근거 유효성 확인 필요)",
            }[category]
            review.append({
                "id": rid, "category": category, "review_reason": reason,
                "question": record["question"], "reference_answer": record["answer"],
                "prediction": on_row["prediction"], "gold_filenames": "; ".join(sorted(gold)),
                "retrieved_filenames": "; ".join(d["filename"] for d in docs),
                "judge_selected_ranks": ";".join(map(str, ranks)),
                "weak_full_reference_ranks": ";".join(map(str, exact_ranks)),
                "manual_label": "", "manual_note": "",
            })
    ans = [c for c in cases if not c["is_impossible"]]
    imp = [c for c in cases if c["is_impossible"]]
    cats = Counter(c["category"] for c in cases)
    if not ans or not imp:
        raise ValueError("답변 가능/불가능 질문이 모두 있어야 합니다")
    judge_refused_anywhere = sum(c["judge_refused"] for c in cases)
    selected_but_abstained = [c["id"] for c in cases if c["prediction_refused"] and not c["judge_refused"]]
    no_selected_but_generated = [c["id"] for c in cases if not c["prediction_refused"] and c["judge_refused"]]
    report = {
        "source": {"val": str(val_path), "judge_on": str(on_path), "judge_off": str(off_path) if off else None},
        "counts": {
            "total": len(cases), "answerable": len(ans), "impossible": len(imp),
            "gold_source_hit_top5": sum(c["gold_source_in_top5"] for c in ans),
            "gold_source_miss_top5": sum(not c["gold_source_in_top5"] for c in ans),
            "category_counts": dict(cats),
            "correct_refusals": sum(c["prediction_refused"] for c in imp),
            "false_refusals": sum(c["prediction_refused"] for c in ans),
            "judge_refused_question_count": judge_refused_anywhere,
            "selected_but_prediction_refused_ids": selected_but_abstained,
            "no_selected_but_prediction_generated_ids": no_selected_but_generated,
        },
        "generation": {
            "answerable_f1_percent": round(statistics.mean(c["token_f1"] for c in ans) * 100, 4),
            "answered_conditional_f1_percent": round(statistics.mean(c["token_f1"] for c in ans if not c["prediction_refused"]) * 100, 4)
            if any(not c["prediction_refused"] for c in ans) else None,
            "judge_off_f1_percent": round(statistics.mean(c["off_token_f1"] for c in ans) * 100, 4) if off else None,
            "on_minus_off_paired_f1_diff_percent_points_ci95": paired_ci(
                [(c["token_f1"] - c["off_token_f1"]) * 100 for c in ans]) if off else None,
        },
        "weak_labels": {
            "answerable_with_full_reference_in_retrieved_chunk": sum(bool(c["weak_full_reference_ranks"]) for c in ans),
            "full_reference_match_chunks": sum(bool(c["weak_full_reference_match"]) for c in chunks if not c["is_impossible"]),
            "full_reference_match_chunks_selected": sum(bool(c["weak_full_reference_match"]) and bool(c["judge_selected"])
                                                       for c in chunks if not c["is_impossible"]),
        },
        "limitations": [
            "정답 파일명 Hit@5는 실제로 답변에 충분한 문장을 찾았다는 증거가 아닙니다.",
            "정답 전문 문자열 일치는 약한 양성 후보 라벨입니다. 미일치는 음성 라벨이 아닙니다.",
            "원본 is_impossible은 검색 문서의 근거 충분성과 동치가 아닙니다.",
            "Judge ON/OFF는 검색 Top5가 완전히 동일한 경우에만 직접 비교합니다.",
            "Corpus 스캔은 --scan-corpus 옵션이 있을 때만 수행합니다.",
        ],
    }
    # Outputs are generated only after all input validation succeeds.
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "cases_135.csv", cases, list(cases[0]))
    write_csv(out / "retrieved_chunks_675.csv", chunks, list(chunks[0]))
    write_csv(out / "manual_review_queue.csv", review,
              ["id", "category", "review_reason", "question", "reference_answer", "prediction",
               "gold_filenames", "retrieved_filenames", "judge_selected_ranks", "weak_full_reference_ranks",
               "manual_label", "manual_note"])
    if args.scan_corpus:
        path = args.chunks or root / "data/processed/corpus_chunks.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"청크 파일이 없습니다: {path}")
        report["corpus_scan"] = scan_corpus_chunks(path, val, out)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = report["counts"]
    print("\n[RAG 평가 기반 진단]")
    print(f"질문: {metrics['total']} / 가능: {metrics['answerable']} / 불가능: {metrics['impossible']}")
    print(f"정답 출처 Top5: {metrics['gold_source_hit_top5']} / 누락: {metrics['gold_source_miss_top5']}")
    for label, value in sorted(cats.items()):
        print(f"  {label}: {value}")
    print(f"정상 거부: {metrics['correct_refusals']} / 잘못된 거부: {metrics['false_refusals']}")
    print(f"ON F1: {report['generation']['answerable_f1_percent']:.2f}%")
    if off:
        print(f"OFF F1: {report['generation']['judge_off_f1_percent']:.2f}%")
        print(f"paired 95% CI (ON-OFF %p): {report['generation']['on_minus_off_paired_f1_diff_percent_points_ci95']}")
    else:
        print(f"[알림] OFF 예측 파일 없음: {off_path}; ON 단독 평가 진행")
    if args.scan_corpus:
        print(f"Oracle 후보: {report['corpus_scan']['weak_oracle_candidate_rows']}개 청크 (수동 검증 필요)")
    print("저장:", out.resolve())


if __name__ == "__main__":
    main()
