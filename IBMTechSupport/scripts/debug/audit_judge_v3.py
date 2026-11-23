"""Judge-only audit for IBM TechQA RAG v3 (no LLM inference required).

Run from the LLM_IBM_TechSupport project root:
    python IBMTechSupport/scripts/debug/audit_judge_v3.py

Reads the frozen v3 predictions and val_106.json, without changing either.
The QA answerability label is a *proxy*, not an independently verified
retrieved-evidence label. Gold file-name matches are not proof that the
retrieved chunk contains sufficient answer evidence.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET = PROJECT_DIR / "data/processed/val_106.json"
PREDICTIONS = PROJECT_DIR / "evaluate_results/106/rag/v3/predictions.jsonl"
OUTPUT = PROJECT_DIR / "evaluate_results/106/judge"


def filename_key(value: str) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percent(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 2) if whole else None


def main() -> None:
    for path in (DATASET, PREDICTIONS):
        if not path.is_file():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    records = json.loads(DATASET.read_text(encoding="utf-8-sig"))
    predictions = read_jsonl(PREDICTIONS)
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("predictions.jsonl에서 QA ID 중복 발견")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("val_106.json에서 QA ID 중복 발견")
    if set(by_id) != {row["id"] for row in records}:
        raise ValueError("예측 파일과 데이터셋의 QA ID가 일치하지 않습니다")

    summary = Counter()
    audit = []
    failures = []

    for qa in records:
        result = by_id[qa["id"]]
        if qa["question"] != result["question"]:
            raise ValueError(f"질문 내용이 불일치합니다: {qa['id']}")
        if qa["is_impossible"] != result["is_impossible"]:
            raise ValueError(f"답변 가능 레이블이 불일치합니다: {qa['id']}")

        guard = result.get("guardrails", {})
        judge_history = guard.get("judge_history", [])
        if not judge_history:
            raise ValueError(f"Judge 판정 기록이 없습니다: {qa['id']}")
        for history in judge_history:
            if history.get("decision") not in ("YES", "NO", "UNCERTAIN"):
                raise ValueError(f"알 수 없는 Judge 판정: {qa['id']} / {history}")

        # This is the *pre-generation* answerability decision, not final status.
        judge_yes = any(h["decision"] == "YES" for h in judge_history)
        label_yes = not qa["is_impossible"]
        if label_yes and judge_yes:
            case = "TP"
        elif label_yes and not judge_yes:
            case = "FN"
        elif not label_yes and judge_yes:
            case = "FP"
        else:
            case = "TN"
        summary[case] += 1

        documents = result.get("retrieved_documents", [])
        gold_names = {filename_key(c.get("filename", "")) for c in qa.get("contexts", []) if isinstance(c, dict) and c.get("filename")}
        gold_matches = [(rank, doc) for rank, doc in enumerate(documents, 1) if filename_key(doc.get("filename", "")) in gold_names]
        match_ranks = [rank for rank, _ in gold_matches]
        hit5 = any(rank <= 5 for rank in match_ranks)
        hit10 = any(rank <= 10 for rank in match_ranks)
        summary["gold_top5" if hit5 else "gold_not_top5"] += 1
        if case == "FN":
            summary["fn_gold_top5" if hit5 else "fn_no_gold_top5"] += 1
            summary["fn_gold_top10" if hit10 else "fn_no_gold_top10"] += 1

        initial = next((h["decision"] for h in judge_history if h.get("stage") == "top_k"), "")
        expanded = next((h["decision"] for h in judge_history if h.get("stage") == "expanded_k"), "")
        if initial == "YES":
            summary["judge_initial_yes"] += 1
        if expanded == "YES":
            summary["judge_expanded_yes"] += 1

        if case == "FN" and hit5:
            priority = "높음: Gold 문서 상위 5개 + Judge NO"
        elif case == "FN" and hit10:
            priority = "높음: Gold 문서 6~10위 + Judge NO"
        elif case == "FN":
            priority = "검색 실패 가능: Gold 문서 미발견"
        elif case == "FP":
            priority = "확인: QA 답변 불가능이나 Judge YES"
        else:
            priority = "일반"

        matched_snippet = " | ".join(
            f"[{rank}] {doc.get('filename', '')}: {str(doc.get('text', '')).replace(chr(10), ' ')[:450]}"
            for rank, doc in gold_matches[:2]
        )
        item = {
            "id": qa["id"],
            "priority": priority,
            "confusion": case,
            "qa_answerable": label_yes,
            "judge_answerable": judge_yes,
            "top5_judge": initial,
            "expanded10_judge": expanded,
            "guardrail_status": guard.get("status", ""),
            "gold_file_ranks": ",".join(map(str, match_ranks)),
            "gold_file_top5": hit5,
            "gold_file_top10": hit10,
            "question": qa["question"],
            "reference_answer": qa.get("answer", ""),
            "matched_gold_chunk_preview": matched_snippet,
        }
        audit.append(item)
        if case in ("FN", "FP"):
            failures.append(item)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(audit[0])
    for name, data in (("judge_audit_106.csv", audit), ("judge_errors_106.csv", failures)):
        path = OUTPUT / name
        if path.exists():
            raise FileExistsError(f"기존 분석 결과 보존을 위해 덮어쓰지 않습니다: {path}")
        with path.open("x", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    tp, fp, tn, fn = (summary[key] for key in ("TP", "FP", "TN", "FN"))
    summary_obj = {
        "warning": "QA 레이블 기반 대리 평가입니다. 실제 검색 청크의 근거 충분성 독립 라벨이 아닙니다.",
        "total": len(audit),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "judge_precision_proxy_percent": percent(tp, tp + fp),
        "judge_recall_proxy_percent": percent(tp, tp + fn),
        "judge_specificity_proxy_percent": percent(tn, tn + fp),
        "judge_first5_yes": summary["judge_initial_yes"],
        "judge_expanded10_yes": summary["judge_expanded_yes"],
        "false_negatives_gold_file_top5": summary["fn_gold_top5"],
        "false_negatives_gold_file_top10": summary["fn_gold_top10"],
        "false_negatives_no_gold_file_top10": summary["fn_no_gold_top10"],
    }
    metrics_path = OUTPUT / "judge_summary_106.json"
    if metrics_path.exists():
        raise FileExistsError(f"기존 분석 결과 보존을 위해 덮어쓰지 않습니다: {metrics_path}")
    metrics_path.write_text(json.dumps(summary_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary_obj, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUTPUT}")
    print("주의: Gold 문서의 파일명 일치만으로 해당 청크가 질문에 답할 수 있다고 확정할 수는 없습니다.")


if __name__ == "__main__":
    main()
