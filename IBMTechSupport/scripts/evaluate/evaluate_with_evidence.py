import argparse
import json
from .common import measure, load_jsonl as read_jsonl
from pathlib import Path


def score_evidence(predictions, labels):
    if len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("예측 JSONL에 중복 ID가 있음")
    if len({r["id"] for r in labels}) != len(labels):
        raise ValueError("레이블 JSONL에 중복 ID가 있음")
    label_map = {r["id"]: r for r in labels}
    if set(label_map) != {r["id"] for r in predictions}:
        raise ValueError("예측 파일과 레이블 파일의 ID 집합이 다름")

    scored = []
    held = []
    for row in predictions:
        evidence = label_map[row["id"]]
        if row["question"] != evidence["question"]:
            raise ValueError(f"질문 불일치: {row['id']}")
        if evidence["evidence_label"] not in ("Y", "N"):
            held.append(row["id"])
            continue
        expected = evidence["evidence_label"] == "Y"
        if evidence["evidence_answerable"] is not expected:
            raise ValueError(f"근거 레이블 불일치: {row['id']}")
        predicted = row["predicted_answerable"]
        outcome = "TP" if predicted and expected else "FP" if predicted else "FN" if expected else "TN"
        scored.append({
            "id": row["id"],
            "question": row["question"],
            "predicted_answerable": predicted,
            "original_qa_answerable": not row["is_impossible"],
            "evidence_answerable": expected,
            "outcome": outcome,
            "review_status": evidence.get("review_status"),
            "review_rationale": evidence.get("review_rationale"),
            "support_document_filename": evidence.get("support_document_filename"),
        })

    return scored, held


def main():
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="검색 근거 레이블로 Answerability 재채점")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    labels = read_jsonl(args.labels)
    scored, held = score_evidence(predictions, labels)

    from .common import check_output
    check_output(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"예측: {len(predictions)}건 / 평가: {len(scored)}건 / 보류: {len(held)}건")
    print("[검색 근거 기준]", json.dumps(measure(scored, "evidence_answerable"), ensure_ascii=False))
    print("[동일 대상 원본 QA 기준]", json.dumps(measure(scored, "original_qa_answerable"), ensure_ascii=False))
    print("참고: 근거 레이블은 1차 검토 자료이며 확정된 정답 데이터는 아님")
    print("저장:", args.output)


if __name__ == "__main__":
    main()
