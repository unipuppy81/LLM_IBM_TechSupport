"""Compare v3 strict vs experimental relaxed evidence judges on SAVED retrieval.

Run from repo root:
    python IBMTechSupport/scripts/debug/compare_judge_modes.py

Defaults: TRAIN_Q270 / TRAIN_Q535 / TRAIN_Q221 and strict judgments stored in
v3 predictions.jsonl. Strict 1-doc is evaluated live; relaxed 1/5/10 are live.
Add --rerun-strict to evaluate strict 5/10 live too.
This script does NOT run retrieval, answer generation, or write evaluation files.
The QA gold answer is printed only after all judge calls for diagnosis.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from load_base_model import load_model
from rag_guardrails import judge_evidence
from rag_guardrails_relaxed import judge_evidence_relaxed


DEFAULT_IDS = ("TRAIN_Q270", "TRAIN_Q535", "TRAIN_Q221")


def load_cases(path, qa_ids):
    if not path.is_file():
        raise FileNotFoundError(f"기존 v3 예측 파일이 없습니다: {path}")
    by_id = {}
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row["id"] in qa_ids:
                    by_id[row["id"]] = row
    missing = [qa_id for qa_id in qa_ids if qa_id not in by_id]
    if missing:
        raise ValueError(f"QA ID 미발견: {missing}")
    return [by_id[qa_id] for qa_id in qa_ids]


def saved_decisions(row):
    history = row.get("guardrails", {}).get("judge_history", [])
    return {int(entry["document_count"]): entry.get("decision", "-") for entry in history}


def run(ids, predictions, config_path, rerun_strict=False):
    rows = load_cases(predictions, ids)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    model, tokenizer, _ = load_model(config)

    for row in rows:
        docs = row["retrieved_documents"]
        question = row["question"]
        print("\n" + "=" * 72)
        print("ID:", row["id"])
        print("질문 (영문 원문):", question)
        print("1위 문서:", docs[0].get("filename") if docs else "없음")
        print("-" * 72)
        print(f"{'문서 수':>8} | {'기존 v3 STRICT':>16} | {'완화 RELAXED':>15}")
        print("-" * 72)
        previous = saved_decisions(row)
        for count in (1, 5, 10):
            subset = docs[:count]
            if not subset:
                continue
            if count == 1 or rerun_strict or count not in previous:
                strict = judge_evidence(model, tokenizer, question, subset)["decision"]
            else:
                strict = f"{previous[count]} (저장값)"
            relaxed = judge_evidence_relaxed(model, tokenizer, question, subset)["decision"]
            print(f"{len(subset):>8} | {strict:>16} | {relaxed:>15}", flush=True)
        print("-" * 72)
        print("기존 최종 판정:", row.get("guardrails", {}).get("status", "-"))
        print("정답 (진단 전용, Judge 입력에 넣지 않음):", row.get("reference_answer", ""))
        print("정답 있음:", not row.get("is_impossible", True))
    print("\n참고: YES 증가만으로 Judge 정확도가 개선됐다고 볼 수 없습니다.")
    print("오답/답변 불가능 사례까지 비교한 후 전체 평가를 진행하세요.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS))
    parser.add_argument("--predictions", type=Path, default=PROJECT_DIR / "evaluate_results/106/rag/v3/predictions.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/rag/v3.yaml")
    parser.add_argument("--rerun-strict", action="store_true")
    args = parser.parse_args(argv)
    run(args.ids, args.predictions, args.config, args.rerun_strict)


if __name__ == "__main__":
    main()
