"""Evaluate one or more IDs using saved v3 retrieval; no retrieval or dataset edits.

Usage:
  python IBMTechSupport/scripts/debug/debug_per_document_judge.py
  python IBMTechSupport/scripts/debug/debug_per_document_judge.py --ids TRAIN_Q270 TRAIN_Q535 TRAIN_Q221
  python IBMTechSupport/scripts/debug/debug_per_document_judge.py --ids DEV_Q061

Each document is judged separately with the RELAXED prompt. Print the decision
for every chunk; show gold labels only AFTER the judge has finished.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from load_base_model import load_model
from rag_guardrails_per_document import judge_documents_individually

DEFAULT_IDS = ["TRAIN_Q270", "TRAIN_Q535", "TRAIN_Q221"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--predictions", type=Path, default=PROJECT_DIR / "evaluate_results/106/rag/v3/predictions.jsonl")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/rag/v4.yaml")
    args = parser.parse_args()

    if not args.predictions.is_file():
        parser.error(f"v3 검색 결과 파일이 없음: {args.predictions}")

    wanted = set(args.ids)
    found = {}
    with args.predictions.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if row.get("id") in wanted:
                    found[row["id"]] = row
    missing = wanted - found.keys()
    if missing:
        parser.error(f"QA ID 없음: {sorted(missing)}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    model, tokenizer, _ = load_model(cfg)
    top_k = cfg["retrieval"]["top_k"]
    retry_top_k = cfg["guardrails"]["retry_top_k"]

    for case_id in args.ids:
        row = found[case_id]
        result = judge_documents_individually(
            model=model, tokenizer=tokenizer, question=row["question"],
            retrieved_documents=row["retrieved_documents"],
            top_k=top_k, retry_top_k=retry_top_k,
        )
        print("\n" + "=" * 72)
        print("ID:", case_id)
        print("질문:", row["question"])
        print("검색 문서 수:", len(row["retrieved_documents"]))
        print("개별 판정:")
        for entry in result["judge_history"]:
            print(f"  [{entry['rank']:>2}] {entry['filename']} "
                  f"({entry['stage']}): {entry['decision']}")
        print("최종 근거 판정:", result["decision"])
        print("선택된 청크:", [d.get("chunk_id") for d in result["selected_documents"]])
        print("Judge 시간:", result["judge_seconds"], "초")
        # Diagnostics only; these are never passed to the model.
        print("원본 정답 여부 (진단용):", not row["is_impossible"])
        print("원본 정답 (진단용):", row.get("reference_answer"))


if __name__ == "__main__":
    main()
