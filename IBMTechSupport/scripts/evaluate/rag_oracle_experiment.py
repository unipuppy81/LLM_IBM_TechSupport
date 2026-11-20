from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def valid_answered_rows(path: Path):
    by_id = {}
    with path.open(encoding="utf-8-sig") as fp:
        for line in fp:
            if line.strip():
                row = json.loads(line)
                if row["id"] in by_id:
                    raise ValueError(f"예측 파일 중복 ID {row['id']}")
                by_id[row["id"]] = row
    return by_id


def fname(s):
    return s.strip().replace("\\", "/").split("/")[-1].casefold()


def load_verified_oracle(path: Path, val: dict):
    approved = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        need = {"id", "filename", "chunk_id", "chunk_text", "verified"}
        if not need.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"필수 CSV 컬럼 누락: {need - set(reader.fieldnames or [])}")
        seen = set()
        for row in reader:
            if row["verified"].strip().casefold() not in {"yes", "y", "true", "1", "승인", "확인"}:
                continue
            rid = row["id"]
            if rid not in val:
                raise ValueError(f"검토 CSV의 ID가 검증셋에 없음: {rid}")
            if val[rid]["is_impossible"]:
                raise ValueError(f"답변 불가능 질문에 Oracle 근거를 승인함: {rid}")
            gold_files = {fname(c["filename"]) for c in val[rid].get("contexts", [])}
            if fname(row["filename"]) not in gold_files:
                raise ValueError(f"{rid}: 승인한 청크가 참조 출처와 일치하지 않음")
            if not row["chunk_text"].strip():
                raise ValueError(f"{rid}: 승인한 청크 텍스트가 비어 있음")
            key = (rid, row["chunk_id"])
            if key in seen:
                raise ValueError(f"중복 승인 청크: {key}")
            seen.add(key)
            approved[rid].append({
                "chunk_id": row["chunk_id"],
                "filename": row["filename"],
                "text": row["chunk_text"],
            })
    return approved


def main():
    p = argparse.ArgumentParser(description="수동 승인한 정답 청크를 직접 제공하는 Oracle 진단")
    p.add_argument("--root", type=Path, default=PROJECT_DIR)
    p.add_argument("--approved", type=Path, help="verified=YES인 oracle_candidates_review.csv")
    p.add_argument("--val", type=Path, help="val_clean.json")
    p.add_argument("--baseline", type=Path, help="기존 Judge ON predictions.jsonl (선택)")
    p.add_argument("--mode", choices=("direct", "judge", "both"), default="both")
    p.add_argument("--limit", type=int, default=0, help="질문 수, 0=승인 질문 전부")
    p.add_argument("--output", type=Path, help="새 결과 JSONL 경로")
    args = p.parse_args()
    if args.limit < 0:
        p.error("--limit은 0 이상")
    root = args.root.resolve()
    reviewed_path = args.approved or root / "results/rag_failure_analysis_v1/oracle_candidates_review.csv"
    val_path = args.val or root / "data/processed/val_clean.json"
    baseline_path = args.baseline or root / "evaluate_results/rag/final_e2e/reranker/predictions.jsonl"
    if not reviewed_path.is_file():
        raise FileNotFoundError(f"검토 대상 CSV 없음. 먼저 진단 --scan-corpus 실행: {reviewed_path}")
    if not val_path.is_file():
        raise FileNotFoundError(val_path)
    val_list = read_json(val_path)
    val = {row["id"]: row for row in val_list}
    if len(val) != len(val_list):
        raise ValueError("검증셋 ID 중복")
    approved = load_verified_oracle(reviewed_path, val)
    if not approved:
        raise ValueError("승인된 Oracle 청크가 없습니다. CSV의 verified 컬럼에 검토 후 YES를 입력하세요.")
    ids = [row["id"] for row in val_list if row["id"] in approved]
    if args.limit:
        ids = ids[:args.limit]
    output = args.output or root / "results" / ("oracle_verified_" + args.mode + (f"_smoke{args.limit}" if args.limit else "") + ".jsonl")
    if output.exists():
        raise FileExistsError(f"기존 결과 보호: {output}. 다른 --output 경로를 사용하세요")
    if output.resolve().is_relative_to((root / "data").resolve()) or output.resolve().is_relative_to((root / "models").resolve()):
        raise ValueError("결과는 data/, models/ 폴더에 저장하지 않습니다")

    from evaluate.common import ABSTAIN, token_f1
    import torch
    import yaml
    from load_base_model import load_model
    from rag_pipeline import generate_rag_answer
    from answerability_detection import select_evidence_documents

    if not torch.cuda.is_available():
        raise RuntimeError("Oracle Generation 실험은 CUDA GPU가 필요합니다")
    config_path = root / "configs/rag/v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline = valid_answered_rows(baseline_path) if baseline_path.is_file() else {}
    for rid in ids:
        if rid in baseline and baseline[rid]["question"] != val[rid]["question"]:
            raise ValueError(f"Baseline 질문 불일치: {rid}")
    print(f"Oracle 승인된 질문: {len(approved)} / 실험 대상: {len(ids)}")
    print("실험 모드:", args.mode)
    print("기존 Baseline 파일:", baseline_path if baseline else "사용 안 함")
    print("기존 검색 인덱스/모델/설정: 변경 없음")
    model, tokenizer, gen_config = load_model(config)
    modes = ("direct", "judge") if args.mode == "both" else (args.mode,)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored = defaultdict(list)
    with output.open("x", encoding="utf-8") as writer:
        for index, rid in enumerate(ids, 1):
            record = val[rid]
            oracle_docs = approved[rid]
            baseline_pred = baseline[rid]["prediction"] if rid in baseline else None
            for mode in modes:
                evidence = oracle_docs if mode == "direct" else select_evidence_documents(
                    model, tokenizer, record["question"], oracle_docs
                )
                if evidence:
                    pred, duration, tokens = generate_rag_answer(
                        model, tokenizer, gen_config, record["question"], evidence, config["prompt"]
                    )
                else:
                    pred, duration, tokens = ABSTAIN, 0.0, 0
                score = token_f1(pred, record["answer"])
                scored[mode].append(score)
                row = {
                    "id": rid, "mode": mode, "question": record["question"],
                    "reference_answer": record["answer"], "prediction": pred,
                    "baseline_prediction": baseline_pred,
                    "baseline_f1": token_f1(baseline_pred, record["answer"]) if baseline_pred is not None else None,
                    "token_f1": score, "refused": pred == ABSTAIN,
                    "approved_oracle_chunks": len(oracle_docs), "judge_selected_chunks": len(evidence),
                    "generation_seconds": duration, "input_tokens": tokens,
                }
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()
                print(f"[{index}/{len(ids)}] {rid} {mode}: F1={score*100:.2f}% / 선택 청크 {len(evidence)}")
    for mode in modes:
        scores = scored[mode]
        print(f"[{mode}] 승인 질문 {len(scores)}개 평균 F1: {sum(scores)/len(scores)*100:.2f}%")
    print("저장:", output.resolve())
    print("주의: 수동 승인된 부분 집합에 대한 Oracle 진단이며 전체 135개 최종 지표가 아닙니다")


if __name__ == "__main__":
    main()
