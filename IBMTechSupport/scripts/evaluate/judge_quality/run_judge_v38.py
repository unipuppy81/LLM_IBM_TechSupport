from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
IBM_DIR = HERE.parents[2]
RESULT_DIR = IBM_DIR / "evaluate_results" / "106" / "judge_quality"
HUMAN = RESULT_DIR / "IBMTechSupport_judge_v36_human_review.json"
PREDICTIONS = IBM_DIR / "evaluate_results" / "106" / "rag" / "v4" / "predictions.jsonl"
SAMPLE = RESULT_DIR / "review_sample_v37.jsonl"
JUDGE_RESULT = RESULT_DIR / "rubric_judge_v38.jsonl"
REPORT = RESULT_DIR / "review_report_v38.json"


def run(*args: object) -> None:
    command = [sys.executable, *map(str, args)]
    print(f"\n[RUN] {Path(command[1]).name}", flush=True)
    subprocess.run(command, check=True, cwd=IBM_DIR.parent)


def main() -> None:
    required = (
        HERE / "judge_quality.py",
        HERE / "make_review_sample_v37.py",
        HERE / "review_report_v37.py",
        HUMAN,
        PREDICTIONS,
    )
    missing = [p for p in required if not p.is_file()]
    if missing:
        print("필수 파일이 없습니다:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        raise SystemExit(1)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run(HERE / "make_review_sample_v37.py",
        "--human", HUMAN,
        "--predictions", PREDICTIONS,
        "--output", SAMPLE)
    run(HERE / "judge_quality.py", "grade",
        "--predictions", SAMPLE,
        "--output", JUDGE_RESULT,
        "--limit", 20)
    run(HERE / "review_report_v37.py",
        "--human", HUMAN,
        "--judge", JUDGE_RESULT,
        "--output", REPORT)
    print(f"\n완료: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
