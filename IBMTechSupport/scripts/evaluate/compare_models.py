
import argparse
import json
from pathlib import Path

from .common import RESULT_DIR, check_output, load_jsonl, load_records, save_json, validate_prediction_rows


def compare(files=None, output=None):
    # 1. 비교할 평가 결과 파일 설정
    if files is None:
        files = {
            "Base Model": RESULT_DIR / "base/v1/metrics.json",
            "Hybrid RAG": RESULT_DIR / "rag/v2/metrics.json",
            "Reranker RAG": RESULT_DIR / "rag/v1/metrics.json",
        }

    # 2. 출력 파일 경로 검사
    check_output(output)

    # 3. 동일한 Validation 데이터셋 사용
    records = load_records("clean")

    compared = {}

    print("Model                 EM%      F1%  Correct Abstain%  False Refusal%")

    # 4. 각 모델 평가 결과 비교
    for name, path in files.items():
        path = Path(path)

        if not path.is_file():
            print(f"{name}: not evaluated (missing {path})")
            continue

        # 평가 지표 로드
        metrics = json.loads(path.read_text(encoding="utf-8-sig"))

        # 데이터셋 및 평가 건수 검증
        if (metrics.get("dataset") != "clean" or metrics.get("total") != len(records)):
            print(f"{name}: excluded (different evaluation dataset)")
            continue

        # 예측 결과 파일 경로
        prediction_path = path.parent / "predictions.jsonl"

        # 예측 결과와 실제 Validation 데이터 일치 여부 확인
        if prediction_path.is_file():
            try:
                predictions = load_jsonl(prediction_path)
                validate_prediction_rows(records, predictions)

            except (KeyError, ValueError) as exc:
                print(f"{name}: excluded ({exc})")
                continue

            verification = "predictions"

        else:
            verification = "metadata_only"
            print(f"{name}: no predictions; metadata only")

        # 비교 결과 저장
        compared[name] = {
            **metrics,
            "comparison_verification": verification,
        }

        # 주요 평가 지표 출력
        keys = (
            "exact_match_percent",
            "token_f1_percent",
            "abstention_rate_percent",
            "false_refusal_rate_percent",
        )

        values = [metrics.get(key) for key in keys]

        print(
            f"{name:<18}"
            + " ".join(
                f"{value:>12.2f}"
                if value is not None
                else f"{'N/A':>12}"
                for value in values
            )
        )

    print("EM/F1: answerable questions; abstention: respective label subsets.")
    print("Base Batch=4 and RAG Batch=1 timings are not directly comparable.")

    # 5. 비교 결과를 파일로 저장
    if output:
        save_json(output, compared)

    return compared


def main(argv=None):
    parser = argparse.ArgumentParser(description="Base LLM / Hybrid RAG / Reranker RAG 성능 비교")

    parser.add_argument("--base-metrics", type=Path)
    parser.add_argument("--hybrid-metrics", type=Path)
    parser.add_argument("--reranker-metrics", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)

    files = {
        name: path
        for name, path in (
            ("Base Model", args.base_metrics),
            ("Hybrid RAG", args.hybrid_metrics),
            ("Reranker RAG", args.reranker_metrics),
        )
        if path is not None
    }

    return compare(files=files or None, output=args.output)


if __name__ == "__main__":
    main()
