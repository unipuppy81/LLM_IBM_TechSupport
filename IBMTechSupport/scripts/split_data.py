import json
import random

from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

# 재현 가능한 데이터 분리
SEED = 42

# 데이터 분리 비율
RATIOS = { "train": 0.80, "val": 0.15, "test": 0.15 }

def question_key(question):
    # 공백과 대소문자 차이 제거하여 중복 검사
    return " ".join(question.casefold().split())

def split_data(records):
    rng = random.Random(SEED)

    # 동일한 질문은 하나의 그룹
    groups = defaultdict(list)

    for record in records:
        key = question_key(record["question"])
        groups[key].append(record)

    # 분리된 데이터 저장
    splits = { "train": [], "val": [], "test": [] }

    # 답변 가능 / 불가능 비율 유지하여 분리
    for impossible in (False, True):
        selected_groups = []

        for group in groups.values():
            # 그룹에서 가장 많은 정답 유형을 기준으로 분류
            impossible_count = sum(r["is_impossible"] for r in group)
            group_impossible = (impossible_count * 2 >= len(group))

            if group_impossible == impossible:
                selected_groups.append(group)

        # 그룹 랜덤
        rng.shuffle(selected_groups)
        selected_groups.sort(key=len, reverse=True)
        total = sum(len(g) for g in selected_groups)
        targets = { 
            name: total* ratio 
            for name, ratio in RATIOS.items() 
        }

        counts = { name: 0 for name in RATIOS }

        for group in selected_groups:
            # 목표 비율 대비 가장 부족한 데이터셋 선택
            destination = max(RATIOS, key=lambda name: (targets[name] - counts[name]) / targets[name])
            splits[destination].extend(group)
            counts[destination] += len(group)

    # 각 데이터셋 내부 순서 섞기
    for name in splits:
        rng.shuffle(splits[name])

    return splits

def validate_splits(records, splits):
    # 전체 ID 보존 및 중복 검증
    original_ids = [r["id"] for r in records]
    split_ids = [
        r["id"]
        for data in splits.values()
        for r in data
    ]

    assert len(split_ids) == len(original_ids)
    assert len(set(split_ids)) == len(split_ids)
    assert set(original_ids) == set(split_ids)

    # 데이터셋 사이 중복 질문 확인
    question_sets = {
        name: {
            question_key(r["question"])
            for r in data
        }
        for name, data in splits.items()
    }

    names = list(splits)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            overlap = question_sets[a] & question_sets[b]

            assert not overlap, (f"Question leakage: {a} / {b}")

    print("ID Leakage: 0")
    print("Question Leakage: 0")
    print("Split Validation: OK")



def main():
    # 전처리 데이터 로드
    input_path = DATA_DIR / "qa_clean.json"

    with open(input_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    # Train / Validation / Test 분리
    splits = split_data(records)

    print("[Dataset Split]")
    print("Total Records:", len(records))

    # 데이터셋별 통계
    for name, data in splits.items():
        total = len(data)
        answerable = sum(not r["is_impossible"] for r in data)
        impossible = sum(r["is_impossible"] for r in data)

        ratio = total / len(records) * 100

        print(
            f"{name}: {total} ({ratio:.2f}%) | "
            f"Answerable: {answerable} | "
            f"Impossible: {impossible}"
        )

    # 데이터 누출 검증
    validate_splits(records, splits)
    
    # 데이터 저장
    for name, data in splits.items():
        output_path = DATA_DIR / f"{name}.json"

        with open(
            output_path, "w", encoding="utf-8"
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

        print("Saved:", output_path)

    print("Dataset Split: OK")


if __name__ == "__main__":
    main()

