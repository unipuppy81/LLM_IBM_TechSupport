"""IBM TechSupport QA 중복 정제·유사도 분석·검토 통합 도구.

실행 순서 (최초 이전 작업 재사용):
  python IBMTechSupport/scripts/analyze_duplicates.py dedup
  python IBMTechSupport/scripts/analyze_duplicates.py merge
  python IBMTechSupport/scripts/analyze_duplicates.py status

검색을 처음부터 재생성할 때만:
  python IBMTechSupport/scripts/analyze_duplicates.py scan
"""

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODEL = ROOT / "models" / "all-MiniLM-L6-v2"
DEDUP_PATH = DATA / "qa_dedup.json"
REVIEW_PATH = DATA / "duplicate_review.csv"
TFIDF_THRESHOLD = 0.80
EMBED_THRESHOLD = 0.85
VALID_LABELS = {"SAME_CASE", "DIFFERENT_CASE", "UNSURE"}


def read_json(filename):
    with open(DATA / filename, encoding="utf-8") as file:
        return json.load(file)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def norm(text):
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


def normalize_value(value):
    if isinstance(value, str):
        return norm(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(item) for key, item in value.items()}
    return value


def content_key(record, normalized=False):
    content = {
        key: record[key]
        for key in ("question", "answer", "is_impossible", "contexts")
    }
    if normalized:
        content = normalize_value(content)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def pair_key(row):
    return tuple(sorted((row["id_a"], row["id_b"])))


def cves(text):
    return set(re.findall(r"\bCVE-\d{4}-\d{4,}\b", text, re.I))


def bool_value(value):
    return str(value).casefold() == "true"


def score(value):
    return float(value) if value not in (None, "") else None


def split_map():
    mapping = {}
    for name in ("train", "val", "test"):
        for record in read_json(f"{name}.json"):
            if record["id"] in mapping:
                raise ValueError(f"Duplicate split ID: {record['id']}")
            mapping[record["id"]] = name
    return mapping


def dedup():
    """ID 제외 완전 중복 → 안전한 정규화 중복 순으로 제거."""
    raw = read_json("qa_clean.json")
    split_ids = split_map()
    if len({r["id"] for r in raw}) != len(raw):
        raise ValueError("Duplicate IDs found")
    if set(split_ids) != {r["id"] for r in raw}:
        raise ValueError("Split IDs differ from qa_clean.json")

    seen_exact = {}
    seen_normal = {}
    cleaned = []
    dropped = Counter()

    for record in raw:
        exact = content_key(record)
        normalized = content_key(record, normalized=True)

        if exact in seen_exact:
            prior = seen_exact[exact]
            if split_ids[prior["id"]] != split_ids[record["id"]]:
                raise ValueError("Cross-split exact duplicate; review first")
            dropped["exact"] += 1
            print("Exact removed:", record["id"], "->", prior["id"])
            continue

        if normalized in seen_normal:
            prior = seen_normal[normalized]
            if split_ids[prior["id"]] != split_ids[record["id"]]:
                raise ValueError("Cross-split normalized duplicate; review first")
            # 정답/불가능 여부/근거 문서가 원문 그대로 같아야 자동 제거
            for field in ("answer", "is_impossible", "contexts"):
                if record[field] != prior[field]:
                    raise ValueError(f"Different {field}: {record['id']}")
            dropped["normalized"] += 1
            print("Normalized removed:", record["id"], "->", prior["id"])
            continue

        cleaned.append(record)
        seen_exact[exact] = record
        seen_normal[normalized] = record

    assert len(cleaned) + sum(dropped.values()) == len(raw)
    assert len({content_key(r, True) for r in cleaned}) == len(cleaned)
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DEDUP_PATH, "w", encoding="utf-8") as file:
        json.dump(cleaned, file, ensure_ascii=False, indent=2)

    print("[Dedup]")
    print("Before:", len(raw))
    print("Exact Removed:", dropped["exact"])
    print("Normalized Removed:", dropped["normalized"])
    print("After:", len(cleaned))
    print("Saved:", DEDUP_PATH)
    print("Dedup: OK")


def scan():
    """기존 검토 라벨을 유지하면서 TF-IDF/Embedding 후보를 재탐색."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoTokenizer

    records = read_json("qa_dedup.json")
    mapping = split_map()
    questions = [norm(record["question"]) for record in records]

    # 동일 질문이지만 답변이 달라 삭제하지 않은 그룹은 따로 집계
    same_question = defaultdict(list)
    for record, question in zip(records, questions):
        same_question[question].append(record)
    repeated = [group for group in same_question.values() if len(group) > 1]
    conflicted = [
        group for group in repeated
        if len({(norm(r["answer"]), r["is_impossible"]) for r in group}) > 1
    ]
    print("Same-Question Groups:", len(repeated))
    print("Different-Answer Groups:", len(conflicted))

    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(questions)
    tfidf_sim = cosine_similarity(tfidf_matrix)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for embedding scan")
    model = SentenceTransformer(str(MODEL), device="cuda", local_files_only=True)
    embeddings = model.encode(
        questions, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    embed_sim = np.asarray(embeddings) @ np.asarray(embeddings).T

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    max_length = model.max_seq_length
    old_max = tokenizer.model_max_length
    tokenizer.model_max_length = 1_000_000  # 길이 측정 경고만 억제
    try:
        full = tokenizer(questions, truncation=False, padding=False)["input_ids"]
    finally:
        tokenizer.model_max_length = old_max
    token_inputs = tokenizer(
        questions, truncation=True, max_length=max_length, padding=False
    )["input_ids"]

    # 재생성 때 사용자가 수동 수정한 판정은 기존 파일에서 복원
    old = {}
    if REVIEW_PATH.exists():
        old = {pair_key(r): r for r in read_csv(REVIEW_PATH)}

    rows = []
    for i, first in enumerate(records):
        for j in range(i + 1, len(records)):
            if questions[i] == questions[j]:
                continue
            ts = float(tfidf_sim[i, j])
            es = float(embed_sim[i, j])
            if ts < TFIDF_THRESHOLD and es < EMBED_THRESHOLD:
                continue
            second = records[j]
            id_a, id_b = first["id"], second["id"]
            prior = old.get(tuple(sorted((id_a, id_b))), {})
            shared = {
                c["filename"] for c in first["contexts"]
            } & {c["filename"] for c in second["contexts"]}
            rows.append({
                "id_a": id_a, "id_b": id_b,
                "split_a": mapping[id_a], "split_b": mapping[id_b],
                "cross_split": mapping[id_a] != mapping[id_b],
                "tfidf_similarity": round(ts, 6) if ts >= TFIDF_THRESHOLD else "",
                "embedding_similarity": round(es, 6) if es >= EMBED_THRESHOLD else "",
                "truncated_a": len(full[i]) > max_length,
                "truncated_b": len(full[j]) > max_length,
                "same_model_input": token_inputs[i] == token_inputs[j],
                "shared_contexts": len(shared),
                "answerability_conflict": first["is_impossible"] != second["is_impossible"],
                "question_a": first["question"], "question_b": second["question"],
                "answer_a": first["answer"], "answer_b": second["answer"],
                "priority": prior.get("priority", ""),
                "review_label": prior.get("review_label", ""),
                "review_notes": prior.get("review_notes", ""),
                "suggested_label": prior.get("suggested_label", ""),
                "suggestion_basis": prior.get("suggestion_basis", ""),
            })

    rows.sort(key=lambda r: (not bool_value(r["cross_split"]),
                              -(score(r["embedding_similarity"]) or 0)))
    write_csv(REVIEW_PATH, rows)
    print("[Scan]")
    print("Records:", len(records))
    print("Candidates:", len(rows))
    print("Cross-Split:", sum(bool_value(r["cross_split"]) for r in rows))
    print("Truncated Questions:", sum(len(x) > max_length for x in full))
    print("Saved:", REVIEW_PATH)
    print("Scan: OK")


def priority(row, original_keys, high_keys):
    key = pair_key(row)
    if not bool_value(row["cross_split"]):
        return "WITHIN_SPLIT"
    if key in original_keys:
        return "FIRST_39"
    if key in high_keys:
        return "HIGH"
    ts = score(row["tfidf_similarity"])
    es = score(row["embedding_similarity"])
    title_a = norm(row["question_a"].splitlines()[0])
    title_b = norm(row["question_b"].splitlines()[0])
    if (int(row["shared_contexts"] or 0) > 0
            or (es is not None and es >= 0.95)
            or (ts is not None and ts >= 0.90)
            or title_a == title_b):
        return "HIGH"
    if bool_value(row["answerability_conflict"]):
        return "MEDIUM"
    return "LOW"


def merge():
    """기존 판정 85쌍을 보존하고 MEDIUM에 검토 제안 추가."""
    source = REVIEW_PATH if REVIEW_PATH.exists() else DATA / "similarity_review.csv"
    if not source.exists():
        raise FileNotFoundError("duplicate_review.csv or similarity_review.csv")

    previous = {pair_key(r): r for r in read_csv(REVIEW_PATH)} if REVIEW_PATH.exists() else {}
    first_path = DATA / "priority_review_labeled.csv"
    high_path = DATA / "high_priority_review_labeled.csv"
    if not previous and (not first_path.exists() or not high_path.exists()):
        raise FileNotFoundError("Need both reviewed CSV files on first migration")

    first = {pair_key(r): r for r in read_csv(first_path)} if first_path.exists() else {}
    high = {pair_key(r): r for r in read_csv(high_path)} if high_path.exists() else {}
    if set(first) & set(high):
        raise ValueError("Overlapping labeled pairs")

    rows = []
    used = set()
    for row in read_csv(source):
        key = pair_key(row)
        if key in used:
            raise ValueError(f"Duplicate pair: {key}")
        used.add(key)
        prior = previous.get(key, {})
        imported = high.get(key) or first.get(key) or {}
        # 기존 통합 파일의 라벨이 비어 있으면 85쌍 원본 판정을 복원
        label = (prior.get("review_label") or imported.get("review_label") or "").strip()
        if label and label not in VALID_LABELS:
            raise ValueError(f"Invalid review label: {key} / {label}")

        computed = priority(row, set(first), set(high))
        # 통합 CSV만 남은 이후에도 우선순위를 그대로 보존
        p = prior.get("priority") or computed
        if p not in {"FIRST_39", "HIGH", "MEDIUM", "LOW", "WITHIN_SPLIT"}:
            raise ValueError(f"Unknown priority: {key} / {p}")
        ca, cb = cves(row["question_a"]), cves(row["question_b"])
        disjoint_cves = bool(ca and cb and ca.isdisjoint(cb))
        suggested = ""
        reason = ""
        if p == "MEDIUM" and not label:
            if disjoint_cves and int(row["shared_contexts"] or 0) == 0:
                suggested = "DIFFERENT_CASE"
                reason = "서로 다른 CVE; 공유 근거 문서 없음. 수동 확인 필요"
            else:
                suggested = "UNSURE"
                reason = "식별자만으로 동일 사례 판정 불가"

        rows.append({
            **{k: row.get(k, "") for k in (
                "id_a", "id_b", "split_a", "split_b", "cross_split",
                "tfidf_similarity", "embedding_similarity", "truncated_a",
                "truncated_b", "same_model_input", "shared_contexts",
                "answerability_conflict", "question_a", "question_b",
                "answer_a", "answer_b")},
            "priority": p,
            "review_label": label,
            "review_notes": prior.get("review_notes") or imported.get("review_notes", ""),
            "suggested_label": suggested or prior.get("suggested_label", ""),
            "suggestion_basis": reason or prior.get("suggestion_basis", ""),
        })

    if not (set(first) | set(high)).issubset(used):
        raise ValueError("Some previously reviewed pairs were not found")
    rows.sort(key=lambda r: (
        {"FIRST_39": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3,
         "WITHIN_SPLIT": 4}[r["priority"]],
        -(score(r["embedding_similarity"]) or 0),
    ))
    write_csv(REVIEW_PATH, rows)
    status()
    print("Merge: OK")


def status():
    if not REVIEW_PATH.exists():
        raise FileNotFoundError(REVIEW_PATH)
    rows = read_csv(REVIEW_PATH)
    cross = [r for r in rows if bool_value(r["cross_split"])]
    pc = Counter(r["priority"] for r in cross)
    lc = Counter(r["review_label"] for r in cross)
    medium = [r for r in cross if r["priority"] == "MEDIUM"]
    suggestions = Counter(r["suggested_label"] for r in medium if not r["review_label"])
    print("[Duplicate Review Status]")
    print("All Candidate Pairs:", len(rows))
    print("Cross-Split Pairs:", len(cross))
    print("Reviewed:", sum(bool(r["review_label"]) for r in cross))
    print("Label Counts:", dict(lc))
    print("Priority Counts:", dict(pc))
    print("MEDIUM Suggestions:", dict(suggestions))
    print("Saved:", REVIEW_PATH)
    print("Review Status: OK")


def conflicts():
    """삭제하지 않은 동일 질문·서로 다른 정답의 사례를 콘솔에서 확인."""
    records = read_json("qa_dedup.json")
    mapping = split_map()
    grouped = defaultdict(list)
    for record in records:
        grouped[norm(record["question"])].append(record)
    cases = [
        group for group in grouped.values()
        if len(group) > 1 and
        len({(norm(r["answer"]), r["is_impossible"]) for r in group}) > 1
    ]
    print("[Conflicting Answers]", len(cases), "groups")
    for index, group in enumerate(cases, 1):
        print(f"Group {index}:", group[0]["question"][:120].replace("\n", " "))
        for record in group:
            print(" ", record["id"], mapping[record["id"]],
                  "impossible=", record["is_impossible"],
                  "answer=", record["answer"][:110].replace("\n", " "))
    print("Conflicts: OK")



# 기존 train/val/test 파일은 그대로 두고 별도 clean Split 생성
CLEAN_SPLITS = ("train_clean", "val_clean", "test_clean")
SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_SEED = 42


def split_guard(row):
    """두 질문을 다른 Split에 두면 안 되는지 판단한다.

    주의: RISK_GROUP은 동일 사례 확정이 아니며 누출 방지용 보수적 묶음이다.
    """
    label = (row.get("review_label") or "").strip()
    if label == "DIFFERENT_CASE":
        return "SEPARATE"
    if label == "SAME_CASE":
        return "SAME_CASE"
    if label not in ("", "UNSURE"):
        raise ValueError(f"Invalid label: {label}")

    # CVE 번호가 서로 다르고 근거 문서도 공유하지 않으면 임의로 결합하지 않는다.
    ca, cb = cves(row["question_a"]), cves(row["question_b"])
    if ca and cb and ca.isdisjoint(cb) and int(row.get("shared_contexts") or 0) == 0:
        return "SEPARATE_CVE"
    return "RISK_GROUP"


class DisjointSet:
    def __init__(self, ids):
        self.parent = {item: item for item in ids}
        self.size = {item: 1 for item in ids}

    def find(self, item):
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def build_guard_groups(records, review_rows):
    """동일 질문/확정 동일 사례/미확정 위험 쌍을 한 Split으로 묶는다."""
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs in qa_dedup.json")
    ids_set = set(ids)
    dsu = DisjointSet(ids)

    # 완전히 같은 질문은 답변이 달라도 반드시 같은 Split에 두기
    seen_questions = {}
    for r in records:
        q = norm(r["question"])
        if q in seen_questions:
            dsu.union(r["id"], seen_questions[q])
        else:
            seen_questions[q] = r["id"]

    decisions = Counter()
    keys = set()
    for row in review_rows:
        key = pair_key(row)
        if key in keys:
            raise ValueError(f"Duplicate review pair: {key}")
        keys.add(key)
        if not set(key).issubset(ids_set):
            raise ValueError(f"Review pair absent from QA: {key}")
        decision = split_guard(row)
        decisions[decision] += 1
        if decision in {"SAME_CASE", "RISK_GROUP"}:
            dsu.union(*key)

    groups = defaultdict(list)
    for record in records:
        groups[dsu.find(record["id"])].append(record)
    return list(groups.values()), dsu, decisions


def allocate_group_splits(groups):
    """그룹을 통째로 유지하면서 70/15/15 및 답변 가능 비율에 근접 배분."""
    import random

    rng = random.Random(SPLIT_SEED)
    total = sum(len(group) for group in groups)
    impossible_total = sum(r["is_impossible"] for g in groups for r in g)
    answerable_total = total - impossible_total
    target_n = [total * ratio for ratio in SPLIT_RATIOS]
    target_i = [impossible_total * ratio for ratio in SPLIT_RATIOS]
    target_a = [answerable_total * ratio for ratio in SPLIT_RATIOS]

    result = {name: [] for name in CLEAN_SPLITS}
    counts = [0, 0, 0]
    impossible = [0, 0, 0]
    answerable = [0, 0, 0]

    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    for group in groups:
        n = len(group)
        i = sum(r["is_impossible"] for r in group)
        a = n - i
        costs = []
        for j in range(3):
            # 제곱 오차의 증가분: 현재 잔여 목표가 큰 Split 선호
            dn = 2 * n * (counts[j] - target_n[j]) + n * n
            di = 2 * i * (impossible[j] - target_i[j]) + i * i
            da = 2 * a * (answerable[j] - target_a[j]) + a * a
            # 목표 초과에 추가 페널티
            overflow = 4 * max(0, counts[j] + n - target_n[j]) ** 2
            costs.append(dn + 0.5 * (di + da) + overflow)
        j = min(range(3), key=lambda k: (costs[k], k))
        result[CLEAN_SPLITS[j]].extend(group)
        counts[j] += n
        impossible[j] += i
        answerable[j] += a

    # 고정된 시드로 각각의 순서도 재현 가능하게 섞기
    for data in result.values():
        rng.shuffle(data)
    return result


def resplit():
    """기존 평가 분할은 보존한 채 누출 위험을 고려한 새 Split 3개를 만든다."""
    if not REVIEW_PATH.exists():
        raise FileNotFoundError(REVIEW_PATH)
    records = read_json("qa_dedup.json")
    review = read_csv(REVIEW_PATH)
    groups, dsu, decisions = build_guard_groups(records, review)
    splits = allocate_group_splits(groups)

    ids = [r["id"] for data in splits.values() for r in data]
    if len(ids) != len(records) or len(set(ids)) != len(records):
        raise AssertionError("Missing or duplicated IDs in clean splits")

    mapping = {r["id"]: name for name, data in splits.items() for r in data}
    questions = defaultdict(set)
    for r in records:
        questions[norm(r["question"])].add(mapping[r["id"]])
    if any(len(sp) > 1 for sp in questions.values()):
        raise AssertionError("Normalized identical question leakage")

    guarded_cross = [pair_key(row) for row in review
                     if split_guard(row) in {"SAME_CASE", "RISK_GROUP"}
                     and mapping[row["id_a"]] != mapping[row["id_b"]]]
    if guarded_cross:
        raise AssertionError(f"Guarded cross-split pairs: {guarded_cross[:5]}")

    # 기존 train/val/test를 절대 덮어쓰지 않음
    for name, data in splits.items():
        destination = DATA / f"{name}.json"
        with open(destination, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    old = split_map()
    moved = sum(old[item] != mapping[item].removesuffix("_clean") for item in ids)
    print("[Leakage-Aware Resplit]")
    print("Records:", len(records))
    print("Groups:", len(groups))
    print("Largest Group:", max(len(g) for g in groups))
    print("Pair Decisions:", dict(decisions))
    for name, data in splits.items():
        print(f"{name}: {len(data)} ({len(data) / len(records) * 100:.2f}%) | "
              f"Answerable: {sum(not x['is_impossible'] for x in data)} | "
              f"Impossible: {sum(x['is_impossible'] for x in data)}")
    print("Moved from previous split:", moved)
    print("ID Leakage: 0")
    print("Normalized Question Leakage: 0")
    print("Guarded Pair Leakage: 0")
    print("Not-guaranteed distinct-case / CVE pairs:",
          decisions["SEPARATE"] + decisions["SEPARATE_CVE"])
    print("Original train/val/test: unchanged")
    print("Resplit: OK")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("dedup", "scan", "merge", "status", "conflicts", "resplit"))
    args = parser.parse_args()
    {
        "dedup": dedup, "scan": scan, "merge": merge,
        "status": status, "conflicts": conflicts, "resplit": resplit,
    }[args.command]()


if __name__ == "__main__":
    main()
