from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def fname(s):
    return s.strip().replace("\\", "/").split("/")[-1].casefold()


def first_gold_rank(row_ids, chunks, gold):
    return next((rank for rank, index in enumerate(row_ids, 1)
                 if fname(chunks[int(index)]["filename"]) in gold), None)


def root_cause(bm, de, hy, limit=10):
    if bm is None and de is None:
        return "not_in_bm25_or_dense_top50"
    if hy is None:
        return "missing_after_rrf_or_doc_dedup"
    if hy > limit:
        return "outside_reranker_candidate_k10"
    return "reranker_top5_ranking_loss"


def main():
    p = argparse.ArgumentParser(description="정답 출처 Top5 검색 실패 30개를 단계별로 추적")
    p.add_argument("--root", type=Path, default=PROJECT_DIR)
    p.add_argument("--val", type=Path)
    p.add_argument("--on", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int, default=0, help="스모크 테스트용 질문 수 0=전체")
    args = p.parse_args()
    if args.limit < 0:
        p.error("--limit은 0 이상")
    root = args.root.resolve()
    val_path = args.val or root / "data/processed/val_clean.json"
    on_path = args.on or root / "evaluate_results/rag/final_e2e/reranker/predictions.jsonl"
    output = args.output or root / "results/rag_failure_analysis_v1/retrieval_trace.csv"
    for path in (val_path, on_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"기존 결과 보호: {output}")
    if output.resolve().is_relative_to((root / "models").resolve()) or output.resolve().is_relative_to((root / "data").resolve()):
        raise ValueError("출력은 models/ 또는 data/ 폴더 밖에 저장하세요")
    val_list = json.loads(val_path.read_text(encoding="utf-8-sig"))
    val = {row["id"]: row for row in val_list}
    on_list = read_jsonl(on_path)
    on = {row["id"]: row for row in on_list}
    if len(val) != 135 or len(on) != 135 or set(val) != set(on):
        raise ValueError("기존 135개 Baseline과 ID가 일치하는지 확인하세요")
    failures = []
    for record in val_list:
        if record["is_impossible"]:
            continue
        row = on[record["id"]]
        if row["question"] != record["question"] or row["reference_answer"] != record["answer"]:
            raise ValueError(f"질문 또는 정답 불일치: {record['id']}")
        gold = {fname(c["filename"]) for c in record["contexts"]}
        docs = row["retrieved_documents"]
        if not gold:
            raise ValueError(f"정답 출처 없음: {record['id']}")
        if not any(fname(d["filename"]) in gold for d in docs):
            failures.append(record)
    print(f"기존 검색 실패 질문: {len(failures)}개")
    if args.limit:
        failures = failures[:args.limit]
    questions = [record["question"] for record in failures]
    if not questions:
        print("분석할 질문이 없습니다")
        return

    # 기존 Baseline 설정과 일치하는지 모델 로딩 전에 검사
    from retrieval_config import load_retrieval_config
    expected_candidate = int(load_retrieval_config()["reranker"]["candidate_k"])
    if expected_candidate != 10:
        raise ValueError(f"기존 Baseline reranker.candidate_k=10과 현재 설정 불일치: {expected_candidate}")

    # Actual existing project search code. No index rebuilding or configuration changes.
    from dense_search import load_retriever
    from hybrid_search import load_bm25, search_all
    from reranker_search import load_reranker, rerank_candidates
    from document_utils import chunk_filename
    embedder, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)
    results = search_all(embedder, faiss_index, bm25_index, questions)
    reranker = load_reranker()
    reranked = rerank_candidates(reranker, questions, [r[2] for r in results], chunks)
    rows = []
    for record, result, reranked_docs in zip(failures, results, reranked):
        rid = record["id"]
        gold = {fname(c["filename"]) for c in record["contexts"]}
        bm, de, hy = (first_gold_rank(rank, chunks, gold) for rank in result[:3])
        k = 10  # inspect configured candidate_k below before describing as loss
        rerank_ids = [item["row_id"] for item in reranked_docs]
        retrieved_current = [str(chunks[x]["chunk_id"]) for x in rerank_ids[:5]]
        retrieved_saved = [str(doc["chunk_id"]) for doc in on[rid]["retrieved_documents"]]
        matched = retrieved_current == retrieved_saved
        top5_now = first_gold_rank(rerank_ids[:5], chunks, gold)
        row = {
            "id": rid,
            "gold_filenames": "; ".join(sorted(gold)),
            "bm25_first_gold_rank": bm,
            "dense_first_gold_rank": de,
            "hybrid_first_gold_rank": hy,
            "candidate_k": k,
            "reranker_first_gold_rank": first_gold_rank(rerank_ids, chunks, gold),
            "saved_vs_current_top5_match": matched,
            "current_reranker_gold_in_top5": top5_now is not None,
            "provisional_cause": root_cause(bm, de, hy, k) if matched else "REPRO_MISMATCH_DO_NOT_INTERPRET",
            "saved_top5_chunk_ids": ";".join(retrieved_saved),
            "current_top5_chunk_ids": ";".join(retrieved_current),
        }
        rows.append(row)
    if not rows:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reproduced = sum(x["saved_vs_current_top5_match"] for x in rows)
    counts = Counter(x["provisional_cause"] for x in rows)
    print(f"기존 Top5 재현: {reproduced}/{len(rows)} (불일치 건은 원인 분류 불가)")
    for cause, number in sorted(counts.items()):
        print(f"  {cause}: {number}")
    print("결과:", output.resolve())
    print("주의: 출처 파일명 기준 위치이며 정답을 포함한 청크 기준 결과는 아닙니다")


if __name__ == "__main__":
    main()
