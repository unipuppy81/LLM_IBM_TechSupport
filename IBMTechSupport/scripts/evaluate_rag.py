import argparse
import json
import time

import torch

from dense_search import RESULT_DIR, VAL_PATH, load_retriever, chunk_filename
from hybrid_search import load_bm25, search_all
from reranker_search import load_reranker, rerank_candidates, get_chunk_text
from rag_pipeline import generate_rag_answer
from evaluate_base_model import load_model, load_jsonl, save_jsonl, score_predictions


# region 경로 및 평가 데이터

BASE_METRICS_PATH = RESULT_DIR / "base_val_clean_b4_metrics.json"


def load_records(limit):
    with VAL_PATH.open(encoding="utf-8") as file:
        records = json.load(file)

    if limit:
        records = records[:limit]

    if not records:
        raise ValueError("평가 데이터가 비어 있습니다.")

    return records


def result_paths(method, top_k, limit):
    name = f"rag_val_clean_{method}_k{top_k}"

    if limit:
        name += f"_smoke{limit}"

    return (
        RESULT_DIR / f"{name}_predictions.jsonl",
        RESULT_DIR / f"{name}_metrics.json",
    )

# endregion


# region Hybrid / Reranker 문서 검색

def retrieve_all(questions, method, top_k):
    # 기존 임베딩 모델, FAISS, BM25 인덱스 로드
    embedding_model, faiss_index, chunks = load_retriever()
    bm25_index = load_bm25(chunks)

    # Reranker 방식이면 모델을 한 번만 로드
    reranker = load_reranker() if method == "reranker" else None

    start = time.perf_counter()

    # 모든 질문에 대해 Hybrid Search 수행
    results = search_all(embedding_model, faiss_index, bm25_index, questions)

    hybrid_rankings = [
        result[2]
        for result in results
    ]

    # Hybrid 결과 사용
    if reranker is None:
        selected = [
            [
                {
                    "row_id": row_id,
                    "score": result[3][row_id],
                }
                for row_id in result[2][:top_k]
            ]
            for result in results
        ]

    # Hybrid 후보를 Reranker로 재정렬
    else:
        reranked = rerank_candidates(
            reranker,
            questions,
            hybrid_rankings,
            chunks,
        )

        selected = [
            items[:top_k]
            for items in reranked
        ]

    # 검색 결과를 실제 문서 내용으로 변환
    documents_by_question = []

    for items in selected:
        documents = []

        for item in items:
            chunk = chunks[item["row_id"]]

            documents.append({
                "chunk_id": chunk.get("chunk_id"),
                "filename": chunk_filename(chunk),
                "text": get_chunk_text(chunk),
                "score": float(item["score"]),
            })

        documents_by_question.append(documents)

    elapsed = time.perf_counter() - start

    return documents_by_question, elapsed

# endregion


# region RAG 답변 생성

def create_predictions(records, method, top_k):
    questions = [
        record["question"]
        for record in records
    ]

    # 1. 질문별 관련 문서 검색
    documents_by_question, retrieval_seconds = retrieve_all(questions, method, top_k)

    # 2. Qwen 모델 로드 (한 번만 로드)
    model, tokenizer, generation_config = load_model()

    torch.cuda.reset_peak_memory_stats()

    rows = []
    generation_seconds = 0.0
    total_input_tokens = 0

    # 3. 각 질문에 대해 RAG 답변 생성
    for index, (record, documents) in enumerate(zip(records, documents_by_question), start=1):
        answer, seconds, input_tokens = generate_rag_answer(model, tokenizer, generation_config, record["question"], documents)

        generation_seconds += seconds
        total_input_tokens += input_tokens

        # 기존 Base Model 예측 파일과 같은 구조 유지
        rows.append({
            "id": record["id"],
            "question": record["question"],
            "reference_answer": record["answer"],
            "is_impossible": record["is_impossible"],
            "prediction": answer,
            "retrieved_documents": documents,
            "input_tokens": input_tokens,
        })

        print(
            f"Generated: {index}/{len(records)} | {record['id']}",
            flush=True,
        )

    # 4. 실행 성능 계산
    stats = {
        "retrieval_seconds": round(retrieval_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "avg_input_tokens": round(
            total_input_tokens / len(records), 1
        ),
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated() / (1024 ** 3),
            3,
        ),
    }

    return rows, stats

# endregion


# region EM / Token F1 / 거부율 평가 및 저장

def evaluate(method, top_k, limit, reuse):
    records = load_records(limit)

    predictions_path, metrics_path = result_paths(method, top_k, limit)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # 기존 예측 결과를 이용해 재평가
    if reuse:
        predictions = load_jsonl(predictions_path)

        with metrics_path.open(encoding="utf-8") as file:
            old_metrics = json.load(file)

        performance = {
            key: old_metrics.get(key)
            for key in (
                "retrieval_seconds",
                "generation_seconds",
                "avg_input_tokens",
                "peak_gpu_memory_gb",
            )
        }

    # 새로 RAG 추론 수행
    else:
        predictions, performance = create_predictions(records, method, top_k)

    # 기존 Base Model의 평가 함수를 그대로 재사용
    metrics = score_predictions(records, predictions, dataset="clean", batch_size=1)

    metrics.update({
        "retriever": method,
        "top_k": top_k,
        **performance,
    })

    # 새로 생성했을 때만 예측 파일 저장
    if not reuse:
        save_jsonl(predictions_path, predictions)

    # 평가 지표 저장
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"\n[{method.upper()} RAG Evaluation]")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("Predictions:", predictions_path)
    print("Metrics:", metrics_path)
    print("RAG Evaluation: OK")

# endregion


# region Base / Hybrid RAG / Reranker RAG 비교

def compare(top_k):
    expected_total = len(load_records(limit=0))

    files = {
        "Base Model": BASE_METRICS_PATH,
        "Hybrid RAG": result_paths("hybrid", top_k, 0)[1],
        "Reranker RAG": result_paths("reranker", top_k, 0)[1],
    }

    print("\n[Base vs RAG Comparison]")
    print("Dataset: val_clean.json")
    print("Total:", expected_total)

    print(
        f"{'Model':<16} "
        f"{'EM%':>8} "
        f"{'F1%':>8} "
        f"{'Correct Abstain%':>17} "
        f"{'False Refusal%':>16}"
    )

    for name, path in files.items():
        if not path.is_file():
            print(f"{name}: 미평가 (파일 없음: {path.name})")
            continue

        with path.open(encoding="utf-8") as file:
            metrics = json.load(file)

        # 동일한 검증 데이터인지 확인
        if (metrics.get("dataset") != "clean"or metrics.get("total") != expected_total):
            print(f"{name}: 비교 제외 (검증 데이터 구성이 다름)")
            continue

        def fmt(key):
            value = metrics.get(key)
            return "N/A" if value is None else f"{value:.2f}"

        print(
            f"{name:<16} "
            f"{fmt('exact_match_percent'):>8} "
            f"{fmt('token_f1_percent'):>8} "
            f"{fmt('abstention_rate_percent'):>17} "
            f"{fmt('false_refusal_rate_percent'):>16}"
        )

    print(
        "비교 기준: EM/F1은 답변 가능 질문, "
        "거부율은 해당 라벨별 질문"
    )
    print(
        "실행 시간은 Base의 Batch=4와 RAG의 Batch=1이므로 "
        "직접 비교하지 않습니다."
    )

# endregion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=("hybrid", "reranker"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.top_k <= 50 or args.limit < 0:
        parser.error("--top-k는 1~50, --limit는 0 이상이어야 합니다.")

    if args.compare:
        compare(args.top_k)
    else:
        evaluate(args.retriever, args.top_k, args.limit, args.reuse)


if __name__ == "__main__":
    main()


