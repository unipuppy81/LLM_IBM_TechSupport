import argparse

from rag_pipeline import run_rag


def main():
    parser = argparse.ArgumentParser(description="IBM 기술지원 RAG 챗봇")
    parser.add_argument("--question", required=True, type=str)
    parser.add_argument(
        "--retriever",
        choices=("hybrid", "reranker"),
        default="reranker",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not 1 <= args.top_k <= 50:
        parser.error("--top-k는 1~50 사이여야 합니다.")

    run_rag(question=args.question, method=args.retriever, top_k=args.top_k)


if __name__ == "__main__":
    main()