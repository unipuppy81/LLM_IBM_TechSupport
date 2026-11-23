import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from load_base_model import load_model
from rag_pipeline import generate_rag_answer, retrieve_all
from rag_guardrails import answer_with_guardrails


def load_case(case_id, config):
    dataset_path = PROJECT_DIR / config["dataset"]["path"]
    records = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    matches = [record for record in records if record["id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"QA ID not found or duplicated: {case_id}")
    return matches[0]


def run(case_id, config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    record = load_case(case_id, config)
    question = record["question"] 

    method = config["retrieval"]["method"]
    top_k = int(config["retrieval"]["top_k"])
    retry_top_k = int(config.get("guardrails", {}).get("retry_top_k", top_k))
    print(f"\nQA ID: {case_id}\nQuestion: {question}")
    print(f"Retriever: {method}, top_k={top_k}, retry_top_k={retry_top_k}")

    docs_by_question, retrieval_seconds = retrieve_all(
        questions=[question], method=method, top_k=retry_top_k
    )
    documents = docs_by_question[0]
    print(f"Retrieved {len(documents)} chunks in {retrieval_seconds:.3f}s")
    for i, doc in enumerate(documents, 1):
        print(f"  {i:>2}. {doc['filename']} | {doc.get('chunk_id')}")

    model, tokenizer, generation_config = load_model(config)

    result = answer_with_guardrails(
        question=question,
        retrieved_documents=documents,
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        prompt_config=config["prompt"],
        generate_answer=generate_rag_answer,
        top_k=top_k,
        retry_top_k=retry_top_k,
    )

    print(f"\nFinal status: {result['status']}")
    for judgment in result["judge_history"]:
        print(
            f"Judge {judgment['stage']} ({judgment['document_count']} documents): "
            f"{judgment['decision']} | raw={judgment['raw']!r}"
        )
    print("Verification:", result["verification"])
    print("Candidate answer:", result["candidate_answer"])
    print("Final answer:", result["answer"])
    print("Gold answer (diagnostic only, never used as model input):", record["answer"])
    print("Gold is_impossible (diagnostic only):", record["is_impossible"])
    return result


def main():
    parser = argparse.ArgumentParser(description="Debug one RAG v3 QA by ID")
    parser.add_argument("--id", default="TRAIN_Q270", help="QA ID to debug")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_DIR / "configs/rag/v3.yaml"
    )
    args = parser.parse_args()
    run(args.id, args.config.resolve())


if __name__ == "__main__":
    main()
