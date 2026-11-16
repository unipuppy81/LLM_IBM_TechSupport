"""Evaluate model answerability against QA labels and optional evidence labels."""
import argparse
from pathlib import Path
from .common import RESULT_DIR, check_output, load_jsonl, measure, report, save_jsonl


def predict(records):
    from load_base_model import load_model
    from answerability_detection import select_evidence_documents
    
    import yaml
    from .common import PROJECT_DIR

    config = yaml.safe_load((PROJECT_DIR / "configs/rag/v1.yaml").read_text(encoding="utf-8"))
    model, tokenizer, _ = load_model(config)
    rows = []
    for record in records:
        # Preserve the existing per-document OR decision; no second judge call.
        selected = select_evidence_documents(model, tokenizer, record["question"], record["retrieved_documents"])
        predicted = bool(selected)
        rows.append({"id": record["id"], "question": record["question"],
                     "is_impossible": record["is_impossible"],
                     "predicted_answerable": predicted,
                     "label_match": predicted == (not record["is_impossible"]),
                     "retrieved_documents": record["retrieved_documents"]})
    return rows


def evaluate(predictions=None, source=None, labels=None, output=None, save_predictions=None):
    check_output(output)
    check_output(save_predictions)
    if output and save_predictions and Path(output).resolve() == Path(save_predictions).resolve():
        raise ValueError("Metrics and predictions need separate paths")
    
    if predictions and source:
        raise ValueError("Choose saved predictions OR new inference source")
    
    if predictions:
        if save_predictions:
            raise ValueError("Cannot save predictions during reuse")
        
        rows = load_jsonl(predictions)
    else:
        from .evaluate_evidence_judge import load_records
        records = load_records(Path(source or RESULT_DIR / "rag_val_clean_reranker_k5_predictions.jsonl"))
        rows = predict(records)

    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Nonempty predictions with unique IDs required")
    
    qa = [{**r, "original_qa_answerable": not r["is_impossible"]} for r in rows]
    metrics = {"original_qa": measure(qa, "original_qa_answerable")}

    if labels:
        from .evaluate_with_evidence import score_evidence
        scored, held = score_evidence(rows, load_jsonl(labels))
        metrics.update(evidence=measure(scored, "evidence_answerable"),
                       original_qa_same_subset=measure(scored, "original_qa_answerable"),
                       held_ids=held, evaluated_evidence=len(scored))
        
    if save_predictions:
        save_jsonl(save_predictions, rows)

    return report(metrics, output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--predictions", type=Path, help="Saved predicted_answerable JSONL, no model load")
    g.add_argument("--input", dest="source", type=Path, help="RAG documents JSONL for new model inference")
    p.add_argument("--labels", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--save-predictions", type=Path)
    a = p.parse_args(argv)

    if a.predictions is None and a.source is None:
        p.error("--predictions 또는 --input을 지정하세요.")

    return evaluate(a.predictions, a.source, a.labels, a.output, a.save_predictions)


if __name__ == "__main__":
    main()
