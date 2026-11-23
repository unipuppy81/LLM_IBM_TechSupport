"""RAG v4: judge retrieved IBM chunks *individually* and aggregate YES evidence.

Retrieval and model weights are unchanged. No QA labels/reference answers are
passed to either judge. Final answer-grounding verification remains v3's.
"""
from rag_guardrails import ABSTAIN, answer_with_guardrails, verify_grounding
from rag_guardrails_relaxed import judge_evidence_relaxed


def choose_guardrail_runner(config):
    """Select a runner without changing the behavior of existing v3 experiments."""
    mode = config.get("guardrails", {}).get("judge_mode", "strict")
    if mode == "strict":
        return answer_with_guardrails
    if mode == "relaxed":
        from rag_guardrails_relaxed import answer_with_relaxed_guardrails
        return answer_with_relaxed_guardrails
    if mode == "per_document":
        return answer_with_per_document_guardrails
    raise ValueError(f"Unknown guardrails.judge_mode: {mode!r}")


def judge_documents_individually(
    *, model, tokenizer, question, retrieved_documents,
    top_k=5, retry_top_k=10, judge_fn=judge_evidence_relaxed,
):
    """Check each chunk separately. First window, then 6..retry_top_k if no YES.

    YES aggregation is an OR over *independent* document judgments. Only YES
    chunks are returned as context for generation. Other chunks are excluded.
    """
    if not (1 <= top_k <= retry_top_k):
        raise ValueError("Require 1 <= top_k <= retry_top_k")

    available = retrieved_documents[:retry_top_k]
    history = []
    selected = []

    def check_batch(documents, first_rank, stage):
        for rank, document in enumerate(documents, start=first_rank):
            # Important: judge receives ONE document, not a concatenated list.
            result = judge_fn(model, tokenizer, question, [document])
            decision = result.get("decision", "UNCERTAIN")
            if decision not in {"YES", "NO", "UNCERTAIN"}:
                decision = "UNCERTAIN"
            history.append({
                "stage": stage,
                "rank": rank,
                "document_count": 1,
                "filename": document.get("filename"),
                "chunk_id": document.get("chunk_id"),
                "decision": decision,
                "raw": result.get("raw", ""),
                "seconds": float(result.get("seconds", 0.0)),
            })
            if decision == "YES":
                selected.append(document)

    check_batch(available[:top_k], 1, "top_k")
    # Only expand when nothing in top-k contains independently sufficient evidence.
    if not selected and len(available) > top_k:
        check_batch(available[top_k:], top_k + 1, "expanded_k")

    if selected:
        decision = "YES"
    elif any(item["decision"] == "UNCERTAIN" for item in history):
        decision = "UNCERTAIN"
    else:
        decision = "NO"
    return {
        "decision": decision,
        "selected_documents": selected,
        "judge_history": history,
        "judge_seconds": round(sum(item["seconds"] for item in history), 4),
    }


def answer_with_per_document_guardrails(
    *, question, retrieved_documents, model, tokenizer, generation_config,
    prompt_config, generate_answer, top_k=5, retry_top_k=10,
    judge_fn=judge_evidence_relaxed, verify_fn=verify_grounding,
):
    """Judge per chunk → aggregate YES chunks → generate → v3 grounding gate.

    Status/result keys mirror v3, so evaluate_rag.py and rag_pipeline.py can
    consume this unchanged. Logs include rank and filename for debugging.
    """
    audit = judge_documents_individually(
        model=model, tokenizer=tokenizer, question=question,
        retrieved_documents=retrieved_documents, top_k=top_k,
        retry_top_k=retry_top_k, judge_fn=judge_fn,
    )
    selected = audit["selected_documents"]
    common = {
        "judge_seconds": audit["judge_seconds"],
        "judge_history": audit["judge_history"],
        "used_documents": selected,
    }

    if not retrieved_documents:
        return {
            **common, "status": "NO_DOCUMENTS", "answer": ABSTAIN,
            "generation_seconds": 0.0, "verification_seconds": 0.0,
            "input_tokens": 0, "verification": None, "candidate_answer": None,
        }

    if not selected:
        return {
            **common, "status": f"EVIDENCE_{audit['decision']}", "answer": ABSTAIN,
            "generation_seconds": 0.0, "verification_seconds": 0.0,
            "input_tokens": 0, "verification": None, "candidate_answer": None,
        }

    candidate_answer, generation_seconds, input_tokens = generate_answer(
        model, tokenizer, generation_config, question, selected, prompt_config,
    )
    if not isinstance(candidate_answer, str):
        raise TypeError("Generator must return a string answer")
    if not candidate_answer.strip() or candidate_answer.strip() == ABSTAIN:
        return {
            **common, "status": "GENERATOR_ABSTAINED", "answer": ABSTAIN,
            "generation_seconds": generation_seconds, "verification_seconds": 0.0,
            "input_tokens": input_tokens, "verification": None,
            "candidate_answer": candidate_answer,
        }

    verification = verify_fn(model, tokenizer, question, selected, candidate_answer)
    verified = verification.get("decision") == "YES"
    return {
        **common,
        "status": "ANSWER_VERIFIED" if verified else f"ANSWER_{verification.get('decision', 'UNCERTAIN')}",
        "answer": candidate_answer if verified else ABSTAIN,
        "generation_seconds": generation_seconds,
        "verification_seconds": float(verification.get("seconds", 0.0)),
        "input_tokens": input_tokens,
        "verification": verification,
        "candidate_answer": candidate_answer,
    }
