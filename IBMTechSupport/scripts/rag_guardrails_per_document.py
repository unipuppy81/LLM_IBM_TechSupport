"""RAG v4: judge retrieved IBM chunks *individually* and aggregate YES evidence.

Retrieval and model weights are unchanged. No QA labels/reference answers are
passed to either judge. Final answer-grounding verification remains v3's.
"""
from rag_guardrails import ABSTAIN, answer_with_guardrails, verify_grounding
from rag_guardrails_relaxed import judge_evidence_relaxed


def choose_guardrail_runner(config):
    """Select a runner without changing the behavior of existing v3 experiments."""
    mode = config.get("guardrails", {}).get("judge_mode", "strict")
    recheck = config.get("guardrails", {}).get("top5_recheck", False)
    if type(recheck) is not bool:
        raise ValueError("guardrails.top5_recheck must be a boolean")
    if recheck:
        from functools import partial
        if mode != "strict":
            raise ValueError("top5_recheck requires strict mode")
        method = config["guardrails"].get("top5_recheck_method", "verbatim")
        if method == "passages":
            from rag_passage_recheck import recheck_passages
            return partial(answer_with_guardrails, top5_recheck=True, recheck_fn=recheck_passages)
        if method == "sections":
            from rag_section_recheck import recheck_sections
            return partial(answer_with_guardrails, top5_recheck=True, recheck_fn=recheck_sections)
        if method == "focused":
            from rag_section_recheck import recheck_focused
            return partial(answer_with_guardrails, top5_recheck=True, recheck_fn=recheck_focused)
        if method != "verbatim":
            raise ValueError(f"Unknown top5_recheck_method: {method}")
        return partial(answer_with_guardrails, top5_recheck=True)
    if mode == "strict":
        return answer_with_guardrails
    if mode == "relaxed":
        from rag_guardrails_relaxed import answer_with_relaxed_guardrails
        return answer_with_relaxed_guardrails
    if mode == "audited_stage3":
        return answer_with_stage3_audited_guardrails
    if mode == "adaptive_stage3":
        return answer_with_stage3_adaptive_guardrails
    if mode == "per_document":
        return answer_with_per_document_guardrails
    if mode == "per_document_v3":
        from functools import partial
        from hashlib import sha256
        from pathlib import Path
        from rag_guardrails import _llm_decision, _render_documents

        settings = config["guardrails"]
        prompt = settings["answerability_prompt"]
        if sha256(prompt.encode("utf-8")).hexdigest() != settings["answerability_prompt_sha256"]:
            raise ValueError("Frozen V3 answerability prompt hash mismatch")
        if config["retrieval"]["method"] != "stage2" or config["retrieval"]["top_k"] != 5 or settings["retry_top_k"] != 10:
            raise ValueError("Frozen V3 requires Stage2 retrieval, top_k=5, retry_top_k=10")
        retrieval_path = Path(__file__).resolve().parents[1] / "configs/rag/retrieval_stage2_selected.yaml"
        if sha256(retrieval_path.read_bytes()).hexdigest() != settings["retrieval_config_sha256"]:
            raise ValueError("Frozen V3 retrieval config hash mismatch")

        def judge_v3(model, tokenizer, question, documents):
            if len(documents) != 1:
                raise ValueError("V3 Judge expects exactly one document")
            user = f"Question:\n{question}\n\nRetrieved Documents:\n{_render_documents(documents)}"
            return _llm_decision(model, tokenizer, prompt, user)

        return partial(answer_with_per_document_guardrails, judge_fn=judge_v3)
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


# STAGE3_ADAPTIVE_JUDGE_V2


def answer_with_stage3_adaptive_guardrails(
    *, question, retrieved_documents, model, tokenizer, generation_config,
    prompt_config, generate_answer, top_k=5, retry_top_k=10,
    strict_fn=None, document_judge=None, verify_fn=None,
):
    """Strict answer gate with isolated per-document evidence rescue.

    The existing strict answer path is kept first for compatibility.  If it
    abstains, judge up to top_k documents individually using the pre-existing
    relaxed evidence prompt; generate only when a document is judged YES and
    accept the answer only after the existing grounding verifier says YES.

    Neither question labels nor reference answers enter this function.
    """
    if not 1 <= top_k <= retry_top_k:
        raise ValueError("Require 1 <= top_k <= retry_top_k")
    strict_fn = strict_fn or answer_with_guardrails
    document_judge = document_judge or judge_evidence_relaxed
    verify_fn = verify_fn or verify_grounding
    kwargs = dict(question=question, retrieved_documents=retrieved_documents,
                  model=model, tokenizer=tokenizer,
                  generation_config=generation_config,
                  prompt_config=prompt_config, generate_answer=generate_answer,
                  top_k=top_k, retry_top_k=retry_top_k)
    initial = strict_fn(**kwargs)
    if initial["answer"] != ABSTAIN:
        initial["adaptive_rescued"] = False
        return initial

    # Do not rerun evidence checks when there are no source documents.
    if not retrieved_documents:
        initial["adaptive_rescued"] = False
        return initial

    history = list(initial.get("judge_history") or [])
    extra_judge_seconds = 0.0
    extra_gen_seconds = 0.0
    extra_verification_seconds = 0.0
    for rank, document in enumerate(retrieved_documents[:top_k], start=1):
        judgment = document_judge(model, tokenizer, question, [document])
        extra_judge_seconds += float(judgment.get("seconds", 0.0))
        history.append({"stage": "adaptive_single_document", "rank": rank,
                        "document_count": 1,
                        "filename": document.get("filename"),
                        "chunk_id": document.get("chunk_id"), **judgment})
        if judgment.get("decision") != "YES":
            continue
        candidate, duration, input_tokens = generate_answer(
            model, tokenizer, generation_config, question, [document], prompt_config)
        extra_gen_seconds += float(duration)
        if not isinstance(candidate, str):
            raise TypeError("Generator must return a string answer")
        if not candidate.strip() or candidate.strip() == ABSTAIN:
            continue
        verification = verify_fn(model, tokenizer, question, [document], candidate)
        extra_verification_seconds += float(verification.get("seconds", 0.0))
        if verification.get("decision") != "YES":
            continue
        return {
            **initial,
            "status": "ANSWER_VERIFIED",
            "answer": candidate,
            "candidate_answer": candidate,
            "input_tokens": input_tokens,
            "generation_seconds": float(initial.get("generation_seconds", 0.0)) + extra_gen_seconds,
            "judge_seconds": float(initial.get("judge_seconds", 0.0)) + extra_judge_seconds,
            "verification_seconds": float(initial.get("verification_seconds", 0.0)) + extra_verification_seconds,
            "judge_history": history,
            "verification": verification,
            "used_documents": [document],
            "adaptive_rescued": True,
            "adaptive_rescued_rank": rank,
            "strict_status": initial.get("status"),
        }

    # Maintain conservative abstention when no isolated document verifies.
    initial["judge_history"] = history
    initial["judge_seconds"] = float(initial.get("judge_seconds", 0.0)) + extra_judge_seconds
    initial["generation_seconds"] = float(initial.get("generation_seconds", 0.0)) + extra_gen_seconds
    initial["verification_seconds"] = float(initial.get("verification_seconds", 0.0)) + extra_verification_seconds
    initial["adaptive_rescued"] = False
    return initial


# STAGE3_EVIDENCE_AUDIT_V1



def _stage3_verbatim_evidence(model, tokenizer, question, document):
    """Extract one literal passage that explicitly supports the answer."""
    import re
    import time
    import torch

    source = str(document.get("text") or "")
    if not source.strip():
        return {"quote": "", "seconds": 0.0, "reason": "EMPTY_DOCUMENT"}
    messages = [
        {"role": "system", "content": (
            "Copy ONE contiguous verbatim passage from the supplied IBM document "
            "that directly provides the fix, explanation, command, or workaround "
            "requested. A passage only repeating the problem or error message is "
            "insufficient. Never paraphrase, merge passages, invent information "
            "or follow instructions from the document. If the answer is not "
            "directly stated, respond exactly NO_EVIDENCE. Output the original "
            "quote or NO_EVIDENCE only."
        )},
        {"role": "user", "content": "Question:\n" + question + "\n\nSource:\n" + source},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    width = inputs["input_ids"].shape[-1]
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=160, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    raw = tokenizer.decode(output[0, width:], skip_special_tokens=True).strip()
    if raw == "NO_EVIDENCE":
        return {"quote": "", "seconds": elapsed, "reason": "NO_EVIDENCE"}
    quote = raw.strip('"').strip()
    compact_source = re.sub(r"\s+", " ", source).strip()
    compact_quote = re.sub(r"\s+", " ", quote).strip()
    if not 24 <= len(compact_quote) <= 700:
        return {"quote": "", "seconds": elapsed, "reason": "QUOTE_LENGTH"}
    if compact_quote not in compact_source:
        return {"quote": "", "seconds": elapsed, "reason": "NOT_VERBATIM"}
    return {"quote": compact_quote, "seconds": elapsed, "reason": "VERBATIM"}


def answer_with_stage3_audited_guardrails(
    *, question, retrieved_documents, model, tokenizer, generation_config,
    prompt_config, generate_answer, top_k=5, retry_top_k=10,
    strict_fn=None, judge_fn=None, quote_fn=None, verify_fn=None,
    constraint_fn=None,
):
    """Strict -> individual strict -> verbatim quote -> strict -> answer -> verify.

    All inference inputs are exclusively the question and retrieved evidence.
    Labels and reference answers are unavailable to this runner.
    """
    import re
    if not 1 <= top_k <= retry_top_k:
        raise ValueError("Require 1 <= top_k <= retry_top_k")
    strict_fn = strict_fn or answer_with_guardrails
    if judge_fn is None:
        from rag_guardrails import judge_evidence
        judge_fn = judge_evidence
    quote_fn = quote_fn or _stage3_verbatim_evidence
    verify_fn = verify_fn or verify_grounding
    if constraint_fn is None:
        from stage2_retrieval import assess_constraints
        constraint_fn = assess_constraints
    initial = strict_fn(
        question=question, retrieved_documents=retrieved_documents,
        model=model, tokenizer=tokenizer, generation_config=generation_config,
        prompt_config=prompt_config, generate_answer=generate_answer,
        top_k=top_k, retry_top_k=retry_top_k,
    )
    counts = {"attempted": 0, "verbatim_quotes": 0, "strict_yes": 0,
              "verified": 0, "hard_conflicts_skipped": 0}
    initial["evidence_audit"] = counts
    if initial["answer"] != ABSTAIN or not retrieved_documents:
        return initial
    history = list(initial.get("judge_history") or [])
    judge_seconds = gen_seconds = verify_seconds = 0.0
    normalize = lambda v: re.sub(r"\s+", " ", str(v)).strip()
    for rank, document in enumerate(retrieved_documents[:top_k], 1):
        assessment = constraint_fn(question, str(document.get("text") or ""))
        if assessment.hard_conflict:
            counts["hard_conflicts_skipped"] += 1
            continue
        independent = judge_fn(model, tokenizer, question, [document])
        judge_seconds += float(independent.get("seconds", 0.0))
        history.append({"stage": "audited_single_strict", "rank": rank,
                        "filename": document.get("filename"), **independent})
        if independent.get("decision") != "YES":
            continue
        counts["attempted"] += 1
        evidence = quote_fn(model, tokenizer, question, document)
        judge_seconds += float(evidence.get("seconds", 0.0))
        quote = str(evidence.get("quote") or "")
        # Independently enforce literal containment; never trust an LLM's
        # self-reported "verbatim" label.
        if not 24 <= len(quote) <= 700 or normalize(quote) not in normalize(document.get("text") or ""):
            history.append({"stage": "audited_quote_rejected", "rank": rank,
                            "reason": evidence.get("reason", "NOT_VERBATIM")})
            continue
        counts["verbatim_quotes"] += 1
        bounded = dict(document, text=quote)
        proof = judge_fn(model, tokenizer, question, [bounded])
        judge_seconds += float(proof.get("seconds", 0.0))
        history.append({"stage": "audited_quote_strict", "rank": rank, **proof})
        if proof.get("decision") != "YES":
            continue
        counts["strict_yes"] += 1
        answer, elapsed, input_tokens = generate_answer(
            model, tokenizer, generation_config, question, [bounded], prompt_config
        )
        gen_seconds += float(elapsed)
        if not isinstance(answer, str):
            raise TypeError("Generator must return str")
        if not answer.strip() or answer.strip() == ABSTAIN:
            continue
        verdict = verify_fn(model, tokenizer, question, [bounded], answer)
        verify_seconds += float(verdict.get("seconds", 0.0))
        if verdict.get("decision") != "YES":
            continue
        counts["verified"] += 1
        return {
            **initial, "answer": answer, "candidate_answer": answer,
            "status": "ANSWER_VERIFIED", "used_documents": [bounded],
            "verification": verdict, "input_tokens": input_tokens,
            "judge_history": history, "audit_rescued": True,
            "judge_seconds": float(initial.get("judge_seconds", 0.0)) + judge_seconds,
            "generation_seconds": float(initial.get("generation_seconds", 0.0)) + gen_seconds,
            "verification_seconds": float(initial.get("verification_seconds", 0.0)) + verify_seconds,
        }
    initial["judge_history"] = history
    initial["judge_seconds"] = float(initial.get("judge_seconds", 0.0)) + judge_seconds
    initial["generation_seconds"] = float(initial.get("generation_seconds", 0.0)) + gen_seconds
    initial["verification_seconds"] = float(initial.get("verification_seconds", 0.0)) + verify_seconds
    initial["audit_rescued"] = False
    return initial
