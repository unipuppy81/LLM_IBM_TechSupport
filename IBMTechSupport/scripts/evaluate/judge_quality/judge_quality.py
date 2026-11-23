from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]  # IBMTechSupport
PREDICTIONS = PROJECT / "evaluate_results/106/rag/v4/predictions.jsonl"
RESULT_DIR = PROJECT / "evaluate_results/106/judge_quality"
POLICY_PATH = PROJECT / "configs/judge_quality/policy.json"
ABSTAIN = "INSUFFICIENT_INFORMATION"
REVIEW_COLUMNS = [
    "id", "qa_label_only_for_sampling", "v4_status", "v4_accepted",
    "question", "model_answer", "candidate_answer", "reference_answer_diagnostic_only",
    "selected_source_files", "selected_source_excerpt",
    "human_pass", "human_correctness", "human_groundedness", "human_completeness",
    "human_actionability", "human_safety", "human_format", "failure_reason", "risk_level",
    "reviewer_notes",
]
JUDGE_SCHEMA_VERSION = "ibm-techqa-judge-claims-v3.6"
RUBRIC_SYSTEM = """You are an independent technical QA evaluator reviewing an IBM support answer.
Use ONLY the user question, the candidate answer, and the provided IBM document BODY.
You must check whether the source documents DIRECTLY support the actual answer.
The candidate answer does NOT need to cite its sources; you must find evidence YOURSELF.
Do not confuse a missing citation in the candidate answer with missing evidence in the IBM documents.

EVALUATION PROCESS (do these checks before scoring):
1. Identify the candidate answer's essential technical claims, remedies, commands, paths, and parameters.
2. Inspect each IBM document individually for a passage supporting those specific claims.
   A shared error code, product name, or topic is NOT enough. Different root causes
   or remedies for the same error code must not be treated as interchangeable.
3. When a document provides a direct supporting remedy, copy one contiguous passage
   of at least 12 characters from its BODY into evidence_quote, EXACTLY as written.
   Put only the passage in evidence_quote. Do NOT include a [Document N | filename |
   chunk_id] label, document header, or other citation marker in that string.
   Do not quote a title, the question, or the candidate answer. Do not invent text.
4. Evaluate the full answer against that evidence; a matching quote is NECESSARY
   for a high groundedness score but NOT SUFFICIENT to establish correctness.

SCORING: Use independent integers from 1 to 5 for each dimension.
correctness: 5 = all key technical claims correct; 4 = substantially correct;
3 = partial or uncertain; 2 = unsupported major remedy; 1 = contradicted by evidence.
groundedness: 5 = every essential claim supported; 4 = the primary remedy is
explicitly supported and only minor context is unsupported; 3 = partial support
for the essential remedy; 2 = main remedy unsupported; 1 = main remedy contradicted.
Do not use a missing source citation in the candidate answer as a reason to
reduce correctness or groundedness. Assess the meaning of the answer itself.
completeness: 5 = full requested solution; 3 = partial; 1 = request not addressed.
actionability: 5 = clear safe steps; 3 = partial steps; 1 = misleading steps.
safety and format must be 'pass' or 'fail'.
risk_level must be 'low', 'medium', or 'high'.
failure_reason must explain the actual unsupported claim, contradiction, or omission,
not repeat these instructions. If there is no problem, use an empty string.

IMPORTANT CONSISTENCY RULES:
- If no supplied document directly supports the essential answer, use an empty
  evidence_quote, set correctness <= 2 and groundedness <= 2, and give a specific
  failure_reason. Do NOT assign neutral/default 3s to avoid deciding.
- If supporting evidence exists in the source but the candidate has NO source
  citations, still copy the evidence_quote from the source and grade normally.
- If candidate answer is exactly INSUFFICIENT_INFORMATION, assess whether refusal
  is justified; no exact quote is required, but refuse to auto-approve a refusal
  when there are retrieved documents without human review.

Return exactly one JSON object, no Markdown or explanation outside JSON.
It must contain ALL nine keys with correct types:
correctness (integer 1..5), groundedness (integer 1..5),
completeness (integer 1..5), actionability (integer 1..5),
safety (string: pass or fail), format (string: pass or fail),
evidence_quote (string: exact passage or empty),
failure_reason (string: actual case-specific reason or empty),
risk_level (string: low, medium or high).
Do not copy sample scores or generic reasons: none are provided.
"""


def jsonl_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"파일 없음: {path}")
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL {path}:{i} 파싱 오류") from exc
    if not rows or len({str(r.get('id')) for r in rows}) != len(rows):
        raise ValueError(f"데이터 비었거나 중복된 QA ID: {path}")
    return rows


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def selected_docs(record):
    docs = record.get("retrieved_documents") or []
    used = record.get("guardrails", {}).get("used_chunk_ids") or []
    by_id = {str(d.get("chunk_id")): d for d in docs}
    return [by_id[str(key)] for key in used if str(key) in by_id]


def evaluation_docs(record):
    """Evaluate accepted answers against used evidence; refusals against top-10 candidates.

    Otherwise a refusal can be incorrectly treated as justified merely because
    v4 selected zero supporting chunks.
    """
    used = selected_docs(record)
    if record.get("prediction", "").strip() == ABSTAIN:
        return (record.get("retrieved_documents") or [])[:10]
    return used


def risk_score(r):
    score = 0
    g = r.get("guardrails", {})
    status = g.get("status", "")
    if r.get("is_impossible") and status == "ANSWER_VERIFIED":
        score += 100  # automated judge may have approved unsupported answer
    if not r.get("is_impossible") and r.get("prediction", "") == ABSTAIN:
        score += 80
    if status in {"ANSWER_NO", "EVIDENCE_NO", "GENERATOR_ABSTAINED"}:
        score += 30
    if status == "ANSWER_VERIFIED" and len(selected_docs(r)) > 1:
        score += 5
    return score


def balanced_sample(records, n: int, seed: int):
    """Stratify by original QA label only for REVIEW SAMPLING, never Judge input/labels."""
    rng = random.Random(seed)
    a = [r for r in records if not r.get("is_impossible")]
    b = [r for r in records if r.get("is_impossible")]
    rng.shuffle(a); rng.shuffle(b)
    a.sort(key=risk_score, reverse=True)
    b.sort(key=risk_score, reverse=True)
    b_n = min(len(b), n // 2)
    a_n = min(len(a), n - b_n)
    selected = a[:a_n] + b[:b_n]
    if len(selected) < n:
        remaining = {r["id"] for r in selected}
        selected.extend(r for r in sorted(records, key=risk_score, reverse=True)
                        if r["id"] not in remaining)
        selected = selected[:n]
    return sorted(selected, key=lambda r: (-risk_score(r), r["id"]))


def prepare(args):
    rows = jsonl_rows(Path(args.predictions))
    outdir = Path(args.output_dir)
    sample = balanced_sample(rows, min(args.sample_size, len(rows)), args.seed)
    ct = Counter(("answerable" if not r.get("is_impossible") else "impossible", r.get("guardrails",{}).get("status", "UNKNOWN")) for r in rows)
    baseline = {
        "source_file": "evaluate_results/106/rag/v4/predictions.jsonl", "total": len(rows),
        "answerable_original_label": sum(not r.get("is_impossible") for r in rows),
        "impossible_original_label": sum(bool(r.get("is_impossible")) for r in rows),
        "status_by_label": [{"label": k[0], "status": k[1], "count": val} for k,val in sorted(ct.items())],
        "v4_answer_verified": sum(r.get("guardrails",{}).get("status") == "ANSWER_VERIFIED" for r in rows),
        "warning": "자동 Judge 승인/원본 QA 레이블은 정답 정확성의 사람 평가 라벨이 아님",
    }
    dump_json(outdir / "baseline_summary.json", baseline)
    human = []
    cases = []
    for r in sample:
        docs = evaluation_docs(r)
        source_names = ", ".join(str(d.get("filename")) for d in docs)
        source_excerpt = "\n---\n".join(f"[{d.get('filename')}] {d.get('text','')[:550]}" for d in docs[:3])
        human.append({
            "id": r["id"], "qa_label_only_for_sampling": "answerable" if not r.get("is_impossible") else "impossible",
            "v4_status": r.get("guardrails",{}).get("status", ""),
            "v4_accepted": int(r.get("guardrails",{}).get("status") == "ANSWER_VERIFIED"),
            "question": r.get("question", ""), "model_answer": r.get("prediction", ""),
            "candidate_answer": r.get("guardrails",{}).get("candidate_answer") or "",
            "reference_answer_diagnostic_only": "",  # human reviewer blind to original QA answer
            "selected_source_files": source_names, "selected_source_excerpt": source_excerpt,
            "human_pass": "", "human_correctness": "", "human_groundedness": "",
            "human_completeness": "", "human_actionability": "", "human_safety": "",
            "human_format": "", "failure_reason": "", "risk_level": "", "reviewer_notes": "",
        })
        cases.append({"id":r["id"], "question":r.get("question"),"answer":r.get("prediction"),
                      "candidate_answer":r.get("guardrails",{}).get("candidate_answer"),
                      "selected_documents":docs,"original_qa_label_diagnostic_only":not r.get("is_impossible")})
    write_csv(outdir / "human_review_template.csv", human, REVIEW_COLUMNS)
    write_jsonl(outdir / "review_cases.jsonl", cases)
    print(f"[prepare] v4={len(rows)}건, 사람 검토 표본={len(sample)}건")
    print(f"  {outdir / 'baseline_summary.json'}\n  {outdir / 'human_review_template.csv'}")


def normalize_space(s):
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def build_grade_prompt(r):
    docs = evaluation_docs(r)
    blocks = "\n\n".join(
        f"[Document {i} | {d.get('filename','unknown')} | {d.get('chunk_id','')}]\n"
        f"{d.get('text','')}"
        for i, d in enumerate(docs, 1)
    ) or "(NO SELECTED EVIDENCE DOCUMENTS)"
    ans = r.get("prediction", "")
    # Put the original evidence before the candidate so the model explicitly
    # evaluates the final answer against those documents, not against its memory.
    return (
        f"USER QUESTION:\n{r.get('question','')}\n\n"
        f"IBM EVIDENCE DOCUMENTS (inspect them one by one):\n{blocks}\n\n"
        f"CANDIDATE ANSWER TO AUDIT:\n{ans}\n\n"
        "A citation in the candidate is NOT required. Find the supporting sentence "
        "in the IBM document BODY yourself. If the answer's essential remedy is "
        "not supported, state the unsupported remedy specifically. "
        "Judge this candidate answer, not the QA dataset label. Return JSON only."
    )


def parse_rubric(raw, docs, answer, policy):
    """Validate every required field. NEVER fabricate omitted Judge fields.

    Exact quote presence is a necessary condition for a supported answer,
    NOT proof of semantic entailment (human calibration remains necessary).
    """
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Judge JSON 객체 없음")
        obj = json.loads(raw[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Judge 출력은 JSON 객체여야 함")
    required = (
        "correctness", "groundedness", "completeness", "actionability",
        "safety", "format", "evidence_quote", "failure_reason", "risk_level",
    )
    missing = [key for key in required if key not in obj]
    if missing:
        raise ValueError("필수 필드 누락: " + ", ".join(missing))
    for k in ("correctness", "groundedness", "completeness", "actionability"):
        if type(obj.get(k)) is not int or not 1 <= obj[k] <= 5:
            raise ValueError(f"{k}: 1..5 정수 필요")
    for k in ("safety", "format"):
        if obj[k] not in ("pass", "fail"):
            raise ValueError(f"{k}: pass/fail 필요")
    if obj["risk_level"] not in ("low", "medium", "high"):
        raise ValueError("risk_level: low/medium/high 필요")
    if not isinstance(obj["failure_reason"], str) or not isinstance(obj["evidence_quote"], str):
        raise ValueError("failure_reason, evidence_quote: 문자열 필요")

    # Small Judge models sometimes prepend the exact prompt document identifier
    # to an otherwise valid BODY quotation. Strip ONLY a header that matches the
    # corresponding retrieved document, never arbitrary text before a colon.
    raw_quote = obj["evidence_quote"].strip()
    header = re.match(
        r"^\[Document\s+(\d+)\s*\|\s*([^|\]\r\n]+?)\s*\|\s*([^\]\r\n]*?)\s*\]\s*:\s*",
        raw_quote,
        flags=re.IGNORECASE,
    )
    if header:
        idx = int(header.group(1)) - 1
        source = docs[idx] if 0 <= idx < len(docs) else None
        if (source is not None
                and header.group(2).strip() == str(source.get("filename", ""))
                and header.group(3).strip() == str(source.get("chunk_id", ""))):
            raw_quote = raw_quote[header.end():].strip()
            # Record original form for debugging, while the canonical quote is
            # checked strictly against the original document BODY.
            obj["evidence_quote_with_header"] = obj["evidence_quote"]
            obj["evidence_quote"] = raw_quote
            obj["evidence_source_filename"] = source.get("filename")
            obj["evidence_source_chunk_id"] = source.get("chunk_id")
    quote = normalize_space(raw_quote)
    # A technote title only identifies a topic; it is not answer support.
    def body_text(doc):
        value = str(doc.get("text", ""))
        if value.startswith("Title:") and "\n\nText:" in value:
            return value.split("\n\nText:", 1)[1]
        return value

    quote_docs = docs
    if "evidence_source_chunk_id" in obj:
        quote_docs = [d for d in docs if str(d.get("chunk_id")) == str(obj["evidence_source_chunk_id"])]
    quote_valid = bool(quote) and len(quote) >= 12 and any(
        quote in normalize_space(body_text(d)) for d in quote_docs
    )
    abstention = answer.strip() == ABSTAIN
    t = policy["quality_thresholds"]
    passes = (
        obj["correctness"] >= t["correctness_min"]
        and obj["groundedness"] >= t["groundedness_min"]
        and obj["completeness"] >= t["completeness_min"]
        and obj["actionability"] >= t["actionability_min"]
        and obj["safety"] == t["safety_required"]
        and obj["format"] == t["format_required"]
    )
    flags = []
    if quote and len(quote) < 12:
        flags.append("QUOTE_TOO_SHORT")
    elif quote and not quote_valid:
        flags.append("QUOTE_NOT_IN_SELECTED_DOCS")
    if not abstention and not quote_valid and obj["groundedness"] >= t["groundedness_min"]:
        flags.append("MISSING_EXACT_SUPPORTING_QUOTE")
    # A 3-point, no-evidence evaluation was previously considered valid.  This
    # allowed a small Judge to mimic the old example JSON on every question.
    # Require the Judge either to find an actual supporting passage or to make
    # an explicit low-confidence rejection with correctness/groundedness <= 2.
    if (not abstention and not quote_valid and
            (obj["correctness"] > 2 or obj["groundedness"] > 2)):
        flags.append("UNSUPPORTED_SCORE_WITHOUT_QUOTE")
    if normalize_space(obj["failure_reason"]) in {
        "explain unsupported claim or leave empty if genuinely correct",
        "explain unsupported claim",
        "your failure reason here",
    }:
        flags.append("PLACEHOLDER_FAILURE_REASON")
    if (min(obj[k] for k in ("correctness", "groundedness", "completeness", "actionability")) <= 2
            or obj["safety"] == "fail" or obj["format"] == "fail") and not obj["failure_reason"].strip():
        flags.append("MISSING_FAILURE_REASON")
    # Without any retrieved evidence, an exact refusal can be approved in principle.
    # With retrieved documents, negative evidence ('no answer anywhere') cannot be
    # demonstrated by merely leaving evidence_quote empty.
    if abstention and docs:
        flags.append("ABSTENTION_REQUIRES_HUMAN_REVIEW")
    obj["evidence_quote_verified"] = quote_valid
    obj["review_flags"] = flags
    obj["auto_pass"] = bool(passes and (quote_valid if not abstention else not docs and not quote) and not flags)
    return obj


RETRYABLE_FLAGS = frozenset({
    "QUOTE_TOO_SHORT", "QUOTE_NOT_IN_SELECTED_DOCS",
    "MISSING_EXACT_SUPPORTING_QUOTE", "MISSING_FAILURE_REASON",
    "UNSUPPORTED_SCORE_WITHOUT_QUOTE", "PLACEHOLDER_FAILURE_REASON",
})


def retry_instruction(problem):
    """Reject unsupported neutral scores; never coerce an answer to pass."""
    return (
        "Your previous evaluation is invalid: " + problem + "\n"
        "Inspect each ORIGINAL IBM document again, individually, and compare it "
        "to the candidate answer. Candidate source citations are NOT required. "
        "If any document directly states the answer's remedy, copy one EXACT "
        "supporting passage of at least 12 characters from the document BODY "
        "into evidence_quote WITHOUT the [Document N | filename | chunk_id] "
        "header or other labels. Do not change correctness merely because of "
        "quotation formatting. Then evaluate correctness and groundedness. "
        "If none directly supports the remedy, leave evidence_quote empty, "
        "set BOTH correctness and groundedness to 1 or 2, and write a "
        "case-specific failure_reason identifying the unsupported claim. "
        "Do not use 3s as placeholders. Return a single JSON object with all "
        "nine fields: correctness, groundedness, completeness, actionability, "
        "safety, format, evidence_quote, failure_reason, risk_level."
    )


def evaluate_case(record, policy, generate_fn, max_attempts=3, debug_hook=None):
    """Pure retry loop. `generate_fn(messages)` is swappable for unit tests."""
    if max_attempts < 1:
        raise ValueError("max_attempts는 1 이상")
    question = build_grade_prompt(record)
    docs = evaluation_docs(record)
    answer = record.get("prediction", "")
    messages = [
        {"role": "system", "content": RUBRIC_SYSTEM},
        {"role": "user", "content": question},
    ]
    attempts = []
    rubric = None
    status = "REVIEW_REQUIRED"
    last_raw = ""
    for i in range(max_attempts):
        last_raw = generate_fn(messages)
        problem = None
        try:
            rubric = parse_rubric(last_raw, docs, answer, policy)
            retryable = [flag for flag in rubric["review_flags"] if flag in RETRYABLE_FLAGS]
            if retryable:
                problem = ", ".join(retryable)
            elif "ABSTENTION_REQUIRES_HUMAN_REVIEW" in rubric["review_flags"]:
                status = "REVIEW_REQUIRED"
            else:
                status = "OK"
        except (ValueError, KeyError) as exc:
            problem = str(exc)
            rubric = {
                "auto_pass": None,
                "review_flags": ["INVALID_JUDGE_OUTPUT"],
                "parse_error": problem,
            }
        attempts.append({"attempt": i + 1, "raw_response": last_raw, "problem": problem})
        if debug_hook is not None:
            # The full model input for this attempt (including retry instructions).
            debug_hook({
                "attempt": i + 1,
                "messages": [dict(message) for message in messages],
                "raw_response": last_raw,
                "parsed_rubric": dict(rubric) if rubric is not None else None,
                "problem": problem,
                "status_at_attempt": status,
            })
        if problem is None:
            break
        if i + 1 < max_attempts:
            # Repeat the original question/evidence via conversation context,
            # with pointed feedback on the previous answer.
            messages.append({"role": "assistant", "content": last_raw})
            messages.append({"role": "user", "content": retry_instruction(problem)})
    return {
        "judge_status": status,
        "score": rubric,
        "raw_response": last_raw,
        "attempt_count": len(attempts),
        "attempts": attempts,
    }


# ---------------------------------------------------------------------------
# Claim-grounded audit (v3): deterministic extraction, per-document LLM checks,
# rule-derived rubric.  The old rubric-only evaluator above is kept for
# backwards-compatible tests and historical comparisons.
# ---------------------------------------------------------------------------
CLAIM_VERIFY_SYSTEM = """You verify ONE technical claim against ONE IBM technical document.
This is evidence checking, not answer generation. Use only the provided IBM
DOCUMENT BODY. You must also check that its product, version, error, and
conditions match the QUESTION. A similar error code or topic is insufficient.

Return exactly one JSON object with these three string keys:
- verdict: one of SUPPORTED, CONTRADICTED, NOT_FOUND
- quote: an EXACT, CONTIGUOUS passage copied from DOCUMENT BODY if the verdict
  is SUPPORTED or CONTRADICTED; otherwise an empty string
- reason: one short explanation of why this passage supports/contradicts the
  EXACT claim, or why evidence is missing.

Rules:
- SUPPORTED only when the quotation DIRECTLY entails the technical claim,
  including specific remedy, setting, version, and conditions where relevant.
- CONTRADICTED only when the quotation DIRECTLY states an incompatible fact.
- NOT_FOUND when relevant evidence is absent or ambiguous. Do not guess.
- Do not quote the QUESTION, candidate answer, a document title, or your memory.
- Copy only the shortest directly supporting passage (12 to 480 characters),
  without [Document ...] labels or explanations. Do not paste entire documents.
- Do not make up a quotation to satisfy the output format.
"""
MAX_CLAIMS = 16
MAX_CLAIM_LENGTH = 520
MAX_SOURCE_QUOTE_LENGTH = 480

# A non-exhaustive list of function words (only substantive overlap is counted).
_CLAIM_STOPWORDS = frozenset("""a an and are as at be been by can could does during for
from had has have how if in into is it its may more must need not of on or our out
should than that the their them there these this those through to was were what when
where which while who will with would you your then after before following above below
once one the same such typically generally recommended seems described about given
provided directly actual specific exact candidate document body user question answer
according any also all another these only itself might more""".split())


def _claim_tokens(text: str) -> set[str]:
    """Content tokens for a conservative lexical-relevance gate; not proof."""
    return {w.casefold() for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+", text)
            if len(w) > 1 and w.casefold() not in _CLAIM_STOPWORDS}


def _matching_source_span(body: str, snippet: str) -> str | None:
    """Locate a REAL contiguous source substring, tolerating formatting only.

    Long model quotes sometimes paste two genuine paragraphs but omit a heading
    between them. Such a noncontiguous quote is NOT accepted whole: at most
    one actually contiguous source passage is recovered from its individual lines.
    """
    snippet = snippet.strip()
    if not snippet:
        return None
    if snippet in body:
        return snippet
    choices = [snippet]
    # Model quotes can omit intermediate headings or join unrelated lines.
    choices.extend(x.strip() for x in snippet.splitlines() if len(x.strip()) >= 12)
    found = []
    seen = set()
    for choice in choices:
        if choice in seen:
            continue
        seen.add(choice)
        # Trailing punctuation omitted in the source is presentation-level only.
        for variant in (choice, choice.rstrip('.!?:; ')):
            if len(variant) < 12:
                continue
            # Normalize ONLY whitespace; never ignore technical literals.
            atoms = re.split(r"(\s+)", variant)
            pat = ''.join(r'\s+' if x.isspace() else re.escape(x)
                          for x in atoms if x)
            try:
                matches = list(re.finditer(pat, body, flags=re.IGNORECASE))
            except re.error:
                continue
            for match in matches[:3]:
                actual = match.group(0)
                if 12 <= len(actual) <= MAX_SOURCE_QUOTE_LENGTH:
                    found.append(actual)
        if found and choice == snippet:
            break
    if not found:
        return None
    # Prefer one substantial passage over an unrelated introductory snippet.
    return max(found, key=len)


def _key_literals(text: str) -> set[str]:
    """Identifiers/numbers that must occur in cited evidence, not only context."""
    result = set(re.findall(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)*(?![A-Za-z0-9_])", text))
    result.update(re.findall(
        r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_+-]*\.)+[A-Za-z_][A-Za-z0-9_*+-]*",
        text))
    result.update(re.findall(
        r"(?<![A-Za-z0-9_])[A-Z]{2,}[A-Z0-9_-]*[0-9][A-Z0-9_-]*(?![A-Za-z0-9_])",
        text))
    # Full paths and placeholders: a quote about 'logging' is insufficient to
    # establish <Atlas_Install_folder>/Atlas/Properties.
    result.update(re.findall(r"<[^<>]+>(?:/[A-Za-z0-9_.-]+)+", text))
    return result


def _quote_relevance_problem(quote: str, claim: dict, verdict: str) -> str | None:
    """Necessary lexical/literal sanity checks, not an entailment model.

    The Qwen verdict remains an *uncalibrated* semantic assessment. A genuine
    source quote with no overlap with the technical claim must not be accepted.
    """
    text = str(claim.get('text') or '')
    important = _key_literals(text)
    if any(lit.casefold() not in quote.casefold() for lit in important):
        return 'IMPORTANT_LITERAL_MISSING_FROM_QUOTE'
    claim_tokens = _claim_tokens(text)
    quote_tokens = _claim_tokens(quote)
    if not claim_tokens:
        return 'CLAIM_TOO_VAGUE_FOR_AUTOMATIC_SUPPORT'
    common = claim_tokens & quote_tokens
    ratio = len(common) / len(claim_tokens)
    # Requires both substantive shared vocabulary and overlap fraction.
    needed = 1 if len(claim_tokens) == 1 else 2
    if len(common) < needed or ratio < 0.48:
        return 'QUOTE_NOT_SEMANTICALLY_ALIGNED_WITH_CLAIM'
    return None


def extract_answer_claims(answer: str) -> list[dict]:
    """Extract verbatim answer spans; never let an LLM silently omit a claim.

    The deterministic splitter is deliberately conservative: a procedural step,
    parameter assignment, standalone command, or declarative sentence becomes
    a separate claim. It does not paraphrase technical text. A too-long/complex
    answer is sent for human review instead of being partially checked.
    """
    answer = answer.strip()
    if not answer or answer == ABSTAIN:
        return []
    lines = answer.splitlines()
    spans = []
    for original in lines:
        line = original.strip()
        if not line or line.startswith("```"):
            continue
        # Keep literals exactly as typed, other than presentation bullets/steps.
        line = re.sub(r"^(?:\d{1,2}[.)]|[-*•])\s+", "", line)
        if not line:
            continue
        # An introductory heading is not itself a factual/technical assertion.
        if (line.endswith(":") and not any(op in line for op in ("=", "->"))
                and not re.match(r"^(?:update|set|change|edit|configure|replace|use)\b",
                                 line, flags=re.IGNORECASE)):
            continue
        # Only split at sentence boundaries outside quoted error messages;
        # IBM error strings may contain multiple punctuation marks.
        split_at = []
        for boundary in re.finditer(r"(?<=[.!?])\s+(?=[A-Z][a-zA-Z]|To\s|If\s)", line):
            prefix = line[:boundary.start()]
            if prefix.count('"') % 2 == 0:
                split_at.append(boundary)
        pieces = []
        pos = 0
        for boundary in split_at:
            pieces.append(line[pos:boundary.start()])
            pos = boundary.end()
        pieces.append(line[pos:])
        for piece in pieces:
            piece = piece.strip()
            if len(piece) >= 8:
                spans.append(piece)
            elif re.search(r"\b(?:rm|mv|cp|del|kill|drop|set|run|restart|reboot|" 
                           r"delete|clear|enable|disable|install)\b", piece,
                           flags=re.IGNORECASE):
                # A short command is still a technical claim; never silently
                # discard it merely because a previous splitter used len>=8.
                spans.append(piece)
    if not spans:
        return []
    # Deduplicate repeated identical instructions without discarding other text.
    unique = list(dict.fromkeys(spans))
    if len(unique) > MAX_CLAIMS or any(len(s) > MAX_CLAIM_LENGTH for s in unique):
        raise ValueError("TOO_MANY_OR_TOO_LONG_CLAIMS")
    essential_re = re.compile(
        r"^(?:the (?:correct|recommended|required)\b|to (?:work around|resolve|fix|configure)\b|"
        r"(?:you|users|administrators?) (?:should|must|need to)\b|"
        r"(?:navigate|edit|change|update|set|add|remove|restart|save|apply|use|run|"
        r"configure|enable|disable|install|upgrade|download|check|execute|open|"
        r"create|replace|select|enter|call|avoid|limit)\b)",
        flags=re.IGNORECASE,
    )
    claims = []
    for idx, span in enumerate(unique, 1):
        essential = bool(essential_re.search(span) or re.search(r"[A-Za-z_][\w-]*(?:\.[\w*+-]+){2,}|\w[\w.]*=\w", span))
        claims.append({"claim_id": idx, "text": span, "essential": essential})
    if not any(item["essential"] for item in claims):
        # Factual answers without commands still have essential claims.
        for item in claims:
            item["essential"] = True
    return claims


def _json_object_from_response(raw: str) -> dict:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        left, right = raw.find("{"), raw.rfind("}")
        if left < 0 or right <= left:
            raise ValueError("JSON_OBJECT_MISSING")
        try:
            obj = json.loads(raw[left:right + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("JSON_INVALID") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON_OBJECT_EXPECTED")
    return obj


def _source_body(doc: dict) -> str:
    body = str(doc.get("text") or doc.get("content") or "")
    if body.startswith("Title:") and "\n\nText:" in body:
        body = body.split("\n\nText:", 1)[1]
    return body


def parse_claim_verification(raw: str, doc: dict, claim: dict | None = None) -> dict:
    """Validate one model verdict; substring match is NOT entailment proof."""
    obj = _json_object_from_response(raw)
    for field in ("verdict", "quote", "reason"):
        if not isinstance(obj.get(field), str):
            raise ValueError("MISSING_OR_INVALID_" + field.upper())
    verdict = obj["verdict"].strip().upper()
    if verdict not in {"SUPPORTED", "CONTRADICTED", "NOT_FOUND"}:
        raise ValueError("INVALID_VERDICT")
    reason = obj["reason"].strip()
    if len(reason) < 8:
        raise ValueError("MISSING_CASE_SPECIFIC_REASON")
    quote = obj["quote"].strip()
    # Model sometimes inserts its document label despite explicit instructions.
    header = re.match(
        r"^\[Document\s+(\d+)\s*\|\s*([^|\]\r\n]+?)\s*\|\s*([^\]\r\n]*?)\s*\]\s*:\s*",
        quote, flags=re.IGNORECASE,
    )
    if header:
        if (int(header.group(1)) != 1
                or header.group(2).strip() != str(doc.get("filename", ""))
                or header.group(3).strip() != str(doc.get("chunk_id", ""))):
            raise ValueError("QUOTE_DOCUMENT_HEADER_MISMATCH")
        quote = quote[header.end():].strip()
    if verdict == "NOT_FOUND":
        if quote:
            raise ValueError("NOT_FOUND_MUST_HAVE_EMPTY_QUOTE")
        return {"verdict": verdict, "quote": "", "quote_verified": False,
                "reason": reason}
    if len(quote) < 12:
        raise ValueError("QUOTE_LENGTH_INVALID")
    body = _source_body(doc)
    if quote in body and len(quote) <= MAX_SOURCE_QUOTE_LENGTH:
        matched = quote
    else:
        # Use a real, continuous excerpt. Never manufacture the quoted text.
        candidates = []
        for piece in [quote, *quote.splitlines()]:
            recovered = _matching_source_span(body, piece)
            if recovered is not None:
                candidates.append(recovered)
        if not candidates:
            raise ValueError("QUOTE_NOT_IN_DOCUMENT_BODY")
        # Select the excerpt most relevant to THIS claim, not the first error
        # code or shared product name in an overlong model-generated quote.
        if claim is None:
            matched = max(candidates, key=len)
        else:
            claim_tokens = _claim_tokens(str(claim.get('text', '')))
            matched = max(candidates, key=lambda x: (
                len(claim_tokens & _claim_tokens(x)) / max(len(claim_tokens), 1),
                len(claim_tokens & _claim_tokens(x)), len(x)))
    if len(matched) > MAX_SOURCE_QUOTE_LENGTH:
        raise ValueError("QUOTE_LENGTH_INVALID")
    if claim is not None:
        problem = _quote_relevance_problem(matched, claim, verdict)
        if problem:
            raise ValueError(problem)
    return {"verdict": verdict, "quote": matched, "quote_verified": True,
            "reason": reason}


def build_claim_verify_prompt(question: str, answer: str, claim: dict, doc: dict) -> str:
    # Avoid putting the entire candidate answer in the prompt: Qwen 1.5B was
    # copying the answer and presenting it as a quote from the IBM document.
    return (
        "USER QUESTION (context, NOT evidence):\n" + question + "\n\n"
        "EXACT CLAIM TO CHECK (NOT evidence):\n" + claim["text"] + "\n\n"
        "DOCUMENT BODY (the only admissible evidence):\n" + _source_body(doc) + "\n\n"
        "Find a passage from DOCUMENT BODY that directly supports or contradicts "
        "the exact claim, including technical identifiers and the stated action. "
        "If the passage only mentions the same product or topic, return NOT_FOUND. "
        "Return a short, contiguous quotation copied directly from DOCUMENT BODY. "
        "Do NOT copy text from the CLAIM or the QUESTION. JSON only."
    )



# v3.2: Deterministically verify verbatim procedures and configuration values.
# This is deliberately a narrow shortcut, not a generic semantic entailment test.
# In particular, error explanations, root causes, and loose paraphrases still
# require the separate semantic Judge. No keyword-only shortcut is permitted.
def _normalized_procedure_words(text: str) -> list[str]:
    result = []
    for token in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+", text):
        token = token.casefold()
        if token in {"the", "a", "an", "to", "for", "of", "in", "on", "at"}:
            continue
        # Only safe plural variants used in ordinary prose. Preserve identifiers
        # and file/path extensions exactly (checked separately).
        if token in {"paths", "properties", "files", "folders"}:
            token = token[:-1] if token != "properties" else "property"
        result.append(token)
    return result


def _procedural_spans(body: str):
    """Return real source offsets for numbered procedure steps; never text from the answer.

    A single line can contain several numbered steps (IBM's technote HTML),
    so splitting only on newlines misses configuration properties 1/2/3.
    """
    boundary = re.compile(r"(?<![A-Za-z0-9_])\b\d{1,2}[.)]\s+(?=[A-Za-z_<])")
    positions = list(boundary.finditer(body))
    for i, m in enumerate(positions):
        end = positions[i+1].start() if i + 1 < len(positions) else len(body)
        raw = body[m.end():end]
        # Keep only one operation and at most the first paragraph to avoid
        # joining a procedure with unrelated FAQ sections.
        raw = raw.split("\n\n", 1)[0].strip()
        if raw:
            yield raw[:MAX_SOURCE_QUOTE_LENGTH]


def _question_context_matches_document(question: str, doc: dict) -> bool:
    """A copied instruction needs matching distinctive case context as well.

    Some RAG retrievals contain unrelated products with generic instructions
    such as 'save the file'. This gate is intentionally conservative: if the
    question has a specific error code or 3-part version, it must appear in
    the retrieved source (body or title) before deterministic approval.
    """
    codes = set(re.findall(r"\b[A-Z]{2,}[A-Z0-9_-]*[0-9][A-Z0-9_-]*\b", question))
    versions = set(re.findall(r"(?<!\d)\d+(?:\.\d+){2,}(?!\d)", question))
    markers = codes | versions
    if not markers:
        return True
    content = str(doc.get("text") or doc.get("content") or "").casefold()
    return any(marker.casefold() in content for marker in markers)


def _deterministic_procedure_support(claim: dict, doc: dict) -> dict | None:
    """Use only almost-verbatim numbered IBM instructions, never topical overlap.

    This may establish that a claim is stated in a *selected* IBM document,
    not that the document is authoritative, applicable, or factually correct.
    """
    text = str(claim.get("text") or "").strip().rstrip(" .:;!")
    if len(text) < 8:
        return None
    if not _question_context_matches_document(str(claim.get("question_context") or ""), doc):
        return None
    body = _source_body(doc)
    literal_set = _key_literals(text)
    for source in _procedural_spans(body):
        # A specific setting, number, version or path cannot be substituted.
        if any(lit.casefold() not in source.casefold() for lit in literal_set):
            continue
        # Explicit configuration assignments must match with the same value.
        claim_assignments = re.findall(r"(?<!\w)[A-Za-z_][\w.+-]*=[^\s,;]+", text)
        if claim_assignments and any(a not in source for a in claim_assignments):
            continue
        left, right = _normalized_procedure_words(text), _normalized_procedure_words(source)
        if len(left) < 2:
            continue
        # Preserve word order and operation. No contradictory polarity.
        if any(x in text.casefold().split() for x in ("not", "never", "without")) != any(
                x in source.casefold().split() for x in ("not", "never", "without")):
            continue
        if not right or left[0] != right[0]:
            # E.g. 'add fewer...' vs 'remove fewer...' must not match.
            # Config assignments may start with log4j... and match exactly.
            continue
        pos = 0
        for word in left:
            while pos < len(right) and right[pos] != word:
                pos += 1
            if pos >= len(right):
                break
            pos += 1
        else:
            # All content words appear in order; extra source words are fine,
            # but the same claim should not be inferred from a long source blob.
            if len(right) > len(left) + 8:
                continue
            actual = source.strip()
            if len(actual) < 12 or len(actual) > MAX_SOURCE_QUOTE_LENGTH:
                continue
            return {
                "verdict": "SUPPORTED", "quote": actual, "quote_verified": True,
                "reason": "Matched an ordered, numbered IBM procedure step with identical technical literals.",
                "verification_method": "exact_numbered_procedure",
            }
    return None


def _required_identifier_absent(claim: dict, doc: dict) -> bool:
    """Skip an IBM document missing an indispensable full technical identifier.

    Ordinary numbers or generic words are deliberately excluded: e.g. '1000'
    may appear as 'one thousand' or in another paragraph, and is not sufficient.
    """
    text = str(claim.get("text") or "")
    identifiers = []
    identifiers.extend(re.findall(r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_+-]*\.){2,}[A-Za-z_][A-Za-z0-9_*+-]*", text))
    identifiers.extend(re.findall(r"<[^<>]+>(?:/[A-Za-z0-9_.-]+)+", text))
    identifiers.extend(re.findall(r"(?<!\w)[A-Za-z_][\w.+-]*=[^\s,;]+", text))
    identifiers.extend(_required_claim_literals(text) - _key_literals(text))
    if not identifiers:
        return False
    body = _source_body(doc).casefold()
    return any(item.casefold() not in body for item in identifiers)


def _verify_claim_in_document(question, answer, claim, doc, generate_fn,
                              max_attempts=2, debug_hook=None):
    # No LLM call when the instruction is directly recoverable from the source.
    verified = _deterministic_procedure_support({**claim, "question_context": question}, doc)
    if verified is not None:
        if debug_hook is not None:
            debug_hook({"stage": "procedure_match", "claim_id": claim["claim_id"],
                        "claim": claim["text"], "essential": claim["essential"],
                        "source_filename": doc.get("filename"),
                        "source_chunk_id": doc.get("chunk_id"),
                        "attempt": 0, "messages": [], "raw_response": "",
                        "parsed_verdict": verified, "problem": None})
        return verified
    # Missing exact module/path/setting is not a contradiction, just no proof.
    if _required_identifier_absent(claim, doc):
        return {"verdict": "NOT_FOUND", "quote": "", "quote_verified": False,
                "reason": "The required technical identifier is absent from this document.",
                "verification_method": "literal_absence"}
    messages = [
        {"role": "system", "content": CLAIM_VERIFY_SYSTEM},
        {"role": "user", "content": build_claim_verify_prompt(question, answer, claim, doc)},
    ]
    for attempt in range(1, max_attempts + 1):
        raw = generate_fn(messages)
        problem = None
        try:
            result = parse_claim_verification(raw, doc, claim)
        except (ValueError, KeyError) as exc:
            result = None
            problem = str(exc)
        if debug_hook is not None:
            debug_hook({
                "stage": "claim_verify", "claim_id": claim["claim_id"],
                "claim": claim["text"], "essential": claim["essential"],
                "source_filename": doc.get("filename"),
                "source_chunk_id": doc.get("chunk_id"),
                "attempt": attempt, "messages": [dict(m) for m in messages],
                "raw_response": raw, "parsed_verdict": result, "problem": problem,
            })
        if result is not None:
            return result
        if attempt < max_attempts:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    "The previous verdict was INVALID: " + problem + ". "
                    "Use the exact DOCUMENT BODY in the original question. "
                    "Only SUPPORTED/CONTRADICTED may provide an exact passage copied "
                    "from that body; NOT_FOUND must have an empty quote. "
                    "All fields verdict, quote, reason are required. Return JSON only."
                ),
            })
    return {"verdict": "UNRESOLVED", "quote": "", "quote_verified": False,
            "reason": "Invalid model output after retry: " + (problem or "unknown")}


def derive_claim_rubric(claims: list[dict], policy: dict) -> tuple[dict, str]:
    """Compute auditable, conservative scores from *verified* claim statuses.

    These scores are proxy judgments, not independently established truth.
    Unverified model verdicts cannot be silently converted to a passing score.
    """
    core = [c for c in claims if c["essential"]]
    # A flagged contradiction is an *unresolved semantic judgment*, not a
    # demonstrated factual contradiction. Keep the raw model verdict on each
    # claim for the audit log, but do not turn it into a certain score of 1.
    def effective_verdict(c):
        return "UNRESOLVED" if c.get('review_flags') else c['verdict']

    all_claims = [effective_verdict(c) for c in claims]
    core_verdicts = [effective_verdict(c) for c in core]
    unknown_core = sum(v != "SUPPORTED" for v in core_verdicts)
    contradictory = any(v == "CONTRADICTED" for v in all_claims)
    unresolved = any(v == "UNRESOLVED" for v in all_claims)
    missing_context = sum(v == "NOT_FOUND" for c, v in zip(claims, all_claims)
                          if not c["essential"])
    fully_supported = bool(claims) and all(v == "SUPPORTED" for v in all_claims)
    primary_supported = bool(core) and unknown_core == 0
    flags = []
    if contradictory:
        flags.append("CLAIM_CONTRADICTED")
    if unresolved:
        flags.append("CLAIM_VERIFICATION_UNRESOLVED")
    if unknown_core:
        flags.append("ESSENTIAL_CLAIM_NOT_SUPPORTED")
    if missing_context:
        flags.append("CONTEXT_CLAIM_NOT_SUPPORTED")
    # Review requirements raised by evidence-ID repair and contradiction
    # consistency checks must propagate to the final case gate.
    semantic_review_flags = sorted({flag for c in claims for flag in c.get('review_flags', [])})
    flags.extend(semantic_review_flags)
    if fully_supported:
        correctness, groundedness, completeness, actionability = (5, 5, 4, 4)
    elif primary_supported and not contradictory and not unresolved:
        correctness, groundedness, completeness, actionability = (3, 3, 3, 3)
    elif contradictory:
        correctness, groundedness, completeness, actionability = (1, 1, 2, 2)
    elif unknown_core:
        correctness, groundedness, completeness, actionability = (2, 2, 2, 2)
    else:
        correctness, groundedness, completeness, actionability = (3, 3, 3, 3)
    quotes = [c["evidence_quote"] for c in claims if effective_verdict(c) == "SUPPORTED"
              and c.get("evidence_quote_verified")]
    failures = [f"claim {c['claim_id']}: {effective_verdict(c)} - {c['reason']}"
                for c in claims if effective_verdict(c) != "SUPPORTED"]
    thresholds = policy["quality_thresholds"]
    scores = {"correctness": correctness, "groundedness": groundedness,
              "completeness": completeness, "actionability": actionability,
              # This is not an independent safety audit; enforcement remains
              # disabled until calibrated human labels are available.
              "safety": "pass", "format": "pass",
              "safety_scope": "not_independently_audited",
              "evidence_quote": quotes[0] if quotes else "",
              "evidence_quote_verified": bool(quotes),
              "failure_reason": "; ".join(failures),
              "risk_level": "high" if contradictory else ("medium" if flags else "low"),
              "review_flags": flags, "claim_verifications": claims,
              "rubric_derivation": "deterministic_from_per_document_claim_verdicts",
              "score_is_model_calibrated": False}
    pass_thresholds = all((
        correctness >= thresholds["correctness_min"],
        groundedness >= thresholds["groundedness_min"],
        completeness >= thresholds["completeness_min"],
        actionability >= thresholds["actionability_min"],
        scores["safety"] == thresholds["safety_required"],
        scores["format"] == thresholds["format_required"],
    ))
    # Fail closed: even one unsupported contextual technical claim bars automatic
    # approval. Human calibration may justify a more nuanced policy later.
    scores["auto_pass"] = bool(pass_thresholds and fully_supported
                                and not contradictory and not unresolved and not flags)
    status = "REVIEW_REQUIRED" if (unresolved or semantic_review_flags) else "OK"
    return scores, status


# v3.3: Evidence-first semantic verification.
# Python selects verbatim source passages; the model NEVER writes citations.
# One Qwen call per claim, with at most one retry for malformed JSON.
# The selected source span is an exact substring of one document.
MAX_EVIDENCE_SNIPPETS = 5
MAX_PASSAGE_CHARS = 430

EVIDENCE_FIRST_SYSTEM = """You check ONE claim against numbered IBM source passages.
Only E1, E2, ... are evidence. QUESTION and CLAIM are NOT evidence.
Evaluate the exact action, product, error, version, numbers, and conditions.

SUPPORTED: a passage DIRECTLY establishes EVERY MATERIAL PART of the claim.
For a claimed remedy or instruction, a passage that mentions only a symptom,
root cause, or desirable outcome is NOT enough; the action must be supported.
Do not approve a claim about installing, deleting, renaming, restarting, or
reconfiguring something from an excerpt that does not discuss that action.
Do not approve a claim with a specific path/version/number unless the source
supports that same detail. If a quote covers just part of a compound claim,
return NOT_FOUND. Matching topic and keyword overlap alone are not support.
CONTRADICTED: a passage explicitly states an incompatible fact (opposite
polarity, mutually exclusive parameter/value or action). The claim and source
saying the SAME thing is NEVER CONTRADICTED, even with different wording.
NOT_FOUND: no passage directly proves either verdict; similarity is not enough.

Return exactly ONE JSON object, with three keys:
{"verdict":"SUPPORTED","evidence_id":"E1","reason":"brief case-specific explanation"}
When verdict is SUPPORTED or CONTRADICTED: use EXACTLY ONE existing E number
shown in the prompt. NEVER return a blank or made-up evidence_id.
When verdict is NOT_FOUND: use evidence_id "" and explain missing evidence.
Do not invent quotations; source text is copied by the program.
"""


def _passage_windows(body: str) -> list[str]:
    """Generate genuine contiguous source windows, never model-made quotes."""
    body = body.strip()
    if not body:
        return []
    lines = [x for x in body.splitlines(keepends=True)]
    # Preserve paragraph content; split exceptionally long lines at boundaries.
    pieces = []
    for line in lines:
        if len(line) <= MAX_PASSAGE_CHARS:
            pieces.append(line)
        else:
            # Slices remain contiguous within the original source line.
            for offset in range(0, len(line), MAX_PASSAGE_CHARS):
                pieces.append(line[offset:offset + MAX_PASSAGE_CHARS])
    candidates = []
    for i in range(len(pieces)):
        if not pieces[i].strip():
            continue
        for width in (1, 2, 3, 4, 5):
            if i + width > len(pieces):
                break
            sample = ''.join(pieces[i:i + width]).strip()
            if 15 <= len(sample) <= MAX_PASSAGE_CHARS:
                candidates.append(sample)
    # Preserve document order and exact source membership.
    return list(dict.fromkeys(candidates))


def _evidence_relevance(claim: dict, excerpt: str) -> float:
    """Retrieve likely evidence, NOT a conclusion about semantic entailment."""
    words = _claim_tokens(claim["text"])
    matches = words & _claim_tokens(excerpt)
    if not matches:
        return 0.0
    literals = _key_literals(claim["text"])
    literal_hits = sum(x.casefold() in excerpt.casefold() for x in literals)
    # More matching claim words and identifier hits; penalize excessive text.
    return (2.0 * len(matches) / max(len(words), 1)
            + 0.08 * len(matches)
            + 0.5 * literal_hits
            - 0.0005 * len(excerpt))


def _candidate_evidence(question: str, claim: dict, docs: list[dict], limit=MAX_EVIDENCE_SNIPPETS):
    """Return original substrings with genuine document identifiers."""
    ranked = []
    for pos, doc in enumerate(docs):
        if not _question_context_matches_document(question, doc):
            continue
        if _required_identifier_absent(claim, doc):
            continue
        body = _source_body(doc)
        scored = [(_evidence_relevance(claim, x), x) for x in _passage_windows(body)]
        scored.sort(key=lambda t: t[0], reverse=True)
        # At most two passages from a given document; no unrelated source flood.
        for score, quote in scored[:2]:
            if score > 0:
                ranked.append((score, pos, quote, doc))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    chosen = []
    seen = set()
    for score, pos, quote, doc in ranked:
        key = (str(doc.get('chunk_id')), quote)
        if key in seen:
            continue
        seen.add(key)
        chosen.append({"evidence_id": f"E{len(chosen) + 1}",
                       "quote": quote, "source_filename": doc.get('filename'),
                       "source_chunk_id": doc.get('chunk_id'),
                       "rank_score": round(score, 4)})
        if len(chosen) >= limit:
            break
    return chosen


def _evidence_first_messages(question: str, claim: dict, evidence: list[dict]):
    fragments = [f"{x['evidence_id']} ({x['source_filename']} | {x['source_chunk_id']}):\n{x['quote']}"
                 for x in evidence]
    user = ("QUESTION (context, not evidence):\n" + question +
            "\n\nCLAIM TO VERIFY (not evidence):\n" + claim['text'] +
            "\n\nSOURCE EVIDENCE (the only admissible text):\n" +
            "\n\n".join(fragments) +
            "\n\nDecide the claim, not the source's general topic. JSON only.")
    return [{"role": "system", "content": EVIDENCE_FIRST_SYSTEM},
            {"role": "user", "content": user}]


def _required_claim_literals(claim_text: str) -> set[str]:
    """Identifiers required to support a claim; exclude source reference numbers.

    Long all-capital module names such as CSQADITB must not silently match an
    unrelated IBM module (CSQQCON2) just because the error code is identical.
    """
    content = re.sub(r"\b(?:Document|Source)\s+\d+\b", "", claim_text, flags=re.IGNORECASE)
    literals = _key_literals(content)
    generic_caps = {"ERROR", "CAUSE", "REQUESTS", "MODULE", "COUNT", "EXCEED",
                    "MAXIMUM", "APPLICATION", "RESTART", "RESOLVING", "PROBLEM",
                    "INSTALL", "CONFIGURE", "WARNING", "VERSION", "VERSIONS",
                    "AFFECTED", "PRODUCTS", "SERVICE", "MESSAGE"}
    literals.update(x for x in re.findall(r"(?<![\w.])[A-Z][A-Z0-9_]{5,}(?![\w.])", content)
                    if x not in generic_caps)
    return literals


def _recover_unique_evidence_id(reason: str, evidence: list[dict]) -> str | None:
    """Recover a *reference* only when it has exactly one verifiable source.

    Recovery establishes which passage was referenced; it is NOT proof that
    a semantic SUPPORTED/CONTRADICTED verdict is correct. Never guess among
    several passages by lexical similarity alone.
    """
    valid_ids = {str(item['evidence_id']) for item in evidence}
    mentioned_ids = set(re.findall(r"(?<![A-Za-z0-9_])E\d+(?![A-Za-z0-9_])",
                                   reason, flags=re.IGNORECASE))
    mentioned_ids = {item.upper() for item in mentioned_ids}
    if len(mentioned_ids) == 1 and mentioned_ids <= valid_ids:
        return next(iter(mentioned_ids))
    if mentioned_ids:
        return None  # Ambiguous or invalid reference in the explanation.
    if len(evidence) == 1:
        return str(evidence[0]['evidence_id'])
    # Some responses contain an exact quoted passage in the explanation,
    # although evidence_id is blank. Match only a substantial verbatim span.
    fragments = re.findall(r'["\']([^"\']{24,})["\']', reason)
    matches = set()
    for fragment in fragments:
        for item in evidence:
            if normalize_space(fragment) in normalize_space(item['quote']):
                matches.add(str(item['evidence_id']))
    return next(iter(matches)) if len(matches) == 1 else None


def _contradiction_review_signal(claim_text: str, quote: str, reason: str) -> str | None:
    """Detect *suspicious* contradictions; never auto-convert to SUPPORTED.

    Overlap is a review signal, not an entailment/contradiction proof. A small
    LLM can produce a convincing reason that does not follow from the quote.
    """
    claim_tokens = _claim_tokens(claim_text)
    quote_tokens = _claim_tokens(quote)
    if not claim_tokens or not quote_tokens:
        return None
    # If the original source literally contains the asserted sentence, asking
    # a human is safer than accepting the model's contradictory classification.
    if len(claim_text.strip()) >= 20 and normalize_space(claim_text.strip()) in normalize_space(quote):
        return 'CLAIM_TEXT_PRESENT_IN_CONTRADICTING_QUOTE'
    overlap = len(claim_tokens & quote_tokens) / max(len(claim_tokens), 1)
    negatives = re.compile(r"\b(?:no|not|never|cannot|can't|without|unsupported|"
                           r"impossible|unavailable|unable|forbidden|disallowed)\b",
                           flags=re.IGNORECASE)
    # Detect high lexical agreement without a basic polarity flip. Differences
    # in wording or extra source context are not proof of contradiction.
    if (overlap >= 0.67 and len(claim_tokens & quote_tokens) >= 3
            and bool(negatives.search(claim_text)) == bool(negatives.search(quote))):
        return 'POTENTIAL_FALSE_CONTRADICTION'
    # An explanation that describes agreement while voting CONTRADICTED is
    # internally inconsistent, even if the source quote is authentic.
    agreement = re.search(
        r"\b(?:supports? (?:the |this )?claim|directly supports?|"
        r"aligns? with (?:the |this )?claim|matches? (?:the |this )?claim)\b",
        reason, flags=re.IGNORECASE)
    if agreement:
        return 'VERDICT_REASON_INCONSISTENT'
    return None


# Lexical guards below only trigger HUMAN REVIEW; they never declare semantic
# entailment or contradiction. This is important for a small local LLM Judge.
_ACTION_PATTERNS = {
    "restart": r"\b(?:restart|reboot|restarting|rebooting|restarted|rebooted)\b",
    "install": r"\b(?:install|installs|installed|installing|reinstall|reinstalled)\b",
    "delete": r"\b(?:delete|deleting|deleted|remove|removed|removing|clear|cleared|clearing|purge|purged|wipe|wiped|drop|dropped|dropping)\b",
    "rename": r"\b(?:rename|renamed|renaming|move|moved|moving)\b",
    "configure": r"\b(?:configure|configured|configuration|reconfigure|update|updated|updating|change|changed|changing|modify|modified|edit|edited|editing|set|setting)\b",
    "generate": r"\b(?:generate|generating|generated|create|created|creating)\b",
    "enable": r"\b(?:enable|enabled|enabling)\b",
    "disable": r"\b(?:disable|disabled|disabling)\b",
    "download": r"\b(?:download|downloaded|downloading)\b",
    "upgrade": r"\b(?:upgrade|upgraded|upgrading)\b",
    "execute": r"\b(?:execute|executed|executing|run|running|ran)\b",
}

_NUMBER_COMPARISONS = re.compile(
    r"\b(less than|fewer than|under|below|at most|no more than|"
    r"more than|greater than|over|above|at least|no fewer than)\s+"
    r"(\d[\d,]*(?:\.\d+)?)\b", flags=re.IGNORECASE)


def _number_comparators(text: str) -> dict[str, set[str]]:
    comparisons = {}
    for phrase, amount in _NUMBER_COMPARISONS.findall(text):
        amount = amount.replace(',', '')
        if phrase.casefold() in ('less than', 'fewer than', 'under', 'below'):
            symbol = '<'
        elif phrase.casefold() in ('at most', 'no more than'):
            symbol = '<='
        elif phrase.casefold() in ('more than', 'greater than', 'over', 'above'):
            symbol = '>'
        else:
            symbol = '>='
        comparisons.setdefault(amount, set()).add(symbol)
    return comparisons


def _support_review_signals(claim_text: str, quote: str) -> list[str]:
    """Conservative review triggers, NOT automatic semantic rejection.

    An exact source quotation verifies provenance, not entailment. A semantic
    model's SUPPORTED decision must not become an automatic approval when the
    source passage omits most of the asserted content or a specific local path.
    Paraphrases can be valid, so suspicious positives become REVIEW_REQUIRED
    rather than being relabeled as CONTRADICTED or NOT_FOUND.
    """
    flags = []
    tokens = _claim_tokens(claim_text)
    supported_tokens = _claim_tokens(quote)
    shared = tokens & supported_tokens
    if tokens and (len(shared) < (1 if len(tokens) == 1 else 2)
                   or len(shared) / len(tokens) < 0.48):
        flags.append('SUPPORTED_QUOTE_WEAK_ALIGNMENT')

    # _required_claim_literals guards versions, modules and dotted keys. Check
    # absolute paths separately because /tmp/foo is otherwise just common words.
    paths = re.findall(r'(?<![\w/])/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+', claim_text)
    paths += re.findall(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:[^\s,;]+)', claim_text)
    if any(path.casefold() not in quote.casefold() for path in paths):
        flags.append('SUPPORTED_PATH_MISSING_FROM_QUOTE')

    # Requiring the action itself, not only its context, prevents a causal
    # description such as 'old class cache' from authorizing 'delete files'.
    # An inference from symptom to remedy is not direct source support.
    missing_actions = [name for name, pattern in _ACTION_PATTERNS.items()
                       if re.search(pattern, claim_text, flags=re.IGNORECASE)
                       and not re.search(pattern, quote, flags=re.IGNORECASE)]
    if missing_actions:
        flags.append('SUPPORTED_ACTION_NOT_IN_QUOTE')

    claim_comp = _number_comparators(claim_text)
    source_comp = _number_comparators(quote)
    for amount, claim_ops in claim_comp.items():
        if amount in source_comp and not (claim_ops & source_comp[amount]):
            flags.append('SUPPORTED_NUMERIC_CONDITION_MISMATCH')
            break
    return flags


def _parse_evidence_first(raw: str, evidence: list[dict], claim: dict) -> dict:
    obj = _json_object_from_response(raw)
    verdict = obj.get('verdict')
    evidence_id = obj.get('evidence_id')
    reason = obj.get('reason')
    if not isinstance(verdict, str) or verdict.strip().upper() not in {'SUPPORTED', 'CONTRADICTED', 'NOT_FOUND'}:
        raise ValueError('INVALID_VERDICT')
    verdict = verdict.strip().upper()
    if not isinstance(evidence_id, str) or not isinstance(reason, str) or len(reason.strip()) < 8:
        raise ValueError('INVALID_EVIDENCE_ID_OR_REASON')
    evidence_id = evidence_id.strip()
    if verdict == 'NOT_FOUND':
        if evidence_id:
            raise ValueError('NOT_FOUND_MUST_HAVE_NO_EVIDENCE')
        return {'verdict': 'NOT_FOUND', 'quote': '', 'quote_verified': False,
                'reason': reason.strip(), 'verification_method': 'evidence_first_llm'}
    selected = next((e for e in evidence if e['evidence_id'] == evidence_id), None)
    recovered_id = False
    if selected is None and not evidence_id:
        # Missing ID is a formatting failure; only uniquely identifiable source
        # references can be repaired without a second semantic judgment.
        inferred = _recover_unique_evidence_id(reason, evidence)
        if inferred is not None:
            selected = next(e for e in evidence if e['evidence_id'] == inferred)
            evidence_id = inferred
            recovered_id = True
    if selected is None:
        raise ValueError('EVIDENCE_ID_NOT_IN_CANDIDATES')
    # No model-created quote: selected['quote'] was copied directly from a source.
    # Keep exact technical identifiers as an additional necessary check for support.
    if verdict == 'SUPPORTED' and any(x.casefold() not in selected['quote'].casefold()
                                      for x in _required_claim_literals(claim['text'])):
        return {'verdict': 'NOT_FOUND', 'quote': '', 'quote_verified': False,
                'reason': 'The chosen passage lacks a required technical identifier.',
                'verification_method': 'literal_guard',
                'evidence_id_recovered': recovered_id}
    review_flags = []
    if verdict == 'SUPPORTED':
        review_flags.extend(_support_review_signals(claim['text'], selected['quote']))
    if verdict == 'CONTRADICTED':
        suspicion = _contradiction_review_signal(claim['text'], selected['quote'], reason)
        if suspicion:
            review_flags.append(suspicion)
        # A quotation's exact provenance does not verify the Judge's inference
        # of logical incompatibility. False contradictions in the 20-case run
        # make such verdicts unsuitable for an automatic *definitive* failure.
        review_flags.append('CONTRADICTION_NOT_INDEPENDENTLY_CONFIRMED')
    return {'verdict': verdict, 'quote': selected['quote'],
            'quote_verified': True, 'reason': reason.strip(),
            'verification_method': 'evidence_first_llm',
            'source_filename': selected['source_filename'],
            'source_chunk_id': selected['source_chunk_id'],
            'evidence_id': evidence_id,
            'evidence_id_recovered': recovered_id,
            'review_flags': review_flags}


def _verify_claim_evidence_first(question, claim, docs, generate_fn,
                                 max_attempts=2, debug_hook=None):
    # First use narrow non-LLM support for almost-verbatim numbered procedures.
    for doc in docs:
        deterministic = _deterministic_procedure_support(
            {**claim, 'question_context': question}, doc)
        if deterministic is not None:
            if debug_hook:
                debug_hook({'stage': 'procedure_match', 'claim_id': claim['claim_id'],
                            'claim': claim['text'], 'essential': claim['essential'],
                            'source_filename': doc.get('filename'),
                            'source_chunk_id': doc.get('chunk_id'),
                            'parsed_verdict': deterministic})
            return {**deterministic, 'source_filename': doc.get('filename'),
                    'source_chunk_id': doc.get('chunk_id'),
                    'documents_checked': 1, 'per_document_verdicts': [deterministic]}, 0, []
    evidence = _candidate_evidence(question, claim, docs)
    if not evidence:
        return {'verdict': 'NOT_FOUND', 'quote': '', 'quote_verified': False,
                'reason': 'No retrievable source passage overlaps this claim.',
                'verification_method': 'no_candidate_passage',
                'documents_checked': len(docs), 'per_document_verdicts': []}, 0, []
    messages = _evidence_first_messages(question, claim, evidence)
    attempts = []
    previous_declared_verdicts = []
    for attempt in range(1, max_attempts + 1):
        raw = generate_fn(messages)
        problem = None
        try:
            result = _parse_evidence_first(raw, evidence, claim)
        except (ValueError, KeyError) as exc:
            result = None
            problem = str(exc)
            # Preserve the model's first *semantic* verdict even when its
            # reference format is invalid. A retry must not quietly change an
            # earlier SUPPORTED finding into a confident CONTRADICTED finding.
            try:
                earlier_verdict = _json_object_from_response(raw).get('verdict')
                if (isinstance(earlier_verdict, str)
                        and earlier_verdict.strip().upper() in
                        {'SUPPORTED', 'CONTRADICTED', 'NOT_FOUND'}):
                    previous_declared_verdicts.append(earlier_verdict.strip().upper())
            except ValueError:
                pass
        if result is not None and any(
                old_verdict != result['verdict'] for old_verdict in previous_declared_verdicts):
            result.setdefault('review_flags', []).append('VERDICT_CHANGED_ON_RETRY')
            result['previous_unverified_verdicts'] = list(previous_declared_verdicts)
        event = {'stage': 'claim_verify', 'claim_id': claim['claim_id'],
                 'claim': claim['text'], 'essential': claim['essential'],
                 'source_filename': evidence[0]['source_filename'],
                 'source_chunk_id': evidence[0]['source_chunk_id'],
                 'attempt': attempt, 'messages': [dict(m) for m in messages],
                 'raw_response': raw, 'parsed_verdict': result, 'problem': problem,
                 'candidate_evidence': evidence}
        attempts.append({'stage': 'claim_verify', 'claim_id': claim['claim_id'],
                         'source_chunk_id': evidence[0]['source_chunk_id'],
                         'attempt': attempt, 'raw_response': raw, 'problem': problem})
        if debug_hook:
            debug_hook(event)
        if result is not None:
            return {**result, 'source_filename': result.get('source_filename'),
                    'source_chunk_id': result.get('source_chunk_id'),
                    'documents_checked': len({e['source_chunk_id'] for e in evidence}),
                    'per_document_verdicts': [result]}, attempt, attempts
        if attempt < max_attempts:
            valid_ids = ', '.join(item['evidence_id'] for item in evidence)
            # Start from the ORIGINAL evidence, rather than appending the
            # defective model completion as an assistant turn. The latter made
            # Qwen 1.5B flip its semantic verdict merely to repair a blank ID.
            messages = _evidence_first_messages(question, claim, evidence)
            original_verdict = previous_declared_verdicts[0] if previous_declared_verdicts else None
            messages.append({'role': 'user', 'content':
                             f'Previous output was invalid ({problem}). Available IDs: '
                             f'{valid_ids}. The previous semantic verdict was '
                             f'{original_verdict or "unparseable"}; do NOT reverse it just '
                             'because of an ID formatting error. Identify the ONE '
                             'source passage supporting the verdict. If no '
                             'source actually proves the verdict, use NOT_FOUND '
                             'with evidence_id "". Otherwise use exactly one '
                             'listed E number. Return only the three-key JSON.'})
    # Format failure is NOT evidence that the claim is false.
    unresolved = {'verdict': 'UNRESOLVED', 'quote': '', 'quote_verified': False,
                  'reason': 'Malformed semantic verdict after bounded retry.',
                  'verification_method': 'model_output_invalid',
                  'documents_checked': len({e['source_chunk_id'] for e in evidence}),
                  'per_document_verdicts': []}
    return unresolved, len(attempts), attempts


def evaluate_case_claims(record, policy, generate_fn, max_attempts=2, debug_hook=None):
    """v3.6: Evidence-first Judge; uncertain judgments require human review."""
    docs = evaluation_docs(record)
    answer = str(record.get('prediction') or '')
    question = str(record.get('question') or '')
    if answer.strip() == ABSTAIN:
        score = {'auto_pass': False, 'review_flags': ['REFUSAL_REQUIRES_HUMAN_REVIEW'],
                 'claim_verifications': [], 'failure_reason': 'Refusal needs human review',
                 'rubric_derivation': 'not_scored_refusal'}
        return {'judge_status': 'REVIEW_REQUIRED', 'score': score,
                'raw_response': '', 'attempt_count': 0, 'attempts': []}
    try:
        claims = extract_answer_claims(answer)
    except ValueError as exc:
        claims = []
        extraction_error = str(exc)
    else:
        extraction_error = 'NO_CLAIMS_EXTRACTED'
    if not claims:
        score = {'auto_pass': False, 'review_flags': [extraction_error],
                 'claim_verifications': [], 'failure_reason': extraction_error,
                 'rubric_derivation': 'extraction_failed'}
        return {'judge_status': 'REVIEW_REQUIRED', 'score': score,
                'raw_response': '', 'attempt_count': 0, 'attempts': []}
    verified_claims = []
    attempt_log = []
    total_calls = 0
    for claim in claims:
        checked, count, attempts = _verify_claim_evidence_first(
            question, claim, docs, generate_fn,
            max_attempts=min(max_attempts, 2), debug_hook=debug_hook)
        total_calls += count
        attempt_log.extend(attempts)
        verified_claims.append({**claim, **checked,
                                'evidence_quote': checked.get('quote', ''),
                                'evidence_quote_verified': checked.get('quote_verified', False)})
    score, status = derive_claim_rubric(verified_claims, policy)
    return {'judge_status': status, 'score': score,
            'raw_response': attempt_log[-1]['raw_response'] if attempt_log else '',
            'attempt_count': total_calls, 'attempts': attempt_log}


def grade(args):
    """Evaluate on local Qwen; existing v1 records regraded after schema upgrade."""
    rows = jsonl_rows(Path(args.predictions))
    output = Path(args.output)
    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[:args.limit]
    saved = {r["id"]: r for r in jsonl_rows(output)} if output.exists() and output.stat().st_size else {}
    force = getattr(args, "force", False)
    to_eval = []
    for row in rows:
        prev = saved.get(row["id"])
        if prev is None or prev.get("judge_schema_version") != JUDGE_SCHEMA_VERSION or force:
            to_eval.append(row)
        else:
            fingerprint = hashlib.sha256(build_grade_prompt(row).encode("utf8")).hexdigest()
            if prev.get("source_fingerprint") != fingerprint:
                raise RuntimeError(f"평가 입력이 변경됨: {row['id']}. --force 옵션 또는 새 --output 필요")
    if not to_eval:
        print("[grade] 모두 평가 완료(현재 Judge 스키마 결과 재사용)")
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT / model_path
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.float16, local_files_only=True
    ).to("cuda")
    model.eval()
    # do_sample=False must not carry sampling-only settings from the model's
    # saved generation_config (otherwise Transformers emits a warning).
    for name in ('temperature', 'top_p', 'top_k'):
        if hasattr(model.generation_config, name):
            setattr(model.generation_config, name, None)
    max_attempts = getattr(args, "max_attempts", 3)
    output.parent.mkdir(parents=True, exist_ok=True)

    debug_enabled = getattr(args, "debug", False)
    debug_log_path = Path(args.debug_log) if getattr(args, "debug_log", None) else output.parent / "judge_debug.jsonl"
    generation_metadata = {}
    if debug_enabled:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        # A new debug run starts a new trace; normal grading cache remains intact.
        debug_log_path.write_text("", encoding="utf-8")
        print(f"[DEBUG] 전체 프롬프트/청크/원본 응답 로그: {debug_log_path}")
        print("[DEBUG] 로그에 질문, 모델 답변, IBM 문서 본문이 포함됩니다. 외부 공개에 주의하세요.")

    def generate_response(messages):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        input_length = encoded["input_ids"].shape[-1]
        completion_ids = generated[0, input_length:]
        if debug_enabled:
            token_count = int(completion_ids.shape[-1])
            eos_ids = tokenizer.eos_token_id
            eos_seen = bool(eos_ids is not None and (completion_ids == eos_ids).any().item())
            generation_metadata.clear()
            generation_metadata.update({
                "input_token_count": int(input_length),
                "generated_token_count": token_count,
                "max_new_tokens": args.max_new_tokens,
                "eos_seen": eos_seen,
                "may_be_truncated_by_max_new_tokens": token_count >= args.max_new_tokens and not eos_seen,
            })
        return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    for i, row in enumerate(to_eval, 1):
        debug_attempts = []
        docs = evaluation_docs(row) if debug_enabled else []
        if debug_enabled:
            print(f"\n[DEBUG] QA={row['id']} | Judge 문서 수={len(docs)} | "
                  f"문서 길이={[len(str(d.get('text', ''))) for d in docs]}")
            for k, doc in enumerate(docs, 1):
                print(f"  [DOC {k}] {doc.get('filename')} | {doc.get('chunk_id')} | "
                      f"본문 {len(str(doc.get('text','')))}자")

        def capture_debug(event):
            if not debug_enabled:
                return
            item = dict(event)
            item["generation_metadata"] = dict(generation_metadata)
            debug_attempts.append(item)
            parsed = event.get("parsed_verdict") or {}
            print(f"  [CLAIM {event.get('claim_id')}] {event.get('claim', '')[:130]}")
            print(f"    [DOC] {event.get('source_filename')} | {event.get('source_chunk_id')}")
            if event.get("stage") == "procedure_match":
                print("    [DIRECT EVIDENCE] Numbered IBM procedure, exact technical literals")
                print(f"    [VERDICT] {json.dumps(parsed, ensure_ascii=False)}")
                return
            print(f"    [ATTEMPT {event['attempt']}] {generation_metadata} | "
                  f"validation_problem={event.get('problem')!r}")
            print(f"    [VERDICT] {json.dumps(parsed, ensure_ascii=False)}")
            print(f"    [RAW] {event['raw_response'][:1100]}")

        evaluated = evaluate_case_claims(
            row, policy, generate_response, max_attempts=max_attempts,
            debug_hook=capture_debug if debug_enabled else None,
        )
        if debug_enabled:
            trace = {
                "id": row["id"],
                "v4_guardrail_status": row.get("guardrails", {}).get("status"),
                "question": row.get("question", ""),
                "candidate_response": row.get("prediction", ""),
                "selected_document_count": len(docs),
                "selected_documents": [
                    {"filename": d.get("filename"), "chunk_id": d.get("chunk_id"),
                     "text": d.get("text", "")} for d in docs
                ],
                "system_prompt": RUBRIC_SYSTEM,
                "user_prompt": build_grade_prompt(row),
                "attempts": debug_attempts,
                "final_status": evaluated["judge_status"],
                "final_auto_pass": evaluated["score"].get("auto_pass"),
            }
            with debug_log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(trace, ensure_ascii=False) + "\n")
            print(f"  [CLAIM SUMMARY] " + json.dumps([
                {"claim_id": c["claim_id"], "essential": c["essential"],
                 "verdict": c["verdict"], "source": c["source_filename"]}
                for c in evaluated["score"].get("claim_verifications", [])
            ], ensure_ascii=False))
            print(f"  [DERIVED RUBRIC] " + json.dumps({
                k: evaluated["score"].get(k) for k in (
                    "correctness", "groundedness", "completeness", "actionability",
                    "auto_pass", "review_flags", "failure_reason")
            }, ensure_ascii=False))
            print(f"  [DEBUG RESULT] {row['id']} -> "
                  f"{evaluated['judge_status']} | auto_pass={evaluated['score'].get('auto_pass')}")
        result = {
            "id": row["id"],
            "judge_model": str(model_path),
            "judge_schema_version": JUDGE_SCHEMA_VERSION,
            "rubric_version": "ibm-techqa-claim-derived-v3.6",
            "policy_rubric_version": policy.get("rubric_version"),
            **evaluated,
            "source_fingerprint": hashlib.sha256(build_grade_prompt(row).encode("utf8")).hexdigest(),
        }
        saved[row["id"]] = result
        # Safe rewrite avoids duplicates when regrading an existing output.
        temp_path = output.with_name(output.name + ".tmp")
        write_jsonl(temp_path, list(saved.values()))
        temp_path.replace(output)
        print(f"[{i}/{len(to_eval)}] {row['id']}: {evaluated['judge_status']}, "
              f"auto_pass={evaluated['score'].get('auto_pass')}, "
              f"attempts={evaluated['attempt_count']}")
    print(f"[grade] {output}")



def safe_rate(n,d):
    return round(n/d,4) if d else None


def confusion(human, pred):
    tp=sum(h==1 and p==1 for h,p in zip(human,pred))
    fp=sum(h==0 and p==1 for h,p in zip(human,pred))
    tn=sum(h==0 and p==0 for h,p in zip(human,pred))
    fn=sum(h==1 and p==0 for h,p in zip(human,pred))
    n=tp+fp+tn+fn
    po=(tp+tn)/n if n else 0
    p_h=(tp+fn)/n if n else 0
    p_j=(tp+fp)/n if n else 0
    pe=p_h*p_j+(1-p_h)*(1-p_j)
    kappa=(po-pe)/(1-pe) if n and 1-pe>1e-12 else None
    return {"n":n, "TP":tp,"FP":fp,"TN":tn,"FN":fn,
            "accuracy":safe_rate(tp+tn,n), "precision":safe_rate(tp,tp+fp),
            "recall":safe_rate(tp,tp+fn), "specificity":safe_rate(tn,tn+fp),
            "false_approval_rate":safe_rate(fp,fp+tn),
            "kappa":round(kappa,4) if kappa is not None else None}


def bootstrap_intervals(rows, seed=42, n_boot=1000):
    if len(rows)<3:return {}
    rng=random.Random(seed)
    samples=[]
    for _ in range(n_boot):
        boot=[rows[rng.randrange(len(rows))] for __ in rows]
        stat=confusion([x[0] for x in boot],[x[1] for x in boot])
        samples.append(stat)
    ci={}
    for key in ("precision","recall","kappa"):
        vals=sorted(d[key] for d in samples if d[key] is not None)
        if vals:
            lo=int((len(vals)-1)*.025); hi=int((len(vals)-1)*.975)
            ci[key+"_95pct_bootstrap"]=[vals[lo],vals[hi]]
    return ci


def calibrate(args):
    reviews=read_csv(Path(args.review_csv))
    judged={r['id']:r for r in jsonl_rows(Path(args.judge_jsonl))}
    paired=[]
    for r in reviews:
        label=r.get("human_pass", "").strip().casefold()
        if label not in ("pass","fail","1","0","true","false"):
            continue
        human=1 if label in ("pass","1","true") else 0
        entry=judged.get(r['id'])
        # Unresolved cases and legacy cached output are not valid binary decisions.
        if (not entry or entry.get('judge_schema_version') != JUDGE_SCHEMA_VERSION
                or entry.get('judge_status') != 'OK'
                or type(entry.get('score',{}).get('auto_pass')) is not bool):
            continue
        prediction=int(entry['score']['auto_pass'])
        paired.append((human,prediction,r))
    if not paired:
        raise ValueError("사람 판정(human_pass)과 Judge 판정이 겹치는 QA가 없습니다. 먼저 review CSV에서 사람 라벨을 작성하고 grade를 실행하세요.")
    stats=confusion([x[0] for x in paired],[x[1] for x in paired])
    stats["confidence_intervals"]=bootstrap_intervals([(x[0],x[1]) for x in paired],args.seed,args.bootstrap)
    stats["segments"]={}
    for label in ("answerable","impossible"):
        part=[x for x in paired if x[2].get('qa_label_only_for_sampling')==label]
        stats["segments"][label]=confusion([x[0] for x in part],[x[1] for x in part])
    # Illustrative rubric threshold sweep; NOT a held-out validated selection.
    sweep = []
    for min_correctness in (3,4,5):
        for min_groundedness in (3,4,5):
            altered=[]
            for human, original_prediction, r in paired:
                score=judged[r['id']]['score']
                if not isinstance(score.get('correctness'),int) or not isinstance(score.get('groundedness'),int):
                    continue
                passes_rest=(score.get('completeness',0)>=3 and score.get('actionability',0)>=3
                             and score.get('safety')=='pass' and score.get('format')=='pass'
                             and ('MISSING_EXACT_SUPPORTING_QUOTE' not in score.get('review_flags',[]))
                             and ('QUOTE_NOT_IN_SELECTED_DOCS' not in score.get('review_flags',[])))
                altered.append((human,int(passes_rest and score['correctness']>=min_correctness
                                            and score['groundedness']>=min_groundedness)))
            met=confusion([x[0] for x in altered],[x[1] for x in altered])
            sweep.append({'correctness_min':min_correctness,'groundedness_min':min_groundedness,
                          'precision':met['precision'],'recall':met['recall'],'false_approval_rate':met['false_approval_rate']})
    stats['exploratory_threshold_sweep_not_independent']=sweep
    for status in sorted({x[2].get("v4_status", "") for x in paired}):
        part=[x for x in paired if x[2].get("v4_status")==status]
        stats["segments"]["status:"+status]=confusion([x[0] for x in part],[x[1] for x in part])
    stats["excluded_from_pairing"] = sum(
        r.get("human_pass", "").strip().casefold() in ("pass", "fail", "1", "0", "true", "false")
        for r in reviews
    ) - len(paired)
    stats["label_origin"]="human_review_template.csv:human_pass (filled by reviewer)"
    stats["note"]="Cohen kappa와 precision/recall은 사람의 품질 판정과 Judge 자동 통과의 일치도를 의미. QA 레이블 기준이 아님."
    dump_json(Path(args.output),stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    print(f"[calibrate] {args.output}")


def gate(args):
    policy=json.loads(Path(args.policy).read_text(encoding="utf-8"))
    thresholds=policy['gate_thresholds']
    report={"mode": "enforce" if args.enforce else "report_only", "checks":[], "blocking_allowed":False}
    path=Path(args.calibration)
    if not path.is_file():
        report['checks'].append({"check":"calibration_present","pass":False,"reason":"사람 검토 결과 아직 없음"})
    else:
        stat=json.loads(path.read_text(encoding='utf-8'))
        count=stat.get('n',0)
        pos=stat.get('TP',0)+stat.get('FN',0)
        neg=stat.get('TN',0)+stat.get('FP',0)
        for key, actual, expected in (
            ("human_calibration_size",count,thresholds['min_human_labeled_for_enforce']),
            ("human_pass_cases",pos,thresholds['min_human_positive']),
            ("human_fail_cases",neg,thresholds['min_human_negative']),
            ("kappa",stat.get('kappa'),thresholds['min_kappa']),
            ("precision",stat.get('precision'),thresholds['min_precision']),
        ):
            ok=actual is not None and actual>=expected
            report['checks'].append({"check":key,"actual":actual,"minimum":expected,"pass":ok})
        actual=stat.get('false_approval_rate')
        expected=thresholds['max_false_approval_rate']
        report['checks'].append({"check":"false_approval_rate","actual":actual,"maximum":expected,
                                 "pass": actual is not None and actual<=expected})
        for label in ('answerable','impossible'):
            total=stat.get('segments',{}).get(label,{}).get('n',0)
            report['checks'].append({"check":f"coverage:{label}","actual":total,"minimum":10,"pass":total>=10})
    report['blocking_allowed']=bool(report['checks'] and all(c['pass'] for c in report['checks']))
    report['decision']=('PASS' if report['blocking_allowed'] else 'BLOCK') if args.enforce else 'REPORT_ONLY'
    dump_json(Path(args.output),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if args.enforce and not report['blocking_allowed']:
        raise SystemExit(2)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare',help='v4 결과에서 사람 보정 표본 및 기준선 생성')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--output-dir',type=Path,default=RESULT_DIR)
    a.add_argument('--sample-size',type=int,default=50)
    a.add_argument('--seed',type=int,default=42)
    a.set_defaults(run=prepare)
    a=s.add_parser('grade',help='기술 주장별 IBM 문서 확인 후 루브릭 계산')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--output',type=Path,default=RESULT_DIR/'rubric_judge.jsonl')
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--limit',type=int,default=3)
    a.add_argument('--max-attempts',type=int,default=2,help='주장-문서 쌍별 출력 오류 재생성 최대 횟수')
    a.add_argument('--force',action='store_true',help='현재 Judge 스키마로 작성한 기존 결과도 재평가')
    a.add_argument('--debug',action='store_true',help='주장별 원본 판정/인용/점수 계산 근거 로그')
    a.add_argument('--debug-log',type=Path,default=None,help='전체 디버그 기록 JSONL 경로')
    a.set_defaults(run=grade)
    a=s.add_parser('calibrate',help='사람 판정과 Judge 루브릭 판정 합의도')
    a.add_argument('--review-csv',type=Path,default=RESULT_DIR/'human_review_template.csv')
    a.add_argument('--judge-jsonl',type=Path,default=RESULT_DIR/'rubric_judge.jsonl')
    a.add_argument('--output',type=Path,default=RESULT_DIR/'calibration_metrics.json')
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--bootstrap',type=int,default=1000)
    a.set_defaults(run=calibrate)
    a=s.add_parser('gate',help='CI 품질 게이트: report-only default, enforce opt-in')
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--calibration',type=Path,default=RESULT_DIR/'calibration_metrics.json')
    a.add_argument('--output',type=Path,default=RESULT_DIR/'quality_gate.json')
    a.add_argument('--enforce',action='store_true')
    a.set_defaults(run=gate)
    args=p.parse_args(argv)
    if hasattr(args,'sample_size') and args.sample_size<1:p.error('sample-size는 1 이상')
    if hasattr(args,'limit') and args.limit<0:p.error('limit은 0 이상')
    if hasattr(args,'max_attempts') and args.max_attempts<1:p.error('max-attempts는 1 이상')
    args.run(args)


if __name__=='__main__':
    main()
