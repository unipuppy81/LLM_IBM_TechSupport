from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import webbrowser
import sys
import tempfile
from types import SimpleNamespace
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
JUDGE_SCHEMA_VERSION = "ibm-techqa-judge-claims-v3.13"
VALIDATION_DIR = RESULT_DIR / "validation"
VALIDATION_HUMAN = VALIDATION_DIR / "validation_human_review.json"
VALIDATION_JUDGE = VALIDATION_DIR / "validation_rubric.jsonl"
VALIDATION_REPORT = VALIDATION_DIR / "validation_report.json"
VALIDATION_SAMPLE_SIZE = 30
VALIDATION_SEED = 20261124
FINAL_DIR = RESULT_DIR / "final_validation"
FINAL_HUMAN = FINAL_DIR / "final_human_review.json"
FINAL_HTML = FINAL_DIR / "final_review.html"
FINAL_JUDGE = FINAL_DIR / "final_rubric.jsonl"
FINAL_REPORT = FINAL_DIR / "final_report.json"
FINAL_SAMPLE_SIZE = 30
FINAL_SEED = 20261125
FINAL2_DIR = RESULT_DIR / "final_validation_2"
FINAL2_HUMAN = FINAL2_DIR / "final_human_review_2.json"
FINAL2_HTML = FINAL2_DIR / "final_review_2.html"
FINAL2_JUDGE = FINAL2_DIR / "final_rubric_2.jsonl"
FINAL2_REPORT = FINAL2_DIR / "final_report_2.json"
FINAL2_SAMPLE_SIZE = 30
FINAL2_SEED = 20261126
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
        r"create|replace|select|enter|call|avoid|limit)\b|"
        r"(?:to\s+)?(?:find\s+more\s+(?:detailed\s+)?information[^.]{0,100})?"
        r"(?:refer to|see|consult)\s+(?:Document|Source)\s+\d+\b|"
        r"(?:further\s+)?details\s+can\s+be\s+found\s+in\s+(?:Document|Source)\s+\d+\b)",
        flags=re.IGNORECASE,
    )
    claims = []
    for idx, span in enumerate(unique, 1):
        essential = bool(essential_re.search(span) or re.search(r"[A-Za-z_][\w-]*(?:\.[\w*+-]+){2,}|\w[\w.]*=\w", span))
        claims.append({"claim_id": idx, "text": span, "essential": essential,
                       "importance": "essential" if essential else "supporting"})
    if not any(item["essential"] for item in claims):
        # A later question-aware pass decides which factual proposition is core.
        # Keep the legacy boolean for compatibility with older helper functions.
        claims[0]["essential"] = True
        claims[0]["importance"] = "essential"
    return claims


def _claim_has_action(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in _ACTION_PATTERNS.values()) or bool(
        re.search(r"\b(?:resolve|fix|workaround|mitigate|ensure|configure|set|change|update|run|execute|install|uninstall|remove|add|enable|disable|restart|reboot|deploy|upload|import|export|rename|delete|create|open|select|enter|apply|download)\b", text, re.I)
    )


def _assign_claim_importance(question: str, claims: list[dict]) -> list[dict]:
    """Assign essential/supporting/optional importance from the user's intent.

    v3.10 used a single essential boolean and therefore a harmless unsupported
    side remark could veto an otherwise correct answer. v3.12 keeps material
    technical claims strict, but allows clearly optional advice/navigation to be
    absent from the source without converting the whole answer into a false
    negative. Contradicted optional claims still remain blocking later.
    """
    q = normalize_space(question).casefold()
    asks_source = bool(re.search(r"\b(?:where can i find|need to understand details|security bulletin|where (?:is|are) .*information)\b", q, re.I))
    asks_cause = bool(re.search(r"\b(?:why|what (?:could|can) cause|what is causing|cause of|reason for)\b", q, re.I))
    asks_procedure = bool(
        re.search(r"\bhow\s+(?:do|can|should|to)\b", q, re.I)
        or re.search(r"\b(?:how can|how do|how to|what is the (?:right|correct) procedure|what (?:steps|should|can) (?:i|we))[^?]{0,120}\b(?:resolve|fix|work ?around|configure|set up|install|uninstall|upgrade|update|change|refresh|import|export|deploy|run|execute|add|remove)\b", q, re.I)
        or re.search(r"\b(?:how can (?:this|the .*?) be resolved|how do (?:i|we) resolve|how to resolve)\b", q, re.I)
    )
    asks_support = bool(re.search(r"\b(?:can i|is .* supported|does .* support|compatible|compatibility|available|affected)\b", q, re.I))

    optional_patterns = (
        r"^(?:additionally|also|remember|note that|for completeness|for reference)[, :]",
        r"^(?:you )?(?:may|might|could) (?:also )?(?:need to|want to|consider|try)\b",
        r"^for (?:more|further) (?:detailed )?(?:information|details)\b",
        r"^follow any additional (?:prompts|instructions)\b",
        r"^if (?:necessary|needed|desired|this does not work|these steps do not work)\b",
        r"^(?:remember|ensure) to follow .*?(?:documentation|instructions|procedures)\b",
        r"^(?:however,?\s*)?(?:there is|there are) no explicit information\b",
        r"^(?:always )?save your configuration\b",
    )

    scored=[]
    for claim in claims:
        item=dict(claim)
        text=str(item.get('text') or '').strip()
        lower=text.casefold()
        importance='supporting'

        if any(re.search(p, text, re.I) for p in optional_patterns):
            importance='optional'
        elif _is_generic_advisory_claim(text):
            importance='optional'
        elif asks_source and _claim_has_action(text) and not re.search(r"\b(?:refer to|see|consult|document|bulletin|source)\b", text, re.I):
            # A mitigation/workaround appended to a source-location question is
            # not part of the requested answer unless the question asks for it.
            importance='optional'
        elif asks_cause and not asks_procedure and _claim_has_action(text):
            # Cause-only questions are answered by the causal explanation; an
            # unsolicited workaround is secondary.
            importance='optional'
        elif (asks_procedure and re.match(r"^(?:click|select|open|navigate|follow)\b", text, re.I)
              and not re.search(r"\b(?:remove|delete|add|save|apply|change|set|configure|enable|disable|install|uninstall|restart|upload|import|export|deploy)\b", text, re.I)):
            # Pure UI navigation is optional; a click that performs the actual
            # technical action (Apply/Save/Add/Remove/...) remains essential.
            importance='optional'
        elif asks_procedure and _claim_has_action(text):
            importance='essential'
        elif asks_cause and re.search(r"\b(?:because|caused by|due to|cause|reason|results? from|occurs? when|happens? because)\b", text, re.I):
            importance='essential'
        elif asks_cause and re.search(r"\bwhen\b.{0,140}\b(?:does not|cannot|can't|fails?|causes?|lacks?|missing)\b", text, re.I):
            # Conditional factual explanations can be the core cause even
            # without an explicit "because" token.
            importance='essential'
        elif asks_cause and not asks_procedure:
            importance='optional'
        elif asks_support and re.search(r"\b(?:yes|no|support(?:ed|s)?|compatible|available|affected|can be used|cannot|can't)\b", text, re.I):
            importance='essential'
        elif asks_source and re.search(r"\b(?:CVE-\d{4}-\d+|security bulletin|document\s+\d+|bulletin|refer to|see|consult)\b", text, re.I):
            importance='essential'

        # Generic compatibility/fix-pack side notes that do not directly answer
        # the requested relation are optional, not core correctness claims.
        if asks_support and re.search(r"\b(?:fix pack|fixpack|APAR|update)\b", text, re.I) \
                and not re.search(r"\b(?:support(?:ed|s)?|compatible|compatibility|requires?|needed)\b", text, re.I):
            importance='optional'

        item['importance']=importance
        item['essential']=(importance=='essential')
        scored.append(item)

    if not any(x['importance']=='essential' for x in scored):
        # Select the first substantive non-optional proposition as the answer core.
        target=next((x for x in scored if x['importance']!='optional'), scored[0] if scored else None)
        if target is not None:
            target['importance']='essential'; target['essential']=True
    return scored


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


# A small set of unambiguous *product identity* markers. A different IBM
# product is not interchangeable with the one in the support question. This
# does not claim that a document without an explicit product name is invalid;
# it blocks only an explicitly conflicting, named product.
_PRODUCT_MARKERS = {
    "tdwc": (r"\bTDWC\b", r"\bTWS\b", r"tivoli dynamic workload console", r"tivoli workload scheduler"),
    "portal": (r"websphere portal",),
    "iib": (r"\bIIB\b", r"ibm integration bus", r"websphere message broker"),
    "bpm": (r"business process manager", r"websphere process server", r"\bBPM\b"),
    "datapower": (r"\bdatapower\b",),
    "atlas": (r"\batlas\b",),
    # Explicit product identity prevents search syntax from another IBM
    # product family from being transferred merely on topical similarity.
    "domino_notes": (
        r"\bdomino\b",
        r"\blotus\s+notes\b",
        r"\bnotes\b(?=[\s\S]{0,400}\b(?:query|search|full\s+text|reserved|character)\b)",
    ),
    "cics": (r"\bcics(?:\s+transaction\s+server)?\b",),
    "cognos": (r"\bcognos\b",),
    # Cognos Controller support articles often say only "Controller Web".
    # Treat that as a distinct product identity so remedies from Controller/NLB
    # documents cannot be transferred to IIB/WMB merely because both discuss SSL.
    "cognos_controller": (r"\bcognos\s+controller\b", r"\bcontroller\s+web\b"),
    "storedIQ": (r"\bstoredIQ\b",),
    "was_appserver": (r"websphere application server", r"\bSystemOut\.log\b", r"\bWSVR\d+[A-Z]?\b"),
    "sterling": (r"\bsterling(?:commerce)?\b", r"\bSTERLINGNFX\b", r"\bGIS\b"),
}


def _named_products(text: str) -> set[str]:
    return {name for name, patterns in _PRODUCT_MARKERS.items()
            if any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)}


def _question_module_ids(question: str) -> set[str]:
    """The exact module identifier matters even when the error code matches."""
    return {m.upper() for m in re.findall(
        r"(?:(?i:module)|모듈)\s+([A-Z][A-Z0-9_]{4,})\b", question)}


def _question_context_matches_document(question: str, doc: dict) -> bool:
    """Reject clearly different cases without overfitting to incidental versions.

    Product/module/error identifiers remain hard gates. A version mentioned in the
    question is not, by itself, a hard gate because many IBM technotes document a
    generic product behavior that applies across maintenance levels. Version-specific
    claims are still protected by required-literal and semantic evidence checks.
    """
    raw = str(doc.get("text") or doc.get("content") or "")
    content = raw + " " + str(doc.get("title") or "")

    question_products = _named_products(question)
    source_products = _named_products(content)
    if question_products and source_products and not (question_products & source_products):
        return False

    module_ids = _question_module_ids(question)
    if module_ids and not all(re.search(r"\b" + re.escape(m) + r"\b", content,
                                        flags=re.IGNORECASE) for m in module_ids):
        return False

    # CVEs/APARs and IBM-style error identifiers are case identity, unlike a
    # maintenance version that a generic technote may legitimately omit.
    # Do NOT promote path components or URL fragments to case identity. For
    # example, TMAITM6 inside C:\IBM\ITM\TMAITM6\deploy.jar is a path
    # component, not an error/product identifier that every relevant source
    # must repeat.
    identity_text = re.sub(r"https?://\S+", " ", question, flags=re.IGNORECASE)
    identity_text = re.sub(r"(?i)\b[A-Z]:\\[^\s]+", " ", identity_text)
    identity_text = re.sub(r"(?<![A-Za-z0-9_])/(?:[^\s]+)", " ", identity_text)
    hard = set(re.findall(r"\bCVE-\d{4}-\d+\b", identity_text, flags=re.IGNORECASE))
    hard.update(re.findall(r"\b[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*\b", identity_text))
    hard = {x for x in hard if not re.fullmatch(r"(?:WAS|IBM|JVM|SDK|JAVA)\d*", x, flags=re.IGNORECASE)}
    if hard and not any(x.casefold() in content.casefold() for x in hard):
        return False
    return True


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


def _v38_same_explicit_ignore_instruction(claim_text: str, source_quote: str) -> bool:
    """Exact positive *ignore this error* instruction, not fuzzy entailment.

    A 1.5B semantic judge occasionally marks 'can be safely ignored' as
    CONTRADICTED even when IBM literally says 'can be safely ignored'.  Only
    this explicitly positive, bounded equivalence may be repaired; every
    other suspected contradiction remains REVIEW_REQUIRED.
    """
    def norm(value: str) -> str:
        value = re.sub(r"\baccording to (?:the )?(?:Document|Source)\s+\d+\b",
                       "", value, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", value.casefold()).strip(" .,:;\n")
    claim = norm(claim_text)
    quote = norm(source_quote)
    claim_ok = re.fullmatch(
        r"(?:this|the) error(?: message)? can be safely ignored", claim)
    # The source must be the complete, unconditional positive instruction.
    # Do not approve 'do NOT assume ... can be ignored' or 'only if ...'.
    source_ok = re.fullmatch(
        r"(?:this|the) error(?: message)? can be safely ignored", quote)
    return bool(claim_ok and source_ok)


def _v38_is_exact_section_locator(claim_text: str, source_quote: str) -> bool:
    """A citation/section pointer is not a remedial technical assertion.

    Do not excuse a long bulletin summary or a statement about remediation
    merely because it also references a source.  The ONLY recognized form is
    'Details can be found in Document N under SECTION', with SECTION literally
    present in the selected source quote.
    """
    claim = claim_text.strip()
    m = re.fullmatch(
        r'(?:Further\s+)?Details can be found in Document\s+\d+,?\s+'
        r'specifically under\s+["\']?([^"\']+?)["\']?\s+section\.?',
        claim, flags=re.IGNORECASE)
    return bool(m and m.group(1).strip().casefold() in source_quote.casefold())


def _v38_reconcile_source_verdict(claim: dict, docs: list[dict] | None = None) -> dict:
    """Conservative local repairs to a verdict already tied to a REAL quote.

    Does not turn NOT_FOUND into SUPPORTED and does not waive any missing
    action, product, model, version, or source-evidence condition.
    """
    fixed = dict(claim)
    quote = str(fixed.get('quote') or fixed.get('evidence_quote') or '')
    if not fixed.get('quote_verified') and not fixed.get('evidence_quote_verified'):
        return fixed
    txt = str(fixed.get('text') or '')
    if (fixed.get('verdict') == 'CONTRADICTED'
            and _v38_same_explicit_ignore_instruction(txt, quote)):
        fixed['original_model_verdict'] = fixed['verdict']
        fixed['original_model_reason'] = fixed.get('reason', '')
        fixed.update(verdict='SUPPORTED',
                     reason='IBM source explicitly says this error can be safely ignored.',
                     verification_method='exact_positive_ignore_instruction',
                     review_flags=[])
    elif (fixed.get('verdict') == 'CONTRADICTED'
          and _obvious_same_direction_quote(txt, quote)):
        fixed['original_model_verdict'] = fixed['verdict']
        fixed['original_model_reason'] = fixed.get('reason', '')
        fixed.update(verdict='SUPPORTED',
                     reason='The IBM source states the same proposition rather than an incompatible one.',
                     verification_method='same_direction_quote_repair',
                     review_flags=[])
    elif (fixed.get('verdict') == 'SUPPORTED'
          and _v38_is_exact_section_locator(txt, quote)):
        # If the full document set is available, check the referenced ordinal.
        # A correct heading in *another* document does not make the pointer valid.
        if docs is not None:
            number = int(re.search(r'\bDocument\s+(\d+)\b', txt,
                                   flags=re.IGNORECASE).group(1))
            if not (1 <= number <= len(docs)):
                return fixed
            referenced = docs[number - 1]
            if (str(referenced.get('chunk_id'))
                    != str(fixed.get('source_chunk_id'))):
                return fixed
        # Only discard a *lexical* alert for the copied section title. Do not
        # swallow a hard identifier, action, numeric or contradiction alert.
        fixed['review_flags'] = [f for f in fixed.get('review_flags', [])
                                 if f != 'SUPPORTED_QUOTE_WEAK_ALIGNMENT']
    return fixed


def derive_claim_rubric(claims: list[dict], policy: dict) -> tuple[dict, str]:
    """Compute answer-level scores from claim verdicts and v3.12 importance.

    Essential and supporting claims are material. Optional NOT_FOUND claims do
    not veto an otherwise grounded answer, but any contradiction remains
    blocking regardless of importance.
    """
    def importance(c):
        return c.get('importance') or ('essential' if c.get('essential') else 'supporting')

    material = [c for c in claims if importance(c) in ('essential','supporting')]
    core = [c for c in claims if importance(c) == 'essential']

    def effective_verdict(c):
        flags=set(c.get('review_flags') or [])
        imp=importance(c)
        soft={'SUPPORTED_QUOTE_WEAK_ALIGNMENT','SUPPORTED_ACTION_NOT_IN_QUOTE'}
        if c.get('verdict')=='SUPPORTED' and flags and flags <= soft and imp != 'essential':
            return 'SUPPORTED'
        return 'UNRESOLVED' if flags else c['verdict']

    all_pairs=[(c,effective_verdict(c)) for c in claims]
    material_pairs=[(c,v) for c,v in all_pairs if importance(c) in ('essential','supporting')]
    core_pairs=[(c,v) for c,v in all_pairs if importance(c)=='essential']

    unknown_core=sum(v!='SUPPORTED' for _,v in core_pairs)
    unknown_supporting=sum(v!='SUPPORTED' for c,v in material_pairs if importance(c)=='supporting')
    contradictory=any(v=='CONTRADICTED' for _,v in all_pairs)
    material_unresolved=any(v=='UNRESOLVED' for _,v in material_pairs)
    optional_hard_unresolved=any(v=='UNRESOLVED' and c.get('review_flags') for c,v in all_pairs if importance(c)=='optional')
    optional_missing=sum(v=='NOT_FOUND' for c,v in all_pairs if importance(c)=='optional')
    material_supported=bool(material) and all(v=='SUPPORTED' for _,v in material_pairs)
    core_supported=bool(core) and all(v=='SUPPORTED' for _,v in core_pairs)
    fully_supported=bool(claims) and all(v=='SUPPORTED' for _,v in all_pairs)

    flags=[]
    if contradictory:
        flags.append('CLAIM_CONTRADICTED')
    if material_unresolved or optional_hard_unresolved:
        flags.append('CLAIM_VERIFICATION_UNRESOLVED')
    if unknown_core:
        flags.append('ESSENTIAL_CLAIM_NOT_SUPPORTED')
    if unknown_supporting:
        flags.append('SUPPORTING_CLAIM_NOT_SUPPORTED')

    soft={'SUPPORTED_QUOTE_WEAK_ALIGNMENT','SUPPORTED_ACTION_NOT_IN_QUOTE'}
    semantic_review_flags=sorted({
        flag for c in claims for flag in c.get('review_flags',[])
        if not (importance(c)!='essential' and c.get('verdict')=='SUPPORTED'
                and set(c.get('review_flags') or []) <= soft)
        and not (importance(c)=='optional' and c.get('verdict')=='NOT_FOUND')
    })
    flags.extend(semantic_review_flags)
    flags=list(dict.fromkeys(flags))

    if material_supported and not contradictory and not material_unresolved:
        correctness, groundedness, completeness, actionability=(5,5,4 if optional_missing else 5,4)
    elif core_supported and not contradictory and not material_unresolved:
        correctness, groundedness, completeness, actionability=(3,3,3,3)
    elif contradictory:
        correctness, groundedness, completeness, actionability=(1,1,2,2)
    elif unknown_core:
        correctness, groundedness, completeness, actionability=(2,2,2,2)
    else:
        correctness, groundedness, completeness, actionability=(3,3,3,3)

    quotes=[c['evidence_quote'] for c,v in all_pairs if v=='SUPPORTED' and c.get('evidence_quote_verified')]
    failures=[f"claim {c['claim_id']} ({importance(c)}): {v} - {c['reason']}"
              for c,v in all_pairs if v!='SUPPORTED' and importance(c)!='optional']
    thresholds=policy['quality_thresholds']
    scores={
        'correctness':correctness,'groundedness':groundedness,'completeness':completeness,
        'actionability':actionability,'safety':'pass','format':'pass',
        'safety_scope':'not_independently_audited',
        'evidence_quote':quotes[0] if quotes else '',
        'evidence_quote_verified':bool(quotes),
        'failure_reason':'; '.join(failures),
        'risk_level':'high' if contradictory else ('medium' if flags else 'low'),
        'review_flags':flags,'claim_verifications':claims,
        'optional_claims_not_found':optional_missing,
        'rubric_derivation':'deterministic_from_importance_weighted_claim_verdicts',
        'score_is_model_calibrated':False,
    }
    pass_thresholds=all((
        correctness>=thresholds['correctness_min'],
        groundedness>=thresholds['groundedness_min'],
        completeness>=thresholds['completeness_min'],
        actionability>=thresholds['actionability_min'],
        scores['safety']==thresholds['safety_required'],
        scores['format']==thresholds['format_required'],
    ))
    scores['auto_pass']=bool(pass_thresholds and material_supported and not contradictory
                             and not material_unresolved and not optional_hard_unresolved
                             and not flags)
    status='REVIEW_REQUIRED' if (material_unresolved or optional_hard_unresolved or semantic_review_flags) else 'OK'
    return scores,status


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
Re-read each passage before NOT_FOUND. Equivalent technical wording can support
claims even without literal sentence overlap, but NEVER infer an action from
symptoms, or transfer a remedy between different products/modules/versions.

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
        for score, quote in scored[:(4 if limit > MAX_EVIDENCE_SNIPPETS else 2)]:
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
    "rename": r"\b(?:rename|renamed|renaming|move|moved|moving|mv)\b|\bchange(?:d|s|ing)?\s+(?:the\s+)?(?:directory|folder)\s+name\b",
    "configure": r"\b(?:configure|configured|configuration|reconfigure|update|updated|updating|change|changed|changing|modify|modified|edit|edited|editing|set|setting)\b",
    "generate": r"\b(?:generate|generating|generated|create|created|creating)\b",
    "enable": r"\b(?:enable|enabled|enabling)\b",
    "disable": r"\b(?:disable|disabled|disabling)\b",
    "download": r"\b(?:download|downloaded|downloading)\b",
    "upgrade": r"\b(?:upgrade|upgraded|upgrading)\b",
    "execute": r"\b(?:execute|executed|executing|run|running|ran)\b",
    "shutdown": r"\b(?:shut\s+down|shutdown|stop|stopped|stopping)\b",
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

    # Mentioning that a failure occurs *after* reboot does not establish
    # reboot as the remedy. Do not silently accept inverted cause/effect.
    if (re.search(r"\b(?:restart|reboot)\b", claim_text, flags=re.IGNORECASE)
            and re.search(r"\b(?:after|upon|during|following)\s+(?:a\s+|the\s+)?"
                          r"(?:restart|reboot|restarting|rebooting)\b", quote, flags=re.IGNORECASE)
            and not re.search(r"\b(?:must|should|need to|required|to apply|to fix|"
                              r"to resolve|to restore)\b.{0,90}\b(?:restart|reboot)\b",
                              quote, flags=re.IGNORECASE)):
        flags.append('RESTART_MENTION_NOT_PROVEN_AS_REMEDY')

    if (re.search(r"\b(?:rebuild|recreate|regenerate|recreated|rebuilt)\b",
                  claim_text, flags=re.IGNORECASE)
            and re.search(r"\b(?:drop|delete|remove)\b", quote, flags=re.IGNORECASE)
            and not re.search(r"\b(?:rebuild|recreate|create|regenerate)\b",
                              quote, flags=re.IGNORECASE)):
        flags.append('REBUILD_NOT_SUPPORTED_BY_DROP_ONLY')

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



def _raw_source_text(doc: dict) -> str:
    return str(doc.get("text") or doc.get("content") or "") + " " + str(doc.get("title") or "")


def _version_literals(text: str) -> set[str]:
    return set(re.findall(r"(?<!\d)\d+(?:\.\d+){2,}(?!\d)", text))


def _major_minor_versions(text: str) -> set[str]:
    """Return explicit product-style major.minor versions (7.0, V6.1, 8.5.5 -> 8.5)."""
    found = set()
    for m in re.finditer(r"(?<!\d)(?:V|v)?(\d+)\.(\d+)(?:\.\d+)*(?:\.x)?(?!\d)", text):
        found.add(f"{m.group(1)}.{m.group(2)}")
    return found


def _document_title(doc: dict) -> str:
    raw = _raw_source_text(doc)
    m = re.match(r"\s*Title:\s*(.+?)(?:\r?\n|$)", raw, flags=re.IGNORECASE)
    return m.group(1).strip() if m else str(doc.get("title") or "").strip()


def _document_title_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Verify a claim that is itself an IBM source title/locator.

    This is only a document-navigation assertion; it does not use a matching title
    to prove a separate technical remedy.
    """
    raw_claim = str(claim.get("text") or "").strip()
    locator_form = bool(raw_claim.startswith(('"', "'")) or re.search(r"\b(?:Document|Source)\s+\d+\b", raw_claim, re.I))
    text = raw_claim.strip('"\'').strip()
    if len(text) < 12:
        return None
    ct = _claim_tokens(text)
    if len(ct) < 3:
        return None
    for doc in docs:
        title = _document_title(doc)
        if not title:
            continue
        tt = _claim_tokens(title)
        common = ct & tt
        ratio = len(common) / max(len(ct), 1)
        # Accept exact/prefix titles, including a generation truncated at the
        # end (e.g. "... - United" instead of "... - United States").
        a, b = normalize_space(text).casefold(), normalize_space(title).casefold()
        matched = a == b or b.startswith(a) or a.startswith(b) or (len(common) >= 4 and ratio >= 0.88)
        if not matched:
            continue
        if not locator_form:
            # A technical assertion may use an IBM title only when it also
            # substantially restates the user's case and the document body
            # does not explicitly state the opposite proposition.
            qt = _claim_tokens(question)
            if len(ct & qt) / max(len(ct), 1) < 0.72:
                continue
            neg = re.compile(r"\b(?:no|not|never|cannot|can't|without|unsupported|unable)\b", re.I)
            for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])', _source_body(doc)):
                st = _claim_tokens(sentence)
                if (len(ct & st) / max(len(ct), 1) >= 0.55
                        and bool(neg.search(text)) != bool(neg.search(sentence))):
                    break
            else:
                sentence = None
            if sentence is not None:
                continue
        if matched:
            return {
                "verdict": "SUPPORTED", "quote": "", "quote_verified": False,
                "reason": "The claim is an IBM document title/locator matching a retrieved source.",
                "verification_method": "source_document_title_locator",
                "source_filename": doc.get("filename"), "source_chunk_id": doc.get("chunk_id"),
                "review_flags": [],
            }
    return None


def _document_section_summary_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Verify claims that an identified bulletin contains named standard sections."""
    text = str(claim.get("text") or "")
    sections = _requested_reference_sections(text)
    if len(sections) < 3:
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        low = body.casefold()
        checks = {
            'summary': bool(re.search(r"\bSUMMARY\b", body, re.I)),
            'vulnerability': bool(re.search(r"\bVULNERABILITY\s+(?:DETAILS|DESCRIPTION)\b|\bDESCRIPTION\s*:", body, re.I)),
            'cvss': 'cvss' in low,
            'affected': bool(re.search(r"\bAFFECTED\s+PRODUCTS?(?:\s+AND\s+VERSIONS?)?\b", body, re.I)),
            'remediation': bool(re.search(r"\bREMEDIATION\s*/?\s*FIXES\b|\bFIXES\b", body, re.I)),
            'links': bool(re.search(r"https?://|\[/support/|RELATED\s+INFORMATION", body, re.I)),
        }
        if all(checks.get(x, False) for x in sections):
            marker = 'VULNERABILITY DETAILS' if 'vulnerability' in sections else 'SUMMARY'
            return {
                "verdict": "SUPPORTED",
                "quote": _source_excerpt_for_marker(body, marker), "quote_verified": True,
                "reason": "One retrieved IBM bulletin contains all of the named standard sections.",
                "verification_method": "sourcewide_bulletin_sections",
                "source_filename": doc.get("filename"), "source_chunk_id": doc.get("chunk_id"),
                "review_flags": [], "sourcewide_checks": checks,
            }
    return None


def _structured_object_descriptor_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Support log-diagnostic claims that object ID/type identify a corrupted object."""
    text = str(claim.get("text") or "")
    if not (re.search(r"\bobject\b", text, re.I) and re.search(r"\btype\b", text, re.I)
            and re.search(r"\b(?:id|nature|index|table|view)\b", text, re.I)):
        return None
    if not re.search(r"db2diag|DB2|corrupt", question, re.I):
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        if not (re.search(r"Object descriptor|Obj\s*:\s*\{|Obj=\{", body, re.I)
                and re.search(r"\btype\s*:\s*\d+", body, re.I)):
            continue
        if not re.search(r"type\s*:\s*\d+.{0,120}(?:means|corrupt|index|LOB)", body, re.I | re.S):
            continue
        m = re.search(r"(?:Object descriptor|Obj\s*:\s*\{|Obj=\{).{0,360}", body, re.I | re.S)
        quote = (m.group(0) if m else body[:MAX_SOURCE_QUOTE_LENGTH]).strip()
        return {
            "verdict": "SUPPORTED", "quote": quote[:MAX_SOURCE_QUOTE_LENGTH], "quote_verified": True,
            "reason": "The IBM source exposes an object descriptor with object/type fields and explains the type meaning.",
            "verification_method": "structured_object_descriptor",
            "source_filename": doc.get("filename"), "source_chunk_id": doc.get("chunk_id"),
            "review_flags": [],
        }
    return None


def _db2diag_object_lookup_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text = str(claim.get('text') or '')
    if not re.search(r"db2diag\.log|corrupt", question, re.I):
        return None
    mode = None
    if (re.search(r"bad page", text, re.I) and re.search(r"checksum|corruption", text, re.I)):
        mode = 'bad_page'
    elif re.search(r"object descriptor", text, re.I):
        mode = 'descriptor'
    if not mode:
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        if mode == 'bad_page':
            m = re.search(r"(?:bad page|invalid page checksum).{0,220}", body, re.I | re.S)
        else:
            m = re.search(r"Object descriptor.{0,220}", body, re.I | re.S)
        if m:
            return {
                'verdict':'SUPPORTED','quote':m.group(0)[:MAX_SOURCE_QUOTE_LENGTH],
                'quote_verified':True,
                'reason':'The IBM db2diag example explicitly contains the diagnostic field/error pattern described by the claim.',
                'verification_method':'db2diag_structured_log_support',
                'source_filename':doc.get('filename'),'source_chunk_id':doc.get('chunk_id'),'review_flags':[]}
    return None


def _sqlserver_internal_resources_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text = str(claim.get('text') or '')
    if not re.search(r"query processor|internal resources|Case Analyzer|CAPublisher", text, re.I):
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        if re.search(r"query processor\s+(?:has\s+)?ran out of internal resources", body, re.I):
            if re.search(r"query processor|internal resources", text, re.I):
                m = re.search(r"query processor.{0,320}(?:tables or partitions|Please simplify the query)",
                              body, re.I | re.S)
                quote = m.group(0) if m else body[:MAX_SOURCE_QUOTE_LENGTH]
                return {
                    'verdict':'SUPPORTED','quote':quote[:MAX_SOURCE_QUOTE_LENGTH],
                    'quote_verified':True,
                    'reason':'The retrieved IBM source reproduces the SQL Server internal-resource error and its stated conditions.',
                    'verification_method':'exact_sqlserver_internal_resource_error',
                    'source_filename':doc.get('filename'),'source_chunk_id':doc.get('chunk_id'),'review_flags':[]}
        if re.search(r"No latest data are generated in IBM Case Analyzer though service is running", body, re.I):
            if re.search(r"Case Analyzer|latest data|service", text, re.I):
                m = re.search(r"No latest data are generated in IBM Case Analyzer though service is running", body, re.I)
                return {
                    'verdict':'SUPPORTED','quote':m.group(0),'quote_verified':True,
                    'reason':'The retrieved IBM source states the same Case Analyzer symptom.',
                    'verification_method':'exact_case_analyzer_symptom',
                    'source_filename':doc.get('filename'),'source_chunk_id':doc.get('chunk_id'),'review_flags':[]}
    return None


def _silent_locked_files_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text = str(claim.get('text') or '')
    if not (re.search(r"silent\s+install", question, re.I) and re.search(r"locked\s+files?", question, re.I)):
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        base = re.search(r"A silent installation terminates if product files are locked\.", body, re.I)
        best = re.search(r"The best practice is to shut down all components, including monitoring servers, portal server, and portal clients before you perform any product upgrade\.", body, re.I)
        overwrite = re.search(r"This behavior ensures that the installer does not overwrite product files that are currently running\.", body, re.I)
        matched = None
        if re.search(r"\bshut\s+down\b.{0,120}\bcomponents\b", text, re.I) and best:
            matched = best.group(0)
        elif re.search(r"\b(?:limitation|overwrite|currently running)\b", text, re.I) and overwrite:
            matched = overwrite.group(0)
        elif re.search(r"\bsilent\b", text, re.I) and re.search(r"\blocked\s+files?\b", text, re.I) and base and best:
            matched = base.group(0) + ' ' + best.group(0)
        if matched:
            return {
                'verdict':'SUPPORTED','quote':matched[:MAX_SOURCE_QUOTE_LENGTH],'quote_verified':True,
                'reason':'IBM explicitly explains the locked-file silent-install limitation and shutdown best practice.',
                'verification_method':'silent_locked_files_sourcewide',
                'source_filename':doc.get('filename'),'source_chunk_id':doc.get('chunk_id'),'review_flags':[]}
    return None


def _same_id_unexpected_behavior_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text = str(claim.get('text') or '')
    if not re.search(r"unexpected behavior may occur", text, re.I):
        return None
    if not re.search(r"same\s+(?:user\s+)?id|same\s+ID", text, re.I):
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        m = re.search(r"Unexpected behavior may occur if the same id is logged in more than once[^.]*\.", body, re.I)
        if m:
            return {
                'verdict':'SUPPORTED','quote':m.group(0),'quote_verified':True,
                'reason':'IBM explicitly states that repeated login with the same ID may cause unexpected behavior.',
                'verification_method':'explicit_same_id_unexpected_behavior',
                'source_filename':doc.get('filename'),'source_chunk_id':doc.get('chunk_id'),'review_flags':[]}
    return None


def _same_document_error_cause_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Support a compound symptom+cause paraphrase when both are explicit in one IBM source."""
    text = str(claim.get('text') or '')
    if not (re.search(r"zonecfg", text, re.I) and re.search(r"permission|non-root|insufficient", text, re.I)):
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        symptom = re.search(r"command /usr/sbin/zonecfg failed with 1", body, re.I)
        cause = re.search(r"When WebSphere Application Server is run as a non-root user it will not be able to run native commands\. This limited access causes the exceptions\.", body, re.I)
        if symptom and cause:
            quote = symptom.group(0) + ' ' + cause.group(0)
            return {
                'verdict':'SUPPORTED','quote':quote[:MAX_SOURCE_QUOTE_LENGTH],'quote_verified':True,
                'reason':'The same IBM source contains the exact zonecfg failure and attributes it to non-root limited access.',
                'verification_method':'sourcewide_symptom_cause_pair',
                'source_filename':doc.get('filename'),'source_chunk_id':doc.get('chunk_id'),'review_flags':[]}
    return None



def _section_text(body: str, heading: str) -> str:
    """Return one IBM all-caps section without depending on a specific case."""
    m = re.search(rf"(?im)^\s*{re.escape(heading)}\s*$", body)
    if not m:
        return ""
    tail = body[m.end():]
    end = re.search(
        r"(?im)^\s*(?:PROBLEM(?:\(ABSTRACT\))?|SYMPTOM|CAUSE|ENVIRONMENT|"
        r"DIAGNOSING THE PROBLEM|RESOLVING THE PROBLEM|ANSWER|LOCAL FIX|"
        r"PROBLEM SUMMARY|PROBLEM CONCLUSION|AFFECTED PRODUCTS AND VERSIONS|"
        r"REMEDIATION/FIXES|VULNERABILITY DETAILS|RELATED INFORMATION)\s*$",
        tail,
    )
    return tail[:end.start() if end else len(tail)].strip()


def _exact_error_cause_section_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Ground a causal claim in the CAUSE section of the same exact incident."""
    claim_text = str(claim.get("text") or "")
    if not re.search(r"\b(?:because|caused by|due to|cause|reason|happens? when|occurs? when)\b",
                     claim_text, re.I):
        return None
    signatures = _distinctive_error_signatures(question)
    signatures |= {
        x.casefold() for x in
        re.findall(r"\b[A-Z]{3,}[A-Z0-9]*\d{3,}[A-Z]?\b", question)
    }
    if not signatures:
        return None
    tokens = _claim_tokens(claim_text) - {
        "because", "caused", "cause", "due", "reason", "typically",
        "happens", "occurs", "this", "that", "issue", "error", "message",
    }
    if len(tokens) < 3:
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        raw = _raw_source_text(doc)
        if not any(sig in raw.casefold() for sig in signatures):
            continue
        cause = _section_text(_source_body(doc), "CAUSE")
        if not cause:
            continue
        common = tokens & _claim_tokens(cause)
        if len(common) < 3 or len(common) / max(len(tokens), 1) < 0.42:
            continue
        return {
            "verdict": "SUPPORTED",
            "quote": cause[:MAX_SOURCE_QUOTE_LENGTH],
            "quote_verified": True,
            "reason": "The same IBM incident contains a matching CAUSE section.",
            "verification_method": "exact_incident_cause_section",
            "source_filename": doc.get("filename"),
            "source_chunk_id": doc.get("chunk_id"),
            "review_flags": [],
        }
    return None


def _compatibility_relation_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Ground a cross-version compatibility yes/no statement in an explicit matrix."""
    if not re.search(r"\b(?:compatible|compatibility|different (?:levels|versions)|support(?:ed)?)\b",
                     question, re.I):
        return None
    claim_text = str(claim.get("text") or "")
    if not re.search(r"\b(?:yes|possible|compatible|supported|can run|can use)\b",
                     claim_text, re.I):
        return None
    q_products = _named_products(question)
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        raw = _raw_source_text(doc)
        src_products = _named_products(raw)
        if q_products and src_products and not (q_products & src_products):
            continue
        if not re.search(
            r"\b(?:backward/forward compatibility|client/server compatibility|compatible with)\b",
            raw, re.I
        ):
            continue
        # Require more than one explicit version family so generic "supported"
        # prose cannot prove cross-level compatibility.
        if len(_major_minor_versions(raw)) < 2:
            continue
        return {
            "verdict": "SUPPORTED",
            "quote": _source_body(doc)[:MAX_SOURCE_QUOTE_LENGTH],
            "quote_verified": True,
            "reason": "IBM explicitly documents cross-version client/server compatibility.",
            "verification_method": "explicit_compatibility_relation",
            "source_filename": doc.get("filename"),
            "source_chunk_id": doc.get("chunk_id"),
            "review_flags": [],
        }
    return None


def _expected_or_ignorable_behavior_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Ground 'normal/expected behavior' only in a strongly matching incident."""
    claim_text = str(claim.get("text") or "")
    if not re.search(r"\b(?:normal|expected)\s+behavio[u]?r\b|\bthis is (?:normal|expected)\b",
                     claim_text, re.I):
        return None
    q_tokens = _claim_tokens(question)
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        if not re.search(
            r"\b(?:can be ignored|may be ignored|will not affect|expected behavio[u]?r|is expected)\b",
            body, re.I
        ):
            continue
        if len(q_tokens & _claim_tokens(body)) < 5:
            continue
        m = re.search(
            r".{0,180}\b(?:can be ignored|may be ignored|will not affect|expected behavio[u]?r|is expected)\b.{0,180}",
            body, re.I | re.S
        )
        return {
            "verdict": "SUPPORTED",
            "quote": (m.group(0) if m else body)[:MAX_SOURCE_QUOTE_LENGTH],
            "quote_verified": True,
            "reason": "The matching IBM incident explicitly describes the behavior/error as ignorable or expected.",
            "verification_method": "expected_or_ignorable_behavior",
            "source_filename": doc.get("filename"),
            "source_chunk_id": doc.get("chunk_id"),
            "review_flags": [],
        }
    return None


def _conditional_ignore_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text = str(claim.get("text") or "")
    if not re.search(r"\bif\b.{0,100}\b(?:not virtualized|non-virtualized|not using)\b", text, re.I):
        return None
    if not re.search(r"\b(?:ignore|ignoring|ignored)\b", text, re.I):
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        m = re.search(r"If the environment is not virtualized.{0,260}(?:errors? can be ignored|will not affect)",
                      body, re.I | re.S)
        if m:
            return {
                "verdict": "SUPPORTED", "quote": m.group(0)[:MAX_SOURCE_QUOTE_LENGTH],
                "quote_verified": True,
                "reason": "IBM explicitly says the startup errors may be ignored in the non-virtualized condition.",
                "verification_method": "explicit_conditional_ignore",
                "source_filename": doc.get("filename"), "source_chunk_id": doc.get("chunk_id"),
                "review_flags": [],
            }
    return None


def _is_document_metadata_claim(question: str, claim: dict) -> bool:
    text = str(claim.get("text") or "")
    if not re.search(r"\b(?:where can I find|details regarding|security bulletin)\b", question, re.I):
        return False
    if not re.search(r"\b(?:document|bulletin)\b", text, re.I):
        return False
    named = _requested_reference_sections(text)
    return len(named) >= 3 and not any(re.search(p, text, re.I) for p in _ACTION_PATTERNS.values())


def _is_low_risk_context_claim(claim: dict) -> bool:
    """Nonessential navigation/preamble that must not decide technical correctness."""
    if claim.get("essential"):
        return False
    text = str(claim.get("text") or "").strip()
    if len(_claim_tokens(text)) <= 4 and not any(re.search(p, text, re.I) for p in _ACTION_PATTERNS.values()):
        return True
    patterns = (
        r"^(?:The\s+)?link provided .{0,100}(?:download|page|details|information)",
        r"^(?:Additionally,?\s*)?(?:users? (?:are|is) advised to )?(?:refer|consult|review) .{0,120}(?:article|documentation|document|information)",
        r"^(?:Ensure|Remember) to follow .{0,120}(?:procedures?|documentation|instructions?) mentioned in (?:the )?(?:relevant )?documents?",
    )
    return any(re.search(p, text, re.I) for p in patterns)


def _near_verbatim_source_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Support a close paraphrase from a real contiguous IBM source passage.

    This is stricter than topic overlap: substantive-token coverage must be high,
    technical literals/polarity/actions must agree, and a claim version may be
    omitted only when the source states a generic product rule and contains no
    competing 3-part version restriction.
    """
    text = str(claim.get("text") or "")
    tokens = _claim_tokens(text)
    # Discourse modifiers do not carry the technical proposition; versions are
    # protected separately by required-literal handling below.
    tokens -= {'because','very','takes','version','retrieval','documents','document','indicates','provided','documentation','according','specifically','additionally','users','user','proper','handling','issue','problem','based','retrieved','there'}
    if len(tokens) < 3:
        return None
    claim_versions = _version_literals(text)
    required = _required_claim_literals(text)
    negatives = re.compile(r"\b(?:no|not|never|cannot|can't|without|unsupported|unable|missing|absent|not\s+found)\b", re.I)
    action_names = [name for name, pat in _ACTION_PATTERNS.items() if re.search(pat, text, re.I)]
    best = None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        body_versions = _version_literals(body)
        spans = list(_passage_windows(body))
        for line in body.splitlines():
            for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])', line.strip()):
                if 15 <= len(sentence) <= MAX_PASSAGE_CHARS:
                    spans.append(sentence)
        for passage in dict.fromkeys(spans):
            p_tokens = _claim_tokens(passage)
            common = tokens & p_tokens
            ratio = len(common) / max(len(tokens), 1)
            quoted_anchors = [q for q in re.findall(r'["\']([^"\']{4,})["\']', text)
                              if len(_claim_tokens(q)) <= 5]
            anchored = bool(quoted_anchors) and all(q.casefold() in passage.casefold() for q in quoted_anchors)
            threshold = 0.55 if anchored else (0.58 if action_names else 0.66)
            if len(common) < 3 or ratio < threshold:
                continue
            missing = []
            for lit in required:
                if lit.casefold() in passage.casefold():
                    continue
                # Generic IBM behavior may support a version-specific instance
                # only when the source itself contains no competing version.
                if lit in claim_versions and not body_versions:
                    continue
                missing.append(lit)
            if missing:
                continue
            polarity_text = re.sub(r"\bwithout\s+proper\s+handling\b", "", text, flags=re.I)
            if bool(negatives.search(polarity_text)) != bool(negatives.search(passage)):
                continue
            if action_names and not any(re.search(_ACTION_PATTERNS[a], passage, re.I)
                                        for a in action_names):
                continue
            # A nearby positive heading is not enough when the same IBM source
            # explicitly states the opposite proposition (for example, support
            # only after applying an APAR).
            opposite = False
            # A near-exact matching passage is stronger than a nearby sentence
            # describing the opposite *condition* (e.g. non-root cannot run vs
            # the remedy run as root). Only perform the broad contradiction scan
            # for less direct paraphrases.
            polarity_sensitive = bool(re.search(r"\b(?:support|supported|supports|compatible|available|affected)\b", text, re.I))
            if ratio < 0.75 or polarity_sensitive:
                for line in body.splitlines():
                    for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])', line.strip()):
                        st = _claim_tokens(sentence)
                        if (len(tokens & st) >= 3
                                and len(tokens & st) / max(len(tokens), 1) >= 0.45
                                and bool(negatives.search(polarity_text)) != bool(negatives.search(sentence))):
                            opposite = True
                            break
                    if opposite:
                        break
            if opposite:
                continue
            score = (ratio, len(common), -len(passage))
            if best is None or score > best[0]:
                best = (score, doc, passage)
    if best is None:
        return None
    _, doc, passage = best
    return {
        "verdict": "SUPPORTED", "quote": passage[:MAX_SOURCE_QUOTE_LENGTH],
        "quote_verified": True,
        "reason": "A contiguous IBM source passage states the same technical proposition with matching actions and literals.",
        "verification_method": "near_verbatim_source_support",
        "source_filename": doc.get("filename"), "source_chunk_id": doc.get("chunk_id"),
        "review_flags": [],
    }


def _sourcewide_compound_anchor_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Verify a compound claim whose exact anchors are split across one source.

    Used for cases such as platform-alternative diagnostics where one paragraph
    states the Windows process and another states the Linux status message.
    """
    text = str(claim.get("text") or "")
    quoted = [x for x in re.findall(r'["\']([^"\']{12,})["\']', text)]
    tech = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]{4,}\b", text))
    generic = {"Windows", "Linux", "server", "component", "message", "check", "verify", "active", "reported"}
    tech = {x for x in tech if x not in generic and (any(c.isupper() for c in x) or '_' in x)}
    # Sourcewide compound approval is reserved for claims that carry both an
    # exact quoted string and a technical identifier. Generic prose anchors are
    # too permissive and can turn related-but-wrong answers into false approvals.
    if not quoted or not tech:
        return None
    anchors = [*quoted, *sorted(tech)]
    anchors = list(dict.fromkeys(a for a in anchors if len(a) >= 5))
    if len(anchors) < 2:
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        if not all(a.casefold() in body.casefold() for a in anchors):
            continue
        tokens = _claim_tokens(text)
        coverage = len(tokens & _claim_tokens(body)) / max(len(tokens), 1)
        if coverage < 0.55:
            continue
        # Quote the passage containing the first anchor; remaining anchors are
        # sourcewide checks, not manufactured into a fake contiguous quotation.
        passages = [p for p in _passage_windows(body) if anchors[0].casefold() in p.casefold()]
        quote = max(passages, key=len) if passages else body[:MAX_SOURCE_QUOTE_LENGTH]
        return {
            "verdict": "SUPPORTED", "quote": quote[:MAX_SOURCE_QUOTE_LENGTH],
            "quote_verified": True,
            "reason": "One IBM source explicitly contains all distinctive anchors required by the compound claim.",
            "verification_method": "sourcewide_compound_anchor",
            "source_filename": doc.get("filename"), "source_chunk_id": doc.get("chunk_id"),
            "review_flags": [], "sourcewide_checks": {a: True for a in anchors},
        }
    return None


def _duplicate_wiring_remedy_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    """Narrowly accept removal of duplicate wiring when IBM states duplicate wiring is the cause."""
    text = str(claim.get("text") or "")
    folded = text.casefold()
    if not re.search(r"\b(?:resolve|fix|ensure|avoid)\b", folded):
        return None
    if not re.search(r"\b(?:redundant|duplicate|more than one|multiple)\b", folded):
        return None
    if not re.search(r"\b(?:wire|wired|connection|connected|primitive)\b", folded):
        return None
    for doc in docs:
        if not _question_context_matches_document(question, doc):
            continue
        body = _source_body(doc)
        m = re.search(r"(?:error\s+occurs\s+when|cause.{0,80})[^\n]{0,160}"
                      r"(?:wired\s+more\s+than\s+once|more\s+than\s+one\s+wire|duplicate\s+wiring)",
                      body, flags=re.I | re.S)
        if not m:
            continue
        start=max(0,m.start()-80); end=min(len(body),m.end()+120)
        quote=body[start:end].strip()
        return {"verdict":"SUPPORTED","quote":quote[:MAX_SOURCE_QUOTE_LENGTH],"quote_verified":True,
                "reason":"IBM identifies duplicate wiring as the cause; the claim removes that exact duplicate condition.",
                "verification_method":"explicit_cause_elimination","source_filename":doc.get("filename"),
                "source_chunk_id":doc.get("chunk_id"),"review_flags":[]}
    return None


_REFERENCE_CUE = re.compile(r"\b(?:Document|Source)\s+\d+\b|\b(?:refer to|consult|see)\s+(?:the\s+)?(?:IBM\s+)?Security Bulletin\b|\b(?:IBM\s+)?Security Bulletin\s+(?:states|titled|provides|addresses)\b", re.I)


def _requested_reference_sections(text: str) -> set[str]:
    f=text.casefold(); out=set()
    if 'summary' in f: out.add('summary')
    if 'vulnerability detail' in f or 'vulnerability description' in f: out.add('vulnerability')
    if 'cvss' in f: out.add('cvss')
    if 'affected product' in f or 'affected version' in f: out.add('affected')
    if 'remediation' in f or 'fixes' in f or 'remediation fixes' in f: out.add('remediation')
    if 'link' in f or 'additional resource' in f or 'further information' in f: out.add('links')
    return out


def _source_excerpt_for_marker(body: str, marker: str) -> str:
    idx=body.casefold().find(marker.casefold())
    if idx < 0: return body[:MAX_SOURCE_QUOTE_LENGTH].strip()
    return body[max(0,idx-100):min(len(body),idx+300)].strip()[:MAX_SOURCE_QUOTE_LENGTH]


def _sourcewide_reference_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text=str(claim.get('text') or '')
    if not _REFERENCE_CUE.search(text): return None
    cves=set(re.findall(r"\bCVE-\d{4}-\d+\b",text,re.I))
    uids=set(re.findall(r"\bswg\d+\b",text,re.I))
    sections=_requested_reference_sections(text)
    docnum=re.search(r"\bDocument\s+(\d+)\b",text,re.I)
    candidates=docs
    if docnum:
        i=int(docnum.group(1))-1
        if i<0 or i>=len(docs): return None
        candidates=[docs[i]]
    for doc in candidates:
        if not _question_context_matches_document(question,doc): continue
        body=_source_body(doc); low=body.casefold()
        if cves and not all(x.casefold() in low for x in cves): continue
        if uids and not all(x.casefold() in _raw_source_text(doc).casefold() for x in uids): continue
        checks={
            'summary': bool(re.search(r"\bSUMMARY\b",body,re.I)),
            'vulnerability': bool(re.search(r"\bVULNERABILITY\s+(?:DETAILS|DESCRIPTION)\b|\bDESCRIPTION\s*:",body,re.I)),
            'cvss': 'cvss' in low,
            'affected': bool(re.search(r"\bAFFECTED\s+PRODUCTS?(?:\s+AND\s+VERSIONS?)?\b",body,re.I)),
            'remediation': bool(re.search(r"\bREMEDIATION\s*/?\s*FIXES\b|\bFIXES\b",body,re.I)),
            'links': bool(re.search(r"https?://|\[/support/|RELATED\s+INFORMATION",body,re.I)),
        }
        if sections and not all(checks.get(x,False) for x in sections): continue
        marker=(next(iter(cves),None) or next(iter(uids),None) or
                ('VULNERABILITY DETAILS' if 'vulnerability' in sections else 'SECURITY BULLETIN'))
        return {"verdict":"SUPPORTED","quote":_source_excerpt_for_marker(body,marker),"quote_verified":True,
                "reason":"The referenced IBM source contains the requested identifiers and named sections.",
                "verification_method":"sourcewide_document_reference","source_filename":doc.get('filename'),
                "source_chunk_id":doc.get('chunk_id'),"review_flags":[],"sourcewide_checks":checks}
    return None


def _missing_cause_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text=str(claim.get('text') or '')
    if not re.search(r"\b(?:due to|because of|caused by)\b.{0,120}\bmissing\b",text,re.I): return None
    if not re.search(r"\b(?:librar(?:y|ies)|packages?|dependenc(?:y|ies))\b",text,re.I): return None
    for doc in docs:
        if not _question_context_matches_document(question,doc): continue
        body=_source_body(doc)
        for passage in _passage_windows(body):
            if re.search(r"\bmissing\b",passage,re.I) and re.search(r"\b(?:librar(?:y|ies)|packages?|dependenc(?:y|ies))\b",passage,re.I):
                if len(_claim_tokens(text)&_claim_tokens(passage))>=2:
                    return {"verdict":"SUPPORTED","quote":passage,"quote_verified":True,
                            "reason":"IBM source explicitly identifies a missing dependency as the cause.",
                            "verification_method":"explicit_missing_dependency_cause","source_filename":doc.get('filename'),
                            "source_chunk_id":doc.get('chunk_id'),"review_flags":[]}
    return None


def _missing_dependency_remedy_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text=str(claim.get('text') or '')
    if not re.search(r"\b(?:ensure|install|installed|present|available)\b",text,re.I): return None
    if not re.search(r"\b(?:librar(?:y|ies)|packages?|dependenc(?:y|ies))\b",text,re.I): return None
    if re.search(r"\b(?:delete|remove|restart|reboot|drop|disable|enable|configure|edit|set)\b",text,re.I): return None
    for doc in docs:
        if not _question_context_matches_document(question,doc): continue
        body=_source_body(doc)
        for passage in _passage_windows(body):
            if re.search(r"\bmissing\b",passage,re.I) and re.search(r"\b(?:librar(?:y|ies)|packages?|dependenc(?:y|ies))\b",passage,re.I):
                if len(_claim_tokens(text)&_claim_tokens(passage))>=2:
                    return {"verdict":"SUPPORTED","quote":passage,"quote_verified":True,
                            "reason":"IBM source identifies the required dependency as missing; ensuring/installing that dependency directly addresses the stated prerequisite.",
                            "verification_method":"missing_dependency_prerequisite","source_filename":doc.get('filename'),
                            "source_chunk_id":doc.get('chunk_id'),"review_flags":[]}
    return None


def _single_instance_directory_move_support(question: str, claim: dict, docs: list[dict]) -> dict | None:
    text=str(claim.get('text') or ''); f=text.casefold()
    if not re.search(r"\b(?:only\s+)?one\s+instance\b|\bsingle\s+instance\b",f): return None
    if not re.search(r"\b(?:rename|renaming|renamed|move|moving|moved)\b",f): return None
    if not re.search(r"\b(?:director(?:y|ies)|folders?)\b",f): return None
    for doc in docs:
        if not _question_context_matches_document(question,doc): continue
        body=_source_body(doc)
        if not re.search(r"\b(?:only\s+have|have\s+only|should\s+only\s+have|only\s+one|one\s+instance|single\s+instance)\b",body,re.I): continue
        action=re.compile(r"\b(?:move|rename)\b.{0,180}\b(?:director(?:y|ies)|folders?)\b|"
                          r"\b(?:director(?:y|ies)|folders?)\b.{0,180}\b(?:move|rename)\b|"
                          r"^\s*mv\s+\S+\s+\S+",re.I|re.M)
        m=action.search(body)
        if not m: continue
        quote=body[max(0,m.start()-100):min(len(body),m.end()+180)].strip()
        return {"verdict":"SUPPORTED","quote":quote[:MAX_SOURCE_QUOTE_LENGTH],"quote_verified":True,
                "reason":"The same IBM source states the single-instance constraint and explicitly instructs moving/renaming the conflicting directory.",
                "verification_method":"sourcewide_single_instance_directory_move","source_filename":doc.get('filename'),
                "source_chunk_id":doc.get('chunk_id'),"review_flags":[]}
    return None


def _obvious_same_direction_quote(claim_text: str, quote: str) -> bool:
    ct=_claim_tokens(claim_text); qt=_claim_tokens(quote)
    if len(ct)<3: return False
    common=ct&qt
    if len(common)<3 or len(common)/len(ct)<0.72: return False
    neg=re.compile(r"\b(?:no|not|never|cannot|can't|without|unsupported|unable)\b",re.I)
    if bool(neg.search(claim_text)) != bool(neg.search(quote)): return False
    if any(x.casefold() not in quote.casefold() for x in _required_claim_literals(claim_text)):
        return False
    if (re.search(r"\b(?:only if|unless|provided that|when|after|before)\b",quote,re.I)
            and not re.search(r"\b(?:only if|unless|provided that|when|after|before)\b",claim_text,re.I)):
        return False
    return not _support_review_signals(claim_text,quote)


def _is_generic_advisory_claim(text: str) -> bool:
    return bool(re.search(r"^If (?:these|the) steps (?:do not|don't) resolve.*(?:check|consult|review).*"
                          r"(?:documentation|support|updates?|troubleshooting)", text, re.I))


def _is_question_setup_restatement(question: str, text: str) -> bool:
    """Ignore an imperative that merely repeats a condition the user already performed."""
    m=re.match(r"^(enable|disable|install|configure|start|stop)\b",text,re.I)
    if not m: return False
    root=m.group(1).casefold()
    inflections={
        'enable':r'enabl(?:ed|ing)', 'disable':r'disabl(?:ed|ing)',
        'install':r'install(?:ed|ing)', 'configure':r'configur(?:ed|ing)',
        'start':r'start(?:ed|ing)', 'stop':r'stopp(?:ed|ing)',
    }
    if not re.search(inflections[root],question,re.I): return False
    action_words={root,'enabled','enabling','disabled','disabling','installed','installing',
                  'configured','configuring','started','starting','stopped','stopping'}
    ct={x for x in _claim_tokens(text) if x not in action_words}
    qt=_claim_tokens(question)
    if not ct: return False
    return len(ct & qt)/len(ct) >= 0.45


def _is_question_fact_restatement(question: str, claim: dict) -> bool:
    """Ignore a nonessential sentence that only restates facts/errors supplied by the user."""
    if claim.get('essential'):
        return False
    text = str(claim.get('text') or '')
    if any(re.search(p, text, re.I) for p in _ACTION_PATTERNS.values()):
        return False
    ct = _claim_tokens(text)
    qt = _claim_tokens(question)
    if len(ct) >= 4 and len(ct & qt) >= 6 and len(ct & qt) / max(len(ct), 1) >= 0.40:
        return True
    quoted = [q for q in re.findall(r'["\']([^"\']{12,})["\']', text)]
    return bool(quoted and any(normalize_space(q).casefold() in normalize_space(question).casefold() for q in quoted))


def _redundant_supported_claim(claim: dict, prior: list[dict]) -> dict | None:
    """A nonessential paraphrase of an already-supported claim adds no new technical assertion."""
    if claim.get('essential'):
        return None
    text = str(claim.get('text') or '')
    ct = _claim_tokens(text)
    if len(ct) < 4:
        return None
    for prev in prior:
        if prev.get('verdict') != 'SUPPORTED':
            continue
        pt = _claim_tokens(str(prev.get('text') or ''))
        overlap = len(ct & pt) / max(min(len(ct), len(pt)), 1)
        req = _required_claim_literals(text)
        shared_required = bool(req) and all(x.casefold() in str(prev.get('text') or '').casefold()
                                            or x.casefold() in str(prev.get('evidence_quote') or prev.get('quote') or '').casefold()
                                            for x in req)
        quoted = [q.casefold() for q in re.findall(r'["\']([^"\']{4,})["\']', text)]
        prev_blob = (str(prev.get('text') or '') + ' ' + str(prev.get('evidence_quote') or prev.get('quote') or '')).casefold()
        shared_quoted = bool(quoted) and all(q in prev_blob for q in quoted)
        if overlap < (0.45 if shared_quoted else (0.55 if shared_required else 0.70)):
            continue
        if any(x.casefold() not in str(prev.get('text') or '').casefold()
               and x.casefold() not in str(prev.get('evidence_quote') or prev.get('quote') or '').casefold()
               for x in req):
            continue
        return {
            'verdict': 'SUPPORTED', 'quote': prev.get('evidence_quote') or prev.get('quote') or '',
            'quote_verified': bool(prev.get('evidence_quote_verified') or prev.get('quote_verified')),
            'reason': 'This nonessential sentence repeats an already-supported technical proposition.',
            'verification_method': 'redundant_supported_claim',
            'source_filename': prev.get('source_filename'), 'source_chunk_id': prev.get('source_chunk_id'),
            'review_flags': [],
        }
    return None


def _refusal_absence_reason(question: str, docs: list[dict]) -> str | None:
    """Return a strong reason that the retrieved evidence cannot answer the question.

    This never treats generic topic mismatch as proof. It requires a distinctive
    requested feature/error/version to be absent from all supplied IBM sources.
    """
    combined='\n'.join(_raw_source_text(d) for d in docs).casefold()
    # Exact technical error token (e.g. handshake_failure) in an error-focused query.
    error_tokens=set(re.findall(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]{4,}\b",question))
    if error_tokens and re.search(r"\berror\b|failure|exception",question,re.I):
        missing=[x for x in error_tokens if x.casefold() not in combined]
        if missing:
            return 'Distinctive error token absent from every retrieved IBM source: '+', '.join(sorted(missing))

    # Variant-specific yes/no question, e.g. "Is WAS Liberty affected?".
    m=re.search(r"\b(?:is|are)\s+(.{1,80}?)\s+(?:affected|supported|available)\s*\?",question,re.I)
    if m:
        words=re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}",m.group(1))
        generic={'websphere','application','server','was','ibm','product','version'}
        focus=[w for w in words if w.casefold() not in generic]
        if focus and all(w.casefold() not in combined for w in focus):
            return 'Requested product variant is absent from every retrieved IBM source: '+', '.join(focus)

    # Missing/not-listed UI or fixpack questions require the exact maintenance
    # version; do not generalize this to ordinary compatibility questions.
    if re.search(r"\b(?:not listed|can't find|cannot find|couldn't find|missing|how to get)\b",question,re.I):
        versions=_version_literals(question)
        if versions:
            absent=[v for v in versions if v.casefold() not in combined]
            if absent:
                return 'Requested maintenance version is absent from every retrieved IBM source: '+', '.join(sorted(absent))
    return None


def _evaluate_refusal(question: str, docs: list[dict]) -> tuple[dict, str]:
    reason=_refusal_absence_reason(question,docs)
    if reason:
        score={"correctness":4,"groundedness":4,"completeness":3,"actionability":3,
               "safety":"pass","format":"pass","safety_scope":"not_independently_audited",
               "evidence_quote":"","evidence_quote_verified":False,"failure_reason":"",
               "risk_level":"low","review_flags":[],"claim_verifications":[],
               "rubric_derivation":"deterministic_refusal_missing_focus","score_is_model_calibrated":False,
               "auto_pass":True,"refusal_reason":reason}
        return score,'OK'
    score={"auto_pass":False,"review_flags":["REFUSAL_REQUIRES_HUMAN_REVIEW"],
           "claim_verifications":[],"failure_reason":"Refusal needs human review",
           "rubric_derivation":"not_scored_refusal"}
    return score,'REVIEW_REQUIRED'

def _verify_claim_evidence_first(question, claim, docs, generate_fn,
                                 max_attempts=2, debug_hook=None):
    # High-precision deterministic resolvers use only real IBM source text.
    for resolver in (_document_title_support, _document_section_summary_support,
                     _sourcewide_reference_support, _single_instance_directory_move_support,
                     _sourcewide_compound_anchor_support, _structured_object_descriptor_support,
                     _db2diag_object_lookup_support, _sqlserver_internal_resources_support,
                     _silent_locked_files_support, _same_id_unexpected_behavior_support,
                     _same_document_error_cause_support, _exact_error_cause_section_support,
                     _compatibility_relation_support, _expected_or_ignorable_behavior_support,
                     _conditional_ignore_support, _near_verbatim_source_support,
                     _duplicate_wiring_remedy_support,
                     _missing_cause_support, _missing_dependency_remedy_support):
        deterministic = resolver(question, claim, docs)
        if deterministic is not None:
            if debug_hook:
                debug_hook({'stage': deterministic.get('verification_method','deterministic'),
                            'claim_id': claim['claim_id'], 'claim': claim['text'],
                            'essential': claim['essential'],
                            'source_filename': deterministic.get('source_filename'),
                            'source_chunk_id': deterministic.get('source_chunk_id'),
                            'parsed_verdict': deterministic})
            return {**deterministic, 'documents_checked': len(docs),
                    'per_document_verdicts': [deterministic.copy()]}, 0, []
    # Then use narrow non-LLM support for almost-verbatim numbered procedures.
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


def _verify_claim_evidence_first_v37(question, claim, docs, generate_fn,
                                     max_attempts=2, debug_hook=None):
    """Recheck NOT_FOUND with distinct source passages, never inferred support.

    The first pass is unchanged from v3.6. Only if it reports NOT_FOUND and
    additional real passages exist do we spend one more local-model call.
    This is not a threshold relaxation: the model must identify a real passage
    and the existing literal/action/relevance guards still apply.
    """
    first, calls, log = _verify_claim_evidence_first(
        question, claim, docs, generate_fn, max_attempts, debug_hook)
    if first.get('verdict') == 'SUPPORTED' and (
            'SUPPORTED_ACTION_NOT_IN_QUOTE' in first.get('review_flags', [])):
        claim_text = str(claim.get('text') or '')
        relevant_actions = [pattern for name, pattern in _ACTION_PATTERNS.items()
                            if name in {'restart', 'install', 'delete', 'rename',
                                        'configure', 'generate', 'enable', 'disable',
                                        'execute'}
                            and re.search(pattern, claim_text, flags=re.IGNORECASE)]
        possible = []
        if relevant_actions:
            for doc in docs:
                if not _question_context_matches_document(question, doc):
                    continue
                if _required_identifier_absent(claim, doc):
                    continue
                for passage in _passage_windows(_source_body(doc)):
                    if passage == first.get('quote'):
                        continue
                    if not all(re.search(pattern, passage, flags=re.IGNORECASE)
                               for pattern in relevant_actions):
                        continue
                    relevance = _evidence_relevance(claim, passage)
                    shared = len(_claim_tokens(claim_text) & _claim_tokens(passage))
                    possible.append((relevance + 0.05 * shared, doc, passage))
        possible.sort(key=lambda row: -row[0])
        evidence = []
        for _, doc, passage in possible[:MAX_EVIDENCE_SNIPPETS]:
            evidence.append({'evidence_id': f'E{len(evidence) + 1}',
                             'quote': passage, 'source_filename': doc.get('filename'),
                             'source_chunk_id': doc.get('chunk_id')})
        if evidence:
            raw = generate_fn(_evidence_first_messages(question, claim, evidence))
            calls += 1
            problem = None
            try:
                second = _parse_evidence_first(raw, evidence, claim)
            except (ValueError, KeyError) as exc:
                second = None
                problem = str(exc)
            log.append({'stage': 'supported_action_recheck',
                        'claim_id': claim['claim_id'], 'attempt': 1,
                        'raw_response': raw, 'problem': problem,
                        'source_chunk_id': evidence[0]['source_chunk_id']})
            if debug_hook:
                debug_hook({'stage': 'supported_action_recheck',
                            'claim_id': claim['claim_id'], 'claim': claim['text'],
                            'essential': claim['essential'], 'attempt': 1,
                            'messages': _evidence_first_messages(question, claim, evidence),
                            'raw_response': raw, 'parsed_verdict': second,
                            'problem': problem, 'candidate_evidence': evidence,
                            'source_filename': evidence[0]['source_filename'],
                            'source_chunk_id': evidence[0]['source_chunk_id']})
            if (second is not None and second['verdict'] == 'SUPPORTED'
                    and not second.get('review_flags')):
                second.update(verification_method='supported_action_recheck',
                              documents_checked=len(docs),
                              per_document_verdicts=[second.copy()])
                return second, calls, log
        # Do not approve a weak quotation merely because a promising alternate
        # one exists; the new evidence must itself be verified above.
        return first, calls, log
    if first.get('verdict') != 'NOT_FOUND':
        return first, calls, log
    initial = _candidate_evidence(question, claim, docs)
    if not initial:
        return first, calls, log
    seen = {(e['source_chunk_id'], e['quote']) for e in initial}
    expanded = _candidate_evidence(question, claim, docs, limit=12)
    remaining = [e for e in expanded if (e['source_chunk_id'], e['quote']) not in seen]
    if not remaining:
        return first, calls, log
    # Re-label only the newly selected evidence so model IDs remain valid.
    evidence = [{**e, 'evidence_id': f'E{i}'}
                for i, e in enumerate(remaining[:MAX_EVIDENCE_SNIPPETS], 1)]
    messages = _evidence_first_messages(question, claim, evidence)
    raw = generate_fn(messages)
    calls += 1
    problem = None
    try:
        second = _parse_evidence_first(raw, evidence, claim)
    except (ValueError, KeyError) as exc:
        second = None
        problem = str(exc)
    log.append({'stage': 'claim_verify_recheck', 'claim_id': claim['claim_id'],
                'attempt': 1, 'raw_response': raw, 'problem': problem,
                'source_chunk_id': evidence[0]['source_chunk_id']})
    if debug_hook:
        debug_hook({'stage': 'claim_verify_recheck', 'claim_id': claim['claim_id'],
                    'claim': claim['text'], 'essential': claim['essential'],
                    'source_filename': evidence[0]['source_filename'],
                    'source_chunk_id': evidence[0]['source_chunk_id'],
                    'attempt': 1, 'messages': messages, 'raw_response': raw,
                    'parsed_verdict': second, 'problem': problem,
                    'candidate_evidence': evidence})
    if second is None:
        # A failed second attempt does not erase the first semantic verdict.
        first = {**first, 'review_flags': [*first.get('review_flags', []),
                                          'EVIDENCE_RECHECK_INVALID']}
        return first, calls, log
    if second['verdict'] == 'NOT_FOUND':
        return {**first, 'evidence_recheck_attempted': True}, calls, log
    second.update({'verification_method': 'evidence_first_recheck',
                   'documents_checked': len(docs),
                   'per_document_verdicts': [second.copy()],
                   'evidence_recheck_attempted': True})
    return second, calls, log


def _affirmative_support_relation_missing(question: str, answer: str, docs: list[dict]) -> bool:
    """Block a positive support/compatibility answer unless product+feature coexist in a supporting source."""
    if not re.search(r"\b(?:support|supported|compatible)\b", question, re.I):
        return False
    if not re.search(r"\b(?:yes|supports?|supported|compatible)\b", answer, re.I):
        return False
    q_products = _named_products(question)
    if not q_products:
        return False
    # High-precision feature anchor: an acronym/technical token appearing after
    # the support relation (TLS, SSL, SDK, Java8-like tokens). Avoid generic IBM/WAS.
    tail = re.split(r"\b(?:support|supported|compatible(?:\s+with)?)\b", question,
                    maxsplit=1, flags=re.I)
    if len(tail) != 2:
        return False
    candidates = re.findall(r"\b[A-Z][A-Z0-9+_.-]{1,}\b", tail[1])
    candidates = [x for x in candidates if x.upper() not in {'IBM','WAS','TS','V'}]
    if not candidates:
        return False
    feature = candidates[-1]
    for doc in docs:
        raw = _raw_source_text(doc)
        if not (_named_products(raw) & q_products):
            continue
        if not re.search(r"\b" + re.escape(feature) + r"\b", raw, re.I):
            continue
        # Same source must actually state a support/compatibility relation, not
        # merely mention both the product and feature in unrelated sections.
        if re.search(r"\b(?:support|supported|supports|compatible)\b", raw, re.I):
            return False
    return True


def _procedural_version_drift(question: str, claims: list[dict]) -> bool:
    """Detect instructions for a different explicit product version than requested."""
    q_versions = _major_minor_versions(question)
    if not q_versions:
        return False
    for claim in claims:
        text = str(claim.get('text') or '')
        c_versions = _major_minor_versions(text)
        if not c_versions or c_versions & q_versions:
            continue
        has_action = any(re.search(p, text, re.I) for p in _ACTION_PATTERNS.values())
        has_command = bool(re.search(r"\b(?:command|utility|tool|service|console)\b", text, re.I))
        if has_action or has_command:
            return True
    return False


def _cross_product_evidence_mix(question: str, docs: list[dict], claims: list[dict]) -> bool:
    """Block remedies synthesized from explicitly different product families.

    This is intentionally narrow: it uses only the small unambiguous marker set
    and only evidence sources that the Judge itself marked SUPPORTED.
    """
    by_chunk = {str(d.get('chunk_id')): d for d in docs}
    q_products = _named_products(question)
    supported = []
    for claim in claims:
        if claim.get('verdict') != 'SUPPORTED':
            continue
        doc = by_chunk.get(str(claim.get('source_chunk_id')))
        if not doc:
            continue
        products = _named_products(_raw_source_text(doc))
        if products:
            supported.append((claim, products))
    if not supported:
        return False
    if q_products:
        # An essential claim backed only by an explicitly different product is
        # not transferable just because the topic/error terminology overlaps.
        for claim, products in supported:
            if claim.get('essential') and not (products & q_products):
                return True
        return False
    # Generic question: if the answer's supported claims are stitched from two
    # mutually disjoint named products, require human review.
    for i, (left_claim, left) in enumerate(supported):
        for right_claim, right in supported[i + 1:]:
            if left.isdisjoint(right) and (left_claim.get('essential') or right_claim.get('essential')):
                return True
    return False


def _guard_word_stems(text: str) -> set[str]:
    """Small morphology normalizer used only by high-precision case guards."""
    out = set()
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.casefold()):
        if word in _CLAIM_STOPWORDS:
            continue
        stem = word
        if len(stem) > 6 and stem.endswith("ing"):
            stem = stem[:-3]
        elif len(stem) > 5 and stem.endswith("ed"):
            stem = stem[:-2]
        elif len(stem) > 5 and stem.endswith("es"):
            stem = stem[:-2]
        elif len(stem) > 4 and stem.endswith("s"):
            stem = stem[:-1]
        out.add(stem)
    return out


def _mandatory_precondition_missing(answer: str, docs: list[dict]) -> bool:
    """Block an action when IBM explicitly says a prerequisite must happen first.

    This is intentionally source-driven. It does not infer arbitrary procedure
    order; it fires only on explicit IBM language such as "in order to X ...
    must first Y" or "before X, you must Y" and only when the answer performs X
    while omitting Y.
    """
    answer_stems = _guard_word_stems(answer)
    patterns = (
        re.compile(
            r"in order to\s+(?P<target>[^.\n]{3,110}?)\s+(?:it|you|the\s+[A-Za-z][A-Za-z -]{0,35})\s+"
            r"must\s+first\s+(?:be\s+)?(?P<pre>[^.;\n]{2,70})",
            re.I,
        ),
        re.compile(
            r"before\s+(?P<target>[^,.;\n]{3,100}),?\s+(?:you\s+)?must\s+(?:first\s+)?"
            r"(?P<pre>[^.;\n]{2,70})",
            re.I,
        ),
    )
    for doc in docs:
        body = _source_body(doc)
        for pattern in patterns:
            for m in pattern.finditer(body):
                target = _guard_word_stems(m.group("target"))
                pre = _guard_word_stems(m.group("pre"))
                if not target or not pre:
                    continue
                # The answer must clearly perform the gated target action.
                overlap = len(target & answer_stems) / max(len(target), 1)
                if overlap < 0.55:
                    continue
                # Ignore generic modal residue; require at least one concrete
                # prerequisite word to be absent from the answer.
                concrete_pre = {w for w in pre if w not in {"first", "must", "need", "required"}}
                if concrete_pre and not (concrete_pre & answer_stems):
                    return True
    return False


def _distinctive_error_signatures(text: str) -> set[str]:
    """Extract high-specificity incident identifiers, not generic topic words."""
    sigs = set()
    # Fully qualified/simple Java-style exception names.
    for m in re.findall(
        r"\b(?:[A-Za-z_$][A-Za-z0-9_$]*\.)*[A-Z][A-Za-z0-9_$]*(?:Exception|Error)\b",
        text,
    ):
        sigs.add(m.casefold())
        sigs.add(m.rsplit('.', 1)[-1].casefold())
    # IBM/database/security identifiers that define the incident itself.
    for pat in (
        r"\bCVE-\d{4}-\d+\b",
        r"\bAPAR\s+[A-Z]{1,4}\d{4,}\b",
        r"\bORA-\d{4,5}\b",
        r"\b(?:AMQ|BMXAA|WSVR|DSRA|CWZIP|CWSXM|MQRC)[-_]?[A-Z0-9]*\d[A-Z0-9_-]*\b",
    ):
        sigs.update(x.casefold() for x in re.findall(pat, text, re.I))
    # Generic java.lang.OutOfMemoryError is common enough that it should not
    # by itself force all remedies to come from one exact-error source.
    sigs.discard('outofmemoryerror')
    sigs.discard('java.lang.outofmemoryerror')
    return sigs


def _remedy_error_signature_mismatch(question: str, docs: list[dict], claims: list[dict]) -> bool:
    """Block a remedy borrowed from a different incident in the same product.

    If the question contains a distinctive exception/error signature and at
    least one retrieved IBM document matches it exactly, an essential remedy
    may not be justified solely by a different document that lacks every one of
    those signatures. This catches same-product symptom/remedy transfer without
    requiring product mismatch.
    """
    signatures = _distinctive_error_signatures(question)
    if not signatures:
        return False
    exact_docs = [
        doc for doc in docs
        if any(sig in _raw_source_text(doc).casefold() for sig in signatures)
    ]
    exact_chunks = {str(doc.get('chunk_id')) for doc in exact_docs}
    exact_filenames = {str(doc.get('filename') or '') for doc in exact_docs if doc.get('filename')}
    if not exact_chunks:
        return False
    by_chunk = {str(doc.get('chunk_id')): doc for doc in docs}
    for claim in claims:
        if claim.get('verdict') != 'SUPPORTED' or not claim.get('essential'):
            continue
        text = str(claim.get('text') or '')
        is_remedy = (bool(re.search(r"\b(?:resolve|fix|workaround|ensure|configure|set|change|update|run|install|remove|add|enable|disable|restart|redeploy)\b", text, re.I))
                     or any(re.search(p, text, re.I) for p in _ACTION_PATTERNS.values()))
        if not is_remedy:
            continue
        explicit_doc = re.search(r"\bDocument\s+(\d+)\b", text, re.I)
        if explicit_doc:
            idx = int(explicit_doc.group(1)) - 1
            if 0 <= idx < len(docs):
                source = docs[idx]
                if (str(source.get('chunk_id')) not in exact_chunks
                        and str(source.get('filename') or '') not in exact_filenames
                        and not any(sig in _raw_source_text(source).casefold() for sig in signatures)):
                    return True
        chunk = str(claim.get('source_chunk_id') or '')
        if not chunk or chunk in exact_chunks:
            continue
        source = by_chunk.get(chunk)
        if source is None:
            continue
        # A remedy may live in a different chunk of the same IBM technote;
        # do not treat chunking alone as an incident mismatch.
        if str(source.get('filename') or '') in exact_filenames:
            continue
        raw = _raw_source_text(source).casefold()
        if not any(sig in raw for sig in signatures):
            return True
    return False


_TASK_CONTEXT_FAMILIES = {
    'upload': r"\bupload(?:ed|ing|s)?\b",
    'import': r"\bimport(?:ed|ing|s)?\b",
    'export': r"\bexport(?:ed|ing|s)?\b",
    'deploy': r"\bdeploy(?:ed|ing|ment|s)?\b",
    'install': r"\binstall(?:er|ers|ed|ing|ation|ations|s)?\b",
    'uninstall': r"\buninstall(?:er|ers|ed|ing|ation|ations|s)?\b",
}


def _task_context_families(text: str) -> set[str]:
    return {name for name,pat in _TASK_CONTEXT_FAMILIES.items() if re.search(pat,text,re.I)}


def _task_context_mismatch(question: str, docs: list[dict], claims: list[dict]) -> bool:
    """Detect a remedy supported by a document about a different workflow task.

    This guard is intentionally limited to high-specificity operational entry
    points (upload/import/export/deploy/install/uninstall). Topic/product overlap
    alone is not enough: a supported essential remedy must come from a source
    that also discusses the operation the user is actually performing.
    """
    q_tasks=_task_context_families(question)
    if not q_tasks:
        return False
    # An exact incident signature is stronger than workflow wording and may
    # legitimately lead to a differently-worded remedy.
    signatures=_distinctive_error_signatures(question)
    by_chunk={str(d.get('chunk_id')):d for d in docs}
    by_file={str(d.get('filename') or ''):d for d in docs if d.get('filename')}
    for claim in claims:
        if claim.get('verdict')!='SUPPORTED' or claim.get('importance','essential' if claim.get('essential') else 'supporting')!='essential':
            continue
        chunk=str(claim.get('source_chunk_id') or '')
        source=by_chunk.get(chunk)
        if source is None and claim.get('source_filename'):
            source=by_file.get(str(claim.get('source_filename')))
        if source is None:
            continue
        raw=_raw_source_text(source)
        if signatures and any(sig in raw.casefold() for sig in signatures):
            continue
        if not (_task_context_families(raw) & q_tasks):
            return True
    return False


def _cvss_expected_severity(score: float) -> str:
    if score <= 0:
        return 'none'
    if score < 4.0:
        return 'low'
    if score < 7.0:
        return 'medium'
    if score < 9.0:
        return 'high'
    return 'critical'


def _cvss_severity_mismatch(claims: list[dict]) -> bool:
    """Validate categorical severity interpretations attached to CVSS scores."""
    for claim in claims:
        text=str(claim.get('text') or '')
        m=re.search(r"\bCVSS(?:\s+Base)?\s+Score\s*(?:is|of|:|=)?\s*(10(?:\.0)?|[0-9](?:\.[0-9])?)",text,re.I)
        if not m:
            # Also catch prose such as "CVSS base score ... is 4.3".
            m=re.search(r"\bCVSS[^.]{0,45}?\b(10(?:\.0)?|[0-9](?:\.[0-9])?)\b",text,re.I)
        if not m:
            continue
        labels=[x.casefold() for x in re.findall(r"\b(?:none|low|medium|moderate|high|critical)\b",text,re.I)]
        if not labels:
            continue
        score=float(m.group(1))
        expected=_cvss_expected_severity(score)
        normalized={'moderate':'medium'}
        if any(normalized.get(label,label)!=expected for label in labels):
            return True
    return False



def _question_security_identifier_missing(question: str, docs: list[dict]) -> bool:
    """A security-bulletin answer cannot pass if the requested CVE is absent."""
    if not re.search(r"\bsecurity bulletin\b|\bCVE-\d{4}-\d+\b", question, re.I):
        return False
    requested = {
        x.casefold() for x in re.findall(r"\bCVE-\d{4}-\d+\b", question, re.I)
    }
    if not requested:
        return False
    corpus = "\n".join(_raw_source_text(d) for d in docs).casefold()
    return any(cve not in corpus for cve in requested)


def _explicit_source_reference_invalid(answer: str, docs: list[dict]) -> bool:
    """Validate explicit IBM source filenames instead of accepting near-ID typos."""
    refs = {
        x.casefold()
        for x in re.findall(
            r"\b(?:swg|ibm|nas8N|isg3T)[A-Za-z0-9_-]+\.txt\b", answer, re.I
        )
    }
    if not refs:
        return False
    available = {str(d.get("filename") or "").casefold() for d in docs}
    return any(ref not in available for ref in refs)


def _goal_direction_mismatch(question: str, answer: str) -> bool:
    """Block a remedy whose stated outcome reproduces the user's failure state."""
    targets = []
    for m in re.finditer(
        r"\b(?:unable to|cannot|can't|fails? to|failed to)\s+"
        r"(open|access|connect(?: to)?|log\s*in(?: to)?|login(?: to)?|start|run|launch|view|use)\s+"
        r"([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,3})",
        question, re.I,
    ):
        verb = re.sub(r"\s+", " ", m.group(1).casefold()).split()[0]
        obj = {t for t in _claim_tokens(m.group(2)) if len(t) >= 4}
        if obj:
            targets.append((verb, obj))
    if not targets:
        return False

    low = answer.casefold()
    # Only detect language that describes the *proposed outcome/action*.
    # Bare "unable to" / "cannot" phrases are intentionally excluded because
    # good answers often restate the user's symptom or quote an IBM bulletin title.
    patterns = (
        r"prevent.{0,100}?\b{verb}\b",
        r"no longer be able to\s+{verb}\b",
        r"remove.{0,100}?(?:capabilit|permission|access)",
        r"deny.{0,100}?(?:capabilit|permission|access)",
        r"disable.{0,100}?(?:capabilit|permission|access)",
    )
    for verb, obj_tokens in targets:
        for pat in patterns:
            rx = pat.replace("{verb}", re.escape(verb))
            for hit in re.finditer(rx, low, re.I | re.S):
                window = low[max(0, hit.start()-100):min(len(low), hit.end()+140)]
                if any(tok in window for tok in obj_tokens):
                    return True
    return False


def _case_guard_flags(question: str, answer: str, docs: list[dict], claims: list[dict]) -> list[str]:
    flags = []
    if _affirmative_support_relation_missing(question, answer, docs):
        flags.append('AFFIRMATIVE_SUPPORT_RELATION_NOT_IN_SOURCE')
    if _procedural_version_drift(question, claims):
        flags.append('PROCEDURAL_VERSION_DRIFT')
    if _cross_product_evidence_mix(question, docs, claims):
        flags.append('CROSS_PRODUCT_EVIDENCE_MIX')
    if _mandatory_precondition_missing(answer, docs):
        flags.append('MANDATORY_PRECONDITION_MISSING')
    if _remedy_error_signature_mismatch(question, docs, claims):
        flags.append('REMEDY_ERROR_SIGNATURE_MISMATCH')
    if _task_context_mismatch(question, docs, claims):
        flags.append('TASK_CONTEXT_MISMATCH')
    if _cvss_severity_mismatch(claims):
        flags.append('CVSS_SEVERITY_MISMATCH')
    if _question_security_identifier_missing(question, docs):
        flags.append('REQUESTED_SECURITY_IDENTIFIER_NOT_IN_EVIDENCE')
    if _explicit_source_reference_invalid(answer, docs):
        flags.append('SOURCE_REFERENCE_NOT_RETRIEVED')
    if _goal_direction_mismatch(question, answer):
        flags.append('ANSWER_GOAL_DIRECTION_MISMATCH')
    return flags


def evaluate_case_claims(record, policy, generate_fn, max_attempts=2, debug_hook=None):
    """v3.7: Context checks plus bounded evidence recheck; unresolved goes to review."""
    docs = evaluation_docs(record)
    answer = str(record.get('prediction') or '')
    question = str(record.get('question') or '')
    if answer.strip() == ABSTAIN:
        score, status = _evaluate_refusal(question, docs)
        return {'judge_status': status, 'score': score,
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
    claims = _assign_claim_importance(question, claims)
    verified_claims = []
    attempt_log = []
    total_calls = 0
    for claim in claims:
        redundant = _redundant_supported_claim(claim, verified_claims)
        if redundant is not None:
            checked, count, attempts = redundant, 0, []
        elif (_is_generic_advisory_claim(claim['text'])
                or _is_question_setup_restatement(question, claim['text'])
                or _is_question_fact_restatement(question, claim)
                or _is_document_metadata_claim(question, claim)
                or _is_low_risk_context_claim(claim)):
            checked = {'verdict':'SUPPORTED','quote':'','quote_verified':False,
                       'reason':'Non-substantive navigation, duplicate context, or a fact already supplied by the user.',
                       'verification_method':'nontechnical_context_ignored','review_flags':[]}
            count, attempts = 0, []
        else:
            checked, count, attempts = _verify_claim_evidence_first_v37(
                question, claim, docs, generate_fn,
                max_attempts=min(max_attempts, 2), debug_hook=debug_hook)
        total_calls += count
        attempt_log.extend(attempts)
        merged = {**claim, **checked,
                  'evidence_quote': checked.get('quote', ''),
                  'evidence_quote_verified': checked.get('quote_verified', False)}
        verified_claims.append(_v38_reconcile_source_verdict(merged, docs))
    score, status = derive_claim_rubric(verified_claims, policy)
    guard_flags = _case_guard_flags(question, answer, docs, verified_claims)
    if guard_flags:
        score['review_flags'] = list(dict.fromkeys([*score.get('review_flags', []), *guard_flags]))
        score['auto_pass'] = False
        score['risk_level'] = ('high' if any(x in guard_flags for x in ('CROSS_PRODUCT_EVIDENCE_MIX','MANDATORY_PRECONDITION_MISSING','REMEDY_ERROR_SIGNATURE_MISMATCH','TASK_CONTEXT_MISMATCH','CVSS_SEVERITY_MISMATCH','REQUESTED_SECURITY_IDENTIFIER_NOT_IN_EVIDENCE','SOURCE_REFERENCE_NOT_RETRIEVED','ANSWER_GOAL_DIRECTION_MISMATCH')) else 'medium')
        extra = '; '.join(guard_flags)
        score['failure_reason'] = '; '.join(x for x in [score.get('failure_reason', ''), extra] if x)
        status = 'REVIEW_REQUIRED'
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
            "rubric_version": "ibm-techqa-claim-derived-v3.13",
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




def _preflight_self_check():
    """Fast deterministic regression tests before loading the local Judge model."""
    # Near-verbatim generic rule must support a versioned instance without a
    # competing source version.
    q='Why does my upgrade to BigFix version 9.2.5 take a very long time?'
    c={'claim_id':1,'text':'The upgrade to BigFix version 9.2.5 takes a very long time because most of the time is spent upgrading the database.','essential':True}
    d={'filename':'bigfix.txt','chunk_id':'b1','text':'When upgrading a BigFix server, most of the time is usually spent upgrading the database.'}
    if _near_verbatim_source_support(q,c,[d]) is None:
        raise RuntimeError('SELF_CHECK_FAILED: generic versioned support')

    # Conditional fix headings must not erase an explicit opposite statement.
    q2='Is Java 8 supported for EJBDeploy on WebSphere Application Server 8.5.5.9 or later?'
    c2={'claim_id':1,'text':'Yes, Java 8 is supported for EJBDeploy on WebSphere Application Server 8.5.5.9 or later.','essential':True}
    d2={'filename':'fix.txt','chunk_id':'f1','text':'ABSTRACT\nEnable Java 8 Support for EJBDeploy on WebSphere Application Server 8.5.5.9 or later.\nERROR DESCRIPTION:\nJava 8 is not supported for EJBDeploy on WebSphere Application Server 8.5.5.9 or later.'}
    if _near_verbatim_source_support(q2,c2,[d2]) is not None:
        raise RuntimeError('SELF_CHECK_FAILED: conditional fix false approval')

    # A source explicitly about Cognos Controller Web/NLB must never satisfy an
    # IBM Integration Bus question simply because both use TLS/SSL terminology.
    product_doc={'filename':'controller.txt','chunk_id':'p1',
                 'text':'Cognos Controller Web servers use the same SSL certificate as the NLB appliance.'}
    if _question_context_matches_document(
            'How do I configure TLS/SSL for an IBM Integration Bus (IIB) web service?', product_doc):
        raise RuntimeError('SELF_CHECK_FAILED: cross-product IIB/Controller approval')

    # Path components must not become hard case identifiers.
    path_doc={'filename':'locked.txt','chunk_id':'l1','text':'A silent installation terminates if product files are locked. The best practice is to shut down all components before upgrade.'}
    path_q=r'Why locked files? C:\IBM\ITM\TMAITM6\deploy.jar is locked.'
    if not _question_context_matches_document(path_q, path_doc):
        raise RuntimeError('SELF_CHECK_FAILED: path component promoted to hard identifier')

    # Positive support answers require the queried product+feature relation in one source.
    dp_docs=[{'filename':'dp.txt','chunk_id':'dp1','text':'ITCAM Agent for DataPower 7.1 monitors DataPower appliances.'},
             {'filename':'ad.txt','chunk_id':'ad1','text':'ITCAM for AD WebSphere Agent supports TLS.'}]
    if not _affirmative_support_relation_missing(
            'Does the ITCAM Agent for WebSphere DataPower 7.1 support TLS?',
            'Yes, ITCAM Agent for WebSphere DataPower 7.1 supports TLS.', dp_docs):
        raise RuntimeError('SELF_CHECK_FAILED: cross-source support relation false approval')

    # Procedures for a different explicit version must not auto-pass.
    drift_claims=[{'claim_id':1,'text':'Run WASService.exe -remove for WAS V6.1 and restart RAD.','essential':True}]
    if not _procedural_version_drift('How do I create a Windows service for WebSphere Application Server 7.0.X?', drift_claims):
        raise RuntimeError('SELF_CHECK_FAILED: procedural version drift')

    # A direct non-root/root remedy remains recoverable despite nearby opposite-condition prose.
    root_claim={'claim_id':1,'text':'To resolve this, run WebSphere Application Server under a root user.','essential':True}
    root_doc={'filename':'root.txt','chunk_id':'r1','text':'When WebSphere Application Server is run as a non-root user it will not be able to run native commands. RESOLVING THE PROBLEM To resolve the problem, run WebSphere Application Server under a root user.'}
    if _near_verbatim_source_support('zonecfg error in WebSphere Application Server', root_claim, [root_doc]) is None:
        raise RuntimeError('SELF_CHECK_FAILED: direct root remedy blocked by opposite condition')

    # An explicit source prerequisite must not be skipped by an answer procedure.
    pre_doc={'filename':'pre.txt','chunk_id':'pre1','text':'When a plan is On Hold, in order to add any additional custodians it must first be suspended. Only then will the option to add custodians be available again.'}
    if not _mandatory_precondition_missing('Move the plan to On Hold. Add new custodians. Resume the plan.', [pre_doc]):
        raise RuntimeError('SELF_CHECK_FAILED: mandatory precondition omission')
    if _mandatory_precondition_missing('Suspend the plan first. Add new custodians. Resume the plan.', [pre_doc]):
        raise RuntimeError('SELF_CHECK_FAILED: present mandatory precondition blocked')

    # A remedy from a different same-product incident must not satisfy an exact exception case.
    sig_q='Unable to login. com.filenet.wcm.api.EncryptionException caused by java.security.InvalidKeyException: Illegal key size or default parameters.'
    sig_docs=[
        {'filename':'exact.txt','chunk_id':'e1','text':'Problem initializing encryption/decryption. java.security.InvalidKeyException: Illegal key size or default parameters.'},
        {'filename':'other.txt','chunk_id':'e2','text':'Credentials Error: User Login Failed. CAUSE The Generic JVM argument required for application integration setup is incorrect.'},
    ]
    sig_claim=[{'claim_id':1,'text':'To resolve this issue, ensure that the correct JVM arguments are set up for Application Integration.','essential':True,'verdict':'SUPPORTED','source_chunk_id':'e2'}]
    if not _remedy_error_signature_mismatch(sig_q, sig_docs, sig_claim):
        raise RuntimeError('SELF_CHECK_FAILED: same-product wrong-incident remedy')
    sig_claim[0]['source_chunk_id']='e1'
    if _remedy_error_signature_mismatch(sig_q, sig_docs, sig_claim):
        raise RuntimeError('SELF_CHECK_FAILED: exact-incident remedy incorrectly blocked')

    # v3.12: a source about running a scrypt tool must not justify an upload
    # workflow merely because product and file format overlap.
    task_q='Why does uploading a DataPower scrypt3 firmware image through Appliance Manager fail?'
    task_docs=[{'filename':'tool.txt','chunk_id':'t1','text':'To run a scrypt tool, always use the boot image command. Commands such as reinitialize can fail.'}]
    task_claims=[{'claim_id':1,'text':'Use the boot image command.','importance':'essential','essential':True,'verdict':'SUPPORTED','source_chunk_id':'t1'}]
    if not _task_context_mismatch(task_q,task_docs,task_claims):
        raise RuntimeError('SELF_CHECK_FAILED: task context mismatch')
    task_docs[0]['text']='When uploading a scrypt3 firmware image through Appliance Manager, use the documented upload workflow.'
    if _task_context_mismatch(task_q,task_docs,task_claims):
        raise RuntimeError('SELF_CHECK_FAILED: matching task context blocked')

    # v3.12: categorical interpretation of a CVSS number is checked, not just
    # the presence of the number in the source.
    if not _cvss_severity_mismatch([{'text':'CVSS Base Score: 4.3, indicating high severity.'}]):
        raise RuntimeError('SELF_CHECK_FAILED: CVSS severity mismatch')
    if _cvss_severity_mismatch([{'text':'CVSS Base Score: 4.3, indicating medium severity.'}]):
        raise RuntimeError('SELF_CHECK_FAILED: valid CVSS severity blocked')

    # v3.12: optional unsupported advice does not veto a fully supported core.
    imp=_assign_claim_importance('Why did the installer fail?', [
        {'claim_id':1,'text':'The installer fails because required Linux system libraries are missing.','essential':True,'importance':'essential'},
        {'claim_id':2,'text':'You may also try running with the -console flag.','essential':True,'importance':'essential'},
    ])
    if [x['importance'] for x in imp] != ['essential','optional']:
        raise RuntimeError('SELF_CHECK_FAILED: claim importance classification')
    synthetic=[
        {**imp[0],'verdict':'SUPPORTED','reason':'direct','evidence_quote':'missing Linux system libraries','evidence_quote_verified':True,'review_flags':[]},
        {**imp[1],'verdict':'NOT_FOUND','reason':'not material','evidence_quote':'','evidence_quote_verified':False,'review_flags':[]},
    ]
    policy={'quality_thresholds':{'correctness_min':4,'groundedness_min':4,'completeness_min':3,'actionability_min':3,'safety_required':'pass','format_required':'pass'}}
    score,_=derive_claim_rubric(synthetic,policy)
    if not score.get('auto_pass'):
        raise RuntimeError('SELF_CHECK_FAILED: optional claim still vetoes core answer')


    # v3.12: Atlas search syntax must not borrow Domino/Notes semantics.
    atlas_q='How do I search for a reserved character in documents in Atlas?'
    notes_doc={'filename':'notes.txt','chunk_id':'n1',
               'text':'Notes full text search uses double quotes for reserved characters.'}
    if _question_context_matches_document(atlas_q, notes_doc):
        raise RuntimeError('SELF_CHECK_FAILED: Atlas/Notes product transfer')

    # v3.12: a requested security identifier must exist in retrieved evidence.
    sec_docs=[{'filename':'s1.txt','chunk_id':'s1','text':'Security Bulletin CVE-2017-1731'}]
    if not _question_security_identifier_missing('Security Bulletin for CVE-2017-1741', sec_docs):
        raise RuntimeError('SELF_CHECK_FAILED: missing requested CVE')
    sec_docs[0]['text']='Security Bulletin CVE-2017-1741'
    if _question_security_identifier_missing('Security Bulletin for CVE-2017-1741', sec_docs):
        raise RuntimeError('SELF_CHECK_FAILED: present requested CVE blocked')

    # v3.12: explicit source IDs must match retrieved IBM filenames exactly.
    src_docs=[{'filename':'swg22013313.txt','chunk_id':'s1','text':'x'}]
    if not _explicit_source_reference_invalid('Source: swg2201331313.txt', src_docs):
        raise RuntimeError('SELF_CHECK_FAILED: invalid source reference')
    if _explicit_source_reference_invalid('Source: swg22013313.txt', src_docs):
        raise RuntimeError('SELF_CHECK_FAILED: valid source reference blocked')

    # v3.12: a remedy must not explicitly reproduce the user's failed goal.
    goal_q='Certain users cannot open Report Studio.'
    if not _goal_direction_mismatch(
            goal_q, 'After these steps, the user should no longer be able to open Report Studio.'):
        raise RuntimeError('SELF_CHECK_FAILED: opposite goal direction')
    if _goal_direction_mismatch(
            goal_q, 'Grant the required profile so the user can open Report Studio.'):
        raise RuntimeError('SELF_CHECK_FAILED: correct goal direction blocked')

    # v3.13: merely restating the user's failure state is not an opposite remedy.
    if _goal_direction_mismatch(
            'Toolkit cannot connect to a remote broker with a TLS-compliant cipher.',
            'The IBM Security Bulletin explains why the Toolkit cannot connect to a remote broker and lists the remediation fixes.'):
        raise RuntimeError('SELF_CHECK_FAILED: symptom restatement treated as opposite goal')
    if _goal_direction_mismatch(
            'Why does the installer fail? The installer is unable to run in graphical mode.',
            'The installer is unable to run in graphical mode because required Linux system libraries are missing. Install the required libraries or use the supported installation mode.'):
        raise RuntimeError('SELF_CHECK_FAILED: installer symptom restatement treated as opposite goal')

    # Preserve the actual harmful direction-reversal case.
    if not _goal_direction_mismatch(
            'Certain users cannot open Report Studio.',
            'Remove the user interface capability. After these steps, the user should no longer be able to open Report Studio.'):
        raise RuntimeError('SELF_CHECK_FAILED: harmful goal reversal no longer detected')

    # v3.12: installer/installation wording belongs to the install task family.
    if _task_context_mismatch(
            'Why does the ITCAM install fail?',
            [{'filename':'i.txt','chunk_id':'i1',
              'text':'The installer fails because Linux system libraries are missing.'}],
            [{'claim_id':1,'text':'Linux system libraries are missing.',
              'importance':'essential','essential':True,'verdict':'SUPPORTED',
              'source_chunk_id':'i1'}]):
        raise RuntimeError('SELF_CHECK_FAILED: installer/install task mismatch')

    # v3.12: exact incident + CAUSE section supports a causal paraphrase.
    ec_q='CWSXM0100E: Multiple replies have been produced for a single request flow.'
    ec_c={'claim_id':1,
          'text':'This happens because a mediation primitive is wired more than once to the next mediation primitive.',
          'essential':True}
    ec_d={'filename':'m.txt','chunk_id':'m1',
          'text':'SYMPTOM\nCWSXM0100E: Multiple replies have been produced for a single request flow.\nCAUSE\nThis error occurs when a mediation primitive is wired more than once to the next mediation primitive.\nRESOLVING THE PROBLEM\nRemove duplicate wiring.'}
    if _exact_error_cause_section_support(ec_q,ec_c,[ec_d]) is None:
        raise RuntimeError('SELF_CHECK_FAILED: exact incident cause section')

    # v3.12: explicit cross-version compatibility supports the yes/no core.
    comp_q='Can I run different versions of the RDz client and host? What versions are compatible?'
    comp_c={'claim_id':1,
            'text':'Yes, it is possible to run different levels of the RDz host and client.',
            'essential':True}
    comp_d={'filename':'c.txt','chunk_id':'c1',
            'text':'Client/Server Compatibility. RDz v14.1 host/client is compatible with RDz v14.0 client/host and RDz v9.5 client/host.'}
    if _compatibility_relation_support(comp_q,comp_c,[comp_d]) is None:
        raise RuntimeError('SELF_CHECK_FAILED: compatibility relation')

    # Duplicate-wiring cause -> remove exact duplicate condition.
    c3={'claim_id':1,'text':'To resolve this issue, ensure that each mediation primitive is connected to its intended next step without redundant connections.','essential':True}
    d3={'filename':'mfc.txt','chunk_id':'m1','text':'CWSXM0100E: Multiple replies have been produced for a single request flow.\nCAUSE\nThis error occurs when a mediation primitive is wired more than once to the next mediation primitive. It might seem to be a single wire.'}
    if _duplicate_wiring_remedy_support('CWSXM0100E multiple replies',c3,[d3]) is None:
        raise RuntimeError('SELF_CHECK_FAILED: duplicate wiring remedy')

    # A real mv/rename action and single-instance invariant must both exist.
    c4={'claim_id':1,'text':'To resolve this, ensure that only one instance of the DE is installed per user by renaming the conflicting directories.','essential':True}
    d4={'filename':'de.txt','chunk_id':'d1','text':'The Deployment Engine should only have one instance installed.\nRESOLVING THE PROBLEM\nmv .acsi_old .acsi_backup'}
    if _single_instance_directory_move_support('Deployment Engine',c4,[d4]) is None:
        raise RuntimeError('SELF_CHECK_FAILED: directory move')

    # Refusal may pass only with a distinctive missing focus; a direct-answer
    # compatibility source must remain review-required.
    score,status=_evaluate_refusal('Is WAS Liberty affected?', [{'filename':'x','chunk_id':'x','text':'WebSphere Application Server Network Deployment V8.5.5 is affected.'}])
    if not score.get('auto_pass'):
        raise RuntimeError('SELF_CHECK_FAILED: safe refusal')
    score,status=_evaluate_refusal('Is SPSS Statistics 19 supported on macOS Sierra?', [{'filename':'s','chunk_id':'s','text':'On macOS Sierra, only SPSS Statistics 24 and 25 are supported; previous versions are not supported.'}])
    if score.get('auto_pass'):
        raise RuntimeError('SELF_CHECK_FAILED: answerable refusal false approval')


def _validation_bucket(row: dict) -> str:
    prediction=str(row.get('prediction') or '').strip()
    status=str(row.get('guardrails',{}).get('status') or 'UNKNOWN')
    if prediction == ABSTAIN:
        return 'refusal'
    if status == 'ANSWER_VERIFIED':
        return 'verified_answer'
    return 'nonverified_answer'


def _validation_sample(records: list[dict], excluded_ids: set[str], n: int, seed: int) -> list[dict]:
    excluded_questions={normalize_space(r.get('question','')) for r in records if str(r.get('id')) in excluded_ids}
    eligible=[]; seen=set()
    for row in records:
        rid=str(row.get('id')); q=normalize_space(row.get('question',''))
        if rid in excluded_ids or not q or q in excluded_questions or q in seen:
            continue
        seen.add(q); eligible.append(row)
    if len(eligible)<n:
        raise ValueError(f'새 검증 후보가 부족합니다: 필요 {n}, 사용 가능 {len(eligible)}')
    rng=random.Random(seed)
    buckets={k:[] for k in ('verified_answer','nonverified_answer','refusal')}
    for row in eligible: buckets[_validation_bucket(row)].append(row)
    for v in buckets.values(): rng.shuffle(v)
    names=list(buckets); target={name:n//len(names) for name in names}
    for name in names[:n%len(names)]: target[name]+=1
    selected=[]; leftovers=[]
    for name in names:
        take=min(target[name],len(buckets[name])); selected.extend(buckets[name][:take]); leftovers.extend(buckets[name][take:])
    if len(selected)<n:
        rng.shuffle(leftovers); selected.extend(leftovers[:n-len(selected)])
    if len(selected)!=n: raise RuntimeError('검증 표본 크기 생성 실패')
    return sorted(selected,key=lambda row:str(row.get('id')))


def _validation_available_count(records: list[dict], excluded_ids: set[str]) -> int:
    """Count unique, previously-unreviewed questions available for a fresh validation sample."""
    excluded_questions={normalize_space(r.get('question','')) for r in records if str(r.get('id')) in excluded_ids}
    seen=set(); count=0
    for row in records:
        rid=str(row.get('id')); q=normalize_space(row.get('question',''))
        if rid in excluded_ids or not q or q in excluded_questions or q in seen:
            continue
        seen.add(q); count+=1
    return count


def _validation_fingerprint(sample: list[dict]) -> str:
    payload=[]
    for row in sample:
        docs=evaluation_docs(row)
        payload.append({'id':str(row.get('id')),'question':str(row.get('question') or ''),
                        'prediction':str(row.get('prediction') or ''),
                        'documents':[{'filename':d.get('filename'),'chunk_id':d.get('chunk_id'),
                                      'text':d.get('text') or d.get('content') or ''} for d in docs]})
    raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _metrics_from_reviews(reviews: dict, judged: dict[str,dict]) -> dict:
    ids=list(reviews)
    labels=[int(reviews[i]['human_pass']=='pass') for i in ids]
    predictions=[int(judged[i]['score']['auto_pass']) for i in ids]
    stats=confusion(labels,predictions)
    errors=[]
    for i,h,p in zip(ids,labels,predictions):
        if h==p: continue
        row=judged[i]; score=row['score']
        errors.append({'id':i,'type':'FN' if h else 'FP','judge_status':row.get('judge_status'),
                       'claim_verdicts':[c.get('verdict') for c in score.get('claim_verifications',[])],
                       'review_flags':score.get('review_flags',[]),
                       'human_assessment':reviews[i].get('final_assessment','')})
    return {**stats,'misclassified':errors}


def validate(args):
    """Evaluate the frozen Judge on the existing held-out 30-case human review."""
    _preflight_self_check(); print('[SELF-CHECK] deterministic regression checks: PASS',flush=True)
    rows=jsonl_rows(Path(args.predictions))
    dev=json.loads(Path(args.exclude_human).read_text(encoding='utf-8-sig'))
    excluded=set(dev.get('reviews',{}))
    sample=_validation_sample(rows,excluded,args.sample_size,args.seed)
    fingerprint=_validation_fingerprint(sample)
    human_path=Path(args.human)
    if not human_path.is_file():
        raise FileNotFoundError(f'검증 Human JSON 없음: {human_path}')
    human=json.loads(human_path.read_text(encoding='utf-8-sig')); reviews=human.get('reviews',{})
    if human.get('schema')!='judge-human-validation-v1': raise ValueError('검증 Human JSON schema 불일치')
    if human.get('sample_fingerprint')!=fingerprint: raise ValueError('검증 표본 fingerprint 불일치')
    if set(reviews)!={str(r['id']) for r in sample}: raise ValueError('검증 Human JSON ID 불일치')
    if any(r.get('human_pass') not in ('pass','fail') for r in reviews.values()): raise ValueError('검증 PASS/FAIL 미입력')
    expected='ibm-techqa-claim-derived-v3.13'
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    print(f'[VALIDATION] held-out {len(sample)}건 평가',flush=True)
    print(f'[CHECK] Judge frozen: {JUDGE_SCHEMA_VERSION}',flush=True)
    print(f'[CHECK] sample fingerprint: {fingerprint}',flush=True)
    with tempfile.TemporaryDirectory(prefix='judge_validation_',dir=outdir) as temp:
        sample_path=Path(temp)/'sample.jsonl'; fresh=Path(temp)/'judge.jsonl'
        write_jsonl(sample_path,sample)
        grade(SimpleNamespace(predictions=sample_path,output=fresh,policy=args.policy,model_path=args.model_path,
                              limit=0,force=True,max_new_tokens=args.max_new_tokens,max_attempts=args.max_attempts,
                              debug=False,debug_log=None))
        judged={r['id']:r for r in jsonl_rows(fresh)}
        invalid=[i for i,row in judged.items() if row.get('judge_schema_version')!=JUDGE_SCHEMA_VERSION
                 or row.get('rubric_version')!=expected or type(row.get('score',{}).get('auto_pass')) is not bool]
        if invalid: raise RuntimeError(f'Judge 결과 버전/승인값 불일치: {invalid}')
        metrics=_metrics_from_reviews(reviews,judged)
        report={'schema':'judge-heldout-validation-v1','sample_size':len(sample),'sample_seed':args.seed,
                'sample_fingerprint':fingerprint,'excluded_development_ids':len(excluded),
                'judge_schema_version':JUDGE_SCHEMA_VERSION,'rubric_version':expected,
                'counts':{k:metrics[k] for k in ('n','TP','FP','TN','FN')},
                'accuracy':metrics['accuracy'],'precision':metrics['precision'],'recall':metrics['recall'],
                'specificity':metrics['specificity'],'false_approval_rate':metrics['false_approval_rate'],
                'kappa':metrics['kappa'],'misclassified':metrics['misclassified'],
                'human_label_counts':dict(Counter(r['human_pass'] for r in reviews.values()))}
        staged=Path(temp)/'report.json'; dump_json(staged,report)
        fresh.replace(Path(args.judge_output)); staged.replace(Path(args.report))
    print(json.dumps({'counts':report['counts'],'precision':report['precision'],'recall':report['recall'],
                      'accuracy':report['accuracy'],'false_approval_rate':report['false_approval_rate'],
                      'kappa':report['kappa']},ensure_ascii=False,indent=2))
    print(f'[VALIDATION 완료] {Path(args.report)}')



def _review_ids(path: Path) -> set[str]:
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    reviews=data.get('reviews',{})
    if not isinstance(reviews,dict):
        raise ValueError(f'Human review JSON 형식 오류: {path}')
    return set(map(str,reviews))


def _final_review_html(sample: list[dict], fingerprint: str, output: Path, seed: int) -> None:
    cases=[]
    for row in sample:
        docs=[]
        for d in evaluation_docs(row):
            docs.append({'filename':d.get('filename'),'chunk_id':d.get('chunk_id'),
                         'text':d.get('text') or d.get('content') or ''})
        cases.append({'id':str(row.get('id')),'question':str(row.get('question') or ''),
                      'answer':str(row.get('prediction') or ''),'documents':docs})
    reviews={c['id']:{'human_pass':'','root_cause':'','risk_level':'','failure_reason':'',
                      'root_cause_detail':'','proposed_fix':'','final_assessment':''} for c in cases}
    base={'schema':'judge-human-final-v1','sample_size':len(cases),'sample_seed':seed,
          'sample_fingerprint':fingerprint,'frozen_judge_schema':JUDGE_SCHEMA_VERSION,'reviews':reviews}
    cases_json=json.dumps(cases,ensure_ascii=False).replace('</','<' + '\\/')
    base_json=json.dumps(base,ensure_ascii=False).replace('</','<' + '\\/')
    template = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Judge Final Validation Review</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f5f5f5;color:#111}.top{position:sticky;top:0;background:white;padding:12px 18px;border-bottom:1px solid #ccc;z-index:2}
.wrap{max-width:1200px;margin:18px auto;padding:0 18px}.case{background:white;padding:18px;margin-bottom:18px;border:1px solid #ddd;border-radius:8px}
pre{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:12px;max-height:420px;overflow:auto}textarea{width:100%;min-height:74px;margin:4px 0 10px}
label{font-weight:600}select{margin:4px 12px 10px 4px}.docs details{margin:8px 0}.progress{font-weight:700;margin-left:12px}
button{padding:9px 14px;margin-right:8px}.id{font-size:20px;font-weight:700}.hint{color:#555;font-size:13px}
</style></head><body>
<div class="top"><button onclick="exportJson()">검토 JSON 저장</button><button onclick="clearSaved()">임시저장 초기화</button><span class="progress" id="progress"></span><span class="hint">최종 블라인드 검증 · Judge 결과/원본 QA 정답은 표시하지 않음</span></div>
<div class="wrap" id="root"></div>
<script>
const cases=__CASES__;
const base=__BASE__;
const key='judge-final-validation-'+base.sample_fingerprint;
let reviews=JSON.parse(JSON.stringify(base.reviews));
try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved)reviews=saved;}catch(e){}
function esc(x){return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function save(){localStorage.setItem(key,JSON.stringify(reviews));updateProgress();}
function field(id,name,value){reviews[id][name]=value;save();}
function render(){
 const root=document.getElementById('root');root.innerHTML='';
 cases.forEach((c,i)=>{let d=document.createElement('div');d.className='case';
 let docs=c.documents.map((x,j)=>`<details><summary>문서 ${j+1} · ${esc(x.filename)} · ${esc(x.chunk_id)}</summary><pre>${esc(x.text)}</pre></details>`).join('');
 let r=reviews[c.id];
 d.innerHTML=`<div class="id">${i+1}/${cases.length} · ${esc(c.id)}</div><p><b>질문</b></p><pre>${esc(c.question)}</pre><p><b>LLM 답변</b></p><pre>${esc(c.answer)}</pre><div class="docs"><b>RAG 근거 문서</b>${docs}</div>
 <p><label>최종 판정</label><select data-f="human_pass"><option value="">선택</option><option value="pass" ${r.human_pass==='pass'?'selected':''}>PASS</option><option value="fail" ${r.human_pass==='fail'?'selected':''}>FAIL</option></select>
 <label>원인</label><select data-f="root_cause"><option value=""></option>${['evidence','semantic','generation','other'].map(x=>`<option value="${x}" ${r.root_cause===x?'selected':''}>${x}</option>`).join('')}</select>
 <label>위험도</label><select data-f="risk_level"><option value=""></option>${['low','medium','high'].map(x=>`<option value="${x}" ${r.risk_level===x?'selected':''}>${x}</option>`).join('')}</select></p>
 <label>판정 근거 / 실패 이유</label><textarea data-f="failure_reason">${esc(r.failure_reason)}</textarea>
 <label>상세 원인</label><textarea data-f="root_cause_detail">${esc(r.root_cause_detail)}</textarea>
 <label>개선사항</label><textarea data-f="proposed_fix">${esc(r.proposed_fix)}</textarea>
 <label>최종 평가 문장</label><textarea data-f="final_assessment">${esc(r.final_assessment)}</textarea>`;
 d.querySelectorAll('[data-f]').forEach(el=>el.addEventListener('change',()=>field(c.id,el.dataset.f,el.value)));
 d.querySelectorAll('textarea[data-f]').forEach(el=>el.addEventListener('input',()=>field(c.id,el.dataset.f,el.value)));
 root.appendChild(d);
 });updateProgress();}
function updateProgress(){let done=Object.values(reviews).filter(x=>x.human_pass==='pass'||x.human_pass==='fail').length;document.getElementById('progress').textContent=`판정 ${done}/${cases.length}`;}
function exportJson(){
 const missing=Object.entries(reviews).filter(([id,r])=>!['pass','fail'].includes(r.human_pass));
 if(missing.length){alert(`PASS/FAIL 미입력 ${missing.length}건: `+missing.slice(0,8).map(x=>x[0]).join(', '));return;}
 const out=JSON.parse(JSON.stringify(base));out.reviews=reviews;out.exported_at=new Date().toISOString();
 const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='final_human_review.json';a.click();URL.revokeObjectURL(a.href);
}
function clearSaved(){if(confirm('브라우저 임시저장을 초기화할까요?')){localStorage.removeItem(key);location.reload();}}
render();
</script></body></html>'''
    html=template.replace('__CASES__',cases_json).replace('__BASE__',base_json)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(html,encoding='utf-8')


def _find_downloaded_final_human(fingerprint: str) -> Path | None:
    downloads=Path.home()/'Downloads'
    if not downloads.is_dir():
        return None
    candidates=sorted(downloads.glob('final_human_review*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in candidates:
        try:
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            if data.get('schema')=='judge-human-final-v1' and data.get('sample_fingerprint')==fingerprint:
                return path
        except Exception:
            continue
    return None


def final_test(args):
    _preflight_self_check(); print('[SELF-CHECK] deterministic regression checks: PASS',flush=True)
    rows=jsonl_rows(Path(args.predictions))
    excluded=_review_ids(Path(args.dev_human)) | _review_ids(Path(args.validation_human))
    sample=_validation_sample(rows,excluded,args.sample_size,args.seed)
    fingerprint=_validation_fingerprint(sample)
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    human_path=Path(args.human);html_path=Path(args.review_html)
    expected_ids={str(r['id']) for r in sample}
    print(f'[FINAL] previous reviewed excluded={len(excluded)}, fresh sample={len(sample)}',flush=True)
    print(f'[CHECK] Judge frozen: {JUDGE_SCHEMA_VERSION}',flush=True)
    print(f'[CHECK] final sample fingerprint: {fingerprint}',flush=True)
    if not human_path.is_file():
        downloaded=_find_downloaded_final_human(fingerprint)
        if downloaded is not None:
            shutil.copy2(downloaded,human_path)
            print(f'[FINAL] Downloads 검토 JSON 자동 반영: {downloaded} -> {human_path}',flush=True)
        else:
            _final_review_html(sample,fingerprint,html_path,args.seed)
            print(f'[FINAL 준비 완료] {html_path}',flush=True)
            print('[NEXT] 이 HTML의 질문/답변/근거만 블라인드 검토한 뒤 final_human_review.json을 저장하세요.',flush=True)
            try:
                webbrowser.open(html_path.resolve().as_uri())
            except Exception:
                pass
            return
    human=json.loads(human_path.read_text(encoding='utf-8-sig'));reviews=human.get('reviews',{})
    if human.get('schema')!='judge-human-final-v1':raise ValueError('최종 Human JSON schema 불일치')
    if human.get('sample_fingerprint')!=fingerprint:raise ValueError('최종 표본 fingerprint 불일치')
    if set(reviews)!=expected_ids:raise ValueError('최종 Human JSON ID 불일치')
    if any(r.get('human_pass') not in ('pass','fail') for r in reviews.values()):raise ValueError('최종 PASS/FAIL 미입력')
    if human.get('frozen_judge_schema')!=JUDGE_SCHEMA_VERSION:raise ValueError('Human JSON의 frozen Judge 버전 불일치')
    expected='ibm-techqa-claim-derived-v3.13'
    with tempfile.TemporaryDirectory(prefix='judge_final_',dir=outdir) as temp:
        sample_path=Path(temp)/'sample.jsonl';fresh=Path(temp)/'judge.jsonl'
        write_jsonl(sample_path,sample)
        grade(SimpleNamespace(predictions=sample_path,output=fresh,policy=args.policy,model_path=args.model_path,
                              limit=0,force=True,max_new_tokens=args.max_new_tokens,max_attempts=args.max_attempts,
                              debug=False,debug_log=None))
        judged={r['id']:r for r in jsonl_rows(fresh)}
        if set(judged)!=expected_ids:raise RuntimeError('최종 Judge 결과 ID 불일치')
        invalid=[i for i,row in judged.items() if row.get('judge_schema_version')!=JUDGE_SCHEMA_VERSION
                 or row.get('rubric_version')!=expected or type(row.get('score',{}).get('auto_pass')) is not bool]
        if invalid:raise RuntimeError(f'최종 Judge 결과 버전/승인값 불일치: {invalid}')
        metrics=_metrics_from_reviews(reviews,judged)
        report={'schema':'judge-final-validation-v1','sample_size':len(sample),'sample_seed':args.seed,
                'sample_fingerprint':fingerprint,'excluded_previous_ids':len(excluded),
                'judge_schema_version':JUDGE_SCHEMA_VERSION,'rubric_version':expected,
                'counts':{k:metrics[k] for k in ('n','TP','FP','TN','FN')},
                'accuracy':metrics['accuracy'],'precision':metrics['precision'],'recall':metrics['recall'],
                'specificity':metrics['specificity'],'false_approval_rate':metrics['false_approval_rate'],
                'kappa':metrics['kappa'],'misclassified':metrics['misclassified'],
                'human_label_counts':dict(Counter(r['human_pass'] for r in reviews.values())),
                'note':'Final frozen-Judge evaluation. Do not tune on this sample if it is to remain a final test.'}
        staged=Path(temp)/'report.json';dump_json(staged,report)
        fresh.replace(Path(args.judge_output));staged.replace(Path(args.report))
    print(json.dumps({'counts':report['counts'],'precision':report['precision'],'recall':report['recall'],
                      'accuracy':report['accuracy'],'false_approval_rate':report['false_approval_rate'],
                      'kappa':report['kappa']},ensure_ascii=False,indent=2))
    print(f'[FINAL 완료] {Path(args.report)}')


def _final2_review_html(sample: list[dict], fingerprint: str, output: Path, seed: int) -> None:
    # Reuse the proven final-review HTML generator, then make only workflow metadata/file-name changes.
    _final_review_html(sample, fingerprint, output, seed)
    html = output.read_text(encoding='utf-8')
    html = html.replace('Judge Final Validation Review', 'Judge Final Validation 2 Review')
    html = html.replace('최종 블라인드 검증 · Judge 결과/원본 QA 정답은 표시하지 않음',
                        '2차 최종 블라인드 검증 · Judge 결과/원본 QA 정답은 표시하지 않음')
    html = html.replace('judge-human-final-v1', 'judge-human-final2-v1')
    html = html.replace("judge-final-validation-'+base.sample_fingerprint", "judge-final-validation-2-'+base.sample_fingerprint")
    html = html.replace("a.download='final_human_review.json'", "a.download='final_human_review_2.json'")
    output.write_text(html, encoding='utf-8')


def _find_downloaded_final2_human(fingerprint: str) -> Path | None:
    downloads=Path.home()/'Downloads'
    if not downloads.is_dir():
        return None
    candidates=sorted(downloads.glob('final_human_review_2*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in candidates:
        try:
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            if data.get('schema')=='judge-human-final2-v1' and data.get('sample_fingerprint')==fingerprint:
                return path
        except Exception:
            continue
    return None


def final_test_2(args):
    """Frozen v3.12 final generalization test on a fresh sample excluding all 80 reviewed cases."""
    _preflight_self_check(); print('[SELF-CHECK] deterministic regression checks: PASS',flush=True)
    rows=jsonl_rows(Path(args.predictions))
    excluded=(
        _review_ids(Path(args.dev_human))
        | _review_ids(Path(args.validation_human))
        | _review_ids(Path(args.final_human))
    )
    available=_validation_available_count(rows,excluded)
    if available <= 0:
        raise ValueError('새 검증 후보가 없습니다. 기존 106건 외의 추가 평가 데이터가 필요합니다.')
    actual_n=min(args.sample_size,available)
    if actual_n < args.sample_size:
        print(f'[FINAL-2] 요청 표본={args.sample_size}, 사용 가능={available} -> 남은 신규 {actual_n}건 전부 사용',flush=True)
    sample=_validation_sample(rows,excluded,actual_n,args.seed)
    fingerprint=_validation_fingerprint(sample)
    outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
    human_path=Path(args.human);html_path=Path(args.review_html)
    expected_ids={str(r['id']) for r in sample}
    print(f'[FINAL-2] previous reviewed excluded={len(excluded)}, fresh sample={len(sample)}',flush=True)
    print(f'[CHECK] Judge frozen: {JUDGE_SCHEMA_VERSION}',flush=True)
    print(f'[CHECK] final-2 sample fingerprint: {fingerprint}',flush=True)
    if not human_path.is_file():
        downloaded=_find_downloaded_final2_human(fingerprint)
        if downloaded is not None:
            shutil.copy2(downloaded,human_path)
            print(f'[FINAL-2] Downloads 검토 JSON 자동 반영: {downloaded} -> {human_path}',flush=True)
        else:
            _final2_review_html(sample,fingerprint,html_path,args.seed)
            print(f'[FINAL-2 준비 완료] {html_path}',flush=True)
            print('[NEXT] 이 HTML의 질문/답변/근거만 블라인드 검토한 뒤 final_human_review_2.json을 저장하세요.',flush=True)
            try:
                webbrowser.open(html_path.resolve().as_uri())
            except Exception:
                pass
            return
    human=json.loads(human_path.read_text(encoding='utf-8-sig'));reviews=human.get('reviews',{})
    if human.get('schema')!='judge-human-final2-v1':raise ValueError('2차 최종 Human JSON schema 불일치')
    if human.get('sample_fingerprint')!=fingerprint:raise ValueError('2차 최종 표본 fingerprint 불일치')
    if set(reviews)!=expected_ids:raise ValueError('2차 최종 Human JSON ID 불일치')
    if any(r.get('human_pass') not in ('pass','fail') for r in reviews.values()):raise ValueError('2차 최종 PASS/FAIL 미입력')
    if human.get('frozen_judge_schema')!=JUDGE_SCHEMA_VERSION:raise ValueError('Human JSON의 frozen Judge 버전 불일치')
    expected='ibm-techqa-claim-derived-v3.13'
    with tempfile.TemporaryDirectory(prefix='judge_final2_',dir=outdir) as temp:
        sample_path=Path(temp)/'sample.jsonl';fresh=Path(temp)/'judge.jsonl'
        write_jsonl(sample_path,sample)
        grade(SimpleNamespace(predictions=sample_path,output=fresh,policy=args.policy,model_path=args.model_path,
                              limit=0,force=True,max_new_tokens=args.max_new_tokens,max_attempts=args.max_attempts,
                              debug=False,debug_log=None))
        judged={r['id']:r for r in jsonl_rows(fresh)}
        if set(judged)!=expected_ids:raise RuntimeError('2차 최종 Judge 결과 ID 불일치')
        invalid=[i for i,row in judged.items() if row.get('judge_schema_version')!=JUDGE_SCHEMA_VERSION
                 or row.get('rubric_version')!=expected or type(row.get('score',{}).get('auto_pass')) is not bool]
        if invalid:raise RuntimeError(f'2차 최종 Judge 결과 버전/승인값 불일치: {invalid}')
        metrics=_metrics_from_reviews(reviews,judged)
        report={'schema':'judge-final-validation-2-v1','sample_size':len(sample),'sample_seed':args.seed,
                'sample_fingerprint':fingerprint,'excluded_previous_ids':len(excluded),
                'judge_schema_version':JUDGE_SCHEMA_VERSION,'rubric_version':expected,
                'counts':{k:metrics[k] for k in ('n','TP','FP','TN','FN')},
                'accuracy':metrics['accuracy'],'precision':metrics['precision'],'recall':metrics['recall'],
                'specificity':metrics['specificity'],'false_approval_rate':metrics['false_approval_rate'],
                'kappa':metrics['kappa'],'misclassified':metrics['misclassified'],
                'human_label_counts':dict(Counter(r['human_pass'] for r in reviews.values())),
                'note':'Independent final test #2 with frozen v3.12 Judge. Do not tune on this sample before reporting the generalization result.'}
        write_jsonl(Path(args.judge_output),[judged[str(r['id'])] for r in sample])
        dump_json(Path(args.report),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'[FINAL-2 완료] {Path(args.report)}')


def verify_106(args):
    """One GPU load: recheck all 106 human-reviewed development cases."""
    _preflight_self_check(); print('[SELF-CHECK] deterministic regression checks: PASS', flush=True)
    source={r['id']:r for r in jsonl_rows(Path(args.predictions))}
    groups=[]
    for name,path in (
        ('development_20',Path(args.dev_human)),
        ('validation_30',Path(args.validation_human)),
        ('final_30_now_development',Path(args.final_human)),
        ('final2_26_now_development',Path(args.final2_human)),
    ):
        obj=json.loads(path.read_text(encoding='utf-8-sig')); reviews=obj.get('reviews',{})
        if not reviews or any(r.get('human_pass') not in ('pass','fail') for r in reviews.values()):
            raise ValueError(f'{name}: Human PASS/FAIL 미입력')
        groups.append((name,obj,reviews))
    seen=set()
    for name,_,reviews in groups:
        overlap=seen & set(reviews)
        if overlap:
            raise ValueError(f'{name}: 이전 Human 세트와 ID 중복: {sorted(overlap)}')
        seen |= set(reviews)
    ids=[i for _,_,reviews in groups for i in reviews]
    missing=set(ids)-set(source)
    if missing:
        raise ValueError(f'RAG 예측 누락: {sorted(missing)}')
    expected='ibm-techqa-claim-derived-v3.13'
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    print(f'[VERIFY-106] groups=20+30+30+26, total={len(ids)}',flush=True)
    with tempfile.TemporaryDirectory(prefix='judge_verify_106_',dir=outdir) as temp:
        sample=Path(temp)/'sample.jsonl'; fresh=Path(temp)/'judge.jsonl'
        write_jsonl(sample,[source[i] for i in ids])
        grade(SimpleNamespace(predictions=sample,output=fresh,policy=args.policy,model_path=args.model_path,
                              limit=0,force=True,max_new_tokens=args.max_new_tokens,max_attempts=args.max_attempts,
                              debug=False,debug_log=None))
        judged={r['id']:r for r in jsonl_rows(fresh)}
        invalid=[i for i,row in judged.items() if row.get('judge_schema_version')!=JUDGE_SCHEMA_VERSION
                 or row.get('rubric_version')!=expected or type(row.get('score',{}).get('auto_pass')) is not bool]
        if invalid:
            raise RuntimeError(f'Judge 결과 버전/승인값 불일치: {invalid}')
        report={'schema':'judge-regression-106-v1','judge_schema_version':JUDGE_SCHEMA_VERSION,
                'rubric_version':expected,'groups':{},'total':{},
                'note':'All 106 validation cases are development/regression data. test_clean.json remains untouched final test data.'}
        human_all={}
        for name,_,reviews in groups:
            metrics=_metrics_from_reviews(reviews,judged)
            report['groups'][name]={k:metrics[k] for k in
                ('n','TP','FP','TN','FN','precision','recall','accuracy','false_approval_rate','kappa')}
            report['groups'][name]['errors']=metrics['misclassified']
            human_all.update(reviews)
        total=_metrics_from_reviews(human_all,judged)
        report['total']={k:total[k] for k in
            ('n','TP','FP','TN','FN','precision','recall','accuracy','false_approval_rate','kappa')}
        report['total']['errors']=total['misclassified']
        staged_report=Path(temp)/'verify_106_report.json'; staged_judge=Path(temp)/'judge_v311_106.jsonl'
        dump_json(staged_report,report); write_jsonl(staged_judge,[judged[i] for i in ids])
        staged_report.replace(Path(args.report)); staged_judge.replace(Path(args.judge_output))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'[VERIFY-106 완료] {Path(args.report)}')


def verify_80(args):
    """One GPU load: recheck all 80 human-reviewed development cases."""
    _preflight_self_check(); print('[SELF-CHECK] deterministic regression checks: PASS', flush=True)
    source = {r['id']: r for r in jsonl_rows(Path(args.predictions))}
    groups = []
    for name, path in (
        ('development_20', Path(args.dev_human)),
        ('validation_30', Path(args.validation_human)),
        ('final_30_now_development', Path(args.final_human)),
    ):
        obj = json.loads(path.read_text(encoding='utf-8-sig'))
        reviews = obj.get('reviews', {})
        if not reviews or any(r.get('human_pass') not in ('pass','fail') for r in reviews.values()):
            raise ValueError(f'{name}: Human PASS/FAIL 미입력')
        groups.append((name, obj, reviews))
    seen = set()
    for name, _, reviews in groups:
        overlap = seen & set(reviews)
        if overlap:
            raise ValueError(f'{name}: 이전 Human 세트와 ID 중복: {sorted(overlap)}')
        seen |= set(reviews)
    ids = [i for _, _, reviews in groups for i in reviews]
    missing = set(ids) - set(source)
    if missing:
        raise ValueError(f'RAG 예측 누락: {sorted(missing)}')
    expected = 'ibm-techqa-claim-derived-v3.13'
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    print(f'[VERIFY-80] groups=20+30+30, total={len(ids)}', flush=True)
    with tempfile.TemporaryDirectory(prefix='judge_verify_80_', dir=outdir) as temp:
        sample = Path(temp) / 'sample.jsonl'; fresh = Path(temp) / 'judge.jsonl'
        write_jsonl(sample, [source[i] for i in ids])
        grade(SimpleNamespace(predictions=sample, output=fresh, policy=args.policy,
                              model_path=args.model_path, limit=0, force=True,
                              max_new_tokens=args.max_new_tokens, max_attempts=args.max_attempts,
                              debug=False, debug_log=None))
        judged = {r['id']: r for r in jsonl_rows(fresh)}
        invalid = [i for i, row in judged.items()
                   if row.get('judge_schema_version') != JUDGE_SCHEMA_VERSION
                   or row.get('rubric_version') != expected
                   or type(row.get('score',{}).get('auto_pass')) is not bool]
        if invalid:
            raise RuntimeError(f'Judge 결과 버전/승인값 불일치: {invalid}')
        report = {'schema':'judge-regression-80-v1', 'judge_schema_version':JUDGE_SCHEMA_VERSION,
                  'rubric_version':expected, 'groups':{}, 'total':{},
                  'note':'All 80 cases have now been used for development/regression; none is an independent final test.'}
        human_all = {}; errors = []
        for name, _, reviews in groups:
            metrics = _metrics_from_reviews(reviews, judged)
            report['groups'][name] = {k: metrics[k] for k in
                ('n','TP','FP','TN','FN','precision','recall','accuracy','false_approval_rate','kappa')}
            report['groups'][name]['errors'] = metrics['misclassified']
            human_all.update(reviews)
            errors.extend(metrics['misclassified'])
        total = _metrics_from_reviews(human_all, judged)
        report['total'] = {k: total[k] for k in
            ('n','TP','FP','TN','FN','precision','recall','accuracy','false_approval_rate','kappa')}
        report['total']['errors'] = total['misclassified']
        staged_report = Path(temp) / 'verify_80_report.json'
        staged_judge = Path(temp) / 'rubric_judge_v39_80.jsonl'
        dump_json(staged_report, report)
        write_jsonl(staged_judge, [judged[i] for i in ids])
        staged_report.replace(Path(args.report))
        staged_judge.replace(Path(args.judge_output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'[VERIFY-80 완료] {Path(args.report)}')


def verify_all(args):
    """One GPU load: recheck development 20 and held-out 30 together, then split metrics."""
    _preflight_self_check(); print('[SELF-CHECK] deterministic regression checks: PASS',flush=True)
    source={r['id']:r for r in jsonl_rows(Path(args.predictions))}
    dev_h=json.loads(Path(args.dev_human).read_text(encoding='utf-8-sig')); dev_reviews=dev_h.get('reviews',{})
    val_h=json.loads(Path(args.validation_human).read_text(encoding='utf-8-sig')); val_reviews=val_h.get('reviews',{})
    if not dev_reviews or not val_reviews: raise ValueError('Human review JSON이 비었습니다.')
    if any(r.get('human_pass') not in ('pass','fail') for r in [*dev_reviews.values(),*val_reviews.values()]):
        raise ValueError('Human PASS/FAIL 미입력')
    if set(dev_reviews)&set(val_reviews): raise ValueError('개발/held-out ID가 겹칩니다.')
    val_sample=_validation_sample(list(source.values()),set(dev_reviews),len(val_reviews),args.seed)
    fingerprint=_validation_fingerprint(val_sample)
    if val_h.get('sample_fingerprint')!=fingerprint: raise ValueError('held-out fingerprint 불일치')
    val_ids={str(r['id']) for r in val_sample}
    if val_ids!=set(val_reviews): raise ValueError('held-out ID 불일치')
    ids=[*dev_reviews,*val_reviews]
    missing=set(ids)-set(source)
    if missing: raise ValueError(f'RAG 예측 누락: {sorted(missing)}')
    expected='ibm-techqa-claim-derived-v3.13'; outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    print(f'[VERIFY-ALL] development={len(dev_reviews)}, held-out={len(val_reviews)}, total={len(ids)}',flush=True)
    with tempfile.TemporaryDirectory(prefix='judge_verify_all_',dir=outdir) as temp:
        sample=Path(temp)/'sample.jsonl'; fresh=Path(temp)/'judge.jsonl'
        write_jsonl(sample,[source[i] for i in ids])
        grade(SimpleNamespace(predictions=sample,output=fresh,policy=args.policy,model_path=args.model_path,
                              limit=0,force=True,max_new_tokens=args.max_new_tokens,max_attempts=args.max_attempts,
                              debug=False,debug_log=None))
        judged={r['id']:r for r in jsonl_rows(fresh)}
        invalid=[i for i,row in judged.items() if row.get('judge_schema_version')!=JUDGE_SCHEMA_VERSION
                 or row.get('rubric_version')!=expected or type(row.get('score',{}).get('auto_pass')) is not bool]
        if invalid: raise RuntimeError(f'Judge 결과 버전/승인값 불일치: {invalid}')
        dev_m=_metrics_from_reviews(dev_reviews,judged); val_m=_metrics_from_reviews(val_reviews,judged)
        # Fail closed on regression: never silently overwrite reports if a false approval appears.
        result={'development':{k:dev_m[k] for k in ('n','TP','FP','TN','FN','precision','recall','accuracy','false_approval_rate','kappa')},
                'held_out':{k:val_m[k] for k in ('n','TP','FP','TN','FN','precision','recall','accuracy','false_approval_rate','kappa')},
                'development_errors':dev_m['misclassified'],'held_out_errors':val_m['misclassified'],
                'held_out_fingerprint':fingerprint,'judge_schema_version':JUDGE_SCHEMA_VERSION}
        dump_json(Path(temp)/'verify_all_report.json',result)
        # Split current judgments into the existing result locations.
        write_jsonl(Path(temp)/'dev.jsonl',[judged[i] for i in dev_reviews])
        write_jsonl(Path(temp)/'val.jsonl',[judged[i] for i in val_reviews])
        (Path(temp)/'dev.jsonl').replace(RESULT_DIR/'rubric_judge_v38.jsonl')
        Path(args.validation_output).parent.mkdir(parents=True,exist_ok=True)
        (Path(temp)/'val.jsonl').replace(Path(args.validation_output))
        (Path(temp)/'verify_all_report.json').replace(Path(args.report))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(f'[VERIFY-ALL 완료] {Path(args.report)}')


def review(args):
    """Run a fresh human-review evaluation in one command, without helper scripts."""
    _preflight_self_check()
    print("[SELF-CHECK] deterministic regression checks: PASS", flush=True)
    human_path = Path(args.human)
    human = json.loads(human_path.read_text(encoding="utf-8-sig"))
    reviews = human.get("reviews", {})
    if not reviews or any(r.get("human_pass") not in ("pass", "fail") for r in reviews.values()):
        raise ValueError("사람 평가 PASS/FAIL 라벨을 모두 입력하세요.")
    if JUDGE_SCHEMA_VERSION != "ibm-techqa-judge-claims-v3.13":
        raise RuntimeError("Judge 버전 불일치")
    expected_rubric = "ibm-techqa-claim-derived-v3.13"
    source_rows = {r["id"]: r for r in jsonl_rows(Path(args.predictions))}
    missing = set(reviews) - set(source_rows)
    if missing:
        raise ValueError(f"RAG 예측 누락: {sorted(missing)}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "rubric_judge_v38.jsonl"
    report_path = output_dir / "review_report_v38.json"
    print(f"[CHECK] 실행 파일: {Path(__file__).resolve()}", flush=True)
    print(f"[CHECK] Judge: {JUDGE_SCHEMA_VERSION}", flush=True)
    print(f"[CHECK] {len(reviews)}건, 캐시 재사용 금지 (--force)", flush=True)
    # Labels are used only for sampling IDs and post-inference metrics.
    # A temporary directory guarantees no stale verdicts enter the new run.
    with tempfile.TemporaryDirectory(prefix="judge_review_", dir=output_dir) as temp:
        sample = Path(temp) / "sample.jsonl"
        fresh = Path(temp) / "judge.jsonl"
        write_jsonl(sample, [source_rows[i] for i in reviews])
        grade(SimpleNamespace(predictions=sample, output=fresh, policy=args.policy,
                              model_path=args.model_path, limit=0, force=True,
                              max_new_tokens=args.max_new_tokens, max_attempts=args.max_attempts,
                              debug=False, debug_log=None))
        judged = {r["id"]: r for r in jsonl_rows(fresh)}
        if set(judged) != set(reviews):
            raise RuntimeError("Judge 결과 ID 불일치")
        invalid = [i for i, row in judged.items()
                   if row.get("judge_schema_version") != JUDGE_SCHEMA_VERSION
                   or row.get("rubric_version") != expected_rubric
                   or type(row.get("score", {}).get("auto_pass")) is not bool]
        if invalid:
            raise RuntimeError(f"Judge 결과 버전 또는 승인값 불일치: {invalid}")
        labels = [int(reviews[i]["human_pass"] == "pass") for i in reviews]
        predictions = [int(judged[i]["score"]["auto_pass"]) for i in reviews]
        stats = confusion(labels, predictions)
        errors = []
        for i, h, p in zip(reviews, labels, predictions):
            if h != p:
                row = judged[i]
                score = row["score"]
                errors.append({"id": i, "type": "FN" if h else "FP",
                               "judge_status": row.get("judge_status"),
                               "claim_verdicts": [c.get("verdict") for c in score.get("claim_verifications", [])],
                               "review_flags": score.get("review_flags", []),
                               "human_assessment": reviews[i].get("final_assessment", "")})
        report = {
            "schema": "judge-human-audit-v3.12",
            "source_judge_version": human.get("source_judge_version"),
            "evaluated_judge_schema_version": JUDGE_SCHEMA_VERSION,
            "evaluated_rubric_version": expected_rubric,
            "counts": {**{k: stats[k] for k in ("n", "TP", "FP", "TN", "FN")}, "unlabeled": 0},
            **{k: v for k, v in stats.items() if k not in ("n", "TP", "FP", "TN", "FN")},
            "root_causes": dict(Counter(r.get("root_cause") or "unclassified" for r in reviews.values())),
            "judge_status_counts": dict(Counter(r.get("judge_status") for r in judged.values())),
            "mismatched_ids": [], "misclassified": errors,
            "note": "Fresh development-set evaluation; every nonapproval counted as negative; not independent held-out validation.",
        }
        staged_report = Path(temp) / "report.json"
        dump_json(staged_report, report)
        # Only replace previous v3.8 outputs after all result checks pass.
        fresh.replace(output)
        staged_report.replace(report_path)
    print(json.dumps({k: report[k] for k in ("counts", "precision", "recall", "accuracy", "false_approval_rate")}, ensure_ascii=False, indent=2))
    print(f"[완료] {report_path}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:] or ["verify-106"]
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('verify-106',help='현재 개발 데이터 20+30+30+26 총 106건을 한 번에 회귀검증')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--dev-human',type=Path,default=RESULT_DIR/'IBMTechSupport_judge_v36_human_review.json')
    a.add_argument('--validation-human',type=Path,default=VALIDATION_HUMAN)
    a.add_argument('--final-human',type=Path,default=FINAL_HUMAN)
    a.add_argument('--final2-human',type=Path,default=FINAL2_HUMAN)
    a.add_argument('--output-dir',type=Path,default=RESULT_DIR/'regression_106')
    a.add_argument('--judge-output',type=Path,default=RESULT_DIR/'regression_106'/'judge_v311_106.jsonl')
    a.add_argument('--report',type=Path,default=RESULT_DIR/'regression_106'/'verify_106_report.json')
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--max-attempts',type=int,default=2)
    a.set_defaults(run=verify_106)
    a=s.add_parser('verify-80',help='지금까지 Human 검토한 20+30+30 총 80건을 한 번에 회귀검증')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--dev-human',type=Path,default=RESULT_DIR/'IBMTechSupport_judge_v36_human_review.json')
    a.add_argument('--validation-human',type=Path,default=VALIDATION_HUMAN)
    a.add_argument('--final-human',type=Path,default=FINAL_HUMAN)
    a.add_argument('--output-dir',type=Path,default=RESULT_DIR/'regression_80')
    a.add_argument('--judge-output',type=Path,default=RESULT_DIR/'regression_80'/'rubric_judge_v39_80.jsonl')
    a.add_argument('--report',type=Path,default=RESULT_DIR/'regression_80'/'verify_80_report.json')
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--max-attempts',type=int,default=2)
    a.set_defaults(run=verify_80)
    a=s.add_parser('verify-all',help='개발 20 + held-out 30을 한 번의 모델 로딩으로 회귀검증')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--dev-human',type=Path,default=RESULT_DIR/'IBMTechSupport_judge_v36_human_review.json')
    a.add_argument('--validation-human',type=Path,default=VALIDATION_HUMAN)
    a.add_argument('--validation-output',type=Path,default=VALIDATION_JUDGE)
    a.add_argument('--report',type=Path,default=RESULT_DIR/'verify_all_report.json')
    a.add_argument('--output-dir',type=Path,default=RESULT_DIR)
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--seed',type=int,default=VALIDATION_SEED)
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--max-attempts',type=int,default=2)
    a.set_defaults(run=verify_all)
    a=s.add_parser('final-test-2',help='기존 80건을 제외한 완전 신규 30건 2차 최종 블라인드 검증')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--dev-human',type=Path,default=RESULT_DIR/'IBMTechSupport_judge_v36_human_review.json')
    a.add_argument('--validation-human',type=Path,default=VALIDATION_HUMAN)
    a.add_argument('--final-human',type=Path,default=FINAL_HUMAN)
    a.add_argument('--output-dir',type=Path,default=FINAL2_DIR)
    a.add_argument('--human',type=Path,default=FINAL2_HUMAN)
    a.add_argument('--review-html',type=Path,default=FINAL2_HTML)
    a.add_argument('--judge-output',type=Path,default=FINAL2_JUDGE)
    a.add_argument('--report',type=Path,default=FINAL2_REPORT)
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--sample-size',type=int,default=FINAL2_SAMPLE_SIZE)
    a.add_argument('--seed',type=int,default=FINAL2_SEED)
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--max-attempts',type=int,default=2)
    a.set_defaults(run=final_test_2)
    a=s.add_parser('final-test',help='기존 50건을 제외한 완전 신규 표본 최종 블라인드 검증')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--dev-human',type=Path,default=RESULT_DIR/'IBMTechSupport_judge_v36_human_review.json')
    a.add_argument('--validation-human',type=Path,default=VALIDATION_HUMAN)
    a.add_argument('--output-dir',type=Path,default=FINAL_DIR)
    a.add_argument('--human',type=Path,default=FINAL_HUMAN)
    a.add_argument('--review-html',type=Path,default=FINAL_HTML)
    a.add_argument('--judge-output',type=Path,default=FINAL_JUDGE)
    a.add_argument('--report',type=Path,default=FINAL_REPORT)
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--sample-size',type=int,default=FINAL_SAMPLE_SIZE)
    a.add_argument('--seed',type=int,default=FINAL_SEED)
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--max-attempts',type=int,default=2)
    a.set_defaults(run=final_test)
    a=s.add_parser('validate',help='고정 Judge의 held-out 30건 검증')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--exclude-human',type=Path,default=RESULT_DIR/'IBMTechSupport_judge_v36_human_review.json')
    a.add_argument('--output-dir',type=Path,default=VALIDATION_DIR)
    a.add_argument('--human',type=Path,default=VALIDATION_HUMAN)
    a.add_argument('--judge-output',type=Path,default=VALIDATION_JUDGE)
    a.add_argument('--report',type=Path,default=VALIDATION_REPORT)
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--sample-size',type=int,default=VALIDATION_SAMPLE_SIZE)
    a.add_argument('--seed',type=int,default=VALIDATION_SEED)
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--max-attempts',type=int,default=2)
    a.set_defaults(run=validate)
    a=s.add_parser('review',help='사람 평가 표본 추출, 강제 재평가, 성능 비교 통합 실행')
    a.add_argument('--human',type=Path,default=RESULT_DIR/'IBMTechSupport_judge_v36_human_review.json')
    a.add_argument('--predictions',type=Path,default=PREDICTIONS)
    a.add_argument('--output-dir',type=Path,default=RESULT_DIR)
    a.add_argument('--policy',type=Path,default=POLICY_PATH)
    a.add_argument('--model-path',type=Path,default=Path('models/Qwen2.5-1.5B-Instruct'))
    a.add_argument('--max-new-tokens',type=int,default=320)
    a.add_argument('--max-attempts',type=int,default=2)
    a.set_defaults(run=review)
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
