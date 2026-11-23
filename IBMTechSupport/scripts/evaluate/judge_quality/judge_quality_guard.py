from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
from typing import Any

POTENTIAL_FALSE_CONTRADICTION = "POTENTIAL_FALSE_CONTRADICTION"
REASON_RESTATES_CLAIM = "REASON_RESTATES_CLAIM"
EXACT_TEXT_CONTRADICTION = "EXACT_TEXT_CONTRADICTION"
EVIDENCE_ID_INVALID = "EVIDENCE_ID_INVALID"
MODEL_RESPONSE_INVALID = "MODEL_RESPONSE_INVALID"
UNRESOLVED_CLAIM = "UNRESOLVED_CLAIM"


def normalize(value: str) -> str:
    """Normalize punctuation and whitespace; keep negation and numbers."""
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def count_overlap_f1(left: str, right: str) -> float:
    """Token overlap screening metric, NOT an entailment probability."""
    a = Counter(normalize(left).split())
    b = Counter(normalize(right).split())
    if not a or not b:
        return 0.0
    shared = sum((a & b).values())
    return 2 * shared / (sum(a.values()) + sum(b.values()))


def same_polarity(claim: str, quote: str) -> bool:
    """Avoid similarity flags when explicit negation counts differ.

    This *does not* prove the sentences have the same meaning.
    """
    negations = {"no", "not", "never", "without", "cannot", "neither", "nor"}
    a = Counter(x for x in normalize(claim).split() if x in negations)
    b = Counter(x for x in normalize(quote).split() if x in negations)
    return a == b


def same_numbers(claim: str, quote: str) -> bool:
    """Prevent similarity alerts based on incompatible numeric facts."""
    a = set(re.findall(r"\d+(?:\.\d+)*", claim))
    b = set(re.findall(r"\d+(?:\.\d+)*", quote))
    return not a or not b or a == b


def contradiction_risk(claim: dict[str, Any]) -> tuple[str, str] | None:
    """Return a review signal; no heuristic here can establish support."""
    if claim.get("verdict") != "CONTRADICTED":
        return None
    text, quote = str(claim.get("text") or ""), str(claim.get("quote") or "")
    lhs, rhs = normalize(text), normalize(quote)
    if not lhs or not rhs:
        return None
    if lhs == rhs and len(lhs) >= 24:
        return EXACT_TEXT_CONTRADICTION, "근거 인용문과 Claim이 문자 수준에서 동일하지만 CONTRADICTED로 판정됨"
    if len(lhs) >= 40 and lhs in rhs:
        return EXACT_TEXT_CONTRADICTION, "Claim 전체가 근거 인용문 안에 있지만 CONTRADICTED로 판정됨"

    # Equal numerical conditions are not alone proof of entailment, but are a useful
    # candidate for manual inspection even when a longer Claim is paraphrased.
    constraint_pattern = r"\b(?:fewer|less|more|greater)\s+than\s+\d+(?:\.\d+)?\b"
    claim_constraints = set(re.findall(constraint_pattern, lhs))
    quote_constraints = set(re.findall(constraint_pattern, rhs))
    if claim_constraints and claim_constraints.intersection(quote_constraints):
        return POTENTIAL_FALSE_CONTRADICTION, (
            "Claim과 근거에 같은 수치 조건이 나타나므로 CONTRADICTED 판정 재검토 필요"
        )

    reason = str(claim.get("reason") or "")
    norm_reason = normalize(reason)
    if ("explicitly states that" in norm_reason
            and "contradict" not in norm_reason
            and same_polarity(text, reason)
            and same_numbers(text, reason)
            and count_overlap_f1(text, reason) >= 0.68):
        return REASON_RESTATES_CLAIM, (
            "CONTRADICTED 판정 사유가 오히려 Claim의 내용을 근거가 명시한다고 설명함"
        )

    similarity = count_overlap_f1(text, quote)
    if similarity >= 0.65 and same_polarity(text, quote) and same_numbers(text, quote):
        return POTENTIAL_FALSE_CONTRADICTION, (
            f"Claim과 근거의 어휘 중첩 F1={similarity:.2f}; "
            "부정 표현·숫자의 명시적 불일치가 없어 사람이 의미적 모순 여부를 검토해야 함"
        )
    return None


def validate_model_verdict(
    response: dict[str, Any], candidate_evidence_ids: set[str]
) -> tuple[bool, str]:
    """Use in judge response parsing before accepting a model output.

    - SUPPORTED/CONTRADICTED must reference one of the provided candidates.
    - NOT_FOUND cannot claim a positive evidence ID.
    - A valid structure is NOT a guarantee of semantic correctness.
    """
    if not isinstance(response, dict):
        return False, "INVALID_JSON_OBJECT"
    verdict = response.get("verdict")
    if verdict not in {"SUPPORTED", "CONTRADICTED", "NOT_FOUND"}:
        return False, "INVALID_VERDICT"
    reason = response.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return False, "MISSING_REASON"
    evidence_id = response.get("evidence_id")
    if verdict in {"SUPPORTED", "CONTRADICTED"}:
        if not isinstance(evidence_id, str) or evidence_id not in candidate_evidence_ids:
            return False, "EVIDENCE_ID_NOT_IN_CANDIDATES"
    elif evidence_id not in (None, ""):
        return False, "UNEXPECTED_EVIDENCE_ID_FOR_NOT_FOUND"
    return True, "OK"


def audit(records: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    issues: list[dict[str, str]] = []
    status_counts, verdict_counts, attempt_issues = Counter(), Counter(), Counter()
    auto_pass_count, attempt_count = 0, 0
    affected_ids: set[str] = set()

    def emit(record_id: str, claim_id: Any, kind: str, detail: str, recommendation: str) -> None:
        affected_ids.add(record_id)
        issues.append({
            "id": record_id,
            "claim_id": "" if claim_id is None else str(claim_id),
            "issue": kind,
            "detail": detail,
            "recommendation": recommendation,
        })

    for record in records:
        record_id = str(record.get("id", "<unknown>"))
        status_counts[str(record.get("judge_status", "<missing>"))] += 1
        score = record.get("score") or {}
        auto_pass_count += score.get("auto_pass") is True
        attempts = record.get("attempts") or []
        attempt_count += len(attempts)
        for item in attempts:
            problem = item.get("problem")
            if problem:
                attempt_issues[problem] += 1
                if problem == "EVIDENCE_ID_NOT_IN_CANDIDATES":
                    emit(record_id, item.get("claim_id"), EVIDENCE_ID_INVALID,
                         f"attempt={item.get('attempt')}, source_chunk_id={item.get('source_chunk_id')}",
                         "후보 ID를 프롬프트에 명시·검증하고 오류 시 1회 재시도; 반복 오류는 UNRESOLVED로 분기")
                else:
                    emit(record_id, item.get("claim_id"), MODEL_RESPONSE_INVALID,
                         f"attempt={item.get('attempt')}, problem={problem}",
                         "스키마 오류를 1회만 재요청하고 이후 REVIEW_REQUIRED 처리")
        for claim in score.get("claim_verifications") or []:
            verdict_counts[str(claim.get("verdict", "<missing>"))] += 1
            claim_id = claim.get("claim_id")
            if claim.get("verdict") == "UNRESOLVED":
                emit(record_id, claim_id, UNRESOLVED_CLAIM,
                     str(claim.get("reason", "")), "사람의 검토 후 판단; 자동 승인 금지")
            risk = contradiction_risk(claim)
            if risk is not None:
                issue, reason = risk
                emit(record_id, claim_id, issue, reason,
                     "모델의 모순 판단을 그대로 수용하지 말고 원문과 Claim을 사람이 대조; 자동 SUPPORTED 전환 금지")

    blocking_codes = {EXACT_TEXT_CONTRADICTION, POTENTIAL_FALSE_CONTRADICTION,
                      REASON_RESTATES_CLAIM, UNRESOLVED_CLAIM}
    blocking_ids = {i["id"] for i in issues if i["issue"] in blocking_codes}
    blocking_ids.update(str(r.get("id", "<unknown>")) for r in records
                        if r.get("judge_status") == "REVIEW_REQUIRED")
    summary = {
        "total_records": len(records),
        "status_counts": dict(status_counts),
        "auto_pass_count": auto_pass_count,
        "verdict_counts": dict(verdict_counts),
        "total_model_attempts": attempt_count,
        "attempt_problems": dict(attempt_issues),
        "audit_issue_counts": dict(Counter(x["issue"] for x in issues)),
        "blocking_record_count": len(blocking_ids),
        "blocking_record_ids": sorted(blocking_ids),
        "audit_affected_record_count": len(affected_ids),
        "audit_affected_record_ids": sorted(affected_ids),
        "disclaimer": "Overlap flags are screening signals, not entailment proofs; source JSONL is unchanged.",
    }
    return issues, summary


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Line {line_num}: JSON object expected")
        records.append(obj)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit"])
    parser.add_argument("jsonl_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("judge_quality_audit"))
    parser.add_argument("--fail-on-review", action="store_true", help="Exit 2 for semantic suspicion or REVIEW_REQUIRED (not transient formatting errors)")
    args = parser.parse_args()
    records = load_jsonl(args.jsonl_path)
    issues, summary = audit(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "audit_issues.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "claim_id", "issue", "detail", "recommendation"])
        writer.writeheader()
        writer.writerows(issues)
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if args.fail_on_review and summary["blocking_record_count"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
