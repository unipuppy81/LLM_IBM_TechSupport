from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_CONFIG = PROJECT_DIR / "configs" / "rag" / "retrieval_stage2_selected.yaml"

# Conservative, explicit identifiers.  Product matching is intentionally soft;
# CVE/APAR/error-code conflicts can be treated as hard conflicts because their
# lexical identity is much less ambiguous.
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,8}\b", re.I)
APAR_RE = re.compile(r"\b(?:IJ|IT|PH|PI|JR|PM|PK|IC|IV|IY|IZ|LI|LO|OA|SE|SW)\d{5,7}\b", re.I)
ERROR_RE = re.compile(
    r"\b(?:ORA-\d{3,6}|SQL(?:CODE)?\s*(?:[=:]\s*)?-?\d{3,5}|"
    r"[A-Z]{2,}[A-Z0-9_-]*\d{3,}[A-Z0-9_-]*)\b",
    re.I,
)
SQLCODE_RE = re.compile(r"^SQL(?:CODE)?\s*(?:[=:]\s*)?(-?\d{3,5})$", re.I)
VERSION_CUED_RE = re.compile(
    r"\b(?:version|ver\.?|release|fix\s*pack|fixpack|fp|v)\s*[:=]?\s*"
    r"(\d+(?:\.\d+){1,3}(?:[-._][A-Za-z0-9]+)?)\b",
    re.I,
)
VERSION_LONG_RE = re.compile(r"(?<![\d.])\d+\.\d+\.\d+(?:\.\d+)?(?![\d.])")

# A deliberately bounded lexicon.  It is used only for soft scoring.  Adding a
# broad/generic token here can hurt retrieval precision, so phrases are kept
# product-specific where possible.
PRODUCT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("websphere_application_server", (r"\bwebsphere application server\b",)),
    ("websphere_portal", (r"\bwebsphere portal\b",)),
    ("websphere_commerce", (r"\bwebsphere commerce\b",)),
    ("websphere_process_server", (r"\bwebsphere process server\b",)),
    ("websphere_extreme_scale", (r"\bwebsphere extreme scale\b",)),
    ("websphere_service_registry", (r"\bwebsphere service registry(?: and repository)?\b",)),
    ("websphere_mq", (r"\bwebsphere mq\b", r"\bibm mq\b", r"\bmqseries\b")),
    ("websphere_message_broker", (r"\bwebsphere message broker\b", r"\bmessage broker\b")),
    ("integration_bus", (r"\bibm integration bus\b", r"\bintegration bus\b")),
    ("datapower", (r"\bdatapower\b",)),
    ("db2", (r"\bdb2\b",)),
    ("tivoli_workload_scheduler", (r"\btivoli workload scheduler\b", r"\bworkload scheduler\b", r"\bibm workload scheduler\b")),
    ("dynamic_workload_console", (r"\btivoli dynamic workload console\b", r"\bdynamic workload console\b", r"\btdwc\b")),
    ("tivoli_storage_manager", (r"\btivoli storage manager\b", r"\bspectrum protect\b")),
    ("tivoli_directory_server", (r"\btivoli directory server\b", r"\bibm security directory server\b")),
    ("tivoli_monitoring", (r"\btivoli monitoring\b", r"\bibm tivoli monitoring\b")),
    ("endpoint_manager", (r"\bibm endpoint manager\b", r"\btivoli endpoint manager\b")),
    ("maximo", (r"\bmaximo\b",)),
    ("qradar", (r"\bqradar\b",)),
    ("cognos", (r"\bcognos\b",)),
    ("filenet", (r"\bfilenet\b",)),
    ("lotus_domino", (r"\blotus domino\b", r"\bdomino server\b")),
    ("sametime", (r"\bsametime\b",)),
    ("informix", (r"\binformix\b",)),
    ("sterling", (r"\bsterling (?:b2b integrator|file gateway|connect)\b",)),
    ("appscan", (r"\bappscan\b",)),
    ("security_access_manager", (r"\bsecurity access manager\b",)),
    ("security_identity_manager", (r"\bsecurity identity manager\b",)),
    ("business_process_manager", (r"\bbusiness process manager\b", r"\bibm bpm\b")),
    ("mobilefirst", (r"\bmobilefirst\b", r"\bworklight\b")),
    ("guardium", (r"\bguardium\b",)),
    ("bigfix", (r"\bbigfix\b",)),
    ("clearcase", (r"\bclearcase\b",)),
    ("clearquest", (r"\bclearquest\b",)),
    ("rational_software_architect", (r"\brational software architect\b",)),
    ("rational_application_developer", (r"\brational application developer\b",)),
    ("rational_team_concert", (r"\brational team concert\b",)),
    ("rational_doors", (r"\bibm rational doors\b", r"\brational doors\b")),
    ("ibm_http_server", (r"\bibm http server\b",)),
    ("ibm_installation_manager", (r"\bibm installation manager\b",)),
    ("ibm_connections", (r"\bibm connections\b",)),
    ("ibm_content_manager", (r"\bibm content manager\b",)),
    ("ibm_case_manager", (r"\bibm case manager\b",)),
    ("netcool_omnibus", (r"\bnetcool(?:/omnibus)?\b", r"\bibm tivoli netcool/omnibus\b",)),
    ("pureapplication", (r"\bpureapplication\b",)),
    ("urbancode", (r"\burbancode\b",)),
)
_COMPILED_PRODUCTS = tuple(
    (name, tuple(re.compile(pattern, re.I) for pattern in patterns))
    for name, patterns in PRODUCT_PATTERNS
)

BOILERPLATE_RE = re.compile(
    r"\b(?:please\s+help|can\s+anyone\s+help|can\s+someone\s+help|"
    r"does\s+anyone\s+know|i\s+would\s+like\s+to\s+know|"
    r"thanks\s+in\s+advance|thank\s+you)\b",
    re.I,
)
TECHNICAL_CUE_RE = re.compile(
    r"\b(?:error|exception|fail(?:ed|ure)?|unable|cannot|can't|hang|crash|"
    r"install|upgrade|configure|configuration|deploy|start|stop|connect|"
    r"certificate|security|vulnerability|timeout|permission|database|server|"
    r"client|firmware|boot|migration|restore|backup|trace|log)\b",
    re.I,
)


@dataclass(frozen=True)
class RetrievalSignals:
    cves: frozenset[str]
    apars: frozenset[str]
    errors: frozenset[str]
    versions: frozenset[str]
    products: frozenset[str]


@dataclass(frozen=True)
class ConstraintAssessment:
    score: float
    hard_conflict: bool
    reasons: tuple[str, ...]


def _upper_set(values: Iterable[str]) -> frozenset[str]:
    return frozenset(v.upper().replace(" ", "") for v in values if v)


def extract_signals(text: str) -> RetrievalSignals:
    text = text or ""
    cves = _upper_set(CVE_RE.findall(text))
    apars = _upper_set(APAR_RE.findall(text))

    errors = set()
    for raw in ERROR_RE.findall(text):
        sql = SQLCODE_RE.fullmatch(raw.strip())
        if sql:
            # Canonicalize formatting variants such as SQLCODE=-911,
            # SQLCODE -911 and SQL -911 to the same exact identifier.
            token = f"SQLCODE{int(sql.group(1))}"
        else:
            token = re.sub(r"\s+", "", raw.upper())
        if CVE_RE.fullmatch(token) or APAR_RE.fullmatch(token):
            continue
        # Ignore bare years / plain numeric fragments accidentally caught by a
        # permissive vendor-code pattern.
        if not any(ch.isalpha() for ch in token):
            continue
        errors.add(token)

    versions = {m.group(1).casefold() for m in VERSION_CUED_RE.finditer(text)}
    versions.update(v.casefold() for v in VERSION_LONG_RE.findall(text))

    products = set()
    for name, patterns in _COMPILED_PRODUCTS:
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            products.add(name)
            # Product names are often written as "DataPower 10.0" without an
            # explicit "version" cue.  Accept only a dotted number immediately
            # following a recognized product phrase; this avoids treating arbitrary
            # CVSS scores, ports, and dates as versions.
            tail = text[match.end():match.end() + 28]
            near = re.match(r"\s*(?:version\s*|v\s*)?(\d+(?:\.\d+){1,3})\b", tail, re.I)
            if near:
                versions.add(near.group(1).casefold())
            break

    return RetrievalSignals(
        cves=cves,
        apars=apars,
        errors=frozenset(errors),
        versions=frozenset(versions),
        products=frozenset(products),
    )


def _compare_exact(
    label: str,
    q_values: frozenset[str],
    d_values: frozenset[str],
    *,
    match_weight: float,
    mismatch_weight: float,
    hard_on_mismatch: bool,
) -> tuple[float, bool, list[str]]:
    if not q_values:
        return 0.0, False, []
    overlap = q_values & d_values
    if overlap:
        return match_weight, False, [f"{label}_MATCH"]
    if d_values:
        return -mismatch_weight, hard_on_mismatch, [f"{label}_MISMATCH"]
    return 0.0, False, []


def assess_constraints(question: str, document_text: str) -> ConstraintAssessment:
    q = extract_signals(question)
    d = extract_signals(document_text)
    score = 0.0
    hard = False
    reasons: list[str] = []

    for label, qv, dv, mw, mmw, hard_mm in (
        ("CVE", q.cves, d.cves, 4.0, 7.0, True),
        ("APAR", q.apars, d.apars, 4.0, 7.0, True),
        ("ERROR", q.errors, d.errors, 3.0, 5.0, True),
        ("VERSION", q.versions, d.versions, 1.5, 2.0, False),
        ("PRODUCT", q.products, d.products, 2.0, 3.0, False),
    ):
        delta, is_hard, why = _compare_exact(
            label, qv, dv,
            match_weight=mw,
            mismatch_weight=mmw,
            hard_on_mismatch=hard_mm,
        )
        score += delta
        hard = hard or is_hard
        reasons.extend(why)

    return ConstraintAssessment(score=score, hard_conflict=hard, reasons=tuple(reasons))


def get_chunk_text(chunk: dict) -> str:
    text = chunk.get("text") or chunk.get("content") or chunk.get("chunk_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"청크 텍스트가 없습니다: {list(chunk.keys())}")
    return text


def document_search_text(chunk: dict) -> str:
    from document_utils import chunk_filename

    parts = [chunk_filename(chunk)]
    metadata = chunk.get("metadata")
    for source in (chunk, metadata if isinstance(metadata, dict) else {}):
        for key in ("title", "product", "product_name", "version", "url"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    parts.append(get_chunk_text(chunk))
    return "\n".join(parts)


def rewrite_query(question: str) -> str:
    """Conservative deterministic rewrite.

    It never invents identifiers.  It keeps the most technical sentence(s),
    removes only obvious conversational boilerplate, and re-appends every
    high-value identifier found in the original question if needed.
    """
    original = " ".join((question or "").split())
    if not original:
        return original

    sentences = [s.strip() for s in re.split(r"(?<=[?.!])\s+|[\r\n]+", original) if s.strip()]
    if not sentences:
        return original

    signal = extract_signals(original)
    high_values = [*sorted(signal.cves), *sorted(signal.apars), *sorted(signal.errors), *sorted(signal.versions)]

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        s_signals = extract_signals(sentence)
        identifier_count = sum(bool(x) for x in (s_signals.cves, s_signals.apars, s_signals.errors, s_signals.versions))
        product_count = len(s_signals.products)
        cue = int(bool(TECHNICAL_CUE_RE.search(sentence)))
        score = identifier_count * 4 + product_count * 2 + cue
        cleaned = BOILERPLATE_RE.sub(" ", sentence)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
        if cleaned:
            scored.append((score, index, cleaned))

    if not scored:
        return original

    # Keep at most two strongest sentences; preserve original order in output.
    chosen = sorted(sorted(scored, key=lambda x: (-x[0], x[1]))[:2], key=lambda x: x[1])
    rewritten = " ".join(item[2] for item in chosen).strip()

    if not rewritten:
        return original

    for value in high_values:
        if value.casefold() not in rewritten.casefold():
            rewritten += " " + value

    # Do not produce an over-aggressive rewrite when the technical signal is too
    # weak.  In that case the original query is safer.
    if len(re.findall(r"\w+", rewritten)) < 4:
        return original
    return rewritten


def _dedupe_by_document(ranking: Iterable[int], chunks: list[dict]) -> list[int]:
    from document_utils import chunk_filename, filename_key

    seen: set[str] = set()
    out: list[int] = []
    for row_id in ranking:
        key = filename_key(chunk_filename(chunks[row_id]))
        if key in seen:
            continue
        seen.add(key)
        out.append(int(row_id))
    return out


def merge_hybrid_results(
    original_result,
    rewritten_result,
    chunks: list[dict],
    rewrite_weight: float = 0.7,
) -> tuple[list[int], dict[int, float]]:
    if not 0.0 <= rewrite_weight <= 2.0:
        raise ValueError("rewrite_weight must be between 0 and 2")

    original_ranking, original_scores = original_result[2], original_result[3]
    if rewritten_result is None:
        ranking = _dedupe_by_document(original_ranking, chunks)
        return ranking, {int(k): float(v) for k, v in original_scores.items()}

    rewritten_ranking, rewritten_scores = rewritten_result[2], rewritten_result[3]
    scores: dict[int, float] = {}
    for row_id in original_ranking:
        scores[int(row_id)] = scores.get(int(row_id), 0.0) + float(original_scores.get(row_id, 0.0))
    for row_id in rewritten_ranking:
        scores[int(row_id)] = scores.get(int(row_id), 0.0) + rewrite_weight * float(rewritten_scores.get(row_id, 0.0))

    ranking = sorted(scores, key=lambda rid: (-scores[rid], rid))
    ranking = _dedupe_by_document(ranking, chunks)
    return ranking, scores


def apply_constraint_filter(
    question: str,
    ranking: list[int],
    chunks: list[dict],
    *,
    strength: float,
) -> tuple[list[int], dict[int, ConstraintAssessment], dict[int, float]]:
    if strength < 0:
        raise ValueError("filter strength must be >= 0")
    if strength == 0:
        assessments = {rid: assess_constraints(question, document_search_text(chunks[rid])) for rid in ranking}
        return list(ranking), assessments, {rid: 1.0 / (10.0 + rank) for rank, rid in enumerate(ranking, 1)}

    assessments: dict[int, ConstraintAssessment] = {}
    adjusted: dict[int, float] = {}
    for rank, row_id in enumerate(ranking, 1):
        assessment = assess_constraints(question, document_search_text(chunks[row_id]))
        assessments[row_id] = assessment
        base = 1.0 / (10.0 + rank)
        signal_bonus = strength * 0.003 * assessment.score
        hard_penalty = strength * 0.03 if assessment.hard_conflict else 0.0
        adjusted[row_id] = base + signal_bonus - hard_penalty

    reranked = sorted(ranking, key=lambda rid: (-adjusted[rid], ranking.index(rid)))
    return reranked, assessments, adjusted


def score_reranker_candidates(
    reranker,
    questions: list[str],
    rankings: list[list[int]],
    chunks: list[dict],
    *,
    candidate_k: int,
    batch_size: int,
    metadata_input: bool = False,
) -> list[dict[int, float]]:
    import numpy as np

    if candidate_k < 1 or batch_size < 1:
        raise ValueError("candidate_k and batch_size must be >= 1")
    if len(questions) != len(rankings):
        raise ValueError("질문 수와 ranking 수가 다릅니다.")

    pairs = []
    positions: list[tuple[int, int]] = []
    for q_idx, (question, ranking) in enumerate(zip(questions, rankings)):
        for row_id in ranking[:candidate_k]:
            evidence = (document_search_text(chunks[row_id]) if metadata_input
                        else get_chunk_text(chunks[row_id]))
            pairs.append((question, evidence))
            positions.append((q_idx, row_id))

    if not pairs:
        raise ValueError("Reranker 후보가 없습니다.")

    raw = reranker.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    scores = np.asarray(raw, dtype=float).reshape(-1)
    if len(scores) != len(positions) or not np.isfinite(scores).all():
        raise RuntimeError("Reranker 점수 개수/값 오류")

    grouped: list[dict[int, float]] = [dict() for _ in questions]
    for (q_idx, row_id), score in zip(positions, scores):
        grouped[q_idx][int(row_id)] = float(score)
    return grouped


def _zscore_map(values: dict[int, float], ids: list[int]) -> dict[int, float]:
    selected = [values[rid] for rid in ids]
    if not selected:
        return {}
    mu = mean(selected)
    sigma = pstdev(selected)
    if sigma < 1e-8:
        return {rid: 0.0 for rid in ids}
    return {rid: (values[rid] - mu) / sigma for rid in ids}


def rerank_with_constraints(
    question: str,
    candidate_ranking: list[int],
    reranker_scores: dict[int, float],
    chunks: list[dict],
    *,
    candidate_k: int,
    filter_enabled: bool,
    filter_strength: float,
) -> list[dict]:
    subset = [rid for rid in candidate_ranking[:candidate_k] if rid in reranker_scores]
    if not subset:
        return []

    zscores = _zscore_map(reranker_scores, subset)
    results = []
    for original_rank, row_id in enumerate(subset):
        assessment = assess_constraints(question, document_search_text(chunks[row_id]))
        final_score = zscores[row_id]
        if filter_enabled:
            # Small calibrated semantic-independent correction.  Hard conflicts
            # are pushed below otherwise comparable candidates, but are never
            # deleted so recall remains observable during evaluation.
            final_score += filter_strength * 0.12 * assessment.score
            if assessment.hard_conflict:
                final_score -= filter_strength * 1.5
        results.append({
            "row_id": row_id,
            "score": float(final_score),
            "reranker_score": float(reranker_scores[row_id]),
            "original_rank": original_rank,
            "constraint_score": float(assessment.score),
            "hard_conflict": bool(assessment.hard_conflict),
            "constraint_reasons": list(assessment.reasons),
        })

    results.sort(key=lambda x: (-x["score"], x["original_rank"], x["row_id"]))
    return results


def rerank_stage3_light(question, candidate_ranking, reranker_scores, chunks,
                        prior_scores, *, candidate_k):
    """One MiniLM pass, calibrated source-RRF prior, conservative identifier score.

    All signals depend only on the query and candidates, never on ground-truth
    labels. Empty candidate features are neutral rather than penalized.
    """
    subset = [rid for rid in candidate_ranking[:candidate_k] if rid in reranker_scores]
    if not subset:
        return []
    priors = {rid: float(prior_scores.get(rid, 0.0)) for rid in subset}
    prior_z = _zscore_map(priors, subset)
    signals = extract_signals(question)
    has_exact_id = bool(signals.cves or signals.apars or signals.errors)
    results = []
    for rank, row_id in enumerate(subset):
        assessment = assess_constraints(question, document_search_text(chunks[row_id]))
        # MiniLM outputs unbounded logits, not probabilities.
        semantic = float(reranker_scores[row_id]) / 5.0
        source_prior = 0.2 * prior_z[row_id]
        # Do not over-penalize product/version ambiguity; explicit code matches
        # can be trusted more strongly than inferred product families.
        correction = (0.16 if has_exact_id else 0.06) * assessment.score
        hard_penalty = 0.25 if has_exact_id and assessment.hard_conflict else 0.0
        score = semantic + source_prior + correction - hard_penalty
        results.append({
            "row_id": row_id,
            "score": score,
            "reranker_score": float(reranker_scores[row_id]),
            "original_rank": rank,
            "constraint_score": float(assessment.score),
            "constraint_reasons": list(assessment.reasons),
            "hard_conflict": bool(assessment.hard_conflict),
        })
    results.sort(key=lambda x: (-x["score"], x["original_rank"], x["row_id"]))
    return results


def _stage3_evidence_index(chunks):
    from collections import defaultdict
    result = defaultdict(list)
    for row_id, chunk in enumerate(chunks):
        key = chunk.get("document_id") or str(chunk.get("filename") or row_id).casefold()
        result[str(key)].append((int(chunk.get("chunk_index", 0)), row_id))
    for group in result.values():
        group.sort()
    return result


def _stage3_expand_evidence(row_id, chunks, evidence_index, *, max_chars=2400):
    """Complete the selected chunk using its contiguous source-document text.

    First include the next chunk (answer sections often follow problem sections),
    then use remaining budget for the preceding chunk.  Overlap is eliminated
    through original character offsets.  Never mix different source documents.
    """
    original = chunks[row_id]
    text = get_chunk_text(original)
    if len(text) >= max_chars:
        return text[:max_chars]
    key = str(original.get("document_id") or str(original.get("filename") or row_id).casefold())
    group = evidence_index.get(key, [])
    index = next((i for i, (_, rid) in enumerate(group) if rid == row_id), None)
    if index is None:
        return text
    selected_start = int(original.get("start_char", 0))
    selected_end = int(original.get("end_char", selected_start + len(text)))
    if selected_end - selected_start != len(text):
        return text

    # Add content that starts after the selected chunk.  Limit to its actual
    # continuation, not a repeated overlap segment.
    end = selected_end
    for _, neighbor_id in group[index+1:index+3]:
        neighbor = chunks[neighbor_id]
        neighbor_start = int(neighbor.get("start_char", -1))
        neighbor_end = int(neighbor.get("end_char", -1))
        neighbor_text = get_chunk_text(neighbor)
        if neighbor_start > end or neighbor_end - neighbor_start != len(neighbor_text):
            break
        suffix = neighbor_text[max(0, end - neighbor_start):]
        remaining = max_chars - len(text)
        if remaining <= 0:
            break
        text += suffix[:remaining]
        end += min(len(suffix), remaining)
        if len(suffix) > remaining:
            break

    # Recover the preceding answer/title boundary only if room remains.
    if index > 0 and len(text) < max_chars:
        neighbor = chunks[group[index-1][1]]
        previous_start = int(neighbor.get("start_char", -1))
        previous_end = int(neighbor.get("end_char", -1))
        previous_text = get_chunk_text(neighbor)
        if (previous_start >= 0 and previous_end >= selected_start
                and previous_end - previous_start == len(previous_text)):
            prefix = previous_text[:max(0, selected_start - previous_start)]
            text = prefix[-(max_chars-len(text)):] + text
    return text


# STAGE3_EVIDENCE_V2

# STAGE3_LIGHT_RUNTIME_V1
def load_selected_config(path: Path | None = None) -> dict:
    config_path = Path(path) if path is not None else DEFAULT_SELECTED_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Stage2 선택 설정이 없습니다: {config_path}\n"
            "먼저 python IBMTechSupport/scripts/evaluate/rag_stage2.py optimize 를 실행하세요."
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    selected = config.get("selected") or {}
    required = {
        "reranker_enabled", "filter_enabled", "filter_strength", "reranker_candidate_k",
        "query_rewrite_enabled", "rewrite_weight",
    }
    missing = required - set(selected)
    if missing:
        raise ValueError(f"Stage2 선택 설정 누락: {sorted(missing)}")
    return config


def retrieve_all_stage2(
    questions: list[str],
    top_k: int,
    *,
    search_system=None,
    selected_config_path: Path | None = None,
    reranker=None,
    stage3_light: bool = False,
):
    """Stage2 baseline and optional lightweight Stage3 route."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    config = load_selected_config(selected_config_path)
    selected = config["selected"]
    runtime = config.get("runtime", {})
    batch_size = int(runtime.get("reranker_batch_size", 8))

    from dense_search import load_retriever
    from hybrid_search import load_bm25, search_all
    from reranker_search import load_reranker
    from document_utils import chunk_filename

    if search_system is None:
        embedding_model, faiss_index, chunks = load_retriever()
        bm25_index = load_bm25(chunks)
        search_system = (embedding_model, faiss_index, bm25_index, chunks)

    embedding_model, faiss_index, bm25_index, chunks = search_system
    reranker_enabled = bool(selected["reranker_enabled"])
    if reranker_enabled and reranker is None:
        reranker = load_reranker()

    start = time.perf_counter()
    original_results = search_all(embedding_model, faiss_index, bm25_index, questions,
        document_fusion=True, depth=100) if stage3_light else search_all(
        embedding_model, faiss_index, bm25_index, questions)

    rewritten_questions = [rewrite_query(q) for q in questions]
    rewrite_enabled = bool(selected["query_rewrite_enabled"])
    rewritten_results = None
    if rewrite_enabled and any(a != b for a, b in zip(questions, rewritten_questions)):
        rewritten_results = search_all(embedding_model, faiss_index, bm25_index, rewritten_questions,
            document_fusion=True, depth=100) if stage3_light else search_all(
            embedding_model, faiss_index, bm25_index, rewritten_questions)

    max_candidates = int(runtime.get("max_candidates", 50))
    if max_candidates < top_k:
        raise ValueError("runtime.max_candidates must be >= top_k")

    candidate_rankings: list[list[int]] = []
    prior_maps: list[dict[int, float]] = []
    for idx, question in enumerate(questions):
        rewrite_result = rewritten_results[idx] if rewritten_results is not None and rewritten_questions[idx] != question else None
        merged, prior_scores = merge_hybrid_results(
            original_results[idx], rewrite_result, chunks,
            rewrite_weight=float(selected["rewrite_weight"]),
        )
        # The offline optimizer tunes inside the same fixed pre-reranker pool.
        # Truncating before constraints keeps production candidate membership
        # identical to what was evaluated.
        merged = merged[:max_candidates]
        if bool(selected["filter_enabled"]):
            merged, _, _ = apply_constraint_filter(
                question, merged, chunks,
                strength=float(selected["filter_strength"]),
            )
        candidate_rankings.append(merged)
        prior_maps.append(prior_scores)

    candidate_k = int(selected["reranker_candidate_k"])
    if reranker_enabled:
        score_maps = score_reranker_candidates(
            reranker, questions, candidate_rankings, chunks,
            candidate_k=max(candidate_k, top_k), batch_size=batch_size,
            metadata_input=stage3_light,
        )
    else:
        score_maps = [None for _ in questions]

    evidence_index = _stage3_evidence_index(chunks) if stage3_light else None
    docs_by_question = []
    for question, ranking, score_map, prior_map in zip(
            questions, candidate_rankings, score_maps, prior_maps):
        if reranker_enabled:
            if stage3_light:
                ranked = rerank_stage3_light(
                    question, ranking, score_map, chunks, prior_map,
                    candidate_k=max(candidate_k, top_k),
                )
            else:
                ranked = rerank_with_constraints(
                    question, ranking, score_map, chunks,
                    candidate_k=max(candidate_k, top_k),
                    filter_enabled=bool(selected["filter_enabled"]),
                    filter_strength=float(selected["filter_strength"]),
                )
        else:
            ranked = []
            for rank, row_id in enumerate(ranking[:top_k], 1):
                assessment = assess_constraints(question, document_search_text(chunks[row_id]))
                ranked.append({
                    "row_id": row_id,
                    "score": 1.0 / (10.0 + rank),
                    "reranker_score": None,
                    "constraint_score": assessment.score,
                    "constraint_reasons": list(assessment.reasons),
                })

        documents = []
        for item in ranked[:top_k]:
            chunk = chunks[item["row_id"]]
            document = {
                "row_id": item["row_id"],
                "chunk_id": chunk.get("chunk_id"),
                "filename": chunk_filename(chunk),
                "text": (_stage3_expand_evidence(item["row_id"], chunks, evidence_index,
                    max_chars=(2400 if len(documents) < 5 else 1400))
                    if stage3_light else get_chunk_text(chunk)),
                "score": float(item["score"]),
                "constraint_score": float(item["constraint_score"]),
                "constraint_reasons": item["constraint_reasons"],
            }
            if item.get("reranker_score") is not None:
                document["reranker_score"] = float(item["reranker_score"])
            documents.append(document)
        docs_by_question.append(documents)

    return docs_by_question, time.perf_counter() - start
