"""Paired Strict RAG experiment on val_106. Run with python -m evaluate.strict_recheck."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from shutil import copyfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/val_106.json"
V3 = ROOT / "configs/rag/v3.yaml"
OUTPUT = ROOT / "evaluate_results/106/strict_top5_recheck"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def settings():
    config = yaml.safe_load(V3.read_text(encoding="utf-8"))
    config["retrieval"]["method"] = "stage3"
    if (ROOT / config["dataset"]["path"] != DATA or config["retrieval"]["top_k"] != 5
            or config["guardrails"]["retry_top_k"] != 10 or not config["guardrails"]["enabled"]
            or config["inference"]["max_new_tokens"] != 160):
        raise ValueError("Expected the diagnosed v3 Strict configuration")
    config["guardrails"]["judge_mode"] = "strict"
    records = read(DATA)
    if (len({r["id"] for r in records}), len(records),
            sum(r["is_impossible"] is False for r in records),
            sum(r["is_impossible"] is True for r in records)) != (106, 106, 77, 29):
        raise ValueError("Expected 106 unique validation cases (77/29)")
    return records, config


def prepare(directory, records, config, method="verbatim"):
    from huggingface_hub import snapshot_download
    import reranker_search
    from rag_pipeline import load_search_system, retrieve_all
    from stage2_retrieval import get_chunk_text

    directory.mkdir(parents=True, exist_ok=False)
    reranker_search.RERANKER_MODEL = snapshot_download(reranker_search.RERANKER_MODEL, local_files_only=True)
    system = load_search_system()
    groups, seconds = retrieve_all([r["question"] for r in records], "stage3", 10, search_system=system)
    rows = []
    for record, docs in zip(records, groups, strict=True):
        if len(docs) != 10 or len({d["chunk_id"] for d in docs}) != 10:
            raise ValueError("Expected ten distinct retrieved chunks")
        for doc in docs:
            chunk = system[3][doc["row_id"]]
            if chunk["chunk_id"] != doc["chunk_id"]:
                raise ValueError("Corpus chunk ID mismatch")
            doc["text"] = get_chunk_text(chunk)
        rows.append({"id": record["id"], "question": record["question"], "documents": docs})
    cache = directory / "retrieval_cache.json"
    save(cache, rows)
    save(directory / "manifest.json", {
        "dataset": str(DATA.relative_to(ROOT)), "dataset_sha256": digest(DATA),
        "cache_sha256": digest(cache), "retrieval_seconds": seconds,
        "evidence": "original representative chunk (diagnose_strict_refusals)",
        "config": config, "recheck_method": method, "reranker": Path(reranker_search.RERANKER_MODEL).name,
        "retrieval_configs": {n: digest(ROOT / "configs/rag" / n) for n in
                              ("retrieval.yaml", "retrieval_stage2_selected.yaml")},
    })


def reuse_strict(source, directory, config, method):
    manifest = read(source / "manifest.json")
    summary = read(source / "comparison.json")
    cache_sha = digest(source / "retrieval_cache.json")
    strict_sha = digest(source / "strict.jsonl")
    if (manifest["config"] != config or manifest["dataset_sha256"] != digest(DATA)
            or manifest["cache_sha256"] != cache_sha or summary["cache_sha256"] != cache_sha
            or summary["prediction_sha256"]["strict"] != strict_sha):
        raise ValueError("Saved Strict baseline or cache does not match its measured run")
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("retrieval_cache.json", "strict.jsonl"):
        copyfile(source / name, directory / name)
    save(directory / "manifest.json", {**manifest, "recheck_method": method,
         "baseline_run": source.name, "baseline_sha256": strict_sha, "replay_strict_decisions": True})


def predict(directory, records, config, variant):
    from load_base_model import load_model
    from rag_guardrails_per_document import choose_guardrail_runner
    from rag_pipeline import generate_rag_answer

    cache = directory / "retrieval_cache.json"
    manifest = read(directory / "manifest.json")
    if (digest(cache) != manifest["cache_sha256"] or digest(DATA) != manifest["dataset_sha256"]
            or config != manifest["config"]):
        raise ValueError("Cache, dataset or configuration changed")
    groups = read(cache)
    if [(r["id"], r["question"]) for r in groups] != [(r["id"], r["question"]) for r in records]:
        raise ValueError("Cache alignment mismatch")
    run = choose_guardrail_runner({**config, "guardrails": {
        **config["guardrails"], "top5_recheck": variant == "recheck",
        "top5_recheck_method": manifest.get("recheck_method", "verbatim")}})
    baseline = None
    if variant == "recheck" and manifest.get("replay_strict_decisions"):
        if digest(directory / "strict.jsonl") != manifest["baseline_sha256"]:
            raise ValueError("Strict predictions changed")
        baseline = [json.loads(line) for line in (directory / "strict.jsonl").read_text("utf8").splitlines()]
        if [row["id"] for row in groups] != [row["id"] for row in baseline]:
            raise ValueError("Strict predictions and retrieval cache differ in order")
    model, tokenizer, generation_config = load_model(config)
    with (directory / f"{variant}.jsonl").open("x", encoding="utf-8") as file:
        for index, row in enumerate(groups, 1):
            kwargs = {}
            if baseline is not None:
                old = baseline[index - 1]
                path = old["judge_history"]
                if [(h["stage"], h["document_count"], h["decision"]) for h in path] != [
                        ("top_k", 5, "NO"), ("expanded_k", 10, "NO")]:
                    file.write(json.dumps(old, ensure_ascii=False) + "\n")
                    continue
                history = iter(path)
                def saved_judge(_model, _tokenizer, question, docs):
                    hit = next(history)
                    if question != row["question"] or docs != row["documents"][:hit["document_count"]]:
                        raise ValueError("Attempted to reuse a judgment with different inputs")
                    return {k: hit[k] for k in ("raw", "decision", "seconds")}
                kwargs["judge_fn"] = saved_judge
            result = run(question=row["question"], retrieved_documents=row["documents"],
                         model=model, tokenizer=tokenizer, generation_config=generation_config,
                         prompt_config=config["prompt"], generate_answer=generate_rag_answer,
                         top_k=5, retry_top_k=10, **kwargs)
            result["used_chunk_ids"] = [d["chunk_id"] for d in result.pop("used_documents")]
            file.write(json.dumps({"id": row["id"], **result}, ensure_ascii=False) + "\n")
            file.flush()
            print(f"{variant} {index}/{len(groups)} {result['status']}", flush=True)


def transition(before, after):
    if before == "REFUSED" and after == "CORRECT":
        return "refused_to_correct"
    if before == "REFUSED" and after not in ("CORRECT", "REFUSED"):
        return "refused_to_noncorrect"
    if before == "CORRECT" and after != "CORRECT":
        return "correct_to_refused" if after == "REFUSED" else "correct_to_noncorrect"
    return None


def score(directory, records, config):
    from evaluate import evaluate_stage2_quality as quality

    manifest = read(directory / "manifest.json")
    if (digest(directory / "retrieval_cache.json") != manifest["cache_sha256"] or
            digest(DATA) != manifest["dataset_sha256"] or manifest["config"] != config):
        raise ValueError("Evaluation inputs changed")
    variants = ("strict", "recheck")
    predictions = {}
    for variant in variants:
        rows = quality.load_jsonl(directory / f"{variant}.jsonl")
        if [r["id"] for r in rows] != [r["id"] for r in records]:
            raise ValueError(f"Incomplete or misaligned predictions: {variant}")
        predictions[variant] = rows
    evaluator = quality.Evaluator(ROOT / config["model"]["path"])
    cases = []
    counts = {v: Counter(judge_correct=0, false_refusals=0, impossible_responses=0) for v in variants}
    transitions = Counter(refused_to_correct=0, refused_to_noncorrect=0, correct_to_noncorrect=0,
                          correct_to_refused=0)
    impossible_changes = Counter(refused_to_response=0, response_to_refused=0)
    rechecks = Counter(attempted=0, passed=0, generator_refused=0, verifier_refused=0)
    reasons = Counter()
    changed_judge_paths = []
    changed_outside_recheck = []
    for i, record in enumerate(records):
        case = {"id": record["id"], "question": record["question"], "is_impossible": record["is_impossible"]}
        judged = {}  # Identical candidate on this question gets the same correctness verdict.
        for variant in variants:
            pred = predictions[variant][i]
            answer = pred["answer"]
            refused = answer == quality.ABSTAIN
            if record["is_impossible"]:
                verdict = quality.fixed("NOT_APPLICABLE", "Impossible question")
                counts[variant]["impossible_responses"] += not refused
            elif refused:
                verdict = quality.fixed("REFUSED", "Answerable question")
                counts[variant]["false_refusals"] += 1
            else:
                if answer not in judged:
                    judged[answer] = evaluator.evaluate(quality.CORRECT_SYSTEM, {
                        "question": record["question"], "candidate_answer": answer,
                        "reference_answer": record["answer"], "reference_documents": quality.gold_excerpts(record)
                    }, quality.CORRECT_LABELS)
                verdict = judged[answer]
                counts[variant]["judge_correct"] += verdict["verdict"] == "CORRECT"
            case[variant] = {"prediction": answer, "status": pred["status"], "correctness": verdict}
        change = transition(case["strict"]["correctness"]["verdict"], case["recheck"]["correctness"]["verdict"])
        case["transition"] = change
        if change:
            transitions[change] += 1
        if record["is_impossible"]:
            before = case["strict"]["prediction"] == quality.ABSTAIN
            after = case["recheck"]["prediction"] == quality.ABSTAIN
            if before != after:
                change = "refused_to_response" if before else "response_to_refused"
                impossible_changes[change] += 1
                case["transition"] = "impossible_" + change
        reviews = [j for j in predictions["recheck"][i]["judge_history"] if j["stage"] == "top5_recheck"]
        paths = [[(j["stage"], j["document_count"], j["decision"], j["raw"]) for j in
                  predictions[v][i]["judge_history"] if j["stage"] != "top5_recheck"] for v in variants]
        if paths[0] != paths[1]:
            changed_judge_paths.append(record["id"])
        if not reviews and case["strict"] != case["recheck"]:
            changed_outside_recheck.append(record["id"])
        case["recheck_reason"] = reviews[0]["reason"] if reviews else None
        case["refusal_after_pass"] = None
        if reviews:
            rechecks["attempted"] += 1
            reasons[reviews[0]["reason"]] += 1
            if reviews[0]["decision"] == "YES":
                rechecks["passed"] += 1
                status = case["recheck"]["status"]
                if status == "GENERATOR_ABSTAINED":
                    case["refusal_after_pass"] = "generator"
                    rechecks["generator_refused"] += 1
                elif status in ("ANSWER_NO", "ANSWER_UNCERTAIN"):
                    case["refusal_after_pass"] = "verifier"
                    rechecks["verifier_refused"] += 1
        cases.append(case)
    with (directory / "cases.jsonl").open("x", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")
    passed = (counts["recheck"]["judge_correct"] > counts["strict"]["judge_correct"] and
              counts["recheck"]["impossible_responses"] <= counts["strict"]["impossible_responses"])
    summary = {"dataset": "val_106.json", "answerable": sum(not r["is_impossible"] for r in records),
               "impossible": sum(r["is_impossible"] for r in records),
               "cache_sha256": manifest["cache_sha256"], "counts": counts, "transitions": transitions,
               "recheck_method": manifest.get("recheck_method", "verbatim"),
               "impossible_transitions": impossible_changes,
               "noncorrect_definition": "Emitted answer with PARTIAL, INCORRECT or UNCERTAIN correctness",
               "rechecks": rechecks, "review_reasons": reasons, "success": passed,
               "changed_original_judge_paths": changed_judge_paths,
               "changed_cases_without_recheck": changed_outside_recheck,
               "correctness_judge": "evaluate_stage2_quality.CORRECT_SYSTEM, FP16 base Qwen",
               "versions": {n: importlib.metadata.version(n) for n in
                            ("torch", "transformers", "faiss-cpu", "sentence-transformers", "bm25s")},
               "source_sha256": {n: digest(ROOT / "scripts" / n) for n in
                                 ("rag_guardrails.py", "rag_guardrails_per_document.py", "rag_top5_recheck.py",
                                  "rag_passage_recheck.py", "rag_section_recheck.py", "rag_pipeline.py",
                                  "stage2_retrieval.py", "evaluate/strict_recheck.py",
                                  "evaluate/evaluate_stage2_quality.py")},
               "prediction_sha256": {v: digest(directory / f"{v}.jsonl") for v in variants}}
    save(directory / "comparison.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "prepare", "strict", "recheck", "score"), default="all", nargs="?")
    parser.add_argument("--run", default=None, help="New run name, or existing run for a later stage")
    parser.add_argument("--method", choices=("verbatim", "passages", "sections", "focused"),
                        help="Review method to freeze for a new run")
    parser.add_argument("--baseline-run", help="Reuse a completed Strict arm and exact cache for an all run")
    args = parser.parse_args()
    if args.method and args.action not in ("all", "prepare"):
        parser.error("--method is fixed by the run manifest after prepare")
    if args.action not in ("all", "prepare") and not args.run:
        parser.error("--run is required for a later stage")
    name = args.run or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    if args.baseline_run and (args.action != "all" or Path(args.baseline_run).name != args.baseline_run
                              or args.baseline_run in (".", "..")):
        parser.error("--baseline-run requires all and a run folder name")
    if Path(name).name != name or name in (".", ".."):
        parser.error("Expected a run folder name")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    records, config = settings()
    directory = OUTPUT / name
    print(directory, flush=True)
    actions = ("prepare", "strict", "recheck", "score") if args.action == "all" else (args.action,)
    if args.baseline_run:
        reuse_strict(OUTPUT / args.baseline_run, directory, config, args.method or "verbatim")
        actions = ("recheck", "score")
    for action in actions:
        if action == "prepare":
            prepare(directory, records, config, args.method or "verbatim")
        elif action == "score":
            score(directory, records, config)
        else:
            predict(directory, records, config, action)
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
