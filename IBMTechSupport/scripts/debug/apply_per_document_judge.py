"""Install v4 per-document Judge without overwriting existing v3 or results.

Run from repository root:
  python IBMTechSupport/scripts/debug/apply_per_document_judge.py

Changes: only two call sites in evaluate_rag.py and rag_pipeline.py.
New modules/config are included in the ZIP at their final script locations.
Backups: *.bak_per_document
"""
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "IBMTechSupport" / "scripts"
IMPORT_LINE = "from rag_guardrails_per_document import choose_guardrail_runner\n"


def replace_one(source, old, new, name):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: 예상된 코드가 없거나 중복입니다 (found={count})")
    return source.replace(old, new, 1)


def remove_old_relaxed_switch(source, target):
    # A previously-installed optional relaxed runner rejects new judge modes.
    if target == "evaluate":
        block = (
            '    judge_mode = guardrails.get("judge_mode", "strict")\n'
            '    if judge_mode not in ("strict", "relaxed"):\n'
            '        raise ValueError("guardrails.judge_mode must be strict or relaxed")\n'
            '    guardrail_runner = (answer_with_relaxed_guardrails if judge_mode == "relaxed"\n'
            '                        else answer_with_guardrails)\n'
        )
    else:
        block = (
            '        judge_mode = config.get("guardrails", {}).get("judge_mode", "strict")\n'
            '        if judge_mode not in ("strict", "relaxed"):\n'
            '            raise ValueError("guardrails.judge_mode must be strict or relaxed")\n'
            '        guardrail_runner = (answer_with_relaxed_guardrails if judge_mode == "relaxed"\n'
            '                            else answer_with_guardrails)\n'
        )
    if "guardrail_runner = (" in source:
        source = replace_one(source, block, "", f"{target}: old relaxed switch")
    return source


def install_import(source, target):
    if IMPORT_LINE in source:
        raise RuntimeError(f"{target}: already patched; not writing twice")
    return replace_one(
        source,
        "from rag_guardrails import answer_with_guardrails\n",
        "from rag_guardrails import answer_with_guardrails\n" + IMPORT_LINE,
        f"{target}: import",
    )


def patch_evaluate(source):
    source = install_import(source, "evaluate")
    source = remove_old_relaxed_switch(source, "evaluate")
    old_calls = (
        "            guarded_result = answer_with_guardrails(\n",
        "            guarded_result = guardrail_runner(\n",
    )
    matches = [old for old in old_calls if old in source]
    if len(matches) != 1:
        raise RuntimeError("evaluate: cannot identify a unique v3 guarded call")
    return replace_one(
        source, matches[0],
        "            guarded_result = choose_guardrail_runner(config)(\n",
        "evaluate: guarded call",
    )


def patch_pipeline(source):
    source = install_import(source, "pipeline")
    source = remove_old_relaxed_switch(source, "pipeline")
    old_calls = (
        "        guarded_result = answer_with_guardrails(\n",
        "        guarded_result = guardrail_runner(\n",
    )
    matches = [old for old in old_calls if old in source]
    if len(matches) != 1:
        raise RuntimeError("pipeline: cannot identify a unique v3 guarded call")
    source = replace_one(
        source, matches[0],
        "        guarded_result = choose_guardrail_runner(config)(\n",
        "pipeline: guarded call",
    )

    # Existing v3 CLI may not have a --config option. Enable it if missing.
    if 'parser.add_argument("--config"' not in source:
        source = replace_one(
            source,
            '    parser.add_argument("--top-k", type=int, default=TOP_K)\n',
            '    parser.add_argument("--top-k", type=int, default=TOP_K)\n'
            '    parser.add_argument("--config", type=Path, '
            'default=PROJECT_DIR / "configs/rag/v3.yaml")\n',
            "pipeline: --config option",
        )
        source = replace_one(
            source,
            '    run_rag(question=args.question, method=args.retriever, top_k=args.top_k)\n',
            '    run_rag(question=args.question, method=args.retriever, '
            'top_k=args.top_k, config_path=args.config)\n',
            "pipeline: config forwarding",
        )
    return source


def main():
    root_files = (
        SCRIPTS / "rag_guardrails.py",
        SCRIPTS / "rag_guardrails_relaxed.py",
        SCRIPTS / "rag_guardrails_per_document.py",
        ROOT / "IBMTechSupport/configs/rag/v4.yaml",
    )
    for file in root_files:
        if not file.is_file():
            raise FileNotFoundError(f"필수 파일 없음: {file}")

    targets = (
        (SCRIPTS / "evaluate/evaluate_rag.py", patch_evaluate),
        (SCRIPTS / "rag_pipeline.py", patch_pipeline),
    )
    for file, _ in targets:
        if not file.is_file():
            raise FileNotFoundError(f"기존 v3 파일 없음: {file}")

    # Compute, validate and check all backups BEFORE touching a single file.
    writes = []
    for path, patcher in targets:
        original = path.read_text(encoding="utf-8-sig")
        changed = patcher(original)
        compile(changed, str(path), "exec")
        backup = path.with_name(path.name + ".bak_per_document")
        if backup.exists():
            raise FileExistsError(f"백업이 이미 존재합니다: {backup}")
        writes.append((path, backup, changed))

    for file in root_files:
        if file.suffix == ".py":
            compile(file.read_text(encoding="utf-8-sig"), str(file), "exec")

    for path, backup, _ in writes:
        shutil.copy2(path, backup)
    for path, _, changed in writes:
        path.write_text(changed, encoding="utf-8")
        print("적용:", path)
    print("기존 파일 백업: *.bak_per_document")
    print("v3는 strict, v3_relaxed는 relaxed, v4는 per_document 모드를 사용합니다.")
    print("평가 결과: IBMTechSupport/evaluate_results/106/rag/v4/")


if __name__ == "__main__":
    main()
