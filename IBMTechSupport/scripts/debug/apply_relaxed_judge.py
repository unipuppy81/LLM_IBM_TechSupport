"""Enable v3_relaxed config without changing the existing v3 behavior.

Patches only the two existing RAG call sites to choose strict or relaxed based
on `guardrails.judge_mode`.  Existing v3.yaml defaults to strict.
Backups are written as *.bak_relaxed; no dataset or result files are touched.
Run from repository root:
    python IBMTechSupport/scripts/debug/apply_relaxed_judge.py
"""

import argparse
import shutil
from pathlib import Path


def replace_once(source, needle, replacement, label):
    n = source.count(needle)
    if n != 1:
        raise RuntimeError(f"패치 대상 불일치: {label} (matches={n}). 코드를 확인하세요.")
    return source.replace(needle, replacement, 1)


def patch_evaluate(source):
    source = replace_once(
        source,
        "from rag_guardrails import answer_with_guardrails\n",
        "from rag_guardrails import answer_with_guardrails\n"
        "from rag_guardrails_relaxed import answer_with_relaxed_guardrails\n",
        "evaluate import",
    )
    source = replace_once(
        source,
        '    guardrails = config.get("guardrails", {})\n',
        '    guardrails = config.get("guardrails", {})\n'
        '    judge_mode = guardrails.get("judge_mode", "strict")\n'
        '    if judge_mode not in ("strict", "relaxed"):\n'
        '        raise ValueError("guardrails.judge_mode must be strict or relaxed")\n'
        '    guardrail_runner = (answer_with_relaxed_guardrails if judge_mode == "relaxed"\n'
        '                        else answer_with_guardrails)\n',
        "evaluate mode",
    )
    source = replace_once(
        source,
        "            guarded_result = answer_with_guardrails(\n",
        "            guarded_result = guardrail_runner(\n",
        "evaluate call",
    )
    return source


def patch_pipeline(source):
    source = replace_once(
        source,
        "from rag_guardrails import answer_with_guardrails\n",
        "from rag_guardrails import answer_with_guardrails\n"
        "from rag_guardrails_relaxed import answer_with_relaxed_guardrails\n",
        "pipeline import",
    )
    source = replace_once(
        source,
        '        guarded_result = answer_with_guardrails(\n',
        '        judge_mode = config.get("guardrails", {}).get("judge_mode", "strict")\n'
        '        if judge_mode not in ("strict", "relaxed"):\n'
        '            raise ValueError("guardrails.judge_mode must be strict or relaxed")\n'
        '        guardrail_runner = (answer_with_relaxed_guardrails if judge_mode == "relaxed"\n'
        '                            else answer_with_guardrails)\n'
        '        guarded_result = guardrail_runner(\n',
        "pipeline call",
    )
    source = replace_once(
        source,
        '    parser.add_argument("--top-k", type=int, default=TOP_K)\n',
        '    parser.add_argument("--top-k", type=int, default=TOP_K)\n'
        '    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/rag/v3.yaml")\n',
        "pipeline parser",
    )
    source = replace_once(
        source,
        '    run_rag(question=args.question, method=args.retriever, top_k=args.top_k)\n',
        '    run_rag(question=args.question, method=args.retriever, top_k=args.top_k, config_path=args.config)\n',
        "pipeline cli",
    )
    return source


def run(root):
    root = root.resolve()
    scripts = root / "IBMTechSupport/scripts"
    files = {
        scripts / "evaluate/evaluate_rag.py": patch_evaluate,
        scripts / "rag_pipeline.py": patch_pipeline,
    }
    needed = [
        scripts / "rag_guardrails.py",
        scripts / "rag_guardrails_relaxed.py",
        root / "IBMTechSupport/configs/rag/v3.yaml",
        root / "IBMTechSupport/configs/rag/v3_relaxed.yaml",
    ]
    for file in [*files, *needed]:
        if not file.is_file():
            raise FileNotFoundError(f"파일 없음: {file}")
    replacements = {}
    for path, patcher in files.items():
        old = path.read_text(encoding="utf-8")
        if "from rag_guardrails_relaxed import answer_with_relaxed_guardrails" in old:
            raise RuntimeError(f"이미 완화 Judge 패치가 적용됨: {path}")
        new = patcher(old)
        compile(new, str(path), "exec")
        replacements[path] = new
    for path in files:
        backup = path.with_name(path.name + ".bak_relaxed")
        if backup.exists():
            raise FileExistsError(f"백업이 이미 존재: {backup}")
    for path in files:
        shutil.copy2(path, path.with_name(path.name + ".bak_relaxed"))
    for path, content in replacements.items():
        path.write_text(content, encoding="utf-8")
        print("패치 완료:", path)
    print("기존 v3는 judge_mode 기본값 strict로 유지됩니다.")
    print("완화 v3 설정: IBMTechSupport/configs/rag/v3_relaxed.yaml")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    run(args.root)


if __name__ == "__main__":
    main()
