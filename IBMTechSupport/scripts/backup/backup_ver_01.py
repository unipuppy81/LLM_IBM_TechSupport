"""Backup the pre-token-chunking IBMTechSupport retrieval experiment.

Run from the LLM_IBM_TechSupport repository root:
    python backup_ver_01.py

This script never overwrites an existing ver_01 backup.
"""

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml


EXPECTED = {
    "chunk_mode": "char",
    "chunk_size": 1250,
    "chunk_overlap": 150,
    "dense_model": "BAAI/bge-small-en-v1.5",
    "dense_max_seq_length": 512,
}

BASELINE_METRICS = """[검증 데이터 기준: 전체 135문항, 정답 존재 90문항]
Dense:    Hit@5 = 54.44%, MRR@10 = 0.4491, nDCG@5 = 0.4650
Hybrid:   Hit@5 = 60.00%, MRR@10 = 0.4800, nDCG@5 = 0.5003
Reranker: Hit@5 = 62.22%, MRR@10 = 0.4947, nDCG@5 = 0.5201

위 수치는 대화에서 사용자가 기록한 'BGE Small / 512, 문자 기반 1250/150' 평가 결과입니다.
백업된 results/retrieval_comparison.json의 구조와 값은 별도로 보존됩니다.
"""


def die(message):
    raise RuntimeError(message)


def count_and_hash(path):
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                lines += 1
    return lines, digest.hexdigest()


def all_files(root):
    return sorted(p for p in root.rglob("*") if p.is_file())


def main():
    repo = Path.cwd().resolve()
    project = repo / "IBMTechSupport"
    if not project.is_dir():
        die("프로젝트 루트(C:\\Projects\\LLM_IBM_TechSupport)에서 실행해야 합니다.")

    config_path = project / "configs" / "rag" / "retrieval.yaml"
    if not config_path.is_file():
        die(f"설정 파일이 없습니다: {config_path}")

    yaml_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(yaml_text)
    chunk = config["chunking"]
    dense = config["dense"]
    observed = {
        "chunk_mode": chunk.get("mode", "char"),
        "chunk_size": chunk["size"],
        "chunk_overlap": chunk["overlap"],
        "dense_model": dense["model"],
        "dense_max_seq_length": dense["max_seq_length"],
    }
    mismatches = [
        f"{key}: 현재={observed[key]!r}, 기대={value!r}"
        for key, value in EXPECTED.items() if observed[key] != value
    ]
    if mismatches:
        die(
            "ver_01 기준 설정과 현재 YAML이 다릅니다.\n"
            + "\n".join(mismatches)
            + "\n기존 문자 기반 1250/150, BGE Small 512 설정으로 되돌린 뒤 "
            "청크/인덱스를 변경하기 전에 백업하세요."
        )

    model_dir = "models/" + dense["model"].split("/")[-1]

    # 이 파일들은 토큰 Chunking 및 재색인 시 덮어써지거나,
    # 저장된 결과를 복원/재현하는 데 필요한 데이터입니다.
    required = [
        "data/raw/corpus.zip",
        "data/processed/corpus_chunks.jsonl",
        "data/processed/corpus_embeddings.npy",
        "data/processed/corpus_faiss.index",
        "data/processed/bm25_index",
        model_dir,
        "configs/rag/retrieval.yaml",
        "results/retrieval_comparison.json",
        "scripts",
    ]
    optional = [
        "docs/retrieval_experiments.txt",
    ]

    missing = [rel for rel in required if not (project / rel).exists()]
    if missing:
        die("필수 백업 대상이 없습니다:\n- " + "\n- ".join(missing))

    # 현재 임베딩, FAISS와 청크 개수가 일치하는지 확인합니다.
    import faiss
    import numpy as np

    chunk_path = project / "data/processed/corpus_chunks.jsonl"
    embedding_path = project / "data/processed/corpus_embeddings.npy"
    index_path = project / "data/processed/corpus_faiss.index"
    metrics_path = project / "results/retrieval_comparison.json"

    chunk_count, corpus_hash = count_and_hash(chunk_path)
    vectors = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    faiss_index = faiss.read_index(str(index_path))
    if len(vectors.shape) != 2 or vectors.shape != (chunk_count, 384):
        die(f"청크·임베딩 불일치: 청크={chunk_count}, 벡터={vectors.shape}, 기대 차원=384")
    if (faiss_index.ntotal, faiss_index.d) != (chunk_count, 384):
        die(f"FAISS·임베딩 불일치: FAISS={faiss_index.ntotal}x{faiss_index.d}")
    del faiss_index, vectors

    with metrics_path.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)
    embedding_meta = metrics.get("embedding_build", {})
    if embedding_meta:
        for key, expected_value in (
            ("model", dense["model"].split("/")[-1]),
            ("max_sequence_length", 512),
            ("total_chunks", chunk_count),
            ("corpus_sha256", corpus_hash),
        ):
            if key in embedding_meta and embedding_meta[key] != expected_value:
                die(
                    f"임베딩 빌드 기록 불일치: {key}="
                    f"{embedding_meta[key]!r}, 실제/기대={expected_value!r}. "
                    "백업 전 현재 데이터와 설정을 확인하세요."
                )

    backup_root = project / "backup"
    final = backup_root / "ver_01"
    staging = backup_root / ".ver_01.incomplete"

    if final.exists():
        die(f"기존 백업을 보호하기 위해 중단: {final} 이미 존재합니다.")
    if staging.exists():
        die(f"미완료 백업 폴더가 존재합니다: {staging} (확인 후 수동 정리 필요)")

    paths = required + [rel for rel in optional if (project / rel).exists()]
    source_files = []
    for rel in paths:
        src = project / rel
        if src.is_dir():
            source_files.extend(all_files(src))
        else:
            source_files.append(src)
    total_bytes = sum(p.stat().st_size for p in source_files)
    free_bytes = shutil.disk_usage(str(project)).free
    if free_bytes < total_bytes + 128 * 1024 * 1024:
        die(
            f"디스크 공간 부족: 필요한 복사 용량 약 {total_bytes / (1024**3):.2f} GiB, "
            f"남은 공간 약 {free_bytes / (1024**3):.2f} GiB"
        )

    print("[원본 확인 완료]")
    print(f"Chunking: {chunk.get('mode','char')} / {chunk['size']} / {chunk['overlap']}")
    print(f"Dense: {dense['model']} / {dense['max_seq_length']} tokens")
    print(f"Chunks / Embeddings / FAISS: {chunk_count:,} (384차원)")
    print(f"Backup size: {total_bytes / (1024**3):.2f} GiB")
    print(f"Destination: {final}\n", flush=True)

    backup_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        for rel in paths:
            src = project / rel
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            print("Copy:", rel, flush=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)

        # 파일 목록과 크기를 확인합니다. (대용량 파일은 재해시하지 않음)
        manifest_lines = []
        for rel in paths:
            src = project / rel
            if src.is_dir():
                for file in all_files(staging / rel):
                    origin = project / file.relative_to(staging)
                    if file.stat().st_size != origin.stat().st_size:
                        die(f"백업 파일 크기 불일치: {file}")
                    manifest_lines.append(f"{file.relative_to(staging)}\t{file.stat().st_size}")
            else:
                dst = staging / rel
                if dst.stat().st_size != src.stat().st_size:
                    die(f"백업 파일 크기 불일치: {dst}")
                manifest_lines.append(f"{rel}\t{dst.stat().st_size}")

        (staging / "file_manifest.txt").write_text(
            "백업 파일 목록 및 크기(bytes)\n\n" + "\n".join(manifest_lines) + "\n",
            encoding="utf-8-sig",
        )

        info = (
            "[ver_01 - 토큰 기반 Chunking 적용 전 백업]\n"
            f"백업 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"원본 프로젝트: {repo}\n"
            "\n[검증 결과]\n"
            f"전체 청크 수: {chunk_count:,}\n"
            "임베딩 차원: 384\n"
            f"Corpus SHA256: {corpus_hash}\n"
            "청크 수, 임베딩 행 개수, FAISS 벡터 수 일치 확인 완료\n"
            "\n[실험 평가 기록]\n"
            + BASELINE_METRICS
            + "\n[설정 전체: configs/rag/retrieval.yaml 원문]\n"
            + yaml_text.rstrip() + "\n"
            "\n[저장 자료 설명]\n"
            "data/raw/corpus.zip : 입력 원본 자료\n"
            "data/processed/corpus_chunks.jsonl : 기존 문자 기반 분할 결과\n"
            "data/processed/corpus_embeddings.npy : 기존 BGE Small 512 임베딩\n"
            "data/processed/corpus_faiss.index : 기존 FAISS 인덱스\n"
            "data/processed/bm25_index/ : 기존 BM25 인덱스 및 어휘\n"
            f"{model_dir}/ : 임베딩 모델 로컬 파일\n"
            "configs/rag/retrieval.yaml : 실행 설정\n"
            "results/retrieval_comparison.json : 저장된 결과/임베딩 빌드 기록\n"
            "scripts/ : 재현에 필요한 당시 Python 코드\n"
            "file_manifest.txt : 포함된 파일 및 파일 크기\n"
            "\n[주의]\n"
            "GitHub에 backup/의 대용량 데이터를 업로드하지 마세요.\n"
            "기존 인덱스를 재사용하려면 관련 파일과 YAML을 함께 복원하세요.\n"
        )
        (staging / "ver_01_settings.txt").write_text(info, encoding="utf-8-sig")

        staging.rename(final)
    except Exception:
        print(f"백업이 완료되지 않았습니다. 중간 파일은 {staging}에 남아 있습니다.", file=sys.stderr)
        raise

    # 'git add .' 시 대용량 백업 데이터가 실수로 포함되지 않도록
    # 현재 저장소에서만 적용되는 Git 제외 규칙을 기록합니다.
    exclude = repo / ".git" / "info" / "exclude"
    if exclude.is_file():
        try:
            lines = exclude.read_text(encoding="utf-8", errors="replace").splitlines()
            rule = "/IBMTechSupport/backup/"
            if rule not in lines:
                with exclude.open("a", encoding="utf-8") as stream:
                    stream.write("\n# Local experiment backups\n" + rule + "\n")
                print("Git local exclude added:", rule)
        except OSError as error:
            print(f"Git 제외 규칙 자동 설정 실패: {error}", file=sys.stderr)
            print(".git/info/exclude에 /IBMTechSupport/backup/ 를 직접 추가하세요.")

    print("\nBackup complete:", final)
    print("Settings:", final / "ver_01_settings.txt")
    print("Manifest:", final / "file_manifest.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("ERROR:", error, file=sys.stderr)
        sys.exit(1)
