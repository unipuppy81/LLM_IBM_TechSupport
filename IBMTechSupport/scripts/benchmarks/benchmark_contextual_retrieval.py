"""Step 4: lightweight *title-context* retrieval ablation for IBMTechSupport.

IMPORTANT: This is a deterministic, title-based approximation of Contextual Retrieval,
NOT the full LLM-generated per-chunk context technique described by Anthropic.
No reference answers, QA labels or model predictions are used when creating context.

Run from the Git repository root on Windows:
  python IBMTechSupport/scripts/benchmarks/benchmark_contextual_retrieval.py --mode inspect
  python IBMTechSupport/scripts/benchmarks/benchmark_contextual_retrieval.py --mode prepare
  python IBMTechSupport/scripts/benchmarks/benchmark_contextual_retrieval.py --mode build-bm25
  python IBMTechSupport/scripts/benchmarks/benchmark_contextual_retrieval.py --mode evaluate --variant bm25-only
  python IBMTechSupport/scripts/benchmarks/benchmark_contextual_retrieval.py --mode build-dense
  python IBMTechSupport/scripts/benchmarks/benchmark_contextual_retrieval.py --mode evaluate --variant both
  python IBMTechSupport/scripts/benchmarks/benchmark_contextual_retrieval.py --mode report

A/B conditions:
  bm25-only : contextual BM25, unchanged original Dense HNSW, original reranker text
  both      : contextual BM25 + contextual Dense HNSW, original reranker text
  baseline  : original stored reranker Top 5, original Dense and BM25

No baseline corpus/index/config/model is written to. Reuses existing fitted PCA
transformation (192d) so dimensionality and query transformation stay fixed.
The reranker ALWAYS sees the original, unmodified chunk text; only candidate
retrieval indices change. Gold source filenames (not gold evidence spans) are used
solely for evaluation on the 90 answerable questions.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SOURCE = PROJECT / 'data/processed/corpus_chunks.jsonl'
BASELINE = PROJECT / 'evaluate_results/rag/final_e2e/reranker/predictions.jsonl'
DATA = PROJECT / 'data/processed/contextual_title_v1'
CONTEXT_CORPUS = DATA / 'contextual_chunks.jsonl'
MANIFEST = DATA / 'manifest.json'
BM25_DIR = DATA / 'bm25_index'
VECTORS = DATA / 'embeddings_pca192.npy'
INDEX = DATA / 'faiss_hnsw_pca192.index'
RESULTS = PROJECT / 'evaluate_results/rag/contextual_retrieval/title_v1'
VERSION = 1
TITLE_LIMIT = 160


def parse_args():
    p = argparse.ArgumentParser(description='Contextual BM25 + Dense retrieval A/B, title metadata only')
    p.add_argument('--mode', choices=['inspect', 'prepare', 'build-bm25', 'build-dense', 'evaluate', 'report'], default='inspect')
    p.add_argument('--variant', choices=['bm25-only', 'both'], default='bm25-only')
    p.add_argument('--limit', type=int, default=0, help='평가 시 처음 N개 answerable 질문만 테스트 (0=90개 전체)')
    p.add_argument('--batch-size', type=int, default=16, help='Dense 임베딩 GPU 배치 크기')
    args = p.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        p.error('limit은 0 이상, batch-size는 1 이상이어야 합니다.')
    return args


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_config():
    from retrieval_config import load_retrieval_config
    c = load_retrieval_config()
    d, f = c['dense'], c['dense']['faiss']
    if (d['model'] != 'BAAI/bge-small-en-v1.5' or not d['pca']['enabled']
            or int(d['pca']['dimension']) != 192 or f['index_type'] != 'hnsw'
            or not d['normalize_embeddings'] or int(d['max_seq_length']) != 512):
        raise ValueError('이번 실험은 기존 BGE Small + PCA192 + HNSW + normalized embedding 설정만 지원합니다.')
    return c


def config_digest(config):
    # Save the settings actually used to produce this retrieval index.
    settings = {k: config[k] for k in ('chunking', 'bm25', 'dense', 'hybrid', 'reranker')}
    return hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_key(s):
    return str(s).replace('\\', '/').split('/')[-1].strip().casefold()


def first_title(text):
    """Only use an explicit Title: field in the source's first chunk."""
    match = re.search(r'(?im)^\s*title\s*:\s*([^\r\n]+)', text[:1400])
    if match is None:
        return ''
    return re.sub(r'\s+', ' ', match.group(1)).strip()[:TITLE_LIMIT]


def title_header(original_text, title):
    if not title or title.casefold() in original_text[:500].casefold():
        return ''
    return f'Document title: {title}\n'


def read_titles(source=None):
    source = SOURCE if source is None else source
    titles, counts = {}, Counter()
    seen = set()
    for row in read_jsonl(source):
        docid = row.get('document_id')
        if not isinstance(docid, str):
            raise ValueError('corpus chunk has no document_id')
        if row.get('chunk_index') == 0:
            if docid in seen:
                raise ValueError(f'duplicate document first chunk: {docid}')
            seen.add(docid)
            titles[docid] = first_title(row['text'])
        counts[docid] += 1
    if not titles or len(titles) != len(counts):
        raise ValueError('각 원본 문서에는 chunk_index=0이 필요합니다.')
    return titles, counts


def load_baseline():
    rows = list(read_jsonl(BASELINE))
    if len(rows) != 135 or len({r['id'] for r in rows}) != 135:
        raise ValueError('Baseline 135건의 중복 없는 원본 예측이 필요합니다.')
    for r in rows:
        docs = r.get('retrieved_documents', [])
        if len(docs) != 5:
            raise ValueError(f'baseline Top5 누락: {r["id"]}')
        if len({d['row_id'] for d in docs}) != 5:
            raise ValueError(f'baseline Top5 row_id 중복: {r["id"]}')
    return rows


def validate_saved_snapshot():
    """Ensure contextualized indices are built from the identical source corpus as baseline."""
    base = load_baseline()
    lookup = {}
    for r in base:
        for d in r['retrieved_documents']:
            rid = d['row_id']
            expected = (d['chunk_id'], source_key(d['filename']), d['text'])
            if rid in lookup and lookup[rid] != expected:
                raise ValueError(f'baseline Top5 source conflict: row_id={rid}')
            lookup[rid] = expected
    for rid, row in enumerate(read_jsonl(SOURCE)):
        if rid in lookup:
            item = (row['chunk_id'], source_key(row['filename']), row['text'])
            if lookup.pop(rid) != item:
                raise ValueError(f'기존 Reranker 결과와 현재 corpus_chunks.jsonl가 다릅니다: row={rid}')
    if lookup:
        raise ValueError(f'원본 corpus 행 번호가 일치하지 않음: {list(lookup)[:3]}')


def inspect():
    titles, counts = read_titles()
    samples = []
    modified = 0
    total = 0
    for row in read_jsonl(SOURCE):
        title = titles[row['document_id']]
        header = title_header(row['text'], title)
        modified += bool(header)
        total += 1
        if header and len(samples) < 3:
            samples.append((row['chunk_id'], header.strip(), row['text'][:90].replace('\n', ' ')))
    print(f'Documents={len(titles):,} | chunks={total:,} | named_documents={sum(bool(x) for x in titles.values()):,}')
    print(f'Context-prepended chunks={modified:,}/{total:,} ({modified/total*100:.1f}%)')
    print('Context source: explicit Title: field from the first chunk; NO LLM generation')
    for chunk_id, header, snippet in samples:
        print(f'  {chunk_id} | {header} | original={snippet!r}')
    print('Original corpus, embeddings, BM25, FAISS and retrieval.yaml remain unchanged.')


def prepare():
    config = load_config()
    DATA.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in (CONTEXT_CORPUS, MANIFEST)):
        raise FileExistsError(f'이미 Context corpus/manifest가 있습니다: {DATA}. 기존 파일을 보호합니다.')
    validate_saved_snapshot()
    titles, doc_counts = read_titles()
    source_hash = sha256_file(SOURCE)
    temp = Path(str(CONTEXT_CORPUS) + '.part')
    if temp.exists():
        raise FileExistsError(f'이전 중단 시 임시 파일 확인: {temp}')
    stats = Counter()
    try:
        with temp.open('x', encoding='utf-8') as out:
            for row in read_jsonl(SOURCE):
                old = row['text']
                title = titles[row['document_id']]
                header = title_header(old, title)
                record = {**row, 'text': header + old, 'original_text': old, 'context_header': header}
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['chunks'] += 1
                stats['modified'] += bool(header)
        if not stats['chunks']:
            raise RuntimeError('No source chunks')
        manifest = {
            'version': VERSION, 'kind': 'deterministic_title_context_not_llm_generated',
            'source_sha256': source_hash,
            'baseline_sha256': sha256_file(BASELINE),
            'config_digest': config_digest(config),
            'documents': len(doc_counts), 'documents_with_title': sum(bool(x) for x in titles.values()),
            'chunks': stats['chunks'], 'modified_chunks': stats['modified'],
            'context_corpus_sha256': sha256_file(temp),
            'title_limit': TITLE_LIMIT,
        }
        temp.replace(CONTEXT_CORPUS)
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('Prepared:', CONTEXT_CORPUS)


def load_manifest():
    if not MANIFEST.is_file() or not CONTEXT_CORPUS.is_file():
        raise FileNotFoundError('먼저 --mode prepare 실행이 필요합니다.')
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if m['version'] != VERSION or m['config_digest'] != config_digest(load_config()):
        raise ValueError('설정 또는 코드 버전이 준비 단계와 달라졌습니다. 기존 실험 파일을 확인하세요.')
    if m['baseline_sha256'] != sha256_file(BASELINE):
        raise ValueError('원본 baseline predictions.jsonl가 변경되었습니다.')
    if m['source_sha256'] != sha256_file(SOURCE):
        raise ValueError('원본 corpus_chunks.jsonl가 변경되었습니다.')
    if m['context_corpus_sha256'] != sha256_file(CONTEXT_CORPUS):
        raise ValueError('준비된 contextual corpus가 변경되었습니다.')
    return m


def build_bm25():
    import bm25s
    from bm25_search import create_tokenizer
    m = load_manifest()
    if BM25_DIR.exists():
        raise FileExistsError(f'이미 Contextual BM25 인덱스가 있습니다: {BM25_DIR}')
    tmp_dir = DATA / 'bm25_index.part'
    if tmp_dir.exists():
        raise FileExistsError(f'이전 실패 임시 폴더가 있습니다: {tmp_dir}')
    config = load_config()['bm25']
    records = []
    for row in read_jsonl(CONTEXT_CORPUS):
        records.append({'chunk_id': row['chunk_id'], 'filename': row['filename'], 'text': row['text']})
    if len(records) != m['chunks']:
        raise ValueError('contextual chunk 개수 불일치')
    tokenizer = create_tokenizer()
    start = time.perf_counter()
    tokens = tokenizer.tokenize([r['text'] for r in records], return_as='tuple')
    retriever = bm25s.BM25(method=config['method'], k1=config['k1'], b=config['b'])
    retriever.index(tokens)
    try:
        tmp_dir.mkdir()
        retriever.save(str(tmp_dir), corpus=records)
        tokenizer.save_vocab(save_dir=str(tmp_dir))
        # Ensure the directory contains the persisted BM25 index.
        if not (tmp_dir / 'params.index.json').is_file():
            raise RuntimeError('BM25 인덱스가 저장되지 않았습니다')
        tmp_dir.rename(BM25_DIR)
    except Exception:
        # Preserve partial files for inspection rather than overwriting them.
        raise
    print(f'Contextual BM25: {len(records):,} docs / {time.perf_counter()-start:.2f}s')
    print('Saved:', BM25_DIR)


def build_dense(batch_size):
    """Build reduced Dense vectors, then HNSW. Supports resuming embedding batches."""
    import faiss
    import joblib
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    m = load_manifest()
    if INDEX.exists():
        raise FileExistsError(f'이미 Contextual Dense 인덱스가 있습니다: {INDEX}')
    tmp_v = Path(str(VECTORS) + '.part')
    tmp_i = Path(str(INDEX) + '.part')
    progress = DATA / 'dense_progress.json'
    if tmp_i.exists():
        # The embedding stage may have finished, but check the index explicitly.
        raise FileExistsError(f'이전 HNSW 임시 파일 확인 후 제거: {tmp_i}')
    config = load_config()
    d, f = config['dense'], config['dense']['faiss']
    model_dir = PROJECT / 'models' / d['model'].split('/')[-1]
    pca_path = PROJECT / f'data/processed/pca_{d["pca"]["dimension"]}.joblib'
    if not model_dir.is_dir() or not pca_path.is_file():
        raise FileNotFoundError(f'로컬 임베딩 모델/PCA 파일 필요: {model_dir}, {pca_path}')
    output_dim = int(d['pca']['dimension'])
    start = time.perf_counter()

    if not VECTORS.exists():
        if not torch.cuda.is_available():
            raise RuntimeError('기존 실험과 같은 GPU 임베딩을 위해 CUDA가 필요합니다.')
        pca = joblib.load(pca_path)   # Locally generated, trusted PCA file only.
        model = SentenceTransformer(str(model_dir), device='cuda', local_files_only=True)
        model.max_seq_length = int(d['max_seq_length'])
        input_dim = model.get_sentence_embedding_dimension()
        if int(pca.n_features_in_) != input_dim:
            raise ValueError('PCA와 Embedding 모델 차원 불일치')
        print(f'Context Dense embeddings: {m["chunks"]:,} chunks | {input_dim} -> {output_dim} | GPU batch={batch_size}')
        if tmp_v.exists() and not progress.is_file():
            raise FileExistsError(f'진행률 파일 없는 임시 벡터가 있습니다: {tmp_v}. 수동 확인 필요')
        if progress.exists() and not tmp_v.exists():
            raise FileExistsError(f'임시 벡터 없는 진행률 파일이 있습니다: {progress}. 수동 확인 필요')
        if progress.exists():
            state = json.loads(progress.read_text(encoding='utf-8'))
            if state.get('manifest_hash') != m['context_corpus_sha256']:
                raise ValueError('이미 시작된 임베딩 작업의 원본 Context corpus가 다릅니다.')
            count = int(state['completed'])
            if not 0 <= count <= m['chunks']:
                raise ValueError('임베딩 진행률 범위가 잘못됨')
            output = np.lib.format.open_memmap(tmp_v, mode='r+')
            if output.shape != (m['chunks'], output_dim):
                raise ValueError('중단된 임베딩 파일 차원 불일치')
            print(f'Resuming embeddings: {count:,}/{m["chunks"]:,}')
        else:
            count = 0
            output = np.lib.format.open_memmap(tmp_v, mode='w+', dtype=np.float32, shape=(m['chunks'], output_dim))
            progress.write_text(json.dumps({'manifest_hash': m['context_corpus_sha256'], 'completed': 0}), encoding='utf-8')

        def checkpoint(completed):
            output.flush()
            temp_progress = Path(str(progress) + '.part')
            temp_progress.write_text(json.dumps({'manifest_hash': m['context_corpus_sha256'], 'completed': completed}), encoding='utf-8')
            os.replace(temp_progress, progress)

        def flush(batch, index):
            embeddings = model.encode(batch, batch_size=len(batch), convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
            reduced = np.ascontiguousarray(pca.transform(embeddings)[:, :output_dim], dtype=np.float32)
            if not np.isfinite(reduced).all() or np.any(np.linalg.norm(reduced, axis=1) < 1e-12):
                raise ValueError('Contextual PCA embeddings contain NaN/Inf/zero vectors')
            output[index:index + len(batch)] = reduced
            return index + len(batch)

        try:
            batch = []
            for row_id, row in enumerate(read_jsonl(CONTEXT_CORPUS)):
                if row_id < count:
                    continue
                batch.append(row['text'])
                if len(batch) >= batch_size:
                    count = flush(batch, count)
                    batch.clear()
                    if count % (batch_size * 100) == 0:
                        checkpoint(count)
                        print(f'Embedded {count:,}/{m["chunks"]:,}', flush=True)
            if batch:
                count = flush(batch, count)
            if count != m['chunks']:
                raise ValueError(f'임베딩 개수 불일치: {count} != {m["chunks"]}')
            checkpoint(count)
        finally:
            output.flush()
            del output
            del model
        tmp_v.rename(VECTORS)
        progress.unlink(missing_ok=True)
        print(f'Dense embeddings complete: {time.perf_counter()-start:.2f}s')
    else:
        if tmp_v.exists() or progress.exists():
            raise FileExistsError('완료된 벡터와 미완료 임베딩 파일이 동시에 있습니다. 확인 필요.')
        print('Dense embeddings already exist; proceeding to HNSW index.')

    vectors = np.load(VECTORS, mmap_mode='r', allow_pickle=False)
    if vectors.shape != (m['chunks'], output_dim) or vectors.dtype != np.float32:
        raise ValueError('Saved PCA embeddings shape/dtype mismatch')
    faiss.omp_set_num_threads(int(f['threads']))
    index = faiss.IndexHNSWFlat(output_dim, int(f['m']), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(f['ef_construction'])
    index.hnsw.efSearch = int(f['ef_search'])
    for i in range(0, len(vectors), 8192):
        j = min(i + 8192, len(vectors))
        index.add(np.ascontiguousarray(vectors[i:j], dtype=np.float32))
        print(f'Indexed {j:,}/{len(vectors):,}', flush=True)
    if index.ntotal != m['chunks']:
        raise RuntimeError('HNSW index count mismatch')
    faiss.write_index(index, str(tmp_i))
    check = faiss.read_index(str(tmp_i))
    if check.ntotal != m['chunks'] or check.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError('Saved HNSW index validation failed')
    tmp_i.rename(INDEX)
    print('Saved:', VECTORS)
    print('Saved:', INDEX)


def score_ranking(gold, ranked):
    """Scores based on source filenames, NOT verified answer-bearing chunk spans."""
    ranked = [source_key(x) for x in ranked]
    top5 = ranked[:5]
    hit = any(name in gold for name in top5)
    rr = next((1.0 / pos for pos, name in enumerate(top5, 1) if name in gold), 0.0)
    import math
    seen = set()
    dcg = 0.0
    for i, name in enumerate(top5, 1):
        if name in gold and name not in seen:
            dcg += 1.0 / math.log2(i + 1)
        seen.add(name)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(5, len(gold)) + 1))
    return hit, rr, dcg / ideal if ideal else 0.0


def evaluate(variant, limit=0):
    import bm25s
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from bm25_search import create_tokenizer
    from dense_search import INDEX_PATH as ORIGINAL_INDEX, encode_questions
    from hybrid_search import ChunkRowMap, search_all
    from reranker_search import load_reranker, rerank_candidates
    from evaluate.common import load_records
    m = load_manifest()
    if not (BM25_DIR / 'params.index.json').is_file():
        raise FileNotFoundError('먼저 --mode build-bm25를 실행하세요.')
    if variant == 'both' and not INDEX.is_file():
        raise FileNotFoundError('먼저 --mode build-dense를 실행하세요.')
    config = load_config()
    rows = list(read_jsonl(CONTEXT_CORPUS))
    if len(rows) != m['chunks'] or any(row.get('original_text') is None for row in rows):
        raise ValueError('Contextual corpus row alignment invalid')
    original_for_reranker = [{'text': row['original_text']} for row in rows]
    source_keys = [source_key(row['filename']) for row in rows]
    mapping = ChunkRowMap({row['chunk_id']: i for i, row in enumerate(rows)}, source_keys)
    if len(mapping) != len(rows):
        raise ValueError('Duplicate contextual chunk IDs')
    bm25_retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=True, mmap=True)
    if int(bm25_retriever.scores['num_docs']) != len(rows):
        raise ValueError('BM25 index != contextual corpus')
    tokenizer = create_tokenizer()
    tokenizer.load_vocab(str(BM25_DIR))
    bm25_tuple = (bm25_retriever, tokenizer, mapping)
    dense_index = faiss.read_index(str(INDEX if variant == 'both' else ORIGINAL_INDEX))
    if dense_index.ntotal != len(rows) or dense_index.d != 192:
        raise ValueError('Dense index does not align with the 111k chunk corpus')
    dense_index.hnsw.efSearch = int(config['dense']['faiss']['ef_search'])
    faiss.omp_set_num_threads(int(config['dense']['faiss']['threads']))
    model_dir = PROJECT / 'models' / config['dense']['model'].split('/')[-1]
    import torch
    embedding_model = SentenceTransformer(str(model_dir), device='cuda' if torch.cuda.is_available() else 'cpu', local_files_only=True)
    embedding_model.max_seq_length = int(config['dense']['max_seq_length'])
    base = load_baseline()
    data = load_records('clean')
    if len(data) != len(base):
        raise ValueError('Validation / baseline size mismatch')
    for r, b in zip(data, base):
        if (r['id'], r['question'], r['answer'], r['is_impossible']) != (b['id'], b['question'], b['reference_answer'], b['is_impossible']):
            raise ValueError(f'Baseline and validation labels differ: {r["id"]}')
    pairs = [(r, b) for r, b in zip(data, base) if not r['is_impossible']]
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError('No answerable questions')
    outdir = RESULTS / variant
    prefix = f'smoke_{limit}_' if limit else ''
    metrics_path = outdir / (prefix + 'metrics.json')
    cases_path = outdir / (prefix + 'comparison_cases.csv')
    ranked_path = outdir / (prefix + 'retrieval_results.jsonl')
    if any(p.exists() for p in (metrics_path, cases_path, ranked_path)):
        raise FileExistsError(f'해당 실험의 결과가 이미 존재합니다: {outdir}. 기존 파일 보호.')
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Evaluate: {variant} | answerable questions={len(pairs)} | contextual BM25 + {"contextual" if variant == "both" else "original"} Dense')
    start = time.perf_counter()
    questions = [r['question'] for r, b in pairs]
    fusion = search_all(embedding_model, dense_index, bm25_tuple, questions)
    rankings = [x[2] for x in fusion]
    reranked = rerank_candidates(load_reranker(), questions, rankings, original_for_reranker)
    elapsed = time.perf_counter() - start

    stats = Counter()
    summary_rows = []
    ranking_rows = []
    for (gold_row, base_row), rerank in zip(pairs, reranked):
        gold = {source_key(item['filename']) for item in gold_row['contexts']}
        if not gold:
            raise ValueError(f'Gold filename absent for answerable query: {gold_row["id"]}')
        baseline_files = [d['filename'] for d in base_row['retrieved_documents']]
        output_files = [rows[x['row_id']]['filename'] for x in rerank]
        old_hit, old_rr, old_ndcg = score_ranking(gold, baseline_files)
        new_hit, new_rr, new_ndcg = score_ranking(gold, output_files)
        stats['baseline_hits'] += old_hit
        stats['new_hits'] += new_hit
        stats['baseline_rr'] += old_rr
        stats['new_rr'] += new_rr
        stats['baseline_ndcg'] += old_ndcg
        stats['new_ndcg'] += new_ndcg
        stats['recovered'] += not old_hit and new_hit
        stats['lost'] += old_hit and not new_hit
        new_docs = [
            {'row_id': item['row_id'], 'chunk_id': rows[item['row_id']]['chunk_id'],
             'filename': rows[item['row_id']]['filename'], 'text': rows[item['row_id']]['original_text'],
             'rerank_score': float(item['score'])}
            for item in rerank[:5]
        ]
        ranking_rows.append({'id': gold_row['id'], 'question': gold_row['question'],
                             'is_impossible': False, 'retrieved_documents': new_docs})
        summary_rows.append({'id': gold_row['id'], 'baseline_hit5': int(old_hit),
                             'contextual_hit5': int(new_hit), 'change': 'recovered' if not old_hit and new_hit else 'lost' if old_hit and not new_hit else 'same',
                             'gold_files': ';'.join(sorted(gold)),
                             'baseline_top5': ';'.join(source_key(x) for x in baseline_files[:5]),
                             'contextual_top5': ';'.join(source_key(x) for x in output_files[:5])})
    n = len(pairs)
    metrics = {
        'variant': variant, 'kind': m['kind'], 'n_answerable': n,
        'baseline_hit_at_5_percent': round(100 * stats['baseline_hits'] / n, 2),
        'contextual_hit_at_5_percent': round(100 * stats['new_hits'] / n, 2),
        'baseline_mrr_at_5': round(stats['baseline_rr'] / n, 4),
        'contextual_mrr_at_5': round(stats['new_rr'] / n, 4),
        'baseline_ndcg_at_5': round(stats['baseline_ndcg'] / n, 4),
        'contextual_ndcg_at_5': round(stats['new_ndcg'] / n, 4),
        'recovered_gold_sources': stats['recovered'], 'lost_gold_sources': stats['lost'],
        'retrieval_rerank_seconds': round(elapsed, 3),
        'note': 'Gold source filename proxy, not exact answer-bearing chunk. No Judge/Generation test.',
    }
    # Write each result atomically to avoid partial published comparisons.
    for path in (metrics_path, cases_path, ranked_path):
        if path.exists():
            raise FileExistsError(path)
    temp_metrics = Path(str(metrics_path) + '.part')
    temp_cases = Path(str(cases_path) + '.part')
    temp_ranks = Path(str(ranked_path) + '.part')
    try:
        with temp_ranks.open('x', encoding='utf-8') as f:
            for row in ranking_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        with temp_cases.open('x', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        temp_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        temp_ranks.rename(ranked_path)
        temp_cases.rename(cases_path)
        temp_metrics.rename(metrics_path)
    finally:
        for temp in (temp_metrics, temp_cases, temp_ranks):
            temp.unlink(missing_ok=True)
    print('\n[Contextual title retrieval vs fixed baseline]')
    for key, val in metrics.items():
        print(f'  {key}: {val}')
    print('Saved:', metrics_path)
    print('Saved:', cases_path)


def report():
    print('[Title Context Retrieval: 90 answerable questions, original Top5 baseline]')
    found = False
    for variant in ('bm25-only', 'both'):
        path = RESULTS / variant / 'metrics.json'
        if path.exists():
            m = json.loads(path.read_text(encoding='utf-8'))
            print(f'  {variant}: baseline Hit@5={m["baseline_hit_at_5_percent"]}%  contextual Hit@5={m["contextual_hit_at_5_percent"]}% '
                  f' MRR@5={m["contextual_mrr_at_5"]} nDCG@5={m["contextual_ndcg_at_5"]} '
                  f' recovered={m["recovered_gold_sources"]} lost={m["lost_gold_sources"]}')
            found = True
        else:
            print(f'  {variant}: not run')
    if not found:
        print('First run prepare -> build-bm25 -> evaluate.')
    print('Note: this is not the original Anthropic LLM-generated contextualization.')


def main():
    args = parse_args()
    if args.mode == 'inspect':
        inspect()
    elif args.mode == 'prepare':
        prepare()
    elif args.mode == 'build-bm25':
        build_bm25()
    elif args.mode == 'build-dense':
        build_dense(args.batch_size)
    elif args.mode == 'evaluate':
        evaluate(args.variant, args.limit)
    elif args.mode == 'report':
        report()


if __name__ == '__main__':
    main()
