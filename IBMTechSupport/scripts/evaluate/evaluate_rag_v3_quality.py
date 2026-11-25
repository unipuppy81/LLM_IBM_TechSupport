"""Evaluate frozen RAG V3 saved predictions without retrieval or regeneration.
Place this file in IBMTechSupport/scripts/evaluate/ and run it directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

RESULTS = ROOT / 'evaluate_results/106/rag/stage2_v3_frozen'
EXPECTED_PROMPT = 'f64397ff47417761a44d4e0512939e0608bf1dec7def48ed39ef1b8d48031511'
EXPECTED_RETRIEVAL_CONFIG = 'ff04ed84d035d1f4921362d4c40ec53ca17fd497c4981097c3aa4fb41778adf9'
EXPECTED_GENERATION = {
    'exact_match_percent': 0.0,
    'token_f1_percent': 28.49,
    'abstention_rate_percent': 65.52,
    'false_refusal_rate_percent': 24.68,
}
EXPECTED_RETRIEVAL = {
    'hit_at_5_percent': 85.71,
    'recall_at_5_percent': 83.77,
    'precision_at_5': 0.1714,
    'mrr_at_10': 0.6994,
    'ndcg_at_5': 0.7236,
}
EXPECTED_QUALITY = {'answer_correctness_percent': 57.14, 'faithfulness_percent': 70.59}


def retrieval_metrics(records, predictions):
    from document_utils import chunk_filename, filename_key
    from evaluate.common import retrieval_scores

    totals = Counter()
    n = 0
    hits = Counter()
    for r in records:
        if r['is_impossible']:
            continue
        gold = {filename_key(c['filename']) for c in r['contexts']}
        docs = predictions[r['id']]['retrieved_documents']
        if not gold or '' in gold or not isinstance(docs, list) or len(docs) < 10:
            raise ValueError(f'Missing gold or top-10 retrieval log: {r["id"]}')
        ranked = [filename_key(chunk_filename(d)) for d in docs[:10]]
        totals.update(retrieval_scores(gold, ranked, 5))
        for k in (1, 3, 5, 10):
            hits[k] += bool(gold.intersection(ranked[:k]))
        n += 1
    if n != 77:
        raise ValueError(f'Expected 77 answerable questions, got {n}')
    return {
        'retriever': 'stage2', 'source': 'saved V3 retrieved_documents',
        'answerable_evaluated': n, 'impossible_skipped': len(records) - n,
        **{f'hit_at_{k}_percent': round(hits[k] / n * 100, 2) for k in (1, 3, 5, 10)},
        'recall_at_5_percent': round(totals['recall'] / n * 100, 2),
        'precision_at_5': round(totals['precision'] / n, 4),
        'mrr_at_10': round(totals['rr'] / n, 4),
        'ndcg_at_5': round(totals['ndcg'] / n, 4),
    }


def comparison(actual, expected):
    return {k: {'expected': v, 'actual': actual[k], 'match': actual[k] == v}
            for k, v in expected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RESULTS / 'quality_evaluation')
    parser.add_argument('--retrieval-only', action='store_true', help='No GPU inference; print retrieval metrics only')
    args = parser.parse_args()

    from evaluate import evaluate_stage2_quality as quality
    from evaluate.common import score_predictions, check_output, save_json, save_jsonl

    path = RESULTS / 'predictions.jsonl'
    saved = quality.load_json(RESULTS / 'metrics.json')
    if (saved.get('judge_mode') != 'per_document_v3'
            or saved.get('answerability_prompt_sha256') != EXPECTED_PROMPT
            or saved.get('retrieval_config_sha256') != EXPECTED_RETRIEVAL_CONFIG
            or saved.get('retriever') != 'stage2' or saved.get('top_k') != 5
            or saved.get('total') != 106):
        raise ValueError('Saved metrics do not identify the reported frozen V3 run')

    # Existing quality module validates dataset IDs, reference answers and labels.
    records, predictions, manifest_ok = quality.read_inputs(path)
    prediction_hash = quality.sha256(path)
    dataset_hash = quality.sha256(quality.DATASET)
    generation = score_predictions(records, [dict(predictions[r['id']]) for r in records], 'val_106.json', 1)
    for k, expected in EXPECTED_GENERATION.items():
        if generation[k] != expected or saved.get(k) != generation[k]:
            raise ValueError(f'Frozen V3 generation metric mismatch: {k}')

    retrieval = retrieval_metrics(records, predictions)
    retrieval_check = comparison(retrieval, EXPECTED_RETRIEVAL)
    print(json.dumps({'retrieval': retrieval, 'retrieval_comparison': retrieval_check}, indent=2), flush=True)
    if args.retrieval_only:
        return

    output = args.output.resolve()
    for name in ('metrics.json', 'cases.jsonl'):
        check_output(output / name)

    # Faithfulness must use only the excerpts actually supplied to the generator.
    for r in records:
        p = predictions[r['id']]
        if p['prediction'].strip() != quality.ABSTAIN:
            used, status = quality.used_excerpts(p)
            if not used:
                raise ValueError(f'Missing or invalid generator context: {r["id"]}: {status}')

    start = time.perf_counter()
    judge_path = ROOT / 'models/Qwen2.5-1.5B-Instruct'
    evaluator = quality.Evaluator(judge_path.resolve())
    rows = []
    for i, r in enumerate(records, 1):
        rows.append(quality.evaluate_case(r, predictions[r['id']], evaluator))
        if i == 1 or i % 5 == 0 or i == len(records):
            print(f'[V3 Quality] {i}/{len(records)}', flush=True)
    result = quality.summarize(rows)
    joint = sum(not r['is_impossible'] and r['correctness']['verdict'] == 'CORRECT'
                and r['faithfulness']['verdict'] == 'SUPPORTED' for r in rows)
    actual_quality = {
        'answer_correctness_percent': result['answer_correctness']['strict_percent'],
        'faithfulness_percent': result['faithfulness']['strict_percent'],
    }
    quality_check = comparison(actual_quality, EXPECTED_QUALITY)
    if quality.sha256(path) != prediction_hash or quality.sha256(quality.DATASET) != dataset_hash:
        raise ValueError('Evaluation inputs changed during execution')
    result.update({
        'dataset': 'val_106.json', 'source': 'saved frozen V3 predictions',
        'predictions_sha256': prediction_hash, 'dataset_sha256': dataset_hash,
        'generation_manifest_dataset_verified': manifest_ok,
        'answerability_prompt_sha256_reported': EXPECTED_PROMPT,
        'retrieval_config_sha256_reported': EXPECTED_RETRIEVAL_CONFIG,
        'quality_evaluator_source_sha256': quality.sha256(Path(quality.__file__)),
        'judge_model': str(judge_path.resolve()),
        'judge_independence': 'Separate evaluator prompts; same base weights as generator',
        'generation_metrics': generation,
        'retrieval_metrics': retrieval,
        'correct_and_supported_answerable_count': joint,
        'correct_and_supported_answerable_percent': quality.pct(joint, 77),
        'regression_comparison': {
            'generation': comparison(generation, EXPECTED_GENERATION),
            'retrieval': retrieval_check, 'quality': quality_check,
        },
        'no_test_access': True,
        'elapsed_seconds': round(time.perf_counter() - start, 3),
    })
    save_jsonl(output / 'cases.jsonl', rows)
    save_json(output / 'metrics.json', result)
    print(json.dumps({**actual_quality,
        'correctness_counts': result['answer_correctness']['counts'],
        'faithfulness_counts': result['faithfulness']['counts'],
        'correct_and_supported_answerable_count': joint,
        'correct_and_supported_answerable_percent': result['correct_and_supported_answerable_percent'],
        'quality_comparison': quality_check,
        'saved': str(output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
