"""Evaluate the saved QLoRA adapter on val_106.json, without RAG.
Place in IBMTechSupport/scripts/evaluate/.
Run from IBMTechSupport: python scripts/evaluate/evaluate_qlora.py
Use --quality-only to resume quality evaluation of saved predictions.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

import yaml
from evaluate import evaluate_stage2_quality as quality
from evaluate.common import score_predictions, save_json, save_jsonl
from evaluate.evaluate_base import make_prediction_rows


BASE_CONFIG = ROOT / 'configs/base/v1_106.yaml'
ADAPTER = ROOT / 'models/Qwen2.5-1.5B-Instruct-QLoRA'
OUTPUT = ROOT / 'evaluate_results/106/qlora'
DATASET = ROOT / 'data/processed/val_106.json'
BASE_MODEL = ROOT / 'models/Qwen2.5-1.5B-Instruct'
QUANTIZATION = {'load_in_4bit': True, 'quant_type': 'nf4', 'double_quantization': True, 'compute_dtype': 'float16'}


def inputs():
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8-sig'))
    if (ROOT / config['dataset']['path']).resolve() != DATASET.resolve():
        raise ValueError('Expected frozen val_106.json')
    if config['dataset']['expected_count'] != 106:
        raise ValueError('Expected 106 validation cases')
    if (ROOT / config['model']['path']).resolve() != BASE_MODEL.resolve():
        raise ValueError('Unexpected base model')
    inf = config['inference']
    if (inf['max_new_tokens'], inf['do_sample'], inf['num_beams']) != (160, False, 1):
        raise ValueError('Expected greedy decoding with 160 new tokens')
    # Standardize to the frozen RAG evaluation batch size for latency comparison.
    inf['batch_size'] = 1
    records = quality.load_json(DATASET)
    if (len(quality.index_rows(records)), sum(r['is_impossible'] is False for r in records),
            sum(r['is_impossible'] is True for r in records)) != (106, 77, 29):
        raise ValueError('Unexpected validation dataset')
    return config, records


def adapter_hash():
    return {n: quality.sha256(ADAPTER / n) for n in ('adapter_config.json', 'adapter_model.safetensors')}


def generate(config, records):
    import torch
    from peft import PeftModel
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

    if any((OUTPUT / name).exists() for name in ('predictions.jsonl', 'metrics.json', 'generation_manifest.json')):
        raise FileExistsError('Existing QLoRA results are protected. Use --quality-only or archive the output directory.')
    fingerprint = adapter_hash()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16, quantization_config=quantization,
        device_map={'': 0}, local_files_only=True)
    if not any(isinstance(m, bnb.nn.Linear4bit) for m in base.modules()):
        raise RuntimeError('4-bit quantization was not applied')
    inf = config['inference']
    generation_config = GenerationConfig(
        max_new_tokens=inf['max_new_tokens'], do_sample=inf['do_sample'],
        num_beams=inf['num_beams'], pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False, local_files_only=True)
    model.eval()
    rows, seconds, memory = make_prediction_rows(
        model, tokenizer, generation_config, records, 1, config['prompt'])
    metrics = score_predictions(records, rows, 'val_106.json', 1)
    metrics.update({'model': 'Qwen2.5-1.5B-Instruct-QLoRA', 'method': 'QLoRA', 'quantization': QUANTIZATION,
                    'generation_seconds': round(seconds, 3),
                    'mean_generation_seconds_per_question': round(seconds / len(records), 4),
                    'questions_per_second': round(len(records) / seconds, 3) if seconds > 0 else None,
                    'peak_gpu_memory_gb': round(memory, 3),
                    'retrieval': None, 'faithfulness_percent': None})
    save_jsonl(OUTPUT / 'predictions.jsonl', rows)
    save_json(OUTPUT / 'metrics.json', metrics)
    save_json(OUTPUT / 'generation_manifest.json', {
        'dataset_sha256': quality.sha256(DATASET), 'expected_count': len(records),
        'predictions_sha256': quality.sha256(OUTPUT / 'predictions.jsonl'),
        'base_config_sha256': quality.sha256(BASE_CONFIG), 'adapter_sha256': fingerprint,
        'inference': config['inference'], 'prompt': config['prompt'], 'quantization': QUANTIZATION})
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    del model, base, tokenizer, generation_config
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_quality(config, records):
    prediction_path = OUTPUT / 'predictions.jsonl'
    manifest = quality.load_json(OUTPUT / 'generation_manifest.json')
    expected = {'dataset_sha256': quality.sha256(DATASET), 'expected_count': len(records),
                'predictions_sha256': quality.sha256(prediction_path),
                'base_config_sha256': quality.sha256(BASE_CONFIG), 'adapter_sha256': adapter_hash(),
                'inference': config['inference'], 'prompt': config['prompt'], 'quantization': QUANTIZATION}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Saved prediction provenance mismatch')
    predictions = quality.index_rows(quality.load_jsonl(prediction_path))
    if set(predictions) != {r['id'] for r in records}:
        raise ValueError('Prediction IDs do not match the dataset')
    ordered = [predictions[r['id']] for r in records]
    scores = score_predictions(records, ordered, 'val_106.json', 1)
    saved_metrics = quality.load_json(OUTPUT / 'metrics.json')
    for k in ('exact_match_percent', 'token_f1_percent', 'abstention_rate_percent', 'false_refusal_rate_percent'):
        if saved_metrics[k] != scores[k]:
            raise ValueError(f'Prediction metric mismatch: {k}')
    out = OUTPUT / 'quality_evaluation'
    if any((out / n).exists() for n in ('metrics.json', 'cases.jsonl')):
        raise FileExistsError('Existing quality results are protected')
    start = time.perf_counter()
    # Independent evaluation prompts, original base weights, never QLoRA weights.
    evaluator = quality.Evaluator(BASE_MODEL)
    cases = []
    for i, r in enumerate(records, 1):
        answer = predictions[r['id']]['prediction']
        refused = answer.strip() == quality.ABSTAIN
        if r['is_impossible']:
            verdict = quality.fixed('NOT_APPLICABLE', 'No reference answer; refusal policy is evaluated separately')
        elif refused:
            verdict = quality.fixed('REFUSED', 'No answer on answerable question')
        elif not answer.strip():
            verdict = quality.fixed('INCORRECT', 'Empty answer')
        else:
            verdict = evaluator.evaluate(quality.CORRECT_SYSTEM,
                {'question': r['question'], 'candidate_answer': answer,
                 'reference_answer': r['answer'], 'reference_documents': quality.gold_excerpts(r)},
                quality.CORRECT_LABELS)
        cases.append({'id': r['id'], 'question': r['question'], 'is_impossible': r['is_impossible'],
                      'reference_answer': r['answer'], 'prediction': answer, 'refused': refused,
                      'correctness': verdict, 'faithfulness': None})
        if i == 1 or i % 5 == 0 or i == len(records):
            print(f'[QLoRA Quality] {i}/{len(records)}', flush=True)
    answerable = [r for r in cases if not r['is_impossible']]
    counts = Counter(r['correctness']['verdict'] for r in answerable)
    metrics = {'model': 'Qwen2.5-1.5B-Instruct-QLoRA', 'dataset': 'val_106.json',
               'answer_correctness_percent': quality.pct(counts['CORRECT'], len(answerable)),
               'answer_correctness': {'definition': 'Strict CORRECT / all answerable questions',
                                      'counts': dict(counts), 'denominator': len(answerable),
                                      'strict_percent': quality.pct(counts['CORRECT'], len(answerable)),
                                      'strict_among_emitted_answerable_percent': quality.pct(counts['CORRECT'], sum(not r['refused'] for r in answerable)),
                                      'adjudicated_only_percent': quality.pct(counts['CORRECT'], counts['CORRECT'] + counts['PARTIAL'] + counts['INCORRECT']),
                                      'uncertain_percent_of_answerable': quality.pct(counts['UNCERTAIN'], len(answerable))},
               'faithfulness_percent': None, 'retrieval': None,
               'not_applicable_reason': 'No retrieved evidence is supplied to the generator',
               'evaluator_invalid_outputs': sum(r['correctness']['format_valid'] is False for r in cases),
               'judge_model': str(BASE_MODEL), 'human_review_required': True,
               'dataset_sha256': expected['dataset_sha256'], 'predictions_sha256': expected['predictions_sha256'],
               'elapsed_seconds': round(time.perf_counter() - start, 3)}
    save_jsonl(out / 'cases.jsonl', cases)
    save_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-only', action='store_true', help='Reuse existing QLoRA predictions')
    args = parser.parse_args()
    config, records = inputs()
    if not args.quality_only:
        generate(config, records)
    evaluate_quality(config, records)
    print('QLoRA Validation: OK')


if __name__ == '__main__':
    main()
