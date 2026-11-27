"""Train LoRA from train_clean_ver2.json and compare with the previous 532-QA run.
Place in IBMTechSupport/scripts/finetuning/ alongside run_generation_prompt_test1_retrain.py.
Validation is for loss/checkpoint selection and evaluation only. No test, RAG or Judge.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
from datetime import datetime

from run_generation_prompt_test1_retrain import (
    ROOT, EXPERIMENT, RUNS, KEYS, sha, read, write, verify, adapter_hash,
    validate_sources,
)

TRAIN = ROOT / 'data/processed/train_clean_ver2.json'
EXPECTED_TRAIN_SHA256 = '9372bbf386fa57b18ff67ac8b8e19be2cbad398f3fc33010485dcf953e86e5c7'
BASELINE = RUNS / 'run_20261127_113947_251249'
RESULTS = ROOT / 'evaluate_results/106/finetuning/train_clean_ver2'
MODELS = ROOT / 'models/finetuning/train_clean_ver2'


def main():
    import torch
    import train_lora as trainer
    from evaluate import evaluate_lora as ev
    from evaluate.common import load_jsonl, score_predictions

    m, old, cfg, original_train, val, records = validate_sources()
    verify(TRAIN, EXPECTED_TRAIN_SHA256)
    train_rows = trainer.load_records(TRAIN)
    if len(train_rows) != 548 or len({r['id'] for r in train_rows}) != 548:
        raise ValueError('Expected 548 unique training records')
    if any(r['is_impossible'] for r in train_rows):
        raise ValueError('Expected answerable-only training data')
    if {r['id'] for r in train_rows} & {r['id'] for r in records}:
        raise ValueError('Train/validation ID overlap; stop without modifying data')

    previous = BASELINE / 'A_original_prompt'
    baseline_manifest = read(previous / 'generation_manifest.json')
    for k, value in {'dataset_sha256': sha(val), 'expected_count': 106,
                     'prompt': cfg['prompt'], 'inference': cfg['inference']}.items():
        if baseline_manifest.get(k) != value:
            raise ValueError(f'Baseline configuration mismatch: {k}')
    verify(previous / 'predictions.jsonl', baseline_manifest['predictions_sha256'])
    baseline_metrics = read(previous / 'metrics.json')
    baseline_rows = load_jsonl(previous / 'predictions.jsonl')
    baseline_scores = score_predictions(records, baseline_rows, 'val_106.json', 1)
    for k in KEYS[:4]:
        if baseline_scores[k] != baseline_metrics[k]:
            raise ValueError(f'Baseline metric mismatch: {k}')
    baseline_train = read(BASELINE / 'training/lora_train_metrics.json')
    if baseline_train['train_samples'] != 532:
        raise ValueError('Unexpected baseline training count')
    baseline_exp = read(BASELINE / 'experiment.json')
    if baseline_exp['training_sha256'] != sha(original_train) or baseline_exp['settings'] != m['settings']:
        raise ValueError('Baseline training provenance mismatch')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if trainer.MODEL_DIR.resolve() != (ROOT / cfg['model']['path']).resolve():
        raise ValueError('Unexpected base model path')
    if (trainer.LORA_R, trainer.LORA_ALPHA, trainer.LORA_DROPOUT) != (8, 16, .05):
        raise ValueError('Unexpected LoRA settings')
    run = RESULTS / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    adapter = MODELS / run.name / 'lora'
    run.mkdir(parents=True, exist_ok=False)
    if adapter.exists():
        raise FileExistsError(adapter)
    snapshot = run / 'train.json'
    shutil.copyfile(TRAIN, snapshot)
    verify(snapshot, EXPECTED_TRAIN_SHA256)
    write(run / 'experiment.json', {
        'train_source': str(TRAIN), 'training_sha256': sha(snapshot),
        'baseline': str(BASELINE), 'baseline_training_sha256': sha(original_train),
        'validation_sha256': sha(val), 'settings': m['settings'],
        'prompt': cfg['prompt'], 'inference': cfg['inference'],
        'adapter': str(adapter), 'judge': False, 'rag': False,
        'test_used': False,
    })
    # Reuse the original trainer, changing only the training dataset and output paths.
    trainer.TRAIN_PATH, trainer.VAL_PATH = snapshot, val
    trainer.RESULT_DIR, trainer.LORA_DIR = run / 'training', adapter
    trainer.SYSTEM_PROMPT = cfg['prompt']['system']
    trainer.MAX_QUESTION_TOKENS, trainer.MAX_ANSWER_TOKENS = 192, 160
    print('[Train Clean Ver2 | 548 QA]', flush=True)
    trainer.train_lora(argparse.Namespace(**m['settings']))
    current_train = read(run / 'training/lora_train_metrics.json')
    if (current_train['train_samples'], current_train['val_samples']) != (548, 106):
        raise ValueError('Training count mismatch')
    if len(current_train['history']) != 2 or any(h['optimizer_steps'] != math.ceil(548/8) for h in current_train['history']):
        raise ValueError('Unexpected optimizer update count')
    ac = read(adapter / 'adapter_config.json')
    if (ac['r'], ac['lora_alpha'], ac['lora_dropout'], sorted(ac['target_modules'])) != (8, 16, .05, ['q_proj','v_proj']):
        raise ValueError('Saved adapter configuration mismatch')
    digest = adapter_hash(adapter)
    write(run / 'training_complete.json', {'adapter': str(adapter), 'adapter_sha256': digest, 'metrics': current_train})
    gc.collect()
    torch.cuda.empty_cache()

    ev.ADAPTER, ev.OUTPUT, ev.BASE_CONFIG = adapter, run / 'generation', EXPERIMENT / 'prompt_config.yaml'
    ev.generate(copy.deepcopy(cfg), records)
    gen = read(ev.OUTPUT / 'generation_manifest.json')
    if gen['adapter_sha256'] != digest or gen['prompt'] != cfg['prompt'] or gen['inference'] != cfg['inference'] or gen['dataset_sha256'] != sha(val):
        raise ValueError('Generation provenance mismatch')
    current = read(ev.OUTPUT / 'metrics.json')
    rows = load_jsonl(ev.OUTPUT / 'predictions.jsonl')
    recalculated = score_predictions(records, rows, 'val_106.json', 1)
    for k in KEYS[:4]:
        if current[k] != recalculated[k]:
            raise ValueError(f'Generation metric mismatch: {k}')
    report = {'baseline_training': baseline_train, 'new_training': current_train,
              'baseline_generation': baseline_metrics, 'new_generation': current,
              'judge': 'NOT RUN', 'rag': False, 'complete': True,
              'limitation': 'Training data size and optimizer update count changed; lexical metrics do not establish technical correctness.'}
    write(run / 'comparison.json', report)
    lines = ['[Train Clean Ver2 Comparison]', f'{"Metric":40} {"532 QA":>15} {"548 QA":>15}']
    for k in ('train_samples','best_val_loss','training_seconds','peak_gpu_memory_gb'):
        lines.append(f'{k:40} {str(baseline_train[k]):>15} {str(current_train[k]):>15}')
    for k in KEYS:
        lines.append(f'{k:40} {str(baseline_metrics.get(k)):>15} {str(current.get(k)):>15}')
    lines += ['Judge: NOT RUN | RAG: NONE | Test: NOT USED', f'Adapter: {adapter}', f'Results: {run}']
    (run / 'comparison.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
