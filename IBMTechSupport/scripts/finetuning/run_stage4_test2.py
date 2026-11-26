from __future__ import annotations

import argparse
import importlib
from datetime import datetime
from pathlib import Path

import run_stage3_test4_prompt as prompt
import run_stage3_test1 as lr
import run_stage2_test4 as weighted
import run_stage4_test1 as target_test

s0, t2, quality = weighted.s0, weighted.t2, weighted.quality
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage4_test2'
SCRIPT = Path(__file__).resolve()
RANKS = {'A_rank_16': 16, 'B_rank_32': 32}
BASE_TARGETS = {'q_proj', 'v_proj'}
WEIGHT = 0.175
BASE_RANK = 8
SCALE = 2
PHASES = ('train', 'generate', 'quality')
TARGET_RUN = target_test.RUNS / 'run_20261126_210609_215297'


def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT / 'models/finetuning/stage4_test2' / run.parent.parent.name / run.name / method


def check_adapter(path, rank):
    cfg = s0.read(path / 'adapter_config.json')
    if (cfg['r'], cfg['lora_alpha'], cfg['lora_dropout']) != (rank, SCALE * rank, 0.05):
        raise ValueError('Saved rank, alpha or dropout mismatch')
    if cfg.get('bias') != 'none' or cfg.get('task_type') != 'CAUSAL_LM' or cfg.get('use_rslora', False) or cfg.get('use_dora', False):
        raise ValueError('Unexpected LoRA configuration')
    targets = cfg['target_modules']
    if isinstance(targets, str) or set(targets) != BASE_TARGETS:
        raise ValueError(f'Unexpected saved targets: {targets}')
    return target_test.tensor_count(path / 'adapter_model.safetensors')


def install_rank(trainer, rank):
    original_config = trainer.LoraConfig
    original_get_peft = trainer.get_peft_model
    audit = {}

    def config(*args, **kwargs):
        if args or set(kwargs.get('target_modules', [])) != BASE_TARGETS:
            raise ValueError('Unexpected original target_modules configuration')
        if (kwargs.get('r'), kwargs.get('lora_alpha'), kwargs.get('lora_dropout')) != (8, 16, 0.05):
            raise ValueError('Unexpected original LoRA hyperparameters')
        if kwargs.get('use_rslora', False) or kwargs.get('use_dora', False):
            raise ValueError('Expected standard LoRA scaling')
        kwargs.update(r=rank, lora_alpha=SCALE * rank)
        return original_config(**kwargs)

    def get_peft(model, config, *args, **kwargs):
        result = original_get_peft(model, config, *args, **kwargs)
        params = [(n, p) for n, p in result.named_parameters() if p.requires_grad]
        modules = set()
        for name, param in params:
            pieces = name.split('.')
            indices = [i for i, value in enumerate(pieces) if value in ('lora_A', 'lora_B')]
            if len(indices) != 1 or indices[0] == 0:
                raise ValueError(f'Unexpected trainable parameter: {name}')
            i = indices[0]
            modules.add(pieces[i - 1])
            axis = 0 if pieces[i] == 'lora_A' else 1
            if len(param.shape) != 2 or param.shape[axis] != rank:
                raise ValueError(f'Actual LoRA rank mismatch: {name}')
        if modules != BASE_TARGETS:
            raise ValueError(f'Actual trainable targets differ: {sorted(modules)}')
        audit.update(target_modules=sorted(modules), trainable_parameters=sum(p.numel() for _, p in params),
                     total_parameters=sum(p.numel() for p in result.parameters()),
                     rank=rank, alpha=SCALE * rank, scaling=SCALE, dropout=0.05)
        # Stage 1 worker checks the original constants before loading the model.
        # Set them only after the actual adapter is created, for correct training metrics.
        trainer.LORA_R, trainer.LORA_ALPHA = rank, SCALE * rank
        return result

    trainer.LoraConfig = config
    trainer.get_peft_model = get_peft
    return audit


def load_parent(parent, target_run):
    manifest = target_test.load_parent(parent)
    target_run = target_run.resolve()
    if target_run.parent != target_test.RUNS.resolve():
        raise ValueError('Expected Stage 4 Test1 run directory')
    previous = s0.read(target_run / 'comparison.json')
    prior_manifest = s0.read(target_run / 'manifest.json')
    if previous.get('complete') is not True or previous.get('provisional_selection') != 'baseline':
        raise ValueError('Expected completed Test1 with q_proj/v_proj baseline selected')
    if (prior_manifest.get('stage'), prior_manifest.get('test')) != (4, 1):
        raise ValueError('Unexpected previous experiment')
    if previous.get('baseline_lora') != manifest['baseline_metrics']:
        raise ValueError('Test1 baseline differs from verified Stage 2 E baseline')
    for key in ('config', 'settings', 'spec'):
        if prior_manifest.get(key) != manifest[key]:
            raise ValueError(f'Test1 {key} differs from current baseline')
    if prior_manifest['baseline_provenance']['recorded_training_sha256'] != manifest['baseline_provenance']['recorded_training_sha256']:
        raise ValueError('Test1 training provenance mismatch')
    hashes = dict(manifest['runner_hashes'])
    hashes[str(SCRIPT)] = s0.sha(SCRIPT)
    manifest.update(stage=4, test=2, experiment='rank', rank_candidates=RANKS,
                    alpha_candidates={name:SCALE * rank for name,rank in RANKS.items()},
                    scaling=SCALE, targets=sorted(BASE_TARGETS), weights=dict.fromkeys(RANKS, WEIGHT),
                    runner_hashes=hashes, target_run=str(target_run),
                    target_comparison_sha256=s0.sha(target_run / 'comparison.json'))
    return manifest


def worker(run, phase):
    m = s0.read(run.parent.parent / 'manifest.json')
    if (m['rank_candidates'] != RANKS or m['alpha_candidates'] != {n:SCALE*r for n,r in RANKS.items()}
            or m['targets'] != sorted(BASE_TARGETS) or m['weights'] != dict.fromkeys(RANKS, WEIGHT)):
        raise ValueError('Rank experiment manifest changed')
    for filename, digest in m['runner_hashes'].items():
        if s0.sha(Path(filename)) != digest:
            raise ValueError(f'Active runner changed: {filename}')
    s0.SETTINGS = dict(m['settings'])
    weighted.WEIGHTS = dict.fromkeys(RANKS, WEIGHT)
    weighted.adapter = adapter
    if phase == 'train':
        trainer = importlib.import_module('train_lora')
        audit = install_rank(trainer, RANKS[run.name])
        weighted.worker(run, phase)
        if not audit:
            raise ValueError('No LoRA rank audit')
        saved_count = check_adapter(adapter(run, 'lora'), RANKS[run.name])
        if saved_count != audit['trainable_parameters'] or saved_count * BASE_RANK != m['baseline_metrics']['trainable_parameters'] * RANKS[run.name]:
            raise ValueError('Saved adapter parameter count differs from trained count')
        s0.write(run / 'training/rank_audit.json', audit)
    else:
        check_adapter(adapter(run, 'lora'), RANKS[run.name])
        weighted.worker(run, phase)


def execute_phase(run, phase):
    old_file, old_adapter = weighted.__file__, weighted.adapter
    try:
        weighted.__file__, weighted.adapter = str(SCRIPT), adapter
        weighted.execute_phase(run, phase)
    finally:
        weighted.__file__, weighted.adapter = old_file, old_adapter


def report(root, baseline, results, statuses):
    chosen = lr.select(results, baseline)
    complete = set(results) == set(RANKS)
    s0.write(root / 'comparison.json', {
        'baseline_lora': baseline, 'baseline_target_modules': sorted(BASE_TARGETS),
        'rank_candidates': RANKS, 'alpha_candidates': {n:SCALE*r for n,r in RANKS.items()}, 'scaling': SCALE, 'candidates': results, 'quality_status': statuses,
        'youden_j_percent_points': {'baseline': lr.j(baseline), **{n: lr.j(r) for n, r in results.items()}},
        'provisional_selection': chosen, 'complete': complete,
        'human_review_required': True, 'qlora': 'NOT RUN', 'hallucination_rate_percent': None,
        'notes': ['Same E dataset, prompt, sample weights, target modules, scaling and training settings. Rank and alpha change together to keep scaling fixed.',
                  'Each candidate trained independently from original base weights.',
                  'Quality may be skipped for failed generation screening. N/A is not zero.']})
    def fmt(v):
        if v is None:
            return 'N/A'
        return f'{v:.4f}'.rstrip('0').rstrip('.')
    cols = [baseline, *results.values()]
    lines = ['[Stage 4 Test2 Rank Comparison]',
             f'{"Metric":36} ' + ' '.join(f'{n:>16}' for n in ['Baseline_r8', *results])]
    for k in (*t2.KEYS, 'youden_j_percent_points', 'trainable_parameters'):
        vals = [lr.j(r) if k == 'youden_j_percent_points' else r.get(k) for r in cols]
        lines.append(f'{k:36} ' + ' '.join(f'{fmt(v):>16}' for v in vals))
    lines += [f'Ranks: {RANKS} | Alpha: { {n:SCALE*r for n,r in RANKS.items()} } | Scaling: {SCALE}', f'Quality: {statuses}', f'Provisional selection: {chosen}',
              f'Complete: {complete}', 'Validation loss: unweighted. Training loss: weighted.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN', f'Results: {root}']
    (root / 'comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description='Stage 4 Test2: Rank')
    p.add_argument('--parent', type=Path, default=prompt.E_ROOT)
    p.add_argument('--target-run', type=Path, default=TARGET_RUN)
    p.add_argument('--resume', type=Path)
    p.add_argument('--through', choices=('A', 'B'), default='B')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--full-quality', action='store_true')
    p.add_argument('--worker', choices=PHASES, help=argparse.SUPPRESS)
    p.add_argument('--run', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        if a.run is None or a.run.resolve().parent.parent.parent != RUNS.resolve() or a.run.name not in RANKS:
            p.error('Invalid worker directory')
        worker(a.run.resolve(), a.worker)
        return
    if a.resume:
        root = a.resume.resolve()
        if root.parent != RUNS.resolve():
            p.error('Invalid resume directory')
        old = s0.read(root / 'manifest.json')
        a.parent = Path(old['parent_run'])
        a.target_run = Path(old['target_run'])
    manifest = load_parent(a.parent.resolve(), a.target_run.resolve())
    if a.resume:
        if old != manifest:
            raise ValueError('Resume blocked: provenance changed')
    else:
        root = RUNS / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True, exist_ok=False)
        s0.write(root / 'manifest.json', manifest)
    baseline = manifest['baseline_metrics']
    print(f'[Stage 4 Test2] Run: {root}', flush=True)
    if a.prepare_only:
        print('Preparation: OK (no training)', flush=True)
        return
    results, statuses = {}, {}
    try:
        for i, name in enumerate(RANKS, 1):
            run = root / 'experiments' / name
            if i == 2 and a.through == 'A' and not (run / 'completed/generate_lora.json').is_file():
                break
            run.mkdir(parents=True, exist_ok=True)
            source = Path(manifest['training_source'])
            dest = run / 'train.json'
            if not dest.exists():
                dest.write_bytes(source.read_bytes())
            if s0.sha(dest) != manifest['training_sha256']:
                raise ValueError('Training snapshot changed')
            cp = t2.config_for(run)
            if not cp.exists():
                s0.write(cp, manifest['config'])
            if s0.read(cp) != manifest['config']:
                raise ValueError('Config snapshot changed')
            print(f'[Experiment] {i}/{len(RANKS)} | {name}', flush=True)
            for phase in ('train', 'generate'):
                execute_phase(run, phase)
            candidate = lr.mid.generation_metrics(run)
            audit = s0.read(run / 'training/rank_audit.json')
            count = check_adapter(adapter(run, 'lora'), RANKS[name])
            if (count != audit['trainable_parameters'] or candidate['train_samples'] != 539
                    or audit['rank'] != RANKS[name] or audit['alpha'] != SCALE * RANKS[name]
                    or s0.read(run/'training/lora_train_metrics.json')['lora_rank'] != RANKS[name]):
                raise ValueError('Training audit mismatch')
            reasons = lr.screen(candidate, baseline)
            s0.write(run / 'screening.json', {'metrics': candidate, 'reasons': reasons, 'passed': not reasons})
            print(f'[Screen] F1={candidate["token_f1_percent"]}% | Correct refusal={candidate["correct_refusal_rate_percent"]}% | False refusal={candidate["false_refusal_rate_percent"]}% | J={lr.j(candidate)}%p', flush=True)
            if not reasons or a.full_quality or (run / 'completed/quality_lora.json').is_file():
                execute_phase(run, 'quality')
                candidate = quality.metrics(run, 'lora')
                status = 'COMPLETE'
            else:
                status = 'SKIPPED: ' + '; '.join(reasons)
            candidate['trainable_parameters'] = count
            if s0.sha(source) != manifest['training_sha256'] or s0.sha(s0.VAL) != manifest['data']['val_sha256']:
                raise ValueError('Dataset changed during run')
            results[name], statuses[name] = candidate, status
            report(root, baseline, results, statuses)
    except BaseException:
        print(f'Resume: python scripts/finetuning/{SCRIPT.name} --resume "{root}"', flush=True)
        raise


if __name__ == '__main__':
    main()
