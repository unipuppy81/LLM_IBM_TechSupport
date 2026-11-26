from __future__ import annotations

import argparse
import importlib
import json
import struct
from datetime import datetime
from pathlib import Path, PureWindowsPath, PurePosixPath

import run_stage3_test4_prompt as prompt
import run_stage3_test1 as lr
import run_stage2_test4 as weighted

s0, t2, quality = weighted.s0, weighted.t2, weighted.quality
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage4_test1'
SCRIPT = Path(__file__).resolve()
TARGETS = {
    'A_attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'B_all_linear': 'all-linear',
}
EXPECTED = {
    'A_attention': set(TARGETS['A_attention']),
    'B_all_linear': {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'},
}
BASE_TARGETS = {'q_proj', 'v_proj'}
WEIGHT = 0.175
PHASES = ('train', 'generate', 'quality')


def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT / 'models/finetuning/stage4_test1' / run.parent.parent.name / run.name / method


def tensor_count(path):
    # Safetensors header only; no model allocation or GPU use.
    with path.open('rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        if header_len > 100_000_000 or header_len > path.stat().st_size - 8:
            raise ValueError(f'Invalid safetensors header: {path}')
        header = json.loads(f.read(header_len))
    from math import prod
    return sum(prod(item['shape']) for key, item in header.items() if key != '__metadata__')


def check_adapter(path, expected):
    cfg = s0.read(path / 'adapter_config.json')
    if (cfg['r'], cfg['lora_alpha'], cfg['lora_dropout']) != (8, 16, 0.05):
        raise ValueError('Rank, alpha or dropout changed unexpectedly')
    if cfg.get('bias') != 'none' or cfg.get('task_type') != 'CAUSAL_LM':
        raise ValueError('Unexpected LoRA bias or task type')
    targets = cfg['target_modules']
    if isinstance(targets, str) or set(targets) != expected:
        raise ValueError(f'Unexpected saved targets: {targets}; expected={sorted(expected)}')
    return tensor_count(path / 'adapter_model.safetensors')


def install_targets(trainer, name):
    # Only the target_modules argument is changed. The original optimizer,
    # dataset, loss-weighting wrapper and training loop remain unchanged.
    original_config = trainer.LoraConfig
    original_get_peft = trainer.get_peft_model
    audit = {}

    def config(*args, **kwargs):
        if args or set(kwargs.get('target_modules', [])) != BASE_TARGETS:
            raise ValueError('Unexpected original target_modules configuration')
        if (kwargs.get('r'), kwargs.get('lora_alpha'), kwargs.get('lora_dropout')) != (8, 16, 0.05):
            raise ValueError('Unexpected original LoRA hyperparameters')
        kwargs['target_modules'] = TARGETS[name]
        return original_config(**kwargs)

    def get_peft(model, config, *args, **kwargs):
        result = original_get_peft(model, config, *args, **kwargs)
        params = [(n, p) for n, p in result.named_parameters() if p.requires_grad]
        modules = set()
        for n, p in params:
            pieces = n.split('.')
            indices = [i for i, value in enumerate(pieces) if value in ('lora_A', 'lora_B')]
            if len(indices) != 1 or indices[0] == 0:
                raise ValueError(f'Unexpected trainable parameter: {n}')
            modules.add(pieces[indices[0] - 1])
        if modules != EXPECTED[name]:
            raise ValueError(f'Actual trainable targets differ: {sorted(modules)}')
        audit.update(target_modules=sorted(modules), trainable_parameters=sum(p.numel() for _, p in params),
                     total_parameters=sum(p.numel() for p in result.parameters()),
                     rank=8, alpha=16, dropout=0.05)
        return result

    trainer.LoraConfig = config
    trainer.get_peft_model = get_peft
    return audit


def relocate_path(filename, *allowed_roots):
    # Historical manifests contain absolute paths from the original machine.
    # Match the entire project-relative directory, not just a basename.
    if not isinstance(filename, str) or not filename:
        raise ValueError(f'Invalid recorded path: {filename!r}')
    win = PureWindowsPath(filename)
    windows = win.is_absolute()
    recorded = win if windows else PurePosixPath(filename)
    if not recorded.is_absolute() or '..' in recorded.parts:
        raise ValueError(f'Invalid recorded absolute path: {filename}')
    normalize = str.casefold if windows else str
    parts = tuple(normalize(p) for p in recorded.parts)
    matches = set()
    for allowed in allowed_roots:
        allowed = allowed.resolve()
        relative = allowed.relative_to(ROOT.resolve())
        prefix = tuple(normalize(p) for p in (ROOT.name, *relative.parts))
        for i in range(len(parts) - len(prefix) + 1):
            if parts[i:i+len(prefix)] != prefix:
                continue
            tail = recorded.parts[i+len(prefix):]
            mapped = allowed.joinpath(*tail).resolve()
            if not mapped.is_relative_to(allowed):
                raise ValueError(f'Recorded path escapes allowed directory: {filename}')
            matches.add(mapped)
    if len(matches) != 1:
        raise ValueError(f'Unexpected or ambiguous recorded path: {filename}')
    return matches.pop()


def verify_artifact(path, digest):
    if not path.is_file():
        raise FileNotFoundError(f'Previously completed artifact missing at current location: {path}')
    # Git checkout can change line endings in text files. Accept only an exact
    # recorded hash or an exact LF/CRLF variant. Weights remain byte-verified.
    return prompt.verify_sha(path, digest, path.suffix.lower() in {
        '.json', '.jsonl', '.yaml', '.yml', '.txt', '.md'})


def load_relocated_baseline(parent):
    parent = parent.resolve()
    if parent.parent != prompt.E_ROOT.parent.resolve():
        raise ValueError('Expected a Stage 2 midpoint run directory')
    m, c = s0.read(parent/'manifest.json'), s0.read(parent/'comparison.json')
    if (m['stage'], m['test'], c['complete'], c['quality_status'], c['provisional_selection']) != (2, 3, True, 'COMPLETE', prompt.E_NAME):
        raise ValueError('Expected completed and selected Stage 2 E run')
    if m['weights'] != {prompt.E_NAME: 0.175} or m['settings'] != dict(s0.SETTINGS, max_length=2048):
        raise ValueError('Unexpected E training settings')
    cfg = m['config']
    spec = m['spec']
    if (spec['question_tokens'], spec['answer_tokens'], spec['max_new_tokens'], spec['max_length']) != (192,160,160,2048):
        raise ValueError('Unexpected E token limits')
    if (ROOT/cfg['dataset']['path']).resolve() != s0.VAL.resolve() or cfg['dataset']['expected_count'] != 106:
        raise ValueError('Only val_106.json may be evaluated')
    if (ROOT/cfg['model']['path']).resolve() != s0.BASE.resolve():
        raise ValueError('Unexpected base model')
    inf = cfg['inference']
    if (inf['batch_size'], inf['max_new_tokens'], inf['do_sample'], inf['num_beams']) != (1,160,False,1):
        raise ValueError('Expected original greedy inference configuration')
    if cfg['prompt']['user_template'] != '{question}' or cfg['prompt']['system'] != spec['prompt']:
        raise ValueError('Unexpected original prompt')
    run = parent/'experiments'/prompt.E_NAME
    adapter = ROOT/'models/finetuning/stage2_test3_midpoint'/parent.name/prompt.E_NAME/'lora'
    # Verify completed artifacts, not irrelevant historical training launchers.
    marked = {}
    for phase in ('train', 'generate', 'quality'):
        marker = s0.read(run/'completed'/f'{phase}_lora.json')['files']
        if not marker:
            raise ValueError(f'Empty completion marker: {phase}')
        for filename, digest in marker.items():
            path = relocate_path(filename, run, adapter)
            if path in marked and marked[path] != digest:
                raise ValueError(f'Conflicting completion marker hashes: {path}')
            verify_artifact(path, digest)
            marked[path] = digest
    required = [adapter/'adapter_model.safetensors', adapter/'adapter_config.json',
                run/'training/lora_train_metrics.json', run/'training/weight_audit.json',
                run/'lora/predictions.jsonl', run/'lora/metrics.json',
                run/'lora/generation_manifest.json', run/'lora/quality_evaluation/metrics.json',
                run/'lora/quality_evaluation/cases.jsonl']
    if not all(p.resolve() in marked for p in required):
        raise ValueError('Missing protected baseline artifacts')
    training_hash = verify_artifact(run/'train.json', m['training_sha256'])
    if s0.read(run/'prompt_config.yaml') != cfg:
        raise ValueError('Baseline prompt snapshot changed')
    audit = s0.read(run/'training/weight_audit.json')
    if (audit['answerable_weight'], audit['impossible_weight'], audit['validation_weight']) != (1.0,0.175,1.0):
        raise ValueError('Baseline weight audit mismatch')
    records = s0.validate(s0.VAL)
    if (len(records),sum(r['is_impossible'] for r in records)) != (106,29):
        raise ValueError('Expected 77 answerable / 29 impossible questions')
    val_hash = verify_artifact(s0.VAL, m['data']['val_sha256'])
    # The old evaluators and shared judge implementation must be unchanged;
    # tolerate only CRLF/LF differences. Trainer/launcher docstrings are irrelevant here.
    sources = {}
    for name in ('scripts/load_base_model.py','scripts/evaluate/evaluate_base.py',
                 'scripts/evaluate/evaluate_lora.py','scripts/evaluate/evaluate_stage2_quality.py',
                 'scripts/evaluate/common.py','scripts/document_utils.py'):
        sources[name] = prompt.verify_sha(ROOT/name,m['sources'][name],True)
    for module in (s0,quality):
        p=Path(module.__file__).resolve()
        sources[p.name]=prompt.verify_sha(p,m['runner_sha256'][p.name],True)
    model_hashes = m['model_files']
    if set(model_hashes) != {p.name for p in s0.model_files()}:
        raise ValueError('Base model file set changed')
    for name,digest in model_hashes.items():
        prompt.verify_sha(s0.BASE/name,digest)
    base_metrics = quality.metrics(run,'lora')
    if base_metrics != c['candidates'][prompt.E_NAME] or base_metrics['train_samples'] != 539:
        raise ValueError('E baseline metrics mismatch')
    gm = s0.read(run/'lora/generation_manifest.json')
    fingerprint = {name:s0.sha(adapter/name) for name in ('adapter_config.json','adapter_model.safetensors')}
    if set(gm['adapter_sha256']) != set(fingerprint):
        raise ValueError('Unexpected generation adapter file set')
    for name, digest in gm['adapter_sha256'].items():
        verify_artifact(adapter/name, digest)
    verify_artifact(s0.VAL, gm['dataset_sha256'])
    verify_artifact(run/'lora/predictions.jsonl', gm['predictions_sha256'])
    verify_artifact(run/'prompt_config.yaml', gm['base_config_sha256'])
    if gm['prompt'] != cfg['prompt'] or gm['inference'] != inf:
        raise ValueError('E generation provenance mismatch')
    base_metrics.update(generation_seconds=s0.read(run/'lora/metrics.json')['generation_seconds'])
    base_metrics['youden_j_percent_points'] = prompt.j(base_metrics)
    provenance = {'parent_run':str(parent), 'baseline':str(run), 'adapter':str(adapter),
                  'parent_manifest_sha256':s0.sha(parent/'manifest.json'),
                  'parent_comparison_sha256':s0.sha(parent/'comparison.json'),
                  'dataset_sha256':val_hash,'training_sha256':training_hash,
                  'recorded_dataset_sha256':m['data']['val_sha256'],
                  'recorded_training_sha256':m['training_sha256'],
                  'recorded_adapter_sha256':gm['adapter_sha256'],
                  'adapter_sha256':fingerprint,'model_files':model_hashes,'sources':sources,
                  'config':cfg,'settings':m['settings'],'loss_weight':0.175}
    return records, adapter, cfg, base_metrics, provenance


def load_parent(parent):
    _, base_adapter, cfg, baseline, provenance = load_relocated_baseline(parent)
    m = s0.read(parent / 'manifest.json')
    source = parent / 'experiments' / prompt.E_NAME / 'train.json'
    if s0.sha(source) != provenance['training_sha256']:
        raise ValueError('E training data changed')
    rows = s0.validate(source)
    if (len(rows), sum(r['is_impossible'] for r in rows)) != (539, 172):
        raise ValueError('Expected 367 answerable and 172 impossible training samples')
    # Verify the trainer in addition to the evaluator/model hashes checked by
    # load_baseline. Record the currently used helper hashes for resume.
    trainer_path = ROOT / 'scripts/train_lora.py'
    prompt.verify_sha(trainer_path, m['sources']['scripts/train_lora.py'], True)
    base_count = check_adapter(base_adapter, BASE_TARGETS)
    baseline = dict(baseline, trainable_parameters=base_count)
    helpers = (weighted, t2, quality, s0, prompt, lr, lr.mid, weighted.previous)
    hashes = {str(Path(mod.__file__).resolve()): s0.sha(Path(mod.__file__)) for mod in helpers}
    hashes[str(SCRIPT)] = s0.sha(SCRIPT)
    hashes[str(trainer_path)] = s0.sha(trainer_path)
    # Existing quality cache expects the original Stage 0 directory here.
    baseline_cache = relocate_path(m['baseline'], s0.RUNS)
    if baseline_cache.parent != s0.RUNS.resolve() or not baseline_cache.name.startswith('run_'):
        raise ValueError(f'Unexpected Stage 0 cache location: {baseline_cache}')
    if not baseline_cache.is_dir():
        raise FileNotFoundError(f'Stage 0 cache directory missing: {baseline_cache}')
    manifest = {'stage': 4, 'test': 1, 'experiment': 'target_modules',
                'parent_run': str(parent), 'baseline': str(baseline_cache),
                'baseline_provenance': provenance, 'baseline_metrics': baseline,
                'training_source': str(source), 'training_sha256': s0.sha(source),
                'data': dict(m['data'], val_sha256=provenance['dataset_sha256']),
                'sources': m['sources'], 'model_files': m['model_files'],
                'spec': m['spec'], 'config': cfg, 'settings': m['settings'],
                'weights': dict.fromkeys(TARGETS, WEIGHT), 'targets': TARGETS,
                'rank': 8, 'alpha': 16, 'dropout': 0.05, 'runner_hashes': hashes}
    return manifest


def worker(run, phase):
    m = s0.read(run.parent.parent / 'manifest.json')
    if m['targets'] != TARGETS or m['weights'] != dict.fromkeys(TARGETS, WEIGHT):
        raise ValueError('Target module manifest changed')
    for filename, digest in m['runner_hashes'].items():
        if s0.sha(Path(filename)) != digest:
            raise ValueError(f'Active runner changed: {filename}')
    s0.SETTINGS = dict(m['settings'])
    weighted.WEIGHTS = dict.fromkeys(TARGETS, WEIGHT)
    weighted.adapter = adapter
    if phase == 'train':
        trainer = importlib.import_module('train_lora')
        audit = install_targets(trainer, run.name)
        weighted.worker(run, phase)
        if not audit:
            raise ValueError('No LoRA target audit')
        saved_count = check_adapter(adapter(run, 'lora'), EXPECTED[run.name])
        if saved_count != audit['trainable_parameters']:
            raise ValueError('Saved adapter parameter count differs from trained count')
        s0.write(run / 'training/target_audit.json', audit)
    else:
        check_adapter(adapter(run, 'lora'), EXPECTED[run.name])
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
    complete = set(results) == set(TARGETS)
    s0.write(root / 'comparison.json', {
        'baseline_lora': baseline, 'baseline_target_modules': sorted(BASE_TARGETS),
        'target_modules': TARGETS, 'candidates': results, 'quality_status': statuses,
        'youden_j_percent_points': {'baseline': lr.j(baseline), **{n: lr.j(r) for n, r in results.items()}},
        'provisional_selection': chosen, 'complete': complete,
        'human_review_required': True, 'qlora': 'NOT RUN', 'hallucination_rate_percent': None,
        'notes': ['Same E dataset, prompt, sample weights, rank, alpha and training settings.',
                  'Each candidate trained independently from original base weights.',
                  'Quality may be skipped for failed generation screening. N/A is not zero.']})
    def fmt(v):
        if v is None:
            return 'N/A'
        return f'{v:.4f}'.rstrip('0').rstrip('.')
    cols = [baseline, *results.values()]
    lines = ['[Stage 4 Test1 Target Modules Comparison]',
             f'{"Metric":36} ' + ' '.join(f'{n:>16}' for n in ['Baseline_qv', *results])]
    for k in (*t2.KEYS, 'youden_j_percent_points', 'trainable_parameters'):
        vals = [lr.j(r) if k == 'youden_j_percent_points' else r.get(k) for r in cols]
        lines.append(f'{k:36} ' + ' '.join(f'{fmt(v):>16}' for v in vals))
    lines += [f'Targets: {TARGETS}', f'Quality: {statuses}', f'Provisional selection: {chosen}',
              f'Complete: {complete}', 'Validation loss: unweighted. Training loss: weighted.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN', f'Results: {root}']
    (root / 'comparison.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description='Stage 4 Test1: Target Modules')
    p.add_argument('--parent', type=Path, default=prompt.E_ROOT)
    p.add_argument('--resume', type=Path)
    p.add_argument('--through', choices=('A', 'B'), default='B')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--full-quality', action='store_true')
    p.add_argument('--worker', choices=PHASES, help=argparse.SUPPRESS)
    p.add_argument('--run', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        if a.run is None or a.run.resolve().parent.parent.parent != RUNS.resolve() or a.run.name not in TARGETS:
            p.error('Invalid worker directory')
        worker(a.run.resolve(), a.worker)
        return
    if a.resume:
        root = a.resume.resolve()
        if root.parent != RUNS.resolve():
            p.error('Invalid resume directory')
        old = s0.read(root / 'manifest.json')
        a.parent = Path(old['parent_run'])
    manifest = load_parent(a.parent.resolve())
    if a.resume:
        if old != manifest:
            raise ValueError('Resume blocked: provenance changed')
    else:
        root = RUNS / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True, exist_ok=False)
        s0.write(root / 'manifest.json', manifest)
    baseline = manifest['baseline_metrics']
    print(f'[Stage 4 Test1] Run: {root}', flush=True)
    if a.prepare_only:
        print('Preparation: OK (no training)', flush=True)
        return
    results, statuses = {}, {}
    try:
        for i, name in enumerate(TARGETS, 1):
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
            print(f'[Experiment] {i}/{len(TARGETS)} | {name}', flush=True)
            for phase in ('train', 'generate'):
                execute_phase(run, phase)
            candidate = lr.mid.generation_metrics(run)
            audit = s0.read(run / 'training/target_audit.json')
            count = check_adapter(adapter(run, 'lora'), EXPECTED[name])
            if count != audit['trainable_parameters'] or candidate['train_samples'] != 539:
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
        print(f'Resume: python IBMTechSupport/scripts/finetuning/{SCRIPT.name} --resume "{root}"', flush=True)
        raise


if __name__ == '__main__':
    main()
