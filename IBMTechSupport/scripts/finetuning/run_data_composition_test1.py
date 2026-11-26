from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import run_stage4_test1 as previous

s0, t2, quality, weighted = previous.s0, previous.t2, previous.quality, previous.weighted
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/data_composition_test1'
SCRIPT = Path(__file__).resolve()
NAME = 'A_answerable_only'
PHASES = ('train', 'generate', 'quality')


def adapter(run, method):
    if method != 'lora':
        raise ValueError('LoRA only')
    return ROOT / 'models/finetuning/data_composition_test1' / run.parent.parent.name / run.name / method


def split_records(rows):
    positives = [r for r in rows if r['is_impossible'] is False]
    negatives = [r for r in rows if r['is_impossible'] is True]
    if (len(rows), len(positives), len(negatives)) != (539, 367, 172):
        raise ValueError('Expected verified E dataset: 539 / 367 / 172')
    return positives, negatives


def load_parent(parent):
    manifest = previous.load_parent(parent)
    source = Path(manifest['training_source'])
    positives, negatives = split_records(s0.validate(source))
    val = s0.validate(s0.VAL)
    if {r['id'] for r in positives} & {r['id'] for r in val}:
        raise ValueError('Train/validation ID overlap')
    if {s0.norm(r['question']) for r in positives} & {s0.norm(r['question']) for r in val}:
        raise ValueError('Train/validation normalized question overlap')
    hashes = dict(manifest['runner_hashes'])
    hashes[str(SCRIPT)] = s0.sha(SCRIPT)
    hashes[str(Path(quality.__file__).resolve())] = s0.sha(Path(quality.__file__))
    manifest.update(stage='data_composition', test=1, experiment='answerable_only',
                    original_training_sha256=manifest.pop('training_sha256'),
                    source_counts={'total':539,'answerable':367,'impossible':172},
                    candidate_counts={'total':367,'answerable':367,'impossible':0},
                    removed_ids=[r['id'] for r in negatives],
                    weights={NAME:1.0}, targets=sorted(previous.BASE_TARGETS),
                    runner_hashes=hashes,
                    validation_checkpoint_selection='minimum unweighted loss over all 106 questions',
                    quality_policy='always evaluate, irrespective of refusal rate',
                    training_loss='ordinary per-sample mean CE; positive weight 1.0',
                    optimizer_updates_per_epoch={'baseline':68,'candidate':46})
    return manifest, positives


def verify_snapshot(run, m):
    for filename, digest in m['runner_hashes'].items():
        if s0.sha(Path(filename)) != digest:
            raise ValueError(f'Active runner changed: {filename}')
    source = Path(m['training_source'])
    if s0.sha(source) != m['original_training_sha256'] or s0.sha(s0.VAL) != m['data']['val_sha256']:
        raise ValueError('Source or validation changed')
    positives, negatives = split_records(s0.validate(source))
    if (s0.sha(run/'train.json') != m['training_sha256'] or s0.validate(run/'train.json') != positives
            or [r['id'] for r in negatives] != m['removed_ids']):
        raise ValueError('Answerable-only snapshot changed')
    if s0.read(t2.config_for(run)) != m['config']:
        raise ValueError('Prompt/config snapshot changed')


def check_adapter(run, m):
    count = previous.check_adapter(adapter(run,'lora'), previous.BASE_TARGETS)
    if count != m['baseline_metrics']['trainable_parameters']:
        raise ValueError('Trainable parameter count changed')
    return count


def worker(run, phase):
    m = s0.read(run.parent.parent/'manifest.json')
    verify_snapshot(run, m)
    s0.SETTINGS = dict(m['settings'])
    weighted.adapter = adapter
    quality.adapter = adapter
    s0.adapter = adapter
    if phase == 'train':
        # Unlike the weighted runner, this uses the original loss directly.
        # All retained examples had weight 1.0 in the baseline as well.
        trainer = t2.configure_trainer(m['spec'])
        old_dataset = trainer.QADataset
        counts = []

        class CheckedDataset(old_dataset):
            def __init__(self, records, tokenizer, max_length):
                counts.append((len(records), sum(r['is_impossible'] for r in records)))
                super().__init__(records, tokenizer, max_length)

        trainer.QADataset = CheckedDataset
        quality.worker(run, 'train', 'lora')
        if sorted(counts) != [(106,29),(367,0)]:
            raise ValueError(f'Unexpected training/validation dataset construction: {counts}')
        count = check_adapter(run, m)
        s0.write(run/'training/composition_audit.json', {
            'training_records':367,'training_impossible':0,'validation_records':106,
            'validation_impossible':29,'trainable_parameters':count,
            'training_weight':1.0,'validation_weight':1.0,
            'optimizer_updates_per_epoch':46, 'dataset_construction_counts':counts})
    else:
        check_adapter(run, m)
        weighted.worker(run, phase)


def execute_phase(run, phase):
    old_file, old_adapter = weighted.__file__, weighted.adapter
    try:
        weighted.__file__, weighted.adapter = str(SCRIPT), adapter
        weighted.execute_phase(run, phase)
    finally:
        weighted.__file__, weighted.adapter = old_file, old_adapter


def report(root, m, candidate):
    baseline = m['baseline_metrics']
    keys = ('train_samples', *t2.KEYS, 'youden_j_percent_points', 'trainable_parameters')
    keys = tuple(dict.fromkeys(keys))
    b = dict(baseline, youden_j_percent_points=previous.lr.j(baseline))
    c = dict(candidate, youden_j_percent_points=previous.lr.j(candidate))
    deltas = {k:round(c[k]-b[k],4) for k in keys if c.get(k) is not None and b.get(k) is not None}
    s0.write(root/'comparison.json', {
        'baseline_lora':b,'candidates':{NAME:c},'delta_vs_baseline':deltas,
        'quality_status':'COMPLETE','complete':True,'human_review_required':True,
        'qlora':'NOT RUN','hallucination_rate_percent':None,
        'diagnostic_only':True,'provisional_selection':None,
        'notes':['Removed only the original impossible training records. No answer edits.',
                 'Training starts from original base weights. Positive loss weight stays 1.0.',
                 'Same full 106-question validation, unweighted validation loss and best-loss checkpoint policy.',
                 'At fixed epochs, removal also changes sample order, total optimizer updates and loss composition; this is not an isolated weight intervention.',
                 'Judge always runs. Impossible non-refusals are unverified; reference unsupported is not a true hallucination rate.',
                 'No automatic candidate selection: this is a diagnostic, not the final training dataset.']})
    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}'.rstrip('0').rstrip('.')
    lines = ['[Data Composition Test1 Comparison]', f'{"Metric":36} {"Baseline_539":>16} {"A_positive_367":>16}']
    for k in keys:
        lines.append(f'{k:36} {fmt(b.get(k)):>16} {fmt(c.get(k)):>16}')
    lines += ['Training composition: 367/172 -> 367/0', 'Optimizer updates/epoch: 68 -> 46',
              'Quality: COMPLETE | Complete: True', 'Diagnostic only; no automatic selection.',
              'Automated quality verdicts require human review. QLoRA: NOT RUN', f'Results: {root}']
    (root/'comparison.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    p = argparse.ArgumentParser(description='Data Composition Test1: answerable-only LoRA diagnostic')
    p.add_argument('--parent', type=Path, default=previous.prompt.E_ROOT)
    p.add_argument('--resume', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--worker', choices=PHASES, help=argparse.SUPPRESS)
    p.add_argument('--run', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        if a.run is None or a.run.resolve().parent.parent.parent != RUNS.resolve() or a.run.name != NAME:
            p.error('Invalid worker directory')
        worker(a.run.resolve(), a.worker)
        return
    if a.resume:
        root = a.resume.resolve()
        if root.parent != RUNS.resolve():
            p.error('Invalid resume directory')
        saved = s0.read(root/'manifest.json')
        a.parent = Path(saved['parent_run'])
    manifest, positives = load_parent(a.parent.resolve())
    if a.resume:
        run = root/'experiments'/NAME
        manifest['training_sha256'] = s0.sha(run/'train.json')
        if saved != manifest:
            raise ValueError('Resume blocked: provenance changed')
    else:
        root = RUNS / ('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True, exist_ok=False)
        run = root/'experiments'/NAME
        s0.write(run/'train.json', positives)
        s0.write(t2.config_for(run), manifest['config'])
        manifest['training_sha256'] = s0.sha(run/'train.json')
        s0.write(root/'manifest.json', manifest)
    verify_snapshot(run, manifest)
    print(f'[Data Composition Test1] Run: {root}', flush=True)
    if a.prepare_only:
        print('Preparation: OK (367 answerable; 0 impossible; no training)', flush=True)
        return
    try:
        print(f'[Experiment] 1/1 | {NAME}', flush=True)
        for phase in PHASES:
            execute_phase(run, phase)
        verify_snapshot(run, manifest)
        candidate = quality.metrics(run,'lora')
        candidate['trainable_parameters'] = check_adapter(run, manifest)
        if candidate['train_samples'] != 367:
            raise ValueError('Unexpected training sample count')
        report(root, manifest, candidate)
    except BaseException:
        print(f'Resume: python scripts/finetuning/{SCRIPT.name} --resume "{root}"', flush=True)
        raise


if __name__ == '__main__':
    main()
