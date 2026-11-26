from __future__ import annotations

import argparse
import gc
import re
import sys
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import run_stage3_test4_prompt as previous

s0 = previous.s0
ROOT = s0.ROOT
RUNS = ROOT / 'evaluate_results/106/finetuning/stage3_test5'
NAME = 'A_base'
ABSTAIN = 'INSUFFICIENT_INFORMATION'


class ProgressLog(previous.ProgressLog):
    def write(self, value):
        self.file.write(value)
        self.file.flush()
        self.pending += value
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            g = re.fullmatch(r'Generated: (\d+/\d+)', line.strip())
            q = re.fullmatch(r'\[base quality\] (\d+/\d+)', line.strip())
            if g:
                print('[Evaluation] Question ' + g[1], file=self.terminal, flush=True)
            if q:
                print('[Quality] Question ' + q[1], file=self.terminal, flush=True)
        return len(value)


def marker(path, files):
    s0.write(path, {'files': {str(p.resolve()): s0.sha(p) for p in files}})


def check_marker(path, files):
    saved = s0.read(path)['files']
    if set(saved) != {str(p.resolve()) for p in files}:
        raise ValueError(f'Completion marker file set mismatch: {path}')
    for file in files:
        previous.verify_sha(file, saved[str(file.resolve())])


def archive(path):
    if path.exists():
        path.rename(path.with_name(path.name + '.incomplete-' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))


def generation_files(run):
    out = run / 'base'
    return [run/'prompt_config.yaml', out/'predictions.jsonl', out/'metrics.json', out/'generation_manifest.json']


def generate(run, cfg, records):
    import torch
    from load_base_model import load_model
    from evaluate.evaluate_base import make_prediction_rows
    from evaluate.common import score_predictions

    out = run / 'base'
    archive(out)
    cp = run / 'prompt_config.yaml'
    s0.write(cp, cfg)
    terminal = sys.stdout
    with (run/'generation.log').open('a', encoding='utf-8') as log:
        with redirect_stdout(ProgressLog(log, terminal)), redirect_stderr(log):
            model, tokenizer, gen = load_model(cfg)
            try:
                rows, seconds, memory = make_prediction_rows(model, tokenizer, gen, records, 1, cfg['prompt'])
                metrics = score_predictions(records, rows, 'val_106.json', 1)
                metrics.update(method='Base', generation_seconds=round(seconds, 3),
                               mean_generation_seconds_per_question=round(seconds/len(records), 4),
                               questions_per_second=round(len(records)/seconds, 3),
                               peak_gpu_memory_gb=round(memory, 3), retrieval=None, faithfulness_percent=None)
                s0.jsonl(out/'predictions.jsonl', rows)
                s0.write(out/'metrics.json', metrics)
                s0.write(out/'generation_manifest.json', {
                    'dataset_sha256': s0.sha(s0.VAL), 'expected_count': len(records),
                    'predictions_sha256': s0.sha(out/'predictions.jsonl'),
                    'base_config_sha256': s0.sha(cp), 'inference': cfg['inference'],
                    'prompt': cfg['prompt']})
            finally:
                del model, tokenizer, gen
                gc.collect()
                torch.cuda.empty_cache()
    marker(run/'generation_complete.json', generation_files(run))


def checked_generation(run, cfg, records):
    from evaluate.common import score_predictions, load_jsonl
    check_marker(run/'generation_complete.json', generation_files(run))
    out = run/'base'
    cp = run/'prompt_config.yaml'
    gm = s0.read(out/'generation_manifest.json')
    expected = {'dataset_sha256': s0.sha(s0.VAL), 'expected_count': len(records),
                'predictions_sha256': s0.sha(out/'predictions.jsonl'),
                'base_config_sha256': s0.sha(cp), 'inference': cfg['inference'], 'prompt': cfg['prompt']}
    if gm != expected or s0.read(cp) != cfg:
        raise ValueError('Base generation provenance mismatch')
    rows = load_jsonl(out/'predictions.jsonl')
    actual = score_predictions(records, rows, 'val_106.json', 1)
    saved = s0.read(out/'metrics.json')
    for k in ('token_f1_percent', 'exact_match_percent', 'abstention_rate_percent', 'false_refusal_rate_percent'):
        if actual[k] != saved[k]:
            raise ValueError(f'Base metric mismatch: {k}')
    result = {'token_f1_percent': saved['token_f1_percent'], 'answer_correctness_percent': None,
              'correct_refusal_rate_percent': saved['abstention_rate_percent'],
              'false_refusal_rate_percent': saved['false_refusal_rate_percent'],
              'exact_match_percent': saved['exact_match_percent'],
              'generation_seconds': saved['generation_seconds'], 'questions_per_second': saved['questions_per_second'],
              'inference_gpu_gib': saved['peak_gpu_memory_gb'], 'reference_contradiction_percent': None,
              'reference_unsupported_percent': None}
    result['youden_j_percent_points'] = previous.j(result)
    return result, rows


def judge(run):
    from evaluate import evaluate_stage2_quality as q
    out = run/'base/quality_evaluation'
    files = [out/n for n in ('metrics.json', 'cases.jsonl', 'review_required.jsonl')]
    done = run/'quality_complete.json'
    if done.exists():
        check_marker(done, files)
    else:
        archive(out)
        original_config, original_adapter, original_evaluator = s0.CONFIG, s0.adapter, q.Evaluator
        terminal = sys.stdout
        try:
            s0.CONFIG = run/'prompt_config.yaml'
            with (run/'quality.log').open('a', encoding='utf-8') as log:
                with redirect_stdout(ProgressLog(log, terminal)), redirect_stderr(log):
                    # Same judge prompts, claims criteria, and exact-input cache as previous LoRA runs.
                    previous.quality.worker(run, 'quality', 'base')
            marker(done, files)
        finally:
            s0.CONFIG, s0.adapter, q.Evaluator = original_config, original_adapter, original_evaluator
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    result = s0.read(out/'metrics.json')
    if result['dataset_sha256'] != s0.sha(s0.VAL) or result['predictions_sha256'] != s0.sha(run/'base/predictions.jsonl'):
        raise ValueError('Base quality provenance mismatch')
    return {k: result[k] for k in ('answer_correctness_percent', 'reference_contradiction_percent', 'reference_unsupported_percent')}


def paired(records, base_rows, lora_rows):
    from evaluate.common import validate_prediction_rows
    validate_prediction_rows(records, base_rows)
    validate_prediction_rows(records, lora_rows)
    counts = {'answerable': Counter(), 'impossible': Counter()}
    cases = []
    for rec, a, b in zip(records, base_rows, lora_rows):
        x, y = a['prediction'] == ABSTAIN, b['prediction'] == ABSTAIN
        key = ('refuse' if x else 'non_refuse') + '_to_' + ('refuse' if y else 'non_refuse')
        counts['impossible' if rec['is_impossible'] else 'answerable'][key] += 1
        cases.append({'id': rec['id'], 'question': rec['question'], 'is_impossible': rec['is_impossible'],
                      'reference_answer': rec['answer'], 'base_prediction': a['prediction'],
                      'lora_prediction': b['prediction'], 'base_refused': x, 'lora_refused': y,
                      'base_empty': not a['prediction'].strip(), 'lora_empty': not b['prediction'].strip(),
                      'transition': key})
    keys = ('refuse_to_refuse', 'refuse_to_non_refuse', 'non_refuse_to_refuse', 'non_refuse_to_non_refuse')
    return {k: {n: v[n] for n in keys} for k,v in counts.items()}, cases


def report(root, base, lora, transitions, complete):
    delta = {k: round(lora[k]-base[k], 4) if base.get(k) is not None and lora.get(k) is not None else None for k in previous.KEYS}
    s0.write(root/'comparison.json', {'stage':3, 'test':5, 'base':base, 'lora':lora,
             'delta_lora_minus_base':delta, 'paired_refusal_transitions':transitions,
             'complete':complete, 'training':'NOT RUN', 'qlora':'NOT RUN', 'human_review_required':True,
             'notes':['Same detailed system prompt and greedy inference settings; fixed trained E adapter versus no adapter.',
                      'Refusal means exact refusal string; non-refusal is not necessarily a valid answer.',
                      'Quality judged by the same base model; human review required. Repeated validation tuning is not independent testing.']})
    def fmt(x):
        return 'N/A' if x is None else f'{x:.4f}'.rstrip('0').rstrip('.')
    lines = ['[Stage 3 Test5 Base vs LoRA Comparison]', f'{"Metric":36} {"A_Base":>15} {"B_LoRA_E":>15} {"LoRA-Base":>15}']
    for k in previous.KEYS:
        lines.append(f'{k:36} {fmt(base.get(k)):>15} {fmt(lora.get(k)):>15} {fmt(delta[k]):>15}')
    lines += ['[Paired Refusal Transitions: Base -> LoRA]',
              f'{"Transition":32} {"Answerable (77)":>18} {"Impossible (29)":>18}']
    for k in transitions['answerable']:
        lines.append(f'{k:32} {transitions["answerable"][k]:>18} {transitions["impossible"][k]:>18}')
    lines += [f'Complete: {complete}', 'New training: NONE | LoRA result: REUSED | QLoRA: NOT RUN',
              'Automated quality verdicts require human review.', f'Results: {root}']
    text = '\n'.join(lines) + '\n'
    (root/'comparison.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Stage 3 Test5: same-prompt Base vs LoRA diagnostic')
    parser.add_argument('--parent', type=Path, default=previous.E_ROOT)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--generation-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.resume:
        root = args.resume.resolve()
        if root.parent != RUNS.resolve():
            parser.error('Invalid resume directory')
        old = s0.read(root/'manifest.json')
        args.parent = Path(old['parent_run'])
    records, adapter, cfg, lora, manifest = previous.load_baseline(args.parent)
    manifest.update(stage=3, test=5, experiment='base_vs_lora',
                    runner_sha256=s0.sha(Path(__file__).resolve()),
                    loader_sha256=s0.sha(Path(previous.__file__).resolve()))
    if args.resume:
        if old != manifest:
            raise ValueError('Resume blocked: provenance changed')
    else:
        root = RUNS/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        root.mkdir(parents=True, exist_ok=False)
        s0.write(root/'manifest.json', manifest)
    print(f'[Stage 3 Test5] Run: {root}', flush=True)
    if args.prepare_only:
        print('Preparation: OK (no GPU work)', flush=True)
        return
    run = root/'experiments'/NAME
    run.mkdir(parents=True, exist_ok=True)
    try:
        print('[Experiment] 1/1 | A_Base (B_LoRA_E reused)', flush=True)
        if not (run/'generation_complete.json').exists():
            generate(run, cfg, records)
        base, base_rows = checked_generation(run, cfg, records)
        from evaluate.common import load_jsonl, score_predictions
        lora_path = Path(manifest['baseline'])/'lora/predictions.jsonl'
        lora_rows = load_jsonl(lora_path)
        scores = score_predictions(records, lora_rows, 'val_106.json', 1)
        for k, n in (('token_f1_percent','token_f1_percent'), ('exact_match_percent','exact_match_percent'),
                     ('abstention_rate_percent','correct_refusal_rate_percent'), ('false_refusal_rate_percent','false_refusal_rate_percent')):
            if scores[k] != lora[n]:
                raise ValueError(f'LoRA baseline metric mismatch: {k}')
        transitions, cases = paired(records, base_rows, lora_rows)
        s0.jsonl(root/'paired_cases.jsonl', cases)
        # Save preliminary results before the expensive judge, without printing duplicate tables.
        s0.write(root/'basic_metrics.json', {'base':base, 'lora':lora, 'paired_refusal_transitions':transitions})
        print(f'[Screen] Base | F1={base["token_f1_percent"]}% | Correct refusal={base["correct_refusal_rate_percent"]}% | False refusal={base["false_refusal_rate_percent"]}% | J={base["youden_j_percent_points"]}%p', flush=True)
        complete = not args.generation_only or (run/'quality_complete.json').exists()
        if complete:
            base.update(judge(run))
        previous.verify_sha(s0.VAL, manifest['dataset_sha256'])
        for name, digest in manifest['adapter_sha256'].items():
            previous.verify_sha(adapter/name, digest)
        for name, digest in manifest['model_files'].items():
            previous.verify_sha(s0.BASE/name, digest)
        report(root, base, lora, transitions, complete)
    except BaseException:
        print(f'Resume: python IBMTechSupport/scripts/finetuning/{Path(__file__).name} --resume "{root}"', flush=True)
        raise


if __name__ == '__main__':
    main()
