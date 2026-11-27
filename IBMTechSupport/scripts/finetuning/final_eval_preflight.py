"""Read-only repository inventory and post-freeze leakage audit (Python 3.10+)."""
import argparse
import ast
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(root, names):
    result = {}
    for name in names:
        path = (root / name).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        paths = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        if not paths:
            raise ValueError(f'Empty asset: {path}')
        for p in paths:
            if p.name.lower().startswith('test_clean'):
                raise ValueError('Do not include test_clean in the freeze assets')
            result[str(p)] = sha(p)
    return result


def load(path):
    if path.suffix == '.jsonl':
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    else:
        rows = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list):
        raise ValueError(f'Expected array: {path}')
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get('id'), (str, int)) or isinstance(r.get('id'), bool) or not str(r['id']).strip() or not isinstance(r.get('question'), str) or not r['question'].strip() or type(r.get('is_impossible')) is not bool:
            raise ValueError(f'Invalid ID/question/is_impossible: {path}')
    return rows


def normalize(question):
    # Independent conservative duplicate check; not asserted to be project normalizer.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', question).casefold()).strip()


def duplicates(items):
    return {k: v for k, v in Counter(items).items() if v > 1}


def save(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(path)


def inventory(root):
    report = {'scripts': [], 'data_paths': [], 'config_paths': [], 'adapters': [], 'records': []}
    for p in sorted((root / 'scripts').rglob('*.py')):
        if not re.search(r'train|lora|eval|composition|rag|judge|prompt|finetun', p.name, re.I):
            continue
        try:
            tree = ast.parse(p.read_text(encoding='utf-8-sig'))
            args = sorted({a.value for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument' for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)})
            report['scripts'].append({'path': str(p), 'cli_arguments': args})
        except (SyntaxError, UnicodeError) as e:
            report['scripts'].append({'path': str(p), 'error': str(e)})
    for sub, key, patterns in [('data/processed','data_paths',('*.json','*.jsonl')), ('configs','config_paths',('*.yaml','*.yml','*.json')), ('models/finetuning','adapters',('adapter_config.json',)), ('evaluate_results/106','records',('comparison.json','*train_metrics.json','training_complete.json'))]:
        for pattern in patterns:
            report[key].extend(str(p) for p in sorted((root/sub).rglob(pattern)))
    return report


def audit(root, lock_path, test):
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if lock.get('kind') != 'final-eval-freeze-v1':
        raise ValueError('Wrong lock format')
    for name, digest in lock['sha256'].items():
        if sha(Path(name)) != digest:
            raise ValueError(f'Changed frozen asset: {name}')
    test_path = (root / test).resolve()
    if test_path.name != 'test_clean.json':
        raise ValueError('Expected test_clean.json')
    test_rows = load(test_path)
    test_ids = [str(r['id']) for r in test_rows]
    test_qs = [normalize(r['question']) for r in test_rows]
    result = {'test_path': str(test_path), 'test_sha256': sha(test_path), 'total': len(test_rows), 'answerable': sum(not r['is_impossible'] for r in test_rows), 'impossible': sum(r['is_impossible'] for r in test_rows), 'test_duplicate_ids': duplicates(test_ids), 'test_duplicate_questions': duplicates(test_qs), 'normalization': 'NFKC + casefold + collapse whitespace; independent check', 'train_comparisons': []}
    for name in lock['train_paths']:
        train = load(Path(name))
        ids = {str(r['id']) for r in train}
        qs = {normalize(r['question']) for r in train}
        result['train_comparisons'].append({'train_path': name, 'train_count': len(train), 'id_overlap_test_ids': [str(r['id']) for r in test_rows if str(r['id']) in ids], 'question_overlap_test_ids': [str(r['id']) for r in test_rows if normalize(r['question']) in qs]})
    result['passed'] = not (result['test_duplicate_ids'] or result['test_duplicate_questions'] or any(x['id_overlap_test_ids'] or x['question_overlap_test_ids'] for x in result['train_comparisons']))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='IBMTechSupport')
    sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('inventory')
    f = sub.add_parser('freeze')
    f.add_argument('--train', nargs='+', required=True, help='Actual selected train snapshots; both input types')
    f.add_argument('--asset', nargs='+', required=True, help='Verified config, scripts, model/adapter assets, prompts and retrieval index files')
    a = sub.add_parser('audit-test')
    a.add_argument('--lock', required=True)
    a.add_argument('--test', default='data/processed/test_clean.json')
    args = p.parse_args()
    root = Path(args.root).resolve()
    if not (root / 'scripts').is_dir():
        p.error(f'Project scripts not found: {root}')
    out = root / 'evaluate_results' / 'final_lora_qlora'
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.mode == 'inventory':
        save(out / f'inventory_{stamp}.json', inventory(root))
    elif args.mode == 'freeze':
        digest = files(root, args.train + args.asset)
        trains = [str((root / name).resolve()) for name in args.train]
        for name in trains:
            if not Path(name).is_file():
                raise ValueError(f'Train snapshot must be a file: {name}')
            load(Path(name))
        save(out / f'freeze_{stamp}.json', {'kind': 'final-eval-freeze-v1', 'train_paths': trains, 'sha256': digest, 'note': 'Operator must verify chosen settings and complete asset list before freeze.'})
    else:
        result = audit(root, Path(args.lock), args.test)
        save(out / f'leakage_audit_{stamp}.json', result)
        if not result['passed']:
            raise SystemExit('Overlap or duplicate detected; stop final evaluation.')


if __name__ == '__main__':
    main()
