"""Prepare the existing LoRA trainer for the frozen 106-case validation split.
Save in IBMTechSupport/scripts/. Run from any working directory.
Does not train or overwrite adapters.
"""
from pathlib import Path
import ast
import json
import unicodedata
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'scripts/train_lora.py'
BASE_CONFIG = ROOT / 'configs/base/v1_106.yaml'
DATA = ROOT / 'data/processed'
ABSTAIN = 'INSUFFICIENT_INFORMATION'


def read_records(path):
    rows = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'Invalid dataset: {path}')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate IDs: {path}')
    for r in rows:
        if not isinstance(r['question'], str) or not r['question'].strip():
            raise ValueError(f'Invalid question: {r["id"]}')
        if type(r['is_impossible']) is not bool:
            raise ValueError(f'Invalid answerability: {r["id"]}')
        if not r['is_impossible'] and (not isinstance(r['answer'], str) or not r['answer'].strip()):
            raise ValueError(f'Invalid answer: {r["id"]}')
    return rows


def question_key(q):
    return ' '.join(unicodedata.normalize('NFKC', q).casefold().split())


def assignment(source, name, expression, *, insert=False):
    tree = ast.parse(source)
    matches = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
               and any(isinstance(t, ast.Name) and t.id == name
                       for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if len(matches) > 1:
        raise ValueError(f'Ambiguous definition: {name}')
    if matches:
        node = matches[0]
        if isinstance(node, ast.Assign) and len(node.targets) != 1:
            raise ValueError(f'Multiple assignment targets: {name}')
        lines = source.splitlines(keepends=True)
        lines[node.lineno - 1:node.end_lineno] = [f'{name} = {expression}\n']
        return ''.join(lines)
    if not insert:
        raise ValueError(f'No assignment found: {name}')
    # Insert before the first function/class; existing imports and constants stay intact.
    first = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    lines = source.splitlines(keepends=True)
    lines[first.lineno - 1:first.lineno - 1] = [f'{name} = {expression}\n\n']
    return ''.join(lines)


def prepare_source(source, prompt):
    source = assignment(source, 'VAL_PATH', 'DATA_DIR / "val_106.json"')
    source = assignment(source, 'SYSTEM_PROMPT', repr(prompt), insert=True)
    source = assignment(source, 'ABSTAIN', repr(ABSTAIN), insert=True)
    # Recent Transformers can return a BatchEncoding by default; the existing
    # QADataset concatenates prompt token IDs with a plain Python list.
    old = 'tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)'
    new = 'tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)'
    if source.count(old) == 1:
        source = source.replace(old, new)
    elif source.count(old) > 1:
        raise ValueError('Multiple matching chat-template calls; inspect the local trainer')
    compile(source, str(TRAINER), 'exec')
    return source


def main():
    cfg = yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8-sig'))
    if cfg['dataset'] != {'path': 'data/processed/val_106.json', 'expected_count': 106}:
        raise ValueError('Unexpected Base validation configuration')
    if cfg['prompt']['user_template'] != '{question}':
        raise ValueError('Unexpected Base user prompt')
    train = read_records(DATA / 'train_clean.json')
    val = read_records(DATA / 'val_106.json')
    if (len(val), sum(not r['is_impossible'] for r in val)) != (106, 77):
        raise ValueError('Expected 106 validation cases: 77 answerable and 29 impossible')
    ids = {r['id'] for r in train} & {r['id'] for r in val}
    questions = {question_key(r['question']) for r in train} & {question_key(r['question']) for r in val}
    if ids or questions:
        raise ValueError(f'Train/validation overlap: {len(ids)} IDs, {len(questions)} normalized questions. Inspect before training.')
    original = TRAINER.read_text(encoding='utf-8-sig')
    updated = prepare_source(original, cfg['prompt']['system'])
    if updated != original:
        TRAINER.write_text(updated, encoding='utf-8')
    print('Training samples:', len(train))
    print('Validation samples:', len(val))
    print('Train/validation ID and normalized question overlap: 0')
    print('Trainer:', TRAINER)
    print('Preparation: OK')
    adapter = ROOT / 'models/Qwen2.5-1.5B-Instruct-LoRA'
    if adapter.exists() and any(adapter.iterdir()):
        print('WARNING: Full LoRA adapter directory already exists. Preserve previous results before full training.')


if __name__ == '__main__':
    main()
