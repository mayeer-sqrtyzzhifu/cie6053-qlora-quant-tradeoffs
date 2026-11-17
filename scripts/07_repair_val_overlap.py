#!/usr/bin/env python
"""07_repair_val_overlap.py — 修复 val 与训练子集共享同一条 AG News 文本的问题。

背景（外部审稿 P0-1）：AG News 训练池内部存在重复文本，01_prepare_data 先按行抽样
val、再对剩余行抽样训练子集，导致同一文本可能同时落入 val 与 train_4000（实测 1/1000）。
该问题不影响任何测试集结果，但会污染"验证集独立"的声明。

本脚本：
1. 从 raw train 重建清洗后的候选池（与 01 的 clean_text/norm_key 一致）；
2. 找出与任一训练子集文本相同的 val 样本，用池中未被任何集合占用的样本替换；
3. 重写 data/processed/val.json；
4. 在 metadata.json 的 audit.checks 中写入 val_vs_train_subset_overlap（应全为 0）。

用法：python scripts/07_repair_val_overlap.py
"""
import json
import random
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'


def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _entity_to_char(m):
    try:
        return chr(int(m.group(1)))
    except (TypeError, ValueError):
        return m.group(0)


def clean_text(text):
    import html as _html
    import re
    text = str(text).replace('\\', ' ')
    text = _html.unescape(text)
    text = re.sub(r'#(\d+);', _entity_to_char, text)
    return ' '.join(text.split())


def norm_key(text):
    import re
    import unicodedata
    t = unicodedata.normalize('NFKC', str(text)).lower()
    return re.sub(r'[^a-z0-9]+', '', t)


def main():
    cfg = load_config()
    raw = ROOT / 'data' / 'raw' / 'ag_news'
    train_df = pd.read_parquet(next(raw.glob('data/train-*.parquet')))
    test_df = pd.read_parquet(next(raw.glob('data/test-*.parquet')))
    proc = ROOT / cfg['paths']['processed_data_dir']
    val = json.loads((proc / 'val.json').read_text(encoding='utf-8'))
    subsets = {n: json.loads((proc / f'train_{n}.json').read_text(encoding='utf-8'))
               for n in (500, 1000, 2000, 4000)}
    class_names = cfg['data']['class_names']
    prompt_style = cfg['data'].get('prompt_style', 'chat')

    # 与 01 一致的文本提取：raw parquet 的 text 列
    test_keys = {norm_key(clean_text(t)) for t in test_df['text']}
    subset_keys = set()
    for rows in subsets.values():
        for r in rows:
            subset_keys.add(norm_key(r['prompt'].split('News:')[-1]
                                     .split('\n\nAnswer')[0]))
    val_keys = {norm_key(r['prompt'].split('News:')[-1].split('\n\nAnswer')[0])
                for r in val}

    def key_of_prompt(p):
        body = p.split('News:')[-1].split('\n\nAnswer')[0]
        return norm_key(body)

    candidates = []
    seen = set()
    for _, row in train_df.iterrows():
        txt = clean_text(row['text'])
        k = norm_key(txt)
        if not k or k in seen:
            continue
        seen.add(k)
        if k in test_keys or k in subset_keys or k in val_keys:
            continue
        candidates.append((txt, int(row['label'])))

    print(f'候选替换样本: {len(candidates)}')

    # 当前 val 中与训练子集重叠的条目
    bad = [(i, r) for i, r in enumerate(val) if key_of_prompt(r['prompt']) in subset_keys]
    print(f'val 与训练子集重叠: {len(bad)} 条')
    if not bad:
        print('无需修复')
    else:
        # 需要 tokenizer 重建 chat prompt
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(ROOT / cfg['model']['local_path']), local_files_only=True)
        system = ('You are a news classifier. Classify the following news into exactly '
                  'one of these categories: ' + ', '.join(class_names) + '.')
        rng = random.Random(cfg['experiment']['seed'] + 7)
        rng.shuffle(candidates)
        pick = iter(candidates)
        for idx, old in bad:
            while True:
                txt, label = next(pick)
                k = norm_key(txt)
                if k in val_keys:
                    continue
                break
            messages = [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': f'News: {txt}\n\nAnswer with only the category name.'},
            ]
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
            val[idx] = {'prompt': prompt, 'completion': class_names[label], 'label': label}
            val_keys.add(k)
            print(f'替换: label={label} 文本前 60 字: {txt[:60]}')
        (proc / 'val.json').write_text(
            json.dumps(val, ensure_ascii=False, indent=2), encoding='utf-8')
        print('已重写 val.json')

    # 重算重叠并写入 audit
    val_keys2 = {key_of_prompt(r['prompt']) for r in val}
    overlaps = {}
    for n, rows in subsets.items():
        sk = {key_of_prompt(r['prompt']) for r in rows}
        overlaps[f'train_{n}'] = len(val_keys2 & sk)
    meta_path = proc / 'metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta.setdefault('audit', {}).setdefault('checks', {})['val_vs_train_subset_overlap'] = overlaps
    meta['audit'].setdefault('repairs', []).append({
        'date': '2026-11-18',
        'issue': 'val 与 train_4000 共享 1 条相同 AG News 文本（原始训练池内部重复）',
        'action': '用池中未被任何集合占用的样本替换该 val 条目；重写 val.json',
        'result': overlaps,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('audit 写入 metadata.json：', overlaps)


if __name__ == '__main__':
    main()
