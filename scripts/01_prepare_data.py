#!/usr/bin/env python
"""
01_prepare_data.py - 数据清洗、去重、分层抽样与提示词构建（v4）

v4 修订（第三轮评审 R3）：
- R3-01 提示词改用模型官方 chat template（`data.prompt_style: chat`），
  zero-shot 400 条实测 82.5%，纯文本 + Category: 为 75.0%；
  同时保证 completion 之后紧跟的 EOS(`<|im_end|>`) 与聊天格式一致。
- R3-02 文本规范化：AG News 原始 text 约 11% 含 CSV 转义残留 "\\"，
  替换为空格并压缩空白；`#NN;` 形式的 HTML 实体残留只统计不修改。
- R3-03 防泄漏：丢弃与测试集归一化文本相同的训练池样本，并在 metadata 中
  记录 train/val/test 的重叠检查结果（实测为 0）。
- R3-04 长度兜底：逐条按 `model.max_seq_length` 收缩新闻正文，
  保证 训练/评估时不再发生右截断（原数据 test 有 2 条、train_2000/4000 各有 1 条超限，
  右截断会切掉 completion，使该样本对 loss/评估无贡献）。
- R3-05 数据审计写入 metadata：类别分布、token 分位数、去重计数、
  子集嵌套与重叠检查，供报告 §3.1 直接引用。

用法：
    conda activate lora-project
    python scripts/01_prepare_data.py
"""
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'

INSTRUCTION = (
    "You are a news classifier. Classify the following news into exactly one "
    "of these categories: {labels}.\n\n"
    "News: {text}\n\n"
    "Answer with only the category name."
)
CATEGORY_CUE = "\nCategory:"
HTML_ENTITY_RESIDUE = re.compile(r'#(\d{1,3});')


def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def _entity_to_char(match):
    """把 AG News 里丢掉 '&' 的 HTML 实体（如 #36; → $）还原。"""
    code = int(match.group(1))
    return chr(code) if 32 <= code < 127 else match.group(0)


def clean_text(text):
    """规范化 AG News 文本：反斜杠转义残留→空格、还原残留 HTML 实体、压缩空白。"""
    t = str(text).replace('\\', ' ')
    t = HTML_ENTITY_RESIDUE.sub(_entity_to_char, t)
    return re.sub(r'\s+', ' ', t).strip()


def norm_key(text):
    """用于跨 split 去重比较的归一化键（不改写实际文本）。"""
    t = unicodedata.normalize('NFKC', text).lower()
    t = re.sub(r'[^a-z0-9]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def build_prompt(text, class_names, tokenizer=None, style='plain'):
    """构造分类提示词；训练与评估共用同一函数（唯一真源）。"""
    body = INSTRUCTION.format(labels=', '.join(class_names), text=text)
    if style == 'chat':
        if tokenizer is None:
            raise ValueError("prompt_style=chat 需要传入 tokenizer")
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': body}],
            tokenize=False, add_generation_prompt=True,
        )
    if style == 'plain':
        return body + CATEGORY_CUE
    raise ValueError(f'未知 prompt_style: {style}')


def prompt_version(style):
    return 'v4_chat_template' if style == 'chat' else 'v4_plain_category'


def stratified_sample(dataset, n_total, seed, label_col='label'):
    """按类别比例分层抽样，返回索引列表（同一 seed 下各档子集互为前缀，保证嵌套）。"""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, label in enumerate(dataset[label_col]):
        by_class[label].append(i)

    n_classes = len(by_class)
    per_class = n_total // n_classes
    remainder = n_total % n_classes
    selected = []
    for c in sorted(by_class.keys()):
        idxs = by_class[c][:]
        rng.shuffle(idxs)
        take = per_class + (1 if remainder > 0 else 0)
        remainder -= 1 if remainder > 0 else 0
        selected.extend(idxs[:take])
    rng.shuffle(selected)
    return selected


def fit_prompt(text, completion, tokenizer, class_names, style, max_seq_length):
    """确保 prompt + completion + eos 不超过 max_seq_length。

    超出时收缩新闻正文（保留开头，信息量最高），而不是在训练/评估阶段右截断——
    右截断会把 completion 切掉，使该样本标签全为 -100（对 loss 无贡献），
    评估时则让最后一个 token 落在正文中间，判定位置错位。
    """
    completion_ids = tokenizer.encode(' ' + completion, add_special_tokens=False)
    prompt, n_tok, truncated = '', 0, False
    for _ in range(5):
        prompt = build_prompt(text, class_names, tokenizer, style)
        n_tok = (len(tokenizer.encode(prompt, add_special_tokens=False))
                 + len(completion_ids) + 1)
        if n_tok <= max_seq_length:
            return prompt, n_tok, truncated
        overflow = n_tok - max_seq_length + 4
        ids = tokenizer.encode(text, add_special_tokens=False)
        keep = max(8, len(ids) - overflow)
        trunc = tokenizer.decode(ids[:keep], skip_special_tokens=True).strip()
        if trunc == text:
            break
        text, truncated = trunc, True
    return prompt, n_tok, truncated


def token_stats(tokenizer, examples):
    lens = sorted(len(tokenizer.encode(e['prompt'], add_special_tokens=False))
                  + len(tokenizer.encode(' ' + e['completion'], add_special_tokens=False)) + 1
                  for e in examples)
    n = len(lens)
    return {
        'min': lens[0], 'median': lens[n // 2],
        'p95': lens[min(n - 1, int(n * 0.95))], 'max': lens[-1],
        'mean': round(sum(lens) / n, 1),
    }


def main():
    cfg = load_config()
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    seed = cfg['experiment']['seed']
    class_names = data_cfg['class_names']
    style = data_cfg.get('prompt_style', 'plain')
    clean_on = data_cfg.get('clean_text', True)
    drop_overlap = data_cfg.get('drop_test_overlap', True)
    max_seq_length = int(model_cfg['max_seq_length'])
    raw_dir = ROOT / data_cfg['raw_dir']
    out_dir = ROOT / cfg['paths']['processed_data_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(ROOT / model_cfg['local_path'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print("=" * 60)
    print("加载 AG News 原始数据")
    print("=" * 60)
    dataset = load_dataset(
        'parquet',
        data_files={
            'train': str(raw_dir / 'data' / 'train-00000-of-00001.parquet'),
            'test': str(raw_dir / 'data' / 'test-00000-of-00001.parquet'),
        },
    )
    print(f"训练集: {len(dataset['train'])} 条 | 测试集: {len(dataset['test'])} 条")

    # ---------- 文本清洗 + 审计 ----------
    audit = {
        'prompt_style': style,
        'prompt_version': prompt_version(style),
        'max_seq_length': max_seq_length,
        'clean_text': clean_on,
        'drop_test_overlap': drop_overlap,
        'raw_rows': {'train': len(dataset['train']), 'test': len(dataset['test'])},
    }
    clean_hits = {'train': 0, 'test': 0}
    entity_hits = {'train': 0, 'test': 0}
    empty_after_clean = 0
    for split in ('train', 'test'):
        texts = []
        for t in dataset[split]['text']:
            if '\\' in t:
                clean_hits[split] += 1
            entity_hits[split] += len(HTML_ENTITY_RESIDUE.findall(t))
            nt = clean_text(t) if clean_on else str(t)
            if not nt:
                empty_after_clean += 1
            texts.append(nt)
        dataset[split] = dataset[split].remove_columns(['text']).add_column('text', texts)
    audit['backslash_fixed'] = clean_hits
    audit['html_entity_restored'] = entity_hits
    audit['empty_after_clean'] = empty_after_clean
    print(f"文本清洗：含反斜杠残留 train {clean_hits['train']} / test {clean_hits['test']} 条；"
          f"还原 HTML 实体残留 {entity_hits['train'] + entity_hits['test']} 处")

    # ---------- 防泄漏：训练池去掉与测试集归一化后相同的文本 ----------
    test_keys = {norm_key(t) for t in dataset['test']['text']}
    train_keys = [norm_key(t) for t in dataset['train']['text']]
    leaked = [i for i, k in enumerate(train_keys) if k in test_keys]
    audit['train_rows_matching_test_normalized'] = len(leaked)
    print(f"训练集与测试集归一化重叠：{len(leaked)} 条 "
          f"({'已从训练池剔除' if drop_overlap else '保留'}）")

    leaked_set = set(leaked)
    keep_rows = [i for i in range(len(dataset['train'])) if i not in leaked_set]
    train_pool_all = dataset['train'].select(keep_rows) if drop_overlap else dataset['train']
    audit['train_pool_rows'] = len(train_pool_all)

    # ---------- 分层抽样：验证集，然后训练池，再四档子集 ----------
    val_indices = stratified_sample(train_pool_all, data_cfg['val_size'], seed=seed)
    val_set = train_pool_all.select(val_indices)
    val_index_set = set(val_indices)
    remaining = [i for i in range(len(train_pool_all)) if i not in val_index_set]
    train_pool = train_pool_all.select(remaining)
    print(f"验证集: {len(val_set)} 条，剩余训练池: {len(train_pool)} 条")

    files = {}
    truncated_counts = {}
    for size in data_cfg['train_subset_sizes']:
        indices = stratified_sample(train_pool, size, seed=seed)
        subset = train_pool.select(indices)
        examples, n_trunc = [], 0
        for ex in subset:
            answer = class_names[ex['label']]
            prompt, _, truncated = fit_prompt(
                ex['text'], answer, tokenizer, class_names, style, max_seq_length)
            n_trunc += int(truncated)
            examples.append({'prompt': prompt, 'completion': answer,
                             'label': int(ex['label'])})
        files[f'train_{size}'] = examples
        truncated_counts[f'train_{size}'] = n_trunc
        with open(out_dir / f'train_{size}.json', 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"[OK] train_{size}.json  ({len(examples)} 条，正文收缩 {n_trunc} 条)")

    val_examples, val_trunc = [], 0
    for ex in val_set:
        answer = class_names[ex['label']]
        prompt, _, truncated = fit_prompt(
            ex['text'], answer, tokenizer, class_names, style, max_seq_length)
        val_trunc += int(truncated)
        val_examples.append({'prompt': prompt, 'completion': answer,
                             'label': int(ex['label'])})
    files['val'] = val_examples
    truncated_counts['val'] = val_trunc
    with open(out_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_examples, f, ensure_ascii=False, indent=2)

    # R4（外部审稿 P0）：AG News 训练池内部有重复文本。先抽 val、再从剩余池抽训练子集时，
    # 同一文本可能同时落入 val 与某个训练子集。这里检查并用池中未被任何集合占用的样本替换。
    def _prompt_key(prompt):
        return norm_key(prompt.split('News:')[-1].split('\n\nAnswer')[0])

    subset_keys = set()
    for name, ex in files.items():
        if name.startswith('train_'):
            subset_keys |= {_prompt_key(e['prompt']) for e in ex}
    bad_val = [i for i, e in enumerate(val_examples)
               if _prompt_key(e['prompt']) in subset_keys]
    if bad_val:
        val_used = {_prompt_key(e['prompt']) for e in val_examples}
        cands = []
        for ex in train_pool:
            txt = clean_text(ex['text'])
            k = norm_key(txt)
            if not k or k in subset_keys or k in val_used or k in test_keys:
                continue
            cands.append((txt, int(ex['label'])))
        rng2 = random.Random(seed + 7)
        rng2.shuffle(cands)
        pick = iter(cands)
        for idx in bad_val:
            while True:
                txt, label = next(pick)
                if norm_key(txt) not in val_used:
                    break
            prompt, _, truncated = fit_prompt(
                txt, class_names[label], tokenizer, class_names, style, max_seq_length)
            val_examples[idx] = {'prompt': prompt, 'completion': class_names[label],
                                 'label': label}
            val_used.add(norm_key(txt))
            val_trunc += int(truncated) - 0
        files['val'] = val_examples
        with open(out_dir / 'val.json', 'w', encoding='utf-8') as f:
            json.dump(val_examples, f, ensure_ascii=False, indent=2)
        print(f"[FIX] val 与训练子集重叠 {len(bad_val)} 条，已用池中样本替换")
    audit['val_subset_overlap_repaired'] = len(bad_val)

    test_examples, test_trunc = [], 0
    for ex in dataset['test']:
        answer = class_names[ex['label']]
        prompt, _, truncated = fit_prompt(
            ex['text'], answer, tokenizer, class_names, style, max_seq_length)
        test_trunc += int(truncated)
        test_examples.append({'prompt': prompt, 'completion': answer,
                              'label': int(ex['label'])})
    files['test'] = test_examples
    truncated_counts['test'] = test_trunc
    with open(out_dir / 'test.json', 'w', encoding='utf-8') as f:
        json.dump(test_examples, f, ensure_ascii=False, indent=2)
    print(f"[OK] val.json / test.json（正文收缩 val {val_trunc} / test {test_trunc} 条）")

    # ---------- 审计：分布、长度、去重、嵌套、重叠 ----------
    audit['files'] = {}
    for name, examples in files.items():
        dist = Counter(e['label'] for e in examples)
        prompts = [e['prompt'] for e in examples]
        audit['files'][name] = {
            'rows': len(examples),
            'class_dist': {class_names[k]: v for k, v in sorted(dist.items())},
            'unique_prompts': len(set(prompts)),
            'duplicate_prompts': len(prompts) - len(set(prompts)),
            'tokens': token_stats(tokenizer, examples),
            'truncated_texts': truncated_counts[name],
        }
    def keyset(name):
        return {norm_key(e['prompt']) for e in files[name]}
    audit['checks'] = {
        'val_vs_test_overlap': len(keyset('val') & keyset('test')),
        'train_subset_vs_test_overlap': {
            name: len(keyset(name) & keyset('test'))
            for name in files if name.startswith('train_')
        },
        'val_vs_train_subset_overlap': {
            name: len(keyset('val') & keyset(name))
            for name in files if name.startswith('train_')
        },
        'nested_subsets': {
            'train_500_in_train_1000': keyset('train_500') <= keyset('train_1000'),
            'train_1000_in_train_2000': keyset('train_1000') <= keyset('train_2000'),
            'train_2000_in_train_4000': keyset('train_2000') <= keyset('train_4000'),
        },
        'samples_over_max_seq_length': {
            name: sum(1 for e in ex
                      if len(tokenizer.encode(e['prompt'], add_special_tokens=False))
                      + len(tokenizer.encode(' ' + e['completion'], add_special_tokens=False))
                      + 1 > max_seq_length)
            for name, ex in files.items()
        },
    }

    label_token_ids = {
        name: tokenizer.encode(' ' + name, add_special_tokens=False)[0]
        for name in class_names
    }
    meta = {
        'seed': seed,
        'class_names': class_names,
        'subset_sizes': data_cfg['train_subset_sizes'],
        'val_size': len(val_examples),
        'test_size': len(test_examples),
        'prompt_style': style,
        'prompt_version': prompt_version(style),
        'prompt_template': build_prompt('<TEXT>', class_names, tokenizer, style),
        'label_token_ids': label_token_ids,
        'eos_token_id': tokenizer.eos_token_id,
        'eos_token': tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
        'source': 'fancyzhx/ag_news (Apache-2.0 / CC0)',
        'audit': audit,
    }
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("数据准备完成")
    print("=" * 60)
    print(f"提示词版本: {meta['prompt_version']}（{style}）")
    print(f"标签首 token: {label_token_ids}")
    print(f"EOS: {meta['eos_token']} ({meta['eos_token_id']})")
    print(f"抽查（train_500[0]）:\n{files['train_500'][0]['prompt'][-260:]}")
    print(f"completion={files['train_500'][0]['completion']!r}")
    print(f"长度（train_500）: {audit['files']['train_500']['tokens']}")
    print(f"审计文件: {out_dir / 'metadata.json'}")


if __name__ == '__main__':
    main()
