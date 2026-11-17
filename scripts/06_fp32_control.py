#!/usr/bin/env python
"""06_fp32_control.py — 零样本非单调性的 fp32/bf16/nf4 对照实验（外审 P1-3）。

背景：报告曾用"400 条上 bf16 0.7500 / fp32 0.7425 / nf4 0.8225"来解释
"零样本准确率随基座精度非单调、但不是 bf16 精度不足"，但没有留下可复核的 artifact。
本脚本在同一批 400 条测试样本、同一提示词、同一判定逻辑下重做三组评估并落盘：

    outputs/results/zero_shot_control400_fp32_eval.json
    outputs/results/zero_shot_control400_bf16_eval.json
    outputs/results/zero_shot_control400_nf4_eval.json

用法：python scripts/06_fp32_control.py [--limit 400]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'


def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def label_token_ids(tokenizer, class_names):
    ids = [tokenizer.encode(' ' + n, add_special_tokens=False)[0] for n in class_names]
    if len(set(ids)) != len(ids):
        raise ValueError(f'类别首 token 冲突: {dict(zip(class_names, ids))}')
    return ids


def build_model(cfg, mode):
    """mode: fp32 / bf16 / nf4。"""
    model_path = str(ROOT / cfg['model']['local_path'])
    kwargs = dict(device_map=None)
    if mode == 'fp32':
        kwargs['dtype'] = torch.float32
    elif mode == 'bf16':
        kwargs['dtype'] = torch.bfloat16
    elif mode == 'nf4':
        kwargs['dtype'] = torch.bfloat16
        kwargs['device_map'] = 'auto'
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(mode)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, **kwargs)
    if kwargs.get('device_map') is None:
        model = model.to('cuda')
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, tokenizer, examples, lab_ids, max_len, batch_size, class_names):
    preds, gold = [], []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        enc = tokenizer([e['prompt'] for e in batch], return_tensors='pt',
                        padding=True, truncation=True, max_length=max_len).to('cuda')
        logits = model(**enc).logits[:, -1, :]
        preds.extend(logits[:, lab_ids].argmax(dim=-1).tolist())
        gold.extend([e['label'] for e in batch])
    total = len(gold)
    correct = sum(1 for p, g in zip(preds, gold) if p == g)
    per_class = {}
    for c, name in enumerate(class_names):
        tp = sum(1 for p, g in zip(preds, gold) if p == c and g == c)
        support = sum(1 for g in gold if g == c)
        predicted = sum(1 for p in preds if p == c)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {'support': support, 'predicted_count': predicted,
                           'recall': round(recall, 4), 'precision': round(precision, 4),
                           'f1': round(f1, 4)}
    max_share = max(v['predicted_count'] for v in per_class.values()) / total if total else 0.0
    return {
        'accuracy': round(correct / total, 4) if total else None,
        'correct': correct, 'total': total,
        'per_class': per_class,
        'max_single_class_pred_share': round(max_share, 4),
        'collapsed': bool(max_share >= 0.5),
        'predictions': preds, 'gold_labels': gold,
        'confusion_matrix': [
            [sum(1 for p, g in zip(preds, gold) if g == t and p == q)
             for q in range(len(class_names))]
            for t in range(len(class_names))
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    max_len = cfg['model']['max_seq_length']
    class_names = cfg['data']['class_names']
    model_path = str(ROOT / cfg['model']['local_path'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 与 03_evaluate.py 保持一致：左填充，保证 logits[:, -1] 是最后一个真实 token
    tokenizer.padding_side = 'left'

    test_path = ROOT / cfg['paths']['processed_data_dir'] / 'test.json'
    examples = json.loads(test_path.read_text(encoding='utf-8'))[:args.limit]
    print(f'[info] control set: {len(examples)} samples (前 {args.limit} 条测试样本)')

    lab_ids = label_token_ids(tokenizer, class_names)
    results = {}
    for mode in ('fp32', 'bf16', 'nf4'):
        print(f'[run ] {mode} ...')
        model = build_model(cfg, mode)
        try:
            res = evaluate(model, tokenizer, examples, lab_ids, max_len,
                           args.batch_size, class_names)
        finally:
            del model
            torch.cuda.empty_cache()
        res.update({
            'run_id': f'zero_shot_control400_{mode}',
            'protocol': 'native', 'quantization': mode,
            'model_device': str(next(iter(torch.cuda.device_count() * ['cuda']) or ['cuda'])),
            'prompt_version': cfg.get('data', {}).get('prompt_version', 'v4_chat_template'),
            'limited_eval': True,
            'note': '400 条固定子集，用于检验零样本准确率随基座精度的非单调性',
            'eval_timestamp': datetime.now().isoformat(timespec='seconds'),
        })
        out = ROOT / cfg['paths']['results_dir'] / f'zero_shot_control400_{mode}_eval.json'
        out.write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
        results[mode] = res
        print(f'[done] {mode}: acc={res["accuracy"]} max_share={res["max_single_class_pred_share"]}')

    print('\n== 对照结果（同一 400 条样本）==')
    for mode, r in results.items():
        print(f'  {mode:5s} accuracy={r["accuracy"]:.4f} '
              f'max_single_class_share={r["max_single_class_pred_share"]:.3f} collapsed={r["collapsed"]}')
    print('\n[OK] artifacts: outputs/results/zero_shot_control400_*_eval.json')


if __name__ == '__main__':
    main()
