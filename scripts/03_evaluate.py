#!/usr/bin/env python
"""
03_evaluate.py - 模型评估脚本（v2）

修订要点：
- 双评估协议：native（各自量化格式）+ cross_nf4（统一 NF4）
- 新增 zero-shot 基线（不加载 adapter）
- 新增推理吞吐/延迟测量
- 不再伪报 perplexity；如需 perplexity 请另行实现

用法：
    conda activate lora-project
    python scripts/03_evaluate.py --run nf4_r16_n2000_s42 --protocol native
    python scripts/03_evaluate.py --all
    python scripts/03_evaluate.py --zero-shot
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# 与 02_train.py 保持一致：变长序列下避免缓存分配器碎片导致的显存打满/分页。
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'


def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_quant_cfg(cfg, name):
    return next(q for q in cfg['quantization'] if q['name'] == name)


def is_formal_eval(path):
    """正式（非 --limit 冒烟）评估结果判定，与 04_summary.py 保持一致。"""
    if path.stem.endswith('_partial'):
        return False
    try:
        d = json.load(open(path, encoding='utf-8'))
    except Exception:
        return False
    return not d.get('limited_eval', False)


def build_model_for_protocol(cfg, protocol, quant_name, compute_dtype):
    """按评估协议加载模型"""
    model_path = str(ROOT / cfg['model']['local_path'])

    if protocol == 'native':
        qcfg = get_quant_cfg(cfg, quant_name)
        if qcfg['load_in_4bit']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=qcfg['bnb_4bit_quant_type'],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif qcfg['load_in_8bit']:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None
    elif protocol == 'cross_nf4':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f'未知评估协议: {protocol}')

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        dtype=compute_dtype if bnb_config is None else None,
        device_map='auto' if bnb_config else None,
        local_files_only=True,
    )
    # P0 修复：非量化（bf16_lora / native）分支原本留在 CPU 上，
    # 导致前向仅 2.4 samples/s 且吞吐被记成 CPU 数字，与其余 GPU 配置不可比。
    if bnb_config is None:
        model = model.to('cuda')
    device = next(model.parameters()).device
    assert device.type == 'cuda', f"模型未位于 CUDA（实际 {device}），吞吐数据不可比"
    return model


def get_label_token_ids(tokenizer, class_names):
    ids = [
        tokenizer.encode(' ' + name, add_special_tokens=False)[0]
        for name in class_names
    ]
    if len(set(ids)) != len(ids):
        raise ValueError(f'类别首 token 冲突（无法用单 token logits 判定）: '
                         f'{dict(zip(class_names, ids))}')
    return ids


def count_overlength(examples, tokenizer, max_seq_length):
    """R3-04：统计会被 max_length 右截断的样本数（应为 0）。

    右截断会切掉末尾的类别判定位置，使该样本的预测落在正文中间——
    数据准备阶段已按预算收缩正文，这里只做防御性检查并告警。
    """
    n = 0
    for ex in examples:
        n_tok = (len(tokenizer.encode(ex['prompt'], add_special_tokens=False))
                 + len(tokenizer.encode(' ' + ex['completion'], add_special_tokens=False)) + 1)
        if n_tok > max_seq_length:
            n += 1
    return n


def evaluate_accuracy(model, tokenizer, test_examples, label_token_ids,
                      max_seq_length, batch_size, class_names):
    """在测试集上评估分类，返回完整指标 + 逐样本预测（供 McNemar 与塌缩检测）。"""
    n_classes = len(class_names)
    all_preds, all_labels = [], []

    for i in range(0, len(test_examples), batch_size):
        batch = test_examples[i:i + batch_size]
        prompts = [ex['prompt'] for ex in batch]
        labels = [ex['label'] for ex in batch]

        encoded = tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits[:, -1, :]
            preds = logits[:, label_token_ids].argmax(dim=-1).tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

        if (i // batch_size) % 50 == 0:
            print(f"  进度: {len(all_labels)}/{len(test_examples)}")

    total = len(all_labels)
    if total == 0:
        return None

    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
    per_class = {}
    for c, name in enumerate(class_names):
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l == c)
        support = sum(1 for l in all_labels if l == c)
        predicted = sum(1 for p in all_preds if p == c)
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[name] = {
            'support': support,
            'predicted_count': predicted,
            'tp': tp,
            'recall': round(recall, 4),
            'precision': round(precision, 4),
            'f1': round(f1, 4),
        }

    confusion = [
        [sum(1 for p, l in zip(all_preds, all_labels) if l == t and p == q)
         for q in range(n_classes)]
        for t in range(n_classes)
    ]
    recalls = [v['recall'] for v in per_class.values()]
    f1s = [v['f1'] for v in per_class.values()]
    max_pred_share = (
        max(v['predicted_count'] for v in per_class.values()) / total
    )

    return {
        'accuracy': round(correct / total, 4),
        'balanced_accuracy': round(sum(recalls) / n_classes, 4),
        'macro_f1': round(sum(f1s) / n_classes, 4),
        'correct': correct,
        'total': total,
        'per_class': per_class,
        'confusion_matrix': confusion,
        'confusion_labels': class_names,
        'max_single_class_pred_share': round(max_pred_share, 4),
        'collapsed': bool(max_pred_share >= 0.5),
        'predictions': all_preds,
        'gold_labels': all_labels,
    }


def measure_inference(model, tokenizer, test_examples, label_token_ids,
                      max_seq_length, bench_cfg):
    """测量推理吞吐与延迟（首个 token 分类）"""
    batch_size = bench_cfg['batch_size']
    warmup = bench_cfg['num_warmup_batches']
    measure = bench_cfg['num_measure_batches']

    batches = []
    for i in range(0, min(len(test_examples), (warmup + measure) * batch_size), batch_size):
        batch = test_examples[i:i + batch_size]
        if len(batch) < batch_size:
            break
        prompts = [ex['prompt'] for ex in batch]
        encoded = tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=max_seq_length,
        ).to(model.device)
        batches.append(encoded)

    if not batches:
        return None

    for i in range(min(warmup, len(batches))):
        with torch.no_grad():
            model(**batches[i])
    torch.cuda.synchronize()

    times = []
    total_samples = 0
    for i in range(warmup, min(warmup + measure, len(batches))):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(**batches[i])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        total_samples += batches[i]['input_ids'].shape[0]

    if not times:
        return None

    avg_time = sum(times) / len(times)
    return {
        'avg_batch_latency_s': round(avg_time, 4),
        'samples_per_s': round(total_samples / sum(times), 2),
        'ms_per_sample': round(avg_time * 1000 / batch_size, 3),
        'batches_measured': len(times),
        'batch_size': batch_size,
    }


def evaluate_run(cfg, run_id, protocol, limit=None):
    """评估单次运行的 adapter"""
    print("=" * 70)
    print(f"评估: {run_id} | 协议: {protocol}")
    print("=" * 70)

    class_names = cfg['data']['class_names']
    model_cfg = cfg['model']
    eval_cfg = cfg['evaluation']
    compute_dtype = (
        torch.bfloat16 if cfg['bnb_config']['bnb_4bit_compute_dtype'] == 'bfloat16'
        else torch.float16
    )

    data_dir = ROOT / cfg['paths']['processed_data_dir']
    with open(data_dir / 'test.json', encoding='utf-8') as f:
        test_examples = json.load(f)
    if limit:
        test_examples = test_examples[:limit]
    print(f"测试集: {len(test_examples)} 条")

    # v2: 从训练结果 JSON 读取 quant，避免靠字符串切分猜（改命名规则会静默取错）
    train_result_path = ROOT / cfg['paths']['results_dir'] / f'{run_id}.json'
    if train_result_path.exists():
        quant_name = json.load(open(train_result_path, encoding='utf-8'))['quantization']
    else:
        quant_name = run_id.split('_r')[0]
        print(f"[WARN] 未找到 {train_result_path.name}，回退字符串解析: {quant_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / model_cfg['local_path']), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    label_token_ids = get_label_token_ids(tokenizer, class_names)
    print(f"类别 token ids: {dict(zip(class_names, label_token_ids))}")
    over = count_overlength(test_examples, tokenizer, model_cfg['max_seq_length'])
    if over:
        print(f"[WARN] {over} 条样本超过 max_seq_length，会被右截断（判定位置错位）；"
              f"请重跑 01_prepare_data.py")

    model = build_model_for_protocol(cfg, protocol, quant_name, compute_dtype)

    adapter_dir = ROOT / cfg['paths']['checkpoints_dir'] / run_id / 'adapter'
    if not adapter_dir.exists():
        print(f"[FAIL] adapter 不存在: {adapter_dir}")
        return None
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    metrics = evaluate_accuracy(
        model, tokenizer, test_examples, label_token_ids,
        model_cfg['max_seq_length'], eval_cfg['batch_size'], class_names,
    )

    bench_result = None
    if eval_cfg['inference_bench']['enabled']:
        bench_result = measure_inference(
            model, tokenizer, test_examples, label_token_ids,
            model_cfg['max_seq_length'], eval_cfg['inference_bench'],
        )

    meta_path = ROOT / cfg['paths']['processed_data_dir'] / 'metadata.json'
    prompt_version = (json.load(open(meta_path, encoding='utf-8'))
                      .get('prompt_version', 'unknown') if meta_path.exists() else 'unknown')

    result = {
        'run_id': run_id,
        'protocol': protocol,
        'model_device': 'cuda',
        'label_token_ids': dict(zip(class_names, label_token_ids)),
        'prompt_version': prompt_version,
        'eval_config': {
            'batch_size': eval_cfg['batch_size'],
            'max_seq_length': model_cfg['max_seq_length'],
            'inference_bench': eval_cfg['inference_bench'],
            'test_size': len(test_examples),
            'overlength_samples': over,
        },
        'inference_bench': bench_result,
        'eval_timestamp': datetime.now().isoformat(),
        'limited_eval': limit is not None,
        **metrics,
    }

    out_dir = ROOT / cfg['paths']['results_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    # 冒烟测试单独命名（含协议），避免被汇总脚本误当作正式结果或互相覆盖
    out_path = (out_dir / f'{run_id}_{protocol}_eval_partial.json' if limit
                else out_dir / f'{run_id}_{protocol}_eval.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] {run_id}/{protocol}: accuracy = {metrics['accuracy']:.4f}, "
          f"balanced = {metrics['balanced_accuracy']:.4f}, "
          f"max_pred_share = {metrics['max_single_class_pred_share']:.3f}")
    if metrics['collapsed']:
        print("  [WARN] 预测分布塌缩（单一类别占比 ≥50%），该结果不代表分类能力")
    if bench_result:
        print(f"  分类前向吞吐: {bench_result['samples_per_s']} samples/s, "
              f"延迟: {bench_result['ms_per_sample']} ms/sample")
    print(f"结果: {out_path.relative_to(ROOT)}")

    del model
    torch.cuda.empty_cache()
    return result


def run_zero_shot(cfg, quant_name='nf4', limit=None):
    """zero-shot 基线：不加载 adapter，可指定任一量化格式的基座。

    R3-21：原实现写死 NF4；现支持 --quant，用同一套 build_model_for_protocol
    加载（含非量化分支的 CUDA 搬运与断言），输出 zero_shot_{quant}_eval.json。
    """
    print("=" * 70)
    print(f"Zero-shot 基线评估（{quant_name} 基座，无 adapter）")
    print("=" * 70)

    class_names = cfg['data']['class_names']
    model_cfg = cfg['model']
    eval_cfg = cfg['evaluation']
    compute_dtype = torch.bfloat16

    data_dir = ROOT / cfg['paths']['processed_data_dir']
    with open(data_dir / 'test.json', encoding='utf-8') as f:
        test_examples = json.load(f)
    if limit:
        test_examples = test_examples[:limit]

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / model_cfg['local_path']), local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    label_token_ids = get_label_token_ids(tokenizer, class_names)
    over = count_overlength(test_examples, tokenizer, model_cfg['max_seq_length'])
    if over:
        print(f"[WARN] {over} 条样本超过 max_seq_length，会被右截断（判定位置错位）")

    model = build_model_for_protocol(cfg, 'native', quant_name, compute_dtype)
    model.eval()

    metrics = evaluate_accuracy(
        model, tokenizer, test_examples, label_token_ids,
        model_cfg['max_seq_length'], eval_cfg['batch_size'], class_names,
    )
    bench_result = measure_inference(
        model, tokenizer, test_examples, label_token_ids,
        model_cfg['max_seq_length'], eval_cfg['inference_bench'],
    )

    meta_path = ROOT / cfg['paths']['processed_data_dir'] / 'metadata.json'
    prompt_version = (json.load(open(meta_path, encoding='utf-8'))
                      .get('prompt_version', 'unknown') if meta_path.exists() else 'unknown')

    result = {
        'run_id': f'zero_shot_{quant_name}',
        'protocol': 'zero_shot',
        'quantization': quant_name,
        'model_device': 'cuda',
        'label_token_ids': dict(zip(class_names, label_token_ids)),
        'prompt_version': prompt_version,
        'eval_config': {
            'batch_size': eval_cfg['batch_size'],
            'max_seq_length': model_cfg['max_seq_length'],
            'inference_bench': eval_cfg['inference_bench'],
            'test_size': len(test_examples),
            'overlength_samples': over,
        },
        'inference_bench': bench_result,
        'eval_timestamp': datetime.now().isoformat(),
        'limited_eval': limit is not None,
        **metrics,
    }

    out_dir = ROOT / cfg['paths']['results_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / f'zero_shot_{quant_name}_eval_partial.json' if limit
                else out_dir / f'zero_shot_{quant_name}_eval.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] zero-shot/{quant_name}: accuracy = {metrics['accuracy']:.4f}, "
          f"balanced = {metrics['balanced_accuracy']:.4f}, "
          f"max_pred_share = {metrics['max_single_class_pred_share']:.3f}")
    if metrics['collapsed']:
        print("  [WARN] zero-shot 预测分布塌缩，CP1 应判为不通过")
    if bench_result:
        print(f"  分类前向吞吐: {bench_result['samples_per_s']} samples/s")
    print(f"结果: {out_path.relative_to(ROOT)}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    cfg = load_config()
    quant_choices = [q['name'] for q in cfg['quantization']]

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='单次运行的 run_id')
    parser.add_argument('--protocol', choices=['native', 'cross_nf4'],
                        default='native', help='评估协议')
    parser.add_argument('--all', action='store_true', help='评估所有已完成的 run')
    parser.add_argument('--zero-shot', action='store_true',
                        help='运行 zero-shot 基线（默认四种量化格式全跑）')
    parser.add_argument('--quant', choices=quant_choices,
                        help='zero-shot 使用的基座格式（默认全部）')
    parser.add_argument('--limit', type=int, help='仅评估前 N 条')
    parser.add_argument('--resume', action='store_true',
                        help='跳过已存在的正式评估结果（中断后续跑）')
    args = parser.parse_args()

    if args.zero_shot:
        targets = [args.quant] if args.quant else quant_choices
        for q in targets:
            formal = ROOT / cfg['paths']['results_dir'] / f'zero_shot_{q}_eval.json'
            if args.resume and not args.limit and formal.exists() and is_formal_eval(formal):
                print(f"[SKIP] 已存在正式 zero-shot 结果: {formal.name}")
                continue
            run_zero_shot(cfg, quant_name=q, limit=args.limit)
        return

    if args.all:
        results_dir = ROOT / cfg['paths']['results_dir']
        run_files = sorted(results_dir.glob('*.json'))
        skip_suffixes = ('_eval', '_partial', '_summary', '_pareto', '_config')
        run_ids = [
            f.stem for f in run_files
            if not f.stem.endswith(skip_suffixes)
            and not f.stem.startswith('zero_shot')
        ]
    elif args.run:
        run_ids = [args.run]
    else:
        parser.print_help()
        return

    protocols = cfg['evaluation']['protocols']
    for run_id in run_ids:
        for protocol in protocols:
            target = (ROOT / cfg['paths']['results_dir'] /
                      f'{run_id}_{protocol}_eval.json')
            if args.resume and not args.limit and target.exists() and is_formal_eval(target):
                print(f"[SKIP] 已存在正式结果: {target.name}")
                continue
            evaluate_run(cfg, run_id, protocol, limit=args.limit)


if __name__ == '__main__':
    main()
