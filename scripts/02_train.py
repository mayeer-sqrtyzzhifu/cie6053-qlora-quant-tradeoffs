#!/usr/bin/env python
"""
02_train.py - QLoRA 微调训练脚本（v3）

v3 修订（依据 PLAN_REVIEW_2026-11-17.md）：
- P0 §2.6: 分离 train_time_pure_s（不含 val 评估）与 train_time_total_s；
  eval_strategy 改为 'no'，val 评估移到训练后单次执行，避免吞吐口径失真。
- P0 §2.4 / 决策③: 新增 --batch-size / --max-seq-length CLI 覆盖，
  覆盖值写入 experiment_snapshot.yaml 与结果 JSON，保证可追溯。
- P0 §2.5: 记录 model_resident_memory_gb（加载后、训练前），把权重占用与激活占用分开。
- P1 §3.3: 训练期 GPU 温度/功耗/利用率采样 → gpu_trace.csv。
- P1 §3.9: console.log 改为 dump 完整 trainer.state.log_history。
- P1 §2.5 / §3.8: 新增 --explore（batch 8/16 探索 + fp4 补种子），共 4 次。
- 数字修正: 主矩阵 15 次（原注释误写 20）。

用法：
    conda activate lora-project
    python scripts/02_train.py --plan --dry-run      # 核对 15 次组合
    python scripts/02_train.py --plan                # 主矩阵 15 次（推荐）
    python scripts/02_train.py --explore             # 增强 4 次
    python scripts/02_train.py --plan --explore      # 合计 19 次
    python scripts/02_train.py --quant nf4 --rank 16 --train-size 2000 --batch-size 8
"""
import argparse
import gc
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

# 必须在 import torch 之前设置：变长序列（88–253 token）+ 梯度检查点会让默认
# 缓存分配器产生大量碎片，显存被 reserved 打满后 Windows/WDDM 会向系统内存分页，
# 表现为「利用率 99% 但功耗只有 70 W、步速掉 3–5 倍」。expandable_segments 可消除该碎片。
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'

# int8（bnb MatMul8bitLt）会把 bf16/fp32 输入转 fp16，日志会刷屏；
# 这是 bnb 的既定行为，不影响结果，但会淹没训练日志。
warnings.filterwarnings('ignore', message='MatMul8bitLt.*')
logging.getLogger('bitsandbytes').setLevel(logging.ERROR)


def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_run_id(quant_name, rank, train_size, seed, overrides=None):
    """run_id 唯一真源：batch 探索运行追加 _bs{N}，不覆盖主矩阵结果。"""
    overrides = overrides or {}
    bs_tag = f"_bs{overrides['batch_size']}" if overrides.get('batch_size') else ''
    return f"{quant_name}_r{rank}_n{train_size}_s{seed}{bs_tag}"


def param_footprint_gb(model):
    """权重占用（设备无关），用于把"模型常驻"与"峰值（含激活）"分开。"""
    try:
        return round(model.get_memory_footprint() / 1024 ** 3, 3)
    except Exception:
        total = 0
        for p in model.parameters():
            try:
                total += p.numel() * p.element_size()
            except Exception:
                continue
        return round(total / 1024 ** 3, 3)


def stoppable_smi_query():
    try:
        return subprocess.run(
            ['nvidia-smi',
             '--query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,clocks.sm,memory.used',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ''


def gpu_mem_used_mb():
    """当前整卡占用（MiB），用于子进程间等待驱动回收显存。"""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        return int(out[0].strip())
    except Exception:
        return None


def wait_gpu_idle(limit_mb=2500, timeout_s=180, poll_s=3):
    """等 GPU 回到空闲水位再启动下一个 run。

    R3-27：子进程退出后，Windows/WDDM 对显存的回收可能滞后；紧接着启动下一个进程会让
    新进程在「显存已被占满」的假象下分配，触发分页（利用率高、功耗低、步速掉 3–10 倍）。
    这里显式等待整卡占用降到阈值以下，最多等 timeout_s 秒。
    """
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = gpu_mem_used_mb()
        if used is None or used <= limit_mb:
            return used
        time.sleep(poll_s)
    return gpu_mem_used_mb()


def save_env_snapshot(out_dir):
    """训练前硬件/驱动/库版本快照"""
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'pre_training',
        'python': sys.version,
        'torch': torch.__version__,
        'cuda_compiled': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        snapshot['gpu_name'] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        snapshot['compute_capability'] = f'sm_{cap[0]}{cap[1]}'
        snapshot['vram_total_gb'] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
        )
    snapshot['nvidia_smi'] = stoppable_smi_query()
    with open(out_dir / 'env_snapshot.json', 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


class GpuTraceSampler:
    """训练期后台采样 GPU 状态（P1 §3.3）"""

    def __init__(self, out_path, interval=2.0):
        self.out_path = out_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        header = ('timestamp,temperature_gpu_c,power_draw_w,utilization_gpu_pct,'
                  'clocks_sm_mhz,memory_used_mb\n')
        with open(self.out_path, 'w', encoding='utf-8') as f:
            f.write(header)
            while not self._stop.is_set():
                line = stoppable_smi_query()
                if line:
                    f.write(line + '\n')
                    f.flush()
                self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)


def save_config_snapshot(cfg, out_dir, overrides=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = dict(cfg)
    snap['_cli_overrides'] = overrides or {}
    snap['_snapshot_note'] = 'CLI 覆盖值见 _cli_overrides；其余字段与 configs/experiment.yaml 一致'
    with open(out_dir / 'experiment_snapshot.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(snap, f, allow_unicode=True, sort_keys=False)


def build_bnb_config(qcfg, bnb_cfg, compute_dtype):
    if qcfg['load_in_4bit']:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qcfg['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bnb_cfg['bnb_4bit_use_double_quant'],
        )
    elif qcfg['load_in_8bit']:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class ClassificationDataset(torch.utils.data.Dataset):
    """仅对 completion 计算 loss 的 causal LM 数据集。

    v4（R3-04）：长度超限时不再右截断（右截断会切掉 completion 与 EOS，
    使该样本 labels 全为 -100、对 loss 无贡献），改为从左侧裁掉 prompt 开头，
    保证 completion 与 EOS 一定保留。数据准备阶段已把正文收缩到预算内，
    正常情况下 truncated == 0，此分支仅作兜底。
    """

    def __init__(self, examples, tokenizer, max_length):
        self.features = []
        self.truncated = 0
        for ex in examples:
            prompt_ids = tokenizer.encode(ex['prompt'], add_special_tokens=False)
            completion_ids = tokenizer.encode(
                ' ' + ex['completion'], add_special_tokens=False
            )
            eos_id = tokenizer.eos_token_id
            completion_ids = completion_ids + [eos_id]
            budget = max_length - len(completion_ids)
            if budget <= 0:
                # 极端兜底：只保留 completion 与 EOS 的尾部
                completion_ids = completion_ids[-max(1, max_length):]
                prompt_ids = []
                self.truncated += 1
            elif len(prompt_ids) > budget:
                prompt_ids = prompt_ids[-budget:]
                self.truncated += 1
            self.features.append({
                'input_ids': prompt_ids + completion_ids,
                'attention_mask': [1] * (len(prompt_ids) + len(completion_ids)),
                'labels': [-100] * len(prompt_ids) + completion_ids,
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


class PaddingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f['input_ids']) for f in features)
        batch = {'input_ids': [], 'attention_mask': [], 'labels': []}
        for f in features:
            pad_len = max_len - len(f['input_ids'])
            batch['input_ids'].append(f['input_ids'] + [self.pad_token_id] * pad_len)
            batch['attention_mask'].append(f['attention_mask'] + [0] * pad_len)
            batch['labels'].append(f['labels'] + [-100] * pad_len)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def run_once(cfg, quant_name, rank, train_size, seed, overrides=None):
    """运行单次实验。overrides: CLI 覆盖的超参（写入留档）。"""
    overrides = overrides or {}
    set_seed(seed)

    qcfg = next(q for q in cfg['quantization'] if q['name'] == quant_name)
    model_cfg = cfg['model']
    lora_cfg = cfg['lora']
    train_cfg = cfg['training']

    batch_size = overrides.get('batch_size', train_cfg['batch_size'])
    max_seq_length = overrides.get('max_seq_length', model_cfg['max_seq_length'])

    # batch 探索运行用独立 run_id，不覆盖主矩阵结果
    run_id = make_run_id(quant_name, rank, train_size, seed, overrides)
    print("\n" + "=" * 70)
    print(f"实验: {run_id}  |  batch_size={batch_size}, max_seq_length={max_seq_length}")
    print("=" * 70)

    run_log_dir = ROOT / cfg['paths']['logs_dir'] / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    save_env_snapshot(run_log_dir)
    save_config_snapshot(cfg, run_log_dir, overrides)

    compute_dtype = (
        torch.bfloat16 if cfg['bnb_config']['bnb_4bit_compute_dtype'] == 'bfloat16'
        else torch.float16
    )

    data_dir = ROOT / cfg['paths']['processed_data_dir']
    with open(data_dir / f'train_{train_size}.json', encoding='utf-8') as f:
        train_examples = json.load(f)
    with open(data_dir / 'val.json', encoding='utf-8') as f:
        val_examples = json.load(f)

    model_path = str(ROOT / model_cfg['local_path'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train_ds = ClassificationDataset(train_examples, tokenizer, max_seq_length)
    val_ds = ClassificationDataset(
        val_examples[:train_cfg['eval_subset_size']], tokenizer, max_seq_length
    )
    if train_ds.truncated or val_ds.truncated:
        print(f"[WARN] 序列超限被兜底截断：train {train_ds.truncated} / val {val_ds.truncated} 条"
              f"（数据准备阶段的正文收缩应已消除此情况）")

    bnb_config = build_bnb_config(qcfg, cfg['bnb_config'], compute_dtype)
    # R3-26：常驻显存用「加载前后差值」，而不是绝对值——绝对值会把同进程残留算进去。
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_before_gb = torch.cuda.memory_allocated() / 1024**3
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        dtype=compute_dtype if bnb_config is None else None,
        device_map='auto' if bnb_config else None,
        local_files_only=True,
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg['gradient_checkpointing']
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * lora_cfg['lora_alpha_multiplier'],
        lora_dropout=lora_cfg['lora_dropout'],
        target_modules=lora_cfg['target_modules'],
        task_type=lora_cfg['task_type'],
        bias='none',
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # R3-06 修复：bf16_lora 分支 device_map=None，此时模型仍在 CPU，
    # 直接读 torch.cuda.memory_allocated() 会得到 0（原实现即如此）。
    # 先显式搬到 GPU，再分别记录"常驻（权重+LoRA）"与"设备无关权重足迹"。
    if bnb_config is None:
        model = model.to('cuda')
    load_time = time.time() - t0

    torch.cuda.synchronize()
    model_resident_gb = max(0.0, torch.cuda.memory_allocated() / 1024**3 - mem_before_gb)
    weight_footprint_gb = param_footprint_gb(model)
    total_params = sum(prm.numel() for prm in model.parameters())
    trainable_params = sum(prm.numel() for prm in model.parameters()
                           if prm.requires_grad)
    avg_seq_len = round(
        sum(len(f['input_ids']) for f in train_ds.features) / len(train_ds.features), 1)
    data_meta_path = ROOT / cfg['paths']['processed_data_dir'] / 'metadata.json'
    data_meta = (json.load(open(data_meta_path, encoding='utf-8'))
                 if data_meta_path.exists() else {})
    prompt_version = data_meta.get('prompt_version', 'unknown')
    prompt_style = data_meta.get('prompt_style', 'unknown')

    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps = steps_per_epoch * train_cfg['num_epochs']
    warmup_steps = max(1, int(total_steps * train_cfg['warmup_ratio']))

    args = TrainingArguments(
        output_dir=str(ROOT / cfg['paths']['checkpoints_dir'] / run_id),
        num_train_epochs=train_cfg['num_epochs'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=cfg['evaluation']['batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        learning_rate=train_cfg['learning_rate'],
        lr_scheduler_type=train_cfg['lr_scheduler'],
        warmup_steps=warmup_steps,
        weight_decay=train_cfg['weight_decay'],
        max_grad_norm=train_cfg['max_grad_norm'],
        optim=train_cfg['optim'],
        logging_steps=train_cfg['logging_steps'],
        # P0 §2.6: 训练期不做 val 评估，避免固定 500 条评估开销污染吞吐口径
        eval_strategy='no',
        save_strategy='no',
        report_to='tensorboard',
        seed=seed,
        data_seed=seed,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        # R3-27：关闭 tqdm。批处理下父进程用 Tee 记录子进程 stdout，进度条每步一行会把
        # 管道/缓冲打满，反压到子进程造成「利用率高但功耗低、步速掉 3–10 倍」的假慢。
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        # 保留 eval_dataset 供训练后单次 evaluate()，但 eval_strategy='no'
        # 使训练过程内不触发评估，吞吐口径纯净
        eval_dataset=val_ds,
        data_collator=PaddingCollator(tokenizer.pad_token_id),
    )

    # P1 §3.3: 启动 GPU 采样
    gpu_trace_path = run_log_dir / 'gpu_trace.csv'
    sampler = GpuTraceSampler(gpu_trace_path, cfg['experiment'].get('gpu_trace_interval_s', 2.0))
    sampler.start()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    train_time_pure = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    # val 评估移到训练后单次执行，单独计时。
    # R3-25：评估前必须释放训练期 reserved 显存——否则 eval 的 logits（batch 8 × 256 × 151k）
    # 会与训练期缓存块叠加，超过 12 GB 预算后被 WDDM 分页，实测步速从 0.4 s/batch 掉到 4.5 s/batch。
    torch.cuda.empty_cache()
    t_eval = time.time()
    eval_loss = trainer.evaluate()['eval_loss']
    eval_time = time.time() - t_eval
    sampler.stop()

    adapter_dir = ROOT / cfg['paths']['checkpoints_dir'] / run_id / 'adapter'
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # P1 §3.9: 完整 log_history 留档
    with open(run_log_dir / 'console.log', 'w', encoding='utf-8') as f:
        f.write(f"run_id={run_id}\n")
        f.write(f"prompt_version={prompt_version}\nprompt_style={prompt_style}\n")
        f.write(f"batch_size={batch_size} max_seq_length={max_seq_length}\n")
        f.write(f"train_truncated={train_ds.truncated} val_truncated={val_ds.truncated}\n")
        f.write(f"train_time_pure_s={train_time_pure:.2f}\n")
        f.write(f"eval_time_s={eval_time:.2f}\n")
        f.write(f"peak_memory_gb={peak_mem:.3f}\n")
        f.write(f"model_resident_memory_gb={model_resident_gb:.3f}\n")
        f.write(f"weight_footprint_gb={weight_footprint_gb:.3f}\n")
        f.write(f"train_loss={train_result.training_loss:.4f}\n")
        f.write(f"eval_loss={eval_loss:.4f}\n\n")
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    # 汇总 GPU 采样
    gpu_stats = summarize_gpu_trace(gpu_trace_path)

    metrics = {
        'run_id': run_id,
        'quantization': quant_name,
        'lora_rank': rank,
        'train_size': train_size,
        'seed': seed,
        'batch_size': batch_size,
        'max_seq_length': max_seq_length,
        'prompt_version': prompt_version,
        'prompt_style': prompt_style,
        'cli_overrides': overrides,
        'train_truncated': train_ds.truncated,
        'val_truncated': val_ds.truncated,
        'load_time_s': round(load_time, 2),
        'train_time_pure_s': round(train_time_pure, 2),
        'eval_time_s': round(eval_time, 2),
        'train_time_total_s': round(train_time_pure + eval_time, 2),
        'model_resident_memory_gb': round(model_resident_gb, 3),
        'weight_footprint_gb': weight_footprint_gb,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'avg_seq_len': avg_seq_len,
        'total_train_tokens': int(avg_seq_len * len(train_ds) * train_cfg['num_epochs']),
        'peak_memory_gb': round(peak_mem, 3),
        'train_samples_per_s': round(
            len(train_ds) * train_cfg['num_epochs'] / train_time_pure, 2
        ),
        'train_loss': round(train_result.training_loss, 4),
        'eval_loss': round(eval_loss, 4),
        'gpu_trace': gpu_stats,
        'adapter_dir': str(adapter_dir.relative_to(ROOT)),
        'log_dir': str(run_log_dir.relative_to(ROOT)),
    }

    results_dir = ROOT / cfg['paths']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f'{run_id}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] {run_id}")
    print(f"  常驻显存 : {metrics['model_resident_memory_gb']} GB")
    print(f"  峰值显存 : {metrics['peak_memory_gb']} GB")
    print(f"  纯训练   : {metrics['train_time_pure_s']} s "
          f"(另含 val 评估 {metrics['eval_time_s']} s)")
    print(f"  训练吞吐 : {metrics['train_samples_per_s']} samples/s（纯训练口径）")
    print(f"  训练 Loss: {metrics['train_loss']}")
    if gpu_stats:
        print(f"  GPU 平均功耗 {gpu_stats['avg_power_w']} W / "
              f"峰值温度 {gpu_stats['max_temp_c']} °C")
    print(f"  结果文件 : {out_path.relative_to(ROOT)}")

    del model, trainer
    torch.cuda.empty_cache()
    return metrics


def smoke_formats(cfg, steps=4, n_samples=32):
    """CP0 的 4 步格式冒烟：对四种量化格式各跑固定步数，验证加载/量化/反传路径可用。

    不写 adapter、不进结果汇总；只写 outputs/logs/smoke_YYYYMMDD_HHMM.json。
    返回 (ok: bool, rows: list[dict])。
    """
    import tempfile
    out_rows = []
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / cfg['model']['local_path']), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with open(ROOT / cfg['paths']['processed_data_dir'] / 'train_500.json',
              encoding='utf-8') as f:
        examples = json.load(f)[:n_samples]
    ds = ClassificationDataset(examples, tokenizer,
                               int(cfg['model']['max_seq_length']))
    collator = PaddingCollator(tokenizer.pad_token_id)

    for qcfg in cfg['quantization']:
        name = qcfg['name']
        row = {'quantization': name, 'steps': steps}
        try:
            t0 = time.time()
            bnb = build_bnb_config(qcfg, cfg['bnb_config'], torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(
                str(ROOT / cfg['model']['local_path']),
                quantization_config=bnb,
                dtype=torch.bfloat16 if bnb is None else None,
                device_map='auto' if bnb else None,
                local_files_only=True,
            )
            if bnb is not None:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
            model = get_peft_model(model, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=cfg['lora']['target_modules'],
                task_type=cfg['lora']['task_type'], bias='none'))
            if bnb is None:
                model = model.to('cuda')
            with tempfile.TemporaryDirectory() as tmp:
                args = TrainingArguments(
                    output_dir=tmp, max_steps=steps,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=1, learning_rate=2e-4,
                    lr_scheduler_type='constant', warmup_steps=1,
                    logging_steps=1, eval_strategy='no', save_strategy='no',
                    report_to='none', bf16=True, gradient_checkpointing=True,
                    dataloader_num_workers=0, remove_unused_columns=False,
                )
                trainer = Trainer(model=model, args=args, train_dataset=ds,
                                  data_collator=collator)
                torch.cuda.reset_peak_memory_stats()
                result = trainer.train()
            row.update({
                'ok': True,
                'load_and_train_s': round(time.time() - t0, 1),
                'last_loss': round(float(result.training_loss), 4),
                'peak_memory_gb': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
                'device': str(next(model.parameters()).device),
            })
            del model, trainer
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            traceback.print_exc()
            row.update({'ok': False, 'error': f'{type(e).__name__}: {e}'})
            try:
                del model, trainer
            except Exception:
                pass
            torch.cuda.empty_cache()
        out_rows.append(row)
        status = 'PASS' if row.get('ok') else 'FAIL'
        print(f"[{status}] {name:10s} loss={row.get('last_loss', 'n/a')} "
              f"peak={row.get('peak_memory_gb', 'n/a')} GB {row.get('error', '')}")

    ok = all(r.get('ok') for r in out_rows)
    out_dir = ROOT / cfg['paths']['logs_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smoke_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'steps': steps, 'n_samples': n_samples, 'ok': ok,
                   'rows': out_rows}, f, ensure_ascii=False, indent=2)
    print(f"\n[smoke] {'全部通过' if ok else '存在失败项'} → {out_path.relative_to(ROOT)}")
    return ok, out_rows


def summarize_gpu_trace(path):
    """从 gpu_trace.csv 汇总平均功耗、峰值温度、是否可能降频。

    R4：早期表头缺 memory_used_mb 一列（数据行有 6 列而表头只有 5 列），
    这里按「已知列名 + 位置兜底」解析，使新旧文件都能读。
    """
    if not path.exists():
        return None
    import csv as _csv
    with open(path, encoding='utf-8', newline='') as f:
        rows = list(_csv.reader(f))
    if not rows:
        return None
    header = [c.strip() for c in rows[0]]
    idx = {name: i for i, name in enumerate(header)}
    pos = {
        'temp': idx.get('temperature_gpu_c', 1),
        'power': idx.get('power_draw_w', 2),
        'util': idx.get('utilization_gpu_pct', 3),
        'clock': idx.get('clocks_sm_mhz', 4),
        'mem': idx.get('memory_used_mb', 5),
    }
    temps, powers, utils, clocks, mems = [], [], [], [], []
    for parts in rows[1:]:
        parts = [p.strip() for p in parts]
        if len(parts) < 5:
            continue
        try:
            temps.append(float(parts[pos['temp']]))
            powers.append(float(parts[pos['power']].replace(' W', '')))
            utils.append(float(parts[pos['util']].replace(' %', '')))
            clocks.append(float(parts[pos['clock']].replace(' MHz', '')))
            if len(parts) > pos['mem'] and parts[pos['mem']] not in ('', '[N/A]'):
                mems.append(float(parts[pos['mem']].replace(' MiB', '')))
        except (ValueError, IndexError):
            continue
    if not temps:
        return None
    stats = {
        'samples': len(temps),
        'avg_temp_c': round(sum(temps) / len(temps), 1),
        'max_temp_c': round(max(temps), 1),
        'thermal_throttle_suspected': bool(max(temps) >= 85),
    }
    # R3-07 修复：部分笔记本 GPU 的功耗/利用率/时钟会返回 [N/A]，
    # 原实现会在列表为空时 ZeroDivisionError。
    if powers:
        stats['avg_power_w'] = round(sum(powers) / len(powers), 1)
        stats['max_power_w'] = round(max(powers), 1)
    if utils:
        stats['avg_utilization_pct'] = round(sum(utils) / len(utils), 1)
    if clocks:
        stats['min_sm_clock_mhz'] = round(min(clocks), 1)
        stats['max_sm_clock_mhz'] = round(max(clocks), 1)
        stats['thermal_throttle_suspected'] = bool(
            max(temps) >= 85 and (max(clocks) - min(clocks)) > 200
        )
    if mems:
        stats['avg_gpu_mem_used_mb'] = round(sum(mems) / len(mems))
        stats['max_gpu_mem_used_mb'] = round(max(mems))
    return stats


def build_combos(cfg, args):
    """构造实验组合。

    --plan   = OFAT 单因素筛选矩阵，15 次（含 3 个关键配置的 3 种子）
    --explore= 4 次增强：nf4 batch 8/16 各 1 次 + fp4 补 seed 43/44
    --sweep  = 单因素扫描（注意：同样会展开多种子，勿与 --plan 叠加）
    --all    = 全因子 54 次，不推荐
    """
    quants = [q['name'] for q in cfg['quantization']]
    ranks = cfg['lora']['ranks']
    sizes = cfg['data']['train_subset_sizes']
    default_seed = cfg['experiment']['seed']
    seeds = cfg['experiment']['seeds']
    multi_seed = {
        (q, r, s) for q, r, s in cfg['experiment']['multi_seed_configs']
    }
    center_q, center_r, center_s = quants[2], ranks[1], sizes[2]

    combos = []

    if args.all or args.exhaustive:
        base = [(q, r, s) for q in quants for r in ranks for s in sizes]
    elif args.plan:
        base = [(q, center_r, center_s) for q in quants]
        base += [(center_q, r, center_s) for r in ranks if r != center_r]
        base += [(center_q, center_r, s) for s in sizes if s != center_s]
    elif args.sweep == 'quant':
        base = [(q, ranks[1], sizes[2]) for q in quants]
    elif args.sweep == 'rank':
        base = [(quants[2], r, sizes[2]) for r in ranks]
    elif args.sweep == 'size':
        base = [(quants[2], ranks[1], s) for s in sizes]
    elif args.quant and args.rank and args.train_size:
        base = [(args.quant, args.rank, args.train_size)]
    else:
        base = []

    seen = set()
    for q, r, s in base:
        if (q, r, s) in seen:
            continue
        seen.add((q, r, s))
        run_seeds = seeds if (q, r, s) in multi_seed else [default_seed]
        for sd in run_seeds:
            combos.append((q, r, s, sd, {}))

    if args.explore:
        for bs in cfg['experiment'].get('explore_batch_sizes', [8, 16]):
            combos.append((center_q, center_r, center_s, default_seed,
                           {'batch_size': bs}))
        fp4 = quants[3]
        for sd in seeds[1:]:
            combos.append((fp4, center_r, center_s, sd, {}))

    return combos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quant', help='量化格式: bf16_lora/int8/nf4/fp4')
    p.add_argument('--rank', type=int, help='LoRA 秩')
    p.add_argument('--train-size', type=int, help='训练样本数')
    p.add_argument('--dry-run', action='store_true', help='只打印组合不执行')
    p.add_argument('--seed', type=int, help='覆盖默认种子')
    p.add_argument('--batch-size', type=int, help='CLI 覆盖 batch_size（OOM 处置 A）')
    p.add_argument('--max-seq-length', type=int, help='CLI 覆盖 max_seq_length')
    p.add_argument('--all', action='store_true', help='全因子 54 次，不推荐')
    p.add_argument('--exhaustive', action='store_true', help='同 --all')
    p.add_argument('--plan', action='store_true', help='主矩阵 15 次（推荐）')
    p.add_argument('--explore', action='store_true', help='增强 4 次')
    p.add_argument('--sweep', choices=['quant', 'rank', 'size'], help='单因素扫描')
    p.add_argument('--retries', type=int, default=2,
                   help='单次运行失败后的重试次数（执行契约 §6：默认 2 次）')
    p.add_argument('--resume', action='store_true',
                   help='跳过已有结果 JSON 的组合（中断后续跑，不覆盖已完成结果）')
    p.add_argument('--smoke', action='store_true',
                   help='CP0：四种量化格式各跑 N 步冒烟（不产出 adapter、不进汇总）')
    p.add_argument('--smoke-steps', type=int, default=4, help='冒烟步数，默认 4')
    args = p.parse_args()

    cfg = load_config()

    if args.smoke:
        ok, _ = smoke_formats(cfg, steps=args.smoke_steps)
        sys.exit(0 if ok else 1)
    if args.seed is not None:
        cfg['experiment']['seed'] = args.seed
        cfg['experiment']['multi_seed_configs'] = []

    cli_overrides = {}
    if args.batch_size:
        cli_overrides['batch_size'] = args.batch_size
    if args.max_seq_length:
        cli_overrides['max_seq_length'] = args.max_seq_length

    combos = build_combos(cfg, args)
    if not combos:
        p.print_help()
        return

    print(f"计划运行 {len(combos)} 次实验:")
    for q, r, s, sd, ov in combos:
        print(f"  - {q}, rank={r}, n={s}, seed={sd}"
              + (f", overrides={ov}" if ov else ""))

    if args.dry_run:
        print("\n[dry-run] 未执行任何训练。")
        return

    all_metrics, failures = [], []
    results_dir = ROOT / cfg['paths']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    # R3-26：批处理模式下每个 run 用「独立子进程」执行。
    # 同进程串行跑多个 run 时，上一轮的分配器残留会污染下一轮的显存读数
    # （实测 nf4 常驻从 1.5 GB 涨到 5.0 GB、峰值涨到 7.3 GB，且随 run 顺序单调增长）。
    # 显存是本项目的核心指标，必须逐 run 隔离进程才能保证可比。
    single_run = bool(args.quant and args.rank and args.train_size)
    script = str(Path(__file__).resolve())

    for q, r, s, sd, ov in combos:
        merged = {**cli_overrides, **ov}
        run_id = make_run_id(q, r, s, sd, merged)
        out_json = results_dir / f'{run_id}.json'

        if args.resume and out_json.exists():
            try:
                done = json.load(open(out_json, encoding='utf-8'))
                if 'train_loss' in done:
                    all_metrics.append(done)
                    print(f"[SKIP] 已存在完成结果: {run_id}")
                    continue
            except Exception:
                pass

        if single_run:
            m, last_err = None, None
            for attempt in range(1, max(0, args.retries) + 2):
                try:
                    m = run_once(cfg, q, r, s, sd, overrides=merged)
                    m['attempts'] = attempt
                    break
                except Exception as e:
                    last_err = e
                    print(f"[ERROR] {run_id} 第 {attempt} 次失败: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    if attempt <= args.retries:
                        print(f"[RETRY] 10 s 后重试 {run_id}")
                        time.sleep(10)
            if m is not None:
                all_metrics.append(m)
            else:
                failures.append({
                    'run_id': run_id, 'quantization': q, 'lora_rank': r,
                    'train_size': s, 'seed': sd, 'overrides': merged,
                    'attempts': max(0, args.retries) + 1,
                    'error': f'{type(last_err).__name__}: {last_err}',
                })
            continue

        # 批处理：为每个 run 起一个干净的子进程
        cmd = [sys.executable, script, '--quant', q, '--rank', str(r),
               '--train-size', str(s), '--seed', str(sd)]
        if merged.get('batch_size'):
            cmd += ['--batch-size', str(merged['batch_size'])]
        if merged.get('max_seq_length'):
            cmd += ['--max-seq-length', str(merged['max_seq_length'])]

        got = None
        wait_gpu_idle()   # R3-27：等上一个子进程的显存被驱动回收
        for attempt in range(1, max(0, args.retries) + 2):
            print(f"\n[SUBPROCESS] {run_id} (attempt {attempt})")
            rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
            if out_json.exists():
                try:
                    got = json.load(open(out_json, encoding='utf-8'))
                    got['attempts'] = attempt
                    break
                except Exception:
                    got = None
            print(f"[ERROR] {run_id} 子进程退出码 {rc}，未产出结果 JSON")
            if attempt <= args.retries:
                print(f"[RETRY] 10 s 后重试 {run_id}")
                time.sleep(10)
        if got is not None:
            all_metrics.append(got)
        else:
            failures.append({
                'run_id': run_id, 'quantization': q, 'lora_rank': r,
                'train_size': s, 'seed': sd, 'overrides': merged,
                'attempts': max(0, args.retries) + 1,
                'error': 'subprocess failed or produced no result JSON',
            })
            print(f"[SKIP] {run_id} 重试耗尽，已记录到 FAILED_RUNS.md")

    summary = results_dir / 'batch_summary.json'
    with open(summary, 'w', encoding='utf-8') as f:
        json.dump({'generated': datetime.now().isoformat(),
                   'runs': all_metrics, 'failures': failures}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n批次汇总已保存: {summary.relative_to(ROOT)} "
          f"（成功 {len(all_metrics)} / 失败 {len(failures)}）")

    if failures:
        with open(results_dir / 'FAILED_RUNS.md', 'w', encoding='utf-8') as f:
            f.write('# 失败/跳过的运行（须在报告中披露）\n\n')
            f.write(f'生成时间：{datetime.now().isoformat()}\n\n')
            f.write('| run_id | 尝试次数 | 错误 |\n|---|---|---|\n')
            for x in failures:
                f.write(f"| `{x['run_id']}` | {x['attempts']} | {x['error']} |\n")


if __name__ == '__main__':
    main()
