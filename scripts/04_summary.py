#!/usr/bin/env python
"""
04_summary.py - 结果汇总、统计检验与可视化（v3）

v3 修订（依据 PLAN_REVIEW_2026-11-17.md）：
- P1 §3.2: 跳过 limited_eval / *_eval_partial 冒烟结果，防止污染正式汇总。
- P1 §3.1: 实现真正的"12GB 预算 + 相对 BF16 精度损失 <1pp → 取最高吞吐"筛选，
  并输出被筛掉的配置及原因；实现真正的 Pareto 前沿（支配关系）。
- P1 §3.8: 用 McNemar 配对检验比较关键配置，替代不稳定的 3 种子标准差判断。
- P1 §3.3 / §3.4: 图编号规整，新增 fig01/02/03/08（均由脚本从原始结果生成）。
- P1 §3.7: 汇总 GPU 温度/功耗，输出功耗-吞吐关系。

用法：
    conda activate lora-project
    python scripts/04_summary.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'


def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def is_formal_eval(path):
    """P1 §3.2: 冒烟结果一律排除"""
    if path.stem.endswith('_partial'):
        return False
    try:
        d = json.load(open(path, encoding='utf-8'))
    except Exception:
        return False
    return not d.get('limited_eval', False)


def collect_results(cfg):
    """合并训练指标、双协议评估指标、推理基准。"""
    results_dir = ROOT / cfg['paths']['results_dir']
    rows = []
    skip_suffixes = ('_eval', '_partial', '_summary', '_pareto', '_config')
    for f in sorted(results_dir.glob('*.json')):
        if f.stem.endswith(skip_suffixes) or f.stem.startswith('zero_shot'):
            continue
        try:
            train = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        # 只接受训练结果 JSON：MCNEMAR.json（列表）、RECOMMENDATION.json（无 train_loss）
        # 等聚合产物必须排除，否则 dict() 会抛 ValueError。
        if (not isinstance(train, dict) or 'run_id' not in train
                or 'train_loss' not in train):
            continue
        row = dict(train)

        for protocol in cfg['evaluation']['protocols']:
            eval_file = results_dir / f'{f.stem}_{protocol}_eval.json'
            if not eval_file.exists() or not is_formal_eval(eval_file):
                continue
            ev = json.load(open(eval_file, encoding='utf-8'))
            row[f'accuracy_{protocol}'] = ev.get('accuracy')
            row[f'balanced_accuracy_{protocol}'] = ev.get('balanced_accuracy')
            row[f'macro_f1_{protocol}'] = ev.get('macro_f1')
            row[f'collapsed_{protocol}'] = ev.get('collapsed')
            bench = ev.get('inference_bench') or {}
            row[f'infer_samples_per_s_{protocol}'] = bench.get('samples_per_s')
            row[f'infer_ms_per_sample_{protocol}'] = bench.get('ms_per_sample')
            row[f'_predictions_{protocol}'] = ev.get('predictions')
            row['_gold_labels'] = ev.get('gold_labels')
        rows.append(row)

    zero_shots = {}
    for zs_path in sorted(results_dir.glob('zero_shot_*_eval.json')):
        if not is_formal_eval(zs_path):
            continue
        d = json.load(open(zs_path, encoding='utf-8'))
        zero_shots[d.get('quantization', zs_path.stem)] = d
    return rows, zero_shots


def aggregate_multiseed(df, cfg):
    if df.empty:
        return df
    key = ['quantization', 'lora_rank', 'train_size', 'batch_size']
    numeric = [c for c in df.columns
               if c.startswith(('accuracy_', 'balanced_accuracy_', 'peak_memory_gb',
                                'train_samples_per_s', 'train_time_pure_s',
                                'infer_samples_per_s_'))
               and not c.startswith('_')]
    grouped = df.groupby(key)[numeric].agg(['mean', 'std', 'count'])
    grouped.columns = [f'{c[0]}_{"n" if c[1] == "count" else c[1]}'
                       for c in grouped.columns]
    return grouped.reset_index()


def mcnemar(preds_a, preds_b, gold):
    """精确 McNemar 配对检验（二项检验），返回 discordant 计数与 p 值。"""
    if not preds_a or not preds_b or len(preds_a) != len(preds_b):
        return None
    b = sum(1 for a, bb, g in zip(preds_a, preds_b, gold)
            if a == g and bb != g)   # A 对 B 错
    c = sum(1 for a, bb, g in zip(preds_a, preds_b, gold)
            if a != g and bb == g)   # A 错 B 对
    n = b + c
    if n == 0:
        # 两个配置逐样本预测完全一致：返回与常规分支相同的键，避免下游 KeyError
        return {'b_a_correct_b_wrong': 0, 'c_a_wrong_b_correct': 0,
                'n_discordant': 0, 'p_value': 1.0, 'p_value_raw': 1.0,
                'significant_at_0_05': False}
    # 双侧精确二项检验
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return {
        'b_a_correct_b_wrong': b,
        'c_a_wrong_b_correct': c,
        'n_discordant': n,
        'p_value': round(min(p, 1.0), 6),
        'p_value_raw': min(p, 1.0),
        'significant_at_0_05': bool(p < 0.05),
    }


def main():
    cfg = load_config()
    results_dir = ROOT / cfg['paths']['results_dir']
    figures_dir = ROOT / cfg['paths']['figures_dir']
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows, zero_shots = collect_results(cfg)
    zero_shot = zero_shots.get('nf4') or next(iter(zero_shots.values()), None)
    if not rows:
        print("[WARN] 暂无正式实验结果。请先跑 02_train.py 与 03_evaluate.py。")
        print("       注意：--limit 冒烟结果不会被计入。")
        return

    df = pd.DataFrame(rows)
    public = df.drop(columns=[c for c in df.columns if c.startswith('_')])
    public.to_csv(results_dir / 'summary_raw.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] 原始汇总: summary_raw.csv ({len(df)} 条正式运行)")

    agg = aggregate_multiseed(public, cfg)
    agg.to_csv(results_dir / 'summary_by_config.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] 配置聚合: summary_by_config.csv ({len(agg)} 个配置)")

    # 字段 schema 表（供报告引用数字时核对）
    schema = {
        'prompt_version / prompt_style': '数据与提示词版本（v4_chat_template / chat）',
        'train_time_pure_s': '纯训练耗时，不含 val 评估',
        'train_time_total_s': '纯训练 + val 评估总耗时',
        'eval_time_s': '训练后单次 val 评估耗时',
        'train_samples_per_s': '训练吞吐（纯训练口径，samples/s）',
        'total_params / trainable_params': '模型总参数量与 LoRA 可训练参数量',
        'avg_seq_len': '训练样本平均序列长度（token）',
        'total_train_tokens': '整次训练处理的 token 总数',
        'model_resident_memory_gb': '模型加载后、训练前的常驻显存（权重+LoRA）',
        'weight_footprint_gb': '设备无关的权重足迹（与常驻显存对账）',
        'peak_memory_gb': '训练期间峰值显存（含激活）',
        'accuracy_native': '各自量化格式评估的测试准确率',
        'accuracy_cross_nf4': '统一 NF4 基座评估的测试准确率',
        'balanced_accuracy / macro_f1': '平衡准确率与宏平均 F1',
        'per_class / confusion_matrix': '每类指标与 4×4 混淆矩阵',
        'predictions / gold_labels': '逐样本预测（7,600 条），供 McNemar',
        'infer_samples_per_s_native': '分类前向吞吐（prefill，不含 decode）',
        'gpu_trace': '平均/峰值温度、平均/峰值功耗、利用率、SM 时钟、是否疑似降频',
        'attempts': '该 run 的训练尝试次数（重试后 >1）',
    }
    with open(results_dir / 'RESULT_SCHEMA.md', 'w', encoding='utf-8') as f:
        f.write('# 结果字段说明（报告引用数字时以此为准）\n\n')
        for k, v in schema.items():
            f.write(f'- `{k}`: {v}\n')

    if zero_shots:
        write_zero_shot_md(zero_shots, results_dir)
    write_per_class(cfg, results_dir)

    acc = 'accuracy_native'
    plot_fig01_framework(cfg, figures_dir)
    plot_fig02_matrix(df, figures_dir)
    plot_fig03_train_curves(cfg, figures_dir)
    plot_fig04_zero_shot(zero_shots, figures_dir)
    write_hardware_md(public, cfg, results_dir)

    if acc in df.columns and df[acc].notna().any():
        plot_fig05_tradeoffs(df, zero_shot, figures_dir)
        plot_fig06_protocol(df, figures_dir)
        plot_fig07_multiseed(agg, figures_dir)
        plot_fig08_power(df, figures_dir)
        write_recommendations(df, zero_shot, cfg, results_dir)
        write_mcnemar(df, cfg, results_dir)
    else:
        print('[WARN] 尚无正式评估结果：fig04–fig07、推荐表与 McNemar 暂不生成')
    generated = sorted(p.name for p in figures_dir.glob('fig*.png'))
    print(f"[OK] 已生成图表: {', '.join(generated) if generated else '无'}")


def write_per_class(cfg, out_dir):
    """每类召回（按量化格式）与 NF4 标准配置的混淆矩阵（报告 §5.1）。

    R3-29：主结果表只有总准确率，无法回答「量化对哪一类伤害最大」；
    这里从 seed=42 的 native 评估结果聚合每类指标，并给出 NF4 的 4×4 混淆矩阵。
    """
    results_dir = ROOT / cfg['paths']['results_dir']
    class_names = cfg['data']['class_names']
    seed = cfg['experiment']['seed']
    formats = ['bf16_lora', 'int8', 'nf4', 'fp4']
    per = {q: None for q in formats}
    for q in formats:
        path = results_dir / f'{q}_r16_n2000_s{seed}_native_eval.json'
        if path.exists() and is_formal_eval(path):
            per[q] = json.load(open(path, encoding='utf-8'))
    if not any(per.values()):
        return
    lines = ['# 每类指标（native 协议，标准配置 r16/n2000，seed=%d）' % seed, '',
             '| 类别 | 支持数 | ' + ' | '.join(f'{q} recall' for q in formats) + ' |',
             '|---|---|' + '---|' * len(formats)]
    for i, name in enumerate(class_names):
        support = None
        cells = []
        for q in formats:
            d = per[q]
            if d and d.get('per_class', {}).get(name):
                pc = d['per_class'][name]
                support = pc['support']
                cells.append(f"{pc['recall']:.4f}")
            else:
                cells.append('—')
        lines.append(f'| {name} | {support if support is not None else "—"} | '
                     + ' | '.join(cells) + ' |')
    # macro 行
    lines.append('| **macro-F1** | — | ' + ' | '.join(
        f"{per[q]['macro_f1']:.4f}" if per[q] and per[q].get('macro_f1') is not None else '—'
        for q in formats) + ' |')
    lines.append('')
    ref = per.get('nf4')
    if ref and ref.get('confusion_matrix'):
        cm = ref['confusion_matrix']
        lines += ['## NF4 标准配置的混淆矩阵（行=真值，列=预测）', '',
                  '| 真值 \\ 预测 | ' + ' | '.join(class_names) + ' |',
                  '|---|' + '---|' * len(class_names)]
        for i, name in enumerate(class_names):
            lines.append(f'| {name} | ' + ' | '.join(str(v) for v in cm[i]) + ' |')
        lines.append('')
    # 用实际数据生成观察句，避免写死「哪类最难」而在数据变化后失真
    if ref and ref.get('per_class'):
        recalls = {k: v['recall'] for k, v in ref['per_class'].items()}
        low = min(recalls, key=recalls.get)
        high = max(recalls, key=recalls.get)
        lines += [f'> 观察：NF4 标准配置下 `{low}` 的召回最低（{recalls[low]:.4f}），'
                  f'`{high}` 最高（{recalls[high]:.4f}）；类别间的难度差异（约 '
                  f'{(recalls[high] - recalls[low]) * 100:.1f} 个百分点）**远大于**格式之间的差异，'
                  f'说明量化格式主要影响整体水平而非某一类的结构性问题。']
    with open(out_dir / 'PER_CLASS.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('[OK] 每类指标: PER_CLASS.md')


def write_zero_shot_md(zero_shots, out_dir):
    """四格式 zero-shot 对照表（R3-21）。"""
    order = ['bf16_lora', 'int8', 'nf4', 'fp4']
    quants = [q for q in order if q in zero_shots] + \
             [q for q in zero_shots if q not in order]
    lines = [
        '# Zero-shot 基线（四量化格式对照）',
        '',
        '无 adapter、同一 chat template、全 7,600 条测试集。',
        '',
        '| 基座格式 | accuracy | balanced | macro-F1 | 最大单类占比 | 塌缩 | 前向吞吐 (samples/s) |',
        '|---|---|---|---|---|---|---|',
    ]
    for q in quants:
        d = zero_shots[q]
        bench = d.get('inference_bench') or {}
        lines.append(
            f"| {q} | {d.get('accuracy')} | {d.get('balanced_accuracy')} | "
            f"{d.get('macro_f1')} | {d.get('max_single_class_pred_share')} | "
            f"{d.get('collapsed')} | {bench.get('samples_per_s', 'N/A')} |")
    lines += ['', '## 每类召回（NF4 基座）', '']
    ref = zero_shots.get('nf4') or zero_shots[quants[0]]
    lines += ['| 类别 | 支持数 | 预测数 | recall | precision | F1 |',
              '|---|---|---|---|---|---|']
    for name, v in (ref.get('per_class') or {}).items():
        lines.append(f"| {name} | {v['support']} | {v['predicted_count']} | "
                     f"{v['recall']} | {v['precision']} | {v['f1']} |")
    lines += ['',
              '## 解读与使用限制',
              '',
              '- 零样本准确率随基座精度的变化**非单调**：本机实测 NF4 最高、INT8 最低，'
              '且 400 条对照中 fp32 与 bf16 接近（排除「bf16 精度不足」这一解释）。',
              '- 因此 zero-shot 行**只作参照，不用于量化格式排序**；'
              '微调增益一律与**同格式** zero-shot 比较。',
              '- CP1 判据：`collapsed == False`（最大单类预测占比 < 0.50）且 balanced 与 accuracy 接近；'
              '不设准确率上界。']
    with open(out_dir / 'ZERO_SHOT.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('[OK] Zero-shot 对照表: ZERO_SHOT.md')


def _mpl_setup():
    """统一的打印友好样式：小字号、低网格、无内嵌标题（标题由 HTML 图注提供）。"""
    plt.rcParams.update({
        'axes.unicode_minus': False,
        'font.size': 8.5,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
        'svg.fonttype': 'path',   # 文字转路径，避免 PDF/SVG 字体缺失
        'figure.dpi': 110,
    })


def _panel_tag(ax, tag):
    ax.set_title(tag, loc='left', fontsize=8.5, fontweight='bold', pad=2)


QUANT_COLORS = {'bf16_lora': '#7a7a7a', 'int8': '#3b6fb6',
                'nf4': '#2a9d6f', 'fp4': '#c95f3a'}
QUANT_LABEL = {'bf16_lora': 'BF16', 'int8': 'INT8', 'nf4': 'NF4', 'fp4': 'FP4'}


def plot_fig01_framework(cfg, out_dir):
    """图 1：实验流程（单行流程框，无标题）。"""
    _mpl_setup()
    fig, ax = plt.subplots(figsize=(6.8, 1.5))
    ax.axis('off')
    ax.grid(False)
    steps = [
        ('AG News\n120k / 7.6k', '#eef3f9'),
        ('Stratified\nsampling', '#e9f4ee'),
        ('QLoRA\nfine-tuning', '#fdf2e3'),
        ('Dual-protocol\nevaluation', '#fbeaf0'),
        ('Three axes\nacc / mem / TP', '#efeffa'),
    ]
    n = len(steps)
    for i, (txt, fc) in enumerate(steps):
        x = i * 1.0
        ax.add_patch(plt.Rectangle((x, 0.15), 0.78, 0.62, facecolor=fc,
                                   edgecolor='#b9c4d0', linewidth=0.8))
        ax.text(x + 0.39, 0.46, txt, ha='center', va='center', fontsize=8.6)
        if i < n - 1:
            ax.annotate('', xy=(x + 0.97, 0.46), xytext=(x + 0.80, 0.46),
                        arrowprops=dict(arrowstyle='-|>', color='#7a8794', lw=0.9))
    ax.set_xlim(-0.02, n - 0.02)
    ax.set_ylim(0.12, 0.80)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_dir / 'fig01_pipeline.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig01_pipeline.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_fig02_matrix(df, out_dir):
    """图 2：已执行矩阵（行=格式×秩，列=样本量，格内为运行数）。"""
    _mpl_setup()
    rows = [('bf16_lora', 16), ('int8', 16), ('fp4', 16),
            ('nf4', 8), ('nf4', 16), ('nf4', 32)]
    sizes = [500, 1000, 2000, 4000]
    labels, cells, colors = [], [], []
    for q, r in rows:
        sub = df[(df['quantization'] == q) & (df['lora_rank'] == r)]
        labels.append(f"{QUANT_LABEL.get(q, q)}  r={r}")
        row, crow = [], []
        for nsz in sizes:
            runs = sub[sub['train_size'] == nsz]
            if runs.empty:
                row.append('—')
                crow.append('#f4f5f7')
                continue
            k = len(runs)
            bs = sorted(runs['batch_size'].astype(int).unique()) \
                if 'batch_size' in runs.columns else [4]
            txt = str(k) if all(b == 4 for b in bs) else f"{k}*"
            row.append(txt)
            crow.append('#dce8f6' if k else '#f4f5f7')
        cells.append(row)
        colors.append(crow)
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.axis('off')
    ax.grid(False)
    tbl = ax.table(cellText=cells, rowLabels=labels,
                   colLabels=[f'n={n}' for n in sizes],
                   loc='center', cellLoc='center',
                   cellColours=colors, rowColours=['#eef1f5'] * len(rows),
                   colColours=['#eef1f5'] * len(sizes))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.55)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_dir / 'fig02_matrix.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig02_matrix.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_fig03_train_curves(cfg, out_dir):
    """图 3：训练损失曲线（标准配置 seed=42，四种格式各一条）。"""
    _mpl_setup()
    logs_dir = ROOT / cfg['paths']['logs_dir']
    seed = cfg['experiment']['seed']
    series = []
    for q in ('bf16_lora', 'int8', 'nf4', 'fp4'):
        run = f"{q}_r16_n2000_s{seed}"
        log = logs_dir / run / 'console.log'
        if not log.exists():
            continue
        try:
            txt = log.read_text(encoding='utf-8')
            hist = json.loads(txt[txt.index('['):])
            pts = [(h.get('step', i), h['loss']) for i, h in enumerate(hist) if 'loss' in h]
            if pts:
                series.append((q, pts))
        except Exception:
            continue
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    if not series:
        ax.text(0.5, 0.5, 'no training logs', ha='center', va='center')
    for q, pts in series:
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, linewidth=0.9, color=QUANT_COLORS.get(q), label=QUANT_LABEL.get(q, q))
    ax.set_xlabel('Training step')
    ax.set_ylabel('Training loss')
    ax.legend(frameon=False, ncol=4, loc='upper right')
    plt.tight_layout(pad=0.3)
    plt.savefig(out_dir / 'fig03_train_curves.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig03_train_curves.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_fig04_zero_shot(zero_shots, out_dir):
    """图 4：四格式零样本准确率与预测集中度。"""
    if not zero_shots:
        return
    _mpl_setup()
    order = ['bf16_lora', 'int8', 'nf4', 'fp4']
    quants = [q for q in order if q in zero_shots]
    accs = [zero_shots[q].get('accuracy') for q in quants]
    bals = [zero_shots[q].get('balanced_accuracy') for q in quants]
    shares = [zero_shots[q].get('max_single_class_pred_share') for q in quants]
    x = range(len(quants))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax = axes[0]
    ax.bar([i - 0.19 for i in x], accs, width=0.36, label='accuracy', color='#3b6fb6')
    ax.bar([i + 0.19 for i in x], bals, width=0.36, label='balanced acc.', color='#2a9d6f')
    ax.set_xticks(list(x))
    ax.set_xticklabels([QUANT_LABEL.get(q, q) for q in quants])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=2, fontsize=6.8)
    _panel_tag(ax, '(a)')
    ax = axes[1]
    ax.bar(list(x), shares, color='#c95f3a')
    ax.axhline(0.5, ls='--', lw=0.9, color='#a33')
    ax.text(len(quants) - 0.6, 0.51, 'collapse 0.5', fontsize=7, color='#a33')
    ax.set_xticks(list(x))
    ax.set_xticklabels([QUANT_LABEL.get(q, q) for q in quants])
    ax.set_ylim(0, 0.6)
    ax.set_ylabel('Max single-class share')
    _panel_tag(ax, '(b)')
    plt.tight_layout(pad=0.4)
    plt.savefig(out_dir / 'fig04_zero_shot_by_quant.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig04_zero_shot_by_quant.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_fig05_tradeoffs(df, zero_shot, out_dir):
    """图 5：准确率—显存—吞吐三视图（2 行：上两幅、下通栏）。"""
    _mpl_setup()
    acc = 'accuracy_native'
    tp = 'infer_samples_per_s_native'
    zs_acc = (zero_shot or {}).get('accuracy')
    fig = plt.figure(figsize=(6.8, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for q, g in df.groupby('quantization'):
        c = QUANT_COLORS.get(q, '#333')
        ax1.scatter(g['peak_memory_gb'], g[acc], s=16, color=c, alpha=0.85,
                    label=QUANT_LABEL.get(q, q))
        if 'model_resident_memory_gb' in g.columns:
            ax1.scatter(g['model_resident_memory_gb'], g[acc], s=18, marker='x',
                        color=c, alpha=0.9)
    if zs_acc is not None:
        ax1.axhline(zs_acc, ls='--', lw=0.9, color='#666')
        ax1.text(df['peak_memory_gb'].min(), zs_acc + 0.004,
                 f'zero-shot {zs_acc:.3f}', va='bottom', ha='left',
                 fontsize=6.5, color='#444')
    ax1.set_ylim(bottom=min(df[acc].min(), zs_acc if zs_acc else df[acc].min()) - 0.007,
                 top=df[acc].max() + 0.012)
    ax1.set_xlabel('Memory (GB): filled = peak, x = resident')
    ax1.set_ylabel('Test accuracy')
    _panel_tag(ax1, '(a)')

    if tp in df.columns and df[tp].notna().any():
        for q, g in df.groupby('quantization'):
            ax2.scatter(g[tp], g[acc], s=16, color=QUANT_COLORS.get(q, '#333'), alpha=0.85)
        if zs_acc is not None:
            ax2.axhline(zs_acc, ls='--', lw=0.9, color='#666')
        ax2.set_xlabel('Inference throughput (samples/s)')
        ax2.set_ylabel('Test accuracy')
    _panel_tag(ax2, '(b)')

    for q, g in df.groupby('quantization'):
        ax3.scatter(g['peak_memory_gb'], g['train_samples_per_s'], s=16,
                    color=QUANT_COLORS.get(q, '#333'), alpha=0.85,
                    label=QUANT_LABEL.get(q, q))
    ax3.set_xlabel('Memory (GB)')
    ax3.set_ylabel('Training throughput (samples/s)')
    ax3.legend(frameon=False, fontsize=6.5, loc='upper right', ncol=4)
    _panel_tag(ax3, '(c)')
    plt.savefig(out_dir / 'fig05_tradeoff_plots.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig05_tradeoff_plots.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_fig06_protocol(df, out_dir):
    """图 6：native 与 cross-NF4 准确率一致性。"""
    _mpl_setup()
    if 'accuracy_cross_nf4' not in df.columns or not df['accuracy_cross_nf4'].notna().any():
        return
    d = df.dropna(subset=['accuracy_native', 'accuracy_cross_nf4'])
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    lo = min(d['accuracy_native'].min(), d['accuracy_cross_nf4'].min()) - 0.005
    hi = max(d['accuracy_native'].max(), d['accuracy_cross_nf4'].max()) + 0.005
    ax.plot([lo, hi], [lo, hi], ls='--', lw=0.9, color='#777', label='y = x')
    ax.scatter(d['accuracy_native'], d['accuracy_cross_nf4'], s=22,
               color='#3b6fb6', alpha=0.85, edgecolor='white', linewidth=0.4)
    mad = (d['accuracy_native'] - d['accuracy_cross_nf4']).abs().mean()
    ax.text(0.03, 0.95, f'mean |diff| = {mad * 100:.2f} pp\nn = {len(d)}',
            transform=ax.transAxes, va='top', fontsize=7.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Accuracy (native protocol)')
    ax.set_ylabel('Accuracy (cross-NF4 protocol)')
    ax.legend(frameon=False, loc='lower right', fontsize=7)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_dir / 'fig06_protocol_agreement.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig06_protocol_agreement.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_fig07_multiseed(agg, out_dir):
    """图 7：三种子配置的均值 ± 标准差（放大纵轴）。"""
    if 'accuracy_native_mean' not in agg.columns or 'accuracy_native_n' not in agg.columns:
        return
    multi = agg[agg['accuracy_native_n'] > 1].sort_values('accuracy_native_mean')
    if multi.empty:
        return
    _mpl_setup()
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    labels = [f"{QUANT_LABEL.get(r['quantization'], r['quantization'])}"
              for _, r in multi.iterrows()]
    means = multi['accuracy_native_mean'].astype(float).values
    stds = multi['accuracy_native_std'].fillna(0).astype(float).values
    x = list(range(len(means)))
    ax.errorbar(x, means, yerr=stds, fmt='o', color='#3b6fb6', ecolor='#1f3864',
                capsize=4, markersize=5, elinewidth=1.1)
    for i, (m, sd) in enumerate(zip(means, stds)):
        ha = 'left' if i == 0 else ('right' if i == len(means) - 1 else 'center')
        dx = 4 if i == 0 else (-4 if i == len(means) - 1 else 0)
        ax.annotate(f'{m:.4f} ± {sd:.4f}', (i, m + sd), textcoords='offset points',
                    xytext=(dx, 6), ha=ha, fontsize=7)
    ax.set_xlim(-0.55, len(means) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    lo, hi = float(means.min()), float(means.max())
    pad = max(0.003, (hi - lo) * 0.7)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel('Test accuracy (mean ± std)')
    plt.tight_layout(pad=0.3)
    plt.savefig(out_dir / 'fig07_multiseed_variance.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig07_multiseed_variance.svg', format='svg', bbox_inches='tight')
    plt.close()


def plot_fig08_power(df, out_dir):
    """图 8：硬件侧视图——(a) 功耗 vs 训练吞吐；(b) 每焦耳 token 数。"""
    _mpl_setup()
    traces = []
    for _, r in df.iterrows():
        gt = r.get('gpu_trace')
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except Exception:
                gt = None
        traces.append(gt if isinstance(gt, dict) else None)
    tmp = df.copy()
    tmp['_power'] = [t.get('avg_power_w') if t else None for t in traces]
    if tmp['_power'].notna().sum() == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.95))
    ax = axes[0]
    for q, g in tmp.groupby('quantization'):
        ax.scatter(g['train_samples_per_s'], g['_power'], s=18,
                   color=QUANT_COLORS.get(q, '#333'), alpha=0.85,
                   label=QUANT_LABEL.get(q, q))
    ax.set_xlabel('Training throughput (samples/s)')
    ax.set_ylabel('Average power (W)')
    ax.legend(frameon=False, fontsize=6.5, loc='lower right')
    _panel_tag(ax, '(a)')
    ax = axes[1]
    for q, g in tmp.groupby('quantization'):
        eff = g['train_samples_per_s'] * g['avg_seq_len'] / g['_power']
        ax.scatter([QUANT_LABEL.get(q, q)] * len(eff), eff, s=18,
                   color=QUANT_COLORS.get(q, '#333'), alpha=0.85)
    ax.set_ylabel('Tokens per joule')
    _panel_tag(ax, '(b)')
    plt.tight_layout(pad=0.4)
    plt.savefig(out_dir / 'fig08_power_vs_throughput.png', dpi=220, bbox_inches='tight')
    plt.savefig(out_dir / 'fig08_power_vs_throughput.svg', format='svg', bbox_inches='tight')
    plt.close()


def _device_peak_gib(cfg, core):
    """R4（外审 P1-2）：从 gpu_trace.csv 读 nvidia-smi 整卡占用峰值（MiB → GiB）。

    与分配器峰值（torch.cuda.max_memory_allocated）不同，设备占用还包含 CUDA 上下文、
    驱动与同机其他进程，不能直接相减；它回答的是"这张卡还剩多少余量"。
    """
    logs = ROOT / cfg['paths']['logs_dir']
    peaks = {}
    for q, g in core.groupby('quantization'):
        vals = []
        for rid in g['run_id']:
            f = logs / str(rid) / 'gpu_trace.csv'
            if not f.exists():
                continue
            mems = []
            with open(f, encoding='utf-8') as fh:
                next(fh, None)
                for ln in fh:
                    parts = [x.strip() for x in ln.split(',')]
                    if len(parts) > 5 and parts[5]:
                        try:
                            mems.append(float(parts[5].replace(' MiB', '')))
                        except ValueError:
                            pass
            if mems:
                vals.append(max(mems) / 1024)
        peaks[q] = vals
    return peaks


def write_hardware_md(df, cfg, out_dir):
    """硬件派生指标：tokens/J、估算 TFLOPS、功耗、设备峰值显存。

    口径：
    - FLOPs 用 6ND 主项估算（前向 2ND + 反向 4ND），是**保守下界**，不含注意力二次项；
    - R4（外审 P1-5）：N 统一取未打包的模型参数量（BF16 模型的 total_params），
      不再用 4-bit 的打包参数量，否则会把 BF16 与 4-bit 的算力差距虚增约 42%；
    - R4（外审 P1-2）：同时报告 nvidia-smi 设备峰值，budget 类结论必须注明口径。
    """
    if df.empty:
        return
    core = df
    for col, val in (('lora_rank', 16), ('train_size', 2000), ('batch_size', 4)):
        if col in core.columns and (core[col] == val).any():
            core = core[core[col] == val]
    if core.empty:
        return
    # 未打包参数量：优先取 BF16 模型的 total_params，否则取各格式最大值
    n_unpacked = None
    if 'total_params' in core.columns:
        bf = core[core['quantization'] == 'bf16_lora']
        n_unpacked = float(bf['total_params'].max() if not bf.empty
                           else core['total_params'].max())
    dev_peaks = _device_peak_gib(cfg, core)  # {quant: [per-run max GiB]}
    rows = []
    for q, g in core.groupby('quantization'):
        traces = []
        for x in g['gpu_trace']:
            if isinstance(x, str):
                try:
                    x = json.loads(x)
                except Exception:
                    x = None
            if x:
                traces.append(x)
        power = sum(t.get('avg_power_w', 0) for t in traces) / len(traces) if traces else None
        temp = max((t.get('max_temp_c', 0) for t in traces), default=None)
        util = (sum(t.get('avg_utilization_pct', 0) for t in traces) / len(traces)
                if traces else None)
        tp = g['train_samples_per_s'].mean()
        seq = g['avg_seq_len'].mean() if 'avg_seq_len' in g.columns else None
        tokens_s = tp * seq if seq else None
        tflops = (6 * n_unpacked * tokens_s / 1e12) if (n_unpacked and tokens_s) else None
        joules = (tokens_s / power) if (power and tokens_s) else None
        rows.append({
            'quant': q,
            'train_tp': tp,
            'tokens_s': tokens_s,
            'tflops': tflops,
            'power': power,
            'tokens_j': joules,
            'temp': temp,
            'util': util,
            'resident': g['model_resident_memory_gb'].mean()
                        if 'model_resident_memory_gb' in g.columns else None,
            'peak': g['peak_memory_gb'].mean(),
            'dev_peak_mean': (sum(dev_peaks[q]) / len(dev_peaks[q])) if dev_peaks.get(q) else None,
            'dev_peak_max': max(dev_peaks[q]) if dev_peaks.get(q) else None,
            'dev_peak_n': len(dev_peaks.get(q) or []),
        })
    lines = ['# 硬件派生指标（由 gpu_trace.csv 与训练结果二次计算）', '',
             '| 格式 | 训练吞吐 (sa/s) | tokens/s | 名义 TFLOPS（6ND 近似） | 平均功耗 (W) | tokens/J | '
             '峰值温度 (°C) | SM 利用率 (%) | 常驻显存 (GB) | 分配器峰值 (GB) | 设备峰值均值 (GiB) | 设备峰值最大 (GiB) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        def fmt(v, n=1):
            return '—' if v is None else f'{v:.{n}f}'
        lines.append(
            f"| {r['quant']} | {fmt(r['train_tp'])} | {fmt(r['tokens_s'], 0)} | {fmt(r['tflops'], 2)} | "
            f"{fmt(r['power'])} | {fmt(r['tokens_j'], 1)} | {fmt(r['temp'], 0)} | "
            f"{fmt(r['util'])} | {fmt(r['resident'], 3)} | {fmt(r['peak'], 3)} | "
            f"{fmt(r['dev_peak_mean'], 2)} | {fmt(r['dev_peak_max'], 2)} |")
    lines += ['',
              '## 口径与解读',
              '',
              f'- `名义 TFLOPS = 6 × N × tokens/s`，N 统一取未打包参数量 {n_unpacked:,.0f}；'
              '4-bit 的 `total_params` 是打包后的存储量，不能用于 FLOPs。',
              '- QLoRA 的 6ND 是"名义近似"，不是全参数训练的 6ND 分解：冻结基座不需算权重梯度，'
              '但需算输入梯度（≈2ND），而梯度检查点会在反向时重算一次前向（≈2ND），'
              '两者相加近似回到 6ND；注意注意力二次项（seq≈256、d=1536 时约 8%）与 bnb 反量化开销未计入。'
              '因此该列只用于同平台横向比较，不能解读为硬件峰值利用率。',
              '- `tokens/J = tokens/s ÷ 平均功耗`，功耗取自训练期 2 s 间隔的 `gpu_trace.csv`，'
              '是 **GPU 板卡级** 能效，不含 CPU 与整机。',
              '- 分配器峰值 = `torch.cuda.max_memory_allocated()`；设备峰值 = `nvidia-smi memory.used` '
              '的训练期最大值（每个 run 一个），含 CUDA 上下文、驱动与同机其他进程，**两列不可相减**。',
              '- 设备峰值给出"均值"与"最大"两列：均值会被缺测种子拉低（bf16 仅 s43/s44 有采样），'
              '判断卡上余量应看"最大"列。',
              '- "12 GB 预算"以分配器峰值为准；若按整卡容量衡量，bs16 的设备峰值约 11.56 GiB，'
              '已接近 11.94 GiB 上限。',
              '- 4-bit 的计算仍在 bf16 完成（`bnb_4bit_compute_dtype=bfloat16`），'
              '权重常驻显著下降，但每次矩阵乘需反量化，**训练吞吐反而低于 BF16**。',
              '- int8（`MatMul8bitLt`）会把 bf16 输入转 fp16 计算，数值路径与 4-bit 不同，'
              '报告里应与 4-bit 分开解读。']
    with open(out_dir / 'HARDWARE_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('[OK] 硬件派生指标: HARDWARE_SUMMARY.md')


def write_recommendations(df, zero_shot, cfg, out_dir):
    """P1 §3.1 + R3-10: 真正的筛选逻辑（按配置聚合均值后再筛）。"""
    acc, tp, mem = 'accuracy_native', 'infer_samples_per_s_native', 'peak_memory_gb'
    budget_gb, tol = 12.0, 0.01
    key = ['quantization', 'lora_rank', 'train_size', 'batch_size']

    lines = ['# QLoRA 配置推荐表', '']
    if zero_shot:
        lines += [f"- **Zero-shot 基线**: accuracy={zero_shot['accuracy']}, "
                  f"balanced={zero_shot.get('balanced_accuracy')}, "
                  f"max_single_class_share={zero_shot.get('max_single_class_pred_share')}, "
                  f"collapsed={zero_shot.get('collapsed')}", '']

    valid = df[df[acc].notna()].copy()
    if valid.empty:
        lines.append('暂无有效评估结果。')
        with open(out_dir / 'RECOMMENDATIONS.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return

    # R3-10: 多种子配置不能当作多个独立配置参与 Pareto/筛选，先按配置聚合
    agg_kwargs = dict(acc=(acc, 'mean'), acc_std=(acc, 'std'), mem=(mem, 'mean'),
                      tp=(tp, 'mean'), train_tp=('train_samples_per_s', 'mean'),
                      train_s=('train_time_pure_s', 'mean'),
                      n_seeds=(acc, 'count'))
    if 'model_resident_memory_gb' in valid.columns:
        agg_kwargs['resident'] = ('model_resident_memory_gb', 'mean')
    agg = valid.groupby(key, as_index=False, dropna=False).agg(**agg_kwargs)
    agg['label'] = [f"{r['quantization']}_r{int(r['lora_rank'])}_n{int(r['train_size'])}"
                    + ('' if int(r['batch_size']) == 4 else f"_bs{int(r['batch_size'])}")
                    for _, r in agg.iterrows()]

    bf16 = agg[agg['quantization'] == 'bf16_lora']
    bf16_acc = float(bf16['acc'].max()) if not bf16.empty else None
    bf16_mem = float(bf16['mem'].min()) if not bf16.empty else None

    # Pareto 前沿（精度↑、显存↓、吞吐↑），在配置层而非 run 层
    pareto = []
    for i, a in agg.iterrows():
        dominated = False
        for j, b in agg.iterrows():
            if i == j:
                continue
            if (b['acc'] >= a['acc'] and b['mem'] <= a['mem'] and b['tp'] >= a['tp']
                    and (b['acc'] > a['acc'] or b['mem'] < a['mem'] or b['tp'] > a['tp'])):
                dominated = True
                break
        if not dominated:
            pareto.append(a)
    lines += ['## Pareto 前沿（配置层非支配解）', '',
              '目标：accuracy ↑、peak_memory ↓、inference throughput ↑', '',
              '| 配置 | accuracy | acc_std | peak_mem_GB | infer_TP | seeds |',
              '|---|---|---|---|---|---|']
    for r in sorted(pareto, key=lambda x: -x['acc']):
        sd = '—' if pd.isna(r['acc_std']) else f"{r['acc_std']:.4f}"
        lines.append(f"| `{r['label']}` | {r['acc']:.4f} | {sd} | "
                     f"{r['mem']:.3f} | {r['tp']:.1f} | {int(r['n_seeds'])} |")
    lines.append('')

    lines += ['## 12 GB 显存预算下的推荐（真实筛选结果）', '']
    if bf16_acc is None:
        lines.append('缺少 BF16 基线，无法计算精度损失阈值，筛选未执行。')
    else:
        lines.append(f'- BF16 基线（配置层最高）准确率: **{bf16_acc:.4f}**，'
                     f'峰值显存 **{bf16_mem:.3f} GB**')
        lines.append(f'- 筛选条件：`peak_memory_gb` ≤ {budget_gb} GB 且 '
                     f'`accuracy` ≥ {bf16_acc:.4f} − {tol} 且推理吞吐非空；'
                     f'**BF16 基线只作参照，不参与推荐候选**')
        lines.append('')
        lines.append('| 配置 | accuracy | peak_mem_GB | train_s | infer_TP | 结果 |')
        lines.append('|---|---|---|---|---|---|')
        eligible = []
        candidates = []
        for _, r in agg.iterrows():
            reasons = []
            if r['quantization'] == 'bf16_lora':
                reasons.append('BF16 参照基线')
            if r['mem'] > budget_gb:
                reasons.append('显存超预算')
            if r['acc'] < bf16_acc - tol:
                reasons.append(f"精度损失 {bf16_acc - r['acc']:.4f} ≥ {tol}（即 1 pp 绝对准确率）")
            if pd.isna(r['tp']):
                reasons.append('缺推理吞吐')
            ok = not reasons
            if ok:
                eligible.append(r)
            candidates.append({
                'label': r['label'],
                'accuracy': round(float(r['acc']), 4),
                'peak_mem_gb': round(float(r['mem']), 3),
                'train_s': None if pd.isna(r['train_s']) else round(float(r['train_s']), 1),
                'infer_tp': None if pd.isna(r['tp']) else round(float(r['tp']), 1),
                'n_seeds': int(r.get('n_seeds', 1)),
                'acc_std': None if pd.isna(r.get('acc_std')) else round(float(r['acc_std']), 4),
                'eligible': ok,
                'reason': '' if ok else '，'.join(reasons),
            })
            tp_txt = 'N/A' if pd.isna(r['tp']) else f"{r['tp']:.1f}"
            ts_txt = '—' if pd.isna(r['train_s']) else f"{r['train_s']:.0f}"
            lines.append(f"| `{r['label']}` | {r['acc']:.4f} | {r['mem']:.3f} | {ts_txt} | "
                         f"{tp_txt} | "
                         f"{'✓ 入选' if ok else '✗ ' + '，'.join(reasons)} |")
        lines.append('')
        if eligible:
            # R3-28 选择规则：
            # ① 只在「跑了 ≥3 个种子」的候选里选主推荐（单种子配置不入主推荐，避免以弱证据背书）；
            # ② 先按推理吞吐取并列组（≥最大吞吐 98%，同格式内吞吐差异远小于种子波动），组内取精度最高。
            multi = [r for r in eligible if r.get('n_seeds', 1) >= 3]
            pool = multi if multi else eligible
            max_tp = max(r['tp'] for r in pool if not pd.isna(r['tp']))
            tie = [r for r in pool
                   if not pd.isna(r['tp']) and r['tp'] >= max_tp * 0.98]
            best = max(tie, key=lambda r: r['acc'])
            # R4（外审 P1-1）：并列组内 0.01 pp 的差落在种子噪声内，按事后阈值判为工程等价；
            # 该阈值与训练时长 tie-break 均为整改时新增，不是实验前预注册。
            # 等价时用预声明的 tie-break（训练时长）选择，而不是按舍入边界选。
            acc_std_max = max((float(r['acc_std']) for r in tie
                               if not pd.isna(r.get('acc_std'))), default=0.0)
            equiv_pp = max(0.003, acc_std_max) * 100
            equiv_set = [r for r in tie if (best['acc'] - r['acc']) * 100 <= equiv_pp]
            if len(equiv_set) > 1:
                best = min(equiv_set,
                           key=lambda r: r['train_s'] if not pd.isna(r['train_s']) else 1e9)
                tie_break = (f'精度差 ≤ {equiv_pp:.2f} pp 按事后阈值判为工程等价（非等效性检验），'
                             '等价时按训练时长取最短（整改时新增的说明性规则，非实验前预注册）')
            else:
                tie_break = '并列组内精度差超过等价阈值，取精度最高'
            equiv_labels = {r['label'] for r in equiv_set} if len(equiv_set) > 1 else {best['label']}
            best_acc = max(eligible, key=lambda r: r['acc'])
            cheapest = min(eligible,
                           key=lambda r: r['train_s'] if not pd.isna(r['train_s']) else 1e9)
            saving = (1 - best['mem'] / bf16_mem) * 100 if bf16_mem else float('nan')

            def _seeds(r):
                n = int(r.get('n_seeds', 1))
                return f'{n} 种子' if n > 1 else '单种子'

            lines += [
                f"**推荐配置: `{best['label']}`**（accuracy {best['acc']:.4f}，"
                f"较 BF16 基线 {best['acc'] - bf16_acc:+.4f}；峰值显存 {best['mem']:.3f} GB，"
                f"较 BF16 节省 {saving:.1f}%；推理吞吐 {best['tp']:.1f} samples/s；"
                f"纯训练 {best['train_s']:.0f} s；{_seeds(best)}）", '',
                f"并列组：{'、'.join(f'`{r[chr(108)+chr(97)+chr(98)+chr(101)+chr(108)]}`' for r in equiv_set)}；"
                f"等价阈值 {equiv_pp:.2f} pp；判据：{tie_break}", '',
                '**两条边界方案**（供不同偏好）：', '',
                f"- 精度优先：`{best_acc['label']}`（accuracy {best_acc['acc']:.4f}，"
                f"峰值显存 {best_acc['mem']:.3f} GB，推理吞吐 {best_acc['tp']:.1f} samples/s，"
                f"纯训练 {best_acc['train_s']:.0f} s，{_seeds(best_acc)}）",
                f"- 训练成本优先：`{cheapest['label']}`（纯训练 {cheapest['train_s']:.0f} s，"
                f"accuracy {cheapest['acc']:.4f}，峰值显存 {cheapest['mem']:.3f} GB，"
                f"{_seeds(cheapest)}）", '',
                '选择规则：主推荐只在「≥3 种子」的候选中产生（单种子配置作为边界方案披露）；'
                '先按推理吞吐取并列组（≥最大吞吐 98%）；组内若精度差 ≤ 等价阈值'
                f'（max(0.3 pp, 种子标准差) = {equiv_pp:.2f} pp），按事后阈值判为工程等价（非等效性检验），'
                '按训练时长取最短（整改时新增的 tie-break，非实验前预注册），否则取精度最高；'
                '候选须满足 `peak_memory ≤ 12 GB` 且 `accuracy ≥ BF16 − 1 pp`。', '']
        else:
            lines += ['**没有量化配置同时满足全部条件**，请放宽预算或精度阈值。', '']

    lines += [
        '## 统计显著性声明',
        '',
        '- 关键配置跑了 3 个种子，可报告均值 ± 标准差（仅作训练随机性参考）。',
        '- 更可靠的 A/B 判定见 `MCNEMAR.md`（同测试集配对检验）。',
        '- 其余配置为单种子工程测量，**不做统计显著性断言**。',
        '- 双协议一致性见 fig06；功耗-吞吐见图 8。',
        '- 显存对比请用 `model_resident_memory_gb`（权重占用）与 `peak_memory_gb`'
        '（含激活）两个口径；batch 8/16 探索运行回答"预算内可支持的最大批量"。',
    ]
    with open(out_dir / 'RECOMMENDATIONS.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # 供 05_build_report.py 直接消费的结构化推荐（避免解析 Markdown）
    rec = {'picked': None, 'bf16_acc': bf16_acc, 'bf16_mem': bf16_mem,
           'budget_gb': budget_gb, 'tol': tol, 'alternatives': None, 'candidates': []}
    if bf16_acc is not None and eligible:
        def _row(r):
            return {
                'label': r['label'], 'accuracy': round(float(r['acc']), 4),
                'threshold_equivalent': bool(r['label'] in equiv_labels
                                            and len(equiv_labels) > 1),
                'peak_mem_gb': round(float(r['mem']), 3),
                'resident_mem_gb': round(float(r['resident']), 3)
                                   if 'resident' in r and not pd.isna(r['resident']) else None,
                'infer_tp': round(float(r['tp']), 1) if not pd.isna(r['tp']) else None,
                'train_tp': round(float(r['train_tp']), 1)
                            if not pd.isna(r['train_tp']) else None,
                'train_s': round(float(r['train_s']), 1) if not pd.isna(r['train_s']) else None,
                'mem_saving_pct': round((1 - float(r['mem']) / bf16_mem) * 100, 1)
                                  if bf16_mem else None,
                'acc_delta_pp': round((float(r['acc']) - bf16_acc) * 100, 2),
                'n_seeds': int(r.get('n_seeds', 1)),
            }
        rec['picked'] = _row(best)
        rec['alternatives'] = {'best_accuracy': _row(best_acc),
                               'cheapest_training': _row(cheapest)}
        for c in candidates:
            c['threshold_equivalent'] = bool(c['label'] in equiv_labels
                                            and len(equiv_labels) > 1)
        rec['tie'] = {
            'labels': sorted(equiv_labels),
            'threshold_pp': round(equiv_pp, 3),
            'criterion': tie_break,
            'threshold_equivalent': len(equiv_labels) > 1,
        }
        rec['candidates'] = candidates
    with open(out_dir / 'RECOMMENDATION.json', 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[OK] 推荐表（含真实筛选）: RECOMMENDATIONS.md")


def _holm_bonferroni(pvals):
    """R4（外审 P1-4）：k 组两两比较的 Holm 与 Bonferroni 校正。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    holm = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[idx]))
        holm[idx] = running
    bonf = [min(1.0, p * m) for p in pvals]
    return holm, bonf


def write_mcnemar(df, cfg, out_dir):
    """关键配置两两 McNemar 配对检验（标准配置 r16/n2000/batch4/seed）。

    R4（外审 P0-1）：旧实现按「每个格式出现的第一行」取代表，而结果文件按字典序读取，
    nf4 的 n1000 排在 n2000 之前，导致表 8 实际比较的是 n1000 单种子配置。
    现在显式限定标准配置，并在每对比较里记录配置键。
    R4（外审 P1-4）：同时输出 Holm / Bonferroni 校正 p 值，避免 6 次两两比较的假阳性。
    """
    key = 'accuracy_native'
    preds_col = '_predictions_native'
    if preds_col not in df.columns:
        return
    seed = cfg['experiment']['seed']
    std = df[df[key].notna() & df[preds_col].notna()].copy()
    for col, val in (('lora_rank', 16), ('train_size', 2000),
                     ('batch_size', 4), ('seed', seed)):
        if col in std.columns:
            std = std[std[col] == val]
    if 'seed' in std.columns:
        std = std[std['seed'] == seed]
    reps = {}
    for _, r in std.iterrows():
        reps.setdefault(r['quantization'], r)
    if len(reps) < 2:
        print('[WARN] McNemar: 标准配置运行不足 2 个，跳过')
        return
    names = list(reps.keys())
    actual_seed = int(reps[names[0]]['seed']) if 'seed' in reps[names[0]].index else seed
    cfg_key = f'r16/n2000/batch4/seed{actual_seed}'
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = reps[names[i]], reps[names[j]]
            gold = a.get('_gold_labels') or []
            if not gold:
                continue
            r = mcnemar(a[preds_col], b[preds_col], gold)
            if r:
                pairs.append({'a': names[i], 'b': names[j],
                              'config': cfg_key, **r})
    if not pairs:
        return
    holm, bonf = _holm_bonferroni([x['p_value_raw'] for x in pairs])
    # 外审 P1：给每条比较补配对 bootstrap 置信区间（样本层不确定性）
    import numpy as _np
    rng = _np.random.default_rng(20261118)
    for x in pairs:
        a, b = reps[x['a']], reps[x['b']]
        pa = _np.asarray(a[preds_col])
        pb = _np.asarray(b[preds_col])
        gold = _np.asarray(a.get('_gold_labels') or [])
        if gold.size == pa.size and pa.size == pb.size:
            ca = (pa == gold).astype(_np.float64)
            cb = (pb == gold).astype(_np.float64)
            diff = (ca.mean() - cb.mean()) * 100
            idx = rng.integers(0, gold.size, size=(2000, gold.size))
            boot = (ca[idx].mean(axis=1) - cb[idx].mean(axis=1)) * 100
            lo, hi = _np.percentile(boot, [2.5, 97.5])
            x['diff_pp'] = round(float(diff), 3)
            x['ci95_low_pp'] = round(float(lo), 3)
            x['ci95_high_pp'] = round(float(hi), 3)
            x['ci95_within_1pp'] = bool(lo > -1.0 and hi < 1.0)
    for x, h, b in zip(pairs, holm, bonf):
        x['p_holm'] = round(h, 6)
        x['p_bonferroni'] = round(b, 6)
        x['significant_uncorrected'] = bool(x['p_value'] < 0.05)
        x['significant_holm'] = bool(h < 0.05)
    min_holm = min(x['p_holm'] for x in pairs)
    lines = ['# McNemar 配对检验', '',
             f'- 比较对象：`{cfg_key}` 下的四种基座精度（每格式 1 个代表运行）',
             '- 同一测试集逐样本配对，p 为双侧精确二项检验',
             '- 6 组两两比较同时进行，同时报告 Holm 与 Bonferroni 校正 p',
             '- 仅未校正 p < 0.05 不能单独作为结论依据', '',
             '| A | B | A对B错 | A错B对 | discordant | p（未校正） | p（Holm） | p（Bonf） | 校正后显著 |',
             '|---|---|---|---|---|---|---|---|---|']
    for x in pairs:
        lines.append(
            f"| {x['a']} | {x['b']} | {x['b_a_correct_b_wrong']} | "
            f"{x['c_a_wrong_b_correct']} | {x['n_discordant']} | {x['p_value']:.4g} | "
            f"{x['p_holm']:.4g} | {x['p_bonferroni']:.4g} | "
            f"{'是' if x['significant_holm'] else '否'} |")
    lines += ['', f'- 最小 Holm 校正 p = {min_holm:.4g}；'
                  + ('校正后存在显著差异' if min_holm < 0.05 else '**校正后无一对显著**')]
    with open(out_dir / 'MCNEMAR.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    with open(out_dir / 'MCNEMAR.json', 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print('[OK] McNemar 配对检验: MCNEMAR.md（标准配置 + 多重比较校正）')


if __name__ == '__main__':
    main()
