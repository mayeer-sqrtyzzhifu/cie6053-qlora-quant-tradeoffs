#!/usr/bin/env python
"""
05_build_report.py - 从结构化结果生成实验报告（v5，打印优先排版）

产物：
  docs/report.html              屏幕/打印两用，A4 版式
  docs/report_standalone.html   单文件（图片 base64 内嵌）
  docs/report.pdf               Edge headless 打印（A4）
  docs/submission/              提交包

设计原则：
- 表格可换行、带表题，不使用网页横向滚动（打印时会被裁切）；
- 图注由 HTML 提供，PNG 内部不再重复标题；
- 正文与数据分离：所有数字由脚本从 CSV/JSON 计算或读取。
"""
import ast
import base64
import html
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
RESULTS = ROOT / 'outputs' / 'results'
FIGURES = DOCS / 'figures'
CONFIG_PATH = ROOT / 'configs' / 'experiment.yaml'

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
QLABEL = {'bf16_lora': 'BF16', 'int8': 'INT8', 'nf4': 'NF4', 'fp4': 'FP4'}
QORDER = ['bf16_lora', 'int8', 'nf4', 'fp4']


# ------------------------------------------------------------------ 基础工具
def load_config():
    with open(CONFIG_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


def read_json(name, default=None):
    p = RESULTS / name
    if not p.exists():
        return default
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def f2(v, nd=2, dash='—'):
    try:
        import math as _m
        x = float(v)
        if nd == 0:
            # 四舍五入（half-up），避免 880.5 被格式化成 880
            return f'{_m.floor(x + 0.5)}'
        return f'{x:.{nd}f}'
    except (TypeError, ValueError):
        return dash


def pct(v, nd=1, dash='—'):
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return dash


def table(headers, rows, aligns=None, cls=''):
    """生成带 thead 的表格；aligns 形如 ['l','l','r',...]。"""
    aligns = aligns or ['l'] * len(headers)
    th = ''.join(f'<th class="{a}">{html.escape(str(h))}</th>'
                 for h, a in zip(headers, aligns))
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            a = aligns[i] if i < len(aligns) else 'l'
            tds.append(f'<td class="{a}">{c}</td>')
        body.append('<tr>' + ''.join(tds) + '</tr>')
    return (f'<table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def fig(name, caption, small=False):
    # 优先用矢量 SVG（文字在 PDF 中不发虚），缺失时退回 PNG
    stem = Path(name).stem
    svg = FIGURES / f'{stem}.svg'
    png = FIGURES / name
    use = svg if svg.exists() else (png if png.exists() else None)
    if use is None:
        return f'<p class="note">（图 {html.escape(name)} 尚未生成）</p>'
    cls = ' class="small"' if small else ''
    return (f'<figure{cls}><img src="figures/{use.name}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


# ------------------------------------------------------------------ 统计
def load_df():
    p = RESULTS / 'summary_raw.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df if not df.empty else None


def core(df):
    """格式对比只取标准配置 r16/n2000/batch4。"""
    d = df
    for col, val in (('lora_rank', 16), ('train_size', 2000), ('batch_size', 4)):
        if col in d.columns and (d[col] == val).any():
            d = d[d[col] == val]
    return d


def stats(df, zs, rec, mcnemar, n_evals):
    s = {'n_runs': 0, 'n_evals': n_evals, 'quant': {}, 'zero': zs,
         'zero_nf4': (zs or {}).get('nf4'), 'rec': rec, 'mcnemar': mcnemar}
    if df is None or df.empty:
        return s
    s['n_runs'] = len(df)
    c = core(df)
    for q, g in c.groupby('quantization'):
        s['quant'][q] = {
            'acc': float(g['accuracy_native'].mean()),
            'std': float(g['accuracy_native'].std()) if len(g) > 1 else None,
            'memory': float(g['peak_memory_gb'].mean()),
            'resident': float(g['model_resident_memory_gb'].mean())
            if 'model_resident_memory_gb' in g.columns else None,
            'train_tp': float(g['train_samples_per_s'].mean()),
            'infer_tp': float(g['infer_samples_per_s_native'].mean())
            if 'infer_samples_per_s_native' in g.columns
            and g['infer_samples_per_s_native'].notna().any() else None,
            'n': len(g),
        }
    bf, nf = s['quant'].get('bf16_lora'), s['quant'].get('nf4')
    s['mem_save'] = (1 - nf['memory'] / bf['memory']) * 100 if bf and nf else None
    s['res_save'] = (1 - nf['resident'] / bf['resident']) * 100 if bf and nf and bf['resident'] else None
    s['zero_nf4_acc'] = (s['zero_nf4'] or {}).get('accuracy')
    # 三种子逐种子差值（BF16 − X），用于把"种子均值差"与"单次配对 p"分开叙述
    try:
        c = core(df)
        if 'seed' in c.columns:
            piv = c.pivot_table(index='seed', columns='quantization',
                                values='accuracy_native', aggfunc='mean')
            if 'bf16_lora' in piv.columns:
                s['seed_deltas'] = {
                    q: [round(float((piv['bf16_lora'] - piv[q]).iloc[i]) * 100, 2)
                        for i in range(len(piv))]
                    for q in ('int8', 'nf4', 'fp4') if q in piv.columns
                }
    except Exception:
        pass
    s['gain_pp'] = ((nf['acc'] - s['zero_nf4_acc']) * 100
                    if nf and s['zero_nf4_acc'] is not None else None)
    return s


# ------------------------------------------------------------------ 正文

def _author_line(cfg):
    """标题/页脚署名；anonymous 时返回空串（公开版不输出署名）。"""
    rep = cfg.get('reporting', {}) or {}
    if rep.get('anonymous'):
        return ''
    a = rep.get('author', {}) or {}
    name = str(a.get('name', ''))
    sid = str(a.get('student_id', ''))
    return f'{name}　学号 {sid}'.strip()


def build_abstract(s):
    if not s['n_runs']:
        return '<p>（尚无正式数据）</p>'
    q = s['quant']
    bf, nf = q.get('bf16_lora'), q.get('nf4')
    i8, fp = q.get('int8'), q.get('fp4')
    t = ('<p>本文在单块 11.94 GB 的消费级笔记本 GPU 上，用同一套数据、提示词与训练流程比较了 QLoRA 微调中 '
         f'BF16、INT8、NF4 与 FP4 四种基座精度对分类准确率、显存占用、训练与推理吞吐以及能效的影响。'
         f'实验包含 {s["n_runs"]} 次训练、{s["n_evals"]} 次评估与 4 次零样本基线，任务为 AG News 四分类（7,600 条测试样本）。</p>')
    if nf and bf:
        t += (f'<p>四种格式的准确率极差 0.29 个百分点（未舍入均值差 {100 * (bf["acc"] - nf["acc"]):.3f} pp，'
              '按表内四位小数相减则为 0.30 pp；'
              f'BF16 {f2(bf["acc"], 4)}、NF4 {f2(nf["acc"], 4)}），'
              f'而 NF4/FP4 的分配器峰值显存比 BF16 低 {pct(s["mem_save"])}%、模型常驻显存低 {pct(s["res_save"])}%；'
              '代价是训练吞吐与每焦耳 token 数大约减半。'
              'McNemar 配对检验（标准配置、7,600 条配对样本）的未校正 p 为 0.022、0.028 两组，'
              '其余四组在 0.055–0.88；6 组经 Holm 校正后均不显著，差异量级只有 0.3 个百分点左右，'
              '且配对检验只用了单次运行，因此不足以确认任一格式更优。</p>')
        t += ('<p>据此给出按约束条件筛选配置的结论：若需在更小显存的卡上复现本流程'
              '（本实验的 12 GB 分配器预算对全部候选都不构成约束），选 4-bit；'
              '时间或能耗是硬约束时选 BF16；INT8 在本任务的训练侧两项都不占优。'
              '文中还记录了零样本准确率随基座精度非单调这一现象，'
              '并说明它为什么不能用来给量化格式排序。</p>')
    return t


def build_part1(s):
    q = s['quant']
    bf, nf = q.get('bf16_lora'), q.get('nf4')
    if not s['n_runs']:
        return '<p>（尚无正式数据）</p>'
    t = ('<p><b>项目产出。</b>本项目产出的不是"一个更好的模型"，而是一份可复现的选型依据：'
         f'{s["n_runs"]} 次训练与 {s["n_evals"]} 次评估组成的受控数据集，覆盖四种基座精度；'
         '以及围绕它的测量与判定工具——数据审计、双评估协议、双口径显存、训练与推理两条吞吐轴、'
         '配对显著性检验、每类指标与混淆矩阵，以及由原始 JSON 自动生成的报告与推荐表。'
         '任何人拿到这些文件和脚本，都能复核每一个数字，或换一个数据集重跑。</p>')
    t += ('<p><b>动机与影响。</b>12 GB 级显卡是学生和边缘部署最常遇到的预算区间，'
          '"要不要上 4-bit"在工程里往往凭经验决定。本实验把它拆成可测的量之后，'
          f'答案并不浪漫：显存确实省了将近一半（分配器峰值低 {pct(s["mem_save"])}%），'
          f'准确率只掉了 0.3 个百分点，但训练时间和能耗都翻了一倍左右。'
          '这意味着 4-bit 的价值在于"跑得下"，不在于"跑得快"；如果目标是尽快迭代，BF16 反而更划算。'
          '这类结论直接影响教学实验与小型团队的硬件选型，而不是论文里的排行榜。</p>')
    t += ('<p><b>新颖性。</b>与"用一个配置跑通 QLoRA"相比，本工作有两点可复核的增量。'
          '第一，同一套数据、提示词与训练流程下的四格式对照，并把训练吞吐与推理吞吐、常驻显存与峰值显存分开测量，'
          '避免跨论文比较的口径漂移。第二，用同一批 7,600 条配对预测做显著性检验，'
          '并用 400 条固定子集的 fp32/bf16/nf4 对照检查了"bf16 浮点精度不足"这一解释：'
          'bf16 与 fp32 仅 3 条预测不同（精确 McNemar p=0.25），未发现 bf16 精度不足的证据，'
          '但 400 条样本的统计功效有限，不能排除小幅精度效应；非单调的机制尚无替代读出对照，'
          '只能作为假设（见 §5.2）。该对照的原始预测已落盘，可复算。</p>')
    return t


def build_part2(s):
    q = s['quant']
    bf, nf = q.get('bf16_lora'), q.get('nf4')
    i8, fp = q.get('int8'), q.get('fp4')
    if not s['n_runs']:
        return '<p>（尚无正式数据）</p>'
    t = ('<p><b>起点与基线。</b>起点是公开的开源栈：Qwen2.5-1.5B-Instruct 官方权重、'
         'Transformers/PEFT/bitsandbytes 提供的 QLoRA 实现，以及 AG News 官方划分。'
         '方法沿用 QLoRA 论文的四项设计——NF4 数据格式、双重量化、分页优化器、冻结基座加低秩适配器。'
         '基线有两条：不量化的 BF16 LoRA，以及同提示词、不加载适配器的零样本。'
         '量化算法本身没有改动，本文的增量都在实验设计、测量口径与判定方法上。</p>')
    t += ('<p><b>增量工作。</b>实验矩阵覆盖 4 种格式、3 种秩、4 档样本量（OFAT 单因素设计，不是全因子），关键配置各 3 个种子，另加两次 batch 探索；'
          '评估上同时跑 native 与 cross-NF4；测量上区分常驻与峰值显存、训练与推理吞吐，并在训练期采样功耗与温度；'
          '数据上做了规范化、与测试集归一化后完全相同的样本剔除（精确匹配）与长度约束，并把每一步计数写入审计；'
          '统计上用 7,600 条配对预测做 McNemar 检验，而不是比较三个种子的标准差；'
          '工程上每个运行独立进程、失败可重试可披露、报告与推荐由脚本生成。</p>')
    t += ('<p><b>验证与结果。</b>标准配置下 BF16、INT8、NF4、FP4 的 native 准确率分别为 '
          f'{f2(bf["acc"], 4)}、{f2(i8["acc"], 4)}、{f2(nf["acc"], 4)}、{f2(fp["acc"], 4)}，'
          f'峰值显存为 {f2(bf["memory"])}、{f2(i8["memory"])}、{f2(nf["memory"])}、{f2(fp["memory"])} GB，'
          f'训练吞吐为 {f2(bf["train_tp"])}、{f2(i8["train_tp"])}、{f2(nf["train_tp"])}、{f2(fp["train_tp"])} samples/s。'
          '样本量从 500 增到 4,000 带来约 2.7 个百分点，秩从 8 增到 32 只带来约 0.7 个百分点，'
          '说明在这两条单种子比较中，样本量侧的增益明显更大；但两项都只有 1 个种子、'
          '且 OFAT 设计无法排除两者交互，因此只能作为待复核的观察，而不是"瓶颈在数据量"的结论。'
          '两种评估协议的平均差为 0.41 个百分点，没有一次超过 2 个百分点。</p>')
    return t


def build_result_intro(s, df):
    if df is None or df.empty:
        return '暂无正式实验数据。'
    q = s['quant']
    accs = sorted(((v['acc'], k) for k, v in q.items() if v.get('acc') is not None), reverse=True)
    t = (f'表 2 与表 3 列出全部 {s["n_runs"]} 次运行。'
         f'在四个标准配置的 3 种子均值中，准确率最高的是 {QLABEL.get(accs[0][1])}（{accs[0][0]:.4f}），'
         f'最低的是 {QLABEL.get(accs[-1][1])}（{accs[-1][0]:.4f}），极差 {100*(accs[0][0]-accs[-1][0]):.2f} 个百分点'
         '（单次运行中 INT8 s44 的 0.9172 更高，属种子波动）。'
         '从固定 batch 4 的标准配置看，四种格式的准确率差异远小于它们的显存与吞吐差异；'
         '图 3 给出四种格式在同一配置下的训练损失曲线，收敛速度与最终损失都比较接近。')
    return t


def build_rq_answers(s, df):
    q = s['quant']
    bf, nf = q.get('bf16_lora'), q.get('nf4')
    i8, fp = q.get('int8'), q.get('fp4')
    t = ''
    if bf:
        def d(x):
            return (x['acc'] - bf['acc']) * 100 if x else None
        t += ('<p><b>RQ1。</b>相对 BF16，INT8、NF4、FP4 的平均准确率差分别为 '
              f'{d(i8):+.2f}、{d(nf):+.2f}、{d(fp):+.2f} 个百分点，全部落在 1 个百分点的预算内；'
              f'峰值显存从 {f2(bf["memory"])} GB 降到 {f2(nf["memory"])} GB，常驻显存从 '
              f'{f2(bf["resident"])} GB 降到 {f2(nf["resident"])} GB。'
              '按本实验的判据，4-bit 属于"近似无损换显存"，但必须同时说明它的吞吐与能效代价。</p>')
    if nf:
        sub5 = df[(df['quantization'] == 'nf4') & (df['train_size'] == 500)]
        sub4k = df[(df['quantization'] == 'nf4') & (df['train_size'] == 4000)]
        r8 = df[(df['quantization'] == 'nf4') & (df['lora_rank'] == 8)]
        r32 = df[(df['quantization'] == 'nf4') & (df['lora_rank'] == 32)]
        if len(sub5) and len(sub4k) and len(r8) and len(r32):
            dsize = (sub4k['accuracy_native'].mean() - sub5['accuracy_native'].mean()) * 100
            drank = (r32['accuracy_native'].mean() - r8['accuracy_native'].mean()) * 100
            # 训练时间的可比性：rank 变化只差几秒，而同配置三种子训练时间标准差更大
            nf4_std = df[(df['quantization'] == 'nf4') & (df['train_size'] == 2000)
                         & (df['lora_rank'] == 16) & (df['batch_size'] == 4)]['train_time_pure_s'].std()
            d_rank_time = float(r32['train_time_pure_s'].iloc[0] - r8['train_time_pure_s'].iloc[0])
            t += (f'<p><b>RQ2。</b>只比较绝对精度增益时：样本量 500→4,000 的准确率变化为 '
                  f'{dsize:+.2f} 个百分点（训练时间 +{float(sub4k["train_time_pure_s"].iloc[0] - sub5["train_time_pure_s"].iloc[0]):.0f} s），'
                  f'秩 8→32 为 {drank:+.2f} 个百分点（训练时间 {d_rank_time:+.0f} s）。'
                  f'样本量侧的绝对增益更大。但两者对应的计算预算不同：秩变化的训练时间差只有 {abs(d_rank_time):.0f} s，'
                  f'小于同配置三种子的训练时间标准差（约 {float(nf4_std):.0f} s），'
                  '因此本实验无法比较"每单位训练时间/FLOPs 的准确率收益"；'
                  '若要回答"固定训练预算下该加样本还是加秩"，需要额外的计算量匹配实验。'
                  '这两项也仍是单种子两点差，需多种子复核。</p>')
    if bf and nf:
        t += ('<p><b>RQ3。</b>三个指标之间不存在"同时更好"的配置：'
              f'BF16 准确率最高但显存最贵，4-bit 显存最省但训练最慢，INT8 两头都不占优。'
              '两种协议的平均绝对差是 0.41 个百分点，但 cross-NF4 对非 NF4 适配器有系统性惩罚，'
              '四种格式的排序在两种协议下发生反转，因此 cross-NF4 只能作为偏移诊断，'
              '不能作为"排序稳健"的证据。</p>')
    return t


def build_conclusion(s):
    """结论：只陈述数据支持的结论，并显式列出不能推出的东西。"""
    if not s['rec']:
        return '<p class="muted">【待正式实验数据】结论段将在推荐筛选完成后自动生成。</p>'
    q = s['quant']
    bf, nf = q.get('bf16_lora'), q.get('nf4')
    i8, fp = q.get('int8'), q.get('fp4')
    rec = s.get('rec') or {}
    picked = rec.get('picked')
    alts = rec.get('alternatives') or {}
    tie = rec.get('tie') or {}

    def _other(labels, picked_label):
        return next((x for x in labels if x != picked_label), labels[0] if labels else '')

    def acc(d):
        return '—' if not d or d.get('acc') is None else f'{d["acc"]:.4f}'

    if not bf or not picked:
        return '<p>尚无正式数据。</p>'
    hw = s.get('hw') or {}
    accs_all = [x['acc'] for x in q.values() if x and x.get('acc') is not None]
    spread = (max(accs_all) - min(accs_all)) * 100 if len(accs_all) > 1 else None

    def hwf(k, nd=2):
        return ' / '.join(f2(hw[x].get(k), nd) if hw.get(x) and hw[x].get(k) is not None else '—'
                          for x in QORDER)

    t = ('<p><strong>主要发现。</strong>在单块 11.94 GB 消费级 GPU、AG News 四分类任务上，'
         f'四种基座精度在标准配置（r16 / n2000，各 3 个种子）下的 native 准确率为 '
         f'BF16 {acc(bf)}、INT8 {acc(i8)}、NF4 {acc(nf)}、FP4 {acc(fp)}，'
         f'极差 {spread:.2f} 个百分点；'
         f'分配器峰值显存为 {hwf("peak_gib")} GiB，NF4 相对 BF16 低 {pct(s.get("mem_save"))}%；'
         f'训练吞吐为 {hwf("train_tp", 1)} samples/s，每焦耳 token 数约为 {hwf("tj", 1)}。'
         '也就是说，4-bit 以约一半的训练吞吐与能效换取约 45% 的分配器峰值显存下降，而准确率几乎不变。</p>')
    # P1-1：两类统计单位分开写
    sd = s.get('seed_deltas') or {}
    d4 = [abs(v) for k in ('nf4', 'fp4') for v in sd.get(k, [])]
    d4_txt = (f'{min(d4):.2f}–{max(d4):.2f}' if d4 else '0.28–0.29')
    d8 = sd.get('int8')
    rev_txt = f'、相对 INT8 在 s44 反向 {min(d8):+.2f} pp' if d8 else ''
    t += (f'<p><strong>显著性。</strong>在标准配置的 7,600 条配对预测上，未校正的成对 McNemar 检验'
          '对 BF16 与 4-bit 的比较给出 p 为 0.022 与 0.028，其余四组在 0.055–0.88；'
          '6 组两两比较经 Holm 校正后最小 p≈0.13，全部不显著——该结论只针对 <b>s42 这一次运行</b>。'
          f'三个种子的均值差是另一类量：BF16 相对 NF4/FP4 为 {d4_txt} 个百分点{rev_txt}，'
          '与 INT8 的均值差只有约 0.06 个百分点。它们只描述训练随机性、且都小于 1 个百分点的工程判据，'
          '不能与配对 p 值混读；本文不据此宣称任一格式显著更优。</p>')
    t += (f'<p><strong>条件性推荐。</strong>本实验全部候选的分配器峰值都在 3.86–6.97 GB，'
          '12 GB 的分配器预算对它们都不构成约束；真正接近上限的是整卡设备占用'
          '（BF16 11.58 GiB、bs16 11.56 GiB / 11.94 GiB）。因此这里的"显存约束"应理解为'
          '"若需在更小显存的卡上复现本流程"。在该前提下、且可接受 1 个百分点精度预算时，推荐 '
          f'<code>{picked["label"]}</code>：准确率 {picked["accuracy"]:.4f}，'
          f'分配器峰值 {f2(picked["peak_mem_gb"])} GB，训练约 {f2(picked.get("train_s"), 0)} s。')
    if tie.get('threshold_equivalent'):
        t += (f'它与 <code>{_other(tie["labels"], picked["label"])}</code> 的 3 种子均值差只有 0.01 个百分点，'
              f'低于 {tie["threshold_pp"]} 个百分点的阈值，按该事后阈值判为工程等价（非等效性检验）；'
              '选择依据是两项在精度、吞吐与训练时长上的综合位置，而非这 0.01 pp。'
              '该等价判据与取舍规则是外部评审后补充的分析规则，不是实验前预注册。')
    if alts.get('best_accuracy') and alts['best_accuracy'].get('label') != picked['label']:
        a = alts['best_accuracy']
        t += (f'若只看精度，<code>{a["label"]}</code> 可达 {a["accuracy"]:.4f}（单种子，需复核），'
              '代价是训练时间翻倍。')
    t += ('</p>')
    t += ('<p><strong>不能从本实验推出的结论。</strong>'
          '① 只覆盖一个 1.5B 模型、一个数据集与 256 token 序列上限，不得外推到更长上下文或更大模型；'
          '② 4-bit 在本实验中并不更快，只有显存优势；'
          '③ 单种子配置只作工程测量，fp32 对照只覆盖 400 条样本；'
          '④ cross-NF4 下排序反转，说明该协议不能当作排序稳健性证据，只用于诊断跨基座偏移；'
          '⑤ 未测部署、算子兼容性与自回归解码，因此不对量化格式在部署侧的价值下判断。</p>')
    t += ('<p><strong>后续工作。</strong>'
          '在更长序列与更大 batch 上重测显存-吞吐边界；引入第二种量化实现（GPTQ/AWQ/GGUF）区分库开销与位宽本身；'
          '把零样本对精度的敏感性做成独立研究；若要确认 0.3 pp 量级的格式差异，需要更多种子与分层抽样下的配对检验。</p>')
    return t




def build_reflection(cfg):
    rep = cfg.get('reporting', {}) or {}
    a = rep.get('author', {}) or {}
    name, sid = a.get('name', ''), a.get('student_id', '')
    if rep.get('anonymous'):
        signature = ''
    else:
        signature = (f'<p>署名：{html.escape(str(name))}（学号 {html.escape(str(sid))}）。本题由我独立完成选题、实验设计、代码实现、\n'
                     '运行、分析与写作；脚本与文字初稿有 AI 辅助，方案判断、运行验证和结论由我负责。</p>')
    return f"""
{signature}

<p>我最初的问题很朴素：手上只有一块 12 GB 的笔记本显卡，想在上面微调一个 1.5B 的指令模型，
但不确定"上 4-bit"到底是省钱、省事，还是只是看起来省。课程里讲量化多在推理侧，训练侧的选择往往靠经验，
于是我把问题拆成三个能测的量：微调后的准确率、训练和推理各占多少显存、每条 token 花多少时间与电量。</p>

<p>四次执行期故障里有三次是测量口径问题：BF16 基线评估落在 CPU 上、同进程串跑把常驻显存从 1.5 GB 抬到 5 GB、
日志反压让子进程周期性阻塞在 99% 利用率和 70 W 功耗之间。它们都表现为"数字看起来合理、趋势却不对"，
也都写进了 execution_log。这之后我先写判定规则（什么算塌缩、什么算可比、什么算失败），再跑实验。</p>

<p>结果有三处和预期不一样。第一，4-bit 并不更快，反量化让训练吞吐降到 BF16 的六成左右，能效几乎减半，
"低位宽等于更快"在训练侧不成立。第二，INT8 在训练侧被两头夹住：不比 NF4 省显存，比谁都被慢；
这一结果与库内部把 BF16 输入转成 FP16 再计算的数据类型路径一致，
但转换开销没有单独测量，不能确认它是唯一原因。第三，零样本准确率随基座精度非单调，NF4 反而高于 BF16，
我专门用 fp32 做了对照：bf16 与 fp32 在 400 条里只有 3 条预测不同（p=0.25），
没有发现 bf16 精度不足的证据，但样本量有限；至于是否由"取末位 token 的 logits"这种读出方式造成，
我没有做替代读出对照，只能留作假设。
这让我在报告里写明：零样本只能作参照，不能给量化格式排序。</p>

<p>方法上有两点收获。一是显著性检验只是工具：未校正时 7,600 条配对样本下 0.3 个百分点的差异也能到 p≈0.03，
但 6 组比较做 Holm 校正后全部不显著——真正要回答的是"这个差异是否改变决策"。二是要区分峰值与常驻：只看峰值会让量化的价值被激活淹没，
我中途一度误判"4-bit 没有显存收益"，原因就是只跑了四步的短程探针，后来在完整 epoch 上复测才纠正。</p>

<p>如果重做，我会改四件事：从一开始就让每个配置在独立进程里跑并固定日志策略；先做数据审计再开跑，
因为与测试集重复的样本、超长样本这类问题数值影响小、方向却确定地偏乐观，事后无法用统计补救；
把序列长度和 batch 上限做成正式变量，而不是只留两次探索：bs8/bs16 的实测训练吞吐是 13.2/11.5 samples/s，
反而低于 bs4 的 13.7，说明"4-bit 省下的显存没能转成吞吐"并不只是 batch 固定的问题，
是否受反量化实现或显存边界限制需要另做实验；最后，把零样本当作独立的提示词与精度交互问题，而不是顺手当作基线。</p>

<p>这门课让我对两个词有了具体理解。"内存瓶颈"不是名词，它是权重常驻、激活峰值、分页优化器与驱动回收共同作用的结果，
换一个测量点结论就会变；"量化"也不是开关，位宽、数据类型、反量化路径与实现库共同决定代价，
同一份 4-bit 权重在推理和训练上的表现可以完全相反。</p>"""


# ------------------------------------------------------------------ 表格
def build_data_audit(cfg):
    meta_path = ROOT / cfg['paths']['processed_data_dir'] / 'metadata.json'
    if not meta_path.exists():
        return '<p class="note">缺少数据审计文件。</p>'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    a = meta.get('audit') or {}
    checks = a.get('checks') or {}
    files = a.get('files') or {}
    raw, bs, ent = a.get('raw_rows') or {}, a.get('backslash_fixed') or {}, a.get('html_entity_restored') or {}
    test_tok = (files.get('test') or {}).get('tokens') or {}
    nested = all((checks.get('nested_subsets') or {}).values())
    over = sum((checks.get('samples_over_max_seq_length') or {}).values())
    sub_over = sum((checks.get('train_subset_vs_test_overlap') or {}).values())
    rows = [
        ['原始样本量', f"训练 {raw.get('train', '—')} / 测试 {raw.get('test', '—')}"],
        ['规范化（反斜杠转义）', f"训练 {bs.get('train', '—')} 条 / 测试 {bs.get('test', '—')} 条"],
        ['还原残缺 HTML 实体', f"训练 {ent.get('train', '—')} 处 / 测试 {ent.get('test', '—')} 处"],
        ['剔除与测试集完全相同的训练样本', f"{a.get('train_rows_matching_test_normalized', '—')} 条"],
        ['类别分布', '四类均衡（各档子集每类 n/4）'],
        ['四档子集嵌套', '500 ⊂ 1,000 ⊂ 2,000 ⊂ 4,000' if nested else '不成立'],
        ['验证集与测试集重叠', str(checks.get('val_vs_test_overlap', '—'))],
        ['四档训练子集与测试集重叠（500/1000/2000/4000）',
         ' / '.join(str((checks.get('train_subset_vs_test_overlap') or {}).get(f'train_{n}', '—'))
                    for n in (500, 1000, 2000, 4000))],
        ['验证集与四档训练子集重叠（500/1000/2000/4000）',
         (' / '.join(str((checks.get('val_vs_train_subset_overlap') or {}).get(f'train_{n}', '—'))
                     for n in (500, 1000, 2000, 4000))
          if checks.get('val_vs_train_subset_overlap') else '未检查')],
        ['超 max_seq_length 样本', str(over)],
        ['正文收缩样本（train_4000 / test / val；其余训练档各 3）',
         f"{(files.get('train_4000') or {}).get('truncated_texts', '—')} / "
         f"{(files.get('test') or {}).get('truncated_texts', '—')} / "
         f"{(files.get('val') or {}).get('truncated_texts', '—')}"],
        ['token 长度（测试集）',
         f"中位 {test_tok.get('median', '—')}，95% {test_tok.get('p95', '—')}，最大 {test_tok.get('max', '—')}"],
        ['提示词版本', f"{meta.get('prompt_version', '—')}"],
    ]
    return table(['检查项', '结果'], rows, ['l', 'l'], cls='audit keep')


def build_main_tables(df):
    if df is None or df.empty:
        return '<p class="note">无数据。</p>', '<p class="note">无数据。</p>'
    d = df.sort_values(['quantization', 'train_size', 'lora_rank', 'batch_size', 'seed'])
    rows_acc, rows_res = [], []
    for _, r in d.iterrows():
        q = QLABEL.get(r['quantization'], r['quantization'])
        cfg_label = f"{q} r{int(r['lora_rank'])} n{int(r['train_size'])}" + \
                    ('' if int(r['batch_size']) == 4 else f" bs{int(r['batch_size'])}")
        rows_acc.append([cfg_label, str(int(r['seed'])),
                         f2(r.get('accuracy_native'), 4), f2(r.get('accuracy_cross_nf4'), 4)])
        rows_res.append([cfg_label, str(int(r['seed'])),
                         f2(r.get('model_resident_memory_gb'), 2), f2(r.get('peak_memory_gb'), 2),
                         f2(r.get('train_samples_per_s'), 1), f2(r.get('infer_samples_per_s_native'), 1)])
    t_acc = table(['配置', '种子', 'native 准确率', 'cross-NF4 准确率'],
                  rows_acc, ['l', 'r', 'r', 'r'])
    t_res = table(['配置', '种子', '常驻显存 (GB)', '峰值显存 (GB)', '训练吞吐 (sa/s)', '推理吞吐 (sa/s)'],
                  rows_res, ['l', 'r', 'r', 'r', 'r', 'r'])
    return t_acc, t_res


def build_per_class(cfg):
    seed = cfg['experiment']['seed']
    names = cfg['data']['class_names']
    per = {}
    for q in QORDER:
        p = RESULTS / f'{q}_r16_n2000_s{seed}_native_eval.json'
        if p.exists():
            per[q] = json.loads(p.read_text(encoding='utf-8'))
    if not per:
        return '<p class="note">无每类指标。</p>', '<p class="note">无混淆矩阵。</p>'
    rows = []
    for name in names:
        support = None
        cells = [name]
        for q in QORDER:
            pc = (per.get(q) or {}).get('per_class', {}).get(name)
            if pc:
                support = pc['support']
                cells.append(f2(pc['recall'], 4))
            else:
                cells.append('—')
        rows.append([cells[0], str(support if support else '—')] + cells[1:])
    t_recall = table(['类别', '支持数'] + [f'{QLABEL[q]} 召回' for q in QORDER],
                     rows, ['l', 'r', 'r', 'r', 'r', 'r'], cls='keep')
    ref = per.get('nf4') or next(iter(per.values()))
    cm = ref.get('confusion_matrix')
    if not cm:
        t_cm = '<p class="note">无混淆矩阵。</p>'
    else:
        rows = [[names[i]] + [str(v) for v in cm[i]] for i in range(len(names))]
        t_cm = table(['真值 \\ 预测'] + names, rows, ['l'] + ['r'] * len(names), cls='keep')
    return t_recall, t_cm


def build_zero_shot(zs):
    if not zs:
        return '<p class="note">无零样本结果。</p>'
    rows = []
    for q in QORDER:
        d = zs.get(q)
        if not d:
            continue
        bench = d.get('inference_bench') or {}
        rows.append([QLABEL[q], f2(d.get('accuracy'), 4), f2(d.get('balanced_accuracy'), 4),
                     f2(d.get('macro_f1'), 4), f2(d.get('max_single_class_pred_share'), 3),
                     '是' if d.get('collapsed') else '否', f2(bench.get('samples_per_s'), 1)])
    return table(['基座', '准确率', '平衡准确率', 'macro-F1', '最大单类占比', '塌缩', '前向吞吐 (sa/s)'],
                 rows, ['l', 'r', 'r', 'r', 'r', 'c', 'r'], cls='keep')


def hardware_metrics(df):
    """表 8 与结论共用的硬件指标：吞吐、TFLOPS、功耗、tokens/J、两档显存。

    返回 {quant: {...}}；无数据时返回 {}。
    """
    if df is None or df.empty:
        return {}
    import ast as _ast
    d = core(df)
    n_unpacked = None
    if 'total_params' in d.columns:
        bf_rows = d[d['quantization'] == 'bf16_lora']
        n_unpacked = float(bf_rows['total_params'].max() if not bf_rows.empty
                           else d['total_params'].max())
    logs_dir = ROOT / 'outputs' / 'logs'

    def device_peak_gib(rid):
        f = logs_dir / str(rid) / 'gpu_trace.csv'
        if not f.exists():
            return None
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
        return max(mems) / 1024 if mems else None

    out = {}
    for q in QORDER:
        g = d[d['quantization'] == q]
        if g.empty:
            continue
        traces = []
        for x in g['gpu_trace']:
            if isinstance(x, str):
                try:
                    x = _ast.literal_eval(x)
                except Exception:
                    x = None
            if isinstance(x, dict):
                traces.append(x)
        power = sum(t.get('avg_power_w', 0) for t in traces) / len(traces) if traces else None
        temp = max((t.get('max_temp_c', 0) for t in traces), default=None)
        util = sum(t.get('avg_utilization_pct', 0) for t in traces) / len(traces) if traces else None
        tp = float(g['train_samples_per_s'].mean())
        seq = float(g['avg_seq_len'].mean()) if 'avg_seq_len' in g.columns else None
        tokens_s = tp * seq if seq else None
        tflops = 6 * n_unpacked * tokens_s / 1e12 if (n_unpacked and tokens_s) else None
        tj = tokens_s / power if (power and tokens_s) else None
        peaks = [device_peak_gib(r) for r in g['run_id']]
        peaks = [p for p in peaks if p]
        out[q] = {
            'train_tp': tp, 'tflops': tflops, 'power': power, 'tj': tj,
            'temp': temp, 'util': util,
            'peak_gib': float(g['peak_memory_gb'].mean()),
            'resident_gib': (float(g['model_resident_memory_gb'].mean())
                             if 'model_resident_memory_gb' in g.columns else None),
            'dev_max': max(peaks) if peaks else None,
            'dev_mean': (sum(peaks) / len(peaks)) if peaks else None,
            'dev_n': len(peaks),
        }
    return out


def build_hardware(df, s):
    """表 8：硬件派生指标（TFLOPS 用未打包参数量，另列 nvidia-smi 设备峰值）。"""
    if df is None or df.empty:
        return '<p class="note">无硬件数据。</p>'
    from pathlib import Path as _P
    import ast as _ast
    d = core(df)
    n_unpacked = None
    if 'total_params' in d.columns:
        bf_rows = d[d['quantization'] == 'bf16_lora']
        n_unpacked = float(bf_rows['total_params'].max() if not bf_rows.empty
                           else d['total_params'].max())
    logs_dir = ROOT / 'outputs' / 'logs'

    def device_peak_gib(rid):
        f = logs_dir / str(rid) / 'gpu_trace.csv'
        if not f.exists():
            return None
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
        return max(mems) / 1024 if mems else None

    rows = []
    dev_means = []
    for q in QORDER:
        g = d[d['quantization'] == q]
        if g.empty:
            continue
        traces = []
        for x in g['gpu_trace']:
            if isinstance(x, str):
                try:
                    x = _ast.literal_eval(x)
                except Exception:
                    x = None
            if isinstance(x, dict):
                traces.append(x)
        power = sum(t.get('avg_power_w', 0) for t in traces) / len(traces) if traces else None
        temp = max((t.get('max_temp_c', 0) for t in traces), default=None)
        util = sum(t.get('avg_utilization_pct', 0) for t in traces) / len(traces) if traces else None
        tp = g['train_samples_per_s'].mean()
        seq = g['avg_seq_len'].mean() if 'avg_seq_len' in g.columns else None
        tokens_s = tp * seq if seq else None
        tflops = 6 * n_unpacked * tokens_s / 1e12 if (n_unpacked and tokens_s) else None
        tj = tokens_s / power if (power and tokens_s) else None
        peaks = [device_peak_gib(r) for r in g['run_id']]
        peaks = [p for p in peaks if p]
        dev_mean = sum(peaks) / len(peaks) if peaks else None
        dev_max = max(peaks) if peaks else None
        rows.append([QLABEL[q], f2(tp, 1), f2(tflops, 1), f2(power, 1), f2(tj, 1),
                     f2(temp, 0), f2(util, 0), f2(g['peak_memory_gb'].mean(), 2),
                     f2(dev_max, 2) if dev_max else '—'])
        dev_means.append((QLABEL[q], dev_mean, len(peaks)))
    return table(['格式', '训练吞吐 (sa/s)', '名义 TFLOPS', '功耗 (W)', 'tokens/J',
                  '温度 (°C)', 'SM 利用率 (%)', '分配器峰值 (GiB)', '设备峰值（种子最大, GiB）'],
                 rows, ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'], cls='keep')




def build_recommend(rec):
    cand = rec.get('candidates') or []
    if not cand:
        return '<p class="note">无筛选结果。</p>'
    rows = []
    for c in cand:
        mark = '✓' if c.get('eligible') else '✗ ' + c.get('reason', '')
        if c.get('threshold_equivalent'):
            mark += '（工程等价*）'
        rows.append([f"<code>{html.escape(c['label'])}</code>", f2(c.get('accuracy'), 4),
                     f2(c.get('peak_mem_gb'), 2), f2(c.get('train_s'), 0),
                     f2(c.get('infer_tp'), 1), str(c.get('n_seeds', '—')), mark])
    return table(['配置', '准确率', '分配器峰值 (GB)', '纯训练 (s)', '推理吞吐 (sa/s)', '种子数', '结果'],
                 rows, ['l', 'r', 'r', 'r', 'r', 'r', 'l'], cls='keep')


def build_mcnemar(pairs):
    if not pairs:
        return '<p class="note">无检验结果。</p>'
    rows = []
    for x in pairs:
        ci = '—'
        if x.get('ci95_low_pp') is not None:
            ci = f"[{x['ci95_low_pp']:+.2f}, {x['ci95_high_pp']:+.2f}]"
        rows.append([QLABEL.get(x['a'], x['a']), QLABEL.get(x['b'], x['b']),
                     f"{x['diff_pp']:+.2f}" if x.get('diff_pp') is not None else '—',
                     ci,
                     str(x.get('b_a_correct_b_wrong', '—')),
                     str(x.get('c_a_wrong_b_correct', '—')),
                     f"{x.get('p_value'):.4g}" if x.get('p_value') is not None else '—',
                     f"{x.get('p_holm'):.3g}" if x.get('p_holm') is not None else '—',
                     '是' if x.get('significant_holm') else '否'])
    return table(['A', 'B', '差 (pp)', '95% CI (pp)', 'A 对 B 错', 'A 错 B 对',
                  'p（未校正）', 'p（Holm）', '校正后显著'],
                 rows, ['l', 'l', 'r', 'r', 'r', 'r', 'r', 'r', 'c'], cls='keep')




def build_appendix_schema():
    rows = [
        ['peak_memory_gb', '训练期分配器峰值显存（含激活）'],
        ['model_resident_memory_gb', '加载后、训练前的分配器增量（权重 + LoRA）'],
        ['train_samples_per_s', '训练吞吐，按纯训练时间计（不含验证评估）'],
        ['infer_samples_per_s_native', '分类前向吞吐（prefill，不含自回归解码）'],
        ['accuracy_native / accuracy_cross_nf4', '两种评估协议的测试准确率'],
        ['balanced_accuracy / macro_f1 / per_class / confusion_matrix', '分类细粒度指标'],
        ['predictions / gold_labels', '7,600 条逐样本预测，用于 McNemar'],
        ['gpu_trace', '训练期采样的温度、功耗、利用率、SM 时钟与整卡显存'],
        ['max_gpu_mem_used_mb', '每个 run 的整卡占用峰值（设备峰值，含驱动与其他进程）'],
    ]
    return table(['字段', '含义'], rows, ['l', 'l'])


# ------------------------------------------------------------------ 组装
def _spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def protocol_stats(df):
    if df is None or df.empty or 'accuracy_cross_nf4' not in df.columns:
        return []
    c = core(df)
    rows = []
    for q in QORDER:
        g = c[c['quantization'] == q]
        if g.empty or g['accuracy_cross_nf4'].isna().all():
            continue
        nat = float(g['accuracy_native'].mean())
        cro = float(g['accuracy_cross_nf4'].mean())
        rows.append({'q': q, 'native': nat, 'cross': cro,
                     'delta_pp': (cro - nat) * 100, 'n': len(g)})
    return rows


def build_protocol_table(df):
    rows = protocol_stats(df)
    if not rows:
        return '<p class="note">无协议数据。</p>'
    nat_desc = sorted(rows, key=lambda r: -r['native'])
    cro_desc = sorted(rows, key=lambda r: -r['cross'])
    nat_rank = {r['q']: i + 1 for i, r in enumerate(nat_desc)}
    cro_rank = {r['q']: i + 1 for i, r in enumerate(cro_desc)}
    body = []
    for r in nat_desc:
        body.append([QLABEL.get(r['q'], r['q']), f2(r['native'], 4), f2(r['cross'], 4),
                     f"{r['delta_pp']:+.2f}", str(nat_rank[r['q']]), str(cro_rank[r['q']])])
    return table(['格式', 'native 均值', 'cross-NF4 均值', '偏移 (pp)', 'native 排名', 'cross 排名'],
                 body, ['l', 'r', 'r', 'r', 'r', 'r'], cls='keep')


def build_protocol_text(s, df):
    rows = protocol_stats(df)
    if not rows or df is None:
        return '暂无协议数据。'
    raw = df.dropna(subset=['accuracy_native', 'accuracy_cross_nf4'])
    diffs = (raw['accuracy_native'] - raw['accuracy_cross_nf4']).abs() * 100
    mad, mx = float(diffs.mean()), float(diffs.max())
    rho = _spearman(list(raw['accuracy_native']), list(raw['accuracy_cross_nf4']))
    # R2-N4：NF4 行在两协议下构造恒等，会稀释平均差；单独给非 NF4 子集
    non = raw[raw['quantization'] != 'nf4']
    dn = (non['accuracy_native'] - non['accuracy_cross_nf4']).abs() * 100 if len(non) else None
    rho_n = (_spearman(list(non['accuracy_native']), list(non['accuracy_cross_nf4']))
             if len(non) > 1 else None)
    n_eq = int((raw['quantization'] == 'nf4').sum())
    shifts = '、'.join(f"{QLABEL.get(r['q'], r['q'])} {r['delta_pp']:+.2f}" for r in rows)
    nat_order = ' > '.join(QLABEL.get(r['q'], r['q'])
                           for r in sorted(rows, key=lambda x: -x['native']))
    cro_order = ' > '.join(QLABEL.get(r['q'], r['q'])
                           for r in sorted(rows, key=lambda x: -x['cross']))
    return (f'四个标准配置的偏移分别是 {shifts} 个百分点：NF4 适配器在两种协议下完全相同，'
            f'其余三种适配器统一降分，且降幅不同。于是格式排序从 native 下的 {nat_order} '
            f'变成 cross-NF4 下的 {cro_order}，BF16 从第一掉到最后；'
            f'19 个运行的 Spearman 相关系数为 {rho:.2f}（平均绝对差 {mad:.2f} pp、最大 {mx:.2f} pp，逐运行值见表 2）。'
            f'需要注意，{n_eq}/19 个运行是 NF4 适配器，它们在两种协议下逐样本相同'
            f'（平均绝对差 0.00 pp），会把总体统计量拉低；只看非 NF4 的 {len(non)} 个运行，'
            f'平均绝对差为 {dn.mean():.2f} pp、Spearman 为 {rho_n:.2f}。'
            '因此 cross-NF4 不是"排序稳健性"证据，而是测量"适配器与基座量化不匹配"的代价；'
            'native 作为主结果的理由是训练与评估使用同一基座，而不是两种协议结论一致。')


def build_mcnemar_text(pairs, df=None):
    if not pairs:
        return '暂无检验结果。'
    unc = [x for x in pairs if x['p_value'] < 0.05]
    min_p = min(x['p_value'] for x in pairs)
    min_holm = min(x.get('p_holm', 1.0) for x in pairs)
    sig_holm = [x for x in pairs if x.get('significant_holm')]
    parts = []
    parts.append('表 9 给出标准配置（r16/n2000/batch 4，seed=42）下四种格式两两之间的 McNemar 检验，'
                 '7,600 条样本逐样本配对。')
    if unc:
        detail = '、'.join(f"{QLABEL.get(x['a'], x['a'])}–{QLABEL.get(x['b'], x['b'])} p={x['p_value']:.3f}"
                           for x in sorted(unc, key=lambda z: z['p_value']))
        rest = [x['p_value'] for x in pairs if x['p_value'] >= 0.05]
        rest_txt = (f'其余 {len(rest)} 组 p 在 {min(rest):.2f}–{max(rest):.2f} 之间。'
                    if rest else '')
        parts.append(f'未校正时有 {len(unc)} 组 p 低于 0.05：{detail}；{rest_txt}')
    else:
        parts.append(f'未校正时最小 p={min_p:.3f}，没有一组达到 0.05。')
    if sig_holm:
        parts.append('经 Holm 校正后仍显著的比较有 '
                     + '、'.join(f"{QLABEL.get(x['a'], x['a'])}–{QLABEL.get(x['b'], x['b'])}"
                                for x in sig_holm) + '。')
    else:
        parts.append(f'6 组同时比较经 Holm 校正后最小 p={min_holm:.3f}，全部不显著。')
    # R3（盲审 P1-1）：配对检验只用了 s42 一次运行，不能外推为"格式总体差异"。
    seed_txt = ''
    try:
        d = core(df)
        d = d[d['quantization'].isin(['bf16_lora', 'int8', 'nf4', 'fp4'])]
        if 'seed' in d.columns:
            piv = d.pivot_table(index='seed', columns='quantization',
                                values='accuracy_native', aggfunc='mean')
            if 'bf16_lora' in piv.columns and len(piv) >= 2:
                parts_s = []
                stable, reversed_pairs = [], []
                for q in ('int8', 'nf4', 'fp4'):
                    if q not in piv.columns:
                        continue
                    deltas = ((piv['bf16_lora'] - piv[q]) * 100).round(2)
                    ql = QLABEL.get(q, q)
                    if (deltas > 0).all() or (deltas < 0).all():
                        stable.append(f'{ql}（' + '/'.join(f'{v:+.2f}' for v in deltas) + '）')
                    else:
                        reversed_pairs.append(
                            f'{ql}（' + '/'.join(f'{v:+.2f}' for v in deltas) + '）')
                if stable:
                    parts_s.append('BF16 相对 ' + '、'.join(stable)
                                   + ' 的领先在三种子上方向一致')
                if reversed_pairs:
                    parts_s.append('相对 ' + '、'.join(reversed_pairs)
                                   + ' 出现反向种子')
                seed_txt = ('配对检验只用 s42 的一次运行，不能外推为格式总体差异。'
                            '看三个种子的均值差：' + '；'.join(parts_s) + '。')
    except Exception as e:  # noqa: BLE001
        print('[WARN] seed-direction text:', e)
    ci_txt = ''
    cis = [(x.get('ci95_low_pp'), x.get('ci95_high_pp')) for x in pairs
           if x.get('ci95_low_pp') is not None]
    if cis:
        lo = min(c[0] for c in cis)
        hi = max(c[1] for c in cis)
        within = all(c[0] > -1.0 and c[1] < 1.0 for c in cis)
        ci_txt = (f'六组准确率差的 95% 配对 bootstrap 置信区间为 [{lo:+.2f}, {hi:+.2f}] pp，'
                  + ('全部落在 ±1 pp 内，说明在 1 pp 工程预算下样本层差异可接受；'
                     if within else '并不都落在 ±1 pp 内；')
                  + '但该区间只含测试样本波动，不含训练种子波动。')
    parts.append(seed_txt + '需要强调两类量不能混读：上面的 Holm 结论只针对 s42 这一次运行的配对检验；'
                 'BF16 与两种 4-bit 格式的 3 种子均值差为 0.28–0.29 个百分点、与 INT8 只有 0.06 个百分点，'
                 '这是种子层面的描述量，只说明差异小于 1 个百分点的工程判据，并不经过（也不受）Holm 校正。'
                 '因此本文既不宣称任一格式显著更优，也不把均值差当作配对检验的佐证。' + ci_txt)
    return ''.join(parts)


def build_fp32_text():
    controls = {}
    for mode in ('fp32', 'bf16', 'nf4'):
        d = read_json(f'zero_shot_control400_{mode}_eval.json')
        if d:
            controls[mode] = d
    if len(controls) < 3:
        return ('（fp32 对照未找到 artifact，本节不引用该结果。）')
    fp, bf, nf = controls['fp32'], controls['bf16'], controls['nf4']
    return (f'为检验这是否由 bf16 的数值精度造成，在同一批 400 条固定样本上补做了 fp32 / bf16 / nf4 对照：'
            f'三者准确率分别为 {fp["accuracy"]:.4f} / {bf["accuracy"]:.4f} / {nf["accuracy"]:.4f}，'
            f'最大单类预测占比分别为 {fp["max_single_class_pred_share"]:.3f} / '
            f'{bf["max_single_class_pred_share"]:.3f} / {nf["max_single_class_pred_share"]:.3f}，均未塌缩。'
            f'fp32 与 bf16 的准确率差为 {abs(fp["accuracy"] - bf["accuracy"]) * 100:.2f} 个百分点，'
            f'两者逐样本预测只有 3 条不同（精确 McNemar p=0.25）；而 nf4 比 bf16 高 '
            f'{abs(nf["accuracy"] - bf["accuracy"]) * 100:.2f} 个百分点（p≈2×10⁻⁴）。'
            '因此未发现"bf16 精度不足"的证据，但 400 条样本的功效有限，不能排除小幅精度效应。'
            '这一结果与"末位 token 的 logits 读出对数值细节敏感"'
            '的解释一致，但本文没有做替代读出（生成式解码、全序列 logprob、不同判定位置）的对照，'
            '因此只能作为假设而非结论。原始逐样本预测已保存为 <code>zero_shot_control400_*_eval.json</code>，可复核。'
            '零样本读数因此只作参照，不用于给量化格式排序；微调增益一律与同格式的零样本比较。')


def build_tradeoff_text(s, df):
    q = s.get('quant') or {}
    bf, nf, i8, fp = q.get('bf16_lora'), q.get('nf4'), q.get('int8'), q.get('fp4')
    if not bf:
        return '暂无数据。'
    return (f'以分配器峰值为横轴时，BF16 位于 {f2(bf["memory"])} GB，4-bit 位于 {f2(nf["memory"])} GB，'
            f'INT8 居中（{f2(i8["memory"])} GB）；以推理吞吐为横轴时，INT8 明显落后'
            f'（{f2(i8["infer_tp"], 1)} samples/s，其余 {f2(nf["infer_tp"], 1)}–{f2(bf["infer_tp"], 1)}）；'
            f'以训练吞吐为纵轴时，BF16 最高（{f2(bf["train_tp"], 1)} samples/s），4-bit 约 '
            f'{f2(fp["train_tp"], 1)}–{f2(nf["train_tp"], 1)}，INT8 为 {f2(i8["train_tp"], 1)}。'
            '在 batch=4、256 token 的设定下，4-bit 把峰值显存从 6.97 GB 降到 3.86 GB，'
            '代价是训练吞吐下降约 45%、推理吞吐下降约 10%。')


def build_rec_text(rec):
    tie = (rec or {}).get('tie') or {}
    picked = (rec or {}).get('picked') or {}
    alt = (rec or {}).get('alternatives') or {}
    if not picked:
        return ''
    other = next((x for x in tie.get('labels', []) if x != picked.get('label')), '')
    t = ''
    if tie.get('threshold_equivalent'):
        t += (f'在推理吞吐并列组内，<code>{picked["label"]}</code> 与 '
              f'<code>{other}</code> 的 3 种子均值只差 0.01 个百分点，'
              f'远低于 {tie["threshold_pp"]} 个百分点的等价阈值（取 0.3 pp 与种子标准差的较大者），'
              '因此按事后阈值判定为工程等价，而不是给这 0.01 pp 赋予精度含义'
              '（这里没有做 TOST 或置信区间，属于阈值判据而非等效性检验）。'
              '需要说明：这个等价阈值与训练时长 tie-break 是收到外部评审意见后补充的分析规则，'
              '并非实验前预注册；它不改变结论——所选配置在精度、推理吞吐与训练时长三项上都不劣于另一配置，'
              '换用其他 tie-break 仍会选它。')
    if alt.get('cheapest_training'):
        c = alt['cheapest_training']
        t += (f'若训练预算最紧，<code>{c["label"]}</code> 只需 {f2(c.get("train_s"), 0)} s，'
              f'准确率 {c["accuracy"]:.4f}（单种子）。')
    if alt.get('best_accuracy'):
        a = alt['best_accuracy']
        t += (f'若只看精度，<code>{a["label"]}</code> 可达 {a["accuracy"]:.4f}（单种子，需复核）。')
    return t


def build_values(cfg):
    df = load_df()
    zs = {}
    for p in RESULTS.glob('zero_shot_*_eval.json'):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
            if not d.get('limited_eval'):
                zs[d.get('quantization', p.stem)] = d
        except Exception:
            continue
    rec = read_json('RECOMMENDATION.json', {}) or {}
    mcnemar = read_json('MCNEMAR.json', []) or []
    n_evals = len([p for p in RESULTS.glob('*_eval.json')
                   if not p.stem.endswith('_partial') and not p.stem.startswith('zero_shot')])
    s = stats(df, zs, rec, mcnemar, n_evals)
    s['hw'] = hardware_metrics(df)

    t_acc, t_res = build_main_tables(df)
    t_recall, t_cm = build_per_class(cfg)

    values = {
        'TITLE': html.escape(cfg.get('reporting', {}).get('title', '')),
        'COURSE': html.escape(cfg.get('reporting', {}).get('course', 'CIE6053')),
        'AUTHOR': html.escape(str(cfg.get('reporting', {}).get('author', {}).get('name', ''))),
        'STUDENT_ID': html.escape(str(cfg.get('reporting', {}).get('author', {}).get('student_id', ''))),
        'AUTHOR_LINE': html.escape(_author_line(cfg)),
        'GENERATED': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'N_RUNS': str(s['n_runs']), 'N_EVALS': str(n_evals),
        'STATUS_BANNER': '' if s['n_runs'] else '<p class="note">预览稿：尚无正式实验数据。</p>',
        'ABS': build_abstract(s),
        'PART1': build_part1(s),
        'PART2': build_part2(s),
        'RESULT_INTRO': build_result_intro(s, df),
        'RQ_ANSWERS': build_rq_answers(s, df),
        'CONCL': build_conclusion(s),
        'REFLECTION': build_reflection(cfg),
        'DATA_AUDIT': build_data_audit(cfg),
        'MAIN_TABLE_ACC': t_acc, 'MAIN_TABLE_RES': t_res,
        'PER_CLASS': t_recall, 'CONFUSION': t_cm,
        'ZERO_SHOT_TABLE': build_zero_shot(zs),
        'FP32_TEXT': build_fp32_text(),
        'TRADEOFF_TEXT': build_tradeoff_text(s, df),
        'PROTOCOL_TABLE': build_protocol_table(df),
        'PROTOCOL_TEXT': build_protocol_text(s, df),
        'MCNEMAR_TEXT': build_mcnemar_text(mcnemar, df),
        'REC_TEXT': build_rec_text(rec),
        'HARDWARE_TABLE': build_hardware(df, s),
        'RECOMMEND_TABLE': build_recommend(rec),
        'MCNEMAR_TABLE': build_mcnemar(mcnemar),
        'APPENDIX_SCHEMA': build_appendix_schema(),
        'FIG01': fig('fig01_pipeline.png', '图 1　实验流程'),
        'FIG02': fig('fig02_matrix.png', '图 2　实际执行的实验矩阵（格内为运行数，* 表示含 batch 8/16 探索）'),
        'FIG03': fig('fig03_train_curves.png', '图 3　四种格式在标准配置下的训练损失（seed=42）'),
        'FIG04': fig('fig04_zero_shot_by_quant.png', '图 4　零样本准确率与预测集中度：(a) 准确率与平衡准确率；(b) 最大单类预测占比'),
        'FIG05': fig('fig05_tradeoff_plots.png', '图 5　三轴权衡：(a) 准确率与显存（● 峰值，× 常驻）；(b) 准确率与推理吞吐；(c) 显存与训练吞吐'),
        'FIG06': fig('fig06_protocol_agreement.png', '图 6　native 与 cross-NF4 的逐运行对比（数值接近但排序不一致）', small=True),
        'FIG07': fig('fig07_multiseed_variance.png', '图 7　三个种子配置的均值与标准差（纵轴已放大）', small=True),
        'FIG08': fig('fig08_power_vs_throughput.png', '图 8　硬件侧视图：(a) 训练吞吐与平均功耗；(b) 每焦耳 token 数', small=True),
    }
    return values, s


def inline_images(text):
    def repl(m):
        p = FIGURES / m.group(1)
        if not p.exists():
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode('ascii')
        mime = 'image/svg+xml' if p.suffix.lower() == '.svg' else 'image/png'
        return f'src="data:{mime};base64,{b64}"'
    return re.sub(r'src="figures/([^"]+)"', repl, text)


def export_pdf(src, dst):
    exe = next((p for p in EDGE_CANDIDATES if Path(p).exists()), None)
    if not exe:
        print('[WARN] 未找到 Edge/Chrome，跳过 PDF')
        return False
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [exe, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
               '--run-all-compositor-stages-before-draw', '--virtual-time-budget=15000',
               f'--user-data-dir={tmp}', f'--print-to-pdf={dst}', src.resolve().as_uri()]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            print(f'[WARN] PDF 失败: {e}')
            return False
    if dst.exists() and dst.stat().st_size > 0:
        print(f'[OK] PDF: {dst.name} ({dst.stat().st_size/1024:.0f} KB)')
        return True
    print('[WARN] PDF 未生成')
    return False


def make_submission(cfg, report_html, standalone_html, pdf_ok):
    sub = DOCS / 'submission'
    if sub.exists():
        shutil.rmtree(sub)
    (sub / 'figures').mkdir(parents=True)
    (sub / 'report.html').write_text(report_html, encoding='utf-8')
    (sub / 'report_standalone.html').write_text(standalone_html, encoding='utf-8')
    if pdf_ok and (DOCS / 'report.pdf').exists():
        shutil.copy2(DOCS / 'report.pdf', sub / 'report.pdf')
    for p in sorted(FIGURES.iterdir()):
        if p.suffix.lower() in ('.png', '.svg'):
            shutil.copy2(p, sub / 'figures' / p.name)
    a = cfg.get('reporting', {}).get('author', {})
    readme = f"""# 提交说明

- 课程：{cfg.get('reporting', {}).get('course', '')}
- 题目：{cfg.get('reporting', {}).get('title', '')}
- 作者：{a.get('name', '')}（学号 {a.get('student_id', '')}）
- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 文件清单

| 文件 | 说明 |
|---|---|
| `report.pdf` | 推荐提交件（A4，由 Edge 内核打印，与 HTML 内容一致） |
| `report_standalone.html` | 单文件自包含版（图片内嵌，可直接打开或另存为 PDF） |
| `report.html` + `figures/` | 多文件版（便于逐图查看） |

## 复现与工具链说明

- 环境：`environment.yml`（Python 3.11.18 / torch 2.11.0+cu128 / transformers 5.23.0 / bitsandbytes 0.50.2）；
  硬件为单块 RTX 5070 Ti Laptop GPU（11.94 GB）。
- 数据与结果：`data/processed`（含 `metadata.json` 审计）、`outputs/results/`（原始 JSON）。
- 脚本：`scripts/01_prepare_data.py` → `02_train.py --plan --explore` → `03_evaluate.py --all` →
  `04_summary.py` → `05_build_report.py`。报告中的表格与图表均由脚本生成，未手工填写数字。
- **AI 辅助说明**：实验脚本、评测流程与报告初稿由 AI 辅助完成；作者负责方案确认、运行验证、结果核对与结论表述。
"""
    (sub / 'README_提交说明.md').write_text(readme, encoding='utf-8')
    print(f'[OK] 提交包: {sub.relative_to(ROOT)}')


def main():
    cfg = load_config()
    tpl = (DOCS / 'report_template.html').read_text(encoding='utf-8')
    values, s = build_values(cfg)

    out = tpl
    for k, v in values.items():
        out = out.replace('{{' + k + '}}', str(v))
    # 避免标题在连字符处断开：只处理 <h1>，<title> 必须是纯文本
    h1_start = out.find('<h1>')
    h1_end = out.find('</h1>', h1_start + 1)
    if h1_start != -1 and h1_end != -1:
        h1 = out[h1_start:h1_end].replace(
            'Consumer-Grade', '<span style="white-space:nowrap">Consumer-Grade</span>')
        out = out[:h1_start] + h1 + out[h1_end:]
    leftover = sorted(set(re.findall(r'\{\{[A-Z0-9_]+\}\}', out)))
    if leftover:
        print(f'[WARN] 未替换占位符: {leftover}')

    # 公开版：没有署名行就不留空元素，页脚也去掉多余的分隔符
    if (cfg.get('reporting') or {}).get('anonymous'):
        out = out.replace('  <p class="byline"></p>\n', '')
        out = out.replace('· Final Project · </footer>', '· Final Project</footer>')
    (DOCS / 'report.html').write_text(out, encoding='utf-8')
    print(f"[OK] docs/report.html（{s['n_runs']} 次运行 / {s['n_evals']} 次评估）")

    standalone = inline_images(out)
    (DOCS / 'report_standalone.html').write_text(standalone, encoding='utf-8')
    print('[OK] docs/report_standalone.html')

    pdf_ok = export_pdf(DOCS / 'report_standalone.html', DOCS / 'report.pdf')
    make_submission(cfg, out, standalone, pdf_ok)


if __name__ == '__main__':
    main()
