# CIE6053 Final Project — QLoRA Quantization Trade-offs on a Consumer GPU

在单块 **11.94 GiB** 消费级笔记本 GPU（RTX 5070 Ti Laptop）上，用同一套数据、提示词与训练流程，
系统比较 QLoRA 微调中 **BF16 / INT8 / NF4 / FP4** 四种基座精度对分类准确率、显存、训练/推理吞吐与能效的影响。

- **最终报告（仅 PDF）**：[docs/report.pdf](docs/report.pdf)（17 页，A4）
- 课程：CIE6053 AI Chips and Applications · 2026

## 主要结果

标准配置 `r16 / n2000 / batch4`，每种格式 3 个随机种子（42/43/44），native 协议：

| 格式 | 准确率 | 分配器峰值 | 模型常驻 | 训练吞吐 | 推理吞吐 | tokens/J |
|---|---|---|---|---|---|---|
| BF16 | **0.9150 ± 0.0014** | 6.97 GiB | 2.88 GiB | **25.0 sa/s** | 79.5 sa/s | **23.6** |
| INT8 | 0.9143 ± 0.0025 | 4.45 GiB | 2.10 GiB | 7.6 sa/s | 39.9 sa/s | 7.4 |
| NF4 | 0.9120 ± 0.0015 | **3.86 GiB** | **1.52 GiB** | 13.7 sa/s | 71.9 sa/s | 12.3 |
| FP4 | 0.9121 ± 0.0008 | 3.86 GiB | 1.52 GiB | 14.0 sa/s | 71.9 sa/s | 12.5 |

主要结论（详细论证与限制见报告 §5–§7）：

1. **准确率差异很小**：四格式 3 种子均值极差 0.29 pp；未校正 McNemar p 为 0.022/0.028（BF16 对两种 4-bit），
   但 6 组两两比较经 Holm 校正后最小 p≈0.134，**均不显著**——本实验不宣称任一格式显著更优。
2. **4-bit 省的是显存，不是时间**：分配器峰值较 BF16 低 44.6%、常驻低 47.4%，
   但训练吞吐与每焦耳 token 数约减半；INT8 在训练侧两项都不占优。
3. **双协议会反转排序**：把全部适配器放到统一 NF4 基座（cross-NF4）评估时，BF16 从第 1 掉到第 4，
   因此 cross-NF4 只能作"基座不匹配"的诊断，不能当作排序稳健性证据。
4. **推荐**：显存受限时 `fp4_r16_n2000` 与 `nf4_r16_n2000` 按 0.30 pp 事后阈值判为工程等价
   （非等效性检验、非实验前预注册），按训练时长取前者；单种子边界方案见报告表 10。

## 仓库结构

```
.
├── README.md
├── environment.yml              # 精确环境（conda export）
├── requirements.txt
├── configs/experiment.yaml      # 唯一配置源
├── scripts/
│   ├── 00_verify_env.py         # 环境自检
│   ├── 01_prepare_data.py       # 清洗/去重/长度约束/审计 + 提示词
│   ├── 02_train.py              # 训练（--plan --explore --smoke --resume）
│   ├── 03_evaluate.py           # 双协议评估 + 四格式零样本
│   ├── 04_summary.py            # 汇总、McNemar、每类指标、硬件派生、8 张图
│   ├── 05_build_report.py       # 由结果生成 HTML / 单文件 / PDF / 提交包
│   ├── 06_fp32_control.py       # 400 条 fp32/bf16/nf4 零样本对照（非单调性检验）
│   └── 07_repair_val_overlap.py # val 与训练子集重复文本的一次性修复
├── data/
│   ├── raw/ag_news/             # AG News 原始 parquet（官方划分）
│   └── processed/               # 4 档训练集 + val + test + metadata.json（含审计）
├── outputs/
│   ├── results/                 # 19 次训练 JSON、38 次评估 JSON、4 零样本、3 对照、
│   │                            #   汇总 CSV、McNemar、推荐、硬件派生、每类指标
│   ├── logs/                    # 每次运行的环境/配置快照、GPU 采样、控制台日志
│   ├── execution_log.md         # 执行流水（含 4 次故障与修复）
│   └── acceptance_checklist.md  # 交付验收自检
└── docs/
    ├── report.pdf               # 最终报告（唯一提交稿，仅 PDF）
    ├── figures/                 # 图 1–8（SVG 矢量 + PNG）
    └── report_template.html     # 报告模板（源文件，不是提交稿）
```

## 复现

环境：Python 3.11.18 / PyTorch 2.11.0+cu128 / Transformers 5.23.0 / PEFT 0.21.3 /
bitsandbytes 0.50.2；完整清单见 `environment.yml`。

```bash
# 1) 环境
conda env create -f environment.yml && conda activate lora-project
# 或 pip install -r requirements.txt

# 2) 基座模型（未随仓库发布，约 2.9 GB）
#    下载 Qwen/Qwen2.5-1.5B-Instruct 到 models/Qwen2.5-1.5B-Instruct/

# 3) 环境自检
python scripts/00_verify_env.py

# 4) 数据（processed 已随仓库提供；如需从 raw 重建）
python scripts/01_prepare_data.py

# 5) 训练（19 次，单卡约 2.5–4 h）与评估（38 次，约 1.5 h）
python scripts/02_train.py --plan --explore
python scripts/03_evaluate.py --zero-shot
python scripts/03_evaluate.py --all

# 6) 汇总 + 8 张图 + 判定
python scripts/04_summary.py

# 7) 可选：重新生成报告（需要 Edge/Chrome 打印 PDF）
python scripts/05_build_report.py
```

## 未包含的大文件

| 内容 | 大小 | 说明 |
|---|---|---|
| `models/Qwen2.5-1.5B-Instruct` | ~2.9 GB | 从 HuggingFace 官方仓库下载 |
| `outputs/checkpoints`（19 个 LoRA 适配器） | ~370 MB | 可用 `02_train.py` 重新生成 |
| `outputs/logs/eval_all.log` | ~48 MB | 评估过程的完整控制台输出，未纳入仓库 |

## 口径与已知限制

- **显存三层**：模型常驻（加载增量）/ 训练分配器峰值（`max_memory_allocated`）/ 设备占用
  （`nvidia-smi memory.used`，含驱动与其他进程）。三者不能换算或相减；本文的"12 GB 预算"
  指分配器峰值口径，且对全部候选都不构成约束。单位均为 GiB（1024³ 字节）。
- **TFLOPS 是名义值**：6ND 近似 + 梯度检查点重算，未计注意力二次项（约 8%）与 bnb 反量化，
  不能解读为硬件峰值利用率。
- **统计单位**：McNemar 只用 seed=42 的同一次运行做逐样本配对，6 组比较报告 Holm 校正；
  3 种子均值差只描述训练随机性，两者不能混读。
- **已知限制**：单一 1.5B 模型、单一数据集、256 token 上限；仅 bitsandbytes 一种量化实现；
  推理只测分类前向（prefill）；11 个配置中 7 个为单种子；未测部署与自回归解码。

## 数据、许可与致谢

- 数据：AG News（`fancyzhx/ag_news`，Apache-2.0 / CC0），官方训练/测试划分。
- 基座：Qwen2.5-1.5B-Instruct（Apache-2.0）。
- 方法：QLoRA（[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)）、LoRA（[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)）、
  LLM.int8()（[arXiv:2208.07339](https://arxiv.org/abs/2208.07339)）。
- 本仓库为课程作业，**未附开源许可证**；如需复用代码或结果，请先联系作者并注明出处。
- AI 辅助说明：实验脚本、数据/评测流程与报告初稿在 AI 辅助下完成；作者负责方案确认、运行验证、
  结果核对与结论表述，并对全部内容负责。
