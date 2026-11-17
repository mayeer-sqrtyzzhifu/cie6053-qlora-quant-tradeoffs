# 交付验收自检（CP6）

> 对照 `docs/AGENT_EXECUTION_PLAN.md` §8.2 逐条核验。执行时间：2026-11-17 21:00–21:40。
> 数据版本 `v4_chat_template`；训练 19 次、评估 38 次、zero-shot 4 次，失败 0。

## 1. 完整性

| 项 | 要求 | 实际 | 证据 |
|---|---|---|---|
| 训练 | 19 次均有结果 JSON | **19/19** | `outputs/results/{run_id}.json` ×19；`batch_summary.json` 记录 `runs=19, failures=0` |
| 评估 | 38 个正式评估 JSON | **38/38** | `outputs/results/{run_id}_{protocol}_eval.json`，全部 `model_device=cuda`、`limited_eval=false` |
| Zero-shot | 四种格式各 1 次 | **4/4** | `zero_shot_{bf16_lora,int8,nf4,fp4}_eval.json` |
| 失败/跳过披露 | 需披露 | 无失败 | 未生成 `FAILED_RUNS.md`，无需披露 |
| 图表 | fig01–fig08 | **8/8** | `docs/figures/`：SVG（矢量，报告用） + PNG（兼容用） |
| 报告 | HTML + 单文件 + PDF | 全部生成 | `docs/report.html`、`report_standalone.html`、`report.pdf`（1.5 MB） |
| 提交包 | 目录 + 说明 | 已生成 | `docs/submission/`（report.* + figures/ + `README_提交说明.md`） |

## 2. 可追溯（关键数字 → 来源）

| 报告中的数字 | 来源文件 / 字段 |
|---|---|
| 各配置准确率 0.9120 ± 0.0015、0.9150 ± 0.0014 等 | `summary_by_config.csv` → `accuracy_native_mean/_std/_n` |
| 每 run 的准确率、显存、吞吐 | `summary_raw.csv`（19 行）；原始 `*_eval.json` / `{run_id}.json` |
| McNemar 的 p 值（0.00012、0.054909 等） | `MCNEMAR.json` / `MCNEMAR.md` |
| 峰值/常驻显存 6.967 / 2.884 / 4.448 / 3.862 GB | `{run_id}.json` → `peak_memory_gb` / `model_resident_memory_gb` |
| tokens/J、估算 TFLOPS、平均功耗 | `HARDWARE_SUMMARY.md`（由 `gpu_trace.csv` 与结果 JSON 二次计算） |
| 零样本四格式 0.7357/0.7289/0.7975/0.7475 | `zero_shot_*_eval.json` → `accuracy` |
| 每类召回与混淆矩阵 | `PER_CLASS.md`（由 seed=42 native 评估聚合） |
| 数据清洗计数、泄漏剔除 28、超长 0 | `data/processed/metadata.json` → `audit` |
| 推荐配置 `fp4_r16_n2000` 与边界方案 | `RECOMMENDATION.json` → `picked` / `alternatives` |

- 每张图由 `04_summary.py` 的对应函数生成（`plot_fig01…plot_fig08`），可无 GPU 重绘。
- 报告所有表格/图由 `05_build_report.py` 从上述文件注入，模板中无手填数字。

## 3. 可复现

| 项 | 状态 |
|---|---|
| `environment.yml` | 已导出精确版本（Python 3.11.18 / torch 2.11.0+cu128 / transformers 5.23.0 / bnb 0.50.2） |
| 配置快照 | 每次运行 `outputs/logs/{run_id}/experiment_snapshot.yaml`（含 `_cli_overrides`） |
| 随机种子 | 42；关键配置 42/43/44；fp4 亦补齐 3 种子 |
| 数据 | 原始 parquet + `metadata.json`（seed、prompt 版本、审计）；`prompt_version=v4_chat_template` |
| 环境注意项 | 批处理逐 run 独立子进程、`disable_tqdm`、`wait_gpu_idle`；Windows 不支持 `expandable_segments`（已在 README/A7 注明） |
| 复现命令 | `python scripts/01_prepare_data.py` → `02_train.py --plan --explore` → `03_evaluate.py --all` → `04_summary.py` → `05_build_report.py` |

## 4. 诚实性

- 单种子配置（n500/n1000/n4000、r8/r32、bs8/bs16）在报告与 `RECOMMENDATIONS.md` 中均标注「单种子」，主推荐只在 ≥3 种子的候选中产生。
- 4-bit **不是**免费加速：报告如实写明训练吞吐与 tokens/J 低于 BF16（约 −45% / −48%），未把量化写成「又快又省」。
- 统计显著与实际重要性分开表述（§5.7 说明：0.2–0.3 pp 也可显著，但不改变工程决策）。
- 局限章节包含：单一数据集/模型、仅 bnb 一种实现、仅测 prefill、固定 batch 的矛盾、笔记本散热、未在测试集调参。
- 方案迭代（§3.6/§6.4）如实记录四类修正，未隐藏中途推翻的中间判断。

## 5. 格式与合规

| 项 | 状态 |
|---|---|
| 报告可直接打开 | `docs/report.html`，A4 版式；8 图、10 张编号表 + 表 A1/B1；0 残留占位符；表格不裁切、无横向滚动 |
| PDF | `docs/report.pdf` 17 页 A4 / 0.63 MB；图以 SVG 矢量嵌入（缩放不虚）；已逐页渲染核对版式 |
| 单文件版 | `report_standalone.html` 740 KB（图 base64 内嵌） |
| **评分点对照** | Part 1 三问 → §1.4；Part 2 三问 → §2.1；两节直接给出数据支撑 |
| Part 1 内容 | §1.4 覆盖「项目产出 / 动机与影响 / 新颖性与独特性」，含实测数字 |
| Part 2 内容 | §2.1 覆盖「起点与基线 / 增量工作 / 验证实验与结果」，含对照数字 |
| 结论口径 | §7 分「主要发现 / 条件性推荐 / 不能推出的结论 / 后续工作」四段，显式排除过度外推 |
| 课程主题 | 量化 / 内存瓶颈 / GPU-XPU / 神经网络（§1.4） |
| 个人反思 | **已撰写**（§9，第一人称，含 AI 辅助声明）；待作者按个人经历润色 |
| 评审入口 | `PROJECT_INDEX.md`（产物清单、关键数字、口径约定、15 分钟核查路径、评审清单） |
| 不在交付内 | PPT 与 5 分钟展示（执行规格 E4） |

## 5.1 外部评审整改（第一轮 2026-11-17 22:30，第二轮 22:55）

| 级别 | 条数 | 状态 | 证据 |
|---|---|---|---|
| P0 | 2 | 已修复 | `MCNEMAR.json` 标准配置（BF16–NF4 p=0.0282, Holm 0.141）；§5.4 表 7 协议排序反转 |
| P1 | 6 | 已修复 | `RECOMMENDATION.json` tie 字段；`HARDWARE_SUMMARY.md` 设备峰值列；`zero_shot_control400_*.json`；Holm 校正；TFLOPS 统一参数量；§7 删除部署外推 |
| P2 | 7 | 已修复 | §1.1/§5.3/§6.2/§9 改写；表 1 措辞；gpu_trace 表头；§6.4 时间线；索引路径 |

第二轮验证（整改回归）又发现 9 条：P1×4（事后规则写成"预声明"、设备峰值实为跨种子均值、§9 batch 因果断言、双协议统计被 NF4 恒等点稀释）+ P2×5（表 7 无正文引用、索引计数过期、"近重复"残留 2 处、bf16 显存列是空格字符串、摘要 44.6% 未绑定分配器口径）。已全部修完并复核：
- "预声明"全文 0 次，§6.1/§7 改为"外部评审后补充的分析规则"；
- 表 8 设备峰值改取种子最大（BF16 11.58/INT8 11.50/NF4 11.19/FP4 11.07），正文注明 bf16 仅 2 种子；
- §9 batch 断言改为实测（bs8/bs16 13.2/11.5 sa/s，低于 bs4 的 13.7）；
- 双协议补报非 NF4 子集（n=9，0.87 pp、Spearman 0.35）；
- 表 7 正文引用补齐；索引计数改为"10 张编号表 + 表 A1"；
- "近重复"0 次；bf16 显存列改为 NaN（113/113）；
- 摘要与 §1.4 的 44.6% 全部绑定"分配器峰值"。

第二轮独立盲审（84/100 B+，0 个 P0）另提 6 个 P1 + 12 个 P2，已修：fp32 对照改为"未发现证据但功效有限"、
§6.2 反量化因果改为"与解释一致但未单独测量"、零样本塌缩区间 0.35–0.43、种子标准差"同量级"、
删除重复表 7 行、LLM.int8() 引文 [4]→[2]、控制组占比 3 位小数、设备峰值差距 0.51 GiB、
摘要补 % 与 0.293 pp、§4.1 硬件参数改为有依据表述、表 8 TFLOPS 脚注、
`stat_equivalent`→`threshold_equivalent`、RQ2/§5.4 措辞、图 5/8 微调。
P2-12（页尾空白）接受为取舍：优先保证图表与题注不跨页。

逐条说明见 `docs/REVIEW_RESPONSE.md`。

## 5.2 人审 Major Revision 整改（2026-11-18 01:2x）

- **P0-1 val 独立性（真问题）**：val 与 train_4000 有 1 条相同文本；01 增加抽样后检查与替换并重跑，
  四档重叠归零（训练子集 MD5 未变，无需重训），表 1 增加三行审计。
- **P0-2 RQ2**：改为绝对增益，并注明单位计算成本不在设计内（秩差 6 s < 种子波动 ~34 s）。
- **P0-3 显存**：三层口径（常驻/分配器峰值/设备占用）+ 统一 GiB + 12 GB 预算口径说明。
- **P0-4 TFLOPS**：改名"名义 TFLOPS"，脚注给出 LoRA+检查点下的 6ND 分解与未计入项。
- **P1**：McNemar 补 95% bootstrap CI（表 9 增加差值与 CI 列）；§6.4 增加"精度+软件路径"限定；
  新增附录 B 表 B1 测量协议；提交包含 PDF/单文件/多文件三种形态。
- **P2**：补 [4][7][8] 引用；修 HTML `<title>` 内嵌 span；表 10 的 ✓ 语义注释；§5.4 拆段；反思 INT8 归因加限定。

## 6. 结论

全部验收项通过（含外部评审 15 条整改）。仍需人工完成：① 报告 §9 个人反思的个性化润色；
② 最终结论措辞确认（报告已按严谨口径自行定稿）；③ PPT 与 5 分钟展示（契约外）。
