# 主结果表

| 模型 | 设置 | Color | Shape | Texture | Spatial | Non-spatial | Complex | Average |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Janus-Pro-1B | Baseline | 0.3414 | 0.2039 | 0.2774 | 0.0735 | 0.2621 | 0.2335 | 0.2320 |
| Janus-Pro-1B | CompGen-GRPO-Full | 0.7834 | 0.5090 | 0.6756 | 0.2976 | 0.3044 | 0.3776 | 0.4913 |
| Janus-Pro-7B | T2I-R1-Reported | 0.8130 | 0.5852 | 0.7243 | 0.3378 | 0.3090 | 0.3993 | 0.5114 |

## 派生指标

- 相比 Janus-Pro-1B baseline 的绝对提升：`0.4913 - 0.2320 = +0.2593`。
- 相比 Janus-Pro-1B baseline 的相对提升：`0.2593 / 0.2320 = 111.8%`。
- 相对 reported T2I-R1 Janus-Pro-7B 平均分的比例：`0.4913 / 0.5114 = 96.1%`。

## 数据来源

Baseline 和 finetuned 分数由本地 `eval_results/` 下的 JSON 文件重新计算得到。每个类别包含 3,000 条评测记录。T2I-R1 Janus-Pro-7B 数字是 README / T2I-R1 对比表中的 reported reference values，并非从本地 `eval_results/` 重新计算。

