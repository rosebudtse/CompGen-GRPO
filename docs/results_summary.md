# 结果总结

本文档用于标准化 Phase 1 的主结果表述，方便后续复用到 README 和论文草稿中。

## 主结果表

| 类别 | Janus-Pro-1B Baseline | CompGen-GRPO Janus-Pro-1B | T2I-R1 Janus-Pro-7B Reported |
|---|---:|---:|---:|
| Color | 0.3414 | 0.7834 | 0.8130 |
| Shape | 0.2039 | 0.5090 | 0.5852 |
| Texture | 0.2774 | 0.6756 | 0.7243 |
| Spatial | 0.0735 | 0.2976 | 0.3378 |
| Non-spatial | 0.2621 | 0.3044 | 0.3090 |
| Complex | 0.2335 | 0.3776 | 0.3993 |
| Average | 0.2320 | 0.4913 | 0.5114 |

## 推荐表述

我们的 Janus-Pro-1B 模型将 T2I-CompBench 平均分从 `0.2320` 提升到 `0.4913`，相当于 `+0.2593` 的绝对提升，以及相对 Janus-Pro-1B baseline 的 `+111.8%` 相对提升。在该 benchmark 表格口径下，它达到 reported T2I-R1 Janus-Pro-7B 平均分的 `96.1%`。

## 重要注意事项

- T2I-R1 Janus-Pro-7B 数值是 reported reference numbers，不是从本地 `eval_results/` 重新计算得到。
- `96.1%` 的 claim 应限定在 T2I-CompBench average score 上。
- 不要声称 1B 模型在一般意义上等价于 7B 模型。
- 在没有 ablation 证据前，不要把某个类别的提升完全归因于单一 reward。
- 当前结果是自动 benchmark 分数；若要做更强的人类感知质量 claim，需要补 human evaluation。

## 本地结果来源

用于计算 baseline 和 finetuned 平均分的本地结果文件如下：

| 类别 | 结果文件 |
|---|---|
| color | `eval_results/{baseline,finetuned}/color/annotation_blip/vqa_result.json` |
| shape | `eval_results/{baseline,finetuned}/shape/annotation_blip/vqa_result.json` |
| texture | `eval_results/{baseline,finetuned}/texture/annotation_blip/vqa_result.json` |
| spatial | `eval_results/{baseline,finetuned}/spatial/labels/annotation_obj_detection_2d/vqa_result.json` |
| non_spatial | `eval_results/{baseline,finetuned}/non_spatial/annotation_clip/vqa_result.json` |
| complex | `eval_results/{baseline,finetuned}/complex/annotation_3_in_1/vqa_result.json` |
