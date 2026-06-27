# CompGen-GRPO 后续开发任务说明文档

## 0. 文档目的

本项目是 `CompGen-GRPO` 的后续增强任务说明，目标是让 Codex 在读取本文档和本地代码后，能够理解：

1. 当前项目的研究定位。
2. 哪些任务需要优先完成。
3. 哪些任务属于暂缓或后续扩展。
4. 如何组织代码、实验、配置、结果和文档。
5. 哪些 claim 可以写，哪些 claim 需要避免。
6. ablation、reward reliability、qualitative analysis 等实验应如何实现和记录。

本项目不是从零提出新的 T2I 强化学习框架，而是基于 T2I-R1 / Bi-CoT-GRPO 的扩展工作。当前重点是把项目整理成一个适合秋招简历、技术面试和 arXiv technical report 的高可信项目。

---

## 1. 当前项目定位

### 1.1 项目一句话定位

本项目基于 T2I-R1 / Bi-CoT-GRPO，将 GRPO 适配到 Janus-Pro-1B 这类紧凑型自回归 Text-to-Image 模型，并通过多维 reward suite 重设计，提升模型在组合语义生成任务中的 prompt following、属性绑定、空间关系、计数和整体语义对齐能力。

### 1.2 推荐论文 / README 表述

可以安全使用如下表述：

> We adapt the T2I-R1 Bi-CoT-GRPO framework to Janus-Pro-1B and study whether a compact autoregressive T2I model can acquire stronger compositional generation ability through multi-dimensional reward redesign.

中文表述：

> 本工作基于 T2I-R1 / Bi-CoT-GRPO，研究紧凑型自回归 T2I 模型能否通过多维奖励重设计获得更强的组合语义生成能力。

### 1.3 贡献边界

必须明确：

1. Bi-CoT-GRPO 框架来自 T2I-R1。
2. 本项目不是从零提出新的 T2I 强化学习算法。
3. 本项目的主要贡献是：

   * 将 T2I-R1 / Bi-CoT-GRPO 适配到 Janus-Pro-1B。
   * 针对组合语义对齐设计 multi-reward suite。
   * 改进 detection-based reward，加入软空间评分和软计数惩罚。
   * 使用 Qwen3-VL 类 VLM reward 进行属性绑定和整体语义对齐。
   * 在 T2I-CompBench 上验证 1B 模型的参数效率。
   * 提供单卡训练、reward model 量化共享、自动评估 pipeline 等工程实现。

### 1.4 禁止或谨慎使用的 claim

不要写：

1. “我们提出了 Bi-CoT-GRPO。”
2. “我们从零提出了新的 T2I RL 框架。”
3. “GDinoEnhanced 直接解决 color / shape / texture 属性绑定。”
4. “我们超过了 T2I-R1 7B。”
5. “1B 模型全面等价于 7B 模型。”
6. “所有提升都严格来自某一个 reward”，除非有完整 ablation 支撑。

可以写：

1. “We adapt T2I-R1 Bi-CoT-GRPO to Janus-Pro-1B.”
2. “We redesign the reward suite for compositional semantic alignment.”
3. “VLMAttr targets fine-grained attribute-object binding.”
4. “GDinoEnhanced provides object existence, soft spatial relation, and soft counting rewards.”
5. “On T2I-CompBench average score, our 1B model reaches 96.1% of the reported T2I-R1 Janus-Pro-7B performance.”
6. “The result suggests strong parameter efficiency under this benchmark setting.”

---

## 2. 当前核心结果

### 2.1 主结果

当前主结果如下：

| Category    | Baseline Janus-Pro-1B | Ours Janus-Pro-1B | T2I-R1 Janus-Pro-7B Reported |
| ----------- | --------------------: | ----------------: | ---------------------------: |
| color       |                0.3414 |            0.7834 |                       0.8130 |
| shape       |                0.2039 |            0.5090 |                       0.5852 |
| texture     |                0.2774 |            0.6756 |                       0.7243 |
| spatial     |                0.0735 |            0.2976 |                       0.3378 |
| non-spatial |                0.2621 |            0.3044 |                       0.3090 |
| complex     |                0.2335 |            0.3776 |                       0.3993 |
| average     |                0.2320 |            0.4913 |                       0.5114 |

### 2.2 结果表达口径

必须区分绝对提升和相对提升：

```text
Absolute gain = 0.4913 - 0.2320 = +0.2593
Relative improvement = 0.2593 / 0.2320 = +111.8%
Ratio to reported T2I-R1 7B average = 0.4913 / 0.5114 = 96.1%
```

推荐写法：

> Our Janus-Pro-1B model improves the T2I-CompBench average score from 0.2320 to 0.4913, corresponding to +0.2593 absolute gain and +111.8% relative improvement over the 1B baseline. It reaches 96.1% of the reported average score of T2I-R1 with Janus-Pro-7B.

---

## 3. 术语统一要求

请 Codex 检查并统一项目中所有 README、docs、scripts、paper draft 中的术语。

### 3.1 模型名称

统一写作：

```text
Janus-Pro-1B
T2I-R1
Bi-CoT-GRPO
T2I-CompBench
GroundingDINO
Qwen3-VL-2B
HPS v2
```

### 3.2 Reward 命名

建议统一为：

| Reward 名称                     | 作用                   | 备注                         |
| ----------------------------- | -------------------- | -------------------------- |
| HPSReward                     | 图像美学 / 人类偏好          | 基于 HPS v2                  |
| GDinoEnhancedReward           | 对象存在、软空间关系、软计数       | 不直接负责属性绑定                  |
| VLMAttrReward                 | 属性-对象绑定判别            | 负责 color / shape / texture |
| VLMAlignReward 或 VLMOrmReward | 整体 prompt-image 语义对齐 | 如果代码确实实现了 VLM 整体评分才使用      |
| FormatReward                  | 输出格式正确性              | 如果是 rule-based，不能称为 VLMOrm |

### 3.3 需要重点核对的问题

Codex 读取本地代码后，需要检查：

1. `VLMOrm` 是否真实使用 Qwen3-VL-2B 对图像和 prompt 做整体语义评分。
2. 如果 `VLMOrm` 实际只是 rule-based format correctness，则需要：

   * 将其重命名为 `FormatReward`。
   * 不要在 README 或论文中声称它是 VLM outcome reward。
3. 如果同时存在 `VLMOrm` 和 `FormatReward`，需要在文档中明确区分二者。
4. 检查 reward aggregation 代码中每个 reward 的权重、归一化、返回值范围是否在 README 中准确描述。

---

## 4. 优先级总览

本项目后续任务分为 3 个阶段：

| 阶段      | 目标                  | 状态    |
| ------- | ------------------- | ----- |
| Phase 1 | 秋招高收益版本             | 最高优先级 |
| Phase 2 | Workshop-quality 版本 | 中优先级  |
| Phase 3 | 更强论文版本              | 暂缓    |

执行原则：

1. 优先完成能直接增强简历、面试和 arXiv v1 可信度的任务。
2. 暂缓大规模新方法开发，避免战线过长。
3. 任何实验必须保证日志、配置、checkpoint、评测结果可追溯。
4. 不能把小规模实验包装成完整训练实验。
5. 不能把 post-hoc reward analysis 包装成 training ablation。

---

# Phase 1：秋招高收益版本

## 5. Phase 1 目标

Phase 1 是当前最优先完成的版本。目标是在较短时间内让项目具备以下能力：

1. README 和代码清晰。
2. 项目贡献边界准确。
3. 主结果可信。
4. 有关键 ablation。
5. 有 qualitative examples。
6. 有面试可讲的技术细节。
7. 可以整理成 arXiv technical report v1。

---

## 6. Task 1：统一 README、docs、paper draft、简历 claim

### 6.1 任务描述

请 Codex 检查以下文件：

```text
README.md
docs/
paper/
scripts/
configs/
results/
```

如果目录不存在，请根据项目结构调整检查路径。

需要完成：

1. 统一项目定位。
2. 统一 reward 命名。
3. 统一 benchmark 名称。
4. 统一主结果数字。
5. 明确 acknowledge T2I-R1。
6. 删除或修正夸大 claim。
7. 添加 “Contribution Boundary” 或 “Relation to T2I-R1” 小节。

### 6.2 README 建议结构

建议 README 包含：

```text
# CompGen-GRPO

## Overview
## Relation to T2I-R1
## Method
## Reward Suite
## Training Setup
## Evaluation
## Main Results
## Ablation Plan / Results
## Qualitative Examples
## Limitations
## Reproducibility
## Citation / Acknowledgement
```

### 6.3 必须新增的小节

建议加入：

```markdown
## Relation to T2I-R1

This project builds upon the Bi-CoT-GRPO framework introduced by T2I-R1. 
Our focus is not to propose a new RL algorithm from scratch, but to study how reward redesign and compact-model adaptation can improve compositional text-to-image generation with Janus-Pro-1B.
```

---

## 7. Task 2：梳理代码结构和入口

### 7.1 任务描述

Codex 需要读取本地代码，输出或补充一份代码结构说明文档。

建议新增：

```text
docs/code_structure.md
```

### 7.2 文档需要说明

至少包括：

1. 训练入口脚本。
2. reward 相关文件。
3. evaluation 相关文件。
4. dataset preprocessing 相关文件。
5. DeepSpeed 配置文件。
6. inference / generation 脚本。
7. qualitative example 生成脚本。
8. checkpoint 保存路径和命名规则。
9. TensorBoard 日志路径。
10. 每个重要配置字段含义。

### 7.3 示例结构

```markdown
# Code Structure

## Training
- `grpo.py`: main GRPO training entry.
- `scripts/train_xxx.sh`: training launcher.

## Rewards
- `rewards/hps_reward.py`
- `rewards/gdino_enhanced.py`
- `rewards/vlm_attr.py`
- `rewards/vlm_align.py`
- `rewards/format_reward.py`

## Evaluation
- `eval/t2i_compbench_eval.py`
- `eval/generate_samples.py`

## Configs
- `configs/grpo_full.yaml`
- `configs/deepspeed_zero2.json`
```

请以真实本地代码结构为准，不要凭空创建错误路径。

---

## 8. Task 3：整理 Method Figure 所需信息

### 8.1 任务描述

暂时不要求 Codex 画图，但需要整理 method figure 的文字内容，方便后续画论文图。

建议新增：

```text
docs/method_figure_spec.md
```

### 8.2 图中必须包含

```text
Prompt
  ↓
Bi-CoT reasoning / textual planning
  ↓
Autoregressive image token generation by Janus-Pro-1B language_model
  ↓
VQ image decoder / gen_encoder
  ↓
Generated image candidates
  ↓
Multi-reward scoring:
    - HPSReward
    - GDinoEnhancedReward
    - VLMAttrReward
    - VLMAlignReward / FormatReward
  ↓
Group-relative advantage estimation
  ↓
GRPO update on language_model
```

### 8.3 图中需要体现冻结模块

需要标注：

```text
Trainable:
- language_model

Frozen:
- vision_model
- aligner
- gen_encoder / VQ decoder
```

---

## 9. Task 4：主结果表和结果文件标准化

### 9.1 任务描述

Codex 需要整理已有结果文件，并生成统一的结果表。

建议新增或更新：

```text
results/main_results.csv
results/main_results.md
docs/results_summary.md
```

### 9.2 结果表字段

```csv
model,setting,color,shape,texture,spatial,non_spatial,complex,average
Janus-Pro-1B,Baseline,0.3414,0.2039,0.2774,0.0735,0.2621,0.2335,0.2320
Janus-Pro-1B,CompGen-GRPO-Full,0.7834,0.5090,0.6756,0.2976,0.3044,0.3776,0.4913
Janus-Pro-7B,T2I-R1-Reported,0.8130,0.5852,0.7243,0.3378,0.3090,0.3993,0.5114
```

### 9.3 自动计算

建议 Codex 新增脚本：

```text
scripts/compute_result_summary.py
```

功能：

1. 读取 `results/main_results.csv`。
2. 计算 absolute gain。
3. 计算 relative improvement。
4. 计算 ratio to T2I-R1 7B。
5. 输出 Markdown 表格。

---

## 10. Task 5：Qualitative Examples 生成与整理

### 10.1 任务描述

Qualitative examples 是 Phase 1 的高优先级任务。它直接服务于 README、论文和面试展示。

建议新增目录：

```text
assets/qualitative/
results/qualitative_examples.json
docs/qualitative_analysis.md
```

### 10.2 需要准备的样例类型

建议至少准备 12 组：

| 类型              | 数量 |
| --------------- | -: |
| color success   |  2 |
| shape success   |  2 |
| texture success |  2 |
| spatial success |  2 |
| complex success |  2 |
| failure case    |  2 |

### 10.3 每组样例应包含

```json
{
  "id": "color_001",
  "category": "color",
  "prompt": "A red dog sitting next to a blue cat.",
  "baseline_image": "assets/qualitative/color_001_baseline.png",
  "ours_image": "assets/qualitative/color_001_ours.png",
  "baseline_score": {},
  "ours_score": {},
  "observation": "Baseline swaps the colors, while ours preserves the red-dog and blue-cat bindings."
}
```

### 10.4 失败案例要求

必须保留失败案例。失败案例可以包括：

1. non-spatial action relation 仍然失败。
2. 小物体被 GroundingDINO 漏检。
3. 遮挡导致 bbox relation 判断错误。
4. VLMAttr 给出 false positive。
5. 视觉合理但语义错误的图像被 VLMAlign 给高分。

---

## 11. Task 6：Reward Ablation

## 11.1 核心原则

Ablation 是 Phase 1 最关键实验之一。

需要明确区分：

1. Training ablation：不同 reward 组合分别训练模型，然后评测。
2. Post-hoc reward analysis：对同一批图像用不同 reward 打分，不重新训练。
3. Short-run / subset ablation：小步数或小数据训练，用于趋势分析。
4. Full-run ablation：完整训练设置，可信度最高。

不能把 post-hoc reward analysis 写成 training ablation。

---

## 11.2 推荐优先级

### 最高优先级

必须已有或必须完成：

| Setting               | 是否需要训练 | 说明       |
| --------------------- | -----: | -------- |
| Baseline Janus-Pro-1B |      否 | 原始模型直接评估 |
| Full CompGen-GRPO     |      是 | 主实验，完整训练 |

### 强烈建议完整训练

如果算力允许，优先完整训练：

| Setting           | 目的                                           |
| ----------------- | -------------------------------------------- |
| w/o VLMAttr       | 验证属性绑定 reward 对 color / shape / texture 的贡献  |
| w/o GDinoEnhanced | 验证 grounding / spatial / counting reward 的贡献 |

这两个是最重要的 leave-one-out ablation。

### 可做 short-run 或 subset ablation

如果算力有限，可以做：

| Setting                      | 建议策略                   |
| ---------------------------- | ---------------------- |
| w/o VLMAlign / VLMOrm        | 500–800 steps 或 subset |
| HPS only                     | 500–800 steps 或 subset |
| HPS + GDinoEnhanced          | 500–800 steps 或 subset |
| Original T2I-R1 reward suite | 500–800 steps 或 subset |

### 暂缓或可选

| Setting                 | 原因                   |
| ----------------------- | -------------------- |
| 多随机种子完整训练               | 成本较高，论文增强版再做         |
| group size 4 vs 8 vs 16 | 显存成本高                |
| beta=0 vs beta>0 系统对比   | 有价值但不是 Phase 1 最高优先级 |

---

## 11.3 推荐 ablation 表格

建议最终输出：

```text
results/ablation_results.csv
results/ablation_results.md
```

字段：

```csv
setting,training_type,steps,train_prompts,eval_prompts,color,shape,texture,spatial,non_spatial,complex,average,notes
```

示例：

```csv
Full,full-run,2000,7223,17960,0.7834,0.5090,0.6756,0.2976,0.3044,0.3776,0.4913,"main result"
w/o VLMAttr,full-run,2000,7223,17960,,,,,,,,"leave-one-out"
w/o GDinoEnhanced,full-run,2000,7223,17960,,,,,,,,"leave-one-out"
w/o VLMAlign,short-run,800,subset,subset,,,,,,,,"short-run ablation; not directly comparable to full-run"
```

### 11.4 配置文件要求

建议为每个 ablation 建立独立 config：

```text
configs/ablation/full.yaml
configs/ablation/wo_vlm_attr.yaml
configs/ablation/wo_gdino_enhanced.yaml
configs/ablation/wo_vlm_align.yaml
configs/ablation/hps_only.yaml
configs/ablation/original_t2i_r1_rewards.yaml
```

每个 config 中必须显式记录：

```yaml
experiment_name:
base_model:
trainable_modules:
frozen_modules:
num_train_steps:
num_generations:
learning_rate:
beta:
reward_weights:
enabled_rewards:
disabled_rewards:
dataset:
eval_setting:
seed:
notes:
```

### 11.5 训练脚本要求

建议新增：

```text
scripts/run_ablation.sh
scripts/run_ablation_eval.sh
```

功能：

1. 传入 config。
2. 自动创建 experiment directory。
3. 保存训练日志。
4. 保存 config snapshot。
5. 保存 checkpoint。
6. 运行或提示运行 evaluation。
7. 将结果写入 `results/ablation_results.csv`。

### 11.6 关于是否每组都完整训练

结论：

1. 严格的 training ablation 中，每个 reward 组合都应该独立训练。
2. 但当前阶段不要求所有组都完整训练 2000 steps。
3. 最少需要：

   * Baseline：不训练。
   * Full：完整训练。
   * w/o VLMAttr：尽量完整训练。
   * w/o GDinoEnhanced：尽量完整训练。
4. 其他组可以先做 short-run / subset ablation，但必须在结果表中明确标注，不能和 full-run 直接等价比较。
5. 如果只是在评估时关闭某个 reward，不算训练 ablation，只能叫 post-hoc reward contribution analysis。

---

# Phase 2：Workshop-quality 版本

## 12. Phase 2 目标

Phase 2 用于把项目从秋招高收益版本推进到 workshop-quality technical report。

重点不是再堆新方法，而是补足证据链：

1. Reward reliability analysis。
2. Human evaluation。
3. Unseen prompt generalization。
4. Reward failure taxonomy。
5. Original reward suite comparison。

---

## 13. Task 7：Reward Reliability Analysis

### 13.1 任务描述

目标：证明每个 reward 大致在测量它声称要测量的东西。

建议新增：

```text
analysis/reward_reliability/
scripts/analyze_reward_reliability.py
results/reward_reliability.csv
docs/reward_reliability.md
```

### 13.2 需要计算的相关性

建议计算 Pearson 和 Spearman：

| Reward              | Expected correlated metric        |
| ------------------- | --------------------------------- |
| GDinoEnhanced       | spatial score                     |
| GDinoEnhanced-count | numeracy-related score            |
| VLMAttr             | color / shape / texture scores    |
| VLMAlign / VLMOrm   | complex / non-spatial scores      |
| HPS                 | visual quality / preference proxy |

### 13.3 输出格式

```csv
reward,metric,pearson,spearman,num_samples,notes
GDinoEnhanced,spatial,,,, 
VLMAttr,color,,,, 
VLMAttr,shape,,,, 
VLMAttr,texture,,,, 
VLMAlign,complex,,,, 
```

### 13.4 注意事项

如果相关性不高，不要直接删除结果。可以在分析中说明：

1. 自动 evaluator 本身有噪声。
2. Reward 是训练信号，不完全等价于 benchmark metric。
3. 多 reward 的目标是互补，而不是单个 reward 完全预测最终分数。
4. 低相关性可能揭示 reward mismatch，是合理的 failure analysis。

---

## 14. Task 8：Reward Failure Taxonomy

### 14.1 任务描述

需要人工或半自动整理 reward 失败案例。

建议新增：

```text
docs/reward_failure_taxonomy.md
results/reward_failure_cases.json
assets/reward_failure_cases/
```

### 14.2 失败类型

至少包括：

| Failure Type                 | Description               |
| ---------------------------- | ------------------------- |
| VLMAttr false positive       | 属性实际错误，但 VLM 判为正确         |
| VLMAttr false negative       | 属性实际正确，但 VLM 判为错误         |
| GDino missed small object    | 小物体漏检                     |
| GDino duplicate detection    | 重复检测导致计数错误                |
| Spatial bbox ambiguity       | bbox 重叠或遮挡导致空间关系判断不稳定     |
| VLMAlign semantic overrating | 图像视觉合理但语义不符合，仍被高分         |
| Reward conflict              | 美学 reward 和语义 reward 方向冲突 |

---

## 15. Task 9：小规模 Human Evaluation

### 15.1 任务描述

建议做 50–100 prompts 的 pairwise comparison。

新增：

```text
human_eval/
human_eval_prompts.json
human_eval_template.csv
human_eval_results.csv
docs/human_eval.md
```

### 15.2 评测维度

| Metric              | Question      |
| ------------------- | ------------- |
| prompt_alignment    | 哪张图更符合 prompt |
| attribute_binding   | 哪张图属性绑定更准确    |
| spatial_correctness | 哪张图空间关系更准确    |
| overall_quality     | 哪张图整体质量更好     |

### 15.3 输出表格

```csv
metric,ours_win,tie,baseline_win,ours_win_rate,num_prompts,notes
prompt_alignment,,,,,,
attribute_binding,,,,,,
spatial_correctness,,,,,,
overall_quality,,,,,,
```

### 15.4 注意事项

1. 如果只有单人评测，需要明确写作 “small-scale human inspection”。
2. 如果有多人评测，可以报告 agreement。
3. 不要夸大为大规模 human preference study。
4. 该实验主要用于辅助证明自动 benchmark 之外的可见提升。

---

## 16. Task 10：Unseen Prompt Generalization

### 16.1 任务描述

构造一个小规模 unseen compositional prompt set，验证模型不是只适配 T2I-CompBench。

新增：

```text
data/unseen_prompts/
data/unseen_prompts/unseen_compositional_prompts.json
results/unseen_eval_results.csv
docs/unseen_generalization.md
```

### 16.2 推荐规模

100–200 条即可。

| Type               | Count |
| ------------------ | ----: |
| color binding      |    30 |
| shape binding      |    30 |
| texture binding    |    30 |
| spatial relation   |    30 |
| complex            |    30 |
| non-spatial action |    30 |

### 16.3 评测方式

优先级：

1. baseline vs ours qualitative comparison。
2. 自动评分，如果现有 evaluator 支持。
3. 小规模 human pairwise comparison。

### 16.4 注意事项

1. 不要一开始做中文 prompt。
2. 不要一开始接入太多新 benchmark。
3. DrawBench / PartiPrompts 可作为后续扩展，不是当前必须任务。

---

# Phase 3：暂缓任务 / 更强论文版本

## 17. Phase 3 总原则

Phase 3 任务有研究价值，但当前暂缓。

暂缓原因：

1. 实验成本高。
2. 方法 claim 更复杂。
3. 结果不稳定时会拖慢主线。
4. 对秋招短期收益不如 ablation、qualitative examples 和 reliability analysis。

---

## 18. Task 11：Adaptive Reward Weighting

### 18.1 任务定位

这是后续潜在方法创新。如果结果能证明优于固定 reward sum，可以将方法升级为：

```text
Task-Adaptive Multi-Reward GRPO
```

### 18.2 暂缓原因

1. 需要重新训练。
2. 可能只提升部分 task，降低其他 task。
3. 容易被质疑为手调权重。
4. 需要更系统 ablation 才能支撑 claim。

### 18.3 如果后续实现，建议采用最小版本

按 task type 设置 reward weights：

| Task Type               | Weight Strategy                             |
| ----------------------- | ------------------------------------------- |
| spatial                 | GDinoEnhanced ↑                             |
| numeracy                | GDino count component ↑                     |
| color / shape / texture | VLMAttr ↑                                   |
| complex                 | GDinoEnhanced + VLMAttr + VLMAlign balanced |
| non-spatial             | VLMAlign ↑                                  |

### 18.4 需要新增配置

```yaml
adaptive_reward_weighting:
  enabled: true
  strategy: task_type_based
  weights:
    spatial:
      hps: 0.5
      gdino: 2.0
      vlm_attr: 0.5
      vlm_align: 1.0
    color:
      hps: 0.5
      gdino: 0.5
      vlm_attr: 2.0
      vlm_align: 1.0
```

具体权重需要以实际代码和实验为准，上面只是配置格式示例。

---

## 19. Task 12：Structured Attribute-Relation Reward

### 19.1 任务定位

这是更强独立贡献方向，但当前不作为 Phase 1 主线。

潜在新方法名称：

```text
Constraint-Level Reward Optimization for Compositional Text-to-Image Generation
```

### 19.2 暂缓原因

1. 需要可靠 constraint parser。
2. 需要将 prompt 拆成 objects、attributes、relations、counts。
3. 需要 constraint-level reward aggregation。
4. 需要 unseen prompt 泛化验证。
5. 工作量接近一个新项目。

### 19.3 当前可先做的轻量版本

不要先做 LLM parser。
先使用数据集已有字段：

```text
nouns
attr_nouns
spatial_info
numeracy_info
task_type
```

实现：

1. 记录 per-constraint reward。
2. 统计不同 constraint 的 satisfaction rate。
3. 分析哪些 constraint 最容易失败。
4. 作为 analysis section，而不是主方法 claim。

### 19.4 后续完整版本

后续如果要完整实现，需要：

1. LLM-based prompt parser。
2. Object existence reward。
3. Attribute-object binding reward。
4. Relation reward。
5. Count reward。
6. Holistic semantic reward。
7. Constraint-level aggregation。
8. Generalization benchmark。

---

## 20. Task 13：Cross-Benchmark Generalization

### 20.1 任务定位

这是期刊或更强会议版本需要的内容。

### 20.2 暂缓原因

1. 接入成本较高。
2. 自动评估不一定统一。
3. 不同 benchmark prompt 风格不同。
4. 短期秋招收益不如现有 benchmark 深挖。

### 20.3 后续可选 benchmark

1. DrawBench compositional subset。
2. PartiPrompts compositional subset。
3. 自建英文 compositional prompt set。
4. 可选中文 prompt set。

---

## 21. Task 14：多随机种子和更大模型

### 21.1 任务定位

这属于强论文版本的稳定性验证。

### 21.2 暂缓原因

1. 训练成本高。
2. 显存和时间压力大。
3. 对秋招不是最高优先级。

### 21.3 后续可做

1. Full model 3 seeds。
2. w/o VLMAttr 3 seeds。
3. w/o GDinoEnhanced 3 seeds。
4. Janus-Pro-7B 适配，如果算力允许。

---

# 22. 实验记录规范

## 22.1 每个实验必须保存

每次训练必须保存：

```text
experiment_name
git_commit_hash
config_snapshot
training_command
start_time
end_time
base_model
dataset_version
num_train_steps
reward_weights
enabled_rewards
disabled_rewards
seed
checkpoint_path
tensorboard_log_path
eval_command
eval_result_path
notes
```

建议新增：

```text
experiments/experiment_registry.csv
```

字段：

```csv
experiment_name,git_commit_hash,config_path,training_command,start_time,end_time,status,checkpoint_path,eval_result_path,notes
```

---

## 23. 推荐目录结构

如果当前项目没有这些目录，建议逐步补充：

```text
CompGen-GRPO/
├── README.md
├── docs/
│   ├── code_structure.md
│   ├── method_figure_spec.md
│   ├── results_summary.md
│   ├── qualitative_analysis.md
│   ├── reward_reliability.md
│   ├── reward_failure_taxonomy.md
│   ├── human_eval.md
│   ├── unseen_generalization.md
│   └── interview_defense.md
├── configs/
│   ├── grpo_full.yaml
│   └── ablation/
│       ├── full.yaml
│       ├── wo_vlm_attr.yaml
│       ├── wo_gdino_enhanced.yaml
│       ├── wo_vlm_align.yaml
│       ├── hps_only.yaml
│       └── original_t2i_r1_rewards.yaml
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_ablation.sh
│   ├── run_ablation_eval.sh
│   └── compute_result_summary.py
├── results/
│   ├── main_results.csv
│   ├── main_results.md
│   ├── ablation_results.csv
│   ├── ablation_results.md
│   ├── reward_reliability.csv
│   └── unseen_eval_results.csv
├── assets/
│   ├── qualitative/
│   └── reward_failure_cases/
├── human_eval/
│   ├── human_eval_prompts.json
│   ├── human_eval_template.csv
│   └── human_eval_results.csv
└── experiments/
    └── experiment_registry.csv
```

实际路径必须以本地代码为准，不要为了匹配本文档强行破坏原有结构。

---

# 24. 面试防御文档

## 24.1 任务描述

建议新增：

```text
docs/interview_defense.md
```

该文档用于秋招技术面。

### 24.2 必须包含的问题

```markdown
# Interview Defense

## Q1: 你和 T2I-R1 的区别是什么？
## Q2: 为什么用 GRPO，而不是 PPO / DPO / SFT？
## Q3: 为什么只训练 language_model？
## Q4: 为什么冻结 vision_model / aligner / gen_encoder？
## Q5: GDinoEnhanced 怎么计算 soft spatial score？
## Q6: VLMAttr 为什么比 GIT caption similarity 更适合属性绑定？
## Q7: reward 会不会被 hack？
## Q8: 为什么 beta=0，不加 KL 不会崩？
## Q9: 为什么 non-spatial 提升最小？
## Q10: 1B 达到 7B 96.1% 是否公平？
## Q11: 如果继续做，你会怎么增强？
## Q12: ablation 怎么证明 reward 有效？
```

### 24.3 推荐核心回答

建议写入：

```text
本项目不是重新提出 Bi-CoT-GRPO，而是在 T2I-R1 的基础上研究小型自回归 T2I 模型的组合语义对齐问题。主要贡献是面向组合约束重设计 reward suite，并在 Janus-Pro-1B 上验证其参数效率。为了证明 reward redesign 不是简单堆模块，需要补充 category-level result、reward ablation、reward reliability 和 qualitative failure analysis。
```

---

# 25. 最终执行顺序

请 Codex 按以下顺序执行，不要直接跳到 Phase 3。

## 25.1 第一批必须执行

```text
1. 检查并统一 README / docs 中的 claim 和术语。
2. 检查 reward 命名是否和代码一致，尤其是 VLMOrm / FormatReward。
3. 新增 docs/code_structure.md。
4. 新增 docs/method_figure_spec.md。
5. 整理 results/main_results.csv 和 results/main_results.md。
6. 新增 scripts/compute_result_summary.py。
7. 新增 docs/interview_defense.md。
```

## 25.2 第二批高优先级执行

```text
8. 整理 qualitative examples 目录和 JSON 格式。
9. 新增 docs/qualitative_analysis.md。
10. 建立 ablation configs。
11. 新增 scripts/run_ablation.sh 和 scripts/run_ablation_eval.sh。
12. 新增 results/ablation_results.csv 模板。
13. 优先准备 w/o VLMAttr 和 w/o GDinoEnhanced 两个 ablation。
```

## 25.3 第三批中优先级执行

```text
14. 实现 reward reliability analysis 脚本。
15. 整理 reward failure taxonomy。
16. 准备 50–100 prompts human evaluation 模板。
17. 准备小规模 unseen compositional prompt set。
```

## 25.4 暂缓执行

```text
18. Adaptive Reward Weighting。
19. Structured Attribute-Relation Reward。
20. Cross-benchmark large-scale generalization。
21. 多随机种子完整训练。
22. 更大模型适配。
```

---

# 26. 完成标准

Phase 1 完成标准：

1. README claim 准确，无明显夸大。
2. 代码结构文档完整。
3. reward 命名与代码一致。
4. 主结果表规范。
5. 至少 12 组 qualitative examples。
6. 至少完成 Full 和 Baseline 结果整理。
7. 至少准备好 w/o VLMAttr 和 w/o GDinoEnhanced 的 ablation config。
8. 如果完成 ablation 训练，需要结果写入统一 CSV。
9. 有 interview defense 文档。
10. 可以基于 README 和 docs 直接写 arXiv technical report v1。

Phase 2 完成标准：

1. 有 reward reliability correlation。
2. 有 reward failure taxonomy。
3. 有小规模 human evaluation。
4. 有 unseen prompt generalization 结果。
5. 有更完整的 qualitative success / failure 分析。

Phase 3 完成标准：

1. Adaptive weighting 有实质提升。
2. Structured reward 不只是 parser，而是带来可验证收益。
3. 有跨 benchmark 结果。
4. 有更强稳定性实验。

---

# 27. 额外注意

1. 所有实验结果必须保留原始文件，不能只写 Markdown 摘要。
2. 所有小规模实验必须明确标注 `short-run`、`subset` 或 `pilot`。
3. 所有 full-run 实验必须记录 config、seed、checkpoint、eval command。
4. 不要把训练 reward 和 evaluation metric 混为一谈。
5. 不要把自动 benchmark 的提升解释成绝对的人类偏好提升，除非有人评支撑。
6. 不要声称 GDinoEnhanced 负责颜色、形状、纹理绑定；这些主要由 VLMAttr 负责。
7. 不要声称 VLM reward 完全可靠；必须保留 reward failure analysis。
8. 不要删除失败案例，失败案例是项目可信度的一部分。
9. 不要为了更强 novelty 过早重命名方法，除非 adaptive / structured reward 确实有额外实验结果。
10. 当前最重要目标是形成清晰、可信、可复现、可面试讲透的项目证据链。
