# 方法图文字规格

本文档用于整理后续绘制论文 / README 方法图所需的信息。本阶段只整理图的文字内容，不实际绘图，也不启动训练。

## 图的目标

展示 CompGen-GRPO 如何基于 T2I-R1 / Bi-CoT-GRPO，使用多维 reward suite 对 Janus-Pro-1B 进行组合语义 text-to-image 生成优化。

## 主流程

```text
输入 prompt
  ->
Reasoning prompt template
  ->
Janus-Pro-1B language_model 生成 textual CoT / visual plan
  ->
将 raw prompt + textual CoT 拼接成 image-generation prompt
  ->
Janus-Pro-1B language_model 自回归生成 image tokens
  ->
冻结的 gen_vision_model / VQ decoder 将 image tokens 解码成候选图像
  ->
Multi-reward scoring
  ->
Group-relative advantage estimation
  ->
对 semantic CoT tokens + image tokens 计算 GRPO loss
  ->
只更新可训练的 language_model
```

## 图中需要标出的模块

### 可训练模块

- `language_model`
  - 负责生成 textual CoT。
  - 负责生成 image tokens。
  - 接收 GRPO 更新。

### 冻结模块

- `vision_model`
- `aligner`
- `gen` 相关模块，包括 image-token decoder / generation vision model。
- 奖励模型：
  - HPS v2。
  - GroundingDINO。
  - Qwen3-VL-2B。

## 奖励体系模块

图中应展示当前正式完整训练启用的四个奖励：

| 图中标签 | 代码类 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| HPSReward | `HPSv2` | prompt + image | scalar | 图像美学 / 人类偏好 |
| GDinoEnhancedReward | `GDinoEnhanced` | image + `nouns` / `spatial_info` / `numeracy_info` | scalar | 对象存在性、软空间关系、软计数 |
| VLMAttrReward | `VLMAttr` | image + `attr_nouns` | scalar | 细粒度属性-对象绑定 |
| VLMOrmReward | `VLMOrm` | image + prompt + task type | scalar | 整体语义对齐 |

注意：图中不要把 `VLMOrm` 标成 `FormatReward`。

## 奖励聚合方式

当前 trainer 行为：

```text
R_total = sum_i R_i
```

`run_train.sh` 没有显式传入 per-reward 权重。因此图中应画成简单求和，除非后续真的实现 adaptive weighting。

## GRPO 模块

对每个输入 prompt：

```text
num_generations = 4 个 textual CoT generations
new_generations_image = 每个 CoT 生成 1 张图
group size = 4
```

组内相对优势计算：

```text
advantage = (reward - group_mean_reward) / (group_std_reward + 1e-4)
```

loss 作用在两类 token 上：

- Semantic CoT token log probabilities。
- Image token log probabilities。

KL 项：

- 主训练脚本 `run_train.sh` 中 `beta = 0`。
- 图中可以标注代码支持 optional KL / reference model，但不要暗示主实验启用了 KL。

## 推荐图布局

```text
┌─────────────────┐
│ 输入 Prompt      │
└────────┬────────┘
         │
         v
┌─────────────────────────────────┐
│ Reasoning Prompt Template        │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│ Janus-Pro-1B language_model      │  可训练
│ Textual CoT / visual plan        │
└────────┬────────────────────────┘
         │ raw prompt + CoT
         v
┌─────────────────────────────────┐
│ Janus-Pro-1B language_model      │  可训练
│ 自回归生成 image tokens          │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│ 冻结的 image token decoder       │
│ 生成候选图像                     │
└────────┬────────────────────────┘
         │
         v
┌──────────────────────────────────────────────────┐
│ Multi-Reward Suite                                │
│ HPS | GDinoEnhanced | VLMAttr | VLMOrm            │
└────────┬─────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│ 组内相对优势估计                 │
└────────┬────────────────────────┘
         │
         v
┌─────────────────────────────────┐
│ GRPO update on language_model    │
└─────────────────────────────────┘
```

## 图注草稿

CompGen-GRPO 基于 T2I-R1 的 Bi-CoT-GRPO pipeline。给定输入 prompt 后，Janus-Pro-1B 首先生成 textual compositional plan，然后自回归生成 image tokens。候选图像由冻结的 generation decoder 解码，并由多维 reward suite 打分。该 reward suite 覆盖图像美学、对象 grounding、空间 / 计数正确性、属性绑定和整体语义对齐。模型在每个 prompt 的候选组内计算 group-relative advantage，并只更新 Janus-Pro-1B 的 `language_model`。
