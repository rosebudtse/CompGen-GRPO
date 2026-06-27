# StructComp-GRPO TODO

## 0. 文档目的

本文档用于定义 CompGen-GRPO 的进阶研究线：**StructComp-GRPO: Structure-Aware Reward Optimization for Compositional Text-to-Image Generation**。

目标是让另一个 agent 或新的会话读完本文档后，可以直接理解：

1. 为什么要做 StructComp-GRPO。
2. 它和 T2I-R1、当前 CompGen-GRPO 的关系。
3. 相关论文和趋势依据。
4. 最小可行版本应该实现什么。
5. 需要改哪些代码位置。
6. 实验、ablation、分析和文档如何推进。

本项目不是从零替代 T2I-R1，而是在 T2I-R1 / BiCoT-GRPO 的训练框架上，进一步研究 **组合语义约束如何被结构化建模，并转化为更可靠的 reward optimization**。

---

## 1. 一句话定位

> StructComp-GRPO builds upon T2I-R1 and CompGen-GRPO, and studies how to decompose compositional prompts into structured constraints and optimize them with constraint-aware, task-adaptive rewards.

中文：

> StructComp-GRPO 基于 T2I-R1 和当前 CompGen-GRPO，研究如何将组合式 prompt 分解为对象、属性、关系和计数约束，并通过结构感知、自适应的奖励聚合进行优化。

---

## 2. 出发点

### 2.1 T2I-R1 的贡献和局限

T2I-R1 的核心贡献是把 reasoning 和 RL 引入自回归 T2I 生成：

- semantic-level CoT：生成前的高层语义规划。
- token-level CoT：patch-by-patch 图像 token 生成过程。
- BiCoT-GRPO：在同一训练步骤中联合优化 semantic CoT 和 image token CoT。
- ensemble rewards：使用 HPS、GIT、GroundingDINO、ORM 等 reward 共同优化生成质量和语义对齐。

参考：

- T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT  
  https://arxiv.org/abs/2505.00703
- T2I-R1 official repository  
  https://github.com/CaraJ7/T2I-R1

局限：

1. reward 仍然是黑盒式 fixed ensemble。
2. 不显式建模 prompt 中的 object / attribute / relation / count。
3. 很难解释到底哪个约束失败、哪个 reward 起作用。
4. 多个 reward 直接相加，存在尺度不一致和 reward conflict。
5. 对 compositional generation 来说，整体 reward 太粗，credit assignment 不清晰。

### 2.2 当前 CompGen-GRPO 的贡献和局限

当前 CompGen-GRPO 已经完成：

- 将 T2I-R1 / BiCoT-GRPO 适配到 Janus-Pro-1B。
- 使用 HPS、GDinoEnhanced、VLMAttr、VLMOrm 组成多维 reward suite。
- 在 T2I-CompBench 上将 Janus-Pro-1B baseline average 从 0.2320 提升到 0.4913。
- 达到 reported T2I-R1 Janus-Pro-7B average 0.5114 的 96.1%。

但当前 trainer 中 reward 聚合仍是：

```python
rewards = rewards_per_func.sum(dim=1)
```

即所有 reward 固定求和，没有 task-adaptive weighting，也没有 constraint-level logging。

StructComp-GRPO 的切入点就是补上这一层。

---

## 3. 核心研究问题

### 3.1 主问题

组合式 T2I prompt 通常不是一个整体语义，而是一组可验证约束：

- Objects：有哪些对象？
- Attributes：对象的颜色、形状、材质、大小是否正确？
- Relations：对象之间的空间 / 动作 / 交互关系是否正确？
- Counts：数量是否正确？
- Holistic alignment：整体图像是否符合 prompt？

当前 fixed reward sum 难以区分这些约束。因此，本项目研究：

> Can structure-aware reward optimization improve compositional text-to-image generation beyond fixed multi-reward summation?

### 3.2 子问题

1. 如何从已有数据字段构建 constraint graph？
2. 如何为不同 constraint 设计或复用 reward？
3. 如何按 task type / constraint type 自适应聚合 reward？
4. 如何记录 per-constraint satisfaction，提升可解释性？
5. structured / adaptive reward 是否优于当前 fixed-sum CompGen？

---

## 4. 相关论文和可借鉴趋势

### 4.1 Reasoning + RL for T2I

**T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT**  
https://arxiv.org/abs/2505.00703

借鉴点：

- 使用 semantic-level CoT 做高层规划。
- 使用 GRPO 优化自回归图像生成。
- StructComp-GRPO 不重复提出 BiCoT-GRPO，而是在其 reward optimization 层做结构化增强。

### 4.2 VQA-style evaluation / reward for compositional T2I

**Evaluating Text-to-Visual Generation with Image-to-Text Generation / VQAScore**  
https://arxiv.org/abs/2404.01291

**GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation**  
https://arxiv.org/abs/2406.13743

借鉴点：

- 整体 CLIP-style similarity 不足以评估组合语义。
- VQA-style 问答更适合验证 object / attribute / relation。
- 可将 VLMAttr / VLMOrm 升级为 localized VQA reward 或 yes-probability reward。

### 4.3 Multi-preference / reward calibration

**Calibrated Multi-Preference Optimization for Aligning Diffusion Models / CaPO**  
https://arxiv.org/abs/2502.02588

**Diffusion-RPO: Aligning Diffusion Models through Relative Preference Optimization**  
https://arxiv.org/abs/2406.06382

借鉴点：

- 多 reward 之间存在尺度和偏好冲突。
- 不应无校准地直接求和。
- 可以做 reward normalization、task-specific weights、Pareto-style 或 conflict-aware aggregation。

### 4.4 Dense / step-level reward and credit assignment

**A Dense Reward View on Aligning Text-to-Image Diffusion with Preference**  
https://arxiv.org/abs/2402.08265

**Step-level Reward for Free in RL-based T2I Diffusion Model Fine-tuning**  
https://arxiv.org/abs/2505.19196

借鉴点：

- 图像生成中的 reward sparse / delayed 问题很严重。
- 虽然这些工作主要面向 diffusion timestep，但 StructComp-GRPO 可以在自回归 T2I 中做类似思想：
  - semantic CoT credit
  - image token credit
  - constraint-level credit

Phase 1 不要求实现 token-level dense credit，但可以作为后续方向。

### 4.5 Scene graph / constraint graph / curriculum for compositional generation

**Scene Graph Disentanglement and Composition for Generalizable Complex Image Generation**  
https://arxiv.org/abs/2410.00447

**Synthetic Curriculum Reinforces Compositional Text-to-Image Generation**  
https://arxiv.org/abs/2511.18378

**HiCoGen: Hierarchical Compositional Text-to-Image Generation via Reinforcement Learning**  
https://arxiv.org/abs/2511.19965

借鉴点：

- 复杂 prompt 应被拆成结构化语义单元。
- 可用 scene graph / semantic units 表达对象、属性、关系、计数。
- 可根据组合复杂度做 curriculum 或 difficulty-aware sampling。

Phase 1 使用已有字段构建轻量 constraint graph；后续再考虑 LLM parser 和 curriculum。

### 4.6 Reward model reliability

**Multimodal LLMs as Customized Reward Models for Text-to-Image Generation / LLaVA-Reward**  
https://arxiv.org/abs/2507.21391

**GenAI Arena: An Open Evaluation Platform for Generative Models**  
https://arxiv.org/abs/2406.04485

借鉴点：

- MLLM-as-judge / reward model 不是绝对可靠。
- 需要 reward reliability analysis 和 failure taxonomy。
- StructComp-GRPO 应记录 reward 与 benchmark metric 的相关性，以及典型误判。

---

## 5. 方法设计

## 5.1 总体框架

当前 CompGen-GRPO：

```text
Prompt
  -> semantic CoT
  -> image token generation
  -> generated image
  -> HPS + GDinoEnhanced + VLMAttr + VLMOrm
  -> fixed sum reward
  -> GRPO update
```

StructComp-GRPO：

```text
Prompt
  -> constraint graph construction
      objects
      attributes
      spatial relations
      counts
      holistic prompt
  -> semantic CoT
  -> image token generation
  -> generated image
  -> constraint-specific rewards
      object existence reward
      attribute binding reward
      relation reward
      count reward
      holistic alignment reward
  -> calibrated / task-adaptive aggregation
  -> per-constraint logging
  -> GRPO update
```

## 5.2 Constraint Graph

Phase 1 不做 LLM parser，直接使用数据集已有字段：

```text
nouns
attr_nouns
spatial_info
numeracy_info
task_type
raw_prompt
```

建议内部结构：

```python
constraint_graph = {
    "objects": ["dog", "cat"],
    "attributes": [
        {"object": "dog", "attribute": "red", "raw": "red dog"},
        {"object": "cat", "attribute": "blue", "raw": "blue cat"}
    ],
    "relations": [
        {"subject": "dog", "relation": "left of", "object": "cat"}
    ],
    "counts": [
        {"object": "cat", "count": 3}
    ],
    "task_type": "color",
    "prompt": "a red dog left of a blue cat"
}
```

## 5.3 Constraint-Specific Rewards

建议对应关系：

| Constraint | Reward source | 当前可复用代码 |
|---|---|---|
| object existence | GroundingDINO / GDinoEnhanced | `get_object_score` |
| attribute binding | VLMAttr | `reward_vlm.py` |
| spatial relation | GDinoEnhanced | `get_spatial_score` |
| count | GDinoEnhanced | `get_numeracy_score` |
| holistic alignment | VLMOrm / VQAScore-style VLM | `reward_vlm.py` |
| visual quality | HPS | `reward_hps.py` |

## 5.4 Adaptive Reward Aggregation

当前 fixed sum：

```python
total_reward = hps + gdino + vlm_attr + vlm_orm
```

StructComp-GRPO v1：

```python
weights = get_task_adaptive_weights(task_type)
total_reward = (
    weights["hps"] * hps +
    weights["object"] * object_reward +
    weights["attribute"] * attribute_reward +
    weights["relation"] * relation_reward +
    weights["count"] * count_reward +
    weights["holistic"] * holistic_reward
)
```

初始权重建议只作为实验起点，必须通过 ablation 验证：

```yaml
attribute:
  hps: 0.5
  object: 0.5
  attribute: 2.0
  relation: 0.0
  count: 0.0
  holistic: 1.0

spatial:
  hps: 0.5
  object: 0.5
  attribute: 0.0
  relation: 2.0
  count: 0.0
  holistic: 1.0

numeracy:
  hps: 0.5
  object: 0.5
  attribute: 0.0
  relation: 0.0
  count: 2.0
  holistic: 1.0

complex:
  hps: 0.5
  object: 0.5
  attribute: 1.0
  relation: 1.0
  count: 1.0
  holistic: 1.0

non_spatial:
  hps: 0.5
  object: 0.5
  attribute: 0.5
  relation: 0.0
  count: 0.0
  holistic: 2.0
```

注意：

- 权重不能直接写成最终结论。
- 必须在文档里标注为 heuristic initial weights。
- 需要 fixed-sum baseline 对比。

## 5.5 Reward Calibration

最低版本：

- 先不做复杂 Pareto。
- 对每个 reward 在 batch 内做均值 / 方差统计。
- 记录 raw reward 和 weighted reward。

可选版本：

```python
calibrated_reward = (reward - running_mean) / (running_std + 1e-6)
```

需要小心：

- GRPO 本身已经做 group-relative advantage。
- reward calibration 不应破坏组内相对差异。
- Phase 1 可以先只记录 calibration statistics，不一定启用。

---

## 6. 代码改造计划

## 6.1 当前关键代码入口

训练入口：

```text
src/t2i-r1/src/open_r1/grpo.py
src/t2i-r1/src/open_r1/trainer/grpo_trainer.py
src/t2i-r1/src/run_train.sh
```

Reward 文件：

```text
src/t2i-r1/src/utils/reward_hps.py
src/t2i-r1/src/utils/reward_gdino_enhanced.py
src/t2i-r1/src/utils/reward_vlm.py
```

当前 fixed sum 位置：

```text
src/t2i-r1/src/open_r1/trainer/grpo_trainer.py
```

查找：

```python
rewards = rewards_per_func.sum(dim=1)
```

## 6.2 新增文件建议

```text
src/t2i-r1/src/utils/constraint_graph.py
src/t2i-r1/src/utils/reward_aggregator.py
configs/structcomp/
configs/structcomp/fixed_sum.yaml
configs/structcomp/task_adaptive.yaml
configs/structcomp/task_adaptive_no_calib.yaml
results/structcomp_results.csv
results/structcomp_ablation_results.csv
docs/structcomp_grpo_todo.md
docs/structcomp_method.md
docs/structcomp_ablation_plan.md
```

## 6.3 `constraint_graph.py`

职责：

1. 从 `nouns / attr_nouns / spatial_info / numeracy_info / task_type / raw_prompt` 构建 constraint graph。
2. 提供统一字段，供 reward aggregator 和 logging 使用。
3. 不修改原始数据文件。

建议函数：

```python
def build_constraint_graph(example: dict) -> dict:
    ...

def parse_attr_noun(attr_noun: str) -> dict:
    ...

def constraint_summary(graph: dict) -> dict:
    ...
```

## 6.4 `reward_aggregator.py`

职责：

1. 将 `rewards_per_func` 和 reward names 映射到结构化 reward dict。
2. 按 task type 获取权重。
3. 输出 total reward、per-constraint reward、weighted reward。
4. 支持 fixed-sum 和 task-adaptive 两种模式。

建议类：

```python
class RewardAggregator:
    def __init__(self, mode: str, weights: dict | None = None):
        ...

    def aggregate(self, rewards_per_func, reward_names, batch_metadata):
        ...
```

输出：

```python
{
    "total_reward": total_reward_tensor,
    "raw_rewards": {...},
    "weighted_rewards": {...},
    "constraint_rewards": {...},
}
```

## 6.5 Trainer 改造

在 `grpo_trainer.py` 中：

1. 保留原有 fixed-sum 路径，确保可回退。
2. 新增参数：

```text
--reward_aggregation_mode fixed_sum | task_adaptive
--reward_weights_config path/to/yaml
--log_constraint_rewards true/false
```

3. 将：

```python
rewards = rewards_per_func.sum(dim=1)
```

替换为：

```python
aggregation_output = self.reward_aggregator.aggregate(
    rewards_per_func=rewards_per_func,
    reward_names=reward_names,
    batch_metadata=reward_kwargs,
)
rewards = aggregation_output["total_reward"]
```

4. 日志中新增：

```text
rewards/raw/hps
rewards/raw/gdino
rewards/raw/vlm_attr
rewards/raw/vlm_orm
rewards/weighted/object
rewards/weighted/attribute
rewards/weighted/relation
rewards/weighted/count
rewards/weighted/holistic
constraint_satisfaction/object
constraint_satisfaction/attribute
constraint_satisfaction/relation
constraint_satisfaction/count
```

---

## 7. 实验计划

## 7.1 Phase 1: 最小可行闭环

目标：证明 StructComp-GRPO 不是空概念，而是在当前 fixed-sum CompGen 基础上有可测收益或更强可解释性。

必须完成：

1. fixed-sum CompGen baseline 结果整理。
2. task-adaptive reward aggregation 实现。
3. constraint-level logging 实现。
4. 小规模或完整训练对比。
5. ablation 表格。

建议实验：

| Setting | 训练类型 | 说明 |
|---|---|---|
| Janus-Pro-1B baseline | no training | 原始模型 |
| CompGen fixed-sum | full-run | 当前主结果 |
| StructComp task-adaptive | full-run or short-run | 核心方法 |
| StructComp w/o adaptive | same as above | 退化为 fixed-sum |
| StructComp w/o VLMAttr | full or short | 验证属性 reward |
| StructComp w/o GDinoEnhanced | full or short | 验证 grounding / relation / count |

如果算力不足：

- 先做 500-800 step short-run。
- 使用 representative prompt subset。
- 明确标注为 pilot / short-run。
- 不能把 short-run 和 full-run 直接等价比较。

## 7.2 Phase 2: Reward Reliability

目标：证明 reward 不是盲目堆叠。

输出：

```text
results/reward_reliability.csv
docs/reward_reliability.md
```

分析：

| Reward | Expected metric |
|---|---|
| GDino relation | spatial score |
| GDino count | numeracy-related score |
| VLMAttr | color / shape / texture |
| VLMOrm | complex / non-spatial |
| HPS | visual quality proxy |

报告 Pearson / Spearman，如果低相关也保留，并分析 reward mismatch。

## 7.3 Phase 3: Qualitative and Failure Taxonomy

输出：

```text
docs/structcomp_qualitative_analysis.md
docs/structcomp_failure_taxonomy.md
assets/structcomp_qualitative/
```

至少包含：

- attribute success
- spatial success
- count success
- complex success
- non-spatial failure
- reward conflict case
- VLM false positive
- GDino missed object

## 7.4 Phase 4: Optional Extensions

暂缓，除非 Phase 1-3 完成：

1. LLM-based prompt parser。
2. difficulty-aware curriculum。
3. dense token-level credit assignment。
4. human evaluation。
5. cross-benchmark generalization。

---

## 8. Ablation 设计

## 8.1 必须区分的实验类型

1. Training ablation：不同配置分别训练。
2. Post-hoc reward analysis：同一批图像用不同 reward 打分。
3. Short-run ablation：少步数 / 子集训练。
4. Full-run ablation：完整训练。

不要把 post-hoc reward analysis 写成 training ablation。

## 8.2 推荐 ablation 表

```csv
setting,aggregation,training_type,steps,train_prompts,eval_prompts,color,shape,texture,spatial,non_spatial,complex,average,notes
CompGen-FixedSum,fixed_sum,full-run,2000,7223,17960,0.7834,0.5090,0.6756,0.2976,0.3044,0.3776,0.4913,current main result
StructComp-Adaptive,task_adaptive,full-run,2000,7223,17960,,,,,,,,main method
StructComp-Adaptive-Short,task_adaptive,short-run,800,subset,subset,,,,,,,,pilot
StructComp-w/o-VLMAttr,task_adaptive,full-or-short,,,,,,,,,,leave-one-out
StructComp-w/o-GDino,task_adaptive,full-or-short,,,,,,,,,,leave-one-out
```

---

## 9. 论文/项目叙事

## 9.1 推荐标题

```text
StructComp-GRPO: Structure-Aware Reward Optimization for Compositional Text-to-Image Generation
```

备选：

```text
From Multi-Reward to Structure-Aware Reward Optimization for Compositional Text-to-Image Generation
```

## 9.2 核心 thesis

```text
While T2I-R1 introduces reasoning-enhanced GRPO for image generation, it treats reward feedback as a fixed ensemble of black-box scores. We argue that compositional T2I generation requires structure-aware reward optimization, where prompts are decomposed into object, attribute, relation, and counting constraints and optimized with constraint-specific, task-adaptive rewards.
```

## 9.3 贡献点草案

1. We formulate compositional T2I alignment as structured constraint satisfaction over objects, attributes, relations, and counts.
2. We propose a constraint-aware reward aggregation strategy that adapts reward weights according to task and constraint types.
3. We extend CompGen-GRPO with constraint-level logging and reward reliability analysis, making reward optimization more interpretable.
4. We evaluate StructComp-GRPO against fixed-sum multi-reward GRPO on T2I-CompBench and report category-level gains and failure modes.

## 9.4 必须避免的 claim

不要写：

- We propose BiCoT-GRPO.
- We propose GRPO for T2I from scratch.
- We surpass T2I-R1 in all settings, unless experiments prove it.
- Our reward models are fully reliable.
- GDinoEnhanced solves attribute binding.
- Adaptive weights are universally optimal.

可以写：

- We build upon T2I-R1 / BiCoT-GRPO.
- We improve the reward optimization layer for compositional T2I.
- We study structured constraints and task-adaptive reward aggregation.
- We provide reward reliability and failure analysis.

---

## 10. 实现优先级

### Priority 0: 准备与对齐

- [ ] 复查当前 CompGen 代码，确认 fixed-sum reward 位置。
- [ ] 确认当前主结果表和训练配置。
- [ ] 新增本文档。
- [ ] 新增 `docs/structcomp_method.md` 草稿。

### Priority 1: 最小方法实现

- [ ] 新增 `constraint_graph.py`。
- [ ] 新增 `reward_aggregator.py`。
- [ ] 在 training args 中加入 `reward_aggregation_mode`。
- [ ] 保留 fixed-sum 路径。
- [ ] 实现 task-adaptive aggregation。
- [ ] 实现 raw reward / weighted reward logging。

### Priority 2: 训练配置

- [ ] 新增 `configs/structcomp/fixed_sum.yaml`。
- [ ] 新增 `configs/structcomp/task_adaptive.yaml`。
- [ ] 新增 `configs/structcomp/wo_vlm_attr.yaml`。
- [ ] 新增 `configs/structcomp/wo_gdino.yaml`。
- [ ] 新增运行脚本或改造现有脚本支持配置。

### Priority 3: 实验

- [ ] 先跑 short-run pilot。
- [ ] 检查 reward 曲线是否稳定。
- [ ] 检查各 task 的 weighted reward 是否符合预期。
- [ ] 如 pilot 有正向趋势，再跑 full-run。
- [ ] 评测 T2I-CompBench。

### Priority 4: 分析

- [ ] 整理 category-level result。
- [ ] 整理 reward ablation。
- [ ] 整理 reward reliability。
- [ ] 整理 qualitative examples。
- [ ] 整理 failure taxonomy。

### Priority 5: 写作

- [ ] 更新 README 的 Relation to T2I-R1。
- [ ] 新增 StructComp 方法图说明。
- [ ] 写 Method section。
- [ ] 写 Experiments section。
- [ ] 写 Limitations。

---

## 11. 完成标准

StructComp-GRPO v1 完成标准：

1. 代码支持 fixed-sum 和 task-adaptive 两种 reward aggregation。
2. 能从已有数据字段构建 constraint graph 或等价 metadata。
3. 日志中能看到 raw reward、weighted reward、constraint-level reward。
4. 至少完成 short-run pilot，并与 fixed-sum 对比。
5. 有 ablation result template。
6. 有 reward reliability 分析计划或初步结果。
7. 有 qualitative / failure case 目录和文档。
8. README / docs 中清楚说明与 T2I-R1 和 CompGen-GRPO 的关系。

更强版本完成标准：

1. 完整训练 StructComp-Adaptive。
2. 在 T2I-CompBench 上相比 fixed-sum CompGen 有平均分或关键类别提升。
3. 至少完成 w/o VLMAttr 和 w/o GDinoEnhanced ablation。
4. 有 reward reliability correlation。
5. 有 12 组以上 qualitative examples。
6. 有 failure taxonomy。

---

## 12. 当前建议路线

最稳路线：

```text
Step 1: 保持 T2I-R1 / CompGen 主线不动
Step 2: 只改 reward aggregation 和 logging
Step 3: 用已有数据字段做 constraint graph
Step 4: short-run 验证 task-adaptive 是否有趋势
Step 5: 若趋势成立，跑 full-run
Step 6: 用 ablation + reliability + failure analysis 讲完整故事
```

不要一开始做：

- LLM parser。
- dense token-level credit assignment。
- large-scale human eval。
- cross-benchmark generalization。
- 多随机种子完整训练。

这些都可以作为后续扩展，而不是 v1 的阻塞项。

