# CompGen-GRPO 论文 TODO

## 当前定位

这个项目应定位为 T2I-R1 的扩展工作，而不是一个完全独立提出的 GRPO 框架。

推荐表述：

> 本工作基于 T2I-R1 / Bi-CoT-GRPO，研究紧凑型自回归 T2I 模型能否通过多维奖励重设计获得更强的组合语义生成能力。

当前可发表性判断：

- arXiv 技术报告：完成论文写作并厘清贡献边界后可以发。
- Workshop paper：补充 reward ablation 和 reward reliability analysis 后有机会。
- 期刊 / 更强会议：需要更强的方法创新、更广泛的评测和更系统的证据。

避免这样写：

- 我们提出了 Bi-CoT-GRPO。
- 我们从零提出了新的 T2I 强化学习框架。
- GDinoEnhanced 直接解决 color / shape / texture 属性绑定。
- 我们超过了 T2I-R1 7B。

可以安全这样写：

- 我们将 T2I-R1 的 Bi-CoT-GRPO 适配到 Janus-Pro-1B。
- 我们针对组合语义对齐重设计了多维 reward suite。
- VLMAttr 负责细粒度属性绑定。
- GDinoEnhanced 提供对象存在性、软空间关系和软计数奖励。
- 1B 模型在 T2I-CompBench 上达到 T2I-R1 7B 报告平均分的 96.1%。

## Priority 0：整理项目叙事

- [ ] 确定论文标题。
  - 候选：Multi-Reward GRPO for Parameter-Efficient Compositional Text-to-Image Generation
  - 后续强化版：Structure-Aware Adaptive Reward Optimization for Compositional Text-to-Image Generation
- [ ] 写一段项目核心 thesis。
  - 问题：T2I 的组合语义生成失败。
  - 基础框架：T2I-R1 / Bi-CoT-GRPO。
  - 贡献：小模型适配 + 多维 reward redesign。
  - 结果：Janus-Pro-1B 在 T2I-CompBench 上从 0.2320 提升到 0.4913。
- [ ] 更新 README 和论文草稿，明确 acknowledge T2I-R1。
  - 说明 Bi-CoT-GRPO 来自 T2I-R1。
  - 说明本项目重点是 reward redesign 和 compact-model empirical validation。
- [ ] 统一术语。
  - 统一写作 T2I-CompBench。
  - 统一写作 Janus-Pro-1B。
  - GDinoEnhanced 只用于 object grounding / spatial / numeracy rewards。
  - VLMAttr 用于 color / shape / texture attribute binding。
  - VLMOrm 用于 holistic semantic alignment。

## Priority 1：arXiv v1

目标：先完成一版可信的 technical report，再继续做更深的扩展。

- [ ] 创建 LaTeX 论文骨架。
  - Abstract
  - Introduction
  - Related Work
  - Method
  - Experiments
  - Analysis
  - Limitations
  - Conclusion
- [ ] 写贡献点。
  - Contribution 1：在有限算力下将 T2I-R1 Bi-CoT-GRPO 适配到 Janus-Pro-1B。
  - Contribution 2：提出面向组合语义对齐的多维 reward suite。
  - Contribution 3：改进 detection-based reward，引入软空间评分和软计数惩罚。
  - Contribution 4：在 T2I-CompBench 上取得显著提升，达到 T2I-R1 7B 报告性能的 96.1%。
- [ ] 添加方法图。
  - Prompt -> reasoning CoT -> image tokens -> generated image。
  - Rewards：HPS、GDinoEnhanced、VLMAttr、VLMOrm。
  - GRPO group-relative advantage update。
- [ ] 添加 reward decomposition table。
  - HPS：美学 / 人类偏好。
  - GDinoEnhanced：对象存在性、空间关系、计数。
  - VLMAttr：属性-对象绑定。
  - VLMOrm：整体语义对齐。
- [ ] 添加数据集统计表。
  - spatial：2,195
  - numeracy：1,172
  - color：879
  - complex：700
  - non / non-spatial：700
  - texture：695
  - shape：682
  - object：200
  - total：7,223
- [ ] 添加主结果表。
  - Baseline Janus-Pro-1B average：0.2320
  - Ours average：0.4913
  - T2I-R1 7B reported average：0.5114
  - Relative improvement over baseline：111.8%
  - Ours / T2I-R1 7B：96.1%
- [ ] 添加 limitations。
  - 基于 T2I-R1，不是新的 RL algorithm。
  - 自动 benchmark 可能存在 evaluator bias。
  - non-spatial action relations 仍然较弱。
  - reward models 可能误判图像或被 reward hacking。
  - 需要完整 reward ablation 支撑更强 causal claim。

## Priority 2：Reward Ablation

这是把项目从“好项目”升级成“研究论文”的最关键实验。

- [ ] 运行或近似运行 reward ablation。
  - Baseline Janus-Pro-1B。
  - hps + gdino + vlm\_attr（验证整体语义Orm对齐 reward 贡献）
  - hps + gdino + vlm\_orm（验证属性级 reward 贡献）
  - hps + vlm\_attr + vlm\_orm（验证 object / spatial / count reward 贡献）
  - HPS + GDinoEnhanced + VLMAttr + VLMOrm（FULL）
- [ ] 如果完整训练太贵，做小规模 ablation。
  - 减少训练步数。
  - 使用代表性训练 prompt 子集。
  - 在 T2I-CompBench 子集上评测。
  - 明确标注为 subset ablation。
- [ ] 如可行，对比原版 T2I-R1 风格 rewards。
  - HPS + original GDino + GIT + ORM。
  - HPS + GDinoEnhanced + VLMAttr + VLMOrm。
- [ ] 报告类别级别变化。
  - VLMAttr 应主要影响 color / shape / texture。
  - GDinoEnhanced 应主要影响 spatial / numeracy-related behavior。
  - VLMOrm 应帮助 complex / non-spatial holistic alignment。
  
  **档位 C：paper-oriented**
  训练 full-run：7–9 组
  实验	用途
  Full Ours	主结果
  w/o VLMAttr	消融
  w/o Grounding	消融
  w/o VLMOrm	消融
  Original T2I-R1 reward suite	同框架公平对比
  StructComp-Adaptive	进阶方法
  StructComp-Normalized	reward scale
  G=4 Full	group sensitivity
  7B Ours Full	scaling validation


## Priority 3：Reward Reliability Analysis

目标：证明每个 reward 确实在测量它声称要测量的东西。

- [ ] 计算 reward score 和 T2I-CompBench 各类别分数的相关性。
  - GDinoEnhanced vs spatial score。
  - VLMAttr vs color / shape / texture scores。
  - VLMOrm vs complex / non-spatial scores。
  - HPS vs image quality / preference，如果有可用指标。
- [ ] 人工检查 reward failure。
  - VLMAttr false positives / false negatives。
  - GDinoEnhanced 漏检小物体。
  - 遮挡或检测错误导致 bbox relation 判断错误。
  - VLMOrm 给视觉合理但语义错误的图像高分。
- [ ] 添加定性案例。
  - 成功案例：属性绑定被修正。
  - 成功案例：空间关系被修正。
  - 失败案例：动作 / 交互关系仍然错误。
  - 失败案例：reward evaluator 和人类判断不一致。
- [ ] 考虑小规模人工评测。
  - 50-100 个 prompts。
  - Pairwise comparison：baseline vs finetuned。
  - 标准：prompt alignment、attribute binding、spatial correctness、overall quality。
  - 报告 win rate。

## Priority 4：Adaptive Reward Weighting

如果结果能证明优于固定 reward 相加，这可以成为更强的方法贡献。

- [ ] 定义 task-adaptive reward weights。
  - spatial：提高 GDinoEnhanced。
  - numeracy：提高 GDinoEnhanced count component。
  - color / shape / texture：提高 VLMAttr。
  - complex：平衡 GDinoEnhanced + VLMAttr + VLMOrm。
  - non-spatial：提高 VLMOrm，或加入 relation-specific VLM prompts。
- [ ] 实现 reward weighting。
  - 在 training args 中加入 reward weights。
  - 加入 task-type based reward aggregation。
  - 记录 per-reward 和 weighted total reward。
- [ ] 对比 fixed vs adaptive weighting。
  - Fixed sum reward。
  - Task-adaptive reward。
  - 可选 curriculum：前期偏 HPS / VLMOrm，后期提高 attribute / spatial rewards。
- [ ] 如果结果提升，以此重命名方法。
  - 候选：Task-Adaptive Multi-Reward GRPO。

## Priority 5：Structured Attribute-Relation Reward

这是走向更强独立贡献的最佳方向。

- [ ] 将 prompts 转换成结构化约束。
  - Objects。
  - Attributes。
  - Relations。
  - Counts。
- [ ] 定义 constraint-level rewards。
  - Object existence reward。
  - Attribute-object binding reward。
  - Relation reward。
  - Count reward。
  - Holistic semantic reward。
- [ ] 实现 constraint parser。
  - 先使用数据集中已有字段：nouns、attr\_nouns、spatial\_info、numeracy\_info。
  - 后续扩展为 LLM-based prompt parsing，用于 unseen prompts。
- [ ] 在 constraint level 聚合 rewards。
  - 对 constraints 取平均。
  - 按 task type 加权。
  - 记录 per-constraint satisfaction rate。
- [ ] 添加分析。
  - 哪类 constraints 最容易失败？
  - RL 是提升所有 constraints，还是只提升简单 constraints？
  - visual quality 和 constraint satisfaction 是否存在 trade-off？

## Priority 6：Generalization Tests

目标：证明方法不是只适配 T2I-CompBench evaluator。

- [ ] 构建 unseen compositional prompt set。
  - Attribute binding prompts。
  - Spatial relation prompts。
  - Numeracy prompts。
  - Complex multi-constraint prompts。
  - Non-spatial action prompts。
- [ ] 在其他 benchmark 或子集上评测。
  - DrawBench compositional subset。
  - PartiPrompts compositional subset。
  - 自建英文 prompt set。
  - 可选中文 prompt set。
- [ ] 定性和定量比较 baseline vs finetuned。
  - 能自动评分则用自动评分。
  - 自动指标较弱时用 human preference。
- [ ] 报告 out-of-distribution prompts 上的 failure modes。

## Priority 7：论文 / Venue 策略

- [ ] arXiv v1。
  - 定位：technical report。
  - 需要：清晰写作、明确引用 T2I-R1、主结果、limitations。
- [ ] arXiv v2。
  - 加入 reward ablation。
  - 加入 reliability analysis。
  - 加入 qualitative examples。
  - 如果完成，加入 generalization tests。
- [ ] Workshop submission。
  - Priority 2 和 Priority 3 完成后最适合。
  - 定位为 compositional T2I 的 empirical study + reward modeling method。
- [ ] Journal / stronger venue。
  - 只有在 Priority 4、5、6 有实质结果后再考虑。
  - 需要更强的方法身份：adaptive 或 structured reward optimization。

## 推荐时间线

### arXiv v1 之前

- [ ] 清理 README 和项目笔记。
- [ ] 写 paper skeleton。
- [ ] 画 method figure。
- [ ] 添加 main result table。
- [ ] 添加 limitations 和 ethical attribution。

### workshop-quality 版本之前

- [ ] 完成 reward ablation。
- [ ] 完成 reward reliability analysis。
- [ ] 添加 qualitative success / failure cases。
- [ ] 如果可行，添加小规模 human evaluation。

### journal-oriented 版本之前

- [ ] 实现 adaptive reward weighting。
- [ ] 实现 structured attribute-relation reward。
- [ ] 加入 cross-benchmark generalization。
- [ ] 加入更强 baselines 和 human evaluation。

## 关键风险

- 如果没有系统评估 reward redesign，工作可能被认为只是 T2I-R1 的工程扩展。
- 如果没有 human evaluation 或 generalization tests，自动 benchmark 的提升可能被质疑。
- VLM rewards 可能引入 evaluator bias 或 reward hacking。
- 关于 GDinoEnhanced 的 claim 必须准确：它不直接解决 color / shape / texture binding。
- 最强的 novelty 路线不是“加更多 reward”，而是“面向组合约束的 structured and adaptive reward modeling”。

## Bottom Line

当前项目足够整理成可信的 arXiv technical report，也对秋招有帮助。若要升级为 paper-level 工作，优先做 reward ablation 和 reward reliability analysis。若要摆脱“只是 T2I-R1 扩展”的印象，需要进一步发展 adaptive reward weighting 和 structured constraint-level rewards。
