# 项目表述与术语一致性检查

本文档根据 `docs/codex_project_plan.md` 的 Phase 1 要求，对 `README.md`、`docs/` 和本地代码中的项目定位、术语和 claim 进行核对。

## 总结

本项目应定位为 **基于 T2I-R1 / Bi-CoT-GRPO 的扩展工作**，而不是从零提出新的 T2I 强化学习算法。当前核心贡献是：将框架适配到 Janus-Pro-1B，并针对组合语义生成任务重设计多维奖励体系。

当前最需要修正的地方是 `VLMOrm`、`ORM`、`FormatReward` 的命名混用：本地当前正式训练脚本启用的是 `vlm_orm`，它是基于 Qwen3-VL 的整体语义对齐奖励；但 `README.md` 里仍有表格把 `ORM` 描述为“基于规则的格式正确性奖励”，这和当前训练代码不一致。

## 可以安全使用的表述

- 本项目基于 T2I-R1 / Bi-CoT-GRPO。
- 当前正式训练脚本将该框架适配到 Janus-Pro-1B。
- `src/t2i-r1/src/run_train.sh` 中启用的 reward 是 `hps gdino vlm_attr vlm_orm`。
- `GDinoEnhanced` 提供对象存在性、软空间关系、软计数奖励。
- `VLMAttr` 使用 Qwen3-VL VQA 做细粒度属性-对象绑定判断。
- `VLMOrm` 使用 Qwen3-VL 做整体 prompt-image 语义对齐评分，输出 0-10 分后归一化到 `[0, 1]`。
- 从本地 T2I-CompBench 结果文件计算，finetuned Janus-Pro-1B 平均分为 `0.4913`，baseline Janus-Pro-1B 平均分为 `0.2320`。
- finetuned 1B 模型在 T2I-CompBench 平均分上达到 T2I-R1 Janus-Pro-7B 报告分数 `0.5114` 的 `96.1%`。

## 需要避免或改写的表述

| 当前/风险表述 | 问题 | 推荐写法 |
|---|---|---|
| 不加限定地写 “Multi-Reward GRPO Framework” | 容易让读者误以为 GRPO 框架本身是本项目提出的。 | “基于 T2I-R1 / Bi-CoT-GRPO 的多奖励扩展工作”。 |
| “achieve 96.1% of T2I-R1's 7B performance using only 1B parameters” | 基本可用，但需要限定为 T2I-CompBench 平均分和 reported 7B 结果。 | “在 T2I-CompBench 平均分上达到 reported T2I-R1 Janus-Pro-7B 的 96.1%”。 |
| “GroundingDINO spatial reward” | 过窄。代码中还包含对象存在性和计数分支。 | “GDinoEnhanced object grounding、spatial relation、numeracy reward”。 |
| “GDinoEnhanced 处理 color / shape / texture 属性绑定” | 当前代码不支持这个 claim。属性类任务主要从 GDinoEnhanced 得到对象存在性约束。 | “VLMAttr 负责属性绑定；GDinoEnhanced 为非 spatial / numeracy 任务提供对象存在性约束”。 |
| “ORM = rule-based format correctness” | 与当前启用的 `vlm_orm` 不一致。 | “VLMOrm 是基于 Qwen3-VL 的整体语义对齐奖励；旧版 `reward_orm.py` 当前正式训练未启用”。 |
| “Reward weights” | 当前训练代码是直接对各奖励输出求和，`run_train.sh` 没有显式传入单项奖励权重。 | “当前启用 HPS、GDinoEnhanced、VLMAttr、VLMOrm，trainer 中直接求和”。 |

## README 需要修正的点

建议后续更新 README 时完成以下事项：

1. 新增 `Relation to T2I-R1` 小节。
2. 将可能暗示“从零提出新 RL 框架”的表述改成“builds upon T2I-R1 / Bi-CoT-GRPO”。
3. 将 reward 表格改成与代码一致的命名：

| 奖励名称 | 代码类 | 当前正式训练是否启用 | 作用 |
|---|---|---|---|
| HPSReward | `HPSv2` | 是 | 图像美学 / 人类偏好分数 |
| GDinoEnhancedReward | `GDinoEnhanced` | 是 | 对象存在性、软空间关系、软计数 |
| VLMAttrReward | `VLMAttr` | 是 | 属性-对象绑定 |
| VLMOrmReward | `VLMOrm` | 是 | Qwen3-VL 0-10 整体语义对齐评分 |
| Legacy ORM | `reward_orm.py` 中的 `ORM` | 当前 `run_train.sh` 未启用 | LLaVA yes/no outcome reward |
| FormatReward | 当前正式训练中未发现独立启用类 | 否 | 不应声称已启用，除非后续单独实现并明确区分 |

4. 修正脚本路径：
   - 当前正式训练入口：`src/t2i-r1/src/run_train.sh`。
   - 当前评测脚本：`src/t2i-r1/src/run_eval.sh`。
   - `src/scripts/run_grpo.sh` 存在，但更像备用或旧版启动脚本。
   - 本地不存在 `src/scripts/run_eval.sh`。

## VLMOrm / FormatReward 一致性结论

当前代码状态：

- `src/t2i-r1/src/utils/reward_vlm.py` 定义了 `VLMOrm`。
- `VLMOrm` 继承 `_Qwen3VLBase`，使用 4-bit 方式加载 Qwen3-VL-2B，针对不同 `task_type` 构造整体语义评分问题，解析 0-10 整数分数，裁剪后返回 `score / 10.0`。
- `src/t2i-r1/src/open_r1/trainer/grpo_trainer.py` 会把包含 `vlm_orm` 的 reward 字符串映射到 `VLMOrm(args)`。
- `src/t2i-r1/src/run_train.sh` 启用了 `--reward_funcs hps gdino vlm_attr vlm_orm`。
- `src/t2i-r1/src/utils/reward_orm.py` 中存在旧版 `ORM` 类，使用 LLaVA yes/no 概率作为 outcome reward，但当前正式训练脚本没有启用它。
- 当前正式完整训练中没有发现独立启用的 `FormatReward` 类。

结论：

当前正式训练配置中的 `VLMOrm` 是真实的基于 VLM 的整体语义对齐奖励，不应描述成“基于规则的格式正确性奖励”。如果文档中需要讨论格式奖励，应明确标注为旧版或当前未启用，除非后续新增独立 `FormatReward` 实现。
