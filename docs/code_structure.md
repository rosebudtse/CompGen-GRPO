# 代码结构说明

本文档用于 Phase 1 的代码结构梳理，只描述当前仓库中真实存在的文件和入口。

## 顶层结构

- `README.md`：项目概览、当前结果、快速开始和致谢。
- `data/geneval_and_t2i_data_final.json`：JSONL 格式训练数据，共 7,223 条样本。
- `data/prompt/reasoning_prompt.txt`：生成图像前用于引导 textual planning / reasoning CoT 的 prompt 模板。
- `eval_results/`：本地保存的 baseline 和 finetuned T2I-CompBench 评测输出。
- `figs/`：已有项目图片。
- `src/requirements.txt`：训练相关 Python 依赖。
- `src/t2i-r1/`：基于 T2I-R1 修改后的训练、reward、生成和评测代码。

## 训练入口

### 当前正式训练脚本

- `src/t2i-r1/src/run_train.sh`

这是当前最应作为 full run 依据的训练脚本。关键配置如下：

- 模型：Janus-Pro-1B snapshot path。
- 数据集：`data/geneval_and_t2i_data_final.json`。
- 训练入口：`open_r1/grpo.py`。
- DeepSpeed 配置：`src/t2i-r1/configs/zero2.json`。
- 最大训练步数：`2000`。
- 单卡 micro batch size：`1`。
- 梯度累积：`4`。
- 每个 prompt 生成 textual CoT 数量：`4`。
- 每个 textual CoT 生成图像数量：`1`。
- 每张图像 token 数：`576`。
- 图像尺寸：训练参数默认 `384`。
- patch size：训练参数默认 `16`。
- CFG weight：`5`。
- beta / KL 系数：`0`。
- 日志：TensorBoard。
- 启用 reward：`hps gdino vlm_attr vlm_orm`。

### 备用 / 旧版训练脚本

- `src/scripts/run_grpo.sh`

这个脚本同样启用了 `hps gdino vlm_attr vlm_orm`，但超参数不同，例如 `max_steps=1600`、`gradient_accumulation_steps=8`、`beta=0.01`、使用 W&B 日志。除非明确说明，否则不要把它当作当前主结果的配置依据。

## GRPO 训练代码

- `src/t2i-r1/src/open_r1/grpo.py`

主要职责：

- 扩展 `GRPOConfig`，加入图像生成和 reward checkpoint 相关参数。
- 读取 JSON / CSV / Parquet 数据集。
- 从 `reasoning_prompt_path` 读取 reasoning prompt。
- 将每条训练样本转换成 Janus 使用的 conversation 格式。
- 根据 `nouns` 构造 GroundingDINO 的 `det_prompt` 和 token spans。
- 将 `nouns`、`attr_nouns`、`spatial_info`、`numeracy_info` 传入后续 reward 计算。
- 注册 reward 字符串名称，包括 `hps`、`gdino`、`vlm_attr`、`vlm_orm`。
- 初始化 `JanusT2IR1Trainer`。

- `src/t2i-r1/src/open_r1/trainer/grpo_trainer.py`

主要职责：

- 通过 `AutoModelForCausalLM` 加载 Janus-Pro-1B。
- 冻结参数名以 `vision_model`、`aligner` 或 `gen` 开头的模块。
- 保持 `language_model` 可训练。
- 先生成 textual CoT。
- 将 raw prompt 与生成的 CoT 拼接成 image-generation prompt。
- 使用 classifier-free guidance 自回归生成 image tokens。
- 通过冻结的 generation vision model 解码图像 tokens。
- 调用启用的 reward functions。
- 对各 reward 输出求和得到总 reward。
- 在 `num_generations * new_generations_image` 组内计算 group-relative advantage。
- 对 semantic CoT tokens 和 image tokens 同时计算 GRPO loss。
- 记录 reward、reward std、KL、loss、completion length 和每个 reward 的均值。

## Reward 相关文件

- `src/t2i-r1/src/utils/reward_hps.py`
  - 类：`HPSv2`。
  - 作用：图像美学 / 人类偏好 reward。
  - 使用 HPS v2 checkpoint，返回 prompt-image preference score。

- `src/t2i-r1/src/utils/reward_gdino_enhanced.py`
  - 类：`GDinoEnhanced`。
  - 作用：对象 grounding、空间关系、计数 reward。
  - 对 `spatial` 任务：检查对象存在性和 soft relative position。
  - 对 `numeracy` 任务：使用 NMS 和 soft count penalty。
  - 对其他任务：通过 `get_object_score` 做对象存在性评分。
  - 不直接验证 color / shape / texture 属性绑定。

- `src/t2i-r1/src/utils/reward_vlm.py`
  - 类：`VLMAttr`、`VLMOrm`。
  - 共享基类：`_Qwen3VLBase`。
  - 使用 4-bit 量化加载 Qwen3-VL-2B，并在两个 VLM reward 间复用同一个模型实例。
  - `VLMAttr`：根据 `attr_nouns` 做属性-对象 VQA 判别。
  - `VLMOrm`：构造整体语义对齐问题，并解析 0-10 整数评分。

- `src/t2i-r1/src/utils/reward_gdino.py`
  - 原始 / legacy GDino reward。

- `src/t2i-r1/src/utils/reward_git.py`
  - 原始 / legacy GIT-style reward。

- `src/t2i-r1/src/utils/reward_orm.py`
  - legacy LLaVA yes/no outcome reward。
  - 当前正式训练脚本未启用。

## 生成与评测

- `src/t2i-r1/src/generate_all_eval.py`
  - 加载模型 checkpoint。
  - 为 T2I-CompBench 各类别批量生成图像。
  - 使用与训练相同的“reasoning prompt -> textual CoT -> image token generation”模式。
  - 默认类别：color、shape、texture、spatial、non_spatial、complex。
  - 默认每个 prompt 生成 `10` 张图像。

- `src/t2i-r1/src/run_eval.sh`
  - 运行 T2I-CompBench 评测，可选择 baseline、finetuned 或 both。
  - 评测器：
    - color / shape / texture：BLIP-VQA。
    - spatial：UniDet 2D spatial evaluator。
    - non_spatial：CLIPScore。
    - complex：BLIP-VQA + UniDet + CLIPScore + 3-in-1 aggregation。
  - 脚本结尾会读取评测 JSON 并打印结果表。

## Inference

- `src/t2i-r1/src/infer/reason_inference.py`
  - prompt-based generation / reasoning 推理脚本。
- `src/t2i-r1/src/infer/prompt.txt`
  - 推理 prompt 文本。
- `src/t2i-r1/src/infer/test_data.txt`
  - 推理测试 prompt。

## 配置文件

- `src/t2i-r1/configs/zero2.json`
  - DeepSpeed ZeRO-2 配置。
  - 使用 bf16。
  - `train_micro_batch_size_per_gpu`: `1`。
  - `gradient_clipping`: `1.0`。
  - `checkpoint.save_optimizer_states`: `false`。

## 输出与日志

训练脚本中的路径主要对应远程训练环境：

- `run_train.sh` 主输出目录：`./outputs/train_main`。
- reward log：`./outputs/train_main/reward_log.txt`。
- 本地 snapshot 中 TensorBoard runs 位于 `src/t2i-r1/src/outputs/train_main/runs/`。
- checkpoint 预期位于 `outputs/train_main/checkpoint-*`。

## 结果文件

本地评测结果位于：

- `eval_results/baseline/`
- `eval_results/finetuned/`

Phase 1 主结果使用的类别级结果文件：

- `color/annotation_blip/vqa_result.json`
- `shape/annotation_blip/vqa_result.json`
- `texture/annotation_blip/vqa_result.json`
- `spatial/labels/annotation_obj_detection_2d/vqa_result.json`
- `non_spatial/annotation_clip/vqa_result.json`
- `complex/annotation_3_in_1/vqa_result.json`

