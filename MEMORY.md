# MEMORY.md — CompGen-GRPO 当前状态快照

> 📌 **本文件是会话间的接力棒**。每次会话结束前必须由 agent 更新（见 `AGENTS.md` §维护协议）。
> 新 agent 接力流程：先读 `AGENTS.md` → 再读本文件 → 再动手。

**Last updated**: 2026-06-28 23:18 UTC, by agent (sonnet) after 启动 baseline run 成功
**Branch**: `advanced`
**Owner**: xiezifan

---

## 1. 当前实验状态

### 正在跑的 run

| 项 | 值 |
|---|---|
| Run name | `train_main_2gpu_g4_2k` |
| WANDB project | `CompGen-GRPO` |
| WANDB group | `advanced` |
| 启动时间 | 2026-06-28 ~23:15 UTC |
| GPU | 2 × H20（CUDA_VISIBLE_DEVICES=0,1） |
| 总 step | 1600（约 16 小时，按 ~36s/step） |
| 进度（最后观察） | step 2/1600 |
| 健康指标 | loss=0.0015、grad_norm=2.97、kl=0.0015、reward_std=0.59 → ✅ 正常 |
| 显存占用 | 36 GB / 97 GB（每卡）→ 余量大 |
| GPU util | 40% / 63% → 还能压 |

### 关键超参（详细见 `AGENTS.md` §4）

- `num_generations=4`、`per_device_bs=1`、`grad_accum=2`、`beta=0.01`、`lr=1e-6`
- `attn_implementation=sdpa`（flash_attn 因 ABI 不匹配未启用）
- `report_to=wandb`（**只能单值**，trl HfArgumentParser 限制）

### 日志位置

- 内层（含 Python traceback）：[train_log.txt](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs/train_main/train_log.txt)
- 外层（torchrun warning + NCCL）：[train_main.log](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs/train_main/train_main.log)
- Reward 详情（DEBUG_MODE=true）：[reward_log.txt](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/src/t2i-r1/src/outputs/train_main/reward_log.txt)

---

## 2. 接力 next steps（按优先级）

### P0 — 必须做

1. **监控当前 run**：每 2-4 小时看一眼 wandb，确认：
   - `loss` 不爆 NaN/inf
   - `grad_norm` 没暴涨到 >100
   - `reward_std` 不持续 < 0.05（否则模型坍塌）
   - 单 step 时间不暴涨到 >60s（否则 reward 阶段卡了）
2. **跑完后立刻保存最终 ckpt 信息到本文件**：
   - 最终 reward 各分量（HPSv2 / GDinoEnhanced / VLMAttr / VLMOrm）
   - reward 曲线是否单调上升
   - 最终 wall time、ckpt 路径（`outputs/train_main/checkpoint-1600`）

### P1 — 当前 run 跑完后做

3. **第二个 run（变量：num_generations）**：
   - 拷一份 run_train.sh 改 `--num_generations 8`、`WANDB_NAME=train_main_2gpu_g8_2k`
   - 单 step 时间会翻倍到 ~72s，1600 step ~32h
   - 对比 G=4 vs G=8 的 reward 终值，验证 T2I-R1 论文用 G=8 的必要性
4. **flash_attn 源码编译（异步）**：
   - 命令：`pip install flash-attn==2.7.4.post1 --no-build-isolation`（在 worker 上）
   - 大约 30 min 编译，成功后改 `run_train.sh` L63 `sdpa → flash_attention_2`
   - 预期吞吐 +30-50%
5. **GeneEval 评估**：当前 run 的 ckpt-1600 跑一遍 GeneEval（256 prompts）
   - 评估脚本位置：TODO（需要确认/写）

### P2 — 论文层面

6. **TODO_cn.md 的 P0 项**：详见该文件，包括 baseline 表格、ablation 设计
7. **`docs/structcomp_grpo_todo.md`**：StructComp-GRPO 扩展研究的规划

---

## 3. 已解决的问题（本会话）

按时间顺序，仅近期：

- **GroundingDINO C++ 编译 torch 2.7 ABI 不匹配** → 改 `value.type()` 调用为 `value.is_cuda()` / `value.scalar_type()`，两个副本都同步
- **`from groundingdino import _C: libc10.so`** → `__init__.py` 加 `import torch`
- **`torchvision::nms` CUDA backend 缺失** → `reward_gdino_enhanced.py` L345 强制 `.cpu()`
- **`wandb login` 报 `.netrc` 权限** → 用 `WANDB_API_KEY` 环境变量
- **`--report_to "wandb,tensorboard"` 逗号串** → transformers 4.57 不再支持
- **`--report_to wandb tensorboard` 空格分隔** → trl 1.4 HfArgumentParser 把 `Union[None, str, list[str]]` 解析为 str，只接受单值 → 改为 `--report_to wandb`
- **同一启动命令贴了 4 次起 4 份 torchrun** → 一行启动，不用反斜杠换行
- **`R/train_log.txt: No such file or directory`** → 同上根因

所有这些都已同步到 [AGENTS.md §5 已知坑](file:///mlx_devbox/users/xiezifan/playground/CompGen-GRPO/AGENTS.md)。

---

## 4. 重要决策记录

- **`--beta=0.01`**（不是 0）：保留轻度 KL 正则；T2I-R1 默认 0（不加载 ref_model 省显存），但显存有富余，0.01 给一点 anchor 也无妨
- **`--num_generations=4`**（不是 8）：先用 4 跑 baseline，下个 run 再换 8 做 ablation
- **不启 vLLM**：Janus 的双 head（VQ + lm）vLLM 没原生支持，改造成本高
- **不写 PR / 设计文档**：直接在代码注释 + AGENTS.md / MEMORY.md 沉淀

---

## 5. 未解决 / 待跟进

- [ ] flash_attn 源码编译还没做（决定先让 baseline 跑完再编）
- [ ] GeneEval 评估脚本路径不确定，需要 grep 一下原 T2I-R1 仓库或自己写
- [ ] `REWARD_WEIGHTS.md` 在 git status 里是 `?`（未追踪），可以 commit 进 git
- [ ] 多个 tfevents 在 `outputs/train_main/runs/` 下，是之前测试遗留，可以清掉
- [ ] `--report_to` 想同时要 wandb + tensorboard 的话需要走 YAML config 路线，或不传该参数（默认 `"all"`）

---

## 6. 给下一个 agent 的提示

- **先看 `AGENTS.md`**，再回本文件
- **如果训练还在跑**：不要重启、不要改正在生效的代码；只能修 `run_train.sh`（下次启动才生效）
- **如果训练已完成**：`outputs/train_main/checkpoint-1600/` 应该存在；wandb 看终值；本文件 §1 状态要更新成"已完成"
- **如果训练崩了**：先看 `train_log.txt` 末尾 Python traceback，再决定是否重启
- **不要在 master 节点跑训练**（GPU 在 worker，见 AGENTS.md §6）
