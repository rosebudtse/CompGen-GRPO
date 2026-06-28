# 复现训练 — Worker 节点操作手册

> 本文件由环境检查生成，用于在 **worker 节点（有 GPU）** 从零复现 CompGen-GRPO 训练。
> master 节点无 GPU，以下命令全部在 **worker 终端**执行。仓库在共享存储，路径一致。

## 0. 前置事实

- worker：2× NVIDIA H20（97GB/卡），byted-torch 2.7.1（cu126，GPU 可用）
- 方案：venv 继承系统包，不动 torch；flash_attn 跳过（脚本用 eager attn）
- janus 是 vendored 本地包，无需安装

## 1. 安装环境（约 10-20 分钟）

```bash
cd /mlx_devbox/users/xiezifan/playground/CompGen-GRPO
bash setup_env.sh
```

完成后所有依赖装在 `.venv/`。若末尾提示 **groundingdino/_C 不可用**，多半是没有 nvcc：

```bash
export CUDA_HOME=/usr/local/cuda          # 指向 cu126 toolkit
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version                            # 确认能看到 12.x
source .venv/bin/activate
pip install -e src/t2i-r1/src/utils/GroundingDINO --no-build-isolation
```

## 2. 下载权重（约几十分钟，取决于网速）

```bash
source .venv/bin/activate
bash download_weights.sh
```

下载到 `src/t2i-r1/reward_weight/`：Janus-Pro-1B、Qwen3-VL-2B-Instruct、HPS_v2.1_compressed.pt、groundingdino_swint_ogc.pth。
连接 HF 慢可先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 3. 复检（确认全绿）

```bash
source .venv/bin/activate
bash check_train_env.sh
```

所有 `[FAIL]` 应消失。`groundingdino` / `janus` 那两项如仍报，分别按第 1 步补编译、或确认会在 `src/t2i-r1/src/` 工作目录下运行（训练脚本已自动 cd，不影响）。

## 4. Smoke test（关键！先跑 5 步验证全链路）

正式跑 2000 步前，务必先小步验证。`run_train.sh` 支持环境变量覆盖：

```bash
source .venv/bin/activate
cd /mlx_devbox/users/xiezifan/playground/CompGen-GRPO
MAX_STEPS=5 CUDA_VISIBLE_DEVICES=0 bash src/t2i-r1/src/run_train.sh
```

**重点观察**（这是已知风险点）：
- transformers / trl / Qwen3VL 三者版本是否冲突（reward_vlm.py 需要 `Qwen3VLForConditionalGeneration`，要求 transformers ≥ 4.57）
- 4 个 reward 模型能否成功加载（HPS、GDino、Qwen3-VL 4-bit）
- 单步 loss / reward 是否打印、是否有 NaN
- 显存是否够（1B 训练 + 3 个 reward 模型，单卡 H20 97G 应充足）

5 步能跑完且 reward 正常，再进入正式训练。

## 5. 正式训练

```bash
source .venv/bin/activate
cd /mlx_devbox/users/xiezifan/playground/CompGen-GRPO
nohup bash src/t2i-r1/src/run_train.sh > train_main.log 2>&1 &
tail -f train_main.log
```

- 默认 2000 步，checkpoint 存到 `src/t2i-r1/src/outputs/train_main/`
- TensorBoard：`tensorboard --logdir src/t2i-r1/src/outputs/train_main/runs`
- 想用双卡：`NPROC=2 CUDA_VISIBLE_DEVICES=0,1 bash src/t2i-r1/src/run_train.sh`（需确认 deepspeed 多卡配置）

## 已知风险 / 排错

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `Qwen3VLForConditionalGeneration` import 失败 | transformers < 4.57 | `pip install -U "transformers>=4.57.0"` |
| trl 与 transformers 版本冲突 | trl 0.16.0 对新版 transformers API 不兼容 | 看报错调 trl 版本，或小改 grpo_trainer 适配 |
| groundingdino `_C` 缺失 | 没装 nvcc / 没编译 | 见第 1 步补编译 |
| bitsandbytes GPU 报错 | bnb 与 cuda 版本不匹配 | 装匹配 cu12x 的 bitsandbytes |
| OOM | batch / 生成数过大 | 已是最小配置；检查是否 reward 模型未量化 |
