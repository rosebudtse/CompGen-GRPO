# Reward Weights 下载指引

本仓库 `.gitignore` 排除了 `src/t2i-r1/reward_weight/` 目录及所有 `*.pt / *.pth / *.bin / *.safetensors`，因此克隆下来的代码**不包含模型权重**。在新机器上跑训练 / 评估前，需要先把下面 5 个权重下载到固定路径。

## 0. 一键脚本（推荐）

仓库里已经提供 [download_weights.sh](./download_weights.sh)，会下载 4 个主要权重（Janus-Pro-1B / Qwen3-VL-2B-Instruct / HPS v2.1 / GroundingDINO）。BERT 因为 HF 上的仓库 id 是裸名 `bert-base-uncased` 与官方仓库格式不兼容，脚本里没含，需要按下面 §2 单独下。

```bash
cd <repo_root>
# 国内机器建议先指向镜像（hf-mirror.com 是 huggingface.co 的镜像）
export HF_ENDPOINT=https://hf-mirror.com

bash download_weights.sh
# 然后补 bert-base-uncased（见 §2）
```

走代理 / 内网时把 `HF_ENDPOINT` 改成对应地址即可，HuggingFace CLI 会复用。

---

## 1. 目标目录结构

所有权重统一落到仓库的 `src/t2i-r1/reward_weight/`，**目录名和大小写必须严格匹配**（训练脚本和 GroundingDINO config 里写死了路径）：

```
src/t2i-r1/reward_weight/
├── Janus-Pro-1B/                  # ~3.9G，基座 + 视觉解码器
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── processor_config.json
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── Qwen3-VL-2B-Instruct/          # ~4.0G，VLMAttr + VLMOrm 奖励
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── chat_template.json
│   ├── preprocessor_config.json
│   ├── video_preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── merges.txt
│   └── vocab.json
│
├── HPSv2.1/                       # ~1.9G，美学奖励
│   └── HPS_v2.1_compressed.pt
│
├── bert-base-uncased/             # ~421M，GroundingDINO 文本编码器
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
│
└── groundingdino_swint_ogc.pth    # ~662M，GroundingDINO 视觉权重（直接放在 reward_weight/ 根下）
```

总占用约 **11 GB**。

> **路径耦合点**：
> - 训练脚本 [src/t2i-r1/src/run_train.sh](./src/t2i-r1/src/run_train.sh) 里的 `REWARD_WEIGHT`、`HPS_CKPT`、`GDINO_CKPT`、`VLM_CKPT`、`MODEL_PATH`
> - GroundingDINO 配置 [src/t2i-r1/src/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py](./src/t2i-r1/src/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py) 里的 `text_encoder_type` 写的是绝对路径 `.../reward_weight/bert-base-uncased`。如果你换了仓库根路径或目录布局，**必须同步修改这两个文件**。

---

## 2. 各权重来源

### 2.1 Janus-Pro-1B（基座模型）

```bash
RW=src/t2i-r1/reward_weight
hf download deepseek-ai/Janus-Pro-1B --local-dir "$RW/Janus-Pro-1B"
```

来源：[deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)。

### 2.2 Qwen3-VL-2B-Instruct（VLMAttr + VLMOrm 奖励）

```bash
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir "$RW/Qwen3-VL-2B-Instruct"
```

来源：[Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。

### 2.3 HPS v2.1（美学奖励）

只需要单个文件 `HPS_v2.1_compressed.pt`：

```bash
mkdir -p "$RW/HPSv2.1"
hf download xswu/HPSv2 HPS_v2.1_compressed.pt --local-dir "$RW/HPSv2.1"
```

来源：[xswu/HPSv2](https://huggingface.co/xswu/HPSv2)。注意脚本默认下到 `reward_weight/` 根目录，需要再 `mv` 进 `HPSv2.1/`，或者像上面一样直接指定 local-dir。

### 2.4 GroundingDINO SwinT-OGC（spatial / numeracy 奖励）

视觉权重从 GitHub release 拿：

```bash
wget -c https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
     -O "$RW/groundingdino_swint_ogc.pth"
```

### 2.5 bert-base-uncased（GroundingDINO 文本编码器）

GroundingDINO config 默认会按 repo id 拉 `bert-base-uncased`，但 HF 把它放在裸名 repo 下，`hf download bert-base-uncased` 会报 "Repo id must be in the form 'repo_name' or 'namespace/repo_name'"。两种绕法二选一：

**方案 A：直接 wget 5 件套（最稳）**

```bash
mkdir -p "$RW/bert-base-uncased"
cd "$RW/bert-base-uncased"
for f in config.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors; do
  wget -c "https://huggingface.co/bert-base-uncased/resolve/main/$f"
done
cd -
```

**方案 B：用 git lfs**

```bash
git lfs install
git clone https://huggingface.co/bert-base-uncased "$RW/bert-base-uncased"
```

下载完毕后，确认 [GroundingDINO_SwinT_OGC.py](./src/t2i-r1/src/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py) 第 34 行 `text_encoder_type` 指向上面这个目录的绝对路径（**不是 repo id**）。如果你把仓库放在了别的位置，记得改。

---

## 3. 校验

下载完一次性检查，期望大小如下：

```bash
du -sh src/t2i-r1/reward_weight/*
```

| 路径 | 期望大小 |
| --- | --- |
| `bert-base-uncased/`           | ~421 M |
| `groundingdino_swint_ogc.pth`  | ~662 M |
| `HPSv2.1/`                     | ~1.9 G |
| `Janus-Pro-1B/`                | ~3.9 G |
| `Qwen3-VL-2B-Instruct/`        | ~4.0 G |

**最终烟测**：

```bash
bash check_train_env.sh   # 会检查权重路径 + 关键依赖
MAX_STEPS=5 bash src/t2i-r1/src/run_train.sh
```

如果 5 步训练能跑完不报 `FileNotFoundError` / `Repo id must be ...` / `OSError: ... does not appear to have a file named config.json`，就说明权重齐了。

---

## 4. 故障速查

| 报错 | 含义 / 修复 |
| --- | --- |
| `Repo id must be in the form 'repo_name' or 'namespace/repo_name'` | bert 目录缺失或路径写错，HF 把 `bert-base-uncased` 当成 repo id 去拉 → 见 §2.5 |
| `Failed to load custom C++ ops. Running on CPU mode Only!` | GroundingDINO `_C` 扩展未编译，与权重无关，参考 README / setup_env 重新 `pip install -e .` |
| `FileNotFoundError: HPS_v2.1_compressed.pt` | HPS 文件没放在 `reward_weight/HPSv2.1/` 目录下，注意子目录大小写 |
| 下载到 0 字节 / 经常断 | 切 `HF_ENDPOINT=https://hf-mirror.com`，或用 `wget -c` 续传 |
