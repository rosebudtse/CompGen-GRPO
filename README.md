# CompGen-GRPO: 

**Multi-Reward GRPO Framework for Compositional Text-to-Image Generation**

A reinforcement learning framework for compositional text-to-image generation, built on [T2I-R1](https://github.com/CaraJ7/T2I-R1) with an enhanced four-dimensional reward system. We train **Janus-Pro-1B** via Bi-CoT-GRPO and achieve **96.1% of T2I-R1's 7B performance using only 1B parameters**.

---

## 🏆 Results on T2I-CompBench

| Category | Baseline (1B) | **Ours (1B)** | T2I-R1 (7B) |
|---|---|---|---|
| Color | 0.3414 | **0.7834** | 0.8130 |
| Shape | 0.2039 | **0.5090** | 0.5852 |
| Texture | 0.2774 | **0.6756** | 0.7243 |
| Spatial | 0.0735 | **0.2976** | 0.3378 |
| Non-spatial | 0.2621 | **0.3044** | 0.3090 |
| Complex | 0.2335 | **0.3776** | 0.3993 |
| **Average** | 0.2320 | **0.4913** | 0.5114 |

- ✅ **+111.8%** average improvement over the 1B baseline
- ✅ **96.1%** of T2I-R1 7B performance with only 1/7 the parameters
- ✅ Non-spatial score (0.3044) nearly matches T2I-R1 7B (0.3090)

---

## 📌 Overview

This project reproduces and extends T2I-R1 with the following contributions:

- **Four-dimensional reward system**: HPS aesthetic reward (`reward_hps.py`) + GroundingDINO spatial reward (`reward_gdino_enhanced.py`) + VLM semantic reward (VLMAttr and VLMOrm) via Qwen3-VL-2B (`reward_vlm.py`)
- **Enhanced GroundingDINO reward**: improved object detection scoring for spatial compositional alignment
- **VLM-based semantic reward**: Qwen3-VL-2B as an outcome reward model for fine-grained semantic verification
- **Full T2I-CompBench evaluation pipeline**: automated scripts for all 6 categories across baseline and finetuned models

### Framework

```
Janus-Pro-1B
    └── Bi-CoT-GRPO Training
            ├── Thinking CoT (compositional reasoning)
            ├── Drawing CoT (token-level generation)
            └── Four-Dimensional Reward
                    ├── HPS v2.1       (aesthetic quality)
                    ├── GroundingDINO  (spatial accuracy)
                    ├── Qwen3-VL-2B   (semantic alignment)
                    └── ORM            (format correctness)
```

---

## 🗂️ Repository Structure

```
CompGen-GRPO/
├── data/
│   ├── geneval_and_t2i_data_final.json   # training data
│   └── prompt/reasoning_prompt.txt
├── eval_results/                          # T2I-CompBench scores (json)
│   ├── baseline/
│   └── finetuned/
├── src/
│   ├── scripts/
│   │   ├── run_grpo.sh                   # GRPO training launcher
│   │   └── run_eval.sh                   # T2I-CompBench evaluation launcher
│   └── t2i-r1/src/
│       ├── open_r1/
│       │   ├── grpo.py                   # GRPO main logic (modified)
│       │   └── trainer/grpo_trainer.py   # Trainer (modified)
│       ├── utils/
│       │   ├── reward_hps.py             # HPS aesthetic reward
│       │   ├── reward_gdino_enhanced.py  # Enhanced GroundingDINO reward ⭐
│       │   ├── reward_vlm.py             # Qwen3-VL semantic reward ⭐
│       │   ├── reward_gdino.py           # Original GDino reward
│       │   └── reward_orm.py             # Format reward
│       ├── generate_for_eval.py          # Single-model image generation
│       ├── generate_all_eval.py          # Batch generation for all categories
│       └── infer/reason_inference.py     # Inference script
├── figs/                                 # Architecture figures
└── README.md
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
pip install -r src/requirements.txt
cd src/t2i-r1/src/utils/GroundingDINO && pip install -e .
```

### 2. Download Model Weights

```bash
# Janus-Pro-1B (base model)
huggingface-cli download deepseek-ai/Janus-Pro-1B --local-dir src/t2i-r1/reward_weight/Janus-Pro-1B

# HPS v2.1 checkpoint
# Place HPS_v2.1_compressed.pt in src/t2i-r1/reward_weight/

# GroundingDINO weights
# Place groundingdino_swint_ogc.pth in src/t2i-r1/reward_weight/
```

### 3. Training

```bash
bash src/t2i-r1/src/run_train.sh
```

Key training arguments (in `run_train.sh`):

| Argument | Value |
|---|---|
| Base model | Janus-Pro-1B |
| Training steps | 2000 |
| DeepSpeed | ZeRO-2 |
| Reward weights | HPS + GDino + VLMAttr + VLMOrm |

### 4. Generate Images for Evaluation

```bash
# Generate for all categories
python src/t2i-r1/src/generate_all_eval.py
```

### 5. Run T2I-CompBench Evaluation

```bash
# Evaluate all models and all tasks
bash src/scripts/run_eval.sh --model both --task all

# Evaluate finetuned model on spatial only
bash src/scripts/run_eval.sh --model finetuned --task spatial
```

The script prints a formatted results table at the end comparing baseline vs. finetuned across all categories.

---

## 📊 Reward System Details

| Reward | Model | Measures |
|---|---|---|
| HPS | HPS v2.1 | Aesthetic quality / human preference |
| GroundingDINO (enhanced) | SwinT-OGC | Object detection & spatial relation accuracy |
| VLM | Qwen3-VL-2B-Instruct | Fine-grained semantic alignment |
| ORM | Rule-based | Output format correctness |

---

## 🙏 Acknowledgements

- [T2I-R1](https://github.com/CaraJ7/T2I-R1) — original Bi-CoT-GRPO framework
- [Janus-Pro](https://github.com/deepseek-ai/Janus) — base multimodal model
- [T2I-CompBench](https://karine-h.github.io/T2I-CompBench/) — evaluation benchmark
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — spatial reward backbone
- [HPS v2](https://github.com/tgxs002/HPSv2) — aesthetic reward

---

## 📄 License

This project follows the license of the original [T2I-R1](https://github.com/CaraJ7/T2I-R1) repository.