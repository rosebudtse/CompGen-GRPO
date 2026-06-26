# CompGen-GRPO: 

**Multi-Reward Extension of T2I-R1 for Compositional Text-to-Image Generation**

A reinforcement learning project for compositional text-to-image generation, built on [T2I-R1](https://github.com/CaraJ7/T2I-R1) with an enhanced multi-reward system. We adapt **Janus-Pro-1B** to the Bi-CoT-GRPO training pipeline and reach **96.1% of the reported T2I-R1 Janus-Pro-7B average score on T2I-CompBench** using a 1B model.

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
- ✅ **96.1%** of the reported T2I-R1 Janus-Pro-7B T2I-CompBench average score
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
            └── Multi-Reward Suite
                    ├── HPS v2.1       (aesthetic quality)
                    ├── GDinoEnhanced  (object grounding / spatial / numeracy)
                    ├── VLMAttr        (attribute-object binding)
                    └── VLMOrm         (holistic semantic alignment)
```

---

## 🗂️ Repository Structure

```
CompGen-GRPO/
├── archive/                              # legacy code and local artifacts
│   ├── legacy_dependencies/LLaVA-NeXT/    # legacy LLaVA ORM dependency
│   ├── legacy_rewards/                    # original GDino / GIT / ORM rewards
│   ├── legacy_scripts/                    # old launch/debug scripts
│   ├── run_artifacts/                     # local training logs / TensorBoard events
│   └── third_party_extras/                # unused third-party demos/assets
├── data/
│   ├── geneval_and_t2i_data_final.json   # training data
│   └── prompt/reasoning_prompt.txt
├── docs/                                  # project notes, results, claim audit
├── eval_results/                          # T2I-CompBench scores (json)
│   ├── baseline/
│   └── finetuned/
├── results/                               # summarized result tables
├── src/
│   ├── requirements.txt
│   └── t2i-r1/src/
│       ├── run_train.sh                  # main GRPO training launcher
│       ├── run_eval.sh                   # T2I-CompBench evaluation launcher
│       ├── generate_all_eval.py          # batch generation for all categories
│       ├── open_r1/
│       │   ├── grpo.py                   # GRPO main logic (modified)
│       │   └── trainer/grpo_trainer.py   # Trainer (modified)
│       ├── utils/
│       │   ├── reward_hps.py             # HPS aesthetic reward
│       │   ├── reward_gdino_enhanced.py  # Enhanced GroundingDINO reward ⭐
│       │   ├── reward_vlm.py             # Qwen3-VL semantic reward ⭐
│       │   └── GroundingDINO/            # vendored GDino runtime dependency
│       ├── janus/                        # vendored Janus runtime dependency
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
bash src/t2i-r1/src/run_eval.sh --model both --task all

# Evaluate finetuned model on spatial only
bash src/t2i-r1/src/run_eval.sh --model finetuned --task spatial
```

The script prints a formatted results table at the end comparing baseline vs. finetuned across all categories.

---

## 📊 Reward System Details

| Reward | Model | Measures |
|---|---|---|
| HPS | HPS v2.1 | Aesthetic quality / human preference |
| GDinoEnhanced | GroundingDINO SwinT-OGC | Object existence, soft spatial relation, soft numeracy |
| VLMAttr | Qwen3-VL-2B-Instruct | Fine-grained attribute-object binding |
| VLMOrm | Qwen3-VL-2B-Instruct | Holistic prompt-image semantic alignment |

Legacy GIT, original GDino, and LLaVA ORM rewards are kept under `archive/legacy_rewards/` and `archive/legacy_dependencies/` for reference or future ablation, but they are not enabled by the main training script.

---

## 🌿 Advanced Branch Workflow

This branch is intended for further exploration of CompGen-GRPO, including reward ablation, reward reliability analysis, and structured constraint-level rewards.

```bash
# Check local changes
git status

# Stage and commit the cleaned repository structure
git add .
git commit -m "Clean up project structure for advanced exploration"

# Push this new branch to GitHub
git push -u origin advanced
```

Pushing `advanced` creates or updates the remote `advanced` branch. It does not change `main` unless this branch is later merged into `main`.

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
