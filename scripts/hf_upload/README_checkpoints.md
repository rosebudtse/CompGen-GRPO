---
license: mit
base_model: deepseek-ai/Janus-Pro-1B
tags:
  - text-to-image
  - compositional-generation
  - grpo
  - reinforcement-learning
  - ablation
library_name: transformers
---

# CompGen-GRPO Checkpoints

Fine-tuned checkpoints of **Janus-Pro-1B** with **GRPO (Group Relative Policy Optimization)** for compositional text-to-image generation, following the **T2I-R1** recipe.

This repo hosts **ablation-study checkpoints** used in our paper. Each ablation removes one reward component from the full reward set:
`{HPSv2.1, GroundingDINO, Qwen3-VL-2B (VLMAttr), Qwen3-VL-2B (VLMOrm)}`.

## Repo layout

```
CompGen-GRPO-checkpoints/
├── ablation_wo_attr/          # 去掉 VLM attribute reward
│   ├── checkpoint-400/
│   ├── checkpoint-600/
│   └── checkpoint-800/
└── ablation_wo_gdino/         # 去掉 GroundingDINO reward
    └── checkpoint-600/
```

Each `checkpoint-*` directory is a full HF Transformers `save_pretrained()` export (inference-ready weights only; optimizer states removed to save disk).

## Reward composition per ablation

| Ablation | HPSv2.1 | GroundingDINO | VLM Attr | VLM Orm |
|---|---|---|---|---|
| `full` (baseline of ablation) | ✓ | ✓ | ✓ | ✓ |
| `wo_attr` | ✓ | ✓ | – | ✓ |
| `wo_gdino` | ✓ | – | ✓ | ✓ |

Full & other ablation runs will be released after paper acceptance.

## Training setup

- **Base model**: [`deepseek-ai/Janus-Pro-1B`](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
- **Framework**: TRL 1.4 + DeepSpeed ZeRO-2
- **Precision**: bf16
- **Hardware**: 4 × A100-80GB
- **Prompt source**: T2I-CompBench train set (color/shape/texture/spatial/non_spatial/complex)
- **KL coefficient**: default TRL
- **Batch**: 8 rollouts / prompt, group size 8

Full training script: [github.com/xiezifan/CompGen-GRPO](https://github.com/xiezifan/CompGen-GRPO) → `src/t2i-r1/src/run_train.sh`.

## Usage (inference)

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import snapshot_download

# Pull one checkpoint
path = snapshot_download(
    repo_id="<HF_USER>/CompGen-GRPO-checkpoints",
    allow_patterns=["ablation_wo_attr/checkpoint-800/*"],
    local_dir="./local_ckpt",
)

model = AutoModelForCausalLM.from_pretrained(
    "./local_ckpt/ablation_wo_attr/checkpoint-800",
    torch_dtype="bfloat16",
    trust_remote_code=True,
).cuda()
```

For the full generation pipeline (Bi-CoT prompting, CFG sampling, T2I-CompBench eval), see `src/t2i-r1/src/generate_all_eval.py` in the code repo.

## Evaluation

We evaluate on **T2I-CompBench** (300 prompts × 6 categories, 10 imgs / prompt = 3000 imgs / category).
Pre-generated samples for downstream analysis are hosted in the companion dataset repo: **[`<HF_USER>/CompGen-GRPO-eval-samples`](https://huggingface.co/datasets/<HF_USER>/CompGen-GRPO-eval-samples)**.

Metrics: BLIP-VQA (color/shape/texture), UniDet 2D (spatial), CLIPScore (non_spatial), 3-in-1 aggregation (complex).

## License

MIT (fine-tuned weights follow the base model's license — see [Janus-Pro-1B license](https://huggingface.co/deepseek-ai/Janus-Pro-1B)).

## Citation

```bibtex
@article{compgen-grpo-2026,
  title  = {Compositional Text-to-Image Generation via Multi-Reward GRPO},
  author = {Xie, Zifan and others},
  year   = {2026},
  note   = {In preparation}
}
```
