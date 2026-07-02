---
license: mit
task_categories:
  - text-to-image
tags:
  - t2i-compbench
  - compositional-generation
  - grpo
  - ablation
size_categories:
  - 100K<n<1M
---

# CompGen-GRPO Eval Samples

Pre-generated images from a series of **CompGen-GRPO** checkpoints (fine-tuned Janus-Pro-1B with GRPO), evaluated on **T2I-CompBench** (color / shape / texture / spatial / non_spatial / complex).

Every image is a 384×384 PNG rendered with the same sampling config so results are directly comparable:

- **Prompt set**: T2I-CompBench eval split (300 prompts × 6 categories)
- **Samples per prompt**: 10
- **Resolution**: 384×384
- **CFG scale**: 5.0
- **Seed**: 42
- **Reasoning prompt**: Bi-CoT template (see companion code repo)

Total: **~162,000 images** across 9 models × 6 categories × 3000 imgs.

## Repo layout

```
CompGen-GRPO-eval-samples/
├── baseline/            # Janus-Pro-1B (no RL, deepseek-ai release)
│   ├── color/samples.tar
│   ├── shape/samples.tar
│   └── ...
├── full_400/
├── full_800/
├── wo_gdino_400/        # ablation: no GroundingDINO
├── wo_gdino_600/
├── wo_gdino_800/
├── wo_attr_400/         # ablation: no VLM Attr
├── wo_attr_600/
├── wo_attr_800/
└── <MODEL>/<CATEGORY>/samples.tar
```

Each `samples.tar` contains a `samples/<PROMPT_ID>_<SAMPLE_ID>.png` directory (uncompressed tar — PNG is already compressed).

Categories:

| Category | Prompts | Total imgs | Eval metric |
|---|---|---|---|
| `color`       | 300 | 3000 | BLIP-VQA |
| `shape`       | 300 | 3000 | BLIP-VQA |
| `texture`     | 300 | 3000 | BLIP-VQA |
| `spatial`     | 300 | 3000 | UniDet-2D |
| `non_spatial` | 300 | 3000 | CLIPScore |
| `complex`     | 300 | 3000 | 3-in-1 (BLIP+UniDet+CLIP) |

Naming convention: `<PROMPT_ID>_<SAMPLE_ID>.png` — `PROMPT_ID ∈ [0, 300)`, `SAMPLE_ID ∈ [0, 10)`.

## Note on `full_400`

For historical reasons `full_400/{color,shape,texture}/samples/` contains **6000** images (20 samples / prompt) instead of the standard 3000. This does not change the mean score meaningfully (n=3000 is already low-variance), but be aware if you rely on per-sample paired comparisons.

## How to download

```bash
pip install -U huggingface_hub

# Full dataset (all zips)
hf download <HF_USER>/CompGen-GRPO-eval-samples \
    --repo-type dataset \
    --local-dir ./eval_samples

# Just one model
hf download <HF_USER>/CompGen-GRPO-eval-samples \
    --repo-type dataset \
    --include "full_800/*" \
    --local-dir ./eval_samples

# Unzip all
find ./eval_samples -name "samples.tar" | while read t; do
    (cd "$(dirname "$t")" && tar -xf "$(basename "$t")" && rm "$(basename "$t")")
done
```

## How to reproduce eval

```bash
git clone https://github.com/xiezifan/CompGen-GRPO
cd CompGen-GRPO
# After unzip, samples land at <MODEL>/<CATEGORY>/samples/*.png — just move
# that tree under eval_results/ and run:
bash src/t2i-r1/src/run_eval.sh --model full_800,wo_gdino_600 --task all --gpu 0
```

See [`run_eval.sh`](https://github.com/xiezifan/CompGen-GRPO/blob/main/src/t2i-r1/src/run_eval.sh) for the full T2I-CompBench eval pipeline (BLIP-VQA + UniDet + CLIPScore + 3-in-1).

## License

Images are released under **CC-BY-4.0**. See paper for detailed attribution.

## Citation

```bibtex
@article{compgen-grpo-2026,
  title  = {Compositional Text-to-Image Generation via Multi-Reward GRPO},
  author = {Xie, Zifan and others},
  year   = {2026},
  note   = {In preparation}
}
```
