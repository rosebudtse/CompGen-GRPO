# CompGen-GRPO Paper TODO

## Current Positioning

This project should be positioned as an extension of T2I-R1 rather than a fully independent GRPO framework.

Recommended framing:

> Building on T2I-R1 / Bi-CoT-GRPO, this work studies whether a compact autoregressive T2I model can achieve strong compositional generation through multi-dimensional reward redesign.

Current publishability:

- arXiv technical report: ready after paper writing and careful contribution framing.
- Workshop paper: plausible after reward ablation and reliability analysis.
- Journal / stronger conference: needs stronger method novelty, broader evaluation, and more systematic evidence.

Avoid claiming:

- We propose Bi-CoT-GRPO.
- We introduce a new T2I RL framework from scratch.
- GDinoEnhanced directly solves color / shape / texture attribute binding.
- We outperform T2I-R1 7B.

Safe claims:

- We adapt T2I-R1's Bi-CoT-GRPO to Janus-Pro-1B.
- We redesign the reward suite for compositional alignment.
- VLMAttr handles fine-grained attribute binding.
- GDinoEnhanced provides object existence, soft spatial, and soft numeracy rewards.
- The 1B model reaches 96.1% of the reported T2I-R1 7B average score on T2I-CompBench.

## Priority 0: Clean Project Narrative

- [ ] Finalize paper title.
  - Candidate: Multi-Reward GRPO for Parameter-Efficient Compositional Text-to-Image Generation
  - Stronger future version: Structure-Aware Adaptive Reward Optimization for Compositional Text-to-Image Generation

- [ ] Write a one-paragraph project thesis.
  - Problem: compositional generation failure in T2I.
  - Base framework: T2I-R1 / Bi-CoT-GRPO.
  - Contribution: compact-model adaptation plus multi-dimensional reward redesign.
  - Result: Janus-Pro-1B improves from 0.2320 to 0.4913 average on T2I-CompBench.

- [ ] Update README and paper draft to clearly acknowledge T2I-R1.
  - Mention that Bi-CoT-GRPO is inherited from T2I-R1.
  - State that this project focuses on reward redesign and compact-model empirical validation.

- [ ] Make terminology consistent.
  - Use "T2I-CompBench" consistently.
  - Use "Janus-Pro-1B" consistently.
  - Use "GDinoEnhanced" only for object grounding / spatial / numeracy rewards.
  - Use "VLMAttr" for color / shape / texture attribute binding.
  - Use "VLMOrm" for holistic semantic alignment.

## Priority 1: arXiv v1

Goal: produce a credible technical report before pursuing deeper extensions.

- [ ] Create a LaTeX paper skeleton.
  - Abstract
  - Introduction
  - Related Work
  - Method
  - Experiments
  - Analysis
  - Limitations
  - Conclusion

- [ ] Draft contributions.
  - Contribution 1: adapt T2I-R1 Bi-CoT-GRPO to Janus-Pro-1B under limited compute.
  - Contribution 2: introduce a multi-dimensional reward suite for compositional alignment.
  - Contribution 3: improve detection-based rewards with soft spatial scoring and soft numeracy penalties.
  - Contribution 4: show strong T2I-CompBench improvement, reaching 96.1% of T2I-R1 7B reported performance.

- [ ] Add a method figure.
  - Prompt -> reasoning CoT -> image tokens -> generated image.
  - Rewards: HPS, GDinoEnhanced, VLMAttr, VLMOrm.
  - GRPO group-relative advantage update.

- [ ] Add a reward decomposition table.
  - HPS: aesthetic / human preference.
  - GDinoEnhanced: object existence, spatial relation, numeracy.
  - VLMAttr: attribute-object binding.
  - VLMOrm: holistic semantic alignment.

- [ ] Add a dataset table.
  - spatial: 2,195
  - numeracy: 1,172
  - color: 879
  - complex: 700
  - non / non-spatial: 700
  - texture: 695
  - shape: 682
  - object: 200
  - total: 7,223

- [ ] Add the main result table.
  - Baseline Janus-Pro-1B average: 0.2320
  - Ours average: 0.4913
  - T2I-R1 7B reported average: 0.5114
  - Relative improvement over baseline: 111.8%
  - Ours / T2I-R1 7B: 96.1%

- [ ] Add limitations.
  - Built on T2I-R1, not a new RL algorithm.
  - Automatic benchmarks may have evaluator bias.
  - Non-spatial action relations remain weak.
  - Reward models may misjudge images or be exploitable.
  - Full reward ablation is still needed for stronger causal claims.

## Priority 2: Reward Ablation

This is the most important experiment for upgrading the work from "good project" to "research paper".

- [ ] Run or approximate reward ablation experiments.
  - Baseline Janus-Pro-1B.
  - HPS only.
  - HPS + GDinoEnhanced.
  - HPS + GDinoEnhanced + VLMAttr.
  - HPS + GDinoEnhanced + VLMAttr + VLMOrm.

- [ ] If full training is too expensive, run a smaller ablation.
  - Train for fewer steps.
  - Use a representative subset of training prompts.
  - Evaluate on a subset of T2I-CompBench.
  - Clearly label it as "subset ablation".

- [ ] Compare against original T2I-R1-style rewards if feasible.
  - HPS + original GDino + GIT + ORM.
  - HPS + GDinoEnhanced + VLMAttr + VLMOrm.

- [ ] Report category-level changes.
  - VLMAttr should mainly affect color / shape / texture.
  - GDinoEnhanced should mainly affect spatial / numeracy-related behavior.
  - VLMOrm should help complex / non-spatial holistic alignment.

## Priority 3: Reward Reliability Analysis

Goal: prove that the reward models measure what they are supposed to measure.

- [ ] Compute reward-score correlation with T2I-CompBench categories.
  - GDinoEnhanced vs spatial score.
  - VLMAttr vs color / shape / texture scores.
  - VLMOrm vs complex / non-spatial scores.
  - HPS vs image quality / preference if available.

- [ ] Manually inspect reward failures.
  - VLMAttr false positives / false negatives.
  - GDinoEnhanced missing small objects.
  - Wrong bbox relation due to occlusion or poor detection.
  - VLMOrm giving high score to visually plausible but semantically wrong images.

- [ ] Add qualitative examples.
  - Success case: attribute binding fixed.
  - Success case: spatial relation fixed.
  - Failure case: action / interaction relation still wrong.
  - Failure case: evaluator reward disagrees with human judgment.

- [ ] Consider a small human evaluation.
  - 50-100 prompts.
  - Pairwise comparison: baseline vs finetuned.
  - Criteria: prompt alignment, attribute binding, spatial correctness, overall quality.
  - Report win rate.

## Priority 4: Adaptive Reward Weighting

This can become a stronger method contribution if results show improvement over fixed reward summation.

- [ ] Define task-adaptive reward weights.
  - spatial: high GDinoEnhanced.
  - numeracy: high GDinoEnhanced count component.
  - color / shape / texture: high VLMAttr.
  - complex: balanced GDinoEnhanced + VLMAttr + VLMOrm.
  - non-spatial: higher VLMOrm, possibly relation-specific VLM prompts.

- [ ] Implement reward weighting.
  - Add reward weights to training args.
  - Add task-type based reward aggregation.
  - Log per-reward and weighted total reward.

- [ ] Compare fixed vs adaptive weighting.
  - Fixed sum reward.
  - Task-adaptive reward.
  - Optional curriculum: start with HPS / VLMOrm, later increase attribute / spatial rewards.

- [ ] If results improve, rename method around this idea.
  - Candidate: Task-Adaptive Multi-Reward GRPO.

## Priority 5: Structured Attribute-Relation Reward

This is the best path toward a stronger independent contribution.

- [ ] Convert prompts into structured constraints.
  - Objects.
  - Attributes.
  - Relations.
  - Counts.

- [ ] Define constraint-level rewards.
  - Object existence reward.
  - Attribute-object binding reward.
  - Relation reward.
  - Count reward.
  - Holistic semantic reward.

- [ ] Implement a constraint parser.
  - Start with dataset-provided fields: nouns, attr_nouns, spatial_info, numeracy_info.
  - Later extend to LLM-based prompt parsing for unseen prompts.

- [ ] Aggregate rewards at the constraint level.
  - Average over constraints.
  - Weight by task type.
  - Track per-constraint satisfaction rate.

- [ ] Add analysis.
  - Which constraints fail most often?
  - Does RL improve all constraints or only easy ones?
  - Are there trade-offs between visual quality and constraint satisfaction?

## Priority 6: Generalization Tests

Goal: prove that the method is not merely overfit to T2I-CompBench evaluators.

- [ ] Build an unseen compositional prompt set.
  - Attribute binding prompts.
  - Spatial relation prompts.
  - Numeracy prompts.
  - Complex multi-constraint prompts.
  - Non-spatial action prompts.

- [ ] Evaluate on another benchmark or subset.
  - DrawBench compositional subset.
  - PartiPrompts compositional subset.
  - Self-built English prompt set.
  - Optional Chinese prompt set.

- [ ] Compare baseline vs finetuned qualitatively and quantitatively.
  - Automatic scores where possible.
  - Human preference where automatic metrics are weak.

- [ ] Report failure modes on out-of-distribution prompts.

## Priority 7: Paper / Venue Strategy

- [ ] arXiv v1.
  - Scope: technical report.
  - Needs: clean writing, clear attribution to T2I-R1, main results, limitations.

- [ ] arXiv v2.
  - Add reward ablation.
  - Add reliability analysis.
  - Add qualitative examples.
  - Add generalization tests if ready.

- [ ] Workshop submission.
  - Best target after Priority 2 and 3 are completed.
  - Position as empirical study plus reward modeling method for compositional T2I.

- [ ] Journal / stronger venue.
  - Only consider after Priority 4, 5, and 6 have meaningful results.
  - Need stronger method identity: adaptive or structured reward optimization.

## Recommended Timeline

### Before arXiv v1

- [ ] Clean README and project notes.
- [ ] Write paper skeleton.
- [ ] Create method figure.
- [ ] Add main result table.
- [ ] Add limitations and ethical attribution.

### Before workshop-quality version

- [ ] Complete reward ablation.
- [ ] Complete reward reliability analysis.
- [ ] Add qualitative success / failure cases.
- [ ] Add small human evaluation if possible.

### Before journal-oriented version

- [ ] Implement adaptive reward weighting.
- [ ] Implement structured attribute-relation reward.
- [ ] Add cross-benchmark generalization.
- [ ] Add stronger baselines and human evaluation.

## Key Risks

- The work may be judged as a T2I-R1 engineering extension unless reward redesign is evaluated systematically.
- Automatic benchmark gains may be questioned without human evaluation or generalization tests.
- VLM rewards may introduce evaluator bias or reward hacking.
- Claims around GDinoEnhanced must remain precise: it does not directly solve color / shape / texture binding.
- The strongest path to novelty is not "more rewards", but "structured and adaptive reward modeling for compositional constraints".

## Bottom Line

The current project is strong enough for a credible arXiv technical report and useful for job applications. To make it paper-level, prioritize reward ablation and reward reliability analysis. To make it stronger than an extension of T2I-R1, develop adaptive reward weighting and structured constraint-level rewards.
