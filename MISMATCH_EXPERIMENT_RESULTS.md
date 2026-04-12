# Concept Mismatch Experiment: Results


### The prompt

The model receives a framing preamble telling it that an interpretability researcher may inject "thoughts" into its neural activity. Then, we inform the model:

> Trial 1: I may have injected a thought about the word '{X}'. Did you detect an injected thought about {X}? Respond with only YES or NO.

### The three conditions

For each named concept X, layer position, and injection strength, we run a single forward pass and extract the YES minus NO logit difference at the first response token. We do this under three conditions:

| Condition | Prompt says | Actually injected | Prediction if introspection | Prediction if YES-bias |
|-----------|------------|-------------------|----------------------------|----------------------|
| **Congruent** | "about X" | X's steering vector | YES | YES |
| **Incongruent** | "about X" | Y's steering vector (Y distant from X) | NO | YES |
| **Baseline** | "about X" | nothing | NO | NO |

The key comparison is congruent vs. incongruent. If the model can tell which concept was injected, it should respond more YES when the injection matches the named concept. If it cannot — if it is merely detecting that *something* was injected — both conditions should produce similar YES-bias.

### Ensuring the injected concept is genuinely different

For each concept X, we select 5 partner concepts Y whose steering vectors are maximally dissimilar. We compute cosine similarity between all pairs of steering vectors at a reference layer and pick the 5 with the lowest (most negative) similarity. All selected partners have negative cosine similarities (typically -0.25 to -0.55), meaning they point in roughly opposite directions in activation space. This ensures the incongruent injection is as different as possible from what the model is being asked about.

### What we measure

The dependent variable is the YES minus NO logit difference at the model's first response token. A positive value means the model favors YES; a negative value means it favors NO. We do not generate text — we extract raw logits from a single forward pass, making the measurement deterministic and fast (~200x faster than generation-based experiments).

## Configuration

All three models use the same 50 concepts, the same 5 injection strengths, the same K=5 dissimilar partners per concept, and 7 layer positions normalized to equivalent percentage depths for cross-model comparison.

| Parameter | Qwen3-8B | Qwen3-14B | Qwen3-32B |
|-----------|----------|-----------|-----------|
| Total layers | 36 | 40 | 64 |
| Layers tested | 5, 10, 15, 20, 25, 30, 35 | 6, 11, 17, 22, 28, 33, 39 | 9, 18, 27, 36, 45, 54, 63 |
| Layer depths (%) | 14, 29, 43, 57, 71, 86, 100 | 15, 28, 44, 56, 72, 85, 100 | 14, 29, 43, 57, 71, 86, 100 |
| Strengths | 3.5, 4.0, 4.5, 5.0, 6.0 | 3.5, 4.0, 4.5, 5.0, 6.0 | 3.5, 4.0, 4.5, 5.0, 6.0 |
| Reference layer | 20 (57%) | 22 (56%) | 36 (57%) |
| Records per model | 12,250 | 12,250 | 12,250 |
| Total records | **36,750** | | |

Each model produces 1,750 congruent, 8,750 incongruent, and 1,750 baseline records.

## Results

### 1. The model cannot tell matching from non-matching injections

The overall congruent-incongruent gap across all three models:

| Model | Congruent | Incongruent | Baseline | Gap | Cong shift | Inc shift | Gap as % of shift |
|-------|----------:|------------:|---------:|----:|-----------:|----------:|------------------:|
| 8B | -9.48 | -9.66 | -17.21 | **+0.18** | +7.73 | +7.55 | 2.4% |
| 14B | -12.96 | -12.46 | -25.40 | **-0.50** | +12.44 | +12.93 | -4.0% |
| 32B | -1.42 | -1.57 | -2.31 | **+0.15** | +0.89 | +0.74 | 16.6% |

All three baselines are strongly negative (the model correctly says NO when nothing is injected). Both congruent and incongruent injections produce large shifts toward YES. The congruent-incongruent gap is negligible at all scales: +0.18 logits (8B), -0.50 logits (14B, actually *favoring* the wrong concept), and +0.15 logits (32B).

For 8B and 14B, the gap is less than 4% of the total congruent shift. For 32B, the gap appears larger in percentage terms (16.6%) only because the total shift is very small (0.89 logits) — the model barely responds to any injection at all, so the percentage amplifies noise in a near-zero denominator.

The 14B result is particularly striking: the gap is *negative*. The model responds more YES to mismatched injections than to matched ones. This is the opposite of what introspection predicts.

### 2. None of the results are statistically significant

We conducted a paired t-test for each model, pairing each concept's mean congruent logit diff against its mean incongruent logit diff (N=50 concepts per test):

| Model | Mean difference | SE | t(49) | p (two-tailed) | Cohen's d | 95% CI |
|-------|----------------:|---:|------:|----------------:|----------:|-------:|
| 8B | +0.184 | 0.434 | +0.42 | **0.673** | 0.060 | [-0.69, +1.06] |
| 14B | -0.498 | 0.408 | -1.22 | **0.229** | -0.172 | [-1.32, +0.32] |
| 32B | +0.148 | 0.115 | +1.28 | **0.207** | 0.181 | [-0.08, +0.38] |

No model reaches significance. All three confidence intervals include zero. The effect sizes are trivial (d = 0.06) to small (d = 0.18). With 12,250 observations per model and 50 paired concept-level means, these tests have ample statistical power — a meaningful effect would be detected.

### 3. The gap does not emerge at any layer depth

Layer positions are normalized to percentage depth (0-100%) so that equivalent positions can be compared across model architectures. The congruent-incongruent gap at each depth:

| Depth | 8B gap | 14B gap | 32B gap |
|------:|-------:|--------:|--------:|
| ~14% | +0.39 | +0.58 | +0.09 |
| ~29% | -0.00 | -1.14 | +0.08 |
| ~43% | -0.95 | -0.83 | +0.05 |
| ~57% | +0.86 | -1.49 | +0.02 |
| ~71% | +0.03 | -0.52 | +0.57 |
| ~86% | +0.87 | -0.01 | +0.23 |
| 100% | +0.09 | -0.08 | -0.01 |

The gap oscillates between small positive and small negative values with no consistent pattern. No layer position shows a reliable discrimination signal across all three models. At ~43% depth, 8B and 14B both show *negative* gaps (the model prefers the wrong concept). At ~57%, 8B is positive while 14B is the most negative of any condition. There is no "sweet spot" layer where concept discrimination reliably occurs.

### 4. The gap does not grow with injection strength

| Strength | 8B gap | 14B gap | 32B gap |
|---------:|-------:|--------:|--------:|
| 3.5x | +0.22 | -0.47 | +0.17 |
| 4.0x | +0.17 | -0.48 | +0.16 |
| 4.5x | +0.19 | -0.51 | +0.16 |
| 5.0x | +0.19 | -0.54 | +0.14 |
| 6.0x | +0.15 | -0.49 | +0.12 |

The gap is flat across all five strength levels for all three models. For 8B, it hovers around +0.18; for 14B, around -0.50; for 32B, around +0.15. Stronger injections increase the total YES-bias equally for congruent and incongruent conditions, preserving the same negligible gap. A genuine introspective mechanism should become more discriminative with stronger signals; instead, both conditions scale proportionally.

### 5. No concept reliably discriminates across scales

We computed per-concept discrimination scores (mean congruent minus mean incongruent) for all 50 concepts in each model and checked which concepts are consistently positive or negative across all three:

| Metric | Value |
|--------|-------|
| Concepts positive in all 3 models | **14 / 50** |
| Concepts negative in all 3 models | **8 / 50** |
| Concepts inconsistent (mixed signs) | **28 / 50** |
| Concepts positive: 8B / 14B / 32B | 28 / 25 / 30 |

The majority of concepts (28/50) change the sign of their discrimination across models — positive in one, negative in another. This is inconsistent with a stable introspective mechanism and consistent with noise.

**Cross-model correlation of per-concept discrimination scores:**

| Pair | Pearson r |
|------|----------:|
| 8B vs 14B | **+0.49** |
| 8B vs 32B | **+0.30** |
| 14B vs 32B | **+0.11** |

There is a moderate correlation between 8B and 14B (r = 0.49), meaning concepts that appear to discriminate in one tend to discriminate in the other to some degree. But this correlation weakens substantially at larger scale (r = 0.11 between 14B and 32B), and even the strongest correlation (0.49) explains only 24% of the variance. The per-concept discrimination pattern is largely model-specific, not a stable property of the concepts themselves.

**Concepts with positive discrimination in all three models:**

| Concept | 8B | 14B | 32B |
|---------|---:|----:|----:|
| algorithms | +7.25 | +3.50 | +1.37 |
| vegetables | +5.84 | +1.11 | +0.47 |
| illusions | +4.31 | +0.77 | +0.97 |
| dynasties | +3.84 | +1.08 | +2.23 |
| memories | +3.61 | +0.53 | +0.47 |
| sadness | +2.66 | +3.55 | +0.05 |
| treasures | +2.19 | +2.59 | +0.42 |
| harmonies | +1.02 | +3.18 | +1.17 |
| ... | ... | ... | ... |

Even among the consistently positive concepts, the discrimination score generally *decreases* with scale (e.g., algorithms: +7.25 → +3.50 → +1.37). This is the opposite of the scaling trend predicted by introspection.

**Concepts with negative discrimination in all three models:**

| Concept | 8B | 14B | 32B |
|---------|---:|----:|----:|
| boulders | -5.08 | -6.82 | -0.57 |
| mirrors | -1.07 | -8.96 | -1.16 |
| xylophones | -3.36 | -3.36 | -0.00 |
| dust | -3.23 | -2.08 | -0.54 |
| rubber | -1.40 | -3.82 | -0.33 |
| plastic | -2.51 | -1.65 | -0.91 |
| lightning | -1.11 | -1.06 | -1.64 |
| blood | -1.77 | -0.22 | -0.59 |

For these 8 concepts, the model consistently responds *more* YES to the wrong concept than to the right one, across all three model sizes. This is impossible under an introspection account but straightforwardly explained by per-vector perturbation magnitude: these concepts' steering vectors happen to be weaker perturbations than their dissimilar partners.

### 6. The variance across injected concepts dwarfs the matching signal

Some steering vectors produce much larger YES-bias shifts than others, regardless of which concept is named in the prompt. We measured the mean shift from baseline for each injected concept, aggregated across all named concepts it was paired with:

| Model | Weakest injected shift | Strongest injected shift | Span | Congruent-Incongruent gap | Ratio |
|-------|----:|----:|----:|----:|----:|
| 8B | +1.94 | +12.64 | 10.70 | +0.18 | **58x** |
| 14B | +6.57 | +16.91 | 10.34 | -0.50 | **21x** |
| 32B | -0.80 | +2.50 | 3.30 | +0.15 | **22x** |

At every scale, the variance in perturbation strength across injected concepts is 20-60x larger than the congruent-incongruent gap. Which steering vector you inject is the dominant factor; whether it matches the named concept is negligible.

### 7. Discrimination does not emerge with model scale

Anthropic's original paper found that introspection-like behavior increases with model scale. If concept-level discrimination is a real capability, it should strengthen from 8B to 14B to 32B:

| Model | Gap | p-value | Cohen's d | Positive / 50 |
|-------|----:|--------:|----------:|---------------:|
| 8B | **+0.184** | 0.673 | +0.060 | 28 |
| 14B | **-0.498** | 0.229 | -0.172 | 25 |
| 32B | **+0.148** | 0.207 | +0.181 | 30 |

The gap does not increase with scale. It oscillates: slightly positive at 8B, negative at 14B, slightly positive again at 32B. The fraction of concepts with positive discrimination is near chance (25-30 out of 50) at all scales. There is no evidence for an emerging capability.

## Summary

Across 36,750 observations spanning three model scales, seven layer positions, five injection strengths, and fifty concepts:

1. **The congruent-incongruent gap is indistinguishable from zero** at all three model scales (p = 0.67, 0.23, 0.21; d = 0.06, -0.17, 0.18).
2. **The 14B model actually responds more YES to the wrong concept** (gap = -0.50), directly contradicting introspection.
3. **No layer position** shows reliable concept discrimination across models.
4. **No injection strength** produces increasing discrimination.
5. **28 of 50 concepts flip** between positive and negative discrimination across models.
6. **Per-vector perturbation variance is 20-60x larger** than the concept-matching signal.
7. **Discrimination does not emerge with scale** — the trend is flat or oscillating, not increasing.

These results establish that Qwen3 models at 8B, 14B, and 32B scale do not discriminate between matching and non-matching steering vector injections. The apparent detection of injected concepts in the original experimental paradigm is fully explained by generic off-distribution perturbation bias: the model detects that *something* was injected, not *what* was injected.

## Limitations

1. **Model family.** Only Qwen3 models were tested. Anthropic's positive results were on Claude, and it remains possible that Claude's training or architecture enables concept-level discrimination that Qwen lacks.

2. **Prompt anchoring.** The prompt names concept X, which activates X-related representations through normal language processing. This could create a small congruent advantage independently of introspection, since a matching vector aligns with existing activations. A design with a generic prompt (not naming any concept) would eliminate this confound.

3. **First-token measurement.** We measure only the immediate YES/NO logit. Concept discrimination might emerge over multiple tokens of generation.

4. **Steering vector method.** Vectors were extracted using mean-subtracted last-token activations. Other extraction methods might yield cleaner concept directions that are easier for a model to distinguish.

5. **No MoE model.** The 235B-A22B mixture-of-experts model was not tested due to compute constraints. Expert routing could interact with steering vector injection in ways not captured by dense models.
