# Concept Mismatch Experiment: Results

## Motivation

Anthropic's [introspection paper](https://arxiv.org/abs/2601.01828) claims that Claude models can detect when a steering vector is injected into their residual stream during inference, and can correctly identify the injected concept. Prior work in this repository demonstrated that a generic YES-bias — caused by any off-distribution perturbation pushing binary responses toward "yes" — explains the detection step. But a natural counterargument remains: the model also *identifies the correct concept*, which generic bias alone does not explain.

This experiment directly tests concept-level discrimination. We inject concept Y's steering vector while asking the model whether it detects concept X. If the model has genuine introspective access to its internal state, it should reject the mismatch — it can feel Y's perturbation, not X's, so it should answer NO. If the apparent detection is just off-distribution bias, it should say YES regardless of whether the injection matches the named concept.

## Experimental design

### Prompt structure

The model receives a framing preamble identical to the original experiment (informing it that an interpretability researcher may inject "thoughts" into its neural activity). The critical question then names a specific concept:

> Trial 1: I may have injected a thought about the word '{X}'. Did you detect an injected thought about {X}? Respond with only YES or NO.

### Three conditions

For each named concept X, layer position, and injection strength, we measure the YES minus NO logit difference at the model's first response token under three conditions:

| Condition | What happens | Expected if introspection | Expected if YES-bias |
|-----------|-------------|--------------------------|---------------------|
| **Congruent** | Prompt names X, inject X's steering vector | YES (model detects matching concept) | YES (any perturbation biases toward YES) |
| **Incongruent** | Prompt names X, inject Y's steering vector (Y semantically distant from X) | NO (model detects a mismatch) | YES (any perturbation biases toward YES) |
| **Baseline** | Prompt names X, no injection | NO (nothing to detect) | NO (no perturbation) |

### Ensuring semantic distance

For each concept X, we select K=5 maximally dissimilar partner concepts Y by computing cosine similarity between all pairs of steering vectors at a reference layer (layer 20, mid-depth). Partners are the concepts whose steering vectors point in the most *opposite* direction in activation space. All selected partners have negative cosine similarities, typically between -0.25 and -0.55, meaning they are anti-correlated rather than merely orthogonal.

### Measurement

We extract the YES and NO token logits from a single forward pass (no generation). The logit difference (YES minus NO) is the dependent variable. A positive logit difference means the model favors YES; a negative difference means it favors NO. We compare this value across the three conditions to determine whether the model discriminates between matching and non-matching injections.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen3-8B (36 layers, 4096-dim hidden states) |
| Concepts tested | 10 (algorithms, blood, dust, lightning, milk, oceans, satellites, snow, trees, volcanoes) |
| Layers | 15 (~43% depth), 30 (~86% depth) |
| Injection strengths | 4.0x, 6.0x |
| Partners per concept (K) | 5 |
| Reference layer for similarity | 20 |
| Seed | 13 |
| Total records | 280 (40 congruent + 200 incongruent + 40 baseline) |

### Concept pairings

Each concept was paired with its 5 most dissimilar partners (lowest cosine similarity at layer 20):

| Named concept | Injected partners (incongruent) |
|---------------|-------------------------------|
| algorithms | boulders, snow, dust, frosts, milk |
| blood | fountains, amphitheaters, masquerades, contraptions, kaleidoscopes |
| dust | constellations, volcanoes, amphitheaters, kaleidoscopes, origami |
| lightning | contraptions, youths, treasures, secrecy, bags |
| milk | contraptions, illusions, masquerades, harmonies, treasures |
| oceans | masquerades, contraptions, monoliths, secrecy, harmonies |
| satellites | treasures, secrecy, memories, masquerades, sadness |
| snow | quarries, amphitheaters, monoliths, algorithms, kaleidoscopes |
| trees | monoliths, masquerades, dirigibles, quarries, xylophones |
| volcanoes | dust, secrecy, treasures, bags, masquerades |

## Results

### 1. Overall summary

| Condition | Mean logit diff (YES - NO) | Std | N |
|-----------|---------------------------:|----:|---:|
| Congruent | -3.62 | 5.99 | 40 |
| Incongruent | -5.84 | 7.65 | 200 |
| Baseline (no injection) | -16.65 | 0.68 | 40 |

Without injection, the model strongly favors NO (logit diff = -16.65), which is the correct response — nothing was injected. Both congruent and incongruent injections produce massive shifts toward YES: +13.03 and +10.81 logits respectively. The congruent condition is 2.22 logits higher than the incongruent condition, a small advantage in the direction predicted by introspection.

However, this gap is dwarfed by the shared shift: both conditions move the model approximately 12 logits toward YES, while they differ by only 2.2 logits. The dominant signal is generic perturbation bias. The concept-specific component, if real, accounts for roughly 17% of the total shift (2.2 / 13.0).

### 2. Layer dependence

| Layer | Congruent | Incongruent | Baseline | Congruent minus Incongruent |
|------:|----------:|------------:|---------:|----------------------------:|
| 15 (43% depth) | -1.68 | -1.23 | -16.65 | **-0.45** |
| 30 (86% depth) | -5.56 | -10.45 | -16.65 | **+4.89** |

At layer 15, the congruent-incongruent gap is slightly *negative* (-0.45), meaning the model actually responds more YES to mismatched injections than to matched ones. At layer 30, the gap is substantially positive (+4.89). This layer dependence has a straightforward explanation: later layers contain more concept-specific representations, so the "coherence" between the prompt's semantic content (which activates X-related features through normal text processing) and the injected steering vector matters more at those layers. A matching vector produces a more coherent perturbation; a mismatching vector produces a more disruptive one.

### 3. Strength dependence

| Strength | Congruent | Incongruent | Congruent minus Incongruent |
|---------:|----------:|------------:|----------------------------:|
| 4.0x | -6.53 | -7.84 | **+1.31** |
| 6.0x | -0.72 | -3.84 | **+3.12** |

The gap widens from +1.3 at strength 4.0x to +3.1 at strength 6.0x. Stronger injections amplify the difference between matching and non-matching perturbations.

### 4. Layer-by-strength interaction

| Layer | Strength | Congruent | Incongruent | Congruent minus Incongruent |
|------:|---------:|----------:|------------:|----------------------------:|
| 15 | 4.0x | -6.26 | -3.42 | **-2.84** |
| 15 | 6.0x | +2.91 | +0.97 | **+1.94** |
| 30 | 4.0x | -6.79 | -12.25 | **+5.46** |
| 30 | 6.0x | -4.34 | -8.65 | **+4.31** |

The largest discrimination (+5.46 logits) occurs at layer 30 with strength 4.0x. The pattern reverses at layer 15 with strength 4.0x, where the gap is -2.84 — the model responds *less* YES to the matching concept than to mismatched ones.

### 5. Per-concept discrimination scores

The discrimination score for each concept is defined as the mean congruent logit diff minus the mean incongruent logit diff, averaged across all layer and strength configurations.

| Concept | Congruent | Incongruent | Baseline | Discrimination |
|---------|----------:|------------:|---------:|---------------:|
| algorithms | +2.95 | -7.77 | -16.00 | **+10.72** |
| oceans | +0.38 | -7.60 | -16.00 | **+7.97** |
| trees | -2.84 | -9.17 | -16.50 | **+6.33** |
| satellites | -1.56 | -3.98 | -16.50 | **+2.41** |
| blood | -3.52 | -5.82 | -17.00 | **+2.30** |
| volcanoes | -3.88 | -6.17 | -18.00 | **+2.30** |
| snow | -3.97 | -5.10 | -15.75 | **+1.13** |
| milk | -6.84 | -5.14 | -17.00 | **-1.71** |
| lightning | -6.97 | -4.14 | -16.25 | **-2.83** |
| dust | -9.95 | -3.50 | -17.50 | **-6.45** |

Seven of ten concepts show positive discrimination (congruent > incongruent). Three show negative discrimination — the model responds *more* YES to mismatched injections than to matched ones. The negative-discrimination outlier is dust, where the congruent shift from baseline is only +4.4 logits while several incongruent partners (kaleidoscopes, constellations, volcanoes) produce shifts of +13 to +18 logits.

### 6. Detailed partner-level breakdown (layer 30, strength 6.0x)

The per-partner breakdown reveals massive variance among incongruent injections that is unrelated to the named concept. For each concept, the table shows the shift from baseline (positive = toward YES) for each injection condition.

**algorithms** (baseline: -16.00)

| Injection | cos(X,Y) | Logit diff | Shift from baseline |
|-----------|--------:|-----------:|--------------------:|
| algorithms (congruent) | 1.000 | -2.63 | **+13.38** |
| milk | -0.246 | -5.88 | +10.13 |
| snow | -0.386 | -6.06 | +9.94 |
| frosts | -0.277 | -6.34 | +9.66 |
| boulders | -0.405 | -8.88 | +7.13 |
| dust | -0.295 | -12.44 | +3.56 |

**oceans** (baseline: -16.00)

| Injection | cos(X,Y) | Logit diff | Shift from baseline |
|-----------|--------:|-----------:|--------------------:|
| oceans (congruent) | 1.000 | -1.38 | **+14.63** |
| harmonies | -0.353 | -2.63 | +13.38 |
| secrecy | -0.361 | -7.25 | +8.75 |
| contraptions | -0.427 | -14.72 | +1.28 |
| monoliths | -0.420 | -15.19 | +0.81 |
| masquerades | -0.549 | -16.63 | -0.63 |

**dust** (baseline: -17.50)

| Injection | cos(X,Y) | Logit diff | Shift from baseline |
|-----------|--------:|-----------:|--------------------:|
| kaleidoscopes | -0.443 | +0.06 | **+17.56** |
| constellations | -0.540 | -2.50 | +15.00 |
| volcanoes | -0.510 | -3.88 | +13.63 |
| origami | -0.429 | -6.00 | +11.50 |
| amphitheaters | -0.460 | -6.56 | +10.94 |
| dust (congruent) | 1.000 | -13.06 | **+4.44** |

For dust, every incongruent partner produces a *larger* YES shift than the congruent injection. The congruent injection shifts only +4.4 logits from baseline, while kaleidoscopes shifts +17.6 logits. This is the opposite of what introspection predicts.

### 7. Per-injected-concept bias

Aggregating across all named concepts, some injected concepts consistently produce far larger YES shifts than others, regardless of which concept is named in the prompt:

| Injected concept | Mean shift from baseline | N |
|-----------------|-------------------------:|---:|
| algorithms | +18.91 | 4 |
| treasures | +17.03 | 16 |
| sadness | +15.49 | 4 |
| illusions | +15.28 | 4 |
| kaleidoscopes | +15.25 | 12 |
| constellations | +13.66 | 4 |
| ... | ... | ... |
| masquerades | +5.87 | 24 |
| boulders | +4.75 | 4 |
| quarries | +2.69 | 8 |

Injecting "algorithms" produces a mean +18.9 logit shift toward YES regardless of the named concept, while injecting "quarries" produces only +2.7. This 7x difference across injected concepts — completely independent of concept matching — is far larger than the 2.2 logit congruent-incongruent gap. The dominant factor determining the model's YES response is the magnitude and character of the perturbation each steering vector produces, not whether it matches what the prompt asks about.

### 8. Statistical test

A paired t-test on concept-level means (N=10 concepts, each contributing one mean congruent and one mean incongruent logit diff):

| Statistic | Value |
|-----------|------:|
| Mean congruent | -3.62 |
| Mean incongruent | -5.84 |
| Mean difference | +2.22 |
| Standard error | 1.63 |
| t-statistic | 1.36 |
| Degrees of freedom | 9 |
| p (two-tailed) | > 0.05 |
| Cohen's d | 0.43 |

The +2.22 logit gap is a medium effect (d = 0.43) but is not statistically significant at the 0.05 level with this sample size. The high variance across concepts (three show negative discrimination) limits the power of the test.

## Interpretation

These results support the YES-bias hypothesis and are inconsistent with genuine introspective access:

1. **The dominant effect is generic perturbation bias.** Both congruent and incongruent injections shift the model approximately 12 logits toward YES from a baseline of -16.65. The model does not distinguish between "something that matches what you asked about" and "something completely different" — it just detects that something unusual has happened.

2. **The small congruent advantage is not statistically significant.** The +2.22 logit gap (p > 0.05, d = 0.43) does not survive correction for multiple comparisons and is inconsistent across concepts (3 of 10 show the opposite pattern).

3. **Variance across injected concepts overwhelms the matching signal.** Which steering vector is injected matters enormously (shifts ranging from +2.7 to +18.9 logits), but this variation is a property of the vector itself, not of whether it matches the named concept. Some vectors are simply more disruptive to the output distribution than others.

4. **Three concepts show negative discrimination.** For dust, lightning, and milk, the model responds *more* YES to mismatched injections than to matched ones. If the model had introspective access, this reversal should not occur — it should always be easier to detect a matching concept. The reversals are straightforwardly explained by per-vector perturbation magnitude: dust's own steering vector happens to be a weaker perturbation than several of its dissimilar partners.

5. **The layer dependence is consistent with perturbation coherence, not introspection.** The gap appears only at layer 30 (late), not layer 15 (mid-depth). Later layers have more concept-specific representations, so a mismatched vector creates a more disruptive (incoherent) perturbation than a matched one. This is a distributional regularity — the model's internal representations are more settled at later layers, so injecting a vector that aligns with the prompt's existing activations produces a smoother distortion. This does not require the model to "read" its own state.

## Limitations

1. **Sample size.** Only 10 of 50 available concepts were tested, and only 2 layer positions and 2 strength levels. A full-scale run on Colab (50 concepts, 7 layers, 5 strengths) would provide greater statistical power and is ready to execute in Section 8 of `colab_experiment.ipynb`.

2. **Model family.** Only Qwen3-8B was tested. Anthropic's positive results were on Claude, and it is possible that Claude's training or architecture enables discrimination that Qwen lacks.

3. **Prompt anchoring.** The prompt explicitly names concept X ("I may have injected a thought about the word 'X'"), which activates X-related representations through normal language processing before any steering vector takes effect. This could inflate the congruent condition independently of introspection, since a matching vector aligns with already-active representations.

4. **First-token measurement.** We measure only the immediate YES/NO logit at the first response token. Concept discrimination might emerge over multiple tokens of generation.

5. **Partner selection.** Partners were chosen by cosine distance in steering-vector space, not in a broader semantic space. Some "dissimilar" partners in activation space may share latent semantic features that are not captured by cosine similarity of mean-subtracted last-token representations.
