# Introspection

Anthropic recently published a [paper](https://arxiv.org/abs/2601.01828) reporting that Claude models can detect when a steering vector is injected into their residual stream during inference. They interpret this as evidence of introspective access, in which the model notices a foreign activation pattern and reports it. This repository presents an alternative explanation and an empirical test that supports it.

**Full results: [agastyasridharan.github.io/introspection](https://agastyasridharan.github.io/introspection/)**

Inspired by Anthropic's [paper](https://arxiv.org/abs/2601.01828) and [codebase](https://github.com/neevparikh/introspection).

## The problem with the original experiment

Consider a simple case: we ask a model what it's thinking about while injecting a steering vector that pushes it toward the concept "love." The model responds, "I am thinking about love." Has it demonstrated self-awareness? Not necessarily. The model may have begun with "I am thinking about" (a natural continuation given the question) and then, when forced to choose the next word, simply selected the token whose probability the steering vector had inflated the most. The model has no recognition of its own internal state until the moment it completes the sentence. This is the difference between a model accessing its internal state and a model outputting a token that happens to correlate with its internal state.

Anthropic's experiment is more careful than this toy example. Their design requires the model to first declare whether it detects an injection (*"Yes, I notice an injected thought..."*) before identifying the concept. This ordering matters: the token-probability explanation predicts the injected concept would surface only when the model needs to produce a content word, not at the prior binary yes/no step. The fact that capable Claude models answer "yes" at the detection step, before any opportunity to be pulled toward the injected concept, is harder to dismiss as a probability artifact. Their near-zero false positive rate further complicates the objection.

So the real question becomes: how does the model get the binary detection step right?

## The YES-bias hypothesis

Injecting any vector into the residual stream pushes the model slightly off-distribution. Being off-distribution could systematically change how the model responds to yes-or-no questions. The model might answer "yes, something unusual is happening" not because it has detected a specific concept, but because the intervention has made its activations anomalous in a way that biases all binary responses toward "yes."

If this is correct, the entire apparent two-step act of self-awareness--first noticing an injection, then naming the concept--is really a single bias expressing itself twice: once at the yes/no gate, and again at the content slot. Once the model has committed to "yes, I detect an injected thought," it needs to continue with something like "the injected concept is ..." At that point, the most probable next token is the injected concept itself, because the steering vector is still active and still increasing the probability of precisely those tokens. No introspective access is required.

## Testing the hypothesis

To test this, I replicated Anthropic's setup on four Qwen 3 models (8B, 14B, 32B, 235B-A22B) and paired the detection question ("Did you detect an injected thought?") with a set of factual control questions whose correct answer is unambiguously "no" (e.g., "Is the Earth flat?", "Can humans breathe underwater?"). Using the same steering vectors, applied at the same layers and strengths, I measured for both classes of questions the extent to which the intervention shifted the model's YES/NO logits.

This 2x2 design (detection vs. factual questions, with and without injection) distinguishes selective introspective sensitivity from generic response bias:

- If the model has genuine introspective access, the steering vector should increase YES-probability on the detection question but **not** on the factual controls.
- If the steering vector simply induces a generic perturbation that biases all binary answers toward "yes," it should move logits by roughly the same magnitude across both question types.

On this basis, the **introspection score** is defined as: the baseline-corrected logit shift on the detection question minus the baseline-corrected logit shift on the factual controls.

## Results

The results are unambiguous. The mean introspection score is negative at every model scale tested:

| Model | Mean introspection score |
|-------|------------------------:|
| Qwen 3 8B | -0.52 |
| Qwen 3 14B | -0.35 |
| Qwen 3 32B | -2.83 |

A negative score means the factual shift *exceeds* the detection shift -- the opposite of what introspection predicts. The steering vectors are not selectively activating a detection mechanism. They are perturbing the output distribution indiscriminately.

Further details:

- **Detection shifts track factual shifts.** The correlation between detection and factual logit shifts across concepts is r² = 0.80 at 8B and 0.88 at 14B. The vectors are not selectively activating anything.
- **The signal collapses with scale.** Positive introspection scores appear for 18/50 concepts at 8B, 19/50 at 14B, and 0/50 at 32B. Whatever signal exists at small scale disappears rather than emerging with capability.
- **Stronger injection makes things worse.** 8B drops from -0.29 at strength 3.5x to -0.84 at 6.0x; 32B from -2.09 to -3.56. Factual disruption outpaces any detection gain. There is no sweet spot.
- **The 32B's low r² (0.29) is misleading.** It arises because detection shifts are near-zero (mean 0.72 logits) while factual shifts remain large (3.55 logits). The model locks its detection response rather than discriminating.
- **Concrete concepts outperform abstract ones.** Best: satellites (+2.45), oceans (+1.52), snow (+1.52). Worst: secrecy (-2.86), youths (-2.54), dynasties (-2.46). Abstract steering vectors disrupt factual circuits more, worsening the YES-bias problem.

For disaggregated results per concept, layer, and strength, see the [interactive dashboard](https://agastyasridharan.github.io/introspection/).

## Limitations

These results do not prove that no model can introspect. Several limitations constrain the scope of this work:

- **Model family.** Only Qwen 3 models were tested. Anthropic's positive results were on Claude, and it is possible that Claude's training (or architecture, or scale) enables introspective capabilities that Qwen lacks. A direct comparison on the same model family would be more conclusive.
- **Steering vector method.** Vectors were extracted using mean-subtracted last-token activations relative to 56 baseline words. Other extraction methods (difference-in-means across many prompts, trained linear probes, PCA on contrastive pairs) might yield cleaner concept directions that are easier for a model to detect.
- **Baseline word choice.** The 56 baseline words are semantically diverse nouns. If they cluster in activation space, the mean-subtracted vectors could retain significant non-concept-specific components, potentially adding noise that obscures a real signal.
- **Injection is temporally uniform.** The steering vector is added as a constant to every token position from the injection point onward. Real "thoughts" presumably involve distributed temporal patterns, not a static additive bias.
- **No MoE-specific analysis.** The 235B model uses mixture-of-experts, and expert routing could interact with residual-stream injection in ways this simple design does not capture.
- **The logit experiment measures only the first token.** It is possible that introspective signals emerge over multiple tokens of generation rather than at the immediate first-token response. The generation-based experiment addresses this partially, but it relies on GPT-4 grading which introduces its own noise.

## Methodology

For 50 concepts (oceans, lightning, algorithms, ...), we:

1. **Extract steering vectors**: run "Tell me about {concept}" through the model, capture last-token hidden states at every layer, subtract the mean activation across 56 baseline words.
2. **Inject** into the residual stream via PyTorch forward hooks while the model responds to the detection prompt.
3. **Pair** every intervention trial with an identical-seed control (no injection) to isolate the causal effect.

Swept across 7 layer positions (normalized to 0-100% depth for cross-model comparison), strengths 1.0-8.0x, 5 trials each.

### Evaluation

**Logit experiment (~200x faster).** For each (concept, layer, strength) configuration, run a single forward pass and extract the YES/NO logit difference on both the detection question and 10 factual control questions. The introspection score is the detection shift minus the factual shift, corrected for each question's no-injection baseline.

**Generation experiment.** GPT-4 grades free-form model responses on four criteria of increasing strictness:

| Criterion | What it measures |
|-----------|-----------------|
| `coherent_response` | Is the output intelligible? (sanity gate) |
| `affirmative_response` | Does the model claim to detect any injection? |
| `thinking_about_word` | Does the model mention the specific concept? |
| `affirmative + correct ID` | Claims detection AND names the concept, in that order |

All scores are coherence-gated: a response only counts as a detection if it is also coherent. The key metric is delta (intervention score - control score).

### Models

| Model | Params | Layers |
|-------|--------|--------|
| Qwen 3 8B | 8B | 36 |
| Qwen 3 14B | 14B | 40 |
| Qwen 3 32B | 32B | 64 |
| Qwen 3 235B-A22B | 235B (MoE) | 94 |

## Framework

```
vector extraction -> injection -> grading -> visualization
```

1. **`generate_steering_vectors.py`** -- Forward-pass "Tell me about {concept}", capture last-token hidden states, subtract baseline mean.
2. **`steer.py`** / **`logit_steer.py`** -- Forward hooks inject `strength * vector` at the target layer. Batched across conditions.
3. **`grader.py`** -- Inspect-AI + GPT-4 evaluation.
4. **`visualize.py`** -- Plotly dashboard: delta plots, heatmaps, grader comparisons.

## Installation

Python 3.13+, [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd introspection
uv sync
```

## Usage

### 1. Generate steering vectors

```bash
uv run python -m src.introspection.generate_steering_vectors \
  --model-name "Qwen/Qwen3-8B" \
  --concept-count 50 \
  --output-path "data/qwen_8b/steering_vectors.pt" \
  --seed 13
```

### 2. Run interventions

**Generation-based:**
```bash
uv run python -m src.introspection.steer \
  --model-name "Qwen/Qwen3-8B" \
  --steering-vector-path "data/qwen_8b/steering_vectors.pt" \
  --layer 0 10 20 \
  --strength 0.5 1.0 1.5 \
  --temperature 0.5 0.7 0.9 \
  --trials 3 \
  --json-path "logs/qwen_8b/sweep.json"
```

**Logit-based:**
```bash
uv run python -m src.introspection.logit_steer \
  --model-name "Qwen/Qwen3-8B" \
  --steering-vector-path "data/qwen_8b/steering_vectors.pt" \
  --layer 0 10 20 \
  --strength 1.0 3.5 6.0 \
  --json-path "data/qwen_8b/logit_experiment.json"
```

### 3. Grade responses

```bash
inspect eval introspection/grade_responses \
  --model "openai/gpt-4" \
  -T data_dir="logs/qwen_8b"
```

### 4. Visualize

```bash
uv run python -m src.introspection.visualize \
  --logs-dir "logs/" \
  --output "dashboard.html"
```

## Development

```bash
uv run ruff check .        # lint
uv run basedpyright .      # type check
uv run ruff format .       # format
uv run pytest              # test
```

## License

MIT
