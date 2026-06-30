# Don't Mask That Period: Min-Based Bayesian Gating for Causal Masked Diffusion Language Models

**Justin Elliott Cobb** · Industrial Algebra · June 2026

---

## Abstract

Masked diffusion language models (MDLMs) convert autoregressive LMs into parallel-sampling
diffusion models by fine-tuning with random token corruption. While recent work
(DiffusionGemma, DeepMind 2026) demonstrates the viability of this approach at scale
with a 26B MoE model generating 256-token blocks at 4× autoregressive speed, we
identify a fundamental failure mode in the standard training procedure: uniform
random masking catastrophically fails in causal-MDLM fine-tuning of smaller
pretrained autoregressive models, collapsing to only periods at every position.

We introduce a **min-based Bayesian gating function** that combines a structural gate
(never mask punctuation) with a frequency gate (rare words mask more) via a simple
product. This eliminates the period collapse and restores factual correctness.
On a 50-prompt evaluation across six categories, our Bayesian-gated MDLM (Qwen 2 1.5B)
achieves 59% of the autoregressive baseline's quality at **5.1× lower latency** (117 ms
vs 599 ms). The gate generalizes across model families (Qwen 1.5B, TinyLlama 1.1B) and
the min operation connects naturally to tropical semiring algebra, suggesting a unified
framework for token-level and block-level diffusion gating. We release all code,
checkpoints, and evaluation manifests under open-source licenses.

## 1. Introduction

Diffusion language models (DLMs) promise a fundamental speed advantage over autoregressive
(AR) generation: instead of producing tokens one at a time, they generate multiple tokens
in parallel through iterative denoising [Austin et al. 2021, Sahoo et al. 2024]. Google
DeepMind's recent release of DiffusionGemma (June 2026) — a 26B Mixture-of-Experts model
generating 256-token blocks at 4× AR speed [DeepMind 2026] — demonstrates that this
approach works at industrial scale.

However, achieving these results requires either training a new model from scratch
(DiffusionGemma was built on the Gemma 4 backbone with a custom architecture) or
fine-tuning an existing pretrained autoregressive LM for the diffusion objective. We
focus on the latter: **converting public AR models into diffusion models through
limited fine-tuning on commodity corpora.** This path is accessible to researchers
without large-scale training infrastructure.

This conversion path faces a critical obstacle that has gone unreported: **the standard
uniform random masking approach catastrophically fails.** When fine-tuning a pretrained
AR model with uniform masking — the approach inherited from BERT [Devlin et al. 2019] —
the model collapses to predicting only periods at every generation position, regardless
of the input prompt. We trace this failure to the interaction between causal attention,
pretrained bigram statistics, and limited fine-tuning data (§3).

Our contribution is threefold:
1. We identify and empirically demonstrate the **uniform mask failure** in causal-MDLM
   fine-tuning — a negative result with implications for the standard training approach.
2. We propose **min-based Bayesian gating** as the solution: a structural gate prevents
   punctuation from being masked; a frequency gate focuses training on informative tokens;
   the two combine as a product gate (§4).
3. We validate across two model families (Qwen 1.5B, TinyLlama 1.1B) and show **5.1×
   speedup at 59% AR quality** on a 50-prompt evaluation set (§5).

The min operation is inspired by tropical semiring algebra, where min replaces conventional
addition as the fundamental combining operation [Maclagan & Sturmfels 2015]. This provides
a principled algebraic foundation and connects naturally to tropical Viterbi decoding for
block-level acceptance decisions (§8).

## 2. Background and Related Work

### 2.1 Absorbing-State Masked Diffusion

In the absorbing-state MDLM framework [Sahoo et al. 2024], a pretrained language model
is fine-tuned to predict randomly masked tokens. At training time, each token is
independently replaced with a special [MASK] token with probability p, following a
noise schedule. The model is trained to predict the original token at each masked
position, minimizing cross-entropy loss over masked positions only.

At inference time, generation begins with a prompt followed by a sequence of [MASK]
tokens. The model iteratively predicts tokens at masked positions, gradually
"unmasking" the sequence until all positions are filled. Each step produces multiple
tokens in parallel, providing the speed advantage over AR generation.

### 2.2 Standard Masking: Uniform Random

The standard approach — inherited from BERT [Devlin et al. 2019] — is uniform random
masking. At each training step, for each token independently: replace with [MASK]
with probability 15%, keep unchanged with probability 85%. This is the default in
virtually all masked language modeling literature [Raffel et al. 2020, Joshi et al.
2020] and is rarely questioned as a design choice.

### 2.3 DiffusionGemma

Concurrent with our work, Google DeepMind released DiffusionGemma [DeepMind 2026],
a 26B Mixture-of-Experts diffusion language model generating 256-token blocks via
iterative denoising with bidirectional attention. It achieves 4× speedup over AR
generation and runs on 24 GB GPUs (18 GB with quantization). DiffusionGemma validates
the AR→diffusion paradigm at industrial scale and is released under Apache 2.0.

Key differences: (1) DiffusionGemma uses a custom MoE architecture trained from
scratch on Gemma 4, while we fine-tune standard public AR models; (2) their scale
(26B MoE, 3.8B active) dwarfs ours (1.5B dense); (3) our Bayesian mask is a
training methodology applicable to any AR model, including DiffusionGemma's training
pipeline. The Bayesian gate could improve data efficiency and address period-bias
problems that may resurface when fine-tuning on domain-specific corpora.

### 2.4 Other Related Work

Masked language modeling traces to BERT [Devlin et al. 2019] and was extended to
span-based masking by T5 [Raffel et al. 2020] and SpanBERT [Joshi et al. 2020].
These approaches use linguistic pre-processing (whole-word masking, span boundaries)
that we avoid. Diffusion models for discrete data were introduced by D3PM [Austin
et al. 2021] and extended to text by SSD-LM [Han et al. 2022] and MDLM [Sahoo et
al. 2024]. Our work contributes to the MDLM line by identifying and fixing a
training pathology specific to causal fine-tuning.

Tropical algebra in machine learning has been applied to neural network analysis
[Zhang et al. 2018] and sequence modeling via Viterbi decoding. The amari-tropical
crate [Industrial Algebra 2025] provides max-plus primitives used in our block-level
extension.

## 3. The Uniform Mask Failure

### 3.1 Experimental Setup

We fine-tune Qwen 2 1.5B-Instruct [Yang et al. 2024] on 1M examples from the C4
corpus [Raffel et al. 2020] (en subset) for 50,000 steps with batch size 4, using
a cosine noise schedule (p = 1 − cos(πu/2), u ~ Uniform(0,1)), AdamW optimizer
(lr=1e-5, weight decay 0.01), gradient clipping at 1.0, and standard uniform random
masking. Training uses a single NVIDIA GB10 GPU (122 GB unified memory). The model
is loaded in bfloat16 precision.

### 3.2 Results

The uniform-mask model produces only periods at every generation position,
regardless of input prompt:

| Prompt | Model Output |
|--------|-------------|
| "The capital of France is" | `.` |
| "The chemical symbol for water is" | `.` |
| "The largest planet in our solar system is" | `.` |
| "What is 7 × 8?" | `.` |

This is not a rare failure mode — it is deterministic. Every prompt produces a period.
The model's token-level logits confirm: at the first masked position after "capital of
France is", the period token has 60.5% probability, followed by space (15.3%), newline
(6.8%), and comma (6.5%). Content-bearing tokens are overwhelmingly suppressed.

### 3.3 Root Cause Analysis

The pretraining corpus of Qwen 2 (and most large LMs) is dominated by a simple
statistical pattern: sentence-ending punctuation. In web text, the bigram
`[word, "."]` is the single most common token pair. When a pretrained AR model
sees [MASK] at the end of a sequence during fine-tuning, its pretrained prior
heavily favors predicting "." — because that is the token most frequently observed
at sequence-final positions across its pretraining data.

Causal attention amplifies this bias. The model only sees left context when predicting
a masked position. At sequence end, the left context is "… is [MASK]" — and the
most common completion of "is" in the pretraining data is a content word followed
by ".", so the model predicts "." directly rather than the intermediate content word.

With 50K steps of fine-tuning on 1M examples, the model receives insufficient training
signal to overcome this prior. The cosine schedule, which samples mask ratios across
[0,1], means most training steps see few masks (low-ratio regime), and high-ratio
steps are too corrupted to learn from. The model converges to the statistical mode.

### 3.4 Why Hasn't This Been Observed Before?

BERT and T5 use **bidirectional** attention and train **from scratch** on massive
datasets (3.3B+ words for BERT, 750 GB for T5). Bidirectional attention allows
the model to see right context, which signals "this position is not the end of the
sentence." Training from scratch means no pretrained period prior exists to overcome.

Causal-MDLM fine-tuning is a different regime: (1) limited fine-tuning data cannot
overcome pretrained bigram statistics, (2) causal attention removes the right-context
signal that would correct the bias, (3) the cosine schedule's high-ratio steps may
exacerbate the problem by presenting corrupted sequences where punctuation predictions
are the only statistically safe choice.

## 4. Min-Based Bayesian Gating

### 4.1 Design

We introduce a masking policy that structurally prevents punctuation from being masked
while statistically focusing training on informative tokens. The masking probability is a
product of three factors:

```
p_mask(t) = r · s(t) · f(t)
```

where r is the base mask ratio from the cosine noise schedule, s(t) is a binary
structural gate, and f(t) is a continuous frequency weight:

| Gate | Symbol | Formula | Effect |
|------|--------|---------|--------|
| Structural | s(t) | 0 if t ∈ punctuation/special, else 1 | Periods never masked |
| Frequency | f(t) | 1 / (1 + log(count(t) + 1)) | Rare words mask more |

Since s(t) ∈ {0,1}, this is equivalent to p_mask(t) = r · f(t) when t passes the
structural gate (s=1), and 0 otherwise. The product form is used throughout because
it makes the decomposition explicit.

### 4.2 Structural Gate s(t)

The structural gate is a binary filter over the tokenizer's vocabulary. Any token
that decodes to a string starting with punctuation (`.`, `,`, `?`, `!`, `;`, `:`,
`"`, `'`, `-`, `(`, `[`, `{`) is excluded from masking. Special tokens (EOS, PAD)
and the mask token itself are also excluded. This is computed once at startup by
scanning the tokenizer vocabulary (~0.1s).

This gate eliminates the period-collapse problem directly: periods are never masked,
so the model never learns to predict them at mask positions. The 60% probability
mass concentrated on "." in the uniform-mask model is redistributed to content tokens.

### 4.3 Frequency Gate f(t)

The frequency gate weights tokens by inverse log frequency. Token frequencies are
estimated from the first 50K lines of the training corpus (~5s at startup). Token
"the" (count ~10⁷) receives weight ~0.14. Token "Paris" (count ~10³) receives
weight ~0.85. The model trains primarily on rare, informative tokens.

### 4.4 The Min Operation

Using `min` rather than product or sum ensures **conservative gating**: both
conditions must independently approve masking before a token is eligible.

Note: For a binary structural gate s ∈ {0,1}, the min formulation
p = r·min(s,f) and the product formulation p = r·s·f are mathematically
identical. We use the product form for clarity. The structural gate provides
a hard veto (s=0 guarantees no masking); the frequency gate modulates the
rate for eligible tokens (s=1 → p = r·f).

The min-based formulation was originally inspired by tropical semiring
algebra [Maclagan & Sturmfels 2015], where min serves as the additive
operation (a ⊕ b = min(a,b)). The amari-tropical crate [Industrial Algebra
2025] provided the algebraic context that suggested the two-gate design.
The tropical connection extends more meaningfully to block-level acceptance
via max-plus Viterbi decoding (§8), where the max operation genuinely
differs from alternatives and provides a principled algebraic foundation.

### 4.5 Implementation

Both gates are O(1) per token. The structural gate uses a precomputed `frozenset`
of token IDs. The frequency gate uses a precomputed `Vec<f32>` of per-token weights.
No linguistic pre-processing, part-of-speech tagging, or architecture changes required.

## 5. Experiments

### 5.1 Main Result: Uniform vs Bayesian Mask

**Config:** Qwen 2 1.5B, bidirectional training, cosine schedule, C4 1M, 50K steps,
causal inference with left-to-right sequential unmasking, early stopping.

| Mask | "capital of France" | "symbol for water" | "largest planet" | tq (50) |
|------|--------------------|--------------------|--------------------|---------|
| Uniform | `.` | `.` | `.` | 0.000 |
| Bayesian | **Paris.** | **H₂O.** | **Jupiter.** | **0.340** |

### 5.2 Cross-Model Validation

Identical Bayesian mask config on TinyLlama 1.1B-Chat [Zhang et al. 2024]:

| Model | "capital of France" | "largest planet" |
|-------|--------------------|--------------------|
| Qwen 2 1.5B | **Paris.** | **Jupiter.** |
| TinyLlama 1.1B | **Paris.** | **Jupiter.** |

### 5.3 Causal vs Bidirectional Training

| Training Attention | "capital of France" | "largest planet" | Best Loss |
|-------------------|--------------------|--------------------|-----------|
| Causal | `Paris the of...` | `Mercury the...` | 3.46 |
| **Bidirectional** | **`Paris.`** | **`Jupiter.`** | **3.10** |

Bidirectional training produces better results with causal inference. This suggests
bidirectional context during training teaches richer representations that transfer
to improved causal predictions — an unexpected finding given that the model is
pretrained causally.

### 5.4 Speed/Quality Tradeoff (Expanded 50-Prompt Eval)

| Model | task_quality | Latency (ms) | Speedup | Memory |
|-------|-------------|-------------|---------|--------|
| AR 1.5B | 0.575 | 599 | 1.0× | 2.9 GB |
| MDLM (ours) | 0.340 | **117** | **5.1×** | 0.3 GB |

Per-category breakdown:

| Category | # Prompts | AR Accuracy | MDLM Accuracy | Ratio |
|----------|----------|------------|---------------|-------|
| Factual knowledge | 10 | 0.90 | 0.60 | 67% |
| Mathematics | 10 | 0.80 | 0.30 | 38% |
| Logic/yes-no | 10 | 0.80 | 0.50 | 63% |
| Code/technical | 10 | 0.30 | 0.10 | 33% |
| Translation | 5 | 0.40 | 0.20 | 50% |
| Writing/keywords | 5 | 0.25 | 0.34 | 136% |

The MDLM matches or exceeds AR quality on writing tasks (keyword-coverage scoring)
but lags on code and mathematics, which require precise token-level generation.

### 5.5 Scaling Experiments

| Run | Steps | Data | Best Loss | Factual Quality |
|-----|-------|------|-----------|-----------------|
| Qwen bidir 50K | 50K | 1M C4 | 3.10 | Paris, H₂O, Jupiter |
| Qwen bidir 100K | 100K | 1M C4 | 2.95 | Paris, H₂O, Jupiter |
| TinyLlama bidir 1M | 50K | 1M C4 | 2.25 | Paris, Jupiter |
| TinyLlama bidir 5M | 50K | 5M C4 | 2.26 | Paris, Jupiter |

Doubling steps and quintupling data diversity did not improve already-correct answers
or fix remaining weak spots. This suggests a model capacity or data quality ceiling
rather than a training budget limitation.

## 6. Sampler Design

### 6.1 Sequential Unmasking

We initially used confidence-based selection (unmask the most confident positions first),
which produced garbled output. The problem: sentence-ending punctuation has higher
prediction confidence than content tokens, so periods were unmasked first, poisoning
subsequent predictions.

Switching to **left-to-right sequential unmasking** (always fill the leftmost mask
position first) fixed this. This respects the causal structure of language and produces
coherent output.

### 6.2 Early Stopping

The model predicts sentence boundaries naturally after training. We detect
sentence-ending punctuation (`.`, `!`, `?`) and EOS tokens in generated tokens,
stopping generation immediately. This produces concise outputs (1-3 tokens for
factual prompts) rather than filling allocated positions with noise.

### 6.3 Block-Level Sampling

We extended the sampler to generate in blocks of k tokens (k=4,8,16,32). Token-trained
models work correctly with block-level sampling without block-specific training (§7.1).

## 7. Ablation Studies

### 7.1 Block Training (Negative Result)

| Corruption | Block Size | Best Loss | Quality |
|-----------|-----------|-----------|---------|
| Token-level | — | **3.10** | **Correct** |
| Block | 4 | 6.05 | Degraded |
| Block | 8 | 6.47 | Degraded |

Block-level training degrades factual recall — masking entire contiguous spans removes
too much context. Token training + block sampling achieves better results.

### 7.2 Gate Ablation

| Gates Active | "capital of France" | Best Loss |
|-------------|--------------------|-----------|
| None (uniform) | `.` | 4.86 |
| Structural only | `Paris.` | — |
| Frequency only | `Paris.` | — |
| Both (Bayesian) | **`Paris.`** | **3.10** |

Both gates independently prevent period collapse. The combination achieves lowest loss.

### 7.3 Speed Profile

| Sampling | Latency | vs AR (1463 ms) | Quality |
|----------|---------|-----------------|---------|
| Token-level | 203 ms | 7.2× | Correct |
| Block 32 | 339 ms | 4.3× | Correct |
| Block 8 | 532 ms | 2.8× | Correct |

## 8. Block Diffusion Extension

### 8.1 Tropical Block-Acceptance

For block-level generation, a critic model scores each token prediction. The block
acceptance decision uses tropical max (addition): accept if max(scores) > threshold.
The sequence of block decisions forms a Viterbi path through the block lattice.

This connects directly to the amari-tropical crate [Industrial Algebra 2025], which
provides max-plus primitives for Viterbi decoding. Implementation and empirical
validation of block-acceptance gating is future work.

## 9. Discussion

### 9.1 Limitations

Our experiments are limited to models up to 1.5B parameters and training corpora up to
5M examples. The quality gap on code and mathematics tasks (33-38% of AR) suggests
that MDLM generation may be better suited to factual and logical queries than
precision-demanding tasks. The 50-prompt evaluation set, while covering six categories,
is not a standard benchmark; results on MMLU, HumanEval, or similar would strengthen
the claims.

### 9.2 Future Work

1. **Block-acceptance critic:** Implement and evaluate the tropical Viterbi block-acceptance
   protocol described in §8.1.
2. **Larger-scale validation:** Apply the Bayesian mask to models at 7B+ scale with
   proportionally larger training corpora.
3. **Benchmark evaluation:** Report MDLM quality on standard benchmarks (MMLU, HumanEval)
   for direct comparison with AR baselines and DiffusionGemma.
4. **Theoretical analysis:** Formalize the connection between tropical gating and absorbing-state
   diffusion noise schedules.

## 10. Conclusion

We have shown that the standard uniform random masking approach catastrophically fails in
causal-MDLM fine-tuning of pretrained autoregressive models, producing models that output
only periods at every generation position. This failure has implications for the standard
MDLM training procedure and may affect practitioners attempting AR→diffusion conversion
at any scale.

Our min-based Bayesian gating function eliminates this failure: a structural gate prevents
punctuation from being masked, a frequency gate focuses training on informative tokens,
and the tropical min operation ensures conservative combination. The resulting MDLM achieves
59% of autoregressive quality at 5.1× lower latency, with cross-model validation across
Qwen 1.5B and TinyLlama 1.1B.

The Bayesian gate is a drop-in modification requiring no architectural changes. The tropical
min operation connects to a growing body of work on algebraic methods in machine learning
and extends naturally to block-level diffusion via Viterbi acceptance decoding.

## References

[1] J. Austin, D. D. Johnson, J. Ho, D. Tarlow, and R. van den Berg. "Structured Denoising
Diffusion Models in Discrete State-Spaces." NeurIPS, 2021.

[2] DeepMind. "DiffusionGemma." Technical report, Google DeepMind, June 2026.
https://diffusiongemma.dev/

[3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. "BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding." NAACL, 2019.

[4] X. Han, S. Kumar, and Y. Tsvetkov. "SSD-LM: Semi-autoregressive Simplex-based
Diffusion Language Model for Text Generation and Modular Control." ACL, 2022.

[5] Industrial Algebra. "amari-tropical: Tropical max-plus algebra and ordinal-weighted
optimization carriers." https://github.com/Industrial-Algebra/Amari, 2025.

[6] M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, and O. Levy. "SpanBERT:
Improving Pre-training by Representing and Predicting Spans." TACL, 2020.

[7] D. Maclagan and B. Sturmfels. "Introduction to Tropical Geometry." AMS, 2015.

[8] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and
P. J. Liu. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text
Transformer." JMLR, 2020.

[9] S. S. Sahoo, M. Arriola, Y. Schiff, A. Gokaslan, E. Marroquin, J. T. Chiu, A. Rush,
and V. Kuleshov. "Simple and Effective Masked Diffusion Language Models." NeurIPS, 2024.

[10] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, et al.
"Qwen2 Technical Report." arXiv:2407.10671, 2024.

[11] P. Zhang, G. Zeng, T. Wang, and W. Lu. "TinyLlama: An Open-Source Small Language
Model." arXiv:2401.02385, 2024.

[12] L. Zhang, G. Naitzat, and L.-H. Lim. "Tropical Geometry of Deep Neural Networks."
ICML, 2018.
