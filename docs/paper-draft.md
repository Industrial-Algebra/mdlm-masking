# Don't Mask That Period: Structural-Frequency Masking for Causal Masked Diffusion Language Models

**Justin Elliott Cobb** · Industrial Algebra · November 2026 (v2 — evidence revision)

> v2 note: this revision supersedes the July 2026 v1. It replaces the informal
> 50-prompt comparisons with a matched 100k-step ablation, adds confidence
> intervals, a diffusion-native NELBO, and a generative factual evaluation, and
> relabels the failure mechanism as a hypothesis. The v1 "59% of AR accuracy"
> framing is withdrawn.

---

## Abstract

Masked diffusion language models (MDLMs) convert autoregressive LMs into
parallel-sampling diffusion models by fine-tuning with random token corruption. We
study a controlled question at small scale: *does the distribution governing which
tokens get corrupted during fine-tuning matter?* In a matched 100k-step ablation on
Qwen2 1.5B (identical model, data, schedule, seed, and attention; only the
token-selection function differs), we find that the training-time mask distribution
**decouples autoregressive discriminative quality from diffusion generative
quality**. A structural-frequency product gate (never mask punctuation; mask rare
tokens more) and uniform-random masking produce checkpoints that are *statistically
indistinguishable* on standard multiple-choice benchmarks — ARC, HellaSwag,
WinoGrande, and MMLU under autoregressive scoring (all |z| < 0.7) — yet differ
drastically in generation: on a 40-prompt gold-answered factual set, the gated model
reproduces the correct fact 95% of the time while the uniform model does so 12.5% of
the time, period-collapsing on 82.5% of prompts. The gap is confirmed by likelihood:
LAMBADA perplexity 6.83 vs 8.21 (z ≈ 5) under AR scoring, and a diffusion-native
NELBO of 3.30 vs 3.91 nats/token (z ≈ 10, upper bounds), with schedule-invariance
validation. **The mask scheme is a generative-quality lever, not a discriminative
one.** Code, manifests, and the ablation protocol are released.

## 1. Introduction

Diffusion LMs promise parallel generation through iterative denoising, and recent
systems demonstrate the approach at production scale. A complementary path —
fine-tuning a public AR model into an MDLM on commodity hardware — is the setting we
study.

The standard recipe is uniform random token corruption, inherited from BERT-era
masked LM. We ask: among the design choices in MDLM fine-tuning, does the
*distribution governing which tokens are corrupted* matter? Prior and concurrent work
modifies the corruption *schedule* (the time axis): mask-ratio schedules,
context-weighted loss reweighting, layer-wise partitioning of the schedule
(DiffusionBlocks). None varies **token selection** — which real tokens absorb the
corruption budget.

We answer with a matched ablation: two runs of Qwen2 1.5B-Instruct, identical in
every configuration field (100k steps, cosine multi-timestep schedule, bidirectional
training, causal inference, batch, seed, data), differing only in the
token-selection function — uniform vs a **structural-frequency product gate**
(structural veto on punctuation × inverse-log-frequency weight).

The result is a decoupling:

1. **Discriminative quality is unchanged.** Under AR scoring the checkpoints are
   statistically indistinguishable on ARC-Challenge, HellaSwag, WinoGrande, MMLU
   (all |z| < 0.7, mixed sign; §5.4).
2. **Generative quality differs drastically.** 40-prompt gold-answered factual set:
   gated 95% gold-match with clean sentence-terminated outputs; uniform 12.5%
   gold-match, bare-period emission on 82.5% of prompts (§5.1).
3. **Likelihood agrees with generation, not the benchmarks.** LAMBADA PPL 6.83 vs
   8.21 (z ≈ 5); diffusion-native NELBO 0.61 nats/token better (z ≈ 10; §5.2).

The masking policy is a *generative-quality lever*. Standard MC evaluation cannot see
the difference because it measures the discriminative pathway; uniform masking's
failure mode lives in the generative pathway, where the model period-collapses rather
than emitting content.

## 2. Background and Related Work

### 2.1 Absorbing-State Masked Diffusion

MDLM (Sahoo et al. 2024): fine-tune a pretrained LM to predict masked tokens;
cross-entropy over masked positions; inference by iterative parallel unmasking.

### 2.2 Three Axes of Masking-Process Design

- **Schedule weighting (time axis):** mask-ratio schedules and time reweighting
  (Sahoo et al.; Context-weighted Discrete Flow Matching, arXiv:2607.21427);
  DiffusionBlocks (Shing, Koyama, Akiba, ICLR 2026, arXiv:2506.14202) partitions the
  schedule across *layer*-blocks for memory-efficient training. Multi-Mask DLMs
  (arXiv:2607.19686) modify the terminal mask state space.
- **Model parameterization:** SUBS-style denoisers, score entropy
  (Mean-to-Score, arXiv:2607.21372).
- **Token selection (this work):** which real tokens absorb the corruption budget.
  To our knowledge no prior or concurrent system varies this; uniform i.i.d.
  selection is universal.

### 2.3 Likelihood Evaluation for Masked Diffusion

Block Diffusion (Arriola et al., ICLR 2025) develops NELBO-based likelihood for
masked/block diffusion and presents the simplified masked-diffusion NELBO of Sahoo et
al. that we use (§5.2); it is schedule-invariant, which provides a built-in
implementation check. DiffusionBlocks reports text8 bpc under the same objective
family.

### 2.4 Deployed Discrete Diffusion

DiffusionGemma (26B MoE, 4× AR speed, Apache 2.0) validates the paradigm at scale;
radiology deployment (arXiv:2607.01436) and prefix KV caching (arXiv:2606.07571)
complement our small-scale conversion-methodology question. BERT-era masking
heuristics used bidirectional from-scratch training — a different regime (§3.4).

## 3. The Uniform-Mask Degradation, Controlled

### 3.1 Matched Ablation Protocol

Two fine-tuning runs of Qwen2 1.5B-Instruct on a local C4 slice, identical in every
configuration field: 100,000 optimizer steps, batch size 4, seed 42, AdamW
(lr 1e-5), bfloat16, bidirectional training attention with causal inference-time
attention, multi-timestep cosine corruption schedule (ratio r = 1 − cos(πu/2), u ~
U(0,1), mean ratio ≈ 0.6) on top of a 0.15 base token-masking rate. The runs differ
in exactly one function:

- **Uniform:** each non-special token masked i.i.d. with equal probability.
- **Gated:** masking probability multiplied by the structural-frequency product gate
  (§4).

Both trained on a single NVIDIA GB10 GPU (122 GB unified memory) to 99,999 steps.
Checkpoints, configs, and run manifests are released with the evaluation artifacts.

### 3.2 Training Loss and the Collapse

Gated best masked-cross-entropy loss **3.124**; uniform **4.857** — a 1.73-nat gap on
the training objective itself. Under greedy diffusion sampling the uniform checkpoint
exhibits *period collapse*: a bare "." at the first generation position regardless of
prompt (period holds ≈60% probability at first masked positions). The gated
checkpoint produces sentence-terminated factual completions (§5.1 quantifies).

### 3.3 Hypothesized Mechanism (Not Established)

**Hypothesis** (consistent with, but not directly established by, our evidence):
web-scale pretraining makes sentence-final punctuation the dominant completion
prior; uniform masking spends its corruption budget mostly on high-frequency tokens,
so the denoising objective is dominated by frequent glue-token prediction and the
period prior is never unlearned. The structural gate removes punctuation from the
pool entirely; the frequency gate shifts budget toward informative rare tokens.

Two corrections to the v1 framing: (i) this is a hypothesis, not a verified
mechanism; (ii) under cosine t-sampling the *mean* mask ratio is ≈ 0.6 — v1's "most
steps see few masks" characterization was wrong.

### 3.4 Why Hasn't This Been Observed Before?

BERT/T5: bidirectional attention, from-scratch training on massive corpora —
right-context corrects the end-of-sequence bias; no inherited period prior.
Production DLMs train from scratch at scale. Small-scale *causal fine-tune of a
pretrained AR model* is the exposed regime. And the failure is *invisible to
discriminative evaluation* (§5.4) — most suites are discriminative.

## 4. Structural-Frequency Product Gating

```
p_mask(t) = r · s(t) · f(t)
```

| Gate | Symbol | Formula | Effect |
|------|--------|---------|--------|
| Structural | s(t) | 0 if t ∈ punctuation/special, else 1 | Periods never masked |
| Frequency | f(t) | 1 / (1 + log(count(t) + 1)) | Rare words mask more |

- **Structural gate:** binary vocabulary filter (~0.1 s startup). Punctuation,
  special tokens, and the mask token are never masked.
- **Frequency gate:** inverse log frequency from the first 50K corpus lines (~5 s
  startup). "the" (count ~10⁷) → ≈0.14; "Paris" (count ~10³) → ≈0.85.
- **Composition:** the structural gate is a binary veto; for s=1 the effective
  probability is r·f(t). Since s ∈ {0,1}, the product form equals r·min(s,f) — the
  product formulation reflects intended semantics (independent probability factors),
  not a different distribution. Both gates are O(1) per token; no linguistic
  pre-processing, no architecture changes.

## 5. Evidence

All evaluation on the matched pair (§3.1); greedy generation unless stated.
Bootstrap SEs from lm-eval-harness; z = (ĝ − û)/√(se_g² + se_u²); |z| ≥ 2
significant.

### 5.1 Generative Factual Evaluation (Primary)

Standard benchmarks measure discrimination; the failure mode is generative. We
evaluate generation directly: 40 factual prompts with gold answers across eight
categories (capitals, chemistry, astronomy, currency, geography, arts, language,
science). **Provenance: author-authored, not a standard benchmark** — deliberately,
because standard suites cannot detect period collapse. Metrics: gold-match
factuality, collapse rate, other-rate, median oracle-PPL (base Qwen2-1.5B-Instruct
judge).

| Metric | Gated | Uniform |
|---|---|---|
| Gold-match factuality | **0.950** | 0.125 |
| Period collapse | **0.025** | 0.825 |
| Other (wrong/tautology) | **0.025** | 0.050 |
| Mean generated tokens | 3.5 | 1.4 |
| Oracle PPL (median) | **26.7** | 635.8 |

Per-category gold-match (gated/uniform): capitals 0.92/0.00 (n=12), chemistry
1.00/0.14 (7), astronomy 1.00/0.33 (6), currency 1.00/0.00 (5), geography 1.00/0.20
(5), arts 1.00/0.00 (3). Uniform period-collapses on 8/8 capital prompts. The
collapsed period is *contextually* surprising to the oracle judge (it expects the
fact) — hence uniform's high oracle PPL.

### 5.2 Diffusion-Native Likelihood (NELBO)

Conditional NELBO of continuation given context, following the simplified
masked-diffusion bound of Sahoo et al. as presented in Block Diffusion (App. B.3):
context never masked (conditioned block), continuation corrupted at 1 − α(t),
weighted masked cross-entropy integrated over t ~ U(0,1). NELBO ≥ NLL, so PPLs are
**upper bounds**. Masks paired across checkpoints. Full LAMBADA test (5,153
examples, 100 t-samples):

| | NELBO (nats/tok) | bpc | PPL (upper bound) |
|---|---|---|---|
| Gated | **3.300** ± 0.042 | 4.76 | 27.1 |
| Uniform | 3.909 ± 0.044 | 5.64 | 49.9 |
| Gap | −0.610 (z ≈ −10) | −0.88 | 1.84× |

Schedule invariance (Sahoo et al.): linear α = 1−t vs quadratic α = (1−t)² on the
gated checkpoint agree to 4.1% — validating the implementation. The diffusion-native
likelihood discriminates the pair an order of magnitude more sharply than AR
benchmark scoring: the metric native to the generative pathway sees the difference
the benchmarks cannot.

### 5.3 Cloze Likelihood: LAMBADA

True AR scoring (real continuation, causal forward):

| | Gated | Uniform |
|---|---|---|
| LAMBADA PPL | **6.83** | 8.21 |
| LAMBADA accuracy | **0.590** | 0.536 |
| Masked-span PL PPL | **759** | 1906 |

PPL/accuracy differences significant (z ≈ −4.8, +5.5). PL row = parallel masked
prediction (MDLM-native, higher-variance view).

### 5.4 Discriminative Equivalence: The Decoupling

Same checkpoints, AR-scored standard MC benchmarks:

| Benchmark | Gated | Uniform | Δ | z |
|---|---|---|---|---|
| ARC-Challenge (acc) | 0.3635 | 0.3558 | +0.0077 | +0.39 |
| ARC-Challenge (acc_norm) | 0.3925 | 0.3968 | −0.0043 | −0.21 |
| HellaSwag (acc) | 0.4928 | 0.4891 | +0.0037 | +0.52 |
| HellaSwag (acc_norm) | 0.6523 | 0.6559 | −0.0037 | −0.55 |
| WinoGrande (acc) | 0.6551 | 0.6582 | −0.0032 | −0.17 |
| MMLU (acc) | 0.5551 | 0.5588 | −0.0036 | −0.64 |

All within noise (|z| < 0.7), mixed sign. Token-level next-token accuracy agrees:
uniform marginally *higher* (0.392 vs 0.356). The generative pathway is where the
mask distribution acts.

### 5.5 Secondary Results (Preliminary Runs, Provenance Stated)

- **Cross-model:** effect reproduces on TinyLlama 1.1B-Chat.
- **Attention regime:** bidirectional training + causal inference beats causal
  training (best loss 3.10 vs 3.46); both matched cells use the bidirectional regime,
  so the matched comparison isolates the gate.
- **Block training (negative result):** block-level corruption degrades factual
  recall; token-trained models support block sampling without block-specific
  training.
- **Gate ablation:** either gate alone prevents collapse; both minimize loss (4.86
  uniform vs 3.10 gated, preliminary).
- **Conciseness (custom 50-prompt set):** 19-char average outputs vs AR's 45;
  117–134 ms time-to-output vs 599 ms — conciseness from sentence-boundary early
  stopping, not per-token speed (≈0.9× AR).

## 6. Sampler Design

- **Sequential unmasking:** left-to-right (most-confident-first fails — punctuation
  carries high confidence and poisons subsequent predictions).
- **Early stopping:** sentence-ending punctuation/EOS stops generation → 1–3-token
  factual answers.
- **Block-level sampling:** blocks of k ∈ {4,8,16,32}; works on token-trained models
  without block-specific training.

## 7. Discussion

### 7.1 Limitations

1. **Generative set provenance:** author-authored 40-prompt set; no standard
   benchmark measures this failure mode (that is the point); set, gold answers, and
   scorer released.
2. **ELBO upper bounds:** NELBO-derived PPL/bpc upper-bound the truth (loosely:
   true LAMBADA PPL is 6.83/8.21). Differences and significance valid; absolutes are
   not perplexities. Variance cap (w ≤ 20) biases absolute NELBO down slightly; the
   gap is cap-robust.
3. **Single seed, single scale:** one seed, one model (1.5B), one corpus slice.
4. **Mechanism unverified** (§3.3 is a hypothesis).
5. **Scope:** "discriminative equivalence" covers four MC suites under AR scoring at
   1.5B — not all discriminative tasks or scales.

### 7.2 Future Work

Multi-seed replication; 3B/7B scale; direct mechanism experiments (corruption-budget
accounting by token class); generative-factuality benchmarks as community resources;
composing token-selection gating with schedule-weighting methods (independent axes).

## 8. Conclusion

In a matched ablation isolating the training-time token-corruption distribution, the
choice between uniform masking and a structural-frequency product gate left AR
discriminative quality unchanged (all benchmark differences within noise) while
transforming generative quality: gold-match 95% vs 12.5%, period collapse 2.5% vs
82.5%, LAMBADA z ≈ 5, diffusion NELBO z ≈ 10. The training mask distribution is a
generative-quality lever — and its effects are invisible to the discriminative
evaluations standard suites provide. MDLM fine-tuning recipes should treat token
selection as a first-class design axis alongside schedule and parameterization.

## References

1. J. Austin et al. "Structured Denoising Diffusion Models in Discrete State-Spaces." NeurIPS 2021.
2. M. Arriola, Z. Qi, A. K. Gokaslan, J. Han, J. T. Chiu, S. S. Sahoo, Z. Yang, V. Kuleshov. "Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models." ICLR 2025. arXiv:2503.09573.
3. M. Shing, M. Koyama, T. Akiba. "DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation." ICLR 2026. arXiv:2506.14202.
4. Multi-Mask Diffusion Language Models. arXiv:2607.19686 (concurrent).
5. Context-Weighted Discrete Flow Matching for Masked Language Generation. arXiv:2607.21427 (concurrent).
6. Mean-to-Score: Score Realizability in Discrete Diffusion. arXiv:2607.21372 (concurrent).
7. DeepMind. "DiffusionGemma." Technical report, June 2026.
8. J. Devlin et al. "BERT." NAACL 2019.
9. X. Han, S. Kumar, Y. Tsvetkov. "SSD-LM." ACL 2022.
10. M. Joshi et al. "SpanBERT." TACL 2020.
11. C. Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR 2020.
12. S. S. Sahoo et al. "Simple and Effective Masked Diffusion Language Models." NeurIPS 2024.
13. A. Yang et al. "Qwen2 Technical Report." arXiv:2407.10671, 2024.
14. P. Zhang et al. "TinyLlama." arXiv:2401.02385, 2024.
15. M. Van Puyvelde et al. "Discrete Diffusion LMs for Interactive Radiology Report Drafting." arXiv:2607.01436, 2026.
16. Y. Go et al. "Enabling KV Caching of Shared Prefix for Diffusion Language Models." arXiv:2606.07571, 2026.

---

**Code:** https://github.com/Industrial-Algebra/mdlm-masking
**v1 (July 2026):** DOI 10.5281/zenodo.24508860 (superseded by this revision).
