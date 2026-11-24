# Don't Mask That Period: Structural-Frequency Masking for Causal Masked Diffusion Language Models

Reproduction code for the Quantizon research paper: **"Don't Mask That Period: Structural-Frequency Masking for Causal Masked Diffusion Language Models"**

> **Paper (preprint, DOI):** [10.5281/zenodo.24508860](https://doi.org/10.5281/zenodo.24508860)

This repository contains the scripts and datasets needed to reproduce the
key results: fine-tuning a pretrained autoregressive language model as a
masked diffusion language model (MDLM) using min-based Bayesian gating
to prevent the uniform mask collapse.

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets tqdm

# Download training data (C4 subset)
./scripts/download_c4_5m.sh

# Train a Bayesian-masked MDLM from a pretrained AR model
python scripts/mdlm_trainer.py \
    --model-name Qwen/Qwen2.5-0.5B-Instruct \
    --bayesian-mask \
    --bidirectional \
    --epochs 1 \
    --batch-size 4 \
    --output-dir artifacts/mdlm-checkpoint

# Sample from the trained model
python scripts/mdlm_sampler.py \
    --checkpoint artifacts/mdlm-checkpoint/checkpoint \
    --prompt "The capital of France is" \
    --block-size 8 \
    --num-steps 8
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `mdlm_trainer.py` | Fine-tune AR model as MDLM with Bayesian masking |
| `mdlm_sampler.py` | Sample from MDLM (token-level, block-level, block-acceptance) |
| `mdlm_eval_backend.py` | Evaluation backend for 50-prompt benchmark |

## Training Modes

`mdlm_trainer.py` supports the configurations described in the paper:

- `--bayesian-mask`: Enable min-based Bayesian gating (structural + frequency gates)
- `--bidirectional`: Train with bidirectional attention (recommended)
- `--mask-schedule cosine`: Cosine noise schedule for mask ratio
- `--num-mask-tokens 1`: Number of mask tokens to add to tokenizer

## Sampling Modes

`mdlm_sampler.py` supports three sampling modes:

- `--mode token`: Token-level MDLM sampling (fastest, best quality)
- `--mode block`: Block-level parallel sampling
- `--mode critic`: Block sampling with acceptance critic and retry

## Hardware Requirements

- Training: 1× GPU with ≥ 16 GB VRAM (tested on A100, GB10)
- Sampling: 1× GPU with ≥ 8 GB VRAM
- Distributed training: 2+ GPUs with NCCL (optional, `--distributed`)

## Reproducing Paper Results

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download C4 data: `./scripts/download_c4_5m.sh`
4. Train the Bayesian bidir MDLM as described in §4-5
5. Evaluate on the 50-prompt set: `python scripts/mdlm_eval_backend.py`

The key finding: uniform random masking (the standard BERT/T5 approach)
catastrophically fails in causal-MDLM fine-tuning, producing only periods.
The min-based Bayesian gating function eliminates this failure.

## Citation

See [CITATION.cff](CITATION.cff).

## License

Apache-2.0. See [LICENSE](LICENSE).

## Originating Project

This repository is a publication snapshot spawned from
[Quantizon](https://github.com/Industrial-Algebra/Quantizon),
a research laboratory for diffusion language models.
