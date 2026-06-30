#!/usr/bin/env python3
"""MDLM (Masked Diffusion Language Model) fine-tuning trainer for Quantizon Phase 2.

This trainer implements absorbing-state masked diffusion fine-tuning on a
pretrained autoregressive language model. Unlike the previous toy masked-denoiser
that trained a tiny encoder from scratch on teacher traces, this trainer:

- loads a real pretrained AR model (e.g. Qwen 0.5B),
- fine-tunes it to predict randomly masked tokens using the causal-MDLM objective,
- trains on real text corpora (C4, WikiText, etc.),
- saves a standard Hugging Face-compatible checkpoint.

The key insight: the causal LM already knows the language. We just teach it to
predict tokens from corrupted left context (absorbing-state diffusion), which is
a much smaller lift than training a denoiser from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from torch.utils.data import DataLoader
except ModuleNotFoundError:
    raise SystemExit("mdlm_trainer requires PyTorch. Install with: pip install torch")

try:
    import datasets as hf_datasets
except ModuleNotFoundError:
    raise SystemExit("mdlm_trainer requires HuggingFace datasets. Install with: pip install datasets")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    raise SystemExit("mdlm_trainer requires HuggingFace transformers. Install with: pip install transformers")

try:
    import tqdm
except ModuleNotFoundError:
    tqdm = None  # type: ignore[assignment]


MASK_TOKEN_TEXT = "<|quantizon_mask|>"
README_FILENAME = "README.md"
CHECKPOINT_DIRNAME = "checkpoint"
CONFIG_FILENAME = "training-config.json"
SUMMARY_FILENAME = "training-summary.json"
METRICS_FILENAME = "training-metrics.jsonl"
ARTIFACT_INTERFACE_FILENAME = "quantizon-artifact-interface.json"
MDLM_ARTIFACT_INTERFACE_PROFILE = "mdlm_checkpoint"
MDLM_ARTIFACT_OPERATION = "mdlm_fine_tuning"


# ── core corruption logic ──────────────────────────────────────────────


def corrupt_sequence(
    token_ids: list[int],
    mask_ratio: float,
    mask_token_id: int,
    rng: random.Random,
    skip_token_ids: frozenset[int] | None = None,
    token_weights: list[float] | None = None,
) -> tuple[list[int], list[int]]:
    """Mask tokens with min-based Bayesian gating.

    For each token t, the mask probability is:
        p_mask(t) = mask_ratio × min(skip_gate(t), freq_gate(t))

    where:
        skip_gate(t) = 0 if t ∈ skip_ids, else 1  (structural: never mask punct)
        freq_gate(t) = w[t]  (frequency: common words mask less)

    The min operation requires both gates to allow masking.

    Returns (corrupted_token_ids, masked_positions).
    """
    if mask_ratio <= 0.0:
        return list(token_ids), []
    corrupted: list[int] = []
    positions: list[int] = []
    for idx, tid in enumerate(token_ids):
        # Structural gate: skip punctuation/special tokens
        if skip_token_ids and tid in skip_token_ids:
            corrupted.append(tid)
            continue
        # Frequency gate: common tokens get lower weight
        freq_weight = token_weights[tid] if token_weights and tid < len(token_weights) else 1.0
        # Min gate: effective mask probability
        effective_p = mask_ratio * freq_weight
        if rng.random() < effective_p:
            corrupted.append(mask_token_id)
            positions.append(idx)
        else:
            corrupted.append(tid)
    return corrupted, positions


def corrupt_blocks(
    token_ids: list[int],
    block_size: int,
    mask_ratio: float,
    mask_token_id: int,
    rng: random.Random,
    skip_token_ids: frozenset[int] | None = None,
    token_weights: list[float] | None = None,
) -> tuple[list[int], list[int]]:
    """Mask entire contiguous blocks of tokens.

    Partitions the sequence into blocks of `block_size` tokens. For each block,
    applies the Bayesian gate: if any token in the block is eligible (not
    punctuation, not too common), the block may be masked as a unit.

    Block-gate: mask_block = (rng.random() < mask_ratio) AND
                              (any token in block passes Bayesian gate)

    Returns (corrupted_token_ids, masked_positions).
    """
    if mask_ratio <= 0.0:
        return list(token_ids), []

    corrupted = list(token_ids)
    positions: list[int] = []
    total_len = len(token_ids)
    n_blocks = (total_len + block_size - 1) // block_size

    for block_idx in range(n_blocks):
        start = block_idx * block_size
        end = min(start + block_size, total_len)
        block_tokens = token_ids[start:end]

        # Bayesian block gate: does any token pass?
        any_eligible = False
        for tid in block_tokens:
            if skip_token_ids and tid in skip_token_ids:
                continue
            any_eligible = True
            break

        if not any_eligible:
            continue

        if rng.random() < mask_ratio:
            for i in range(start, end):
                corrupted[i] = mask_token_id
                positions.append(i)

    return corrupted, positions


def build_skip_token_ids(tokenizer: Any) -> frozenset[int]:
    """Return the set of token IDs that should never be masked.

    Includes punctuation, whitespace, and special tokens.
    """
    skip_texts = {
        ".", ",", "?", "!", ";", ":", "(", ")", "[", "]", "{", "}",
        '"', "'", "-", "–", "—", "…", "`", "\n", " ", "\t", "\r",
        "\n\n", "\n ",
    }
    skip_ids: set[int] = set()
    for text in skip_texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        skip_ids.update(ids)
    # Include any token that starts with punctuation
    for tid in range(min(200, len(tokenizer))):
        decoded = tokenizer.decode([tid])
        stripped = decoded.strip()
        if not stripped or stripped[0] in (".", ",", "?", "!", ";", ":", "(", "[", "{", '"', "'", "-"):
            skip_ids.add(tid)
    # Also skip special tokens and the mask token itself
    skip_ids.update(tokenizer.all_special_ids)
    if tokenizer.mask_token_id is not None:
        skip_ids.add(tokenizer.mask_token_id)
    return frozenset(skip_ids)


def build_token_weights(
    tokenizer: Any,
    local_dataset_path: str | None,
    vocab_size: int,
) -> list[float]:
    """Compute inverse-frequency weights for each token ID.

    w(t) = 1 / (1 + log(count(t) + 1))

    Rare tokens get weights near 1.0; common tokens get weights near 0.
    These weights act as the frequency gate in the min-based Bayesian mask.

    When local_dataset_path is provided, frequencies are estimated from
    that dataset. Otherwise all tokens get uniform weight 1.0 (no frequency
    gating).
    """
    weights = [1.0] * vocab_size
    if local_dataset_path is None:
        return weights

    import json as _json
    from pathlib import Path as _Path
    data_path = _Path(local_dataset_path)
    if not data_path.exists():
        return weights

    counts: dict[int, int] = {}
    sample_lines = 50000
    for line_no, line in enumerate(data_path.read_text(encoding="utf-8").strip().splitlines()):
        if line_no >= sample_lines:
            break
        row = _json.loads(line)
        text = row.get("text", "")
        ids = tokenizer.encode(text, add_special_tokens=False)
        for tid in ids:
            if tid < vocab_size:
                counts[tid] = counts.get(tid, 0) + 1

    if not counts:
        return weights

    import math as _math
    for tid in range(vocab_size):
        c = counts.get(tid, 0)
        weights[tid] = 1.0 / (1.0 + _math.log(c + 1))

    return weights


# ── training helpers ───────────────────────────────────────────────────


def collate_batch(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    mask_ratio: float,
    mask_schedule: str,
    mask_token_id: int,
    pad_token_id: int,
    skip_token_ids: frozenset[int],
    token_weights: list[float],
    rng: random.Random,
) -> dict[str, Any]:
    """Tokenize a batch of text examples, corrupt them using Bayesian mask, return tensors."""
    texts = [example["text"] for example in batch]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Corrupt each sequence independently
    corrupted_ids = input_ids.clone()
    label_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    label_ids[:] = -100

    for b in range(batch_size):
        token_ids = input_ids[b].tolist()
        effective_mask_ratio = mask_ratio
        if mask_schedule == "cosine":
            effective_mask_ratio = sample_cosine_mask_ratio(rng)
        corrupted_list, positions = corrupt_sequence(
            token_ids, effective_mask_ratio, mask_token_id, rng,
            skip_token_ids, token_weights,
        )


def collate_batch_blocks(
    batch: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    block_size: int,
    mask_ratio: float,
    mask_schedule: str,
    mask_token_id: int,
    pad_token_id: int,
    skip_token_ids: frozenset[int],
    token_weights: list[float],
    rng: random.Random,
) -> dict[str, Any]:
    """Tokenize and corrupt using block-structured masking."""
    texts = [example["text"] for example in batch]
    encoded = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    corrupted_ids = input_ids.clone()
    label_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    label_ids[:] = -100

    for b in range(batch_size):
        token_ids = input_ids[b].tolist()
        effective_mask_ratio = mask_ratio
        if mask_schedule == "cosine":
            effective_mask_ratio = sample_cosine_mask_ratio(rng)
        corrupted_list, positions = corrupt_blocks(
            token_ids, block_size, effective_mask_ratio, mask_token_id, rng,
            skip_token_ids, token_weights,
        )
        valid_positions = [p for p in positions
                          if p < (attention_mask[b].sum().item())]
        for p in positions:
            corrupted_ids[b, p] = mask_token_id
        for p in valid_positions:
            label_ids[b, p] = token_ids[p]

    return {"input_ids": corrupted_ids, "attention_mask": attention_mask, "labels": label_ids}
def compute_mdlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Compute cross-entropy loss on non-ignored positions.

    Uses -100 as ignore index (HuggingFace convention for padding/masked tokens).
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def bidirectional_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Any:
    """Run the model with bidirectional (non-causal) attention."""
    batch_size, seq_len = input_ids.shape
    model_dtype = next(model.parameters()).dtype
    mask_4d = torch.zeros(batch_size, 1, seq_len, seq_len, device=input_ids.device, dtype=model_dtype)
    padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
    mask_4d = mask_4d.masked_fill(padding_mask, float("-inf"))
    return model(input_ids=input_ids, attention_mask=mask_4d, use_cache=False)


def build_training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "mask_schedule": args.mask_schedule,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def load_local_dataset(path: str, seed: int, rank: int = 0, world_size: int = 1) -> list[dict[str, Any]]:
    """Load a local JSONL dataset file into memory with shuffling."""
    data_path = Path(path)
    if not data_path.exists():
        raise SystemExit(f"local dataset not found: {data_path}")
    examples: list[dict[str, Any]] = []
    for line in data_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("text", "").strip():
            examples.append(row)
    rng = random.Random(seed)
    rng.shuffle(examples)
    # Shard across ranks if distributed
    if world_size > 1:
        examples = examples[rank::world_size]
    print(f"  loaded {len(examples)} examples" + (f" (rank {rank}/{world_size} shard)" if world_size > 1 else ""))
    return examples


def cycle_dataset(examples: list[dict[str, Any]], seed: int):
    """Yield examples in a cycle, reshuffling each epoch."""
    if not examples:
        raise SystemExit("empty dataset")
    rng = random.Random(seed)
    while True:
        rng.shuffle(examples)
        yield from examples


# ── cosine mask schedule ─────────────────────────────────────────────


def cosine_mask_ratio_at(t: float) -> float:
    """Cosine mask schedule at normalised time t in [0, 1].

    Returns the probability of masking a token at timestep t.
    0 = no corruption, 1 = all masked.
    """
    return 1.0 - math.cos(t * math.pi / 2.0)


def sample_cosine_mask_ratio(rng: random.Random) -> float:
    """Sample a mask ratio by drawing t ~ Uniform(0, 1) and applying the cosine schedule."""
    t = rng.random()
    return cosine_mask_ratio_at(t)


def build_mdlm_artifact_id(model: str) -> str:
    """Derive a stable Quantizon artifact ID from the model name."""
    # Strip redundant prefix like "qwen-" when model is "Qwen/Qwen2-0.5B-Instruct"
    slug = slugify(model)
    # If slug starts with the org name and then repeats it, collapse
    parts = slug.split("-")
    if len(parts) >= 2 and parts[0] == parts[1]:
        slug = "-".join(parts[1:])
    return f"artifact.{slug}.mdlm-phase2"


def build_registered_artifact_documents(
    model: str,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Build Quantizon artifact interface and manifest documents for this checkpoint."""
    trained_artifact_id = build_mdlm_artifact_id(model)
    # Resolve workspace root from output_dir
    parts = output_dir.resolve().parts
    workspace_root = Path.cwd()
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        workspace_root = Path(*parts[:idx])
    manifest_root = workspace_root / "artifacts" / "manifests" / "generated"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{slugify(trained_artifact_id)}.json"

    interface_payload = {
        "schema_version": 1,
        "artifact_id": trained_artifact_id,
        "driver_mode": "trainer",
        "evaluation_interface": {
            "profile": MDLM_ARTIFACT_INTERFACE_PROFILE,
            "checkpoint_dir": str(output_dir / CHECKPOINT_DIRNAME),
            "sampling_backend": "mdlm_sampler",
            "notes": [
                "MDLM absorbing-state diffusion fine-tuning checkpoint.",
                "Sampling uses iterative unmasking via scripts/mdlm_sampler.py.",
                "This is a causal LM fine-tuned for bidirectional masked-token prediction.",
            ],
        },
        "lineage": {
            "base_model": model,
            "operation": MDLM_ARTIFACT_OPERATION,
        },
    }

    manifest_payload = {
        "manifest_kind": "model_artifact",
        "spec": {
            "schema_version": {"major": 1, "minor": 0},
            "manifest_id": f"manifest.{trained_artifact_id}",
            "name": f"MDLM fine-tuned {model}",
            "artifact": {
                "artifact_id": trained_artifact_id,
                "family": "diffusion",
                "weight_compression": {"kind": "uncompressed"},
                "kv_compression": {"kind": "untracked"},
            },
            "lineage": {
                "stage": "training",
                "operation": MDLM_ARTIFACT_OPERATION,
            },
            "provenance": {
                "backend": {
                    "name": "mdlm-trainer",
                    "version": "0.1.0",
                    "entrypoint": "python scripts/mdlm_trainer.py",
                },
                "source_uri": model,
            },
            "storage": {
                "location": str(output_dir),
            },
            "tags": [
                "phase2",
                "training",
                "mdlm",
                "diffusion",
                "fine_tuned",
                "mdlm_checkpoint",
                "bidirectional_attention",
            ],
        },
    }
    return interface_payload, manifest_payload, manifest_path


def write_registered_artifact_documents(
    model: str,
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Write artifact interface and manifest files to disk."""
    interface_payload, manifest_payload, manifest_path = build_registered_artifact_documents(
        model, output_dir, config,
    )
    (output_dir / ARTIFACT_INTERFACE_FILENAME).write_text(
        json.dumps(interface_payload, indent=2),
    )
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))
    return {
        "artifact_id": interface_payload["artifact_id"],
        "manifest_path": str(manifest_path),
        "interface_path": str(output_dir / ARTIFACT_INTERFACE_FILENAME),
        "evaluation_profile": MDLM_ARTIFACT_INTERFACE_PROFILE,
    }


class _TeeWriter:
    """Writes to two streams simultaneously."""
    def __init__(self, stream1, stream2):
        self._s1 = stream1
        self._s2 = stream2
    def write(self, data):
        self._s1.write(data)
        self._s2.write(data)
    def flush(self):
        self._s1.flush()
        self._s2.flush()


# ── main training loop ─────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    # Set up log file if requested
    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        # Tee stdout to log file
        import contextlib
        sys.stdout = _TeeWriter(sys.stdout, log_handle)
        sys.stderr = _TeeWriter(sys.stderr, log_handle)

    rng = random.Random(args.seed)
    torch_manual_seed = args.seed
    torch.manual_seed(torch_manual_seed)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── distributed setup ──────────────────────────────────────────────
    is_distributed = args.distributed
    rank = 0
    world_size = 1
    if is_distributed:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        if rank == 0:
            print(f"distributed: rank={rank}/{world_size}")

    print(f"device: {device}")

    # Load model and tokenizer
    print(f"loading model {args.model!r}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.to(device)

    # Compile for speed if requested (~1.3-1.5x throughput)
    if getattr(args, 'compile', False):
        print("compiling model (first forward pass will be slow)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Ensure mask token exists BEFORE DDP wrap (DDP can't resize embeddings)
    mask_token_id = _ensure_mask_token(tokenizer, model)

    # Wrap in DDP if distributed
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index] if device.index is not None else None)

    # Verify pad token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    pad_token_id = tokenizer.pad_token_id

    # Pre-compute token IDs to skip during masking (punctuation, special tokens)
    skip_token_ids = build_skip_token_ids(tokenizer)
    # Pre-compute inverse-frequency weights for Bayesian min-gate masking
    token_weights = build_token_weights(
        tokenizer, getattr(args, 'local_dataset', None), len(tokenizer),
    )
    if getattr(args, 'no_bayesian_mask', False):
        skip_token_ids = frozenset()
        token_weights = [1.0] * len(tokenizer)

    # Load dataset
    if args.local_dataset:
        print(f"loading local dataset {args.local_dataset!r}...")
        dataset = load_local_dataset(args.local_dataset, args.seed, rank, world_size)
    else:
        print(f"loading dataset {args.dataset_name!r}...")
        dataset = hf_datasets.load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            streaming=True,
            trust_remote_code=True,
        )
        # Shuffle with buffer (no map needed — WikiText already has a "text" column)
        dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    num_training_steps = args.max_steps if args.max_steps > 0 else args.epochs * 500
    # OneCycleLR needs enough steps for its phase structure
    effective_steps = max(num_training_steps, 100)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=effective_steps,
        pct_start=0.1,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume
    epoch_metrics: list[dict[str, Any]] = []
    global_step = 0
    best_loss: float | None = None
    total_tokens = 0
    start_epoch = 1

    if args.resume:
        resume_dir = Path(args.resume)
        resume_summary_path = resume_dir / SUMMARY_FILENAME
        if not resume_summary_path.exists():
            raise SystemExit(f"no training summary found in resume dir: {resume_dir}")
        with open(resume_summary_path) as f:
            resume_summary = json.load(f)
        global_step = resume_summary.get("global_steps", 0)
        best_loss = resume_summary.get("best_loss")
        total_tokens = resume_summary.get("total_masked_tokens", 0)
        epoch_metrics = resume_summary.get("epoch_metrics", [])
        start_epoch = len(epoch_metrics) + 1
        # Load optimizer and scheduler state if available
        opt_path = resume_dir / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        sched_path = resume_dir / "scheduler.pt"
        if sched_path.exists():
            scheduler.load_state_dict(torch.load(sched_path, map_location=device))
        print(f"resumed from step {global_step}, starting epoch {start_epoch}")

    if args.overwrite and not args.resume:
        for stale in output_dir.glob("*"):
            if stale.is_file():
                stale.unlink()

    config = build_training_config(args)
    if rank == 0:
        (output_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2))

    epoch_metrics: list[dict[str, Any]] = []
    global_step = 0
    best_loss: float | None = None
    total_tokens = 0

    model.train()
    accumulation_steps = max(1, getattr(args, 'gradient_accumulation_steps', 1))
    accum_step = 0

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_tokens = 0
        epoch_nan_steps = 0

        # Use cycling iterator for local datasets, streaming for HF
        if isinstance(dataset, list):
            data_iter = cycle_dataset(dataset, args.seed + epoch)
        else:
            data_iter = dataset

        def batch_iterator():
            batch_buffer: list[dict[str, Any]] = []
            use_blocks = getattr(args, 'block_size', 0) > 0
            for example in data_iter:
                batch_buffer.append(example)
                if len(batch_buffer) >= args.batch_size:
                    if use_blocks:
                        yield collate_batch_blocks(
                            batch_buffer, tokenizer, args.max_length,
                            args.block_size, args.mask_ratio, args.mask_schedule,
                            mask_token_id, pad_token_id, skip_token_ids, token_weights, rng,
                        )
                    else:
                        yield collate_batch(
                            batch_buffer, tokenizer, args.max_length,
                            args.mask_ratio, args.mask_schedule, mask_token_id, pad_token_id,
                            skip_token_ids, token_weights, rng,
                        )
                    batch_buffer = []
                    if args.max_steps > 0 and epoch_steps >= args.max_steps // max(args.epochs, 1):
                        break

        iterator = batch_iterator()
        if tqdm is not None:
            iterator = tqdm.tqdm(iterator, desc=f"epoch {epoch}/{args.epochs}", unit="batch")

        for batch in iterator:
            batch = {k: v.to(device) for k, v in batch.items()}

            if getattr(args, 'bidirectional', False):
                outputs = bidirectional_forward(
                    model, batch["input_ids"], batch["attention_mask"],
                )
                loss = compute_mdlm_loss(outputs.logits, batch["labels"], mask_token_id)
            else:
                outputs = model(**batch)
                loss = outputs.loss
                if loss is None:
                    loss = compute_mdlm_loss(outputs.logits, batch["labels"], mask_token_id)

            # Skip batches with NaN loss (numeric instability in some text examples)
            if torch.isnan(loss):
                epoch_nan_steps += 1
                if tqdm is not None and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss="nan (skipped)")  # type: ignore[union-attr]
                continue

            global_step += 1
            epoch_steps += 1

            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            accum_step += 1

            if accum_step >= accumulation_steps:
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_step = 0

            epoch_loss += loss.item()
            epoch_tokens += batch["labels"].ne(-100).sum().item()

            if tqdm is not None:
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(loss=f"{loss.item():.4f}")  # type: ignore[union-attr]

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        avg_loss = epoch_loss / max(epoch_steps, 1)
        total_tokens += epoch_tokens
        print(
            f"epoch {epoch} | loss={avg_loss:.4f} | "
            f"steps={epoch_steps} | tokens={epoch_tokens} | lr={scheduler.get_last_lr()[0]:.2e}"
            + (f" | nan_skipped={epoch_nan_steps}" if epoch_nan_steps > 0 else "")
        )

        epoch_metrics.append({
            "epoch": epoch,
            "train_mean_loss": avg_loss,
            "train_steps": epoch_steps,
            "train_masked_tokens": epoch_tokens,
            "learning_rate": scheduler.get_last_lr()[0],
        })

        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            if rank == 0:
                _save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir / CHECKPOINT_DIRNAME)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save final checkpoint if no best was saved (rank 0 only)
    checkpoint_dir = output_dir / CHECKPOINT_DIRNAME
    if rank == 0 and not checkpoint_dir.exists():
        _save_checkpoint(model, tokenizer, optimizer, scheduler, checkpoint_dir)

    # Write summary
    summary = {
        "trainer": "mdlm_trainer",
        "model": args.model,
        "dataset": args.dataset_name,
        "epochs": args.epochs,
        "global_steps": global_step,
        "total_masked_tokens": total_tokens,
        "final_mean_loss": epoch_metrics[-1]["train_mean_loss"] if epoch_metrics else None,
        "best_loss": best_loss,
        "output_dir": str(output_dir),
        "config": config,
        "epoch_metrics": epoch_metrics,
        "notes": [
            "MDLM absorbing-state diffusion fine-tuning checkpoint.",
            "Trained to predict masked tokens with bidirectional attention.",
            "Use scripts/mdlm_sampler.py for iterative diffusion sampling.",
        ] + (["distributed training across multiple GPUs"] if is_distributed else []),
    }
    if rank == 0:
        (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))

        # Write metrics JSONL
        metrics_lines = [json.dumps(em) for em in epoch_metrics]
        (output_dir / METRICS_FILENAME).write_text("\n".join(metrics_lines) + "\n")

        # Write README
        readme = (
            f"# MDLM Fine-Tuning Checkpoint\n\n"
            f"Model: `{args.model}`\n"
            f"Dataset: `{args.dataset_name}`\n"
            f"Epochs: {args.epochs}\n"
            f"Steps: {global_step}\n"
            f"Final loss: {epoch_metrics[-1]['train_mean_loss']:.4f}\n\n"
            f"## Sampling\n\n"
            f"```bash\n"
            f"python scripts/mdlm_sampler.py \\\n"
            f"    --checkpoint {output_dir / CHECKPOINT_DIRNAME} \\\n"
            f'    --prompt "Your prompt here"\n'
            f"```\n"
        )
        (output_dir / README_FILENAME).write_text(readme)

        # Register as Quantizon artifact
        registered = write_registered_artifact_documents(args.model, output_dir, config)
        summary["registered_artifact"] = registered
        (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, indent=2))

        print(f"artifact registered as {registered['artifact_id']}")
        print(f"manifest written to {registered['manifest_path']}")

    if rank == 0:
        print(f"training complete: {global_step} steps, best loss {best_loss:.4f}")
        print(f"checkpoint saved to {output_dir / CHECKPOINT_DIRNAME}")

    if log_handle:
        log_handle.close()


# ── helpers ────────────────────────────────────────────────────────────


def _ensure_mask_token(tokenizer: Any, model: Any) -> int:
    """Ensure the tokenizer has a mask token and the model's embedding supports it."""
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id

    mask_text = MASK_TOKEN_TEXT
    if mask_text in tokenizer.get_vocab():
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_text)
        return tokenizer.mask_token_id

    num_added = tokenizer.add_tokens([mask_text], special_tokens=True)
    if num_added > 0:
        base_model = model.module if hasattr(model, "module") else model
        base_model.resize_token_embeddings(len(tokenizer))
        # Initialize the new mask token embedding to a reasonable value
        embed_weight = base_model.get_input_embeddings().weight
        with torch.no_grad():
            embed_weight[-num_added:] = embed_weight[:-num_added].mean(dim=0)
    tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_text)
    return tokenizer.mask_token_id


def _save_checkpoint(
    model: Any, tokenizer: Any, optimizer: Any, scheduler: Any, checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_model = model.module if hasattr(model, "module") else model
    base_model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    # Save optimizer and scheduler state for resumption
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MDLM absorbing-state diffusion fine-tuning trainer"
    )
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--dataset-name", default="allenai/c4")
    parser.add_argument("--dataset-config", default="en")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--local-dataset", default=None,
                        help="path to local JSONL dataset (bypasses HF streaming)")
    parser.add_argument("--resume", default=None,
                        help="path to previous training output dir to resume from")
    parser.add_argument("--log-file", default=None,
                        help="write output to log file (in addition to stdout)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-ratio", type=float, default=0.15,
                        help="mask token probability (used when mask-schedule=fixed)")
    parser.add_argument("--mask-schedule", default="cosine",
                        choices=["fixed", "cosine"],
                        help="mask schedule: fixed ratio or cosine noise schedule (default: cosine)")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="accumulate gradients over N batches before stepping (default: 1)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="max total steps (0 = unlimited)")
    parser.add_argument("--distributed", action="store_true",
                        help="enable multi-GPU distributed training via torchrun")
    parser.add_argument("--no-bayesian-mask", action="store_true",
                        help="disable Bayesian mask (use flat uniform masking)")
    parser.add_argument("--bidirectional", action="store_true",
                        help="train with bidirectional attention (4D mask)")
    parser.add_argument("--block-size", type=int, default=0,
                        help="block diffusion: mask contiguous blocks of N tokens (0 = token-level)")
    parser.add_argument("--compile", action="store_true",
                        help="use torch.compile() for ~1.3-1.5x speedup")
    parser.add_argument("--output-dir", default="artifacts/generated/mdlm-training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
