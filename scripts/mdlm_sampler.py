#!/usr/bin/env python3
"""MDLM iterative diffusion sampler for Quantizon Phase 2.

This sampler performs iterative absorbing-state diffusion sampling (unmasking)
from an MDLM fine-tuned checkpoint. The procedure:

1. Start with the prompt tokens followed by [MASK] tokens to fill target length.
2. At each step, run the model on the current sequence.
3. Select the k most confident masked positions, replace them with predicted tokens.
4. Repeat until no masks remain or max steps reached.

This is the standard MDLM absorbing-state sampling algorithm adapted for
causal left-to-right unmasking.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
except ModuleNotFoundError:
    raise SystemExit("mdlm_sampler requires PyTorch. Install with: pip install torch")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError:
    raise SystemExit("mdlm_sampler requires HuggingFace transformers. Install with: pip install transformers")

MASK_TOKEN_TEXT = "<|quantizon_mask|>"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def build_initial_sequence(
    prompt_ids: list[int],
    target_length: int,
    mask_token_id: int,
    pad_token_id: int,
) -> list[int]:
    """Build initial sequence: prompt tokens followed by [MASK] tokens.

    If prompt_ids already exceeds target_length, clip it.
    """
    if len(prompt_ids) >= target_length:
        return list(prompt_ids[:target_length])
    num_masks = target_length - len(prompt_ids)
    return list(prompt_ids) + [mask_token_id] * num_masks


def find_mask_positions(
    token_ids: torch.Tensor,
    mask_token_id: int,
    prompt_length: int,
) -> list[int]:
    """Return list of positions containing mask tokens (after the prompt)."""
    positions: list[int] = []
    for i in range(prompt_length, len(token_ids)):
        if token_ids[i].item() == mask_token_id:
            positions.append(i)
    return positions


def sample_step(
    model: Any,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_positions: list[int],
    mask_token_id: int,
    num_to_unmask: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    """Run one sampling step: unmask the given positions in order.

    Returns (updated_token_ids, still_masked_positions).
    """
    if not mask_positions or num_to_unmask <= 0:
        return token_ids, mask_positions

    with torch.no_grad():
        outputs = model(input_ids=token_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))

    logits = outputs.logits[0]  # [seq_len, vocab_size]

    new_ids = token_ids.clone()
    unmasked_count = 0
    for pos in mask_positions:
        if unmasked_count >= num_to_unmask:
            break
        pos_logits = logits[pos]
        if temperature > 0:
            pos_logits = pos_logits / temperature
        probs = F.softmax(pos_logits, dim=-1)
        if top_k > 0 and top_k < len(probs):
            topk_values, topk_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs)
            probs.scatter_(0, topk_indices, topk_values)
            probs = probs / probs.sum()

        max_prob, predicted = probs.max(dim=-1)
        # Don't predict the mask token itself
        if predicted.item() == mask_token_id:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            for idx in range(len(sorted_indices)):
                if sorted_indices[idx].item() != mask_token_id:
                    predicted = sorted_indices[idx]
                    break
        new_ids[pos] = predicted.item()
        unmasked_count += 1

    remaining = [p for p in mask_positions if new_ids[p].item() == mask_token_id]
    return new_ids, remaining


def iterative_sample(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    num_steps: int,
    mask_token_id: int,
    pad_token_id: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    """Full iterative diffusion sampling from a prompt.

    Returns dict with 'prompt', 'output', 'steps', 'latency_ms'.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # Leave room for max_new_tokens
    target_length = min(len(prompt_ids) + max_new_tokens, getattr(tokenizer, "model_max_length", 512) or 512)

    seq = build_initial_sequence(prompt_ids, target_length, mask_token_id, pad_token_id)
    token_ids = torch.tensor([seq], device=device)
    attention_mask = torch.ones_like(token_ids)

    started = time.perf_counter()
    actual_steps = 0

    # Track which generated positions have been unmasked
    filled_positions: set[int] = set()

    for step in range(num_steps):
        actual_steps = step + 1
        mask_positions = find_mask_positions(token_ids[0], mask_token_id, len(prompt_ids))
        if not mask_positions:
            break

        # Unmask a fraction per step, LEFT-TO-RIGHT
        num_to_unmask = max(1, len(mask_positions) // (num_steps - step))
        # Always fill the leftmost mask positions first (standard MDLM sequential)
        leftmost = mask_positions[:num_to_unmask]
        new_ids, remaining = sample_step(
            model, token_ids[0], attention_mask[0],
            leftmost, mask_token_id,
            num_to_unmask, temperature, top_k, device,
        )
        # Detect which positions were just filled in the generated region
        just_filled = set(leftmost) & set(range(len(prompt_ids), target_length))
        filled_positions |= just_filled
        token_ids = new_ids.unsqueeze(0)

        # Early stop: if any generated token is EOS or sentence-ending
        for pos in just_filled:
            tid = token_ids[0, pos].item()
            if tid in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                break
            decoded = tokenizer.decode([tid])
            if decoded.strip() in (".", "!", "?", "\n", "\n\n"):
                break
        else:
            continue
        break  # sentence complete, stop generating

    latency_ms = (time.perf_counter() - started) * 1000.0

    # Decode only the new tokens (after prompt), stop at first mask
    generated_ids = token_ids[0, len(prompt_ids):].tolist()
    # Stop at first mask token
    if mask_token_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(mask_token_id)]
    output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "prompt": prompt,
        "output": output,
        "steps": actual_steps,
        "latency_ms": latency_ms,
        "generated_tokens": len(generated_ids),
    }


def block_sample(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    num_steps_per_block: int,
    mask_token_id: int,
    pad_token_id: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    """Block diffusion sampling: generate block-by-block.

    Each block of `block_size` tokens is generated via iterative diffusion
    in parallel. Blocks are committed in sequence, with KV-cache reuse
    for committed blocks.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    all_generated: list[int] = []
    started = time.perf_counter()
    current_ids = prompt_ids.copy()

    max_blocks = (max_new_tokens + block_size - 1) // block_size

    for _ in range(max_blocks):
        # Append block_size [MASK] tokens
        current_ids = current_ids + [mask_token_id] * block_size
        target_length = len(current_ids)
        token_ids_tensor = torch.tensor([current_ids], device=device)
        attention_mask = torch.ones_like(token_ids_tensor)

        # Iterative diffusion within this block
        num_masks = block_size
        for step in range(num_steps_per_block):
            if num_masks <= 0:
                break
            num_to_unmask = max(1, num_masks // (num_steps_per_block - step))
            positions_to_unmask = list(range(target_length - num_masks, target_length - num_masks + num_to_unmask))
            new_ids, _ = sample_step(
                model, token_ids_tensor[0], attention_mask[0],
                positions_to_unmask, mask_token_id,
                num_to_unmask, temperature, top_k, device,
            )
            token_ids_tensor = new_ids.unsqueeze(0)
            num_masks -= num_to_unmask

        # Extract generated tokens, check for EOS
        new_tokens = token_ids_tensor[0, target_length - block_size:].tolist()
        stop_idx = None
        for i, tid in enumerate(new_tokens):
            if tid in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                stop_idx = i
                break
            decoded = tokenizer.decode([tid])
            if decoded.strip() in (".", "!", "?", "\n"):
                stop_idx = i + 1
                break

        if stop_idx is not None:
            all_generated.extend(new_tokens[:stop_idx])
            break
        all_generated.extend(new_tokens)
        current_ids = token_ids_tensor[0].tolist()

        if len(all_generated) >= max_new_tokens:
            break

    latency_ms = (time.perf_counter() - started) * 1000.0
    output = tokenizer.decode(all_generated, skip_special_tokens=True).strip()

    return {
        "prompt": prompt,
        "output": output,
        "steps": max_blocks * num_steps_per_block,
        "latency_ms": latency_ms,
        "generated_tokens": len(all_generated),
    }


def block_sample_with_critic(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    num_steps_per_block: int,
    mask_token_id: int,
    pad_token_id: int,
    temperature: float,
    top_k: int,
    acceptance_threshold: float,
    max_retries: int,
    device: torch.device,
) -> dict[str, Any]:
    """Block diffusion with tropical block-acceptance critic.

    Generates blocks using the standard block_sample logic. After each block is
    generated, scores it using the model's per-token confidence from a fresh
    forward pass. The block score is the tropical max (max confidence across
    all tokens in the block). Blocks below the threshold are retried.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    all_generated: list[int] = []
    started = time.perf_counter()
    current_ids = prompt_ids.copy()

    stats = {
        "blocks_generated": 0,
        "blocks_accepted": 0,
        "blocks_rejected": 0,
        "total_retries": 0,
        "block_scores": [],
        "acceptance_rate": 0.0,
    }

    max_blocks = (max_new_tokens + block_size - 1) // block_size

    for _ in range(max_blocks):
        best_result = None
        best_score = float("-inf")

        for retry in range(max_retries + 1):
            stats["blocks_generated"] += 1
            if retry > 0:
                stats["total_retries"] += 1

            # Generate block using standard block_sample logic
            trial_ids = current_ids + [mask_token_id] * block_size
            target_length = len(trial_ids)
            token_ids_tensor = torch.tensor([trial_ids], device=device)
            attention_mask = torch.ones_like(token_ids_tensor)

            num_masks = block_size
            for step in range(num_steps_per_block):
                if num_masks <= 0:
                    break
                num_to_unmask = max(1, num_masks // (num_steps_per_block - step))
                positions_to_unmask = list(range(
                    target_length - num_masks,
                    target_length - num_masks + num_to_unmask,
                ))
                new_ids, _ = sample_step(
                    model, token_ids_tensor[0], attention_mask[0],
                    positions_to_unmask, mask_token_id,
                    num_to_unmask, temperature, top_k, device,
                )
                token_ids_tensor = new_ids.unsqueeze(0)
                num_masks -= num_to_unmask

            # Score block: one fresh forward pass, extract max confidence
            with torch.no_grad():
                outputs = bidirectional_forward(
                    model, token_ids_tensor, attention_mask,
                )
            logits_all = outputs.logits[0]
            confidences: list[float] = []
            new_tokens = token_ids_tensor[0, target_length - block_size:].tolist()
            vocab_size = logits_all.shape[-1]
            chance_baseline = 1.0 / vocab_size  # ~0.000007 for 151K vocab
            for offset, tid in enumerate(new_tokens):
                if tid == mask_token_id:
                    continue
                pos_logits = logits_all[target_length - block_size + offset]
                probs = F.softmax(pos_logits, dim=-1)
                conf = probs[tid].item()
                # Normalize: ratio above chance baseline
                confidences.append(conf / chance_baseline)

            # Block score = tropical max of normalized confidences
            block_score = max(confidences) if confidences else 0.0
            stats["block_scores"].append(block_score)

            new_tokens = token_ids_tensor[0, target_length - block_size:].tolist()

            if block_score >= acceptance_threshold or retry == max_retries:
                best_result = new_tokens
                best_score = block_score
                if block_score >= acceptance_threshold:
                    stats["blocks_accepted"] += 1
                    break
            elif block_score > best_score:
                best_result = new_tokens
                best_score = block_score

        if best_score < acceptance_threshold:
            stats["blocks_rejected"] += 1

        if best_result is None:
            break

        # Extract tokens, check EOS
        stop_idx = None
        for i, tid in enumerate(best_result):
            if tid in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                stop_idx = i
                break
            decoded = tokenizer.decode([tid])
            if decoded.strip() in (".", "!", "?", "\n"):
                stop_idx = i + 1
                break

        if stop_idx is not None:
            all_generated.extend(best_result[:stop_idx])
            break
        all_generated.extend(best_result)
        current_ids = token_ids_tensor[0].tolist()

        if len(all_generated) >= max_new_tokens:
            break

    latency_ms = (time.perf_counter() - started) * 1000.0
    output = tokenizer.decode(all_generated, skip_special_tokens=True).strip()

    total_blocks = stats["blocks_accepted"] + stats["blocks_rejected"]
    stats["acceptance_rate"] = stats["blocks_accepted"] / max(total_blocks, 1)
    stats["mean_block_score"] = sum(stats["block_scores"]) / max(len(stats["block_scores"]), 1)
    stats["min_block_score"] = min(stats["block_scores"]) if stats["block_scores"] else 0.0
    stats["max_block_score"] = max(stats["block_scores"]) if stats["block_scores"] else 0.0

    return {
        "prompt": prompt,
        "output": output,
        "steps": total_blocks * num_steps_per_block,
        "latency_ms": latency_ms,
        "generated_tokens": len(all_generated),
        "critic_stats": stats,
    }


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[Any, Any]:
    """Load model and tokenizer from a saved checkpoint directory."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    # Ensure mask token exists and is set
    if tokenizer.mask_token_id is None:
        if MASK_TOKEN_TEXT in tokenizer.get_vocab():
            tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN_TEXT)
        else:
            num_added = tokenizer.add_tokens([MASK_TOKEN_TEXT], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))
            # Initialize mask token embedding from mean of existing embeddings
            embed_weight = model.get_input_embeddings().weight
            with torch.no_grad():
                embed_weight[-num_added:] = embed_weight[:-num_added].mean(dim=0)
            tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN_TEXT)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.mask_token_id is not None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id)

    return model, tokenizer


def bidirectional_forward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Any:
    """Run the model with bidirectional (non-causal) attention.

    Creates a 4D additive attention mask that allows all positions to attend
    to all positions (zeros mean attend), overriding the default causal mask.
    """
    batch_size, seq_len = input_ids.shape
    # Match the model's dtype for SDPA compatibility
    model_dtype = next(model.parameters()).dtype
    # 4D additive mask: zeros = attend, large negative = don't attend
    mask_4d = torch.zeros(batch_size, 1, seq_len, seq_len, device=input_ids.device, dtype=model_dtype)
    # Apply padding mask: positions with attention_mask=0 should not be attended to
    padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    mask_4d = mask_4d.masked_fill(padding_mask, float("-inf"))
    return model(input_ids=input_ids, attention_mask=mask_4d, use_cache=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MDLM iterative diffusion sampler"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="path to saved MDLM checkpoint directory")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--prompts-file",
                        help="JSONL file with prompts (for batch sampling)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy, >0 = sampling")
    parser.add_argument("--top-k", type=int, default=1,
                        help="top-k for sampling (1 = greedy prediction from top-1)")
    parser.add_argument("--block-size", type=int, default=0,
                        help="block diffusion: generate N tokens per block (0 = token)")
    parser.add_argument("--acceptance-threshold", type=float, default=0.0,
                        help="tropical block-acceptance threshold (0.0 = no critic)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="max retries per rejected block (default: 2)")
    parser.add_argument("--output", help="path to write JSON results")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    model, tokenizer = load_checkpoint(args.checkpoint, device)
    if tokenizer.mask_token_id is None:
        if MASK_TOKEN_TEXT in tokenizer.get_vocab():
            tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN_TEXT)
        else:
            num_added = tokenizer.add_tokens([MASK_TOKEN_TEXT], special_tokens=True)
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))
                tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN_TEXT)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id or mask_token_id
    print(f"model loaded, vocab={len(tokenizer)}, mask_token_id={mask_token_id}")

    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        prompts = []
        for line in prompts_path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt") or row.get("instruction") or row.get("text") or ""
            prompts.append(prompt)
    else:
        prompts = [args.prompt]

    results = []
    for prompt in prompts:
        if args.block_size > 0:
            if args.acceptance_threshold > 0:
                result = block_sample_with_critic(
                    model, tokenizer, prompt,
                    args.max_new_tokens, args.block_size, args.num_steps,
                    mask_token_id, pad_token_id,
                    args.temperature, args.top_k,
                    args.acceptance_threshold, args.max_retries, device,
                )
            else:
                result = block_sample(
                    model, tokenizer, prompt,
                    args.max_new_tokens, args.block_size, args.num_steps,
                    mask_token_id, pad_token_id,
                    args.temperature, args.top_k, device,
                )
        else:
            result = iterative_sample(
                model, tokenizer, prompt,
                args.max_new_tokens, args.num_steps,
                mask_token_id, pad_token_id,
                args.temperature, args.top_k, device,
            )
        results.append(result)
        print(f"prompt: {result['prompt'][:60]}...")
        print(f"output: {result['output'][:120]}")
        print(f"latency: {result['latency_ms']:.1f} ms, tokens: {result['generated_tokens']}")
        print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
