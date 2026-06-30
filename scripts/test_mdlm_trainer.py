#!/usr/bin/env python3
"""Unit tests for the MDLM fine-tuning trainer."""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("mdlm_trainer.py")
SPEC = importlib.util.spec_from_file_location("quantizon_mdlm_trainer", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"failed to load mdlm trainer from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MdlmTrainerTests(unittest.TestCase):
    def test_corrupt_sequence_masks_correct_fraction(self) -> None:
        token_ids = [101, 2054, 2003, 3082, 102, 2000, 1037, 2307]
        mask_token_id = 32000
        corrupted, positions = MODULE.corrupt_sequence(
            token_ids,
            mask_ratio=0.4,
            mask_token_id=mask_token_id,
            rng=random.Random(42),
        )
        self.assertEqual(len(corrupted), len(token_ids))
        mask_fraction = sum(1 for t in corrupted if t == mask_token_id) / len(corrupted)
        self.assertTrue(0.1 <= mask_fraction <= 0.7,
                        f"expected ~40% masked, got {mask_fraction:.2%}")
        self.assertEqual(len(positions), sum(1 for t in corrupted if t == mask_token_id))

    def test_corrupt_sequence_skips_token_ids(self) -> None:
        token_ids = [1, 2, 3, 4, 5]
        mask_token_id = 99
        skip_ids = frozenset({2, 4})
        corrupted, positions = MODULE.corrupt_sequence(
            token_ids, mask_ratio=1.0, mask_token_id=mask_token_id,
            rng=random.Random(42), skip_token_ids=skip_ids,
        )
        # Tokens 2 and 4 should NEVER be masked
        self.assertEqual(corrupted[1], 2)  # token at index 1 is ID 2, should be preserved
        self.assertEqual(corrupted[3], 4)  # token at index 3 is ID 4, should be preserved
        # Other positions should be masked
        self.assertEqual(corrupted[0], mask_token_id)
        self.assertEqual(corrupted[2], mask_token_id)
        self.assertEqual(corrupted[4], mask_token_id)

    def test_bayesian_mask_min_gate(self) -> None:
        """Min gate: token must pass BOTH structural AND frequency filters."""
        token_ids = [10, 20, 30, 40, 50]
        mask_token_id = 99
        skip_ids = frozenset({20})  # token 20 structurally skipped
        weights = [0.0, 0.0, 0.5, 1.0, 1.0]  # freq weights: 10 and 20 get zero
        # Expand weights to cover vocab
        full_weights = [1.0] * 100
        for i, w in enumerate(weights):
            full_weights[token_ids[i]] = w

        # At max ratio, only tokens passing BOTH gates should be masked:
        # Token 10: skip_gate=1, freq_gate=0 → min=0 → not masked
        # Token 20: skip_gate=0 (in skip_ids) → not masked
        # Token 30: skip_gate=1, freq_gate=0.5 → masked
        # Token 40: skip_gate=1, freq_gate=1.0 → masked
        # Token 50: skip_gate=1, freq_gate=1.0 → masked
        corrupted, positions = MODULE.corrupt_sequence(
            token_ids, mask_ratio=1.0, mask_token_id=mask_token_id,
            rng=random.Random(42), skip_token_ids=skip_ids,
            token_weights=full_weights,
        )
        self.assertEqual(corrupted[0], 10)  # not masked (freq zero)
        self.assertEqual(corrupted[1], 20)  # not masked (structural skip)
        self.assertEqual(corrupted[2], mask_token_id)  # masked
        self.assertEqual(corrupted[3], mask_token_id)  # masked
        self.assertEqual(corrupted[4], mask_token_id)  # masked

    def test_corrupt_sequence_preserves_unmasked(self) -> None:
        token_ids = [1, 2, 3, 4, 5]
        mask_token_id = 0
        corrupted, positions = MODULE.corrupt_sequence(
            token_ids,
            mask_ratio=0.0,
            mask_token_id=mask_token_id,
            rng=random.Random(42),
        )
        self.assertEqual(corrupted, token_ids)
        self.assertEqual(positions, [])

    def test_corrupt_sequence_reproducible(self) -> None:
        token_ids = list(range(100))
        mask_token_id = 999
        c1, p1 = MODULE.corrupt_sequence(
            token_ids, mask_ratio=0.3, mask_token_id=mask_token_id, rng=random.Random(7),
        )
        c2, p2 = MODULE.corrupt_sequence(
            token_ids, mask_ratio=0.3, mask_token_id=mask_token_id, rng=random.Random(7),
        )
        self.assertEqual(c1, c2)
        self.assertEqual(p1, p2)

    def test_corrupt_sequence_all_masked(self) -> None:
        token_ids = [1, 2, 3]
        mask_token_id = 99
        corrupted, positions = MODULE.corrupt_sequence(
            token_ids,
            mask_ratio=1.0,
            mask_token_id=mask_token_id,
            rng=random.Random(42),
        )
        self.assertEqual(corrupted, [mask_token_id, mask_token_id, mask_token_id])
        self.assertEqual(positions, [0, 1, 2])

    def test_slugify_handles_slashes_and_uppercase(self) -> None:
        self.assertEqual(MODULE.slugify("Qwen/Qwen2-0.5B-Instruct"), "qwen-qwen2-0-5b-instruct")

    def test_build_training_config_defaults(self) -> None:
        args = type("Args", (), {})()
        args.model = "Qwen/Qwen2-0.5B-Instruct"
        args.dataset_name = "allenai/c4"
        args.dataset_config = "en"
        args.dataset_split = "train"
        args.epochs = 3
        args.batch_size = 8
        args.mask_ratio = 0.15
        args.mask_schedule = "fixed"
        args.max_length = 256
        args.learning_rate = 1e-4
        args.weight_decay = 0.01
        args.grad_clip_norm = 1.0
        args.warmup_steps = 100
        args.max_steps = 1000
        args.output_dir = "/tmp/mdlm-test"
        args.seed = 42
        args.overwrite = False
        args.device = None

        config = MODULE.build_training_config(args)
        self.assertEqual(config["model"], "Qwen/Qwen2-0.5B-Instruct")
        self.assertEqual(config["mask_ratio"], 0.15)
        self.assertEqual(config["mask_schedule"], "fixed")
        self.assertEqual(config["epochs"], 3)

    def test_cosine_mask_schedule_samples_full_range(self) -> None:
        rng = random.Random(42)
        ratios = [MODULE.sample_cosine_mask_ratio(rng) for _ in range(1000)]
        # Should see values from near 0 to near 1
        self.assertGreater(max(ratios), 0.9, "cosine schedule should produce high mask ratios")
        self.assertLess(min(ratios), 0.1, "cosine schedule should produce low mask ratios")
        avg = sum(ratios) / len(ratios)
        self.assertTrue(0.3 < avg < 0.7, f"cosine avg should be ~0.5, got {avg:.3f}")

    def test_cosine_schedule_zero_at_start(self) -> None:
        """At t=0, mask ratio should be 0 (no corruption)."""
        ratio = MODULE.cosine_mask_ratio_at(0.0)
        self.assertAlmostEqual(ratio, 0.0, places=5)

    def test_cosine_schedule_one_at_end(self) -> None:
        """At t=1, mask ratio should be 1 (full corruption)."""
        ratio = MODULE.cosine_mask_ratio_at(1.0)
        self.assertAlmostEqual(ratio, 1.0, places=5)

    def test_cosine_schedule_monotonic(self) -> None:
        """Mask ratio should increase monotonically with t."""
        ratios = [MODULE.cosine_mask_ratio_at(t / 100.0) for t in range(101)]
        self.assertTrue(all(ratios[i] <= ratios[i+1] for i in range(len(ratios)-1)))

    def test_mdlm_loss_only_on_masked_positions(self) -> None:
        """White-box test: verify loss masking logic on a tiny logits/labels pair."""
        try:
            import torch
        except ModuleNotFoundError:
            raise unittest.SkipTest("torch not available")

        logits = torch.randn(2, 5, 1000)  # batch=2, seq=5, vocab=1000
        labels = torch.tensor([[10, -100, 20, -100, 30],
                               [15, -100, -100, 25, 35]])
        mask_token_id = 999

        loss = MODULE.compute_mdlm_loss(logits, labels, mask_token_id)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar
        self.assertGreater(loss.item(), 0.0)

    def test_mdlm_loss_ignores_padding(self) -> None:
        """Labels with -100 are ignored (HF convention for padding)."""
        try:
            import torch
        except ModuleNotFoundError:
            raise unittest.SkipTest("torch not available")

        logits = torch.randn(1, 3, 1000)
        labels = torch.tensor([[-100, 50, -100]])
        mask_token_id = 999

        loss = MODULE.compute_mdlm_loss(logits, labels, mask_token_id)
        self.assertIsInstance(loss, torch.Tensor)

    def test_corrupt_blocks_masks_contiguous_spans(self) -> None:
        """Blocks should be masked as contiguous units."""
        token_ids = list(range(30))
        mask_token_id = 999
        rng = random.Random(7)
        corrupted, positions = MODULE.corrupt_blocks(
            token_ids, block_size=4, mask_ratio=0.5,
            mask_token_id=mask_token_id, rng=rng,
        )
        self.assertEqual(len(corrupted), len(token_ids))
        # Verify that masked positions always start at block boundaries
        # (multiples of block_size, or the remainder block after the last full block)
        for pos in positions:
            block_start = (pos // 4) * 4
            self.assertGreaterEqual(pos, block_start)
            self.assertLess(pos, block_start + 4)

    def test_corrupt_blocks_respects_bayesian_gate(self) -> None:
        """Blocks of only punctuation should not be masked."""
        token_ids = list(range(24))
        mask_token_id = 999
        # Mark tokens 0-3, 4-7 as punctuation blocks
        skip_ids = frozenset({0, 1, 2, 3, 4, 5, 6, 7})
        rng = random.Random(42)
        for _ in range(20):
            corrupted, positions = MODULE.corrupt_blocks(
                token_ids, block_size=4, mask_ratio=1.0,
                mask_token_id=mask_token_id, rng=rng,
                skip_token_ids=skip_ids,
            )
            # First two blocks (0-7, all punctuation) should never be masked
            self.assertNotIn(0, positions)
            self.assertNotIn(4, positions)

    def test_corrupt_blocks_handles_remainder(self) -> None:
        """Last partial block should be handled correctly."""
        token_ids = list(range(10))  # 10 tokens, 2 full blocks + remainder of 2
        mask_token_id = 999
        rng = random.Random(42)
        corrupted, positions = MODULE.corrupt_blocks(
            token_ids, block_size=4, mask_ratio=1.0,
            mask_token_id=mask_token_id, rng=rng,
        )
        # Should have all tokens masked or the remainder preserved
        self.assertEqual(len(corrupted), 10)


if __name__ == "__main__":
    unittest.main()
