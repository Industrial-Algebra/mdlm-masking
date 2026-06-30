#!/usr/bin/env python3
"""Unit tests for the MDLM iterative diffusion sampler."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("mdlm_sampler.py")
SPEC = importlib.util.spec_from_file_location("quantizon_mdlm_sampler", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"failed to load mdlm sampler from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MdlmSamplerTests(unittest.TestCase):
    def test_build_initial_sequence_all_masked(self) -> None:
        prompt_ids = [1, 2, 3]
        target_length = 8
        mask_token_id = 999
        pad_token_id = 0

        seq = MODULE.build_initial_sequence(
            prompt_ids, target_length, mask_token_id, pad_token_id,
        )
        self.assertEqual(len(seq), target_length)
        # First 3 tokens are the prompt, rest are mask
        self.assertEqual(seq[:3], prompt_ids)
        self.assertTrue(all(t == mask_token_id for t in seq[3:]))

    def test_build_initial_sequence_clips_prompt(self) -> None:
        prompt_ids = [1, 2, 3, 4, 5]
        target_length = 4
        mask_token_id = 999
        pad_token_id = 0

        seq = MODULE.build_initial_sequence(
            prompt_ids, target_length, mask_token_id, pad_token_id,
        )
        self.assertEqual(len(seq), target_length)
        # Prompt is clipped to fit
        self.assertEqual(seq, [1, 2, 3, 4])

    def test_parse_sampling_args_defaults(self) -> None:
        import sys as _sys
        _sys.argv = ["mdlm_sampler.py", "--checkpoint", "/tmp/test"]
        args = MODULE.parse_args()
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.num_steps, 16)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_k, 1)

    def test_slugify_handles_paths(self) -> None:
        self.assertEqual(
            MODULE.slugify("reports/phase2-masked-denoiser-eval.json"),
            "reports-phase2-masked-denoiser-eval-json",
        )


if __name__ == "__main__":
    unittest.main()
