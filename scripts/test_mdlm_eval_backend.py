#!/usr/bin/env python3
"""Unit tests for the MDLM evaluation backend."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("mdlm_eval_backend.py")
SPEC = importlib.util.spec_from_file_location("quantizon_mdlm_eval_backend", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"failed to load mdlm eval backend from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MdlmEvalBackendTests(unittest.TestCase):
    def test_resolve_artifact_checkpoint_path_from_interface(self) -> None:
        interface = {
            "artifact_id": "artifact.qwen2-0-5b-instruct.mdlm-phase2",
            "evaluation_interface": {
                "profile": "mdlm_checkpoint",
                "checkpoint_dir": "artifacts/generated/mdlm-training/checkpoint",
            },
        }
        path = MODULE.resolve_artifact_checkpoint_path(interface)
        self.assertEqual(path, Path("artifacts/generated/mdlm-training/checkpoint"))

    def test_resolve_artifact_checkpoint_path_absolute(self) -> None:
        interface = {
            "artifact_id": "artifact.test",
            "evaluation_interface": {
                "profile": "mdlm_checkpoint",
                "checkpoint_dir": "/absolute/path/to/checkpoint",
            },
        }
        path = MODULE.resolve_artifact_checkpoint_path(interface)
        self.assertTrue(path.is_absolute())

    def test_reject_wrong_interface_profile(self) -> None:
        interface = {
            "artifact_id": "artifact.test",
            "evaluation_interface": {
                "profile": "masked_denoiser_checkpoint",
                "checkpoint_dir": "/tmp/checkpoint",
            },
        }
        with self.assertRaises(SystemExit):
            MODULE.resolve_artifact_checkpoint_path(interface)

    def test_reject_missing_checkpoint_dir(self) -> None:
        interface = {
            "artifact_id": "artifact.test",
            "evaluation_interface": {
                "profile": "mdlm_checkpoint",
            },
        }
        with self.assertRaises(SystemExit):
            MODULE.resolve_artifact_checkpoint_path(interface)

    def test_parse_mdlm_eval_args_has_required(self) -> None:
        import sys as _sys
        _sys.argv = [
            "mdlm_eval_backend.py",
            "--manifest", "/tmp/manifest.json",
            "--run-root", "/tmp/run",
            "--quantizon-output", "/tmp/output.json",
            "--quantizon-artifact-registry", "/tmp/registry.json",
        ]
        args = MODULE.parse_args()
        self.assertEqual(args.manifest, "/tmp/manifest.json")
        self.assertEqual(args.run_root, "/tmp/run")

    def test_score_mdlm_output_exact_match(self) -> None:
        result = MODULE.score_mdlm_output(
            output="42",
            expected_output="42",
            evaluation={"mode": "numeric_exact"},
        )
        self.assertEqual(result, 1.0)

    def test_score_mdlm_output_wrong_answer(self) -> None:
        result = MODULE.score_mdlm_output(
            output="41",
            expected_output="42",
            evaluation={"mode": "numeric_exact"},
        )
        self.assertEqual(result, 0.0)

    def test_score_mdlm_output_keyword_coverage(self) -> None:
        result = MODULE.score_mdlm_output(
            output="Block diffusion models use iterative denoising for language generation",
            expected_output=None,
            evaluation={"mode": "keyword_coverage", "keywords": ["block", "diffusion", "language"]},
        )
        self.assertEqual(result, 1.0)

    def test_score_mdlm_output_partial_keyword(self) -> None:
        result = MODULE.score_mdlm_output(
            output="Diffusion models are interesting",
            expected_output=None,
            evaluation={"mode": "keyword_coverage", "keywords": ["block", "diffusion", "language"]},
        )
        # Only "diffusion" matches, "block" and "language" don't
        self.assertAlmostEqual(result, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
