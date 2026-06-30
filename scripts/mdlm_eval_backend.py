#!/usr/bin/env python3
"""MDLM evaluation backend for Quantizon Phase 2.

This backend consumes:
- a Quantizon evaluation manifest,
- a Quantizon artifact-registry snapshot,
- and a prompt dataset (JSON/JSONL),

then evaluates MDLM fine-tuned checkpoints via iterative diffusion sampling.
It scores outputs using the Phase 2 evaluation metadata (exact_match,
numeric_exact, keyword_coverage) and emits Quantizon-compatible backend output.

Unlike the teacher-trace reconstruction evaluator, this backend evaluates
actual generation quality through iterative sampling, making it comparable
to the transformers AR evaluation backend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the scripts directory is importable
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

MASK_TOKEN_TEXT = "<|quantizon_mask|>"
MDLM_INTERFACE_PROFILE = "mdlm_checkpoint"


# ── scoring (mirrored from transformers_eval_backend) ──────────────────


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def numeric_values(value: str) -> list[float]:
    return [float(match) for match in re.findall(r"[-+]?\d+(?:\.\d+)?", value)]


def leading_answer_match(output: str, expected: str) -> bool:
    expected_text = expected.strip()
    if not expected_text:
        return False
    return re.match(
        rf"^\s*{re.escape(expected_text)}(?:$|[\s\.,;:!\?\)])",
        output, flags=re.IGNORECASE,
    ) is not None


def exact_match_score(output: str, accepted_outputs: tuple[str, ...]) -> float:
    if not accepted_outputs:
        return 0.0
    normalized_output = normalize_text(output)
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    normalized_first_line = normalize_text(first_line)
    for value in accepted_outputs:
        normalized_value = normalize_text(value)
        if not normalized_value:
            continue
        if normalized_output == normalized_value or normalized_first_line == normalized_value:
            return 1.0
        if leading_answer_match(output, value):
            return 1.0
    return 0.0


def numeric_exact_score(output: str, accepted_outputs: tuple[str, ...]) -> float:
    output_values = numeric_values(output)
    if not output_values:
        return 0.0
    target_values = [value for candidate in accepted_outputs for value in numeric_values(candidate)]
    if not target_values:
        return 0.0
    return 1.0 if any(
        abs(ov - tv) <= 1.0e-9 for ov in output_values for tv in target_values
    ) else 0.0


def keyword_coverage_score(output: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 0.0
    normalized_output = normalize_text(output)
    matched = sum(1 for k in keywords if normalize_text(k) in normalized_output)
    return matched / len(keywords)


def score_mdlm_output(
    output: str,
    expected_output: str | None,
    evaluation: dict[str, Any] | None,
) -> float | None:
    """Score a generated output against evaluation metadata."""
    if evaluation is None:
        if expected_output is None:
            return None
        return exact_match_score(output, (expected_output,))

    mode = str(evaluation.get("mode", "") or "").strip().lower()
    if mode in {"exact_match", ""}:
        accepted_outputs = tuple(
            str(v).strip() for v in evaluation.get("accepted_outputs", [])
            if str(v).strip()
        )
        if not accepted_outputs and expected_output:
            accepted_outputs = (expected_output,)
        return exact_match_score(output, accepted_outputs)
    if mode == "numeric_exact":
        accepted_outputs = tuple(
            str(v).strip() for v in evaluation.get("accepted_outputs", [])
            if str(v).strip()
        )
        if not accepted_outputs and expected_output:
            accepted_outputs = (expected_output,)
        return numeric_exact_score(output, accepted_outputs)
    if mode == "keyword_coverage":
        keywords = tuple(
            str(v).strip().lower() for v in evaluation.get("keywords", [])
            if str(v).strip()
        )
        return keyword_coverage_score(output, keywords)

    raise SystemExit(f"unsupported evaluation mode in MDLM backend: {mode!r}")


# ── artifact resolution ───────────────────────────────────────────────


def resolve_artifact_checkpoint_path(interface: dict[str, Any]) -> Path:
    """Extract the checkpoint directory from an artifact interface."""
    eval_iface = interface.get("evaluation_interface")
    if not isinstance(eval_iface, dict):
        raise SystemExit(
            f"artifact {interface.get('artifact_id', 'unknown')!r} "
            "has no evaluation_interface"
        )
    profile = str(eval_iface.get("profile", "") or "").strip()
    if profile != MDLM_INTERFACE_PROFILE:
        raise SystemExit(
            f"MDLM evaluation backend only supports artifacts with "
            f"evaluation_interface.profile={MDLM_INTERFACE_PROFILE!r}; "
            f"got {profile!r}"
        )
    checkpoint_dir = eval_iface.get("checkpoint_dir")
    if not checkpoint_dir:
        raise SystemExit(
            f"artifact {interface.get('artifact_id', 'unknown')!r} "
            "evaluation_interface missing checkpoint_dir"
        )
    return Path(str(checkpoint_dir))


# ── prompt loading ─────────────────────────────────────────────────────


PROMPT_KEYS = ("prompt", "instruction", "question", "input", "text")
EXPECTED_KEYS = ("expected_output", "output", "answer", "target", "reference")


def load_prompts(dataset_path: str) -> list[dict[str, Any]]:
    """Load prompt rows from a JSON or JSONL dataset."""
    path = Path(dataset_path)
    if not path.exists():
        raise SystemExit(f"prompt dataset not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise SystemExit(f"unexpected dataset format in {path}")


def extract_prompt(row: dict[str, Any]) -> str:
    for key in PROMPT_KEYS:
        if key in row:
            return str(row[key]).strip()
    raise SystemExit(f"prompt row missing recognized prompt key: {list(row.keys())}")


def extract_expected_output(row: dict[str, Any]) -> str | None:
    for key in EXPECTED_KEYS:
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return None


# ── document loading ───────────────────────────────────────────────────


def load_document(path: str) -> dict[str, Any]:
    doc_path = Path(path)
    if doc_path.suffix.lower() == ".json":
        return json.loads(doc_path.read_text(encoding="utf-8"))
    if doc_path.suffix.lower() == ".toml":
        with open(doc_path, "rb") as handle:
            return tomllib.load(handle)
    raise ValueError(f"unsupported format: {doc_path}")


def load_artifact_registry(path: str) -> dict[str, dict[str, Any]]:
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    bindings = {}
    for binding in snapshot.get("artifacts", []):
        bindings[binding["artifact_id"]] = binding
    return bindings


# ── main backend entry point ───────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MDLM evaluation backend")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report")
    parser.add_argument("--quantizon-output", required=True)
    parser.add_argument("--quantizon-artifact-registry", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy imports to fail fast on missing deps
    try:
        import torch
    except ModuleNotFoundError:
        raise SystemExit("MDLM eval backend requires PyTorch")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError:
        raise SystemExit("MDLM eval backend requires transformers")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load manifest
    manifest = load_document(args.manifest)
    spec = manifest.get("spec", manifest)
    candidates = spec.get("candidates", [])
    dataset_path = spec.get("dataset", {}).get("name", None)
    if not dataset_path:
        raise SystemExit("evaluation manifest missing dataset.name")

    # Load artifact registry
    registry = load_artifact_registry(args.quantizon_artifact_registry)

    # Load prompts
    prompts = load_prompts(dataset_path)

    metrics = spec.get("metrics", ["task_quality"])
    run_root = Path(args.run_root)

    comparisons = []
    for candidate_spec in candidates:
        candidate_id = candidate_spec["artifact_id"]
        print(f"evaluating candidate: {candidate_id}")

        # Resolve checkpoint
        binding = registry.get(candidate_id)
        if binding is None:
            print(f"  artifact {candidate_id!r} not in registry, skipping")
            continue

        interface_path = binding.get("interface_path")
        if interface_path:
            interface = json.loads(Path(interface_path).read_text(encoding="utf-8"))
        else:
            # Try to find interface at artifact location
            storage = binding.get("manifest", {}).get("storage", {})
            location = storage.get("location", "")
            iface_file = Path(location) / "quantizon-artifact-interface.json"
            interface = json.loads(iface_file.read_text(encoding="utf-8"))

        checkpoint_path = resolve_artifact_checkpoint_path(interface)
        print(f"  checkpoint: {checkpoint_path}")

        # Load model and tokenizer via importlib (scripts/ isn't a package)
        import importlib.util as _iu
        _sampler_spec = _iu.spec_from_file_location(
            "mdlm_sampler", _scripts_dir / "mdlm_sampler.py"
        )
        if _sampler_spec is None or _sampler_spec.loader is None:
            raise SystemExit("failed to load mdlm_sampler.py")
        _sampler = _iu.module_from_spec(_sampler_spec)
        sys.modules["mdlm_sampler"] = _sampler
        _sampler_spec.loader.exec_module(_sampler)
        load_checkpoint = _sampler.load_checkpoint
        iterative_sample = _sampler.iterative_sample

        model, tokenizer = load_checkpoint(str(checkpoint_path), device)
        if tokenizer.mask_token_id is None:
            tokenizer.add_tokens([MASK_TOKEN_TEXT], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN_TEXT)
        mask_token_id = tokenizer.mask_token_id
        pad_token_id = tokenizer.pad_token_id or mask_token_id

        # Evaluate each prompt
        scores: list[float] = []
        latencies: list[float] = []
        total_tokens = 0
        started_total = time.perf_counter()

        for row in prompts:
            prompt_text = extract_prompt(row)
            expected = extract_expected_output(row)
            evaluation = row.get("evaluation")

            result = iterative_sample(
                model, tokenizer, prompt_text,
                args.max_new_tokens, args.num_steps,
                mask_token_id, pad_token_id,
                temperature=0.0, top_k=1, device=device,
            )

            score = score_mdlm_output(result["output"], expected, evaluation)
            if score is not None:
                scores.append(score)
            latencies.append(result["latency_ms"])
            total_tokens += result["generated_tokens"]

        total_latency_ms = (time.perf_counter() - started_total) * 1000.0

        task_quality = sum(scores) / len(scores) if scores else None
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        throughput = len(prompts) / (total_latency_ms / 1000.0) if total_latency_ms > 0 else None

        comparisons.append({
            "reference": spec.get("reference", {}).get("artifact_id", "unknown"),
            "candidate": candidate_id,
            "diagnostics": {
                "agreement_score": None,
                "convergence_signal": None,
            },
            "metrics": [
                {"metric": "task_quality", "value": task_quality, "unit": None},
                {"metric": "latency", "value": avg_latency, "unit": "ms"},
                {"metric": "throughput", "value": throughput, "unit": "items_per_second"},
            ],
        })

        # Clean up model memory
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build backend output
    backend_output = {
        "backend_output_kind": "evaluation",
        "spec": {
            "comparisons": comparisons,
        },
    }
    Path(args.quantizon_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.quantizon_output).write_text(json.dumps(backend_output, indent=2))
    print(f"backend output written to {args.quantizon_output}")


if __name__ == "__main__":
    main()
