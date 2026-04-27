#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormerValidation – Polaris Workload with Integrated Validation Suite

Extends BEVFormer to additionally execute every test_*.py script found in
workloads/BEVFormer/reference/Validation/ (excluding layer_ops.py and
ttsim_utils.py) the first time the workload is instantiated.

All captured stdout / stderr is written to a timestamped Markdown report
located in workloads/BEVFormer/validation_output/. The absolute path to that
report is printed to the terminal once the run is complete.

After running the validation suite the workload behaves identically to the
plain BEVFormer workload – it produces the same forward-graph and the same
Polaris JSON projection output.

Usage
-----
    python polaris.py \\
        -w config/ip_workloads.yaml \\
        -a config/all_archs.yaml \\
        -m config/wl2archmapping.yaml \\
        --filterwlg ttsim \\
        --filterwl bevformer_validation \\
        -o ODDIR_bevformer_validation \\
        -s SIMPLE_RUN \\
        --outputformat json
"""

import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from workloads.BEVFormer.ttsim_models.bevformer import BEVFormer  # noqa: E402
from workloads.validation_helpers import (  # noqa: E402
    run_subprocess,
    collect_test_files,
    write_simple_markdown,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute path to the Validation folder that holds all test scripts
_VALIDATION_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "reference", "Validation")
)

# Absolute path of the polaris workspace root (three levels above this file)
_POLARIS_ROOT: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

# Directory into which the Markdown report is written
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "validation_output")
)

# Files inside _VALIDATION_DIR that are *not* test entry-points
_EXCLUDED_FILES = {"layer_ops.py", "ttsim_utils.py"}

# ---------------------------------------------------------------------------
# Per-file descriptions (shown in the Markdown report)
# ---------------------------------------------------------------------------

_TEST_DESCRIPTIONS: dict[str, str] = {
    "test_bbox_utils.py": (
        "Validates bounding-box normalization and denormalization utilities "
        "(`normalize_bbox`, `denormalize_bbox`) by comparing the TTSim "
        "implementation against the PyTorch reference across 7-D and 10-D "
        "inputs, round-trip consistency, batch processing, and edge cases."
    ),
    "test_bevformer.py": (
        "Full BEVFormer architecture layer-by-layer validation. Checks model "
        "construction, shape propagation throughout the backbone → FPN → "
        "encoder → decoder → classification-head pipeline, and numerical "
        "accuracy of TTSim computations against PyTorch reference tensors."
    ),
    "test_bevformer_encoder.py": (
        "Validates BEVFormer encoder components: 3-D and 2-D reference-point "
        "generation, point sampling, camera-projection shapes, and "
        "BEVFormerLayer / BEVFormerEncoder module construction and parameter "
        "counts."
    ),
    "test_bevformer_head.py": (
        "Validates the BEVFormer detection head including inverse-sigmoid, "
        "bbox normalisation helpers, multi-apply utility, bias initialisation "
        "with probability, and construction of BEVFormerHead and "
        "BEVFormerHeadGroupDETR."
    ),
    "test_bricks.py": (
        "Validates the `run_time` timing/profiling decorator used throughout "
        "BEVFormer: basic decoration, multiple functions, statistics reset and "
        "retrieval, summary printing, args/kwargs forwarding, metadata "
        "preservation, nested decorators, and exception handling."
    ),
    "test_custom_base_transformer_layer.py": (
        "Validates the custom base transformer layer: LayerNorm, feed-forward "
        "network (FFN), layer construction and operation-order enforcement, "
        "and a numerical comparison against a PyTorch reference."
    ),
    "test_decoder.py": (
        "Validates BEVFormer decoder components including inverse-sigmoid, "
        "custom multi-scale deformable self-attention (MSDA) construction and "
        "forward pass, decoder construction, decoder forward pass, and "
        "intermediate-output return."
    ),
    "test_grid_mask.py": (
        "Validates the GridMask data-augmentation module: tile-operation "
        "correctness, element-wise masking, offset-mode data validation, "
        "mode-1 inverted mask, training-vs-inference behaviour, different "
        "input sizes, configuration variants, and parameter counting."
    ),
    "test_multi_scale_deform_attn.py": (
        "Validates Multi-Scale Deformable Attention (MSDA) computations "
        "across single-level and multi-level feature maps, varying batch "
        "sizes, head counts, sampling points, boundary sampling locations, "
        "and a full BEVFormer-scale scenario."
    ),
    "test_nms_free_coder.py": (
        "Validates the NMS-free detection coder: bounding-box "
        "denormalization, single-sample and batch decode, top-k selection, "
        "score-threshold filtering, and center-range (BEV-area) filtering."
    ),
    "test_position_embedding.py": (
        "Validates position-embedding generation for BEV queries: module "
        "construction, forward-pass output shape, various spatial sizes, "
        "parameter count, operation without normalization, and data-value "
        "validation."
    ),
    "test_spatial_cross_attention.py": (
        "Validates Spatial Cross-Attention (SCA): module construction, "
        "MSDA-3D forward pass, SCA forward pass, different configurations, "
        "parameter counting, batch-first flag behaviour, masking, and edge "
        "cases."
    ),
    "test_temporal_self_attention.py": (
        "Validates Temporal Self-Attention (TSA): module construction, "
        "forward pass with weighted combination across frames, parameter "
        "counting, and multiple alternative configurations."
    ),
    "test_transformer.py": (
        "Validates the full BEVFormer transformer module by running TTSim "
        "implementations of all sub-components (reference-point prediction, "
        "CAN-bus processing, embeddings) and confirming that computed values "
        "match the PyTorch reference implementation within numerical "
        "tolerance."
    ),
}

_DEFAULT_TEST_DESCRIPTION = (
    "Runs BEVFormer-related validation tests and reports pass/fail status."
)

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once across all workload instances
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: str | None = None

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _collect_bev_test_files() -> list[tuple[str, str]]:
    """Return test files from _VALIDATION_DIR excluding _EXCLUDED_FILES."""
    if not os.path.isdir(_VALIDATION_DIR):
        return []
    entries = []
    for fname in sorted(os.listdir(_VALIDATION_DIR)):
        if (
            fname.startswith("test_")
            and fname.endswith(".py")
            and fname not in _EXCLUDED_FILES
        ):
            entries.append((fname, os.path.join(_VALIDATION_DIR, fname)))
    return entries


def _run_one_test(fpath: str, timeout: int = 300) -> tuple[int, str]:
    return run_subprocess(fpath, _POLARIS_ROOT, timeout=timeout)


def _run_validation_tests() -> str:
    """Run all validation test scripts and write a Markdown report.

    Returns the **absolute path** to the generated Markdown file.
    Subsequent calls are no-ops; the cached path is returned instead.
    """
    global _VALIDATION_DONE, _VALIDATION_MD_PATH  # noqa: PLW0603

    if _VALIDATION_DONE and _VALIDATION_MD_PATH is not None:
        return _VALIDATION_MD_PATH

    os.makedirs(_OUTPUT_BASE, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(_OUTPUT_BASE, f"validation_report_{timestamp}.md")

    test_files = _collect_bev_test_files()
    if not test_files:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# BEVFormer Validation Test Report\n\n")
            fh.write("**Result:** No test files found.\n")
        _VALIDATION_DONE = True
        _VALIDATION_MD_PATH = os.path.abspath(md_path)
        return _VALIDATION_MD_PATH

    print(f"\n{'=' * 72}")
    print(f"[BEVFormerValidation] Running {len(test_files)} validation test(s)…")
    print(f"{'=' * 72}\n")

    results: list[tuple[str, int, str]] = []
    for fname, fpath in test_files:
        print(f"  › Running {fname} …", end="", flush=True)
        rc, output = _run_one_test(fpath)
        label = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {label}")
        results.append((fname, rc, output))

    write_simple_markdown(
        md_path,
        "BEVFormer Validation Test Report",
        results,
        _TEST_DESCRIPTIONS,
        default_description=_DEFAULT_TEST_DESCRIPTION,
    )

    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"[BEVFormerValidation] {passed}/{total} tests passed.")
    print(f"[BEVFormerValidation] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# Polaris workload class
# ---------------------------------------------------------------------------


class BEVFormerValidation(BEVFormer):

    def create_input_tensors(self) -> None:
        # Run validation tests (no-op on subsequent instances).
        _run_validation_tests()
        # Create the TTSim input tensors for the BEVFormer forward graph.
        super().create_input_tensors()
