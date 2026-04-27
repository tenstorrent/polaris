#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DiffusionDriveValidation – Polaris Workload with Integrated Validation Suite

Extends V2TransfuserModel to additionally execute every test_*.py script found
under workloads/DiffusionDrive/navsim/agents/diffusiondrive/reference/ (all
subdirectories, excluding run_all_tests.py and comparison_results.md) the first
time the workload is instantiated.

All captured stdout / stderr is written to a timestamped Markdown report located
in workloads/DiffusionDrive/validation_output/. The absolute path to that report
is printed to the terminal once the run is complete.

After running the validation suite the workload behaves identically to the plain
DiffusionDrive workload – it produces the same forward-graph and the same Polaris
JSON projection output.

Usage
-----
    python polaris.py \\
        -w config/ip_workloads.yaml \\
        -a config/all_archs.yaml \\
        -m config/wl2archmapping.yaml \\
        --filterwlg ttsim \\
        --filterwl DiffusionDrive_Validation \\
        -o ODDIR_diffusiondrive_validation \\
        -s SIMPLE_RUN \\
        --outputformat json
"""

import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

from workloads.DiffusionDrive.navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (  # noqa: E402
    V2TransfuserModel,
)
from workloads.validation_helpers import (  # noqa: E402
    run_subprocess,
    write_sectioned_markdown,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute path to the reference folder that holds all subdirectory test suites
_REFERENCE_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "reference")
)

# Absolute path of the polaris workspace root (five levels above this file)
_POLARIS_ROOT: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../..")
)

# Directory into which the Markdown report is written
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "validation_output")
)

# Subdirectory names under _REFERENCE_DIR that contain test entry-points,
# listed in the order they should appear in the report.
_TEST_SUBDIRS: list[str] = [
    os.path.join("Modules", "blocks"),
    os.path.join("Modules", "Conditional_unet1d"),
    "Transfuser_backbone",
    "transfuser_model_v2",
]

# ---------------------------------------------------------------------------
# Section titles shown in the Markdown report
# ---------------------------------------------------------------------------

_SECTION_TITLES: dict[str, str] = {
    os.path.join("Modules", "blocks"): "Modules / Blocks",
    os.path.join("Modules", "Conditional_unet1d"): "Modules / Conditional UNet1D",
    "Transfuser_backbone": "Transfuser Backbone",
    "transfuser_model_v2": "Transfuser Model V2",
}

# ---------------------------------------------------------------------------
# Per-file descriptions (shown in the Markdown report)
# ---------------------------------------------------------------------------

_TEST_DESCRIPTIONS: dict[str, str] = {
    # Modules / blocks
    "test_1_grid_sample_bev_attention.py": (
        "Validates the GridSampleCrossBEVAttention module: shape inference and "
        "numerical equivalence between the TTSIM implementation and the PyTorch "
        "reference across different batch sizes, lidar resolutions, and BEV "
        "feature configurations."
    ),
    # Modules / Conditional_unet1d
    "test_0_sinusoidal_pos_emb.py": (
        "Validates the SinusoidalPosEmb positional-embedding layer: output "
        "shape, numerical accuracy against the PyTorch reference for various "
        "embedding dimensions and input time-step sequences."
    ),
    "test_1_conv1d_block.py": (
        "Validates the Conv1dBlock module: shape propagation and numerical "
        "equivalence of grouped 1-D convolution, batch normalisation, and "
        "Mish-activation against the PyTorch reference."
    ),
    "test_2_downsample1d.py": (
        "Validates the Downsample1d module: output shape and numerical "
        "correctness of the 1-D strided-convolution downsampling layer "
        "compared to the PyTorch reference."
    ),
    "test_3_upsample1d.py": (
        "Validates the Upsample1d module: output shape and numerical "
        "correctness of the 1-D transposed-convolution upsampling layer "
        "compared to the PyTorch reference."
    ),
    "test_4_cond_residual_block.py": (
        "Validates ConditionalResidualBlock1D: shape propagation, residual "
        "shortcut correctness, FiLM-style conditioning injection, and "
        "numerical equivalence against the PyTorch reference."
    ),
    "test_5_conditional_unet1d.py": (
        "Full ConditionalUnet1D (diffusion U-Net) validation: encoder "
        "down-sampling path, bottleneck, decoder up-sampling path, skip "
        "connections, and end-to-end numerical equivalence against the "
        "PyTorch reference."
    ),
    # Transfuser_backbone
    "test_1_selfattention.py": (
        "Validates the SelfAttention module: shape inference and numerical "
        "equivalence of multi-head self-attention (key / query / value "
        "projections, scaled dot-product, residual dropout) against the "
        "PyTorch reference."
    ),
    "test_2_block.py": (
        "Validates the Transformer Block module: shape propagation through "
        "LayerNorm → SelfAttention → FFN stack and numerical accuracy against "
        "the PyTorch reference implementation."
    ),
    "test_3_gpt.py": (
        "Validates the full GPT-style transformer: positional encoding, "
        "stacked Block layers, final LayerNorm, and end-to-end numerical "
        "equivalence with the PyTorch reference across different sequence "
        "lengths and embedding dimensions."
    ),
    "test_4_multihead_attention.py": (
        "Validates MultiheadAttentionWithAttention: cross-attention and "
        "self-attention modes, attention-weight extraction, shape correctness, "
        "and numerical equivalence against the PyTorch reference."
    ),
    "test_5_decoder_layer.py": (
        "Validates TransformerDecoderLayerWithAttention: masked self-attention, "
        "cross-attention, feed-forward sub-layer, residual connections, and "
        "numerical equivalence with the PyTorch reference."
    ),
    "test_6_decoder.py": (
        "Validates TransformerDecoderWithAttention: stacked decoder layers, "
        "intermediate attention-weight return, output shape, and numerical "
        "accuracy against the PyTorch reference."
    ),
    "test_7_transfuser_backbone.py": (
        "Full TransfuserBackbone validation: ResNet image and LiDAR encoders, "
        "cross-attention fusion, BEV feature extraction, and end-to-end "
        "numerical equivalence with the PyTorch reference across multiple "
        "input resolutions."
    ),
    # transfuser_model_v2
    "test_1_agent_head.py": (
        "Validates the AgentHead detection head: bounding-box regression and "
        "classification MLP stacks, output shapes, and numerical accuracy "
        "against the PyTorch reference."
    ),
    "test_2_diff_motion_planning.py": (
        "Validates DiffMotionPlanningRefinementModule: diffusion-based "
        "trajectory refinement, sinusoidal time embeddings, modulation "
        "injection, and numerical equivalence with the PyTorch reference."
    ),
    "test_3_modulation_layer.py": (
        "Validates ModulationLayer: FiLM conditioning (scale + shift) "
        "parameter generation, application to feature maps, and numerical "
        "equivalence against the PyTorch reference."
    ),
    "test_4_custom_transformer_decoder_layer.py": (
        "Validates CustomTransformerDecoderLayer: BEV cross-attention via "
        "GridSampleCrossBEVAttention, self-attention on query tokens, "
        "feed-forward block, and numerical equivalence with the PyTorch "
        "reference."
    ),
    "test_5_custom_transformer_decoder.py": (
        "Validates the full CustomTransformerDecoder: stacked decoder layers, "
        "BEV-feature cross-attention propagation, output shape, and numerical "
        "accuracy for multi-layer configurations against the PyTorch reference."
    ),
    "test_6_trajectory_head.py": (
        "Validates TrajectoryHead: plan-anchor loading, anchor-conditioned "
        "trajectory prediction, diffusion scheduler interaction, output "
        "shapes, and numerical correctness against the PyTorch reference."
    ),
    "test_7_v2_transfuser_full_validation.py": (
        "Full end-to-end V2TransfuserModel validation: camera and LiDAR "
        "feature encoding, BEV semantic map output, agent-state predictions, "
        "agent classification labels, and numerical equivalence with the "
        "PyTorch reference (trajectory output uses a mock scheduler)."
    ),
}

_DEFAULT_TEST_DESCRIPTION = (
    "Runs DiffusionDrive-related validation tests and reports pass/fail status."
)

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once across all workload instances
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: str | None = None

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _collect_test_files() -> list[tuple[str, str, str]]:
    """Return sorted list of (subdir_key, filename, absolute_path) for every test_*.py."""
    entries: list[tuple[str, str, str]] = []
    for subdir_key in _TEST_SUBDIRS:
        full_subdir = os.path.join(_REFERENCE_DIR, subdir_key)
        if not os.path.isdir(full_subdir):
            continue
        for fname in sorted(os.listdir(full_subdir)):
            if fname.startswith("test_") and fname.endswith(".py"):
                entries.append((subdir_key, fname, os.path.join(full_subdir, fname)))
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

    test_files = _collect_test_files()
    if not test_files:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# DiffusionDrive Validation Test Report\n\n")
            fh.write("**Result:** No test files found.\n")
        _VALIDATION_DONE = True
        _VALIDATION_MD_PATH = os.path.abspath(md_path)
        return _VALIDATION_MD_PATH

    print(f"\n{'=' * 72}")
    print(
        f"[DiffusionDriveValidation] Running {len(test_files)} validation test(s)…"
    )
    print(f"{'=' * 72}\n")

    results: list[tuple[str, str, int, str]] = []
    current_section: str | None = None
    for subdir_key, fname, fpath in test_files:
        if subdir_key != current_section:
            current_section = subdir_key
            section_title = _SECTION_TITLES.get(subdir_key, subdir_key)
            print(f"  [{section_title}]")
        print(f"    › Running {fname} …", end="", flush=True)
        rc, output = _run_one_test(fpath)
        label = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {label}")
        results.append((subdir_key, fname, rc, output))

    write_sectioned_markdown(
        md_path,
        "DiffusionDrive Validation Test Report",
        results,
        _SECTION_TITLES,
        _TEST_DESCRIPTIONS,
        default_description=_DEFAULT_TEST_DESCRIPTION,
    )

    passed = sum(1 for _, _, rc, _ in results if rc == 0)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"[DiffusionDriveValidation] {passed}/{total} tests passed.")
    print(f"[DiffusionDriveValidation] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# Polaris workload class
# ---------------------------------------------------------------------------


class DiffusionDriveValidation(V2TransfuserModel):

    def create_input_tensors(self) -> None:
        # Run validation tests (no-op on subsequent instances).
        _run_validation_tests()
        # Create the TTSim input tensors for the DiffusionDrive forward graph.
        super().create_input_tensors()
