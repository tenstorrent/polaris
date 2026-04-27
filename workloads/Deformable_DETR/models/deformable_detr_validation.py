#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DeformableDETRValidation – Polaris Workload with Integrated Validation Suite

Extends DeformableDETR to additionally execute every test_*.py script found
recursively under workloads/Deformable_DETR/reference/tests/ the first time
the workload is instantiated.

All captured stdout / stderr is written to a timestamped Markdown report
located in workloads/Deformable_DETR/validation_output/. The absolute path
to that report is printed to the terminal once the run is complete.

After running the validation suite the workload behaves identically to the
plain DeformableDETR workload – it produces the same forward-graph and the
same Polaris JSON projection output.

Usage
-----
    python polaris.py \\
        -w config/ip_workloads.yaml \\
        -a config/all_archs.yaml \\
        -m config/wl2archmapping.yaml \\
        --filterwlg ttsim \\
        --filterwl deformable_detr_validation \\
        -o ODDIR_deformable_detr_validation \\
        -s SIMPLE_RUN \\
        --outputformat json
"""

import datetime
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from workloads.Deformable_DETR.models.deformable_detr_ttsim import DeformableDETR  # noqa: E402
from workloads.validation_helpers import (  # noqa: E402
    run_subprocess,
    collect_test_files,
    write_simple_markdown,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute path to the tests folder (searched recursively for test_*.py)
_VALIDATION_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "reference", "tests")
)

# Absolute path of the polaris workspace root (three levels above this file)
_POLARIS_ROOT: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

# Directory into which the Markdown report is written
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "validation_output")
)

# ---------------------------------------------------------------------------
# Per-file descriptions (shown in the Markdown report)
# Keyed by the filename basename; relative path is shown separately.
# ---------------------------------------------------------------------------

_TEST_DESCRIPTIONS: dict[str, str] = {
    # ── top-level tests ────────────────────────────────────────────────────
    "test_backbone_comparision.py": (
        "Comprehensive backbone validation comparing PyTorch and TTSim "
        "implementations for FrozenBatchNorm2d (numerical), ResNetBottleneck "
        "(numerical with pretrained weights), Backbone, and Joiner (shape-only "
        "due to Conv2d cost). Includes relative/absolute error analysis."
    ),
    "test_box_ops_comparision.py": (
        "Box-operations comparison (box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, "
        "box_area, box_iou, generalized_box_iou, masks_to_boxes) with "
        "full numerical analysis and relative error reporting."
    ),
    "test_matcher_comparision.py": (
        "Validates the HungarianMatcher bipartite matching module: shape "
        "correctness, numerical equivalence of focal-loss, L1, and GIoU "
        "cost components, and end-to-end matching results."
    ),
    "test_misc_comparision.py": (
        "Misc utility comparison for NestedTensor data container, "
        "interpolate, and nested_tensor_from_tensor_list between PyTorch "
        "and TTSim implementations with numerical output analysis."
    ),
    "test_ms_deform_attn_comparision.py": (
        "MSDeformAttn module validation: shape inference and numerical "
        "comparison of the multi-scale deformable attention module against "
        "the PyTorch reference across multiple configurations."
    ),
    "test_ms_deform_attn_func_comparision.py": (
        "Validates the ms_deform_attn_core function: shape inference and "
        "numerical comparison of the deformable attention core computation "
        "against the PyTorch reference."
    ),
    "test_position_encoding_numerical.py": (
        "Exhaustive element-wise position-encoding numerical validation: "
        "PositionEmbeddingSine across normalised/unnormalised configs, "
        "non-square spatial sizes, mask patterns, y/x component separation, "
        "parameter sweeps, and edge cases; PositionEmbeddingLearned shape "
        "check; and build_position_encoding factory verification."
    ),
    "test_position_encoding_shape.py": (
        "Shape and numerical comparison for PositionEmbeddingSine, "
        "PositionEmbeddingLearned (shape only – random init), and "
        "the build_position_encoding factory function."
    ),
    # ── deformable_detr/ ──────────────────────────────────────────────────
    "test_numerical.py": (
        "Comprehensive DeformableDETR numerical validation covering 15 "
        "component and end-to-end pipeline tests: MLP, PostProcess, "
        "sigmoid_focal_loss, dice_loss, inverse_sigmoid, SetCriterion "
        "(with and without auxiliary outputs), PostProcessSegm, MLP "
        "intermediates, and full DeformableDETR pipelines (1-layer and "
        "3-layer encoder + decoder)."
    ),
    "test_shapes.py": (
        "Comprehensive DeformableDETR shape validation covering 15 component "
        "and end-to-end pipeline shape tests: MLP, PostProcess, loss "
        "scalars, inverse_sigmoid, SetCriterion (with aux), PostProcessSegm, "
        "and full DeformableDETR pred_logits / pred_boxes / aux_outputs "
        "output shapes."
    ),
    # ── deformable_transformer/deformable_transformer_numerical/ ──────────
    "test_decoder_debug.py": (
        "Block-by-block decoder debugger that isolates divergence in the "
        "multi-layer decoder by independently comparing reference_points "
        "expansion and each sub-block (pos embedding, self-attn + norm, "
        "cross-attn + norm, FFN) to pinpoint accumulation errors."
    ),
    "test_decoder_layer_numerical.py": (
        "Numerical validation of DeformableTransformerDecoderLayer: "
        "self-attention (MHA), cross-attention (MSDeformAttn), FFN, and "
        "LayerNorm residual connections with weight-synced PyTorch and "
        "TTSim instances (accounts for TTSim norm-index swap)."
    ),
    "test_decoder_numerical.py": (
        "Numerical validation of the full DeformableTransformerDecoder "
        "with stacked layers and return_intermediate=False; weights synced "
        "from PyTorch including the norm-index swap between implementations."
    ),
    "test_encoder_layer_numerical.py": (
        "Numerical validation of DeformableTransformerEncoderLayer: "
        "MSDeformAttn + FFN + LayerNorm residual connections with "
        "weight-synced inputs and tolerance-based output comparison."
    ),
    "test_encoder_numerical.py": (
        "Numerical validation of the full DeformableTransformerEncoder "
        "with stacked encoder layers and shared weight synchronisation "
        "from PyTorch to TTSim."
    ),
    "test_full_transformer_numerical.py": (
        "End-to-end numerical validation of the complete "
        "DeformableTransformer (multi-scale flatten → level_embed → "
        "encoder → query split → reference-point linear → decoder) with "
        "full weight synchronisation across all sub-components."
    ),
    # ── deformable_transformer/deformable_transformer_shape/ ─────────────
    "test_decoder_layer_simple.py": (
        "Shape validation for DeformableTransformerDecoderLayer: "
        "verifies output tensor dimensions from the self-attn + "
        "cross-attn + FFN stack."
    ),
    "test_decoder_simple.py": (
        "Shape validation for the full DeformableTransformerDecoder "
        "across multiple stacked decoder layers."
    ),
    "test_encoder_layer_simple.py": (
        "Shape validation for DeformableTransformerEncoderLayer output "
        "dimensions from the MSDeformAttn + FFN stack."
    ),
    "test_encoder_simple.py": (
        "Shape validation for the full DeformableTransformerEncoder "
        "across multiple stacked encoder layers."
    ),
    "test_full_transformer_simple.py": (
        "Shape validation for the complete DeformableTransformer: "
        "verifies hs (decoder hidden states) and reference_points "
        "output dimensions."
    ),
    # ── segmentation/ ─────────────────────────────────────────────────────
    "test_segmentation_shape_inference.py": (
        "Shape inference validation for Deformable DETR segmentation "
        "modules (MHAttentionMap, MaskHeadSmallConv, DETRsegm), diagnosing "
        "unsqueeze ipos mismatches, concat shape mismatches, and tile "
        "batch-dimension expansion issues."
    ),
    # ── segmentation/segmentation_numerical/ ─────────────────────────────
    "test_detrsegm.py": (
        "Numerical validation of DETRsegm: component integration test "
        "(MHAttentionMap + MaskHeadSmallConv), output shape validation, "
        "reshape/squeeze operations, and full end-to-end DETRsegm forward "
        "pass using mock DETR outputs."
    ),
    "test_helper_functions.py": (
        "Numerical validation of segmentation helper functions: "
        "masked_fill_impl, interpolate_nearest, and conv2d_functional "
        "against PyTorch references with tolerance-based comparison."
    ),
    "test_maskhead_smallconv.py": (
        "Numerical validation of MaskHeadSmallConv: GroupNorm, "
        "Conv2d + GroupNorm + ReLU, 1×1 Conv FPN adapter, "
        "interpolate nearest, expand (batch expansion), and full "
        "forward pass against PyTorch reference."
    ),
    "test_mhattention_map.py": (
        "Numerical validation of MHAttentionMap: linear q-projection, "
        "1×1 Conv k-projection, multi-head reshape, einsum attention "
        "scores (bqnc,bnchw→bqnhw), softmax, and dropout (test mode) "
        "against PyTorch reference."
    ),
}

_DEFAULT_TEST_DESCRIPTION = (
    "Runs Deformable DETR-related validation tests and reports pass/fail status."
)

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once across all workload instances
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: str | None = None

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _run_one_test(fpath: str, timeout: int = 300) -> tuple[int, str]:
    return run_subprocess(fpath, _POLARIS_ROOT, timeout=timeout)


def _write_report(md_path: str, results: list) -> None:
    # results from _run_validation_tests: [(rel_path, rc, output)]
    write_simple_markdown(
        md_path,
        "Deformable DETR Validation Test Report",
        results,
        _TEST_DESCRIPTIONS,
        default_description=_DEFAULT_TEST_DESCRIPTION,
    )


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

    test_files = collect_test_files(_VALIDATION_DIR)
    if not test_files:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# Deformable DETR Validation Test Report\n\n")
            fh.write("**Result:** No test files found.\n")
        _VALIDATION_DONE = True
        _VALIDATION_MD_PATH = os.path.abspath(md_path)
        return _VALIDATION_MD_PATH

    print(f"\n{'=' * 72}")
    print(
        f"[DeformableDETRValidation] Running {len(test_files)} validation test(s)…"
    )
    print(f"{'=' * 72}\n")

    results: list[tuple[str, int, str]] = []
    for rel_path, fpath in test_files:
        print(f"  › Running {rel_path} …", end="", flush=True)
        rc, output = _run_one_test(fpath)
        label = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {label}")
        results.append((rel_path, rc, output))

    _write_report(md_path, results)

    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"[DeformableDETRValidation] {passed}/{total} tests passed.")
    print(f"[DeformableDETRValidation] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# Polaris workload class
# ---------------------------------------------------------------------------


class DeformableDETRValidation(DeformableDETR):

    def create_input_tensors(self) -> None:
        # Run validation tests (no-op on subsequent instances).
        _run_validation_tests()
        # Create the TTSim input tensors for the DeformableDETR forward graph.
        super().create_input_tensors()
