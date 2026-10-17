#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
EALSSValidation – TT-Polaris Workload with Integrated Validation Suite

Extends EALSS to additionally execute every test_*.py script found in
workloads/EA-LSS/Reference/Validation/ (excluding ttsim_utils.py) the first
time the workload is instantiated.

All captured stdout / stderr is written to a timestamped Markdown report
located in workloads/EA-LSS/validation_output/. The absolute path to that
report is printed to the terminal once the run is complete.

After running the validation suite the workload behaves identically to the
plain EALSS workload – it produces the same forward-graph and the same
TT-Polaris JSON projection output.

Usage
-----
    python polaris.py \\
        -w config/ip_workloads.yaml \\
        -a config/all_archs.yaml \\
        -m config/wl2archmapping.yaml \\
        --filterwlg ttsim \\
        --filterwl ealss_validation \\
        -o ODDIR_ealss_validation \\
        -s SIMPLE_RUN \\
        --outputformat json
"""

import datetime
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_polaris_root = os.path.normpath(os.path.join(_this_dir, "../.."))

for _p in [_polaris_root, _this_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workloads.EA_LSS.ealss_ttsim import EA_LSS as EALSS  # noqa: E402
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
    os.path.join(_this_dir, "reference", "Validation")
)

# Absolute path of the TT-Polaris workspace root (two levels above this file)
_POLARIS_ROOT: str = _polaris_root

# Directory into which the Markdown report is written
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(_this_dir, "validation_output")
)

# ---------------------------------------------------------------------------
# Per-file descriptions (shown in the Markdown report)
# ---------------------------------------------------------------------------

_TEST_DESCRIPTIONS: dict[str, str] = {
    "test_base_3d_detector.py": (
        "Validates that `Base3DDetector` acts as an abstract pass-through shell "
        "with zero learnable parameters, returning its input unchanged, and "
        "serving as the base class for all EA-LSS detector modules."
    ),
    "test_box3d_nms.py": (
        "Validates pure-NumPy (CPU-only) implementations of 3D NMS "
        "post-processing utilities (`circle_nms`, `aligned_3d_nms`, "
        "`box3d_multiclass_nms`) that filter redundant 3D detections after "
        "TransFusionHead."
    ),
    "test_cam_stream_lss.py": (
        "Validates the Lift-Splat-Shoot camera-to-BEV encoder that lifts "
        "per-camera image features into 3D using depth distributions and splats "
        "them onto a unified bird's-eye-view grid."
    ),
    "test_cam_stream_lss_quickcumsum.py": (
        "Validates the QuickCumsum voxel-pooling operation that efficiently "
        "aggregates camera-lifted point features into BEV voxels using a single "
        "cumulative-sum pass over sorted rank indices."
    ),
    "test_cbnet.py": (
        "Validates the Composite Backbone Swin Transformer (CBNet) that couples "
        "two Swin Transformer backbones with cross-backbone feature exchange via "
        "learned linear projections, producing four pyramid-level outputs."
    ),
    "test_clip_sigmoid.py": (
        "Validates the clamped sigmoid function (`Sigmoid → Clip`) that produces "
        "numerically stable probability outputs in `[eps, 1-eps]` to prevent "
        "infinite focal-loss values in TransFusionHead."
    ),
    "test_coord_transform.py": (
        "Validates the 3D point cloud coordinate transformation helpers "
        "(`apply_3d_transformation`, `extract_2d_info`) that apply sequential "
        "augmentation ops (T, S, R, HF, VF) to align LiDAR and camera coordinate "
        "systems."
    ),
    "test_ealss.py": (
        "Validates the full EA-LSS multi-modal 3D detector that fuses camera "
        "(CBSwinTransformer → FPNC → LSS) and LiDAR (VFE → SparseEncoder → "
        "SECOND → SECONDFPN) BEV streams via concatenation and TransFusionHead."
    ),
    "test_ealss_cam.py": (
        "Validates the camera-only EA-LSS detector variant with a wider image "
        "feature channel (imc=512) and optional LiDAR fusion toggle "
        "(`lc_fusion`), producing the same detection prediction dict as the full "
        "model."
    ),
    "test_ffn.py": (
        "Validates the multi-head feed-forward prediction network that decodes "
        "BEV transformer query features into per-attribute detection outputs "
        "(center, height, dim, rot, vel, heatmap) via stacked 1D point-wise "
        "convolutions."
    ),
    "test_fpn.py": (
        "Validates the Feature Pyramid Network neck that builds a multi-scale "
        "feature hierarchy from backbone outputs via lateral 1×1 convolutions, "
        "top-down upsampling/addition, and 3×3 FPN convolutions at uniform "
        "channel width."
    ),
    "test_fpnc.py": (
        "Validates the FPN-with-Context neck that extends FPN with global average "
        "pooling context, resizes all pyramid levels to a target resolution, and "
        "reduces them to a single unified feature map for the LiftSplatShoot "
        "module."
    ),
    "test_gaussian.py": (
        "Validates Gaussian heatmap utilities (`gaussian_2d`, "
        "`draw_heatmap_gaussian`, `gaussian_radius`, `GaussianDepthTarget`) used "
        "for heatmap-based label generation and depth supervision target "
        "generation in TransFusionHead."
    ),
    "test_mlp.py": (
        "Validates the MLP module (stacked 1D point-wise Conv+BN+ReLU) and its "
        "foundational `BatchNorm1d` / `ConvModule1d` building blocks that are "
        "reused by FFN, PositionEmbeddingLearned, VoxelEncoder, and "
        "VoxelEncoderUtils."
    ),
    "test_multihead_attention.py": (
        "Validates the custom multi-head attention module used in "
        "TransFusionHead's transformer decoder, supporting both self-attention "
        "and cross-attention between BEV proposals and BEV feature maps."
    ),
    "test_mvx_faster_rcnn.py": (
        "Validates that `MVXFasterRCNN` and `DynamicMVXFasterRCNN` are thin "
        "wrapper classes over `MVXTwoStageDetector` with no additional parameters "
        "of their own, distinguishing static vs. dynamic voxelization modes."
    ),
    "test_mvx_two_stage.py": (
        "Validates the generic multi-modality two-stage detector base class that "
        "assembles arbitrary sub-modules (image backbone, LiDAR encoder, "
        "detection head) and chains them in a pts → img → head forward order."
    ),
    "test_norm.py": (
        "Validates inference-only `NaiveSyncBatchNorm1d/2d/3d` implementations "
        "that perform standard BN using stored running statistics, omitting "
        "distributed AllReduce logic not needed at inference."
    ),
    "test_position_embedding_learned.py": (
        "Validates the learned absolute position embedding module that maps "
        "3D/6D query coordinates to positional encodings via a two-layer 1D "
        "convolutional network for use in TransFusionHead's transformer decoder."
    ),
    "test_se_block.py": (
        "Validates the Squeeze-and-Excitation Block that learns channel-wise "
        "feature recalibration via global average pooling and a Conv2d+Sigmoid "
        "gate, used optionally in the EALSS BEV fusion stage."
    ),
    "test_second.py": (
        "Validates the SECOND 2D convolutional backbone that processes LiDAR BEV "
        "pseudo-images through strided `SECONDStage` blocks to extract three "
        "progressively downsampled multi-scale feature maps."
    ),
    "test_second_fpn.py": (
        "Validates the SECONDFPN neck that upsamples multi-scale SECOND backbone "
        "outputs via transposed convolutions and concatenates them into a single "
        "unified BEV feature map for camera-LiDAR fusion."
    ),
    "test_swin_transformer.py": (
        "Validates the Swin Transformer backbone with shifted-window "
        "self-attention, hierarchical patch merging, and relative position bias, "
        "producing four pyramid-level outputs at scales H/4 through H/32."
    ),
    "test_transfusion_bbox_coder.py": (
        "Validates the TransFusion bounding box coder that encodes GT boxes to "
        "regression targets (NumPy) and decodes TransFusionHead outputs back to "
        "3D bounding box coordinates `(cx, cy, cz, w, l, h, yaw)`."
    ),
    "test_transfusion_detector.py": (
        "Validates the `TransFusionDetector` wrapper that extends "
        "`MVXTwoStageDetector` to route BEV features through `TransFusionHead`, "
        "with passthrough behavior when no head is attached."
    ),
    "test_transfusion_head.py": (
        "Validates the full TransFusion detection head that generates query "
        "proposals from heatmap top-k, refines them through transformer decoder "
        "layers, and outputs six per-object prediction tensors (center, height, "
        "dim, rot, vel, heatmap)."
    ),
    "test_voxel_encoder.py": (
        "Validates `HardSimpleVFE` (mean-pooling voxel encoder) and "
        "`HardSimpleVFE_ATT` (temporal-attention + PFN voxel encoder) that "
        "compress raw LiDAR point cloud voxels into fixed-size per-voxel "
        "descriptors."
    ),
    "test_voxel_encoder_utils.py": (
        "Validates the VFE building blocks (`get_paddings_indicator`, `VFELayer`, "
        "`DynamicVFELayer`) that provide point validity masking and per-voxel "
        "feature projection for static and dynamic voxelization pipelines."
    ),
}

_DEFAULT_TEST_DESCRIPTION = (
    "Runs EA-LSS-related validation tests and reports pass/fail status."
)

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once across all workload instances
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: str | None = None

# ---------------------------------------------------------------------------
# Validation driver
# ---------------------------------------------------------------------------


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
        # Nothing to run – write a placeholder report.
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# EA-LSS Validation Test Report\n\n")
            fh.write("**Result:** No test files found.\n")
        _VALIDATION_DONE = True
        _VALIDATION_MD_PATH = os.path.abspath(md_path)
        return _VALIDATION_MD_PATH

    print(f"\n{'=' * 72}")
    print(f"[EALSSValidation] Running {len(test_files)} validation test(s)…")
    print(f"{'=' * 72}\n")

    results: list[tuple[str, int, str]] = []
    for rel_path, fpath in test_files:
        fname = os.path.basename(fpath)
        print(f"  › Running {fname} …", end="", flush=True)
        rc, output = run_subprocess(fpath, _POLARIS_ROOT)
        label = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {label}")
        results.append((fname, rc, output))

    write_simple_markdown(
        md_path,
        "EA-LSS Validation Test Report",
        results,
        _TEST_DESCRIPTIONS,
        default_description=_DEFAULT_TEST_DESCRIPTION,
    )

    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"[EALSSValidation] {passed}/{total} tests passed.")
    print(f"[EALSSValidation] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# TT-Polaris workload class
# ---------------------------------------------------------------------------


class EALSSValidation(EALSS):

    def create_input_tensors(self) -> None:
        # Run validation tests (no-op on subsequent instances).
        _run_validation_tests()
        # Create the TTSim input tensors for the EALSS forward graph.
        super().create_input_tensors()
