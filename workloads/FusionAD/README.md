# FusionAD Workload Validation

This directory contains the FusionAD end-to-end autonomous driving model implementation and validation scripts for verifying ttsim outputs against PyTorch.

---

## 📁 Directory Structure

```
FusionAD/
├── projects/
│   └── mmdet_plugin/
│       ├── core/
│       │   └── bbox/
│       │       └── util.py
│       │
│       ├── fusionad/
│       │   ├── dense_heads/
│       │   │   ├── motion_head.py
│       │   │   ├── occ_head.py
│       │   │   ├── panseg_head.py
│       │   │   ├── planning_head.py
│       │   │   ├── track_head.py
│       │   │   │
│       │   │   ├── motion_head_plugin/
│       │   │   │   ├── base_motion_head.py
│       │   │   │   ├── modules.py
│       │   │   │   └── motion_deformable_attn.py
│       │   │   │
│       │   │   ├── occ_head_plugin/
│       │   │   │   ├── modules.py
│       │   │   │   └── utils.py
│       │   │   │
│       │   │   ├── seg_head_plugin/
│       │   │   │   ├── seg_deformable_transformer.py
│       │   │   │   ├── seg_detr_head.py
│       │   │   │   └── seg_mask_head.py
│       │   │   │
│       │   │   └── track_head_plugin/
│       │   │       ├── modules.py
│       │   │       ├── tracker.py
│       │   │       └── track_instance.py
│       │   │
│       │   ├── detectors/
│       │   │   ├── fusionad_e2e.py
│       │   │   └── fusionad_track.py
│       │   │
│       │   └── modules/
│       │       ├── builder_utils.py
│       │       ├── custom_base_transformer_layer.py
│       │       ├── decoder.py
│       │       ├── encoder.py
│       │       ├── fpn.py
│       │       ├── init_utils.py
│       │       ├── multi_scale_deformable_attn_function.py
│       │       ├── multihead_attention.py
│       │       ├── pts_cross_attention.py
│       │       ├── resnet.py
│       │       ├── spatial_cross_attention.py
│       │       ├── temporal_self_attention.py
│       │       └── transformer.py
│       │
│       └── models/
│           ├── backbones/
│           │   ├── sparse_encoder_hd.py
│           │   └── vovnet.py
│           │
│           └── utils/
│               ├── bricks.py
│               ├── functional.py
│               └── grid_mask.py
│
├── reference/
│   └── mmdet_plugin/
│       ├── core/
│       │   └── bbox/
│       │       ├── run_all.py
│       │       ├── comparison_results.md
│       │       ├── test_bbox_util.py
│       │       └── test_util.py
│       │
│       ├── fusionad/
│       │   ├── dense_heads/
│       │   │   ├── run_all.py
│       │   │   ├── comparison_results.md
│       │   │   ├── test_motion_head.py
│       │   │   ├── test_occ_head.py
│       │   │   ├── test_panseg_head.py
│       │   │   ├── test_planning_head.py
│       │   │   ├── test_track_head.py
│       │   │   │
│       │   │   ├── motion_head_plugin/
│       │   │   │   ├── run_all.py
│       │   │   │   ├── comparison_results.md
│       │   │   │   ├── test_base_motion_head.py
│       │   │   │   ├── test_modules.py
│       │   │   │   └── test_motion_deformable_attn.py
│       │   │   │
│       │   │   ├── occ_head_plugin/
│       │   │   │   ├── run_all.py
│       │   │   │   ├── comparison_results.md
│       │   │   │   └── test_occ_modules.py
│       │   │   │
│       │   │   ├── seg_head_plugin/
│       │   │   │   ├── run_all.py
│       │   │   │   ├── comparison_results.md
│       │   │   │   ├── test_seg_deformable_transformer.py
│       │   │   │   ├── test_seg_detr_head.py
│       │   │   │   └── test_seg_mask_head.py
│       │   │   │
│       │   │   └── track_head_plugin/
│       │   │       ├── run_all.py
│       │   │       ├── comparison_results.md
│       │   │       └── test_modules.py
│       │   │
│       │   ├── detectors/
│       │   │   ├── run_all.py
│       │   │   ├── comparison_results.md
│       │   │   ├── test_fusionad_e2e.py
│       │   │   └── test_fusionad_track.py
│       │   │
│       │   └── modules/
│       │       ├── run_all.py
│       │       ├── comparison_results.md
│       │       ├── test_custom_base_transformer_layer.py
│       │       ├── test_decoder.py
│       │       ├── test_encoder.py
│       │       ├── test_multi_scale_deformable_attn_function.py
│       │       ├── test_pts_cross_attention.py
│       │       ├── test_spatial_cross_attention.py
│       │       ├── test_temporal_self_attention.py
│       │       └── test_transformer.py
│       │
│       └── models/
│           ├── backbones/
│           │   ├── run_all.py
│           │   ├── comparison_results.md
│           │   ├── test_sparse_encoder_hd.py
│           │   └── test_vovnet.py
│           │
│           └── utils/
│               ├── run_all.py
│               ├── comparison_results.md
│               ├── test_bricks.py
│               ├── test_functional.py
│               └── test_grid_mask.py
│
└── ttsim_models/
```

---

## Validation Type

### `reference/` — ttsim vs PyTorch

Validates that the **ttsim** (TensTorrent Simulator) implementation produces outputs that match PyTorch.

- **What it does**: Builds each module using ttsim ops, transfers weights from PyTorch, runs inference, and compares outputs.
- **Use case**: Ensuring ttsim correctly implements each FusionAD layer/module.

---

## How to Run the Scripts

### Run All Tests per Module Group

```bash
# Core bbox utilities
python -m workloads.FusionAD.reference.mmdet_plugin.core.bbox.run_all

# Dense heads (motion, occ, panoptic seg, planning, track)
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.dense_heads.run_all

# Motion head plugin sub-modules
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.dense_heads.motion_head_plugin.run_all

# Occupancy head plugin sub-modules
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.dense_heads.occ_head_plugin.run_all

# Segmentation head plugin sub-modules
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.dense_heads.seg_head_plugin.run_all

# Track head plugin sub-modules
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.dense_heads.track_head_plugin.run_all

# End-to-end detectors
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.detectors.run_all

# Transformer & attention modules
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.modules.run_all

# Backbone networks (VoVNet, SparseEncoderHD)
python -m workloads.FusionAD.reference.mmdet_plugin.models.backbones.run_all

# Model utilities (bricks, functional, grid mask)
python -m workloads.FusionAD.reference.mmdet_plugin.models.utils.run_all
```

### Run Individual Module Validation

```bash
python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.dense_heads.test_motion_head

python -m workloads.FusionAD.reference.mmdet_plugin.fusionad.modules.test_encoder

python -m workloads.FusionAD.reference.mmdet_plugin.models.backbones.test_vovnet
```

---

## Output & Results

Each `run_all.py` script generates:

1. **Console output** — Real-time validation status (PASS/FAIL)
2. **Markdown report** — Saved to `comparison_results.md` in the respective folder

---

# FusionAD Architecture Documentation

---

## Core Utilities

### Bbox Utilities (`core/bbox/`)

| Module | Description |
|--------|-------------|
| `util` | Bounding box coordinate utilities and transformations |

---

## Backbones (`models/backbones/`)

| Module | Description |
|--------|-------------|
| `VoVNet` | One-Shot Aggregation backbone (VoVNet-99) with eSE attention and depthwise separable convolutions |
| `SparseEncoderHD` | 3D sparse convolution-based LiDAR backbone (4-stage encoder with SubMConv3d approximated as Conv2d) |

---

## Model Utilities (`models/utils/`)

| Module | Description |
|--------|-------------|
| `GridMask` | Grid-based data augmentation mask |
| `functional` | Functional utilities for model operations |
| `bricks` | Profiling utilities (timing decorators, performance maps) |

---

## Transformer & Attention Modules (`fusionad/modules/`)

| Module | Description |
|--------|-------------|
| `ResNet (Bottleneck)` | ResNet-101 backbone with optional DCNv2 deformable convolutions (4 stages: 3/4/23/3 blocks) |
| `FPN` | Feature Pyramid Network — multi-scale feature extraction from [512, 1024, 2048] to uniform 256 channels |
| `PerceptionTransformer` | Top-level orchestrator wiring encoder and decoder pipelines |
| `CanBusMLP` | Feed-forward network encoding CAN-bus signals (ego velocity, heading) |
| `BEVFormerEncoder` | Sequence of encoder layers with 3D reference point generation |
| `BEVFormerLayer` | Single encoder layer: TSA → norm → SCA → norm → FFN → norm |
| `BEVFormerFusionLayer` | Extended encoder layer adding LiDAR cross-attention (pts_cross_attn) |
| `DetectionTransformerDecoder` | Decoder with iterative reference-point refinement |
| `CustomMSDeformableAttention` | Multi-scale deformable cross-attention |
| `MultiheadAttention` | Standard multi-head self/cross-attention |
| `TemporalSelfAttention` | Temporal self-attention for historical BEV feature aggregation |
| `SpatialCrossAttention` | Spatial cross-attention between BEV queries and camera image features |
| `PtsCrossAttention` | Multi-scale deformable cross-attention for LiDAR point features |
| `MyCustomBaseTransformerLayer` | Flexible transformer layer composing attention/FFN/norm in configurable order |
| `MultiScaleDeformableAttnFunction_fp16` | CPU-only multi-scale deformable attention implementation |

---

## Dense Heads (`fusionad/dense_heads/`)

### Track Head

| Module | Description |
|--------|-------------|
| `BEVFormerTrackHead` | 3D object detection head with transformer decoder and iterative bbox refinement |
| `ClsBranch` | Classification branch (Linear → LN → ReLU → ... → Linear) |
| `RegBranch` | Regression branch (Linear → ReLU → ... → Linear) |
| `TrajRegBranch` | Trajectory regression branch |

### Track Head Plugin (`track_head_plugin/`)

| Module | Description |
|--------|-------------|
| `QueryInteractionModule` | Query interaction module for track queries |
| `MemoryBank` | Memory bank for tracking state across frames |
| `RuntimeTrackerBase` | Numpy-based runtime scoring/filtering between frames |
| `Instances` | Track instance data structure |

### Motion Head

| Module | Description |
|--------|-------------|
| `MotionHead` | Trajectory prediction with multimodal motion forecasting |

### Motion Head Plugin (`motion_head_plugin/`)

| Module | Description |
|--------|-------------|
| `BaseMotionHead` | Base class for motion layer construction and anchor loading |
| `MotionTransformerDecoder` | Motionformer decoder for trajectory prediction |
| `TwoLayerMLP` | Linear → ReLU → Linear |
| `TrackQueryFuser` | Linear → LayerNorm → ReLU fuser |
| `TrajClsBranch` | Classification branch with LayerNorm |
| `TrajRegBranch` | Regression branch for trajectory outputs |
| `MotionDeformableAttn` | Deformable attention for motion prediction |

### Panoptic Segmentation Head

| Module | Description |
|--------|-------------|
| `PansegformerHead` | Panoptic segmentation head (inherits from SegDETRHead) |

### Segmentation Head Plugin (`seg_head_plugin/`)

| Module | Description |
|--------|-------------|
| `SegDETRHead` | DETR-based segmentation head (base class) |
| `SegDeformableTransformer` | Deformable transformer for segmentation |
| `SegMaskHead` | Mask prediction head for panoptic segmentation |

### Occupancy Head

| Module | Description |
|--------|-------------|
| `OccHead` | Occupancy prediction with transformer-based future state forecasting |

### Occupancy Head Plugin (`occ_head_plugin/`)

| Module | Description |
|--------|-------------|
| `BevFeatureSlicer` | Grid-sample a sub-region of BEV features |
| `MLP` | Multi-layer perceptron (Linear + ReLU) |
| `SimpleConv2d` | Sequential Conv2d layers with BN + ReLU |
| `CVT_DecoderBlock` | Upsample + Conv + BN with optional residual |
| `CVT_Decoder` | Stack of CVT_DecoderBlocks |
| `UpsamplingAdd` | Upsample + Conv + BN + skip connection |
| `Bottleneck` | Residual bottleneck with optional down/upsample |

### Planning Head

| Module | Description |
|--------|-------------|
| `PlanningHeadSingleMode` | Main planning head for ego-trajectory prediction |
| `PlanningDecoder` | Stack of PlanningDecoderLayers |
| `PlanningDecoderLayer` | Standard post-norm transformer decoder layer |
| `MLPFuser` | Linear → LayerNorm → ReLU fusion module |
| `PlanMLP` | 3-layer MLP for plan-info encoding (37 → 256) |
| `PlanRegBranch` | 2-layer MLP for trajectory regression |

---

## End-to-End Detectors (`fusionad/detectors/`)

| Module | Description |
|--------|-------------|
| `FusionADTrack` | Base tracking detector with BEV encoder, backbone (VoVNet + ResNet-101), FPN neck, and detection head |
| `FusionAD` | Full end-to-end model extending FusionADTrack with seg_head, motion_head, occ_head, and planning_head |

---

### Run with Polaris

```bash
python polaris.py -w config/ip_workloads.yaml -a config/all_archs.yaml -m config/wl2archmapping.yaml --filterwlg ttsim --filterwl FusionAD -o ODDIR_fusionad -s SIMPLE_RUN --outputformat json
```

### Export to ONNX

```bash
python polaris.py -w config/ip_workloads.yaml -a config/all_archs.yaml -m config/wl2archmapping.yaml --filterwlg ttsim --filterwl FusionAD -o ODDIR_fusionad -s SIMPLE_RUN --outputformat json --dump_ttsim_onnx
```

---

## Notes

- Each `projects/` file contains both commented-out PyTorch source code (under `#----------------------------------PyTorch----------------------------------------#`) and the ttsim implementation (under `#-------------------------------TTSIM-------------------------------------------#`)
- Results files are generated with timestamps by each `run_all.py`
