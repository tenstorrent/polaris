# MapTracker Workload Validation

This directory contains the MapTracker model implementation and validation scripts for verifying ttsim outputs against PyTorch.

---

## 📁 Directory Structure

```
MapTracker/
├── plugin/
│   └── models/
│       ├── backbones/
│       │   ├── bevformer_backbone.py
│       │   └── bevformer/
│       │       ├── builder_utils.py
│       │       ├── custom_base_transformer_layer.py
│       │       ├── encoder.py
│       │       ├── grid_mask.py
│       │       ├── multi_scale_deformable_attn.py
│       │       ├── spatial_cross_attention.py
│       │       ├── temporal_net.py
│       │       ├── temporal_self_attention.py
│       │       └── transformer.py
│       │
│       ├── heads/
│       │   ├── MapDetectorHead.py
│       │   └── Map_Seg_Head.py
│       │
│       ├── losses/
│       │   ├── detr_loss.py
│       │   └── seg_loss.py
│       │
│       ├── maper/
│       │   ├── MapTracker.py
│       │   ├── base_mapper.py
│       │   └── vector_memory.py
│       │
│       ├── necks/
│       │   └── gru.py
│       │
│       ├── transformer_utils/
│       │   ├── base_transformer.py
│       │   ├── custom_msdeformable_attention.py
│       │   ├── MapTransformer.py
│       │   └── multihead_attention.py
│       │
│       └── utils/
│           └── query_update.py
│
└── reference/
    ├── comparison_bevformer/
    │   ├── run_all.py
    │   ├── comparison_results.md
    │   ├── test_bevformer_backbone.py
    │   ├── test_bevformer_encoder.py
    │   ├── test_custom_base_transformer_layer.py
    │   ├── test_grid_mask.py
    │   ├── test_multi_scale_deform_attn.py
    │   ├── test_MyResBlock.py
    │   ├── test_placeholder_encoder.py
    │   ├── test_spatial_cross_attention.py
    │   ├── test_temporal_net.py
    │   ├── test_temporal_self_attention.py
    │   └── test_transformer.py
    │
    ├── comparison_heads/
    │   ├── run_all.py
    │   ├── comparison_results.md
    │   ├── test_mapdetectorhead.py
    │   └── test_map_seg_head.py
    │
    ├── comparison_losses/
    │   ├── run_all.py
    │   ├── comparison_results.md
    │   ├── test_detr_loss.py
    │   └── test_seg_loss.py
    │
    ├── comparison_mapers/
    │   ├── run_all.py
    │   ├── comparison_results.md
    │   ├── test_base_mapper.py
    │   ├── test_maptracker.py
    │   ├── test_positional_encoding1d.py
    │   ├── test_upsample_block.py
    │   └── test_vectorinstancememory.py
    │
    ├── comparison_necks/
    │   ├── run_all.py
    │   ├── comparison_results.md
    │   └── test_conv_gru.py
    │
    ├── comparison_transformer_utils/
    │   ├── run_all.py
    │   ├── comparison_results.md
    │   ├── test_custom_msdeformable_attention.py
    │   ├── test_maptransformer.py
    │   └── test_multihead_attention.py
    │
    └── comparison_utils/
        ├── run_all.py
        ├── comparison_results.md
        ├── test_embedder.py
        └── test_motion_mlp.py
```

---

## Validation Type

### `reference/` — ttsim vs PyTorch

Validates that the **ttsim** (TensTorrent Simulator) implementation produces outputs that match PyTorch.

- **What it does**: Builds each module using ttsim ops, transfers weights from PyTorch, runs inference, and compares outputs.
- **Use case**: Ensuring ttsim correctly implements each MapTracker layer/module.

---

## How to Run the Scripts

### Run All Tests per Module Group

```bash
# BEVFormer backbone & encoder modules
python -m workloads.MapTracker.reference.comparison_bevformer.run_all

# Detection & segmentation heads
python -m workloads.MapTracker.reference.comparison_heads.run_all

# Loss functions
python -m workloads.MapTracker.reference.comparison_losses.run_all

# MapTracker main pipeline & sub-modules
python -m workloads.MapTracker.reference.comparison_mapers.run_all

# Neck (ConvGRU)
python -m workloads.MapTracker.reference.comparison_necks.run_all

# Transformer utilities
python -m workloads.MapTracker.reference.comparison_transformer_utils.run_all

# Utility modules (Embedder, MotionMLP)
python -m workloads.MapTracker.reference.comparison_utils.run_all
```

### Run Individual Module Validation

```bash
python -m workloads.MapTracker.reference.comparison_bevformer.test_transformer

python -m workloads.MapTracker.reference.comparison_heads.test_map_seg_head

python -m workloads.MapTracker.reference.comparison_mapers.test_maptracker
```

---

## Output & Results

Each `run_all.py` script generates:

1. **Console output** — Real-time validation status (PASS/FAIL)
2. **Markdown report** — Saved to `comparison_results.md` in the respective folder

---

# MapTracker Architecture Documentation

---

## Core Modules

## Backbone (`plugin/models/backbones/bevformer`)

| Module | Description |
|--------|------------|
| `MyResBlock` | Residual block used in backbone stages |
| `GridMask` | Grid-based data augmentation mask |
| `MultiScaleDeformAttn` | Multi-scale deformable attention |
| `SpatialCrossAttention` | Spatial cross-attention  |
| `TemporalSelfAttention` | Temporal self-attention for BEV feature fusion |
| `TemporalNet` | Temporal network for multi-frame aggregation |
| `CustomBaseTransformerLayer` | Base transformer layer with configurable attention |
| `PlaceholderEncoder` | Simplified encoder for testing |
| `BEVFormerEncoder` | Full BEVFormer encoder  |
| `Transformer (PerceptionTransformer)` | Top-level perception transformer |

## Complete Backbone pipeline (`plugin/models/backbones`)
| `BEVFormerBackbone` | Complete BEVFormer backbone |

---

## Heads (`plugin/models/heads/`)

| Module | Description |
|--------|------------|
| `MapSegHead` | BEV semantic segmentation head |
| `MapDetectorHead` | Vector map detection head with transformer decoder |

---

## Losses (`plugin/models/losses/`)

| Module | Description |
|--------|------------|
| `DETRLoss` | DETR-style set prediction loss |
| `SegLoss` | Segmentation cross-entropy + Dice loss |

---

## MapTracker Pipeline (`plugin/models/mapers/`)

| Module | Description |
|--------|------------|
| `BaseMapper` | Base class for map prediction models |
| `MapTracker` | Full end-to-end pipeline |
| `PositionalEncoding1D` | 1D positional encoding for queries |
| `UpsampleBlock` | Upsampling block for segmentation output |
| `VectorInstanceMemory` | Instance-level memory bank for tracking |

---

## Neck (`plugin/models/necks/`)

| Module | Description |
|--------|------------|
| `ConvGRU` | Convolutional GRU for temporal BEV fusion |

---

## Transformer Utilities (`plugin/models/transformer_utils/`)

| Module | Description |
|--------|------------|
| `CustomMSDeformableAttention` | Custom multi-scale deformable attention |
| `MapTransformer` | Transformer decoder for vector map queries |
| `MultiheadAttention` | Standard multi-head attention module |

---

## Utilities (`plugin/models/utils/`)

| Module | Description |
|--------|------------|
| `Embedder` | Fourier feature embedding (sin/cos encoding) |
| `MotionMLP` | MLP for query propagation using ego-motion |

---

### Run with Polaris

```bash
python polaris.py -w config/ip_workloads.yaml -a config/all_archs.yaml -m config/wl2archmapping.yaml --filterwlg ttsim --filterwl MapTracker -o ODDIR_maptracker -s SIMPLE_RUN --outputformat json
```

### Export to ONNX (Note: The constructed graph is extremely large, and exporting the graph can take up to 30 minutes to complete. This is a very resource-intensive task so please make sure no heavy background processes are running and sufficient RAM is available to execute this command to prevent the process from crashing.)

```bash
python polaris.py -w config/ip_workloads.yaml -a config/all_archs.yaml -m config/wl2archmapping.yaml --filterwlg ttsim --filterwl MapTracker -o ODDIR_maptracker -s SIMPLE_RUN --outputformat json --dump_ttsim_onnx
```
---

## Notes

- Each `plugin/models/` file contains both commented-out PyTorch source code (under `#------PyTorch------`) and the ttsim implementation (under `#------TTSIM------`)
- Results files are generated with timestamps by each `run_all.py`
