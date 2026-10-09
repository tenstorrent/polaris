# DiffusionDrive Workload Validation

This directory contains the DiffusionDrive model implementation and validation scripts for verifying ttsim outputs against PyTorch.

---

## 📁 Directory Structure

```
DiffusionDrive/
└── navsim/
    └── agents/
        └── diffusiondrive/
            ├── modules/                          # TTSIM
            │   ├── blocks.py
            │   └── conditional_unet1d_ttsim.py
            │
            ├── transfuser_backbone_ttsim.py
            ├── transfuser_config.py
            ├── transfuser_model_v2_ttsim.py
            │
            └── reference/
                ├── torch_code/                   # Original PyTorch reference code
                │   └── navsim/
                │       ├── agents/
                │       │   └── diffusiondrive/
                │       │       ├── modules/
                │       │       │   ├── blocks.py
                │       │       │   ├── conditional_unet1d.py
                │       │       │   └── multimodal_loss.py
                │       │       ├── transfuser_backbone.py
                │       │       ├── transfuser_config.py
                │       │       └── transfuser_model_v2.py
                │       ├── common/               # navsim data classes & enums
                │       └── planning/             # navsim training/evaluation pipeline
                │
                ├── Modules/                      # Comparison tests: modules
                │   ├── blocks/
                │   │   ├── run_all_tests.py
                │   │   ├── comparison_results.md
                │   │   └── test_1_grid_sample_bev_attention.py
                │   │
                │   └── Conditional_unet1d/
                │       ├── run_all_tests.py
                │       ├── comparison_results.md
                │       ├── test_0_sinusoidal_pos_emb.py
                │       ├── test_1_conv1d_block.py
                │       ├── test_2_downsample1d.py
                │       ├── test_3_upsample1d.py
                │       ├── test_4_cond_residual_block.py
                │       └── test_5_conditional_unet1d.py
                │
                ├── Transfuser_backbone/          # Comparison tests: backbone
                │   ├── run_all_tests.py
                │   ├── comparison_results.md
                │   ├── test_1_selfattention.py
                │   ├── test_2_block.py
                │   ├── test_3_gpt.py
                │   ├── test_4_multihead_attention.py
                │   ├── test_5_decoder_layer.py
                │   ├── test_6_decoder.py
                │   └── test_7_transfuser_backbone.py
                │
                └── transfuser_model_v2/          # Comparison tests: full model
                    ├── run_all_tests.py
                    ├── comparison_results.md
                    ├── test_1_agent_head.py
                    ├── test_2_diff_motion_planning.py
                    ├── test_3_modulation_layer.py
                    ├── test_4_custom_transformer_decoder_layer.py
                    ├── test_5_custom_transformer_decoder.py
                    ├── test_6_trajectory_head.py
                    └── test_7_v2_transfuser_full_validation.py
```

---

## Validation Type

### `reference/` — ttsim vs PyTorch

Validates that the **ttsim** (TensTorrent Simulator) implementation produces outputs that match PyTorch.

- **What it does**: Builds each module using ttsim ops, transfers weights from PyTorch, runs inference, and compares outputs.
- **Use case**: Ensuring ttsim correctly implements each DiffusionDrive layer/module.

---

## How to Run the Scripts

### Run All Tests per Module Group

```bash
# BEV attention blocks
python -m workloads.DiffusionDrive.navsim.agents.diffusiondrive.reference.Modules.blocks.run_all_tests

# Conditional UNet 1D modules
python -m workloads.DiffusionDrive.navsim.agents.diffusiondrive.reference.Modules.Conditional_unet1d.run_all_tests

# Transfuser backbone modules
python -m workloads.DiffusionDrive.navsim.agents.diffusiondrive.reference.Transfuser_backbone.run_all_tests

# Full TransfuserV2 model & sub-modules
python -m workloads.DiffusionDrive.navsim.agents.diffusiondrive.reference.transfuser_model_v2.run_all_tests
```

### Run Individual Module Validation

```bash
python -m workloads.DiffusionDrive.navsim.agents.diffusiondrive.reference.Transfuser_backbone.test_7_transfuser_backbone

python -m workloads.DiffusionDrive.navsim.agents.diffusiondrive.reference.Modules.Conditional_unet1d.test_5_conditional_unet1d

python -m workloads.DiffusionDrive.navsim.agents.diffusiondrive.reference.transfuser_model_v2.test_7_v2_transfuser_full_validation
```

---

## Output & Results

Each `run_all_tests.py` script generates:

1. **Console output** — Real-time validation status (PASS/FAIL)
2. **Markdown report** — Saved to `comparison_results.md` in the respective folder

---

# DiffusionDrive Architecture Documentation

---

## Core Modules

## Modules (`navsim/agents/diffusiondrive/modules/`)

| Module | Description |
|--------|-------------|
| `blocks.py` | `GridSampleCrossBEVAttention_TTSIM` — BEV cross-attention (TTSIM only; torch reference: `reference/torch_code/navsim/agents/diffusiondrive/modules/blocks.py`) |
| `conditional_unet1d_ttsim.py` | Conditional UNet 1D diffusion backbone (TTSIM only; torch reference: `reference/torch_code/navsim/agents/diffusiondrive/modules/conditional_unet1d.py`) |

---

## Transfuser Backbone (`transfuser_backbone_ttsim.py`)

| Module | Description |
|--------|-------------|
| `SelfAttention` | Self-attention block for sequence modelling |
| `Block` | Transformer block (attention + MLP) |
| `GPT` | GPT-style transformer for feature fusion |
| `MultiheadAttention` | Standard multi-head attention |
| `DecoderLayer` | Single transformer decoder layer |
| `Decoder` | Full transformer decoder stack |
| `TransfuserBackbone` | Complete backbone fusing camera + LiDAR features |

---

## Conditional UNet 1D (`modules/conditional_unet1d_ttsim.py`)

| Module | Description |
|--------|-------------|
| `SinusoidalPosEmb` | Sinusoidal positional embedding for diffusion timestep |
| `Conv1dBlock` | 1D convolution block with GroupNorm |
| `Downsample1d` | 1D downsampling block |
| `Upsample1d` | 1D upsampling block |
| `CondResidualBlock1D` | Conditional residual block for UNet |
| `ConditionalUnet1D` | Full conditional UNet 1D diffusion model |

---

## TransfuserV2 Full Model (`transfuser_model_v2_ttsim.py`)

| Module | Description |
|--------|-------------|
| `AgentHead` | Predicts agent bounding boxes from BEV features |
| `DiffMotionPlanning` | Diffusion-based motion planning head |
| `ModulationLayer` | Feature modulation via FiLM conditioning |
| `CustomTransformerDecoderLayer` | Custom transformer decoder layer |
| `CustomTransformerDecoder` | Full custom transformer decoder |
| `TrajectoryHead` | Trajectory prediction head |
| `TransfuserV2` | Full end-to-end DiffusionDrive model |

---

### Run with TT-Polaris

```bash
python polaris.py -w config/ip_workloads.yaml -a config/all_archs.yaml -m config/wl2archmapping.yaml --filterwlg ttsim --filterwl DiffusionDrive -o ODDIR_DiffusionDrive -s SIMPLE_RUN --outputformat json
```

### Export to ONNX

```bash
python polaris.py -w config/ip_workloads.yaml -a config/all_archs.yaml -m config/wl2archmapping.yaml --filterwlg ttsim --filterwl DiffusionDrive -o ODDIR_DiffusionDrive -s SIMPLE_RUN --outputformat json --dump_ttsim_onnx
```

---

## Notes

- Each ttsim module file contains the ttsim implementation only. The original PyTorch (torch) reference code lives under `reference/torch_code/navsim/` (with `agents/`, `common/`, and `planning/` sub-packages), preserving the full original `navsim` package structure.
- `multimodal_loss.py` and `scheduler.py` are torch-only and live at `reference/torch_code/navsim/agents/diffusiondrive/modules/`; they are not used by any ttsim file.
- Results files are generated with timestamps by each `run_all_tests.py`
