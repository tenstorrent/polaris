# ScaledYOLOv4 YOLO Modules - ttsim vs PyTorch Validation Results

Generated: 2026-02-03 17:07:15

**2/2 modules passed** | Tolerances: rtol=0.0001, atol=0.0001

---

## Detect - PASS

**Input**
- Shapes: `[[1, 128, 8, 8], [1, 256, 4, 4], [1, 512, 2, 2]]`
- Values: `multi-scale feature maps`

**Output Shapes**
- Layer 0: PyTorch `[1, 3, 8, 8, 85]`, ttsim `[1, 3, 8, 8, 85]`
- Layer 1: PyTorch `[1, 3, 4, 4, 85]`, ttsim `[1, 3, 4, 4, 85]`
- Layer 2: PyTorch `[1, 3, 2, 2, 85]`, ttsim `[1, 3, 2, 2, 85]`

**Output Values**
- Layer 0:
  - PyTorch: `[-0.699101, -0.143604, 0.615410, -0.373563, 0.992676, ...]`
  - ttsim: `[-0.699101, -0.143604, 0.615410, -0.373563, 0.992676, ...]`
- Layer 1:
  - PyTorch: `[-0.281496, -0.464550, -0.471637, -0.348027, -0.253249, ...]`
  - ttsim: `[-0.281496, -0.464550, -0.471637, -0.348027, -0.253249, ...]`
- Layer 2:
  - PyTorch: `[-0.717933, -1.872519, -0.145058, -0.602282, 1.310944, ...]`
  - ttsim: `[-0.717933, -1.872520, -0.145058, -0.602282, 1.310944, ...]`

**Per-Layer Results (3/3 passed)**

| Layer | Type | PyTorch Shape | ttsim Shape | Max Diff | Status |
|-------|------|---------------|-------------|----------|--------|
| P3 | Detect | `[1, 3, 8, 8, 85]` | `[1, 3, 8, 8, 85]` | 4.77e-07 | ✓ |
| P4 | Detect | `[1, 3, 4, 4, 85]` | `[1, 3, 4, 4, 85]` | 4.77e-07 | ✓ |
| P5 | Detect | `[1, 3, 2, 2, 85]` | `[1, 3, 2, 2, 85]` | 1.07e-06 | ✓ |

---

## Model - PASS

**Input**
- Shape: `[1, 3, 64, 64]`
- Values: `[0.496714, -0.138264, 0.647689, 1.523030, -0.234153, ...] (max=3.93e+00, mean=7.96e-01)`

**Configuration**
- Config file: `yolov4-p5.yaml`
- Number of layers: 32
- Weights transferred: 783

**Output Shapes**
- Layer 0: PyTorch `[1, 4, 8, 8, 85]`, ttsim `[1, 4, 8, 8, 85]`
- Layer 1: PyTorch `[1, 4, 4, 4, 85]`, ttsim `[1, 4, 4, 4, 85]`
- Layer 2: PyTorch `[1, 4, 2, 2, 85]`, ttsim `[1, 4, 2, 2, 85]`

**Output Values**
- Layer 0:
  - PyTorch: `[2.07e-09, 2.78e-08, -2.99e-08, -2.22e-09, 3.05e-08, ...] (max=1.88e-07, mean=2.78e-08)`
  - ttsim: `[2.07e-09, 2.80e-08, -3.01e-08, -2.23e-09, 3.07e-08, ...] (max=1.90e-07, mean=2.81e-08)`
- Layer 1:
  - PyTorch: `[9.74e-12, -1.03e-10, 1.25e-10, -1.93e-10, 4.99e-10, ...] (max=1.33e-09, mean=2.09e-10)`
  - ttsim: `[9.73e-12, -1.04e-10, 1.27e-10, -1.95e-10, 5.04e-10, ...] (max=1.34e-09, mean=2.11e-10)`
- Layer 2:
  - PyTorch: `[-3.15e-12, 3.28e-12, 1.24e-12, 6.84e-12, 1.31e-12, ...] (max=2.07e-11, mean=5.36e-12)`
  - ttsim: `[-3.19e-12, 3.32e-12, 1.25e-12, 6.93e-12, 1.33e-12, ...] (max=2.10e-11, mean=5.43e-12)`

**Per-Layer Results (3/3 passed)**

| Layer | Type | PyTorch Shape | ttsim Shape | Max Diff | Status |
|-------|------|---------------|-------------|----------|--------|
| Detect_0 | Detect | `[1, 4, 8, 8, 85]` | `[1, 4, 8, 8, 85]` | 1.61e-09 | ✓ |
| Detect_1 | Detect | `[1, 4, 4, 4, 85]` | `[1, 4, 4, 4, 85]` | 1.42e-11 | ✓ |
| Detect_2 | Detect | `[1, 4, 2, 2, 85]` | `[1, 4, 2, 2, 85]` | 2.59e-13 | ✓ |

---
