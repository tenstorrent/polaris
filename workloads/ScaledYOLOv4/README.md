# ScaledYOLOv4 Workload Validation

This directory contains ScaledYOLOv4 model implementation and validation scripts for verifying ttsim outputs against PyTorch.

---

## 📁 Directory Structure

```
ScaledYOLOv4/
├── models/
│   ├── common.py
│   ├── experimental.py
│   ├── yolo.py
│   ├── yolov4-csp.yaml
│   ├── yolov4-p5.yaml
│   ├── yolov4-p6.yaml
│   ├── yolov4-p7.yaml
│   └── utils/
│
├── reference/
    ├── common/
    │   ├── run_all.py
    │   ├── conv.py, bottleneck.py, ...
    │   └── results.md
    │
    └── yolo/
        ├── run_all.py
        ├── detect.py
        └── model.py
```

---

## Validation Types

### `reference/` — ttsim vs PyTorch

Validates that the **ttsim** (Tenstorrent Simulator) implementation produces outputs that match PyTorch.

- **What it does**: Builds the module using ttsim ops, transfers weights from PyTorch, runs inference, and compares outputs.
- **Use case**: Ensuring ttsim correctly implements each ScaledYOLOv4 layer/module.

---

## How to Run the Scripts

### ttsim vs PyTorch Validation

**Common Modules:**
```bash
python -m workloads.ScaledYOLOv4.reference.common.run_all
```

**Final YOLO Modules:**
```bash
python -m workloads.ScaledYOLOv4.reference.yolo.run_all
```

### Run Individual Module Validation

You can run individual module scripts directly:

```bash
python -m workloads.ScaledYOLOv4.reference.common.conv
```

---

## Output & Results

Each `run_all.py` script generates:

1. **Console output** — Real-time validation status (PASS/FAIL)
2. **Markdown report** — Saved to `results.md` in the respective folder

### Sample Results Summary

```
# ScaledYOLOv4 ttsim vs PyTorch Validation Results

**18/18 modules passed** | Tolerances: rtol=1e-05, atol=1e-06

## Conv - PASS
- Input Shape: [1, 32, 16, 16]
- PyTorch Output: [1, 64, 16, 16]
- ttsim Output: [1, 64, 16, 16]
- Max Diff: 1.55e-06
```

---

## Modules Validated

| Module          | Description                          |
|-----------------|--------------------------------------|
| `Conv`          | Convolution + BatchNorm + Activation |
| `DWConv`        | Depthwise Convolution                |
| `Bottleneck`    | Residual bottleneck block            |
| `BottleneckCSP` | CSP bottleneck                       |
| `BottleneckCSP2`| CSP bottleneck variant               |
| `Focus`         | Focus layer (space-to-depth)         |
| `SPP`           | Spatial Pyramid Pooling              |
| `SPPCSP`        | CSP variant of SPP                   |
| `VoVCSP`        | VoVNet CSP block                     |
| `Concat`        | Tensor concatenation                 |
| `Upsample`      | Upsampling layer                     |
| `MaxPool`       | Max pooling                          |
| `Flatten`       | Tensor flattening                    |
| `Classify`      | Classification head                  |
| `Detect`        | Detection head                       |
| `Model`         | Full YOLOv4 model                    |

---

## Tolerances

| Validation Type | rtol   | atol   |
|-----------------|--------|--------|
| Common modules  | 1e-5   | 1e-6   |
| YOLO modules    | 1e-4   | 1e-4   |

---

## Notes

- Results files are timestamped when generated
- YOLO model validation requires model config files from `models/`
- `KMP_DUPLICATE_LIB_OK=TRUE` is handled automatically in YOLO scripts
