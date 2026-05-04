# Module 16: MVXFasterRCNN & DynamicMVXFasterRCNN ✅

**Location**: `ttsim_modules/mvx_faster_rcnn.py`
**Original**: `mmdet3d/models/detectors/mvx_faster_rcnn.py`

## Description
Thin wrapper classes over `MVXTwoStageDetector` for multi-modality 3D detection with voxel-based feature extraction. `MVXFasterRCNN` uses static voxelization while `DynamicMVXFasterRCNN` uses dynamic voxelization. Neither class introduces any additional learnable parameters — both fully inherit from `MVXTwoStageDetector`.

## Purpose
Provides the standard mmdet3d detector interfaces for the MVX (Multi-Modality VoxelNet) Faster R-CNN variants. Acts as named detector classes that route forward calls through the common `MVXTwoStageDetector` infrastructure.

## Module Specifications
- **Input**: BEV feature tensor or raw multi-modal data
- **Output**: Prediction dict `{center, height, dim, rot, vel, heatmap}`
- **Parameters**: Equal to the `pts_bbox_head` attached (inherited from MVXTwoStageDetector)
- **MVXFasterRCNN**: static voxelization
- **DynamicMVXFasterRCNN**: dynamic voxelization (same params)

## Validation Methodology
The module is validated through three tests:
1. **MVXFasterRCNN forward**: Confirms output is a prediction dict with canonical keys
2. **DynamicMVXFasterRCNN forward**: Same check for the dynamic variant with B=2
3. **Param delegation**: Verifies `analytical_param_count()` equals head params

## Validation Results

**Test File**: `Validation/test_mvx_faster_rcnn.py`

```
================================================================================
Test 1: MVXFasterRCNN forward shape (dict output)
================================================================================

✓ MVXFasterRCNN output is prediction dict
  keys=['center', 'height', 'dim', 'rot', 'vel', 'heatmap']
--------------------------------------------------------------------------------

================================================================================
Test 2: DynamicMVXFasterRCNN forward shape
================================================================================

✓ DynamicMVXFasterRCNN output is prediction dict
  B=2 keys=['center', 'height', 'dim', 'rot', 'vel', 'heatmap']
--------------------------------------------------------------------------------

================================================================================
Test 3: MVXFasterRCNN params == head params
================================================================================

✓ param_count delegated to head  model=938654, head=938654
--------------------------------------------------------------------------------

============================================================
RESULTS: 3/3 tests passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | MVXFasterRCNN prediction dict | ✅ PASS |
| Test 2 | DynamicMVXFasterRCNN (B=2) | ✅ PASS |
| Test 3 | Param count = head params (938,654) | ✅ PASS |

**Status**: All 3/3 tests passed ✅
