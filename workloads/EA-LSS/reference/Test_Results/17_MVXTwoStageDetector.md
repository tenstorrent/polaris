# Module 17: MVXTwoStageDetector ✅

**Location**: `ttsim_modules/mvx_two_stage.py`
**Original**: `mmdet3d/models/detectors/mvx_two_stage.py`

## Description
Generic multi-modality two-stage 3D detector base class. Accepts pre-built TTSim sub-module instances (image backbone, image neck, LiDAR voxel encoder, LiDAR backbone, LiDAR neck, detection head) and aggregates their parameter counts. The `__call__` method chains available sub-modules in a pts → img → detection-head order.

## Purpose
Provides the base detector infrastructure inherited by EALSS, EALSS_CAM, TransFusionDetector, MVXFasterRCNN, and DynamicMVXFasterRCNN. Decouples sub-module assembly from the concrete forward implementations in derived classes.

## Module Specifications
- **Input**: BEV features or raw sensor data
- **Output**: Prediction dict from `pts_bbox_head` (if attached), else passthrough
- **Parameters**: Sum of all provided sub-module parameter counts
- **Optional sub-modules**: `img_backbone`, `img_neck`, `pts_voxel_encoder`, `pts_backbone`, `pts_neck`, `pts_bbox_head`

## Validation Methodology
The module is validated through three tests:
1. **No sub-modules**: Confirms params=0 and passthrough when no sub-modules attached
2. **With pts_bbox_head only**: Verifies `__call__` delegates to head, params are correct
3. **Param sum across variants**: Validates total params match child sub-module sum

## Validation Results

**Test File**: `Validation/test_mvx_two_stage.py`

```
================================================================================
Test 1: MVXTwoStageDetector with no sub-modules
================================================================================

✓ No sub-modules: params==0, passthrough OK  p=0, out=[1, 384, 32, 32]
--------------------------------------------------------------------------------

================================================================================
Test 2: MVXTwoStageDetector with pts_bbox_head only
================================================================================

✓ MVX delegates __call__ to head, params correct
  p=938654, keys=['center', 'height', 'dim', 'rot', 'vel', 'heatmap']
--------------------------------------------------------------------------------

================================================================================
Test 3: MVXTwoStageDetector param sum across head variants
================================================================================

✓ Param sum matches child  p=289812
--------------------------------------------------------------------------------

============================================================
RESULTS: 3/3 tests passed
```

## Summary Table

| Test | Description | Result |
|------|-------------|--------|
| Test 1 | No sub-modules: params=0, passthrough | ✅ PASS |
| Test 2 | With head: delegates call + params=938,654 | ✅ PASS |
| Test 3 | Param sum matches child modules | ✅ PASS |

**Status**: All 3/3 tests passed ✅
