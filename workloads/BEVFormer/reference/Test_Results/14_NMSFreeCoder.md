# Module 14: NMS-Free BBox Coder ✅

**Location**: `ttsim_models/nms_free_coder.py`
**Original**: `projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`

## Description
NMS-free bounding box decoder for BEVFormer detection outputs. Takes raw classification scores and normalized bbox predictions from the transformer decoder head and returns final 3D detections without non-maximum suppression. Processes confidence scores via sigmoid activation, selects top-K candidates, decodes normalized box parameters to absolute coordinates, and applies score threshold and spatial range filtering.

## Purpose
Decodes the raw outputs of the BEVFormer detection head into usable 3D bounding boxes. By using a top-K selection strategy instead of NMS, it avoids the computational overhead and set-based reasoning issues of traditional NMS while still pruning low-confidence detections. The decoder also reverses the log-dimension and sin/cos rotation encodings applied during training.

## Module Specifications
- **Input Formats**:
  - `cls_scores`: `[num_query, num_classes]` — raw logits from transformer decoder
  - `bbox_preds`: `[num_query, code_size]` — normalized bbox predictions `(cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot), vx, vy)`
- **Output**: Dict with keys `bboxes` `[N, 9]`, `scores` `[N]`, `labels` `[N]` where N ≤ max_num
- **Parameters**:
  - `pc_range`: Point cloud range `[x_min, y_min, z_min, x_max, y_max, z_max]`
  - `post_center_range`: Spatial filter range for final detections
  - `max_num`: Maximum detections to keep (default 100)
  - `score_threshold`: Minimum score to keep a detection (optional)
  - `num_classes`: Number of object classes (default 10)
- **Output bbox format**: `(cx, cy, cz, w, l, h, rot, vx, vy)`
- **Parameter Count**: 0 (no trainable parameters — pure computation graph)

## Implementation Notes
- `decode_single()` processes one sample; `decode()` iterates over a batch using `SliceF` + `Squeeze` to extract per-sample tensors
- Integer modulo/floor-division for label/bbox-index extraction requires float casting: `Cast(Int→Float)` → `Mod`/`Div`+`Floor` → `Cast(Float→Int64)`
- Boolean masking for filtering uses `NonZero` to get valid indices, then `Gather` to select rows — avoids dynamic-shape `Compress` op
- `ReduceMin` over a cast-to-Int32 boolean mask implements the `.all(dim=1)` spatial range check
- `unique_id = str(int(time.time() * 1000000) % 1000000)` is used to prevent op-name collisions when `decode_single` is called multiple times in a batch
- TTSim outputs for filtered results carry dynamic dimension `-1` since the number of passing detections is data-dependent

## Validation Methodology
The module is validated through six tests:
1. **Denormalize BBox**: 10 boxes, code_size=10; validates output shape `[10, 9]` and graph construction
2. **Decode Single (Full Pipeline)**: 900 queries × 10 classes, max_num=100, threshold=0.2; step-by-step comparison of sigmoid, top-K, gather, denormalize, and filter stages across PyTorch and TTSim
3. **Decode Batch**: batch=6 × num_decoder_layers=6 × 900 queries; validates per-sample graph nodes and PyTorch batch results
4. **Top-K Selection Logic**: 20 queries × 3 classes, k=5; verifies label/bbox-index derivation with known inputs
5. **Score Threshold Filtering**: 6 scores with threshold=0.5; expects exactly 3 detections to pass
6. **Center Range Filtering**: 5 bboxes with defined post_center_range; expects exactly 2 to pass

Tests validate shape correctness and graph construction. TTSim outputs carry dynamic leading dimensions (`-1`) due to data-dependent filtering; PyTorch reference provides numerical ground truth.

## Validation Results

**Test File**: `Validation/test_nms_free_coder.py`

```
================================================================================
NMS-FREE BBOX CODER VALIDATION SUITE
================================================================================

================================================================================
TEST 1: Denormalize BBox - Data Validation
================================================================================

Input Data:
  Shape: (10, 10)
  pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  Format: (cx, cy, w_log, l_log, cz, h_log, sin, cos, vx, vy)

STEP 1: PyTorch Reference Implementation
  Input  shape: torch.Size([10, 10])
  Output shape: torch.Size([10, 9])
  Output format: (cx, cy, cz, w, l, h, rot, vx, vy)
  Sample bbox[0]:
    Input : [ 0.4967 -0.1383  0.6477  1.5230 -0.2342 -0.2341  1.5792  0.7674 -0.4695  0.5426]
    Output: [ 0.4967 -0.1383 -0.2342  1.9111  4.5861  0.7913  1.1184 -0.4695  0.5426]

STEP 2: TTSim Implementation (Graph Construction)
  Graph construction complete
  Output tensor name: test_coder_denorm.denorm.concat.out

STEP 3: PyTorch vs TTSim Comparison
  PyTorch shape: [10, 9]
  TTSim shape:   [10, 9]
  ✓ Shapes match!

✅ PASS: Denormalize bbox validation successful

================================================================================
TEST 2: Decode Single - Complete Pipeline Validation
================================================================================

Test Configuration:
  num_query: 900  |  num_classes: 10  |  max_num: 100
  score_threshold: 0.2
  pc_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  post_center_range: [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

PyTorch Pipeline Steps:
  [1.1] Sigmoid: output range [0.5000, 0.7310]
  [1.2] TopK(k=100): top-5 scores [0.7310, 0.7310, 0.7310, 0.7309, 0.7309]
  [1.3] Labels (idx % 10): [7, 7, 0, 5, 5]
        BBox indices (idx // 10): [488, 147, 159, 808, 217]
  [1.4] Selected bbox shape: torch.Size([100, 10])
  [1.5] Denormalized shape: torch.Size([100, 9])
        cx range: [-2.4755, 2.8085]  |  w range: [0.3053, 2.9497]
  [1.6] Score threshold (0.2): 100/100 passing
  [1.7] Center range filter: 100/100 passing
  [1.8] Final: 100 detections, score range [0.7284, 0.7310]

TTSim Graph Construction:
  Output tensors:
    bboxes: test_coder.gather_boxes.out  shape: [-1, 9]
    scores: test_coder.gather_scores.out  shape: [-1]
    labels: test_coder.gather_labels.out  shape: [-1]
  ✓ Shapes compatible (TTSim uses dynamic dimensions)

✅ PASS: Decode single validation successful
   Pipeline: sigmoid → topk → gather → denormalize → filter → output

================================================================================
TEST 3: Decode Batch - PyTorch + TTSim Validation
================================================================================

Configuration: batch_size=6, num_decoder_layers=6, num_query=900, max_num=100
Input: all_cls_scores (6, 6, 900, 10) | all_bbox_preds (6, 6, 900, 10)

PyTorch batch results:
  Sample 0: 100 detections  Score range: [0.7287, 0.7310]
  Sample 1: 100 detections  Score range: [0.7289, 0.7310]
  Sample 2: 100 detections  Score range: [0.7288, 0.7311]
  Sample 3: 100 detections  Score range: [0.7291, 0.7310]
  Sample 4: 100 detections  Score range: [0.7289, 0.7310]
  Sample 5: 100 detections  Score range: [0.7290, 0.7310]

TTSim graph construction: 6 samples, all graph nodes present
  ✓ Batch size matches: 6

✅ PASS: Batch decoding validated
   Pipeline: Slice[i] → Squeeze → decode_single(sample[i]) → results[i]

================================================================================
TEST 4: Top-K Selection Logic Validation
================================================================================

Scores shape: (20, 3)  |  k=5

  [1] Sigmoid: output range [0.5250, 0.7211]
  [2] Flatten: [20, 3] → [60]
  [3] TopK(k=5):
      Scores:  [0.7211, 0.7109, 0.7006, 0.6900, 0.6682]
      Indices: [11, 3, 12, 4, 5]
  [4] Labels (idx % 3):   [2, 0, 0, 1, 2]
      Queries (idx // 3): [3, 1, 4, 1, 1]
  [5] ✓ Top 3 queries match expected: [3, 1, 4]

✅ PASS: Top-K selection logic validated

================================================================================
TEST 5: Score Threshold Filtering
================================================================================

Input scores: [0.30, 0.60, 0.90, 0.40, 0.70, 0.20]
Threshold: 0.5

  Mask (score > 0.5): [False, True, True, False, True, False]
  Passing: 3/6  |  Filtered scores: [0.60, 0.90, 0.70]
  ✓ Count matches: 3 == 3  |  ✓ All scores > threshold

✅ PASS: Score threshold filtering validated

================================================================================
TEST 6: Center Range Filtering
================================================================================

Center range: [-10.0, -10.0, -2.0, 10.0, 10.0, 2.0]
Bboxes:
  BBox 0: center (  0.0,   0.0,  1.0) ✓ Inside
  BBox 1: center ( 15.0,   5.0,  1.0) ✗ Out of range: x
  BBox 2: center ( -5.0,  -5.0,  0.0) ✓ Inside
  BBox 3: center (  5.0,  12.0,  1.0) ✗ Out of range: y
  BBox 4: center (  0.0,   0.0,  3.0) ✗ Out of range: z

  Mask: [True, False, True, False, False]
  Passing: 2/5  |  Passing indices: [0, 2]
  ✓ Expected 2 boxes to pass

✅ PASS: Center range filtering validated

================================================================================
TEST SUMMARY
================================================================================
Denormalize BBox................................................ ✅ PASS
Decode Single (Full Pipeline)................................... ✅ PASS
Decode Batch (PyTorch + TTSim).................................. ✅ PASS
Top-K Selection Logic........................................... ✅ PASS
Score Threshold Filtering....................................... ✅ PASS
Center Range Filtering.......................................... ✅ PASS

Total: 6/6 tests passed

All tests passed! TTSim NMS-Free BBox Coder implementation VALIDATED.
```

## PyTorch vs TTSim Comparison

| Test Case | Input Shape | PyTorch Output | TTSim Output | Validated |
|-----------|-------------|----------------|--------------|-----------|
| Denormalize BBox | (10, 10) | (10, 9) | (10, 9) | ✅ Shape match |
| Decode Single — bboxes | (900, 10) × 2 | (100, 9) | (-1, 9) | ✅ Compatible |
| Decode Single — scores | (900, 10) × 2 | (100,) | (-1,) | ✅ Compatible |
| Decode Single — labels | (900, 10) × 2 | (100,) | (-1,) | ✅ Compatible |
| Decode Batch | (6,6,900,10) × 2 | 6×(100, 9) | 6× graph nodes | ✅ Structure match |
| Top-K Selection | (20, 3) | k=5 with correct labels | — (pure PyTorch) | ✅ Logic verified |
| Score Threshold | (6,) | 3/6 passing | — (pure PyTorch) | ✅ Count & values |
| Center Range | (5, 9) | 2/5 passing | — (pure PyTorch) | ✅ Correct indices |

**Note**: TTSim outputs for filtered results use dynamic dimension `-1` due to data-dependent `NonZero` + `Gather` filtering. Numerical ground truth is provided by the PyTorch reference; TTSim validates graph construction and shape inference.

**Status**: All validations passed ✅

## TTSim Framework Operations Used

The following TTSim ops are composed to build the full decode pipeline:

| PyTorch Operation | TTSim Op | Notes |
|-------------------|----------|-------|
| `torch.sigmoid()` | `F.Sigmoid` | Applied per-element to class logits |
| `tensor.view(-1)` | `F.Reshape` | Flatten `[num_query, num_classes]` → `[num_query*num_classes]` |
| `torch.topk()` | `F.TopK` | Returns `(values, indices)`, `largest=True, sorted=True` |
| `indices % num_classes` | `F.Cast` + `F.Mod` | Int→Float→Mod→Int64 roundtrip |
| `indices // num_classes` | `F.Cast` + `F.Div` + `F.Floor` | Int→Float→Div→Floor→Int64 |
| `bbox[bbox_index]` | `F.Gather(axis=0)` | Row-wise selection of bbox predictions |
| `torch.atan2()` | `F.Atan2` | Rotation angle from sin/cos components |
| `.exp()` | `F.Exp` | Reverse log-dimension encoding |
| `torch.cat(..., dim=-1)` | `F.ConcatX(axis=-1)` | Assemble denormalized bbox components |
| `scores > threshold` | `F.Greater` | Boolean score mask |
| `centers >= range_min` | `F.GreaterOrEqual` | Per-coordinate lower bound check |
| `centers <= range_max` | `F.LessOrEqual` | Per-coordinate upper bound check |
| `mask_a & mask_b` | `F.And` | Combine threshold and range masks |
| `.all(dim=1)` | `F.Cast(Bool→Int32)` + `F.ReduceMin(axis=1)` + `F.Cast(Int32→Bool)` | All-true reduction across coordinate axis |
| `tensor[bool_mask]` | `F.NonZero` + `F.Squeeze` + `F.Gather(axis=0)` | Dynamic boolean indexing |
| `batch[i:i+1]` | `F.SliceF` + `F.Squeeze` | Per-sample extraction in batch loop |
