# Module 19: PositionEmbeddingLearned ✅

**Location**: `ttsim_modules/position_embedding_learned.py`
**Original**: `mmdet3d/models/dense_heads/transfusion_head.py` (class ~line 120)

## Description
Learned absolute position embedding module that maps 3D (or 6D) query coordinates to positional encodings via a two-layer 1D convolutional network. The input coordinate array is transposed, passed through `Conv1d + BN1d + ReLU`, then a final `Conv1d`, producing per-point position embeddings.

## Purpose
Enriches transformer queries and keys with spatial position information in the TransformerDecoderLayer of TransFusionHead. Enables the attention mechanism to reason about geometric relationships between BEV proposals and BEV features.

## Module Specifications
- **Input**: `[B, P, input_channel]` coordinate array (transposed internally)
- **Output**: `[B, num_pos_feats, P]` position embeddings
- **Parameters** (input_channel=3, num_pos_feats=288): 84,960
- **Parameter formula**: `in_ch × F + F + 2F + F² + F` (F = num_pos_feats)
- **Architecture**: Transpose → Conv1d(in_ch, F, 1) + BN1d(F) + ReLU → Conv1d(F, F, 1)

## Validation Methodology
The module is validated through five tests:
1. **Construction**: Default param count (in=3, F=288) = 84,960
2. **Output shape default**: `[B, 288, 200]` for default config
3. **Variant config**: `[1, 128, 100]` for (input_channel=6, num_pos_feats=128)
4. **Param count formula**: Verifies three configurations analytically
5. **Batch-size independence**: B=1 and B=4 produce correct shapes

## Validation Results

**Test File**: `Validation/test_position_embedding_learned.py`

```
================================================================================
Test 1: Construction (default: input_channel=3, num_pos_feats=288)
================================================================================

✓ PEL default param count  got 84,960 expected 84,960
--------------------------------------------------------------------------------

================================================================================
Test 2: Output shape [B, num_pos_feats, P]
================================================================================

✓ PEL output shape  got [2, 288, 200] expected [2, 288, 200]
--------------------------------------------------------------------------------

================================================================================
Test 3: Variant config (input_channel=6, num_pos_feats=128)
================================================================================

✓ PEL (ic=6, F=128) output shape  got [1, 128, 100] expected [1, 128, 100]
--------------------------------------------------------------------------------

================================================================================
Test 4: Param count formula for various configs
================================================================================

✓ PEL(ic=3, F=64)    got 4,544  expected 4,544
✓ PEL(ic=6, F=128)   got 17,664 expected 17,664
✓ PEL(ic=3, F=288)   got 84,960 expected 84,960
--------------------------------------------------------------------------------

================================================================================
Test 5: Batch-size independence
================================================================================

✓ B=1  got [1, 128, 50]
✓ B=4  got [4, 128, 50]
--------------------------------------------------------------------------------

============================================================
  PASS  construction_default
  PASS  output_shape_default
  PASS  output_shape_variant
  PASS  param_count_formula
  PASS  batch_independence

5/5 passed
```

## Parameter Count Table

| Config | Expected | Actual |
|--------|----------|--------|
| ic=3, F=64 | 4,544 | 4,544 |
| ic=6, F=128 | 17,664 | 17,664 |
| ic=3, F=288 | 84,960 | 84,960 |

**Status**: All 5/5 tests passed ✅
