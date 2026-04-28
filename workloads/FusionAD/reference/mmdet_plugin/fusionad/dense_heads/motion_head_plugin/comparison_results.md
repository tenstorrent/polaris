# Comparison Results (heads) -- 2026-04-15 10:24:41

## test_base_motion_head.py  --  PASS

### stdout

```
================================================================================
TEST 1: TwoLayerMLP — PyTorch vs TTSim
================================================================================

  TwoLayerMLP output:
    PyTorch shape: (1, 5, 64)
    TTSim   shape: (1, 5, 64)
    Max diff: 1.490116e-07, Mean diff: 3.628666e-08
    [OK] Match (atol=1e-05)

[OK] TEST 1

================================================================================
TEST 2: TwoLayerMLP — 4D input (B, A, P, D)
================================================================================

  TwoLayerMLP 4D output:
    PyTorch shape: (1, 5, 6, 64)
    TTSim   shape: (1, 5, 6, 64)
    Max diff: 2.384186e-07, Mean diff: 4.185052e-08
    [OK] Match (atol=1e-05)

[OK] TEST 2

================================================================================
TEST 3: TrackQueryFuser — PyTorch vs TTSim (no LN affine)
================================================================================

  TrackQueryFuser output:
    PyTorch shape: (1, 5, 64)
    TTSim   shape: (1, 5, 64)
    Max diff: 7.152557e-07, Mean diff: 4.665512e-08
    [OK] Match (atol=0.0001)

[OK] TEST 3

================================================================================
TEST 4: TrajClsBranch — num_reg_fcs=1
================================================================================

  TrajClsBranch output (nrfc=1):
    PyTorch shape: (1, 5, 6, 1)
    TTSim   shape: (1, 5, 6, 1)
    Max diff: 1.788139e-07, Mean diff: 6.382664e-08
    [OK] Match (atol=0.0001)

[OK] TEST 4

================================================================================
TEST 5: TrajClsBranch — num_reg_fcs=2
================================================================================

  TrajClsBranch output (nrfc=2):
    PyTorch shape: (1, 5, 6, 1)
    TTSim   shape: (1, 5, 6, 1)
    Max diff: 2.942979e-07, Mean diff: 8.555750e-08
    [OK] Match (atol=0.0001)

[OK] TEST 5

================================================================================
TEST 6: TrajRegBranch — traj_reg (out=predict_steps*5)
================================================================================

  TrajRegBranch output (reg):
    PyTorch shape: (1, 5, 6, 60)
    TTSim   shape: (1, 5, 6, 60)
    Max diff: 8.940697e-08, Mean diff: 1.764028e-08
    [OK] Match (atol=1e-05)

[OK] TEST 6

================================================================================
TEST 7: TrajRegBranch — traj_refine (out=predict_steps*2)
================================================================================

  TrajRegBranch output (refine):
    PyTorch shape: (1, 5, 6, 24)
    TTSim   shape: (1, 5, 6, 24)
    Max diff: 1.043081e-07, Mean diff: 1.710141e-08
    [OK] Match (atol=1e-05)

[OK] TEST 7

================================================================================
TEST 8: _extract_tracking_centers — numpy arrays
================================================================================

  _extract_tracking_centers:
    Output shape: (1, 3, 2), expected: (1, 3, 2)
    Max diff: 0.000000e+00
    [OK] Match

[OK] TEST 8

================================================================================
TEST 9: _extract_tracking_centers — torch tensors via .tensor attr
================================================================================

  _extract_tracking_centers (torch):
    Output shape: (1, 2, 2), expected: (1, 2, 2)
    [OK] Match

[OK] TEST 9

================================================================================
TEST 10: _extract_tracking_centers — batch_size=2
================================================================================

  _extract_tracking_centers (batch=2):
    Output shape: (2, 1, 2)
    [OK] All batches match

[OK] TEST 10

================================================================================
TEST 11: Analytical param count — TwoLayerMLP
================================================================================

  TwoLayerMLP param count:
    TTSim:   262912
    PyTorch: 262912
    [OK] Match: True

[OK] TEST 11

================================================================================
TEST 12: Analytical param count — TrajRegBranch
================================================================================

  TrajRegBranch param count:
    TTSim:   147004
    PyTorch: 147004
    [OK] Match: True

[OK] TEST 12

================================================================================
RESULTS: 12/12 tests passed, 0 failed.
ALL TESTS PASSED!
================================================================================
```

---

## test_modules.py  --  PASS

### stdout

```
================================================================================
TEST 1: IntentionInteraction — PyTorch vs TTSim
================================================================================

  IntentionInteraction output:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 5.960464e-07, Mean diff: 8.877273e-08
    [OK] Match (atol=0.0001)

[OK] TEST 1

================================================================================
TEST 2: IntentionInteraction — param count
================================================================================
  Expected: 33,216
  Actual:   33,216

[OK] TEST 2

================================================================================
TEST 3: TrackAgentInteraction — PyTorch vs TTSim
================================================================================

  TrackAgentInteraction output:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 1.180993e-07
    [OK] Match (atol=0.0001)

[OK] TEST 3

================================================================================
TEST 4: TrackAgentInteraction — no positional encodings
================================================================================

  TrackAgentInteraction (no pos):
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 1.033555e-07
    [OK] Match (atol=0.0001)

[OK] TEST 4

================================================================================
TEST 5: TrackAgentInteraction — param count
================================================================================
  Expected: 49,856
  Actual:   49,856

[OK] TEST 5

================================================================================
TEST 6: MapInteraction — PyTorch vs TTSim
================================================================================

  MapInteraction output:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 9.925613e-08
    [OK] Match (atol=0.0001)

[OK] TEST 6

================================================================================
TEST 7: MapInteraction — no positional encodings
================================================================================

  MapInteraction (no pos):
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 5.960464e-07, Mean diff: 9.281572e-08
    [OK] Match (atol=0.0001)

[OK] TEST 7

================================================================================
TEST 8: MapInteraction — param count
================================================================================
  Expected: 49,856
  Actual:   49,856

[OK] TEST 8

================================================================================
TEST 9: MotionTransformerDecoder — construction & param count
================================================================================
  embed_dims   = 64
  num_layers   = 3
  param count  = 7,442,944
  [OK]

[OK] TEST 9

================================================================================
TEST 10: MotionTransformerDecoder — MLP fuser weight copy
================================================================================

  dynamic_embed_fuser:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 2.980232e-07, Mean diff: 5.897027e-08
    [OK] Match (atol=1e-05)

  in_query_fuser:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 2.384186e-07, Mean diff: 4.929673e-08
    [OK] Match (atol=1e-05)

  out_query_fuser:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 3.576279e-07, Mean diff: 6.140783e-08
    [OK] Match (atol=1e-05)

[OK] TEST 10

================================================================================
TEST 11: IntentionInteraction — varying agent/mode counts
================================================================================

  IntentionInteraction B=1 A=4 P=3:
    PyTorch shape: (1, 4, 3, 64)
    TTSim   shape: (1, 4, 3, 64)
    Max diff: 4.768372e-07, Mean diff: 8.713626e-08
    [OK] Match (atol=0.0001)

  IntentionInteraction B=1 A=12 P=6:
    PyTorch shape: (1, 12, 6, 64)
    TTSim   shape: (1, 12, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 8.444611e-08
    [OK] Match (atol=0.0001)

  IntentionInteraction B=1 A=8 P=10:
    PyTorch shape: (1, 8, 10, 64)
    TTSim   shape: (1, 8, 10, 64)
    Max diff: 7.152557e-07, Mean diff: 8.261323e-08
    [OK] Match (atol=0.0001)

[OK] TEST 11

================================================================================
TEST 12: MapInteraction — varying map lane count
================================================================================

  MapInteraction M=20:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 9.436803e-08
    [OK] Match (atol=0.0001)

  MapInteraction M=100:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 9.261361e-08
    [OK] Match (atol=0.0001)

  MapInteraction M=200:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 9.567743e-08
    [OK] Match (atol=0.0001)

[OK] TEST 12

================================================================================
RESULTS: 12/12 passed, 0/12 failed
================================================================================
[OK] All tests passed!
```

---

## test_motion_deformable_attn.py  --  PASS

### stdout

```
================================================================================
TEST 1: CustomModeMultiheadAttention — PyTorch vs TTSim
================================================================================

  CustomModeMultiheadAttention:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 2.384186e-07, Mean diff: 4.831842e-08
    [OK] Match (atol=0.0001)

[OK] TEST 1

================================================================================
TEST 2: CustomModeMultiheadAttention — param count
================================================================================
  Expected: 16,640
  Actual:   16,640

[OK] TEST 2

================================================================================
TEST 3: CustomModeMultiheadAttention — no positional encodings
================================================================================

  CMMA (no pos):
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 2.384186e-07, Mean diff: 3.770613e-08
    [OK] Match (atol=0.0001)

[OK] TEST 3

================================================================================
TEST 4: CustomModeMultiheadAttention — varying A, P
================================================================================

  CMMA A=4 P=3:
    PyTorch shape: (1, 4, 3, 64)
    TTSim   shape: (1, 4, 3, 64)
    Max diff: 2.384186e-07, Mean diff: 4.926551e-08
    [OK] Match (atol=0.0001)

  CMMA A=12 P=6:
    PyTorch shape: (1, 12, 6, 64)
    TTSim   shape: (1, 12, 6, 64)
    Max diff: 2.682209e-07, Mean diff: 4.954503e-08
    [OK] Match (atol=0.0001)

  CMMA A=8 P=10:
    PyTorch shape: (1, 8, 10, 64)
    TTSim   shape: (1, 8, 10, 64)
    Max diff: 2.384186e-07, Mean diff: 4.208687e-08
    [OK] Match (atol=0.0001)

[OK] TEST 4

================================================================================
TEST 5: MotionDeformableAttention — construction & param count
================================================================================
  Expected: 128,256
  Actual:   128,256

[OK] TEST 5

================================================================================
TEST 6: MotionDeformableAttention — PyTorch vs TTSim
================================================================================

  MotionDeformableAttention:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 8.940697e-07, Mean diff: 8.834104e-08
    [OK] Match (atol=0.001)

[OK] TEST 6

================================================================================
TEST 7: MotionDeformableAttention — with query_pos
================================================================================

  MDA with query_pos:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 9.536743e-07, Mean diff: 9.037103e-08
    [OK] Match (atol=0.001)

[OK] TEST 7

================================================================================
TEST 8: MotionTransformerAttentionLayer — construction & param count
================================================================================
  MDA params:  128,256
  FFN params:  16,576
  Expected:    144,832
  Actual:      144,832

[OK] TEST 8

================================================================================
TEST 9: MotionTransformerAttentionLayer — CustomModeMultiheadAttention cfg
================================================================================
  MHA params:  16,640
  FFN params:  16,576
  Expected:    33,216
  Actual:      33,216

[OK] TEST 9

================================================================================
TEST 10: MotionDeformableAttention — weight shapes
================================================================================
  sampling_offsets param: [768, 64] [OK]
  attention_weights param: [384, 64] [OK]
  value_proj param: [64, 64] [OK]
  output_proj_linear param: [64, 768] [OK]

[OK] TEST 10

================================================================================
TEST 11: MotionDeformableAttention — varying BEV size
================================================================================

  MDA BEV=30x30:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 9.536743e-07, Mean diff: 8.083831e-08
    [OK] Match (atol=0.001)

  MDA BEV=50x100:
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 7.152557e-07, Mean diff: 8.226743e-08
    [OK] Match (atol=0.001)

[OK] TEST 11

================================================================================
TEST 12: MotionTransformerAttentionLayer — full forward (MDA cross_attn)
================================================================================

  MTAL full forward (MDA):
    PyTorch shape: (1, 8, 6, 64)
    TTSim   shape: (1, 8, 6, 64)
    Max diff: 9.536743e-07, Mean diff: 1.264270e-07
    [OK] Match (atol=0.0001)

[OK] TEST 12

================================================================================
RESULTS: 12/12 passed, 0/12 failed
================================================================================
[OK] All tests passed!
```

---


