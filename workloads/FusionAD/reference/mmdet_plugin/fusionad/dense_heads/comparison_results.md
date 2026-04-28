# Comparison Results (heads) -- 2026-04-15 10:28:43

## test_motion_head.py  --  PASS

### stdout

```
================================================================================
TEST 1: MotionHead construction + param count
================================================================================

  MotionHead param count: 8,156,559
  [OK] param_count > 0

[OK] TEST 1

================================================================================
TEST 2: _norm_points_np vs PyTorch norm_points
================================================================================

  _norm_points_np:
    PyTorch shape: (2, 6, 2)
    TTSim   shape: (2, 6, 2)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-06)

[OK] TEST 2

================================================================================
TEST 3: _pos2posemb2d_np vs PyTorch pos2posemb2d
================================================================================

  _pos2posemb2d_np:
    PyTorch shape: (2, 6, 64)
    TTSim   shape: (2, 6, 64)
    Max diff: 5.960464e-08, Mean diff: 2.990419e-09
    [OK] Match (atol=1e-05)

[OK] TEST 3

================================================================================
TEST 4: _anchor_transform_np — rotation only
================================================================================

  anchor_transform (rot only):
    PyTorch shape: (4, 2, 6, 12, 2)
    TTSim   shape: (4, 2, 6, 12, 2)
    Max diff: 3.576279e-07, Mean diff: 5.459232e-08
    [OK] Match (atol=1e-05)

[OK] TEST 4

================================================================================
TEST 5: _anchor_transform_np — translation only
================================================================================

  anchor_transform (translate only):
    PyTorch shape: (4, 2, 6, 12, 2)
    TTSim   shape: (4, 2, 6, 12, 2)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-05)

[OK] TEST 5

================================================================================
TEST 6: _anchor_transform_np — rotation + translation
================================================================================

  anchor_transform (rot+translate):
    PyTorch shape: (4, 2, 6, 12, 2)
    TTSim   shape: (4, 2, 6, 12, 2)
    Max diff: 4.768372e-07, Mean diff: 5.446944e-08
    [OK] Match (atol=1e-05)

[OK] TEST 6

================================================================================
TEST 7: _group_mode — class-to-group selection
================================================================================

  _group_mode:
    Input shape:    (1, 4, 2, 6, 64)
    Output shape:   (1, 4, 6, 64)
    Expected shape: (1, 4, 6, 64)
    [OK] Match

[OK] TEST 7

================================================================================
TEST 8: _select_last_dec — select last decoder layer
================================================================================

  _select_last_dec:
    Input shape:  (1, 6, 4, 64)
    Output shape: (1, 4, 64)
    [OK] Match

[OK] TEST 8

================================================================================
TEST 9: _unflatten_and_activate vs PyTorch
================================================================================

  _unflatten_and_activate:
    PyTorch shape: (1, 4, 6, 12, 5)
    TTSim   shape: (1, 4, 6, 12, 5)
    Max diff: 1.907349e-06, Mean diff: 4.832271e-08
    [OK] Match (atol=1e-05)

[OK] TEST 9

================================================================================
TEST 10: Log-softmax (Softmax+Log) vs torch.nn.LogSoftmax(dim=2)
================================================================================

  LogSoftmax composition:
    PyTorch shape: (1, 4, 6)
    TTSim   shape: (1, 4, 6)
    Max diff: 2.384186e-07, Mean diff: 5.712112e-08
    [OK] Match (atol=1e-05)

[OK] TEST 10

================================================================================
TEST 11: TrajClsBranch forward match (weight-copied)
================================================================================

  TrajClsBranch forward:
    PyTorch shape: (1, 4, 6, 1)
    TTSim   shape: (1, 4, 6, 1)
    Max diff: 1.788139e-07, Mean diff: 6.239861e-08
    [OK] Match (atol=0.0001)

[OK] TEST 11

================================================================================
TEST 12: TrajRegBranch forward match (weight-copied)
================================================================================

  TrajRegBranch forward:
    PyTorch shape: (1, 4, 6, 60)
    TTSim   shape: (1, 4, 6, 60)
    Max diff: 2.384186e-07, Mean diff: 3.624866e-08
    [OK] Match (atol=1e-05)

[OK] TEST 12

================================================================================
TEST 13: Cls branch -> squeeze -> log_softmax (end-to-end)
================================================================================

  cls -> squeeze -> log_softmax:
    PyTorch shape: (1, 4, 6)
    TTSim   shape: (1, 4, 6)
    Max diff: 2.384186e-07, Mean diff: 6.953875e-08
    [OK] Match (atol=0.0001)

[OK] TEST 13

================================================================================
TEST 14: Reg branch -> unflatten_activate (end-to-end)
================================================================================

  reg -> unflatten -> cumsum -> gaussian:
    PyTorch shape: (1, 4, 6, 12, 5)
    TTSim   shape: (1, 4, 6, 12, 5)
    Max diff: 4.768372e-07, Mean diff: 6.785606e-08
    [OK] Match (atol=0.0001)

[OK] TEST 14

================================================================================
TEST 15: _compute_anchor_embeddings — PyTorch vs TTSim
================================================================================

  agent_emb:
    PyTorch shape: (1, 4, 2, 6, 64)
    TTSim   shape: (1, 4, 2, 6, 64)
    Max diff: 5.960464e-08, Mean diff: 4.522614e-09
    [OK] Match (atol=0.0001)

  ego_emb:
    PyTorch shape: (1, 4, 2, 6, 64)
    TTSim   shape: (1, 4, 2, 6, 64)
    Max diff: 5.960464e-08, Mean diff: 3.318064e-09
    [OK] Match (atol=0.0001)

  offset_emb:
    PyTorch shape: (1, 4, 2, 6, 64)
    TTSim   shape: (1, 4, 2, 6, 64)
    Max diff: 5.960464e-08, Mean diff: 3.192478e-09
    [OK] Match (atol=0.0001)

  learn_emb:
    PyTorch shape: (1, 4, 2, 6, 64)
    TTSim   shape: (1, 4, 2, 6, 64)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-06)

    [OK] scene_anchors shape: (1, 4, 2, 6, 12, 2)

[OK] TEST 15

================================================================================
TEST 16: boxes_query_embedding_layer forward match
================================================================================

  boxes_query_embedding forward:
    PyTorch shape: (1, 4, 64)
    TTSim   shape: (1, 4, 64)
    Max diff: 1.788139e-07, Mean diff: 3.694004e-08
    [OK] Match (atol=1e-05)

[OK] TEST 16

================================================================================
RESULTS: 16/16 tests passed, 0 failed.
ALL TESTS PASSED!
================================================================================
```

---

## test_occ_head.py  --  PASS

### stdout

```
================================================================================
TEST 1: OccHead construction + param count
================================================================================

  OccHead created with 5 future blocks
  has base_ds_0: True
  has dense_decoder: True
  [OK] construction check

[OK] TEST 1

================================================================================
TEST 2: merge_queries
================================================================================

  merge_queries:
    PyTorch shape: [1, 6, 16]
    TTSim   shape: [1, 6, 16]
    Max diff: 2.384186e-07, Mean diff: 6.942233e-08
    [OK] Match (atol=0.0001)

[OK] TEST 2

================================================================================
TEST 3: get_attn_mask (block 0)
================================================================================

  ins_embed:
    PyTorch shape: [1, 6, 16]
    TTSim   shape: [1, 6, 16]
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=0.0001)

  attn_mask:
    PyTorch shape: [1, 4, 4, 6]
    TTSim   shape: [1, 4, 4, 6]
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=0.001)

[OK] TEST 3

================================================================================
TEST 4: Full forward — output shape
================================================================================

  Output shape: [1, 6, 5, 16, 16]
  Expected:     [1, 6, 5, 16, 16]
  [OK] shape check

[OK] TEST 4

================================================================================
TEST 5: Full forward — numerical (PyTorch vs TTSim)
================================================================================

  full forward ins_occ_logits:
    PyTorch shape: [1, 6, 5, 16, 16]
    TTSim   shape: [1, 6, 5, 16, 16]
    Max diff: 5.364418e-07, Mean diff: 1.196617e-07
    [OK] Match (atol=0.05)

[OK] TEST 5

================================================================================
TEST 6: merge_queries output shape
================================================================================

  Output shape: [1, 6, 16]
  Expected:     [1, 6, 16]
  [OK] shape check

[OK] TEST 6

================================================================================
RESULTS: 6/6 tests passed, 0 failed.
ALL TESTS PASSED!
================================================================================
```

---

## test_panseg_head.py  --  PASS

### stdout

```
======================================================================
PansegformerHead: PyTorch vs TTSim comparison
======================================================================
Config: DIM=32, NHEAD=4, NUM_LEVELS=1, NUM_POINTS=4
  Encoder: 1 layers, FFN=64
  Decoder: 2 layers, FFN=64, return_intermediate=True
  Batch=1, Queries=6, BEV=4x4
  Things=3, Stuff=1

======================================================================
TEST 1: RegBranch (PT Sequential vs TT RegBranch)
======================================================================
  RegBranch output [B,nq,4]:
    PT shape: (1, 6, 4)  TT shape: (1, 6, 4)
    Max diff: 3.725290e-08, Mean diff: 1.684142e-08
    [OK] Match (atol=1e-05)

======================================================================
TEST 2: SinePositionalEncoding (numpy vs TTSim precomputed)
======================================================================
  SinePosEnc [1,32,4,4]:
    PT shape: (1, 32, 4, 4)  TT shape: (1, 32, 4, 4)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-06)

======================================================================
TEST 3: PansegformerHead full forward (PT vs TT)
======================================================================

  Decoder outputs_classes: 2 layers
  outputs_classes[0] [B,nq,3]:
    PT shape: (1, 6, 3)  TT shape: (1, 6, 3)
    Max diff: 3.576279e-07, Mean diff: 9.768539e-08
    [OK] Match (atol=0.001)
  outputs_classes[1] [B,nq,3]:
    PT shape: (1, 6, 3)  TT shape: (1, 6, 3)
    Max diff: 3.576279e-07, Mean diff: 9.768539e-08
    [OK] Match (atol=0.001)

  Decoder outputs_coords: 2 layers
  outputs_coords[0] [B,nq,4]:
    PT shape: (1, 6, 4)  TT shape: (1, 6, 4)
    Max diff: 5.960464e-08, Mean diff: 1.738469e-08
    [OK] Match (atol=0.001)
  outputs_coords[1] [B,nq,4]:
    PT shape: (1, 6, 4)  TT shape: (1, 6, 4)
    Max diff: 8.940697e-08, Mean diff: 1.490116e-08
    [OK] Match (atol=0.001)

  Memory and reference:
  memory [B,S,C]:
    PT shape: (1, 16, 32)  TT shape: (1, 16, 32)
    Max diff: 4.768372e-07, Mean diff: 1.287563e-07
    [OK] Match (atol=0.001)
  query_pos [B,nq,C]:
    PT shape: (1, 6, 32)  TT shape: (1, 6, 32)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=0.001)
  reference [B,nq,ref_dim]:
    PT shape: (1, 6, 4)  TT shape: (1, 6, 4)
    Max diff: 8.940697e-08, Mean diff: 1.490116e-08
    [OK] Match (atol=0.001)

======================================================================
TEST 4: PansegformerHead sub-module existence
======================================================================
  [OK] transformer
  [OK] things_mask_head
  [OK] stuff_mask_head
  [OK] cls_branches
  [OK] reg_branches
  [OK] cls_thing_branches
  [OK] cls_stuff_branches
  [OK] reg_branches2
  [OK] query_embedding_weight
  [OK] stuff_query_weight
  [OK] pos_embed

======================================================================
TEST 5: args_tuple completeness (memory_mask, memory_pos, query)
======================================================================
  args_tuple[1] memory_mask [B,S]:
    PT shape: (1, 16)  TT shape: (1, 16)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-06)
  args_tuple[2] memory_pos [B,S,C]:
    PT shape: (1, 16, 32)  TT shape: (1, 16, 32)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=0.001)
  args_tuple[3] query [B,nq,C]:
    PT shape: (1, 6, 32)  TT shape: (1, 6, 32)
    Max diff: 9.536743e-07, Mean diff: 2.696349e-07
    [OK] Match (atol=0.001)

======================================================================
RESULTS: 5/5 tests passed
ALL TESTS PASSED
======================================================================
```

---

## test_planning_head.py  --  PASS

### stdout

```
================================================================================
TEST 1: MLPFuser — PyTorch vs TTSim (no LN affine)
================================================================================

  MLPFuser output:
    PyTorch shape: (1, 6, 64)
    TTSim   shape: (1, 6, 64)
    Max diff: 3.576279e-07, Mean diff: 2.977443e-08
    [OK] Match (atol=0.0001)

[OK] TEST 1

================================================================================
TEST 2: PlanMLP — PyTorch vs TTSim
================================================================================

  PlanMLP output:
    PyTorch shape: (1, 1, 64)
    TTSim   shape: (1, 1, 64)
    Max diff: 8.940697e-08, Mean diff: 2.175511e-08
    [OK] Match (atol=1e-05)

[OK] TEST 2

================================================================================
TEST 3: PlanRegBranch — PyTorch vs TTSim
================================================================================

  PlanRegBranch output:
    PyTorch shape: (1, 1, 12)
    TTSim   shape: (1, 1, 12)
    Max diff: 5.960464e-08, Mean diff: 2.359351e-08
    [OK] Match (atol=1e-05)

[OK] TEST 3

================================================================================
TEST 4: PlanRegBranch shape — various batch dimensions
================================================================================
  input=(1, 1, 128) -> output=(1, 1, 12), expected=(1, 1, 12) [OK]
  input=(2, 1, 128) -> output=(2, 1, 12), expected=(2, 1, 12) [OK]
  input=(4, 1, 128) -> output=(4, 1, 12), expected=(4, 1, 12) [OK]

[OK] TEST 4

================================================================================
TEST 5: PlanningHeadSingleMode construction
================================================================================
  BEV adapter: 3 blocks present: OK
  adapter params (expected): 61,728
  embed_dims      = 64
  planning_steps  = 6
  bev_h x bev_w   = 10 x 10
  param count     = 547,884

[OK] TEST 5

================================================================================
TEST 6: PlanningDecoderLayer construction
================================================================================
  d_model=64, dim_ff=128, nhead=8
  Expected params: 49,856
  Actual params:   49,856
  [OK]

[OK] TEST 6

================================================================================
TEST 7: PlanningDecoder — 3 layers
================================================================================
  3 layers × 49,856 = 149,568
  Actual: 149,568

[OK] TEST 7

================================================================================
TEST 8: MLPFuser + max-pool (PyTorch vs TTSim)
================================================================================

  Fuser + max-pool:
    PyTorch shape: (1, 1, 64)
    TTSim   shape: (1, 1, 64)
    Max diff: 4.768372e-07, Mean diff: 9.039650e-08
    [OK] Match (atol=0.0001)

[OK] TEST 8

================================================================================
TEST 9: PlanMLP analytical param count
================================================================================
  Expected: 314,944
  Actual:   314,944

[OK] TEST 9

================================================================================
TEST 10: Full pipeline — concat + fuser + max + reg_branch
================================================================================

  Full pipeline output:
    PyTorch shape: (1, 1, 12)
    TTSim   shape: (1, 1, 12)
    Max diff: 2.980232e-08, Mean diff: 1.490116e-08
    [OK] Match (atol=0.0001)

[OK] TEST 10

================================================================================
RESULTS: 10/10 passed, 0/10 failed
================================================================================
[OK] All tests passed!
```

---

## test_track_head.py  --  PASS

### stdout

```
================================================================================
TEST 1: ClsBranch — PyTorch vs TTSim (no LN affine)
================================================================================

  ClsBranch output:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 3.576279e-07, Mean diff: 8.065253e-08
    [OK] Match (atol=0.0001)

[OK] TEST 1

================================================================================
TEST 2: RegBranch — PyTorch vs TTSim
================================================================================

  RegBranch output:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 6.705523e-08, Mean diff: 1.790468e-08
    [OK] Match (atol=1e-05)

[OK] TEST 2

================================================================================
TEST 3: TrajRegBranch — PyTorch vs TTSim
================================================================================

  TrajRegBranch output:
    PyTorch shape: (2, 8, 16)
    TTSim   shape: (2, 8, 16)
    Max diff: 1.192093e-07, Mean diff: 2.007073e-08
    [OK] Match (atol=1e-05)

[OK] TEST 3

================================================================================
TEST 4: Reference-point refinement — PyTorch vs TTSim numpy
================================================================================

  Coord output after pc_range scaling:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 7.629395e-06, Mean diff: 7.599592e-08
    [OK] Match (atol=0.0001)

  last_ref_points (inverse sigmoid):
    PyTorch shape: (2, 8, 3)
    TTSim   shape: (2, 8, 3)
    Max diff: 3.576279e-07, Mean diff: 2.367111e-08
    [OK] Match (atol=0.0001)

[OK] TEST 4

================================================================================
TEST 5: BEVFormerTrackHead construction — with_box_refine=True
================================================================================

  Branch counts: cls=3, reg=3, traj=3
  Analytical param count: 116,234

[OK] TEST 5

================================================================================
TEST 6: BEVFormerTrackHead construction — with_box_refine=False
================================================================================
  All layers share the same branch instance.

[OK] TEST 6

================================================================================
TEST 7: Multi-layer branch forward — PyTorch vs TTSim
================================================================================

  ClsBranch L0:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 4.768372e-07, Mean diff: 8.329516e-08
    [OK] Match (atol=0.0001)

  RegBranch L0:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 5.960464e-08, Mean diff: 1.799781e-08
    [OK] Match (atol=1e-05)

  TrajBranch L0:
    PyTorch shape: (2, 8, 16)
    TTSim   shape: (2, 8, 16)
    Max diff: 1.192093e-07, Mean diff: 1.880653e-08
    [OK] Match (atol=1e-05)

  ClsBranch L1:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 2.980232e-07, Mean diff: 7.636845e-08
    [OK] Match (atol=0.0001)

  RegBranch L1:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 1.192093e-07, Mean diff: 1.860317e-08
    [OK] Match (atol=1e-05)

  TrajBranch L1:
    PyTorch shape: (2, 8, 16)
    TTSim   shape: (2, 8, 16)
    Max diff: 7.450581e-08, Mean diff: 1.634908e-08
    [OK] Match (atol=1e-05)

  ClsBranch L2:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 3.576279e-07, Mean diff: 7.845229e-08
    [OK] Match (atol=0.0001)

  RegBranch L2:
    PyTorch shape: (2, 8, 10)
    TTSim   shape: (2, 8, 10)
    Max diff: 8.940697e-08, Mean diff: 1.903973e-08
    [OK] Match (atol=1e-05)

  TrajBranch L2:
    PyTorch shape: (2, 8, 16)
    TTSim   shape: (2, 8, 16)
    Max diff: 8.940697e-08, Mean diff: 1.786611e-08
    [OK] Match (atol=1e-05)

[OK] TEST 7

================================================================================
TEST 8: ClsBranch shape — various batch dimensions
================================================================================
  input=(1, 8, 64) -> output=(1, 8, 10), expected=(1, 8, 10) [OK]
  input=(4, 8, 64) -> output=(4, 8, 10), expected=(4, 8, 10) [OK]

[OK] TEST 8

================================================================================
RESULTS: 8/8 passed, 0/8 failed
================================================================================
[OK] All tests passed!
```

---


