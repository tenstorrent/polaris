# Comparison Results (heads) -- 2026-04-15 10:26:49

## test_seg_deformable_transformer.py  --  PASS

### stdout

```
======================================================================
SegDeformableTransformer: PyTorch vs TTSim comparison
======================================================================
Config: DIM=32, NHEAD=4, NUM_LEVELS=2, NUM_POINTS=4
  Encoder: 2 layers, FFN=128
  Decoder: 2 layers, FFN=128, return_intermediate=True
  Batch=1, Queries=6, Spatial=[4x4, 2x2]

======================================================================
TEST: SegDeformableTransformer (full forward)
======================================================================
  memory (S,B,D):
    PT shape: (20, 1, 32)  TT shape: (20, 1, 32)
    Max diff: 5.960464e-07, Mean diff: 1.400320e-07
    [OK] Match (atol=0.001)
  init_reference_out (B,nq,2):
    PT shape: (1, 6, 2)  TT shape: (1, 6, 2)
    Max diff: 5.960464e-08, Mean diff: 4.967054e-09
    [OK] Match (atol=1e-05)

  Decoder intermediate states: 2 layers
  inter_state[0] (nq,B,D):
    PT shape: (6, 1, 32)  TT shape: (6, 1, 32)
    Max diff: 4.842877e-07, Mean diff: 1.373701e-07
    [OK] Match (atol=0.001)
  inter_state[1] (nq,B,D):
    PT shape: (6, 1, 32)  TT shape: (6, 1, 32)
    Max diff: 1.132488e-06, Mean diff: 2.012239e-07
    [OK] Match (atol=0.001)

  Decoder intermediate references: 2 layers
  inter_ref[0]:
    PT shape: (1, 6, 4)  TT shape: (1, 6, 4)
    Max diff: 1.490116e-07, Mean diff: 2.235174e-08
    [OK] Match (atol=0.001)
  inter_ref[1]:
    PT shape: (1, 6, 4)  TT shape: (1, 6, 4)
    Max diff: 2.086163e-07, Mean diff: 3.228585e-08
    [OK] Match (atol=0.001)

======================================================================
RESULT: ALL CHECKS PASSED
======================================================================
```

---

## test_seg_detr_head.py  --  PASS

### stdout

```
======================================================================
SegDETRHead — PyTorch vs TTSim Comparison Tests
======================================================================

============================================================
Tests 1-3: Forward branches (cls + reg)
============================================================

  Test 1: fc_cls output:
    PyTorch shape: [6, 1, 300, 4]
    TTSim   shape: [6, 1, 300, 4]
    Max diff: 1.668930e-06, Mean diff: 1.274767e-07
    [OK] Match (atol=0.0001)

  Test 2: reg branch (FFN→ReLU→Linear→Sigmoid):
    PyTorch shape: [6, 1, 300, 4]
    TTSim   shape: [6, 1, 300, 4]
    Max diff: 1.192093e-07, Mean diff: 9.458098e-09
    [OK] Match (atol=0.0001)

  Test 3: Shape match: [OK]  cls=[6, 1, 300, 4], bbox=[6, 1, 300, 4]

============================================================
Test 4: input_proj (Conv2d 1×1)
============================================================

  Test 4: input_proj:
    PyTorch shape: [1, 256, 16, 32]
    TTSim   shape: [1, 256, 16, 32]
    Max diff: 9.536743e-07, Mean diff: 8.267313e-08
    [OK] Match (atol=0.0001)

============================================================
Test 5: analytical_param_count
============================================================
      input_proj: 65,536
      fc_cls: 1,028
      reg_ffn: 131,584
      fc_reg: 1,028
      query_embedding: 76,800
    Total SegDETRHead params: 275,976

  PyTorch param count: 275,976
  TTSim  param count: 275,976
  [OK] Exact match

============================================================
Test 6: Sigmoid cls mode
============================================================

  Test 6a: cls (sigmoid mode):
    PyTorch shape: [3, 1, 50, 5]
    TTSim   shape: [3, 1, 50, 5]
    Max diff: 4.768372e-07, Mean diff: 6.495292e-08
    [OK] Match (atol=0.0001)

  Test 6b: bbox (sigmoid mode):
    PyTorch shape: [3, 1, 50, 4]
    TTSim   shape: [3, 1, 50, 4]
    Max diff: 5.960464e-08, Mean diff: 6.755193e-09
    [OK] Match (atol=0.0001)

======================================================================
SUMMARY
======================================================================
  [PASS] Test 1: fc_cls
  [PASS] Test 2: reg branch
  [PASS] Test 3: shapes
  [PASS] Test 4: input_proj
  [PASS] Test 5: param_count
  [PASS] Test 6a: sigmoid cls
  [PASS] Test 6b: sigmoid bbox

  7/7 passed
  [OK] All tests passed!
```

---

## test_seg_mask_head.py  --  PASS

### stdout

```
============================================================
SegMaskHead Comparison Test: PyTorch vs TTSim
============================================================

============================================================
TEST 1: Mlp
============================================================
  Mlp output:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 8.940697e-08, Mean diff: 1.477892e-08
    [OK] Match (atol=1e-05)

============================================================
TEST 2: SelfAttention
============================================================
  SelfAttention output:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 1.788139e-07, Mean diff: 3.222376e-08
    [OK] Match (atol=1e-05)

============================================================
TEST 3: Attention
============================================================
  Attention output:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 2.384186e-07, Mean diff: 5.733455e-08
    [OK] Match (atol=1e-05)
  Attention mask:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 4.768372e-07, Mean diff: 6.377697e-08
    [OK] Match (atol=1e-05)

============================================================
TEST 4: AttentionTail
============================================================
  AttentionTail mask:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 5.960464e-07, Mean diff: 1.193583e-07
    [OK] Match (atol=1e-05)

============================================================
TEST 5: Block (no self_attn)
============================================================
  Block query output:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 5.960464e-07, Mean diff: 1.048837e-07
    [OK] Match (atol=0.0001)
  Block mask output:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 3.576279e-07, Mean diff: 4.984438e-08
    [OK] Match (atol=1e-05)

============================================================
TEST 6: Block (with self_attn)
============================================================
  Block(sa) query output:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 3.576279e-07, Mean diff: 9.947435e-08
    [OK] Match (atol=0.0001)
  Block(sa) mask output:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 8.344650e-07, Mean diff: 7.122755e-08
    [OK] Match (atol=1e-05)

============================================================
TEST 7: SegMaskHead (thing-like, no self_attn, no pos)
============================================================
  SegMaskHead attn_mask:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 1.430511e-06, Mean diff: 1.816451e-07
    [OK] Match (atol=0.0001)
  SegMaskHead mask[0]:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 4.768372e-07, Mean diff: 8.657575e-08
    [OK] Match (atol=0.0001)
  SegMaskHead mask[1]:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 5.364418e-07, Mean diff: 4.053116e-08
    [OK] Match (atol=0.0001)
  SegMaskHead inter_query[0]:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 7.152557e-07, Mean diff: 1.286971e-07
    [OK] Match (atol=0.0001)
  SegMaskHead inter_query[1]:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 8.940697e-07, Mean diff: 1.858083e-07
    [OK] Match (atol=0.0001)

============================================================
TEST 8: SegMaskHead (stuff-like, self_attn, with pos_query)
============================================================
  SegMaskHead(stuff) attn_mask:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 2.264977e-06, Mean diff: 2.974272e-07
    [OK] Match (atol=0.0001)
  SegMaskHead(stuff) mask[0]:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 2.682209e-07, Mean diff: 2.786517e-08
    [OK] Match (atol=0.0001)
  SegMaskHead(stuff) mask[1]:
    PT shape: (1, 5, 20, 1)  TT shape: (1, 5, 20, 1)
    Max diff: 9.536743e-07, Mean diff: 1.226365e-07
    [OK] Match (atol=0.0001)
  SegMaskHead(stuff) inter_query[0]:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 6.556511e-07, Mean diff: 1.584493e-07
    [OK] Match (atol=0.0001)
  SegMaskHead(stuff) inter_query[1]:
    PT shape: (1, 5, 64)  TT shape: (1, 5, 64)
    Max diff: 9.536743e-07, Mean diff: 2.043602e-07
    [OK] Match (atol=0.0001)

============================================================
SUMMARY
============================================================
  [OK] TEST 1: Mlp
  [OK] TEST 2: SelfAttention
  [OK] TEST 3: Attention
  [OK] TEST 4: AttentionTail
  [OK] TEST 5: Block (no self_attn)
  [OK] TEST 6: Block (with self_attn)
  [OK] TEST 7: SegMaskHead (thing)
  [OK] TEST 8: SegMaskHead (stuff)

  8/8 tests passed.
  ALL TESTS PASSED.
```

---


