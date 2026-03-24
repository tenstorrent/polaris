# Comparison Results (bevformer) — 2026-03-21 00:15:10

## test_custom_msdeformable_attention.py  —  PASS

### stdout

```
================================================================================
CustomMSDeformableAttention TTSim Module Test Suite
================================================================================


Running CustomMSDeformableAttention Test Suite...
================================================================================

================================================================================
TEST 1: CustomMSDeformableAttention Construction
================================================================================

1. Testing default configuration:
   [OK] Module constructed successfully
   - Module name: test_attention
   - Embed dims: 256
   - Num heads: 8
   - Num levels: 4
   - Num points: 4
   - Use sampling offsets: True
   - Batch first: True

2. Testing parameter counting:
    CustomMSDeformableAttention 'test_attention':
      sampling_offsets: 65,792
      attention_weights: 32,896
      value_proj: 65,792
      output_proj: 65,792
    Total CustomMSDeformableAttention params: 230,272
   [OK] Total parameters: 230,272

3. Testing without sampling offsets:
   [OK] Module without sampling offsets constructed
  Total CustomMSDeformableAttention params: 164,480

   Parameter reduction: 65,792

================================================================================
TEST 1: [OK] PASSED
================================================================================

================================================================================
TEST 2: CustomMSDeformableAttention Forward Pass
================================================================================

Configuration:
  Batch size: 2
  Num queries: 100
  Num values: 3294 (computed from spatial_shapes)
  Embed dims: 128
  Num heads: 4
  Num levels: 3
  Num points: 4
  Spatial shapes: [[50, 50], [25, 25], [13, 13]]

1. Creating PyTorch reference model...
   PyTorch sampling_offsets weight: torch.Size([96, 128])
   PyTorch sampling_offsets bias: torch.Size([96])
   PyTorch attention_weights weight: torch.Size([48, 128])
   PyTorch value_proj weight: torch.Size([128, 128])
   PyTorch output_proj weight: torch.Size([128, 128])
2. Creating TTSim model...
3. Creating test inputs...
4. Initializing TTSim parameters from PyTorch...
   Copying 12,288 sampling_offsets weights
   Copying 6,144 attention_weights weights
   Copying 16,384 value_proj weights
   Copying 16,384 output_proj weights
     - sampling_offsets: weight shape (96, 128), bias shape (96,)
     - attention_weights: weight shape (48, 128), bias shape (48,)
     - value_proj: weight shape (128, 128), bias shape (128,)
     - output_proj: weight shape (128, 128), bias shape (128,)
   [OK] All weights copied successfully
5. Running PyTorch forward pass...
6. Running TTSim forward pass...
7. Comparing outputs...

Attention Output Comparison:
  PyTorch shape: torch.Size([2, 100, 128])
  PyTorch stats: mean=-0.001972, std=0.187070
  PyTorch range: [-0.908061, 0.848749]
  TTSim shape: (2, 100, 128)
  TTSim stats: mean=-0.001972, std=0.187070
  TTSim range: [-0.908061, 0.848748]

  Absolute differences:
    Max:    1.564622e-06
    Mean:   2.250626e-07
    Median: 1.792796e-07

  [PASS] [PASS] Outputs match (rtol=1e-05, atol=1e-05)

================================================================================
TEST 2: [OK] PASSED
================================================================================

================================================================================
TEST 3: Multi-Point Reference Points (all points vs first-only)
================================================================================
     - sampling_offsets: weight shape (64, 128), bias shape (64,)
     - attention_weights: weight shape (32, 128), bias shape (32,)
     - value_proj: weight shape (128, 128), bias shape (128,)
     - output_proj: weight shape (128, 128), bias shape (128,)

Multi-point reference output Comparison:
  PyTorch shape: torch.Size([2, 20, 128])
  PyTorch stats: mean=-0.000384, std=0.210924
  PyTorch range: [-0.546211, 0.629009]
  TTSim shape: (2, 20, 128)
  TTSim stats: mean=-0.000384, std=0.210924
  TTSim range: [-0.546211, 0.629009]

  Absolute differences:
    Max:    2.384186e-07
    Mean:   5.896820e-08
    Median: 4.470348e-08

  [PASS] [PASS] Outputs match (rtol=0.0001, atol=0.0001)

[OK] TEST 3 PASSED: All reference points correctly used

================================================================================
TEST SUMMARY
================================================================================
Construction: [OK] PASSED
Forward Pass: [OK] PASSED
Multi-Point Reference Points: [OK] PASSED
================================================================================
ALL TESTS PASSED [OK]
================================================================================
```

### stderr

```
C:\Users\SaSagar\AppData\Local\miniforge3\envs\polaris\Lib\site-packages\requests\__init__.py:109: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (None)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
```

---

## test_maptransformer.py  —  PASS

### stdout

```
================================================================================
MapTransformer Numerical Comparison Suite (PyTorch vs TTSim)
================================================================================

  embed_dims=256  num_heads=8  num_levels=1  num_points=4
  ffn_dim=1024  num_layers=2  batch_size=2  num_queries=50  num_pts_per_query=20
  BEV feature map: 16x16

TEST: Deformable attention standalone
----------------------------------------------------------------------
    Pytorch  shape=(2, 50, 256)  range=[-1.4179e-01, 1.6137e-01]  mean=5.2761e-04
    TTSim   shape=(2, 50, 256)  range=[-1.4179e-01, 1.6137e-01]  mean=5.2761e-04
    range=[0.0000e+00, 1.0803e-07]  mean=1.1161e-08
  [PASS] deformable attention: max=1.08e-07  mean=1.12e-08

PASSED

TEST: Self-attention (MHA) standalone
----------------------------------------------------------------------
    Pytorch  shape=(50, 2, 256)  range=[-1.6075e-02, 1.8919e-02]  mean=4.1343e-05
    TTSim   shape=(50, 2, 256)  range=[-1.6075e-02, 1.8919e-02]  mean=4.1343e-05
    range=[0.0000e+00, 9.3132e-09]  mean=1.2317e-09
  [PASS] self-attention (MHA): max=9.31e-09  mean=1.23e-09

PASSED

TEST: FFN standalone
----------------------------------------------------------------------
    Pytorch  shape=(50, 2, 256)  range=[-4.3936e-01, 4.3884e-01]  mean=-4.6020e-03
    TTSim   shape=(50, 2, 256)  range=[-4.3936e-01, 4.3884e-01]  mean=-4.6020e-03
    range=[0.0000e+00, 5.9605e-08]  mean=7.6130e-09
  [PASS] FFN (2-layer + residual): max=5.96e-08  mean=7.61e-09

PASSED

TEST: Single layer via MapTransformerLayer.__call__
----------------------------------------------------------------------
    Pytorch  shape=(50, 2, 256)  range=[-4.4332e+00, 4.2559e+00]  mean=8.1956e-10
    TTSim   shape=(50, 2, 256)  range=[-4.4332e+00, 4.2559e+00]  mean=1.3644e-09
    range=[0.0000e+00, 9.5367e-07]  mean=1.0148e-07
  [PASS] MapTransformerLayer.__call__: max=9.54e-07  mean=1.01e-07

PASSED

TEST: Full 2-layer decoder via MapTransformerDecoder_new.__call__
----------------------------------------------------------------------
      Pytorch L0  shape=(2, 50, 256)  range=[-4.1003e+00, 3.9631e+00]  mean=1.4575e-09
      TTSim  L0  shape=(2, 50, 256)  range=[-4.1003e+00, 3.9631e+00]  mean=1.4901e-10
    range=[0.0000e+00, 7.1526e-07]  mean=1.0102e-07
  [PASS] decoder layer 0: max=7.15e-07  mean=1.01e-07
      Pytorch L1  shape=(2, 50, 256)  range=[-3.9434e+00, 3.9458e+00]  mean=2.8871e-10
      TTSim  L1  shape=(2, 50, 256)  range=[-3.9434e+00, 3.9458e+00]  mean=-6.5193e-10
    range=[0.0000e+00, 1.1921e-06]  mean=1.4528e-07
  [PASS] decoder layer 1: max=1.19e-06  mean=1.45e-07

PASSED

TEST: Decoder with regression branches
----------------------------------------------------------------------
    range=[0.0000e+00, 9.5367e-07]  mean=1.0021e-07
  [PASS] decoder+reg layer 0 output: max=9.54e-07  mean=1.00e-07
    range=[0.0000e+00, 1.3113e-06]  mean=1.5192e-07
  [PASS] decoder+reg layer 1 output: max=1.31e-06  mean=1.52e-07
      Pytorch ref_pts  shape=(2, 50, 20, 2)  range=[1.1509e-01, 8.8748e-01]  mean=5.0435e-01
      TTSim  ref_pts  shape=(2, 50, 20, 2)  range=[1.1509e-01, 8.8748e-01]  mean=5.0435e-01
    range=[0.0000e+00, 2.3842e-07]  mean=3.3474e-08
  [PASS] final reference points: max=2.38e-07  mean=3.35e-08

PASSED

TEST: Varied inputs (edge refs, small dims)
----------------------------------------------------------------------
    Pytorch  shape=(1, 20, 128)  range=[-1.2733e-01, 1.4497e-01]  mean=4.3140e-04
    TTSim   shape=(1, 20, 128)  range=[-1.2733e-01, 1.4497e-01]  mean=4.3140e-04
    range=[0.0000e+00, 6.7055e-08]  mean=8.9904e-09
  [PASS] varied inputs (edge refs, small dims): max=6.71e-08  mean=8.99e-09

PASSED

TEST: Weight transfer fidelity
----------------------------------------------------------------------
    range=[0.0000e+00, 0.0000e+00]  mean=0.0000e+00
  [PASS] value_proj: max=0.00e+00  mean=0.00e+00
    range=[0.0000e+00, 0.0000e+00]  mean=0.0000e+00
  [PASS] output_proj: max=0.00e+00  mean=0.00e+00
    range=[0.0000e+00, 0.0000e+00]  mean=0.0000e+00
  [PASS] sampling_offsets: max=0.00e+00  mean=0.00e+00
    range=[0.0000e+00, 0.0000e+00]  mean=0.0000e+00
  [PASS] attention_weights: max=0.00e+00  mean=0.00e+00
    range=[0.0000e+00, 0.0000e+00]  mean=0.0000e+00
  [PASS] MHA q_proj: max=0.00e+00  mean=0.00e+00
    range=[0.0000e+00, 4.4703e-08]  mean=7.6673e-09
  [PASS] FFN (no residual): max=4.47e-08  mean=7.67e-09

PASSED

TEST: Memory bank query_bev + query_memory fusion
----------------------------------------------------------------------
    Pytorch (with memory)  shape=(50, 2, 256)  range=[-4.1748e+00, 4.1742e+00]  mean=2.6380e-09
    TTSim  (with memory)  shape=(50, 2, 256)  range=[-4.1748e+00, 4.1742e+00]  mean=1.1665e-09
    range=[0.0000e+00, 9.5367e-07]  mean=1.0896e-07
  [PASS] memory bank query_bev+query_memory fusion: max=9.54e-07  mean=1.09e-07

PASSED

================================================================================
SUMMARY
================================================================================
  [PASS]  Deformable attention standalone
  [PASS]  Self-attention (MHA) standalone
  [PASS]  FFN standalone
  [PASS]  Single layer via MapTransformerLayer.__call__
  [PASS]  Full 2-layer decoder via MapTransformerDecoder_new.__call__
  [PASS]  Decoder with regression branches
  [PASS]  Varied inputs (edge refs, small dims)
  [PASS]  Weight transfer fidelity
  [PASS]  Memory bank query_bev + query_memory fusion

All 9 tests passed.
```

### stderr

```
C:\Users\SaSagar\AppData\Local\miniforge3\envs\polaris\Lib\site-packages\requests\__init__.py:109: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (None)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
```

---

## test_multihead_attention.py  —  PASS

### stdout

```

================================================================================
MultiheadAttention TTSim Implementation Tests
================================================================================

================================================================================
TEST 1: Self-Attention
================================================================================
Configuration:
  embed_dims=256, num_heads=8
  batch_size=2, seq_len=50

Input shape: (50, 2, 256)
Output shape: (50, 2, 256)

Difference vs PyTorch:
  Max:  0.0000001341
  Mean: 0.0000000167
  Total MultiheadAttention params: 263,168

PyTorch params: 263,168

[OK] Self-Attention test PASSED (threshold=1e-05)

================================================================================
TEST 2: Cross-Attention
================================================================================
Configuration:
  embed_dims=256, num_heads=8
  batch_size=2
  seq_q=30, seq_kv=50

Query shape: (30, 2, 256)
Key shape: (50, 2, 256)
Value shape: (50, 2, 256)
Output shape: (30, 2, 256)

Difference vs PyTorch:
  Max:  0.0000001192
  Mean: 0.0000000159

[OK] Cross-Attention test PASSED (threshold=1e-05)

================================================================================
TEST 3: Batch-First Format
================================================================================
Configuration:
  embed_dims=128, num_heads=4
  batch_size=3, seq_len=40
  batch_first=True

Input shape: (3, 40, 128) (bs, seq, embed)
Output shape: (3, 40, 128)

Difference vs PyTorch:
  Max:  0.0000000969
  Mean: 0.0000000156

[OK] Batch-First test PASSED (threshold=1e-05)

================================================================================
TEST 4: Attention Weights Return
================================================================================

Test 1: need_weights=False
  Return type: <class 'ttsim.ops.tensor.SimTensor'>
  Output shape: [20, 2, 128]

Test 2: need_weights=True, average_attn_weights=True
  Return type: tuple
  Output shape: [20, 2, 128]
  Attention weights shape: [2, 1, 20, 20]
  Expected attn shape: (batch=2, seq_q=20, seq_k=20)

Test 3: need_weights=True, average_attn_weights=False
  Return type: tuple
  Output shape: [20, 2, 128]
  Attention weights shape: [2, 8, 20, 20]
  Expected attn shape: (batch=2, num_heads=8, seq_q=20, seq_k=20)

Attention weights properties:
  Min value: 0.047706
  Max value: 0.052656
  Sum over seq_k (should be ~1.0): min=1.000000, max=1.000000

[OK] Attention Weights test PASSED
  - Softmax property verified: True

================================================================================
TEST 5: Key Padding Mask
================================================================================
Configuration:
  embed_dims=256, num_heads=8, batch_first=True
  batch_size=2, seq_q=10, seq_k=5
  Masked positions: last 2 of 5

Output shape: (2, 10, 256)

Difference vs PyTorch:
  Max:  0.0000000596
  Mean: 0.0000000052

[OK] Key Padding Mask test PASSED (threshold=1e-05)

================================================================================
TEST SUMMARY
================================================================================
Self-Attention: [OK] PASSED
Cross-Attention: [OK] PASSED
Batch-First: [OK] PASSED
Attention Weights: [OK] PASSED
Key Padding Mask: [OK] PASSED

================================================================================
ALL TESTS PASSED [OK]
================================================================================
```

### stderr

```
C:\Users\SaSagar\AppData\Local\miniforge3\envs\polaris\Lib\site-packages\requests\__init__.py:109: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (None)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
```

---


