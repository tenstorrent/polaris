# Comparison Results (heads) -- 2026-04-15 12:23:04

## test_custom_base_transformer_layer.py  --  PASS

### stdout

```
================================================================================
CUSTOM BASE TRANSFORMER LAYER VALIDATION TEST
================================================================================

This script validates the TTSim implementation of MyCustomBaseTransformerLayer
by comparing with PyTorch reference implementations and numerical validation.

================================================================================
TEST 1: LayerNorm
================================================================================

1. PyTorch LayerNorm:
   Input shape: torch.Size([2, 10, 256])
   Output shape: torch.Size([2, 10, 256])
   Output mean: -0.000000
   Output std: 0.999995
   Output range: [-3.308985, 3.954201]

2. TTSim LayerNorm:
   Input shape: Shape([2, 10, 256])
   Output shape: Shape([2, 10, 256])
   [OK] LayerNorm constructed successfully

3. Numerical Comparison:
   LayerNorm output:
     PyTorch range: [-3.308985, 3.954201]
     TTSim range: [-3.308985, 3.954201]
     Max diff: 4.768372e-07, Rel diff: 1.205900e-07
     Match: [OK]

4. Parameter Count:
   TTSim params: 512
   Expected params: 512
   Match: True

[OK] LayerNorm test passed!

================================================================================
TEST 2: Feed-Forward Network
================================================================================

1. PyTorch FFN:
   Input shape: torch.Size([2, 10, 64])
   Output shape: torch.Size([2, 10, 64])
   Output mean: 0.040304
   Output std: 0.987728
   Output range: [-3.238370, 3.851585]

2. TTSim FFN:
   Input shape: Shape([2, 10, 64])
   Output shape: Shape([2, 10, 64])
   [OK] FFN constructed successfully

3. Numerical Comparison:
   Note: Outputs will differ due to different weight initialization
   This test validates data computation in TTSim
   [OK] TTSim output computed successfully
   TTSim output range: [-3.693249, 4.683387]
   TTSim output mean: 0.024928
   TTSim output std: 1.170389

4. Parameter Count:
   TTSim params: 16,576
   Expected params: 16,576
   Match: True

[OK] FFN test passed!

================================================================================
TEST 3: Custom Base Transformer Layer Construction
================================================================================

1. Testing: FFN-only transformer layer
   [OK] Layer constructed successfully
   - Operation order: ('ffn', 'norm')
   - Num attentions: 0 (expected: 0)
   - Num FFNs: 1 (expected: 1)
   - Num norms: 1 (expected: 1)
   - Pre-norm: False
   - Batch first: True
   - Embed dims: 256

2. Testing forward pass:
   [OK] Forward pass successful
   - Input shape: Shape([2, 10, 256])
   - Output shape: Shape([2, 10, 256])
   [OK] Output computed successfully
   - Output range: [-3.299653, 3.687034]

3. Testing: Prenorm FFN transformer layer
   [OK] Prenorm layer constructed successfully
   - Pre-norm: True

4. Testing: Multiple FFNs and norms
   [OK] Multi-FFN layer constructed successfully
   - Num FFNs: 2 (expected: 2)
   - Num norms: 3 (expected: 3)

5. Parameter Count:
   Total params: 526,080

[OK] Custom Base Transformer Layer construction test passed!

================================================================================
TEST 4: Operation Order Validation
================================================================================

1. Testing invalid operation order:
   [OK] Correctly rejected invalid operation

2. Testing None operation order:
   [OK] Correctly rejected None operation_order

3. Testing empty operation order:
   [OK] Empty operation order accepted (edge case)

[OK] Operation order validation test passed!

================================================================================
TEST 5: PyTorch vs TTSim Full Comparison
================================================================================

1. PyTorch Simplified Transformer Layer:
   Input shape: torch.Size([2, 10, 64])
   Output shape: torch.Size([2, 10, 64])
   Output mean: -0.000000
   Output std: 0.999995
   Total parameters: 16,832

2. TTSim Transformer Layer:
   Input shape: Shape([2, 10, 64])
   Output shape: Shape([2, 10, 64])
   [OK] Output computed successfully
   Output mean: -0.000000
   Output std: 0.999996
   Total parameters: 16,832

3. Structure Validation:
   Shape match: True
   Parameter count match: True

[OK] PyTorch comparison test passed!

================================================================================
VALIDATION SUMMARY
================================================================================
[OK] PASS: LayerNorm
[OK] PASS: FFN
[OK] PASS: Construction
[OK] PASS: Validation
[OK] PASS: PyTorch Comparison

Total: 5/5 tests passed

[PASS] All validation tests passed!
```

---

## test_decoder.py  --  PASS

### stdout

```

================================================================================
FusionAD Decoder TTSim Comparison Test Suite
================================================================================

================================================================================
TEST 1: inverse_sigmoid
================================================================================

[numpy]  max_diff = 2.38e-07,  mean_diff = 1.34e-08
[TTSim]  max_diff = 2.38e-07,  mean_diff = 1.34e-08

[OK] inverse_sigmoid (threshold=1e-05)

================================================================================
TEST 2a: CustomMSDeformableAttention Construction
================================================================================
  Actual params:   156,256
  Expected params: 156,256
  [OK] param count

================================================================================
TEST 2b: CustomMSDeformableAttention Forward Pass
================================================================================

  Output shape: [100, 2, 256]
  Max  diff (core, excl. residual): 3.73e-08
  Mean diff:                        4.88e-09
  [OK] CustomMSDeformableAttention forward (thr=0.0001)

================================================================================
TEST 3a: MultiheadAttention – Self-Attention
================================================================================
  Shape: [50, 2, 256]
  Max diff:  1.34e-07
  Mean diff: 1.63e-08
  PyTorch params: 263,168  TTSim params: 263,168
  [OK] Self-Attention (thr=1e-05)

================================================================================
TEST 3b: MultiheadAttention – Cross-Attention
================================================================================
  Q shape: (30, 2, 256), KV shape: (50, 2, 256)
  Output shape: [30, 2, 256]
  Max diff:  1.34e-07
  Mean diff: 1.56e-08
  [OK] Cross-Attention (thr=1e-05)

================================================================================
TEST 3c: MultiheadAttention – query_pos + identity residual
================================================================================
  Output shape: [20, 2, 128]
  Max diff:  2.98e-08
  Mean diff: 1.57e-09
  [OK] query_pos + identity (thr=1e-05)

================================================================================
TEST 4a: DetectionTransformerDecoder Construction
================================================================================
  num_layers:          6
  return_intermediate: True
  layers built:        6
  total params:        4,103,232
  [OK] Decoder constructed

================================================================================
TEST 4b: DetectionTransformerDecoder Forward (no reg_branches)
================================================================================
  num intermediate outputs: 2
    layer 0: shape=[50, 1, 256]
    layer 1: shape=[50, 1, 256]
  [OK] Forward pass shapes correct

================================================================================
TEST 4c: DetectionTransformerDecoder Forward (with reg_branches)
================================================================================
  num intermediate outputs: 2
  num ref-pt snapshots:     2
    layer 0: out_shape=[50, 1, 256], ref_shape=[1, 50, 3]
    layer 1: out_shape=[50, 1, 256], ref_shape=[1, 50, 3]
  [OK] Forward with refinement

================================================================================
TEST 4d: DetectionTransformerDecoder Parameter Count
================================================================================
  Per-layer breakdown:
    MHA:  263,168
    CMDA: 156,256
    FFN:  262,912
    LN:   1,536
    Subtotal: 683,872
  Expected (6 layers): 4,103,232
  Actual:                          4,103,232
  [OK] Parameter count

================================================================================
TEST SUMMARY
================================================================================
  inverse_sigmoid........................................ [OK] PASSED
  CMDA Construction...................................... [OK] PASSED
  CMDA Forward Pass...................................... [OK] PASSED
  MHA Self-Attention..................................... [OK] PASSED
  MHA Cross-Attention.................................... [OK] PASSED
  MHA query_pos + identity............................... [OK] PASSED
  Decoder Construction................................... [OK] PASSED
  Decoder Forward (no refine)............................ [OK] PASSED
  Decoder Forward (with refine).......................... [OK] PASSED
  Decoder Parameter Count................................ [OK] PASSED

  10/10 tests passed
================================================================================
ALL TESTS PASSED!
================================================================================
```

---

## test_encoder.py  --  PASS

### stdout

```

================================================================================
BEVFormer Encoder TTSim Module Test Suite
================================================================================

================================================================================
TEST 1: get_reference_points (3D) vs PyTorch
================================================================================
  PyTorch ref_3d shape: (2, 4, 100, 3)
  TTSim   ref_3d shape: (2, 4, 100, 3)
  Max diff:  5.960464e-08
  Mean diff: 4.967054e-09
  PyTorch range: [0.0500, 0.9500]
  TTSim   range: [0.0500, 0.9500]
[OK] 3D reference points match PyTorch exactly

================================================================================
TEST 2: get_reference_points (2D) vs PyTorch
================================================================================
  PyTorch ref_2d shape: (2, 100, 1, 2)
  TTSim   ref_2d shape: (2, 100, 1, 2)
  Max diff:  0.000000e+00
  Mean diff: 0.000000e+00
  PyTorch range: [0.0500, 0.9500]
  TTSim   range: [0.0500, 0.9500]
[OK] 2D reference points match PyTorch exactly

================================================================================
TEST 3: point_sampling vs PyTorch
================================================================================
  PyTorch rpc shape: (6, 2, 100, 4, 2), mask shape: (6, 2, 100, 4)
  TTSim   rpc shape: (6, 2, 100, 4, 2), mask shape: (6, 2, 100, 4)
  rpc  max diff:  1.000000e+00, mean diff: 9.905541e-03
  mask max diff:  0.000000e+00
[OK] point_sampling matches PyTorch

================================================================================
TEST 4: BEVFormerLayer Construction + Param Count
================================================================================
  embed_dims:      256
  operation_order: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
  num_attn:        2
  num_ffns:        1
  num_norms:       3
  pre_norm:        False
  param_count:     1,086,144

  Actual breakdown (from sub-modules):
    TSA:   230,080
    SCA:   328,960
    FFN:   525,568
    Norms: 1,536
    Sum:   1,086,144
[OK] Param count matches sum of sub-modules

================================================================================
TEST 5: BEVFormerFusionLayer Construction + Param Count
================================================================================
  embed_dims:      256
  operation_order: ('self_attn', 'norm', 'pts_cross_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
  num_attn:        3
  num_ffns:        1
  num_norms:       4
  pre_norm:        False
  param_count:     1,242,912
[OK] BEVFormerFusionLayer constructed correctly
  (3 attentions: TSA + PtsCrossAttn + SCA, 1 FFN, 4 norms)

================================================================================
TEST 6: BEVFormerEncoder Construction + Param Count
================================================================================
  num_layers: 3
  num actual layers: 3
  single layer params: 1,086,144
  total params:        3,258,432
  expected (N*single): 3,258,432
[OK] BEVFormerEncoder params = 3 x 1,086,144 = 3,258,432

================================================================================
TEST 7: Reference Points -- Various BEV Sizes
================================================================================
  Case 1: H=50, W=50, Z=8.0, D=4, bs=1 -- 3D: match, 2D: match [OK]
  Case 2: H=100, W=100, Z=8.0, D=4, bs=2 -- 3D: match, 2D: match [OK]
  Case 3: H=30, W=30, Z=10.0, D=8, bs=1 -- 3D: match, 2D: match [OK]
  Case 4: H=200, W=200, Z=8.0, D=4, bs=1 -- 3D: match, 2D: match [OK]
[OK] All size variants match

================================================================================
TEST 8: build_encoder_layer Factory
================================================================================
  [OK] BEVFormerLayer: embed_dims=128, params=235,584
  [OK] BEVFormerFusionLayer: embed_dims=128, params=275,056
  [OK] ValueError raised for unknown type

================================================================================
TEST 9: BEVFormerLayer Forward -- TSA Stage PyTorch vs TTSim
================================================================================
  Config: embed_dims=256, heads=8, bev=10x10, bs=2

[1] PyTorch TSA forward...
  PyTorch output: shape=(2, 100, 256), mean=-1.335660e-04, std=1.224832e-01

[2] TTSim TSA forward (via BEVFormerLayer.attentions[0])...

[3] Numerical comparison...
  TTSim output:   shape=(2, 100, 256), mean=-1.335660e-04, std=1.224832e-01
  Max diff:  5.960464e-08
  Mean diff: 1.280803e-09
[OK] TSA outputs match PyTorch within tolerance

================================================================================
TEST 10: BEVFormerEncoder Forward -- End-to-End Shape & Data Check
================================================================================
  Config: embed=128, heads=4, layers=2, bev=8x8, bs=1
  Built: 2 layers, params=471,168

[1] Initializing parameters...
  Initialized 40 parameter tensors

[2] Creating inputs...
  bev_query: (64, 1, 128)
  key/value: (6, 512, 1, 128)
  spatial_shapes: [(16, 16), (16, 16)]

[3] Running encoder forward pass...
  Output shape: Shape([1, 64, 128])
  [OK] Shape matches expected [1, 64, 128]
  Stats: mean=-1.491571e-10, std=9.999963e-01, range=[-3.581291e+00, 3.810070e+00]
  [OK] Output is finite and non-zero

[OK] BEVFormerEncoder forward pass completed successfully

================================================================================
TEST SUMMARY
================================================================================
Reference Points 3D vs PyTorch.............................. [OK] PASSED
Reference Points 2D vs PyTorch.............................. [OK] PASSED
Point Sampling vs PyTorch................................... [OK] PASSED
BEVFormerLayer Construction................................. [OK] PASSED
BEVFormerFusionLayer Construction........................... [OK] PASSED
BEVFormerEncoder Construction............................... [OK] PASSED
Reference Points Various Sizes.............................. [OK] PASSED
build_encoder_layer Factory................................. [OK] PASSED
BEVFormerLayer Forward PyTorch vs TTSim..................... [OK] PASSED
BEVFormerEncoder Forward PyTorch vs TTSim................... [OK] PASSED

Total: 10/10 tests passed

All tests passed! The encoder module is working correctly.
```

---

## test_multi_scale_deformable_attn_function.py  --  PASS

### stdout

```
================================================================================
Multi-Scale Deformable Attention Validation Tests
================================================================================

Single Level Test:
  Output shape: torch.Size([2, 10, 256])
  PyTorch - mean: -8.888071e-03, std: 3.541140e-01, min: -1.356553e+00, max: 1.399872e+00
  TTSim   - mean: -8.888070e-03, std: 3.541140e-01, min: -1.356553e+00, max: 1.399872e+00
  Max diff: 2.384186e-07
  Mean diff: 1.619560e-08

4-Level Test:
  Output shape: torch.Size([1, 20, 256])
  Spatial shapes: [[50, 50], [25, 25], [13, 13], [7, 7]]
  PyTorch - mean: -1.637657e-03, std: 1.766103e-01, min: -7.557563e-01, max: 7.555073e-01
  TTSim   - mean: -1.637657e-03, std: 1.766103e-01, min: -7.557564e-01, max: 7.555073e-01
  Max diff: 8.940697e-08
  Mean diff: 1.203962e-08

Batch Size 1 Test:
  PyTorch - mean: -3.187767e-03, std: 2.505292e-01
  TTSim   - mean: -3.187769e-03, std: 2.505292e-01
  Max diff: 1.192093e-07, Mean diff: 1.547814e-08

Batch Size 2 Test:
  PyTorch - mean: 7.703266e-04, std: 2.463593e-01
  TTSim   - mean: 7.703262e-04, std: 2.463593e-01
  Max diff: 1.192093e-07, Mean diff: 1.541449e-08

Batch Size 4 Test:
  PyTorch - mean: 2.003724e-03, std: 2.436672e-01
  TTSim   - mean: 2.003724e-03, std: 2.436672e-01
  Max diff: 1.788139e-07, Mean diff: 1.533360e-08

Num Heads 4 Test:
  PyTorch - mean: 7.298634e-03, std: 2.524594e-01
  TTSim   - mean: 7.298636e-03, std: 2.524594e-01
  Max diff: 1.192093e-07, Mean diff: 1.600815e-08

Num Heads 8 Test:
  PyTorch - mean: 1.047077e-04, std: 2.478859e-01
  TTSim   - mean: 1.047079e-04, std: 2.478859e-01
  Max diff: 1.192093e-07, Mean diff: 1.553194e-08

Num Heads 16 Test:
  PyTorch - mean: -3.909258e-03, std: 2.499491e-01
  TTSim   - mean: -3.909257e-03, std: 2.499491e-01
  Max diff: 1.490116e-07, Mean diff: 1.546845e-08

Num Points 1 Test:
  PyTorch - mean: 5.901085e-03, std: 4.792777e-01
  TTSim   - mean: 5.901086e-03, std: 4.792777e-01
  Max diff: 2.384186e-07, Mean diff: 1.740655e-08

Num Points 4 Test:
  PyTorch - mean: -2.379824e-03, std: 2.467906e-01
  TTSim   - mean: -2.379823e-03, std: 2.467906e-01
  Max diff: 1.192093e-07, Mean diff: 1.561292e-08

Num Points 8 Test:
  PyTorch - mean: -2.255548e-03, std: 1.949683e-01
  TTSim   - mean: -2.255548e-03, std: 1.949683e-01
  Max diff: 8.940697e-08, Mean diff: 1.237941e-08

Boundary Sampling Test:
  PyTorch - mean: 6.687523e-03, std: 9.786373e-02
  TTSim   - mean: 6.687523e-03, std: 9.786373e-02
  Max diff: 2.980232e-08, Mean diff: 4.054129e-09

Large Scale BEVFormer-like Test:
  Output shape: torch.Size([1, 900, 256])
  Num queries: 900
  Num keys: 30825
  PyTorch - mean: -4.352739e-04, std: 1.229281e-01, min: -5.597671e-01, max: 5.535631e-01
  TTSim   - mean: -4.352740e-04, std: 1.229281e-01, min: -5.597671e-01, max: 5.535630e-01
  Max diff: 8.940697e-08
  Mean diff: 9.282883e-09

Shape Correctness Test: PASSED

--------------------------------------------------------------------------------
Full Module Tests (covers normalizer shape + identity transpose fixes)
--------------------------------------------------------------------------------

Full Module Normalizer Shape Test:
  embed_dims=256, num_heads=8
  num_levels=4, num_points=8 (intentionally different)
  Output shape: Shape([2, 10, 256]) (expected (2, 10, 256))
  Output range: [-3.4134, 4.1777]
  [OK] Full module normalizer shape test PASSED

Full Module Identity Non-Batch-First Test:
  bs=2, num_query=15 (intentionally different)
  batch_first=False
  Output shape: Shape([15, 2, 256]) (expected (15, 2, 256))
  [OK] Identity non-batch-first test PASSED

--------------------------------------------------------------------------------
Single-Level Test (covers seg head / BEV encoder pattern)
--------------------------------------------------------------------------------

----------------------------------------
Test: num_levels == 1 (seg head pattern)
----------------------------------------
  Output shape: Shape([100, 1, 256]) (expected (100, 1, 256))
  num_levels=1
  [OK] single-level attention test PASSED

================================================================================
ALL TESTS PASSED!
================================================================================
```

---

## test_pts_cross_attention.py  --  PASS

### stdout

```

================================================================================
PtsCrossAttention TTSim Module Test Suite
================================================================================

================================================================================
TEST 1: PtsCrossAttention Construction
================================================================================
[OK] Module constructed successfully
  - Module name: test_pca
  - Embed dims: 256
  - Num heads: 8
  - Num levels: 1
  - Num points: 4

================================================================================
TEST 2: PtsCrossAttention Forward Pass (with Data Validation)
================================================================================

Configuration:
  - Batch size: 2
  - Num queries (object queries): 100
  - Num value (BEV features): 900
  - Embed dims: 256
  - Num heads: 8
  - Num levels: 1
  - Spatial shapes: [(30, 30)]

[1] Creating test inputs...

[2] Running PyTorch reference implementation...
  PyTorch output shape: torch.Size([2, 100, 256])
  PyTorch: mean=1.606887e-04, std=1.063494e-01, min=-5.393406e-01, max=4.557647e-01

[3] Running TTSim implementation...
  Copying PyTorch weights to TTSim...
  TTSim output shape: Shape([2, 100, 256])
  TTSim:   mean=1.606887e-04, std=1.063494e-01, min=-5.393406e-01, max=4.557647e-01

  Numerical comparison:
    Max diff: 5.960464e-08
    Mean diff: 4.336566e-09
    [OK] Numerical outputs match within tolerance

[OK] Forward pass successful with data validation

================================================================================
TEST 3: Parameter Count
================================================================================
PtsCrossAttention parameter breakdown:
  - Sampling offsets: 16,448
  - Attention weights: 8,224
  - Value projection: 65,792
  - Output projection: 65,792
  - Expected total: 156,256
  - Actual total: 156,256
[OK] Parameter count matches expected

================================================================================
TEST 4: Different Configurations (with Data Validation)
================================================================================

Test case 1: embed_dims=128, num_heads=4, num_levels=1, num_points=4
  Spatial shapes: [(20, 20)], num_query: 50
  PyTorch output: shape=torch.Size([2, 50, 128]), range=[-0.491755, 0.415698], mean=0.002008
  TTSim output: shape=Shape([2, 50, 128])
    Max diff: 4.470348e-08, Mean diff: 3.619780e-09
  [OK] Shapes match! Parameter count: 39,216

Test case 2: embed_dims=256, num_heads=8, num_levels=4, num_points=4
  Spatial shapes: [(30, 30), (15, 15), (8, 8), (4, 4)], num_query: 100
  PyTorch output: shape=torch.Size([2, 100, 256]), range=[-0.392312, 0.428514], mean=-0.000002
  TTSim output: shape=Shape([2, 100, 256])
    Max diff: 2.980232e-08, Mean diff: 2.311541e-09
  [OK] Shapes match! Parameter count: 230,272

Test case 3: embed_dims=512, num_heads=16, num_levels=2, num_points=8
  Spatial shapes: [(15, 15), (8, 8)], num_query: 200
  PyTorch output: shape=torch.Size([2, 200, 512]), range=[-0.460398, 0.450229], mean=0.000106
  TTSim output: shape=Shape([2, 200, 512])
    Max diff: 2.980232e-08, Mean diff: 2.064566e-09
  [OK] Shapes match! Parameter count: 919,296

================================================================================
TEST 5: Value Defaults to Query
================================================================================
  Output shape: Shape([2, 49, 128]) -- correct
  Max diff vs PyTorch: 2.980232e-08
[OK] value=None defaults to query correctly, numerical match

================================================================================
TEST 6: Query Position Encoding (query_pos)
================================================================================
  Output shape: Shape([2, 50, 128]) -- correct
  Max diff: 2.980232e-08, Mean diff: 3.358282e-09
[OK] query_pos correctly added, numerical match

================================================================================
TEST SUMMARY
================================================================================
PtsCrossAttention Construction.............................. [OK] PASSED
PtsCrossAttention Forward Pass.............................. [OK] PASSED
Parameter Count............................................. [OK] PASSED
Different Configurations.................................... [OK] PASSED
Value Defaults to Query..................................... [OK] PASSED
Query Position Encoding..................................... [OK] PASSED

Total: 6/6 tests passed

All tests passed! The module is working correctly.
```

---

## test_spatial_cross_attention.py  --  PASS

### stdout

```

================================================================================
Spatial Cross Attention TTSim Module Test Suite
================================================================================

================================================================================
TEST 1: Initialization Utilities (Python 3.13 Compatible)
================================================================================

[1] Testing xavier_init...
  [OK] xavier_init succeeded
    - Weight std: 0.062712
    - Bias max abs: 0.000000
  [OK] xavier_init values look correct

[2] Testing constant_init...
  [OK] constant_init succeeded
    - Weight max abs: 0.000000
    - Bias max abs: 0.000000
  [OK] constant_init values are zero

[3] Testing _is_power_of_2...
  [OK] _is_power_of_2 works correctly

[4] Testing warning for non-power-of-2 dims...
  [OK] Warning triggered as expected: You'd better set embed_dims in MultiScaleDeformAttention to make the dimension of each attention head a power of 2 which is more efficient in our CUDA implementation.

[5] Testing PyTorch vs TTSim output comparison with initialized weights...
    Creating PyTorch model with custom initialization...
    PyTorch output: shape=torch.Size([2, 10, 128]), range=[-0.092352, 0.082533], mean=0.001655
    Creating TTSim model...
    TTSim output: shape=Shape([2, 10, 128])
    [OK] PyTorch and TTSim output shapes match!
      Both have shape: [2, 10, 128]
    [OK] Initialization utilities successfully used in PyTorch-TTSim comparison

================================================================================
TEST 2: MSDeformableAttention3D Construction
================================================================================
[OK] Module constructed successfully
  - Module name: test_msda3d
  - Embed dims: 256
  - Num heads: 8
  - Num levels: 4
  - Num points: 8

================================================================================
TEST 3: SpatialCrossAttention Construction
================================================================================
[OK] Module constructed successfully
  - Module name: test_sca
  - Embed dims: 256
  - Num cameras: 6
  - PC range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

================================================================================
TEST 4: MSDeformableAttention3D Forward Pass (with Data Validation)
================================================================================

Configuration:
  - Batch size: 2
  - Num queries: 10
  - Embed dims: 256
  - Num levels: 4
  - Num Z anchors: 4
  - Spatial shapes: [(50, 50), (25, 25), (13, 13), (7, 7)]
  - Total num_value: 3343

[1] Creating test inputs...

[2] Running PyTorch reference implementation...
  PyTorch output shape: torch.Size([2, 10, 256])
  PyTorch: mean=9.316051e-05, std=1.659306e-02, min=-6.373456e-02, max=7.881641e-02

[3] Running TTSim implementation...
  Copying PyTorch weights to TTSim...
  TTSim output shape: Shape([2, 10, 256])
  TTSim:   mean=9.316047e-05, std=1.659306e-02, min=-6.373456e-02, max=7.881641e-02

  Numerical comparison:
    Max diff: 1.117587e-08
    Mean diff: 1.098050e-09
    [OK] Numerical outputs match within tolerance

[4] Validating outputs...
  Expected shape: [2, 10, 256]

  MSDA3D Output Comparison:
    PyTorch shape: [2, 10, 256]
    TTSim shape: [2, 10, 256]
    PyTorch range: [-0.063735, 0.078816]
    PyTorch mean: 0.000093, std: 0.016593
    [OK] Shapes match!

[OK] Forward pass successful with data validation

================================================================================
TEST 5: SpatialCrossAttention Forward Pass (with Data Validation)
================================================================================
Configuration:
  - Batch size: 1
  - Num queries: 20
  - Num cameras: 3
  - Embed dims: 128
  - Num heads: 4
  - Spatial shapes: [(10, 10), (5, 5)]
  - Total feature size (l): 125

[1] Running PyTorch reference...
  PyTorch output shape: [1, 20, 128]
  PyTorch: mean=3.429583e-03, std=1.003521e-01, min=-3.335913e-01, max=3.853339e-01

[2] Running TTSim implementation...
  Copying PyTorch weights to TTSim...
  TTSim output shape: [1, 20, 128]
  [OK] Shapes match: [1, 20, 128]
  TTSim:   mean=3.429583e-03, std=1.003521e-01, min=-3.335913e-01, max=3.853339e-01

  Numerical comparison:
    Max diff:  2.980232e-08
    Mean diff: 2.168281e-09
    [OK] Numerical outputs match within tolerance

[OK] SpatialCrossAttention forward pass with data validation passed

================================================================================
TEST 6: Different Configurations (with Data Validation)
================================================================================

Test case 1: embed_dims=128, num_heads=4, num_levels=2, num_points=4
  Spatial shapes: [(10, 10), (5, 5)]
  PyTorch output: shape=torch.Size([2, 10, 128]), range=[-0.085349, 0.091056], mean=-0.000270
  TTSim output: shape=Shape([2, 10, 128])
    Max diff: 1.117587e-08, Mean diff: 1.460467e-09
  [OK] Shapes match! Parameter count: 28,896

Test case 2: embed_dims=256, num_heads=8, num_levels=4, num_points=8
  Spatial shapes: [(50, 50), (25, 25), (13, 13), (7, 7)]
  PyTorch output: shape=torch.Size([2, 10, 256]), range=[-0.062551, 0.055376], mean=-0.000290
  TTSim output: shape=Shape([2, 10, 256])
    Max diff: 7.450581e-09, Mean diff: 1.113773e-09
  [OK] Shapes match! Parameter count: 263,168

Test case 3: embed_dims=512, num_heads=16, num_levels=3, num_points=4
  Spatial shapes: [(20, 20), (10, 10), (5, 5)]
  PyTorch output: shape=torch.Size([2, 10, 512]), range=[-0.083013, 0.077137], mean=0.000326
  TTSim output: shape=Shape([2, 10, 512])
    Max diff: 1.117587e-08, Mean diff: 1.528943e-09
  [OK] Shapes match! Parameter count: 558,144

================================================================================
TEST 7: Parameter Count
================================================================================
MSDeformableAttention3D parameter breakdown:
  - Sampling offsets: 131,584
  - Attention weights: 65,792
  - Value projection: 65,792
  - Expected total: 263,168
  - Actual total: 263,168
[OK] Parameter count matches expected

================================================================================
TEST 8: Batch First Flag (with Data Validation)
================================================================================

Testing with batch_first=True
  PyTorch output: shape=torch.Size([2, 10, 128]), range=[-0.086023, 0.095484]
  TTSim output: shape=Shape([2, 10, 128])
    Max diff: 1.490116e-08, Mean diff: 1.509341e-09
  [OK] Output shape correct and matches PyTorch (accounting for batch_first)

Testing with batch_first=False
  PyTorch output: shape=torch.Size([2, 10, 128]), range=[-0.092470, 0.098135]
  TTSim output: shape=Shape([10, 2, 128])
    Max diff: 1.490116e-08, Mean diff: 1.555748e-09
  [OK] Output shape correct and matches PyTorch (accounting for batch_first)

================================================================================
TEST 9: With Key Padding Mask (with Data Validation)
================================================================================

  [0] Creating PyTorch reference model...
  Copying PyTorch weights to TTSim...

  [1] Running without mask (baseline)...
    PyTorch (no mask): shape=torch.Size([2, 10, 128])
    TTSim (no mask): shape=Shape([2, 10, 128])
    PyTorch vs TTSim: Max diff: 1.490116e-08, Mean diff: 1.428441e-09
    [OK] PyTorch and TTSim match without mask

  [2] Running with mask...
    TTSim (with mask): shape=Shape([2, 10, 128])

  [3] Verifying mask effect on TTSim output...
    Difference (masked vs unmasked):
      Max diff: 4.202094e-02, Mean diff: 7.306271e-03
    [OK] Masking has measurable effect on output

  [OK] Masking test passed - PyTorch comparison done, mask functionality verified

================================================================================
TEST 10: Edge Cases (with Data Validation)
================================================================================

Edge case 1: Single query
  PyTorch: shape=torch.Size([1, 1, 128]), range=[-0.063698, 0.053254]
  TTSim: shape=Shape([1, 1, 128])
    Max diff: 5.587935e-09, Mean diff: 1.196895e-09
  [OK] Single query test passed, shapes match PyTorch

Edge case 2: Single level
  PyTorch: shape=torch.Size([2, 10, 128]), range=[-0.160086, 0.149424]
  TTSim: shape=Shape([2, 10, 128])
    Max diff: 1.490116e-08, Mean diff: 1.475428e-09
  [OK] Single level test passed, shapes match PyTorch

================================================================================
TEST SUMMARY
================================================================================
Initialization Utilities.................................... [OK] PASSED
MSDeformableAttention3D Construction........................ [OK] PASSED
SpatialCrossAttention Construction.......................... [OK] PASSED
MSDeformableAttention3D Forward Pass........................ [OK] PASSED
SpatialCrossAttention Forward Pass.......................... [OK] PASSED
Different Configurations.................................... [OK] PASSED
Parameter Count............................................. [OK] PASSED
Batch First Flag............................................ [OK] PASSED
With Key Padding Mask....................................... [OK] PASSED
Edge Cases.................................................. [OK] PASSED

Total: 10/10 tests passed

[PASS] All tests passed! The modules are working correctly.
```

---

## test_temporal_self_attention.py  --  PASS

### stdout

```

================================================================================
Temporal Self Attention TTSim Module Test Suite
================================================================================

================================================================================
TEST 1: TemporalSelfAttention Construction
================================================================================
[OK] Module constructed successfully
  - Module name: test_tsa
  - Embed dims: 256
  - Num heads: 8
  - Num levels: 4
  - Num points: 4
  - Num BEV queue: 2

================================================================================
TEST 2: TemporalSelfAttention Forward Pass (with Data Validation)
================================================================================

Configuration:
  - Batch size: 2
  - Num queries: 1205
  - Embed dims: 256
  - Num levels: 4
  - Num BEV queue: 2
  - Spatial shapes: [(30, 30), (15, 15), (8, 8), (4, 4)]
  - Num value (per BEV): 1205

[1] Creating test inputs...

[2] Running PyTorch reference implementation...
  PyTorch output shape: torch.Size([2, 1205, 256])
  PyTorch: mean=-3.646277e-04, std=1.026675e-01, min=-4.645285e-01, max=4.799971e-01

[3] Running TTSim implementation...
  Copying PyTorch weights to TTSim...
  TTSim output shape: Shape([2, 1205, 256])
  TTSim:   mean=-3.646277e-04, std=1.026675e-01, min=-4.645285e-01, max=4.799972e-01

  Numerical comparison:
    Max diff: 5.960464e-08
    Mean diff: 3.977541e-09
    [OK] Numerical outputs match within tolerance

[OK] Forward pass successful with data validation

================================================================================
TEST 3: Parameter Count
================================================================================
TemporalSelfAttention parameter breakdown:
  - Sampling offsets: 262,656
  - Attention weights: 131,328
  - Value projection: 65,792
  - Output projection: 65,792
  - Expected total: 525,568
  - Actual total: 525,568
[OK] Parameter count matches expected

================================================================================
TEST 4: Different Configurations (with Data Validation)
================================================================================

Test case 1: embed_dims=128, num_heads=4, num_levels=2, num_points=4
  Spatial shapes: [(20, 20), (10, 10)]
  PyTorch output: shape=torch.Size([2, 500, 128]), range=[-0.498172, 0.449546], mean=0.000195
  TTSim output: shape=Shape([2, 500, 128])
    Max diff: 4.470348e-08, Mean diff: 4.664343e-09
  [OK] Shapes match! Parameter count: 82,368

Test case 2: embed_dims=256, num_heads=8, num_levels=4, num_points=4
  Spatial shapes: [(30, 30), (15, 15), (8, 8), (4, 4)]
  PyTorch output: shape=torch.Size([2, 1205, 256]), range=[-0.489575, 0.490585], mean=-0.000331
  TTSim output: shape=Shape([2, 1205, 256])
    Max diff: 5.960464e-08, Mean diff: 4.007792e-09
  [OK] Shapes match! Parameter count: 525,568

Test case 3: embed_dims=512, num_heads=16, num_levels=3, num_points=8
  Spatial shapes: [(15, 15), (8, 8), (4, 4)]
  PyTorch output: shape=torch.Size([2, 305, 512]), range=[-0.470848, 0.505399], mean=-0.000156
  TTSim output: shape=Shape([2, 305, 512])
    Max diff: 4.470348e-08, Mean diff: 4.716016e-09
  [OK] Shapes match! Parameter count: 2,886,912

================================================================================
TEST SUMMARY
================================================================================
TemporalSelfAttention Construction.......................... [OK] PASSED
TemporalSelfAttention Forward Pass.......................... [OK] PASSED
Parameter Count............................................. [OK] PASSED
Different Configurations.................................... [OK] PASSED

Total: 4/4 tests passed

! All tests passed! The module is working correctly.
```

---

## test_transformer.py  --  PASS

### stdout

```
================================================================================
TEST 1: CanBusMLP Construction & Parameter Count
================================================================================
  [OK] CanBusMLP constructed
  Expected params: 35968
  Actual params:   35968
  [OK] Param count matches
  [OK] Without norm: 35456 (expected 35456)

[OK] TEST 1 PASSED

================================================================================
TEST 2: CanBusMLP Forward Pass vs PyTorch
================================================================================

  CanBusMLP output comparison:
    PyTorch shape: (2, 256)
    TTSim   shape: (2, 256)
    Max diff: 4.768372e-07, Mean diff: 1.101562e-07
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

  CanBusMLP bs=1 comparison:
    PyTorch shape: (1, 256)
    TTSim   shape: (1, 256)
    Max diff: 7.748604e-07, Mean diff: 1.118722e-07
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

  CanBusMLP bs=4 comparison:
    PyTorch shape: (4, 256)
    TTSim   shape: (4, 256)
    Max diff: 9.536743e-07, Mean diff: 7.637163e-08
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

  CanBusMLP bs=8 comparison:
    PyTorch shape: (8, 256)
    TTSim   shape: (8, 256)
    Max diff: 9.536743e-07, Mean diff: 4.672620e-08
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

[OK] TEST 2 PASSED

================================================================================
TEST 3: PerceptionTransformer Construction & Parameter Count
================================================================================
  [OK] PerceptionTransformer constructed
    embed_dims       = 256
    num_feature_lvls = 4
    num_cams         = 6
    Expected params: 39299
    Actual params:   39299
  [OK] Param count matches
  [OK] Config (128,2,4): 10947 params
  [OK] Config (512,3,8): 144643 params

[OK] TEST 3 PASSED

================================================================================
TEST 4: get_states_and_refs – Explicit reference_points
================================================================================

  init_reference_out comparison:
    PyTorch shape: (2, 50, 3)
    TTSim   shape: (2, 50, 3)
    Max diff: 5.960464e-08, Mean diff: 1.288950e-08
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

  decoder query comparison:
    PyTorch shape: (50, 2, 256)
    TTSim   shape: (50, 2, 256)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

  decoder query_pos comparison:
    PyTorch shape: (50, 2, 256)
    TTSim   shape: (50, 2, 256)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

[OK] TEST 4 PASSED

================================================================================
TEST 5: get_states_and_refs – Computed reference_points
================================================================================

  decoder query (computed rp) comparison:
    PyTorch shape: (50, 2, 256)
    TTSim   shape: (50, 2, 256)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

  decoder query_pos (computed rp) comparison:
    PyTorch shape: (50, 2, 256)
    TTSim   shape: (50, 2, 256)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

  reference_points (computed) comparison:
    PyTorch shape: (2, 50, 3)
    TTSim   shape: (2, 50, 3)
    Max diff: 1.788139e-07, Mean diff: 2.870957e-08
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

[OK] TEST 5 PASSED

================================================================================
TEST 6: get_bev_features – CAN-bus MLP Encoding
================================================================================

  BEV queries after CAN-bus MLP comparison:
    PyTorch shape: (100, 2, 256)
    TTSim   shape: (100, 2, 256)
    Max diff: 9.536743e-07, Mean diff: 5.395883e-08
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

[OK] TEST 6 PASSED

================================================================================
TEST 7: get_bev_features – Camera & Level Embeddings
================================================================================

  feat_flatten after cam+level embeds comparison:
    PyTorch shape: (6, 18, 1, 256)
    TTSim   shape: (6, 18, 1, 256)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

[OK] TEST 7 PASSED

================================================================================
TEST 8: Ego-motion Shift Computation
================================================================================
  Expected shift:
    [[-0.01253863  0.06521175]
 [-0.07578483 -0.02723316]]
  Actual shift:
    [[-0.01253863  0.06521175]
 [-0.07578483 -0.02723316]]
  [OK] Shift matches

[OK] TEST 8 PASSED

================================================================================
TEST 9: LiDAR Feature Processing (pts_feats)
================================================================================

  pts_feats (flattened & transposed) comparison:
    PyTorch shape: (2, 25, 256)
    TTSim   shape: (2, 25, 256)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

[OK] TEST 9 PASSED

================================================================================
TEST 10: bev_pos Processing
================================================================================

  bev_pos (flattened, transposed, tiled) comparison:
    PyTorch shape: (16, 2, 64)
    TTSim   shape: (16, 2, 64)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match within tol (rtol=0.0001, atol=1e-05)

[OK] TEST 10 PASSED

================================================================================
TEST SUMMARY
================================================================================
  TEST  1: CanBusMLP Construction & Parameter Count.................... [OK] PASSED
  TEST  2: CanBusMLP Forward Pass vs PyTorch........................... [OK] PASSED
  TEST  3: PerceptionTransformer Construction & Parameter Count........ [OK] PASSED
  TEST  4: get_states_and_refs – Explicit reference_points............. [OK] PASSED
  TEST  5: get_states_and_refs – Computed reference_points............. [OK] PASSED
  TEST  6: get_bev_features – CAN-bus MLP Encoding..................... [OK] PASSED
  TEST  7: get_bev_features – Camera & Level Embeddings................ [OK] PASSED
  TEST  8: Ego-motion Shift Computation................................ [OK] PASSED
  TEST  9: LiDAR Feature Processing (pts_feats)........................ [OK] PASSED
  TEST 10: bev_pos Processing.......................................... [OK] PASSED

Total: 10/10 tests passed

================================================================================
All tests passed!
================================================================================
```

---


