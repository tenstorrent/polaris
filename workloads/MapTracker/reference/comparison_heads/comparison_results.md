# Comparison Results (bevformer) — 2026-04-27 09:28:01

## test_map_seg_head.py  —  PASS

### stdout

```

================================================================================
MapSegHead Test Suite: PyTorch vs TTSim
================================================================================

================================================================================
TEST 1: Module Construction
================================================================================

Configuration:
  Classes: 3
  Input channels: 64
  Embed dims: 64
  BEV size: (50, 25)
  Canvas size: (100, 50)

[OK] PyTorch model constructed
[OK] TTSim model constructed

[OK] TEST PASSED: Both models constructed successfully

================================================================================
TEST 2: Forward Pass Comparison
================================================================================

Test configuration:
  Batch size: 1
  Input: [1, 64, 50, 25]
  Expected output preds: [1, 3, 100, 50]
  Expected output feats: [1, 64, 50, 25]

[OK] Input created: (1, 64, 50, 25)

--------------------------------------------------------------------------------
Injecting PyTorch weights into TTSim model
--------------------------------------------------------------------------------
[OK] Injected conv_in weight: (64, 64, 3, 3)
[OK] Injected conv_up_0 weight: (64, 64, 3, 3), bias: (64,)
[OK] Injected conv_out weight: (3, 64, 1, 1), bias: (3,)

[OK] All weights injected successfully

--------------------------------------------------------------------------------
PyTorch Forward Pass
--------------------------------------------------------------------------------
Predictions shape: torch.Size([1, 3, 100, 50])
Predictions range: [-0.314742, 0.503101]
Seg features shape: torch.Size([1, 64, 50, 25])
Seg features range: [0.000000, 0.911500]

--------------------------------------------------------------------------------
TTSim Forward Pass
--------------------------------------------------------------------------------
Predictions shape: Shape([1, 3, 100, 50])
Predictions range: [-0.314743, 0.503101]
Seg features shape: Shape([1, 64, 50, 25])
Seg features range: [0.000000, 0.911501]

================================================================================
Output Comparison
================================================================================

Predictions comparison:
  Max absolute diff: 3.5762786865e-07
  Mean absolute diff: 5.3588525617e-08
  Median absolute diff: 4.4703483582e-08

Seg features comparison:
  Max absolute diff: 2.1457672119e-06
  Mean absolute diff: 1.4788963654e-07
  Median absolute diff: 7.4505805969e-08

Validation (rtol=1e-5, atol=1e-5):
  Predictions match: [OK] YES
  Seg features match: [OK] YES

[OK] TEST PASSED: All outputs match within tolerance

================================================================================
TEST 3: Different Upsampling Configurations
================================================================================

2x upsampling (square):
  BEV: (25, 25), Canvas: (50, 50)
  Preds diff: max=1.937151e-07, match=True
  Feats diff: max=1.847744e-06, match=True
  [OK] PASSED

2x upsampling (rectangular):
  BEV: (50, 25), Canvas: (100, 50)
  Preds diff: max=2.682209e-07, match=True
  Feats diff: max=2.205372e-06, match=True
  [OK] PASSED

[OK] TEST PASSED: All configurations work correctly

================================================================================
Test Summary
================================================================================
Construction................................................ [OK] PASSED
Forward Pass................................................ [OK] PASSED
Different Sizes............................................. [OK] PASSED

Total: 3/3 tests passed

! All tests passed!
```

---

## test_mapdetectorhead.py  —  PASS

### stdout

```

================================================================================
MapDetectorHead TTSim Implementation Tests
================================================================================

================================================================================
TEST 1: RegressionBranch
================================================================================
Input shape: (2, 100, 256)
Output shape: (2, 100, 40)
Max difference: 8.344650e-07
Mean difference: 1.407595e-07
  Total RegressionBranch params: 414,760
PyTorch params: 416,808
[OK] RegressionBranch test PASSED

================================================================================
TEST 2: ClassificationBranch
================================================================================
Input shape: (2, 100, 256)
Output shape: (2, 100, 3)
Max difference: 4.768372e-07
Mean difference: 5.484074e-08
  ClassificationBranch params: 771
PyTorch params: 771
[OK] ClassificationBranch test PASSED

================================================================================
TEST 3: MapDetectorHead Component Construction
================================================================================
Created MapDetectorHead:
  num_queries: 100
  num_classes: 3
  embed_dims: 256
  num_points: 20
  num_layers: 6
  Classification branches: 6
  Regression branches: 6

BEV features processing:
  Input shape: (2, 128, 50, 100)
  Output shape: Shape([2, 256, 50, 100])
    MapDetectorHead 'map_detector_head':
      Input projection: 33,024
      Query embedding: 25,600
      Reference points: 10,280
      Cls branch 0: 771
      Cls branch 1: 771
      Cls branch 2: 771
      Cls branch 3: 771
      Cls branch 4: 771
      Cls branch 5: 771
      Reg branch 0: 414,760
      Reg branch 1: 414,760
      Reg branch 2: 414,760
      Reg branch 3: 414,760
      Reg branch 4: 414,760
      Reg branch 5: 414,760
    Total MapDetectorHead params: 2,562,090

[OK] MapDetectorHead component test PASSED

================================================================================
TEST 4: BEV Positional Encoding Values
================================================================================
  Correct (cumsum) range: [-1.0000, 1.0000]
  TTSim (linspace)  range: [-1.0000, 1.0000]
  Max diff:  0.000000e+00
  Mean diff: 0.000000e+00
[OK] BEV positional encoding values match SinePositionalEncoding

================================================================================
TEST SUMMARY
================================================================================
RegressionBranch: [OK] PASSED
ClassificationBranch: [OK] PASSED
MapDetectorHead Components: [OK] PASSED
BEV Pos Encoding Values: [OK] PASSED

================================================================================
ALL TESTS PASSED [OK]
================================================================================
```

---


