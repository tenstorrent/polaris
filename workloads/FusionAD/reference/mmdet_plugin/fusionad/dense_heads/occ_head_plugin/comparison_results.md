# Comparison Results (heads) -- 2026-04-15 10:25:02

## test_occ_modules.py  --  PASS

### stdout

```
================================================================================
TEST 1: BevFeatureSlicer (identity)
================================================================================

  identity passthrough:
    PyTorch shape: [2, 64, 200, 200]
    TTSim   shape: [2, 64, 200, 200]
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-06)

[OK] TEST 1

================================================================================
TEST 2: BevFeatureSlicer (grid sample)
================================================================================

  grid_sample vs PyTorch:
    PyTorch shape: [1, 32, 200, 200]
    TTSim   shape: [1, 32, 200, 200]
    Max diff: 4.768372e-07, Mean diff: 1.981184e-08
    [OK] Match (atol=0.0001)

[OK] TEST 2

================================================================================
TEST 3: MLP (3 layers, 256->256->256->64)
================================================================================

  MLP forward:
    PyTorch shape: [2, 10, 64]
    TTSim   shape: [2, 10, 64]
    Max diff: 1.192093e-07, Mean diff: 1.473163e-08
    [OK] Match (atol=0.0001)

[OK] TEST 3

================================================================================
TEST 4: SimpleConv2d (num_conv=1, shape check)
================================================================================

  Shape: Shape([1, 32, 50, 50])  (expected [1, 32, 50, 50])
  [OK] shape correct

[OK] TEST 4

================================================================================
TEST 5: SimpleConv2d (num_conv=4, shape check)
================================================================================

  Shape: Shape([1, 256, 50, 50])  (expected [1, 256, 50, 50])
  [OK] shape correct

[OK] TEST 5

================================================================================
TEST 6: SimpleConv2d (4 convs, numerical weight-copy match)
================================================================================

  SimpleConv2d numerical:
    PyTorch shape: [1, 32, 16, 16]
    TTSim   shape: [1, 32, 16, 16]
    Max diff: 1.024455e-07, Mean diff: 1.841559e-08
    [OK] Match (atol=0.001)

[OK] TEST 6

================================================================================
TEST 7: UpsamplingAdd (numerical weight-copy match)
================================================================================

  UpsamplingAdd forward:
    PyTorch shape: [1, 64, 24, 24]
    TTSim   shape: [1, 64, 24, 24]
    Max diff: 4.768372e-07, Mean diff: 3.704253e-08
    [OK] Match (atol=0.001)

[OK] TEST 7

================================================================================
TEST 8: Bottleneck (no downsample, numerical weight-copy match)
================================================================================

  Bottleneck no-ds forward:
    PyTorch shape: [1, 64, 20, 20]
    TTSim   shape: [1, 64, 20, 20]
    Max diff: 2.384186e-07, Mean diff: 1.150033e-08
    [OK] Match (atol=0.001)

[OK] TEST 8

================================================================================
TEST 9: Bottleneck (downsample, even dims=20x20, numerical)
================================================================================

  Bottleneck ds-even forward:
    PyTorch shape: [1, 64, 10, 10]
    TTSim   shape: [1, 64, 10, 10]
    Max diff: 5.960464e-07, Mean diff: 6.050405e-08
    [OK] Match (atol=0.001)

[OK] TEST 9

================================================================================
TEST 10: Bottleneck (downsample, odd dims=25x25, numerical)
================================================================================

  Bottleneck ds-odd forward:
    PyTorch shape: [1, 64, 13, 13]
    TTSim   shape: [1, 64, 13, 13]
    Max diff: 5.960464e-07, Mean diff: 5.808246e-08
    [OK] Match (atol=0.001)

[OK] TEST 10

================================================================================
TEST 11: CVT_DecoderBlock (no residual, no upsample, numerical)
================================================================================

  CVT_DecoderBlock no-res forward:
    PyTorch shape: [1, 64, 16, 16]
    TTSim   shape: [1, 64, 16, 16]
    Max diff: 5.364418e-07, Mean diff: 4.353393e-08
    [OK] Match (atol=0.001)

[OK] TEST 11

================================================================================
TEST 12: CVT_DecoderBlock (residual + upsample, numerical)
================================================================================

  CVT_DecoderBlock res+up forward:
    PyTorch shape: [1, 64, 32, 32]
    TTSim   shape: [1, 64, 32, 32]
    Max diff: 7.450581e-07, Mean diff: 5.771866e-08
    [OK] Match (atol=0.001)

[OK] TEST 12

================================================================================
TEST 13: predict_instance_segmentation_and_trajectories
================================================================================

  Output dtype: int64  (expected int64)
  Output shape: [1, 3, 10, 10]  (expected [1, 3, 10, 10])
  Background=0 in border: True
  Unique IDs: [0 1 2 3 4 5]  (count=6, max allowed=6)
  [OK] dtype
  [OK] shape
  [OK] background is 0 in border
  [OK] consecutive IDs

[OK] TEST 13

================================================================================
RESULTS: 13/13 tests passed, 0 failed.
ALL TESTS PASSED!
================================================================================
```

---


