# Comparison Results (heads) -- 2026-04-15 10:31:14

## test_bricks.py  --  PASS

### stdout

```
======================================================================
bricks.py `run_time` Decorator Validation: PyTorch vs TTSIM
======================================================================

----------------------------------------------------------------------
Test 1: Scalar input / output
----------------------------------------------------------------------
  Input:       7.0
  PyTorch out: 49.0
  TTSIM  out:  49.0
  Value match: [PASS]
  Count maps match: [PASS]  PT=[1.0]  TTSIM=[1.0]

----------------------------------------------------------------------
Test 2: 1-D numpy array (shape + numerical)
----------------------------------------------------------------------
  Input shape:  (128,)
  PT  output shape: (128,)
  TT  output shape: (128,)
  Shape match: [PASS]
  Tolerance: atol=1e-06, rtol=1e-06
  Max  absolute difference: 0.0000000000
  Mean absolute difference: 0.0000000000
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 3: Multi-dimensional numpy array (shape + numerical)
----------------------------------------------------------------------
  Input shape:  (2, 3, 64, 64)
  PT  output shape: (2, 3, 64, 64)
  TT  output shape: (2, 3, 64, 64)
  Shape match: [PASS]
  Tolerance: atol=1e-06, rtol=1e-06
  Max  absolute difference: 0.0000000000
  Mean absolute difference: 0.0000000000
  Output stats PT:   min=0.000000, max=20.062193, mean=0.999250
  Output stats TT:   min=0.000000, max=20.062193, mean=0.999250
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 4: Torch tensor input (shape + numerical)
----------------------------------------------------------------------
  Input shape:  torch.Size([4, 16])
  PT  output shape: (4, 16)
  TT  output shape: (4, 16)
  Shape match: [PASS]
  Tolerance: atol=1e-06, rtol=1e-06
  Max  absolute difference: 0.0000000000
  Mean absolute difference: 0.0000000000
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 5: Multiple invocations — counter accumulation
----------------------------------------------------------------------
  Number of calls: 10
  PT count_maps:  {'multi : sample_fn': 10.0}
  TT count_maps:  {'multi : sample_fn': 10.0}
  Count values match: [PASS]
  Timer keys match:  [PASS]

======================================================================
OVERALL: [PASS] ALL TESTS PASSED — TTSIM matches PyTorch
======================================================================

```

---

## test_functional.py  --  PASS

### stdout

```
======================================================================
FusionAD functional.py - TTSim vs PyTorch Comparison
======================================================================


======================================================================
TEST 1: bivariate_gaussian_activation
======================================================================
Input shape: (4, 8, 5)
Input stats: min=-2.619745, max=2.463242, mean=-0.067602
  PyTorch output shape: (4, 8, 5)
  PyTorch output stats: min=-1.918771, max=11.742821, mean=0.559626
  TTSim  output shape: (4, 8, 5)
  TTSim  output stats: min=-1.918771, max=11.742820, mean=0.559626

------------------------------------------------------------
  Comparison: bivariate_gaussian_activation
------------------------------------------------------------
  PyTorch shape: (4, 8, 5)
  TTSim  shape:  (4, 8, 5)
  Tolerance:     atol=1e-06, rtol=1e-05
  Max abs diff:  0.0000009537
  Mean abs diff: 0.0000000240
  allclose:      True
  [PASS] TTSim matches PyTorch


======================================================================
TEST 2: norm_points
======================================================================
Input shape: (6, 10, 2)
pc_range:    [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
Input stats: min=-162.063370, max=192.636566, mean=2.449786
  PyTorch output shape: (6, 10, 2)
  PyTorch output stats: min=-1.082650, max=2.381216, mean=0.523924
  TTSim  output shape: (6, 10, 2)
  TTSim  output stats: min=-1.082650, max=2.381216, mean=0.523924

------------------------------------------------------------
  Comparison: norm_points
------------------------------------------------------------
  PyTorch shape: (6, 10, 2)
  TTSim  shape:  (6, 10, 2)
  Tolerance:     atol=1e-06, rtol=1e-05
  Max abs diff:  0.0000000000
  Mean abs diff: 0.0000000000
  allclose:      True
  [PASS] TTSim matches PyTorch


======================================================================
TEST 3: pos2posemb2d
======================================================================
Input shape:   (3, 5, 2)
num_pos_feats: 64
temperature:   10000
Input stats: min=-1.952088, max=2.133033, mean=0.145605
  PyTorch output shape: (3, 5, 128)
  PyTorch output stats: min=-0.999994, max=1.000000, mean=0.410828
  TTSim  output shape: (3, 5, 128)
  TTSim  output stats: min=-0.999994, max=1.000000, mean=0.410828

------------------------------------------------------------
  Comparison: pos2posemb2d
------------------------------------------------------------
  PyTorch shape: (3, 5, 128)
  TTSim  shape:  (3, 5, 128)
  Tolerance:     atol=1e-06, rtol=1e-05
  Max abs diff:  0.0000000596
  Mean abs diff: 0.0000000034
  allclose:      True
  [PASS] TTSim matches PyTorch


======================================================================
TEST 4: rot_2d
======================================================================
Input shape: (8,)
Input stats: min=-0.310267, max=1.305479, mean=0.527274
  PyTorch output shape: (8, 2, 2)
  TTSim  output shape: (8, 2, 2)

------------------------------------------------------------
  Comparison: rot_2d
------------------------------------------------------------
  PyTorch shape: (8, 2, 2)
  TTSim  shape:  (8, 2, 2)
  Tolerance:     atol=1e-06, rtol=1e-05
  Max abs diff:  0.0000000298
  Mean abs diff: 0.0000000019
  allclose:      True
  [PASS] TTSim matches PyTorch


======================================================================
TEST 5: anchor_coordinate_transform
======================================================================
Anchors shape:  (2, 4, 12, 2)  (G, M, T, 2)
Yaw shape:      (3, 1)       (A, 1)
Centers shape:  (3, 3)    (A, 3)
  PyTorch output shape: (3, 2, 4, 12, 2)
  TTSim  output shape: (3, 2, 4, 12, 2)

------------------------------------------------------------
  Comparison: anchor_coordinate_transform
------------------------------------------------------------
  PyTorch shape: (3, 2, 4, 12, 2)
  TTSim  shape:  (3, 2, 4, 12, 2)
  Tolerance:     atol=1e-06, rtol=1e-05
  Max abs diff:  0.0000002384
  Mean abs diff: 0.0000000102
  allclose:      True
  [PASS] TTSim matches PyTorch


======================================================================
TEST 6: trajectory_coordinate_transform
======================================================================
Trajectory shape: (3, 2, 4, 12, 2)  (A, G, P, T, 2)
Yaw shape:        (3, 1)       (A, 1)
Centers shape:    (3, 3)    (A, 3)
  PyTorch output shape: (3, 2, 4, 12, 2)
  TTSim  output shape: (3, 2, 4, 12, 2)

------------------------------------------------------------
  Comparison: trajectory_coordinate_transform
------------------------------------------------------------
  PyTorch shape: (3, 2, 4, 12, 2)
  TTSim  shape:  (3, 2, 4, 12, 2)
  Tolerance:     atol=1e-06, rtol=1e-05
  Max abs diff:  0.0000004768
  Mean abs diff: 0.0000000364
  allclose:      True
  [PASS] TTSim matches PyTorch


======================================================================
SUMMARY
======================================================================
  [PASS]  bivariate_gaussian_activation
  [PASS]  norm_points
  [PASS]  pos2posemb2d
  [PASS]  rot_2d
  [PASS]  anchor_coordinate_transform
  [PASS]  trajectory_coordinate_transform

All 6 tests PASSED!
======================================================================
```

---

## test_grid_mask.py  --  PASS

### stdout

```
======================================================================
grid_mask.py Validation: PyTorch vs TTSIM
======================================================================

----------------------------------------------------------------------
Test 1: GridMask — simple masking (offset=False, mode=0)
----------------------------------------------------------------------
  Input shape:  (2, 3, 32, 32)
  Mask shape:   (32, 32)
  PT  output shape: (2, 3, 32, 32)
  TT  output shape: (2, 3, 32, 32)
  Shape match: [PASS]
  Tolerance: atol=1e-06, rtol=1e-06
  Max  abs diff:  0.0000000000
  Mean abs diff:  0.0000000000
  PT  stats: min=-3.019512, max=3.109919, mean=-0.011365
  TT  stats: min=-3.019512, max=3.109919, mean=-0.011365
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 2: GridMask — inverted mode (mode=1)
----------------------------------------------------------------------
  PT  output shape: (2, 3, 32, 32)
  TT  output shape: (2, 3, 32, 32)
  Shape match: [PASS]
  Max  abs diff:  0.0000000000
  Mean abs diff:  0.0000000000
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 3: GridMask — with offset (offset=True)
----------------------------------------------------------------------
  PT  output shape: (2, 3, 32, 32)
  TT  output shape: (2, 3, 32, 32)
  Shape match: [PASS]
  Max  abs diff:  0.0000000000
  Mean abs diff:  0.0000000000
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 4: Grid class — [C,H,W] tensor, offset=False
----------------------------------------------------------------------
  Input shape:  [3,24,24]
  PT  output shape: (3, 24, 24)
  TT  output shape: [3, 24, 24]
  Shape match: [PASS]
  Max  abs diff:  0.0000000000
  Mean abs diff:  0.0000000000
  PT  stats: min=-2.686723, max=4.787365, mean=0.014300
  TT  stats: min=-2.686723, max=4.787365, mean=0.014300
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 5: Grid class — [C,H,W] tensor, offset=True
----------------------------------------------------------------------
  PT  output shape: (3, 24, 24)
  TT  output shape: [3, 24, 24]
  Shape match: [PASS]
  Max  abs diff:  0.0000000000
  Mean abs diff:  0.0000000000
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 6: GridMask — larger batch [4,8,64,64]
----------------------------------------------------------------------
  Input shape:  (4,8,64,64)
  PT  output shape: (4, 8, 64, 64)
  TT  output shape: (4, 8, 64, 64)
  Shape match: [PASS]
  Max  abs diff:  0.0000000000
  Mean abs diff:  0.0000000000
  PT  stats: min=-4.295065, max=4.158942, mean=-0.002153
  TT  stats: min=-4.295065, max=4.158942, mean=-0.002153
  Numerical match: [PASS]

----------------------------------------------------------------------
Test 7: set_prob — probability update consistency
----------------------------------------------------------------------
  epoch=5, max_epoch=10
  PT  prob: 0.4
  TT  prob: 0.4
  Prob match: [PASS]

======================================================================
OVERALL: [PASS] ALL TESTS PASSED — TTSIM matches PyTorch
======================================================================

```

### stderr

```
C:\Users\SaSagar\Downloads\TensTorrent\polaris\workloads\FusionAD\reference\mmdet_plugin\models\utils\test_grid_mask.py:142: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\torch\csrc\utils\tensor_numpy.cpp:219.)
  mask = torch.from_numpy(mask).float()
```

---


