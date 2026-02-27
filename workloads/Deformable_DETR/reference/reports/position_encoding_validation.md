# Position Encoding Validation Report

**Generated:** 2026-02-15 22:00:52

**Test Suite:** PyTorch vs TTSim Position Encoding Implementation Comparison

This report compares the PyTorch and TTSim implementations of position encoding modules.
Includes detailed numerical comparison for sine embeddings (deterministic) and shape testing for learned embeddings (random init).

---


================================================================================
## Test 1: PositionEmbeddingSine
================================================================================

### Configuration
```
Input tensor shape:  [2, 256, 28, 28]
Num pos feats:       128
Temperature:         10000
Normalize:           True
Scale:               2*pi
Output shape:        [2, 256, 28, 28]
```

### Input NestedTensor
```
Tensor shape: [2, 256, 28, 28]
Mask shape:   [2, 28, 28]
Masked pixels: 280/1568
Tensor mean:  -0.00155397
Tensor std:   1.00172794
Sample (first 10): [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
```

### Output Position Embeddings

**Position Embeddings:**
```
PyTorch shape: [2, 256, 28, 28]
TTSim shape:   [2, 256, 28, 28]

PyTorch statistics:
  Mean: 0.46680030
  Std:  0.53112853
  Min:  -1.00000000
  Max:  1.00000000

TTSim statistics:
  Mean: 0.46680030
  Std:  0.53112853
  Min:  -1.00000000
  Max:  1.00000000

Sample values (first 10):
  PyTorch: [-0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, ...]
  TTSim:   [-0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, -0.136167, ...]
```

### Embedding Structure Analysis
```
Output channels: 256 (y_pos: 128, x_pos: 128)

Y-position component:
  PyTorch - Mean: 0.46680027, Std: 0.53112853
  TTSim   - Mean: 0.46680027, Std: 0.53112853

X-position component:
  PyTorch - Mean: 0.46680021, Std: 0.53112853
  TTSim   - Mean: 0.46680021, Std: 0.53112853
```

#### Shape Comparison
```
PyTorch: [2, 256, 28, 28]
TTSim:   [2, 256, 28, 28]
```
**Result:** [PASSED] Shapes match

#### Numerical Comparison
```
Absolute Error:
  Max:  5.960464e-08
  Mean: 3.085485e-09

Relative Error:
  Max:  1.173757e-07
  Mean: 4.476744e-09

Tolerance:
  rtol: 0.0001
  atol: 1e-05

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] PositionEmbeddingSine test

================================================================================
## Test 2: PositionEmbeddingLearned
================================================================================

### Configuration
```
Input tensor shape:  [2, 256, 28, 28]
Num pos feats:       128
Output shape:        [2, 256, 28, 28]
```

### Input NestedTensor
```
Tensor shape: [2, 256, 28, 28]
Mask shape:   [2, 28, 28]
Sample (first 10): [1.926915, 1.487284, 0.900717, -2.105521, 0.678418, -1.234545, -0.043067, -1.604667, -0.752135, 1.648723, ...]
```

### Weight Sync
```
row_embed: PT [50, 128] → TT [50, 128]
col_embed: PT [50, 128] → TT [50, 128]
```

### Output Position Embeddings

**Position Embeddings:**
```
PyTorch shape: [2, 256, 28, 28]
TTSim shape:   [2, 256, 28, 28]

PyTorch statistics:
  Mean: 0.49469829
  Std:  0.28857699
  Min:  0.00006807
  Max:  0.99974877

TTSim statistics:
  Mean: 0.49469829
  Std:  0.28857699
  Min:  0.00006807
  Max:  0.99974877

Sample values (first 10):
  PyTorch: [0.519401, 0.148397, 0.099883, 0.235930, 0.993025, 0.102550, 0.560232, 0.442588, 0.929906, 0.809647, ...]
  TTSim:   [0.519401, 0.148397, 0.099883, 0.235930, 0.993025, 0.102550, 0.560232, 0.442588, 0.929906, 0.809647, ...]
```

#### Shape Comparison
```
PyTorch: [2, 256, 28, 28]
TTSim:   [2, 256, 28, 28]
```
**Result:** [PASSED] Shapes match

#### Numerical Comparison
```
Absolute Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Relative Error:
  Max:  0.000000e+00
  Mean: 0.000000e+00

Tolerance:
  rtol: 0.0001
  atol: 1e-05

Result: PASSED (within tolerance)
```

**Result:** [PASSED] Numerical comparison

### [PASSED] PositionEmbeddingLearned test

================================================================================
## Test 3: build_position_encoding Factory
================================================================================

### Test 3.1: Sine Position Encoding
```
Args:
  hidden_dim: 256
  position_embedding: 'sine'
```

**PyTorch module type:** `PositionEmbeddingSine`
**TTSim module type:** `PositionEmbeddingSine`

#### Shape Comparison
```
PyTorch: [2, 256, 28, 28]
TTSim:   [2, 256, 28, 28]
```
**Result:** [PASSED] Shapes match

**Result:** [PASSED] Sine encoding factory test

### Test 3.2: Learned Position Encoding
```
Args:
  hidden_dim: 256
  position_embedding: 'learned'
```

**PyTorch module type:** `PositionEmbeddingLearned`
**TTSim module type:** `PositionEmbeddingLearned`

#### Shape Comparison
```
PyTorch: [2, 256, 28, 28]
TTSim:   [2, 256, 28, 28]
```
**Result:** [PASSED] Shapes match

**Result:** [PASSED] Learned encoding factory test

### [PASSED] build_position_encoding factory test

================================================================================
# Test Summary
================================================================================

| Metric | Value |
|--------|-------|
| **Tests Passed** | 3/3 |
| **Tests Failed** | 0/3 |
| **Success Rate** | 100.0% |

## Test Details

| Test | Numerical Comparison | Status |
|------|---------------------|--------|
| PositionEmbeddingSine | Yes (deterministic math) | Recommended |
| PositionEmbeddingLearned | Yes (weights synced) | Full numerical |
| build_position_encoding | Partial (sine only) | Factory test |

## [PASSED] All position encoding tests completed successfully!

---


*End of Report*