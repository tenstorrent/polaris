# Shape Inference Rules Documentation

## Overview
This document describes the shape inference rules for operators in the Polaris simulator. Shape inference is performed without allocating memory for tensors (except where explicitly required, e.g., Reshape operator).

## General Principles

### Broadcasting Rules
When operators support broadcasting (e.g., element-wise operations), the following rules apply:
1. Dimensions are aligned from right to left
2. Each dimension pair must either:
   - Be equal
   - Have one dimension = 1 (which will be broadcast)
   - Have one input missing the dimension (treated as 1)

Example:
```
Shape1:     (3, 1, 4)
Shape2:     (   2, 4)
Result:     (3, 2, 4)
```

## Operator Shape Inference Rules

### Matrix Multiplication (MatMul)
- Input shapes: `(M, K)`, `(K, N)` → Output shape: `(M, N)`
- For batched inputs:
  - `(B1, ..., Bn, M, K)`, `(B1, ..., Bn, K, N)` → `(B1, ..., Bn, M, N)`
  - Batch dimensions must be broadcastable
- Special cases:
  - Vector-vector: `(K)`, `(K)` → `()` (scalar)
  - Matrix-vector: `(M, K)`, `(K)` → `(M)`
  - Vector-matrix: `(K)`, `(K, N)` → `(N)`

### Convolution (Conv)
- Input shapes: `(N, C, D1, ..., Dn)`, `(O, C, K1, ..., Kn)`
- Output shape: `(N, O, D1', ..., Dn')`
- For each spatial dimension i:
  ```
  Di' = floor((Di + pad_start + pad_end - (Ki - 1) * dilation - 1) / stride + 1)
  ```
- Auto padding modes:
  - SAME_UPPER/LOWER: Output size = ceil(input_size / stride)
  - VALID: No padding
  - NOTSET: Use explicit padding values

### Pooling (MaxPool, AveragePool)
- Input shape: `(N, C, D1, ..., Dn)`
- Output shape: `(N, C, D1', ..., Dn')`
- Shape calculation same as convolution
- Global pooling:
  - Input shape: `(N, C, D1, ..., Dn)`
  - Output shape: `(N, C, 1, ..., 1)`

### Element-wise Operations
- Binary (Add, Mul):
  - Inputs broadcast according to broadcasting rules
  - Output shape is the broadcast shape
- Unary (Relu, Tanh):
  - Output shape identical to input shape

### Reshape
- Input shape: `(D1, ..., Dn)`
- New shape: `(D1', ..., Dm')`
- Rules:
  - Product of input dimensions must equal product of output dimensions
  - One dimension can be -1 (automatically calculated)
  - Requires actual shape tensor as second input

### Transpose
- Input shape: `(D1, ..., Dn)`
- Output shape: `(Dp1, ..., Dpn)` where `pi` is the i-th permutation index
- Default: Reverses all dimensions
- Custom permutation must include all axes exactly once

### Concatenation
- Input shapes: `[(D1, ..., Dn), ...]`
- Output shape: Same as inputs except concatenation axis
- Rules:
  - All inputs must have same rank
  - All dimensions except concat axis must match
  - Output dimension on concat axis is sum of input dimensions

### Split
- Input shape: `(D1, ..., Dn)`
- Output shapes: `[(D1, ..., Di', ..., Dn), ...]`
- Rules:
  - Sum of split sizes must equal input dimension on split axis
  - All other dimensions remain unchanged

### Batch Normalization
- Input shape: `(N, C, D1, ..., Dn)`
- Additional inputs (scale, bias, mean, var): `(C)`
- Output shape: Same as input shape

## Performance Statistics

Each operator provides performance statistics including:
- Input elements and bytes
- Output elements and bytes
- Operation-specific instruction counts:
  - MatMul: FMA (Fused Multiply-Add) operations
  - Conv: FMA operations per output element
  - Element-wise: One operation per output element
  - Memory operations: Bytes read/written
