# Operator Sequence Differences: Transformers vs Polaris-TTNN ViTAttention

## Executive Summary

The operator sequences are **fundamentally different** due to:
1. **Framework-level optimizations**: Transformers-based uses automatic layout management with explicit conversions
2. **Graph compilation differences**: Different graph optimization passes
3. **Hardware abstraction levels**: Transformers goes through PyTorch → TT-NN conversion, while TT-NN is direct

**Transformers-based**: 29 operations (includes 8 layout conversion operations)
**Polaris-TTNN-based**: 21 operations (no explicit layout conversions in the graph)

---

## Detailed Operator Sequence Comparison

### Transformers-Based Sequence (29 operations)

| Op# | Operation | Purpose | Notes |
|-----|-----------|---------|-------|
| 1024 | **MatMul** | Query projection | `(8,224,768) @ (768,768)` |
| 2048 | **Add** | Query bias | Broadcast add |
| 3072 | **Untilize** | Layout: TILE → ROW_MAJOR | ⚠️ Layout conversion |
| 4096 | **Reshape** | Reshape for heads | `(8,224,768) → (8,224,12,64)` |
| 5120 | **TilizeWithValPadding** | Layout: ROW_MAJOR → TILE | ⚠️ Layout conversion |
| 6144 | **Transpose** | Permute query | `(8,224,12,64) → (8,12,224,64)` |
| 7168 | **MatMul** | Key projection | `(8,224,768) @ (768,768)` |
| 8192 | **Add** | Key bias | Broadcast add |
| 9216 | **Untilize** | Layout: TILE → ROW_MAJOR | ⚠️ Layout conversion |
| 10240 | **Reshape** | Reshape for heads | `(8,224,768) → (8,224,12,64)` |
| 11264 | **TilizeWithValPadding** | Layout: ROW_MAJOR → TILE | ⚠️ Layout conversion |
| 12288 | **PermuteDeviceOperation** | Permute key | `(8,224,12,64) → (8,12,64,224)` |
| 13312 | **MatMul** | Value projection | `(8,224,768) @ (768,768)` |
| 14336 | **Add** | Value bias | Broadcast add |
| 15360 | **Untilize** | Layout: TILE → ROW_MAJOR | ⚠️ Layout conversion |
| 16384 | **Reshape** | Reshape for heads | `(8,224,768) → (8,224,12,64)` |
| 17408 | **TilizeWithValPadding** | Layout: ROW_MAJOR → TILE | ⚠️ Layout conversion |
| 18432 | **Transpose** | Permute value | `(8,224,12,64) → (8,12,224,64)` |
| 19456 | **MatMul** | Attention scores | `(8,12,224,64) @ (8,12,64,224) → (8,12,224,224)` |
| 20480 | **Mul** | Scale attention | Multiply by `1/sqrt(64)` |
| 21504 | **Add** | Attention mask | Broadcast add mask |
| 22528 | **Softmax** | Attention probabilities | Softmax on dim=-1 |
| 23552 | **MatMul** | Context layer | `(8,12,224,224) @ (8,12,224,64) → (8,12,224,64)` |
| 24576 | **Transpose** | Permute context | `(8,12,224,64) → (8,224,12,64)` |
| 25600 | **UntilizeWithUnpadding** | Layout: TILE → ROW_MAJOR | ⚠️ Layout conversion |
| 26624 | **Reshape** | Reshape back | `(8,224,12,64) → (8,224,768)` |
| 27648 | **Tilize** | Layout: ROW_MAJOR → TILE | ⚠️ Layout conversion |
| 28672 | **MatMul** | Output projection | `(8,224,768) @ (768,768)` |
| 29696 | **Add** | Output bias | Broadcast add |

**Total: 29 operations**
- **Layout conversions: 8** (Untilize, TilizeWithValPadding, Tilize, UntilizeWithUnpadding)
- **Core operations: 21**

---

### Polaris-TTNN-Based Sequence (21 operations)

| Op# | Operation | Purpose | Notes |
|-----|-----------|---------|-------|
| 0 | **MatMul** | Query projection | `(8,224,768) @ (768,768)` |
| 1 | **MatMul** | Key projection | `(8,224,768) @ (768,768)` |
| 2 | **MatMul** | Value projection | `(8,224,768) @ (768,768)` |
| 3 | **Add** | Query bias | Broadcast add |
| 4 | **Add** | Key bias | Broadcast add |
| 5 | **Add** | Value bias | Broadcast add |
| 6 | **Reshape** | Reshape query | `(8,224,768) → (8,224,12,64)` |
| 7 | **Reshape** | Reshape key | `(8,224,768) → (8,224,12,64)` |
| 8 | **Reshape** | Reshape value | `(8,224,768) → (8,224,12,64)` |
| 9 | **Transpose** | Permute query | `(8,224,12,64) → (8,12,224,64)` |
| 10 | **Transpose** | Permute key | `(8,224,12,64) → (8,12,64,224)` |
| 11 | **Transpose** | Permute value | `(8,224,12,64) → (8,12,224,64)` |
| 12 | **MatMul** | Attention scores | `(8,12,224,64) @ (8,12,64,224) → (8,12,224,224)` |
| 13 | **Mul** | Scale attention | Multiply by `1/sqrt(64)` |
| 14 | **Add** | Attention mask | Broadcast add mask |
| 15 | **Softmax** | Attention probabilities | Softmax on dim=-1 |
| 16 | **MatMul** | Context layer | `(8,12,224,224) @ (8,12,224,64) → (8,12,224,64)` |
| 17 | **Transpose** | Permute context | `(8,12,224,64) → (8,224,12,64)` |
| 18 | **Reshape** | Reshape back | `(8,224,12,64) → (8,224,768)` |
| 19 | **MatMul** | Output projection | `(8,224,768) @ (768,768)` |
| 20 | **Add** | Output bias | Broadcast add |

**Total: 21 operations**
- **Layout conversions: 0** (handled implicitly or optimized away)
- **Core operations: 21**

---

## Key Differences Explained

### 1. **Layout Conversion Operations**

#### Transformers-Based (8 layout operations):
- **Untilize** (3x): Converts from TILE_LAYOUT to ROW_MAJOR_LAYOUT
- **TilizeWithValPadding** (3x): Converts from ROW_MAJOR_LAYOUT to TILE_LAYOUT with padding
- **UntilizeWithUnpadding** (1x): Converts from TILE_LAYOUT to ROW_MAJOR_LAYOUT with unpadding
- **Tilize** (1x): Converts from ROW_MAJOR_LAYOUT to TILE_LAYOUT

**Why they appear:**
- Transformers model is converted from PyTorch (row-major) to TT-NN
- The conversion process inserts explicit layout conversions
- These are needed because:
  - PyTorch tensors are row-major by default
  - TT-NN operations require TILE_LAYOUT for efficiency
  - The graph compiler doesn't optimize these away

#### Polaris-TTNN-Based (0 layout operations):
- No explicit layout conversion operations in the graph
- Layout management is either:
  - **Implicit**: Handled by the framework automatically
  - **Optimized away**: Graph compiler fuses or eliminates unnecessary conversions
  - **Pre-converted**: Inputs are already in the correct layout

**Why they don't appear:**
- Direct TT-NN implementation starts with correct layouts
- Graph optimization passes remove redundant conversions
- Framework handles layout management transparently

---

### 2. **Operation Ordering**

#### Transformers-Based:
```
For each Q/K/V:
  MatMul → Add → Untilize → Reshape → TilizeWithValPadding → Transpose
```
- **Interleaved**: Layout conversions between each step
- **Sequential**: One projection at a time with full layout pipeline

#### Polaris-TTNN-Based:
```
All projections first:
  MatMul (Q) → MatMul (K) → MatMul (V)
Then all biases:
  Add (Q) → Add (K) → Add (V)
Then all reshapes:
  Reshape (Q) → Reshape (K) → Reshape (V)
Then all transposes:
  Transpose (Q) → Transpose (K) → Transpose (V)
```
- **Batched**: Similar operations grouped together
- **Optimized**: Better for parallel execution and memory access patterns

---

### 3. **Permute vs Transpose**

#### Transformers-Based:
- Uses **PermuteDeviceOperation** for key permutation
- Uses **Transpose** for query and value

#### Polaris-TTNN-Based:
- Uses **Transpose** for all permutations (Q, K, V)

**Why different:**
- Different graph compilation paths
- PermuteDeviceOperation may be a device-specific optimization
- Both achieve the same result: `(8,224,12,64) → (8,12,64,224)` for key

---

### 4. **Graph Optimization Level**

#### Transformers-Based:
- **Less optimized**: Preserves explicit layout conversions
- **More verbose**: Shows all intermediate steps
- **Framework conversion**: PyTorch → TT-NN conversion adds overhead

#### Polaris-TTNN-Based:
- **More optimized**: Layout conversions removed/fused
- **Cleaner graph**: Only essential operations
- **Direct implementation**: No conversion overhead

---

## Why These Differences Exist

### 1. **Framework Conversion Pipeline**

**Transformers path:**
```
PyTorch Model (row-major)
  ↓
PyTorch → ONNX conversion
  ↓
ONNX → TT-NN conversion
  ↓
TT-NN Graph (with explicit layout ops)
```

**Polaris-TTNN path:**
```
TT-NN Functional Code
  ↓
Direct TT-NN Graph Construction
  ↓
Graph Optimization Passes
  ↓
Optimized TT-NN Graph (layout ops removed)
```

### 2. **Graph Compiler Optimizations**

The Polaris-TTNN implementation benefits from:
- **Layout inference**: Framework infers optimal layouts
- **Operation fusion**: Layout conversions fused with operations
- **Dead code elimination**: Unnecessary conversions removed
- **Memory optimization**: Layouts chosen to minimize conversions

The Transformers path:
- **Explicit conversions**: Must preserve for correctness
- **Conservative optimization**: Can't remove conversions without analysis
- **Conversion safety**: Maintains compatibility with PyTorch semantics

### 3. **Hardware Abstraction**

**Transformers-based:**
- Higher-level abstraction (PyTorch)
- More generic (works with multiple backends)
- Explicit control flow preserved

**Polaris-TTNN-based:**
- Lower-level abstraction (TT-NN)
- Hardware-specific optimizations
- Compiler can make more aggressive optimizations

---

## Impact on Performance

### Operation Count:
- **Transformers**: 29 operations (38% more)
- **TT-NN**: 21 operations

### Layout Conversion Overhead:
- **Transformers**: 8 explicit layout conversions
- **TT-NN**: 0 (or implicit, optimized)

### Memory Access Patterns:
- **Transformers**: More memory traffic due to layout conversions
- **TT-NN**: Better memory locality (fewer conversions)

### Execution Efficiency:
- **Transformers**: Sequential, interleaved operations
- **TT-NN**: Batched, grouped operations (better for parallel execution)

---

## Summary

| Aspect | Transformers-Based | Polaris-TTNN-Based |
|--------|-------------------|-------------------|
| **Total Operations** | 29 | 21 |
| **Layout Conversions** | 8 explicit | 0 (implicit/optimized) |
| **Operation Ordering** | Interleaved | Batched |
| **Graph Optimization** | Less aggressive | More aggressive |
| **Framework Path** | PyTorch → TT-NN | Direct TT-NN |
| **Abstraction Level** | High (PyTorch) | Low (TT-NN) |

**Key Takeaway**: The differences arise from the **framework conversion pipeline** and **graph optimization level**. The Polaris-TTNN implementation benefits from direct TT-NN graph construction and aggressive compiler optimizations that eliminate unnecessary layout conversions, resulting in a cleaner, more efficient operator sequence.

