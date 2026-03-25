# Operator Mapping: Transformers vs Polaris-TTNN ViTAttention

## Complete Operator Mapping

### Query Projection Operations

| Transformers | Polaris-TTNN | Match Type |
|--------------|--------------|------------|
| **Op 1024**: MatMul (Q projection) | **Op 0**: MatMul (Q projection) | ✅ **EXACT MATCH** |
| **Op 2048**: Add (Q bias) | **Op 3**: Add (Q bias) | ✅ **EXACT MATCH** |
| **Op 3072**: Untilize | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 4096**: Reshape (Q) | **Op 6**: Reshape (Q) | ✅ **EXACT MATCH** |
| **Op 5120**: TilizeWithValPadding | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 6144**: Transpose (Q) | **Op 9**: Transpose (Q) | ✅ **EXACT MATCH** |

### Key Projection Operations

| Transformers | Polaris-TTNN | Match Type |
|--------------|--------------|------------|
| **Op 7168**: MatMul (K projection) | **Op 1**: MatMul (K projection) | ✅ **EXACT MATCH** |
| **Op 8192**: Add (K bias) | **Op 4**: Add (K bias) | ✅ **EXACT MATCH** |
| **Op 9216**: Untilize | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 10240**: Reshape (K) | **Op 7**: Reshape (K) | ✅ **EXACT MATCH** |
| **Op 11264**: TilizeWithValPadding | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 12288**: PermuteDeviceOperation (K) | **Op 10**: Transpose (K) | ⚠️ **EQUIVALENT** (different op type) |

### Value Projection Operations

| Transformers | Polaris-TTNN | Match Type |
|--------------|--------------|------------|
| **Op 13312**: MatMul (V projection) | **Op 2**: MatMul (V projection) | ✅ **EXACT MATCH** |
| **Op 14336**: Add (V bias) | **Op 5**: Add (V bias) | ✅ **EXACT MATCH** |
| **Op 15360**: Untilize | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 16384**: Reshape (V) | **Op 8**: Reshape (V) | ✅ **EXACT MATCH** |
| **Op 17408**: TilizeWithValPadding | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 18432**: Transpose (V) | **Op 11**: Transpose (V) | ✅ **EXACT MATCH** |

### Attention Computation Operations

| Transformers | Polaris-TTNN | Match Type |
|--------------|--------------|------------|
| **Op 19456**: MatMul (Q@K) | **Op 12**: MatMul (Q@K) | ✅ **EXACT MATCH** |
| **Op 20480**: Mul (scale) | **Op 13**: Mul (scale) | ✅ **EXACT MATCH** |
| **Op 21504**: Add (mask) | **Op 14**: Add (mask) | ✅ **EXACT MATCH** |
| **Op 22528**: Softmax | **Op 15**: Softmax | ✅ **EXACT MATCH** |
| **Op 23552**: MatMul (attn@V) | **Op 16**: MatMul (attn@V) | ✅ **EXACT MATCH** |

### Output Projection Operations

| Transformers | Polaris-TTNN | Match Type |
|--------------|--------------|------------|
| **Op 24576**: Transpose (context) | **Op 17**: Transpose (context) | ✅ **EXACT MATCH** |
| **Op 25600**: UntilizeWithUnpadding | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 26624**: Reshape (context) | **Op 18**: Reshape (context) | ✅ **EXACT MATCH** |
| **Op 27648**: Tilize | *(No equivalent - layout op)* | ⚠️ **LAYOUT CONVERSION** |
| **Op 28672**: MatMul (output) | **Op 19**: MatMul (output) | ✅ **EXACT MATCH** |
| **Op 29696**: Add (output bias) | **Op 20**: Add (output bias) | ✅ **EXACT MATCH** |

---

## Summary Statistics

### Matched Operations
- **Exact Matches**: 18 operations
- **Equivalent Operations**: 1 operation (PermuteDeviceOperation ↔ Transpose)
- **Layout Conversions (Transformers only)**: 8 operations
- **Total Transformers Operations**: 29
- **Total TT-NN Operations**: 21

### Match Rate
- **Core Operations Matched**: 19/19 (100%)
- **Layout Operations**: 0/8 matched (expected - these are framework-specific)

---

## Detailed Mapping with Operation Numbers

### Query Path

```
Transformers                    Polaris-TTNN
─────────────────               ───────────────
Op 1024: MatMul(Q)      ←→      Op 0: MatMul(Q)          ✅
Op 2048: Add(Q)         ←→      Op 3: Add(Q)             ✅
Op 3072: Untilize       ←→      (none - layout op)       ⚠️
Op 4096: Reshape(Q)     ←→      Op 6: Reshape(Q)         ✅
Op 5120: TilizeWithVal  ←→      (none - layout op)       ⚠️
Op 6144: Transpose(Q)   ←→      Op 9: Transpose(Q)       ✅
```

### Key Path

```
Transformers                    Polaris-TTNN
─────────────────               ───────────────
Op 7168: MatMul(K)      ←→      Op 1: MatMul(K)           ✅
Op 8192: Add(K)         ←→      Op 4: Add(K)              ✅
Op 9216: Untilize       ←→      (none - layout op)       ⚠️
Op 10240: Reshape(K)    ←→      Op 7: Reshape(K)          ✅
Op 11264: TilizeWithVal ←→      (none - layout op)       ⚠️
Op 12288: Permute(K)    ←→      Op 10: Transpose(K)      ⚠️ (equivalent)
```

### Value Path

```
Transformers                    Polaris-TTNN
─────────────────               ───────────────
Op 13312: MatMul(V)     ←→      Op 2: MatMul(V)           ✅
Op 14336: Add(V)        ←→      Op 5: Add(V)              ✅
Op 15360: Untilize      ←→      (none - layout op)       ⚠️
Op 16384: Reshape(V)    ←→      Op 8: Reshape(V)          ✅
Op 17408: TilizeWithVal ←→      (none - layout op)       ⚠️
Op 18432: Transpose(V)  ←→      Op 11: Transpose(V)       ✅
```

### Attention Computation

```
Transformers                    Polaris-TTNN
─────────────────               ───────────────
Op 19456: MatMul(Q@K)  ←→      Op 12: MatMul(Q@K)        ✅
Op 20480: Mul(scale)   ←→      Op 13: Mul(scale)          ✅
Op 21504: Add(mask)     ←→      Op 14: Add(mask)           ✅
Op 22528: Softmax      ←→      Op 15: Softmax             ✅
Op 23552: MatMul(attn@V)←→     Op 16: MatMul(attn@V)      ✅
```

### Output Path

```
Transformers                    Polaris-TTNN
─────────────────               ───────────────
Op 24576: Transpose    ←→      Op 17: Transpose           ✅
Op 25600: UntilizeUnpad←→     (none - layout op)         ⚠️
Op 26624: Reshape       ←→      Op 18: Reshape             ✅
Op 27648: Tilize        ←→      (none - layout op)         ⚠️
Op 28672: MatMul(output)←→     Op 19: MatMul(output)      ✅
Op 29696: Add(output)   ←→      Op 20: Add(output)        ✅
```

---

## Operation Ordering Differences

### Transformers Order (Sequential per Q/K/V):
```
Q: MatMul → Add → Untilize → Reshape → Tilize → Transpose
K: MatMul → Add → Untilize → Reshape → Tilize → Permute
V: MatMul → Add → Untilize → Reshape → Tilize → Transpose
Attention: MatMul → Mul → Add → Softmax → MatMul
Output: Transpose → Untilize → Reshape → Tilize → MatMul → Add
```

### TT-NN Order (Batched):
```
All projections: MatMul(Q) → MatMul(K) → MatMul(V)
All biases: Add(Q) → Add(K) → Add(V)
All reshapes: Reshape(Q) → Reshape(K) → Reshape(V)
All transposes: Transpose(Q) → Transpose(K) → Transpose(V)
Attention: MatMul → Mul → Add → Softmax → MatMul
Output: Transpose → Reshape → MatMul → Add
```

**Key Difference**: Transformers processes each Q/K/V completely before moving to the next. TT-NN batches similar operations together.

---

## Layout Conversion Operations (Transformers Only)

These 8 operations have no direct equivalent in TT-NN because they're optimized away:

1. **Op 3072**: Untilize (after Q projection)
2. **Op 5120**: TilizeWithValPadding (before Q transpose)
3. **Op 9216**: Untilize (after K projection)
4. **Op 11264**: TilizeWithValPadding (before K permute)
5. **Op 15360**: Untilize (after V projection)
6. **Op 17408**: TilizeWithValPadding (before V transpose)
7. **Op 25600**: UntilizeWithUnpadding (after context transpose)
8. **Op 27648**: Tilize (before output projection)

**Why they don't exist in TT-NN:**
- Layout management is implicit or handled by the framework
- Graph compiler optimizes them away
- Operations are fused with compute operations
- Inputs/outputs are already in optimal layouts

---

## Visual Mapping Diagram

```
TRANSFORMERS (29 ops)          POLARIS-TTNN (21 ops)
══════════════════════          ════════════════════

Query Path:
  MatMul(Q) ────────────────→  MatMul(Q) [Op 0]
  Add(Q) ──────────────────→  Add(Q) [Op 3]
  Untilize ──────────────────→  (optimized away)
  Reshape(Q) ───────────────→  Reshape(Q) [Op 6]
  TilizeWithVal ─────────────→  (optimized away)
  Transpose(Q) ──────────────→  Transpose(Q) [Op 9]

Key Path:
  MatMul(K) ────────────────→  MatMul(K) [Op 1]
  Add(K) ──────────────────→  Add(K) [Op 4]
  Untilize ──────────────────→  (optimized away)
  Reshape(K) ───────────────→  Reshape(K) [Op 7]
  TilizeWithVal ─────────────→  (optimized away)
  Permute(K) ───────────────→  Transpose(K) [Op 10] ⚠️

Value Path:
  MatMul(V) ────────────────→  MatMul(V) [Op 2]
  Add(V) ──────────────────→  Add(V) [Op 5]
  Untilize ──────────────────→  (optimized away)
  Reshape(V) ───────────────→  Reshape(V) [Op 8]
  TilizeWithVal ─────────────→  (optimized away)
  Transpose(V) ─────────────→  Transpose(V) [Op 11]

Attention:
  MatMul(Q@K) ──────────────→  MatMul(Q@K) [Op 12]
  Mul(scale) ───────────────→  Mul(scale) [Op 13]
  Add(mask) ────────────────→  Add(mask) [Op 14]
  Softmax ──────────────────→  Softmax [Op 15]
  MatMul(attn@V) ───────────→  MatMul(attn@V) [Op 16]

Output:
  Transpose ────────────────→  Transpose [Op 17]
  UntilizeWithUnpadding ─────→  (optimized away)
  Reshape ──────────────────→  Reshape [Op 18]
  Tilize ───────────────────→  (optimized away)
  MatMul(output) ───────────→  MatMul(output) [Op 19]
  Add(output) ───────────────→  Add(output) [Op 20]
```

---

## Key Insights

1. **100% Core Operation Match**: All 19 core computational operations have direct matches
2. **Layout Operations**: 8 layout conversions in Transformers are optimized away in TT-NN
3. **Operation Ordering**: Transformers is sequential, TT-NN is batched for better parallelism
4. **Permute vs Transpose**: One operation type difference (PermuteDeviceOperation vs Transpose) but functionally equivalent
5. **Efficiency**: TT-NN has 27% fewer operations due to layout optimization

