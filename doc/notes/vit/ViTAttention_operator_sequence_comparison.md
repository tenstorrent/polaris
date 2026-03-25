# Operator Sequence, Tensor Shapes, and Types Comparison
## transformers.models.vit.modeling_vit.ViTAttention vs workloads.ttnn.vit.ttnn_functional_vit.vit_attention

## Executive Summary

**Answer: NO - The two implementations will NOT result in the same sequence of operators, but they WILL produce the same tensor shapes and types at equivalent computation points.**

### Key Differences:
1. **Operator Sequence**: TT-NN has additional layout conversion operations (`ttnn.to_layout`) that Transformers doesn't have
2. **Tensor Shapes**: **SAME** at equivalent computation points (after accounting for layout differences)
3. **Tensor Types**: **SAME** (both use bfloat16)
4. **Dropout**: Transformers has dropout in output layer (disabled in eval mode), TT-NN doesn't have dropout operations

---

## Detailed Step-by-Step Comparison

### Input Assumptions
- `batch_size = 8`
- `sequence_size = 224` 
- `hidden_size = 768`
- `num_heads = 12`
- `head_size = 64` (768 / 12)
- `dtype = bfloat16`
- Model in `eval()` mode (dropout disabled)

### Input Tensor
**Both implementations:**
- Shape: `(8, 224, 768)`
- Type: `bfloat16`
- Layout: PyTorch (row-major) vs TT-NN (TILE_LAYOUT after conversion)

---

## Step-by-Step Operator Sequence

### 1. Query Projection

#### Transformers ViTAttention:
```python
# Inside ViTSelfAttention.forward()
query = self.query(hidden_states)  # Linear layer
# Operations: torch.nn.functional.linear(input, weight, bias)
# Shape: (8, 224, 768) -> (8, 224, 768)
# Type: bfloat16
```

#### TT-NN vit_attention:
```82:98:workloads/ttnn/vit/ttnn_functional_vit.py
def vit_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = hidden_states @ parameters.attention.query.weight
    query = query + parameters.attention.query.bias
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
    query = ttnn.permute(query, (0, 2, 1, 3))
```

**Operators:**
1. `@` (matmul): `(8, 224, 768) @ (768, 768)` → `(8, 224, 768)`
2. `+` (add bias): `(8, 224, 768) + (1, 768)` → `(8, 224, 768)`
3. `ttnn.to_layout(ROW_MAJOR_LAYOUT)`: Layout conversion (no shape change)
4. `ttnn.reshape`: `(8, 224, 768)` → `(8, 224, 12, 64)`
5. `ttnn.to_layout(TILE_LAYOUT)`: Layout conversion (no shape change)
6. `ttnn.permute(0, 2, 1, 3)`: `(8, 224, 12, 64)` → `(8, 12, 224, 64)`

**Final Query Shape:**
- Transformers: `(8, 12, 224, 64)` (after view/transpose)
- TT-NN: `(8, 12, 224, 64)`
- **SAME SHAPE** ✅

**Differences:**
- ❌ TT-NN has **2 extra layout conversion operations** per Q/K/V
- ✅ Final tensor shapes are identical

---

### 2. Key Projection

#### Transformers ViTAttention:
```python
key = self.key(hidden_states)  # Linear layer
# Shape: (8, 224, 768) -> (8, 224, 768)
# Then: view -> (8, 224, 12, 64), transpose -> (8, 12, 64, 224)
```

#### TT-NN vit_attention:
```100:105:workloads/ttnn/vit/ttnn_functional_vit.py
    key = hidden_states @ parameters.attention.key.weight
    key = key + parameters.attention.key.bias
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.permute(key, (0, 2, 3, 1))
```

**Operators:**
1. `@` (matmul): `(8, 224, 768) @ (768, 768)` → `(8, 224, 768)`
2. `+` (add bias): `(8, 224, 768) + (1, 768)` → `(8, 224, 768)`
3. `ttnn.to_layout(ROW_MAJOR_LAYOUT)`: Layout conversion
4. `ttnn.reshape`: `(8, 224, 768)` → `(8, 224, 12, 64)`
5. `ttnn.to_layout(TILE_LAYOUT)`: Layout conversion
6. `ttnn.permute(0, 2, 3, 1)`: `(8, 224, 12, 64)` → `(8, 12, 64, 224)`

**Final Key Shape:**
- Transformers: `(8, 12, 64, 224)` (transposed for efficient matmul)
- TT-NN: `(8, 12, 64, 224)`
- **SAME SHAPE** ✅

**Note:** Key is transposed differently than Query/Value for efficient attention score computation.

---

### 3. Value Projection

#### Transformers ViTAttention:
```python
value = self.value(hidden_states)  # Linear layer
# Shape: (8, 224, 768) -> (8, 224, 768)
# Then: view -> (8, 224, 12, 64), transpose -> (8, 12, 224, 64)
```

#### TT-NN vit_attention:
```107:112:workloads/ttnn/vit/ttnn_functional_vit.py
    value = hidden_states @ parameters.attention.value.weight
    value = value + parameters.attention.value.bias
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = ttnn.permute(value, (0, 2, 1, 3))
```

**Final Value Shape:**
- Transformers: `(8, 12, 224, 64)`
- TT-NN: `(8, 12, 224, 64)`
- **SAME SHAPE** ✅

---

### 4. Attention Scores Computation

#### Transformers ViTAttention:
```python
attention_scores = torch.matmul(query, key)  # (8, 12, 224, 64) @ (8, 12, 64, 224) -> (8, 12, 224, 224)
attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # Scale
if attention_mask is not None:
    attention_scores = attention_scores + attention_mask  # Broadcast add
```

#### TT-NN vit_attention:
```114:117:workloads/ttnn/vit/ttnn_functional_vit.py
    attention_scores = query @ key
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
```

**Operators:**
- Transformers: `torch.matmul`, division, optional broadcast add
- TT-NN: `@` (matmul), multiply, optional broadcast add

**Shape:**
- Both: `(8, 12, 224, 224)`
- **SAME SHAPE** ✅

**Type:**
- Both: `bfloat16`
- **SAME TYPE** ✅

**Differences:**
- Transformers uses division (`/`), TT-NN uses multiplication (`* (1/sqrt(...))`)
- Mathematically equivalent, but different operator

---

### 5. Attention Probabilities

#### Transformers ViTAttention:
```python
attention_probs = nn.functional.softmax(attention_scores, dim=-1)
# Optional: attention_probs = self.dropout(attention_probs)  # Disabled in eval mode
```

#### TT-NN vit_attention:
```119:119:workloads/ttnn/vit/ttnn_functional_vit.py
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
```

**Operators:**
- Transformers: `torch.nn.functional.softmax`
- TT-NN: `ttnn.softmax`

**Shape:**
- Both: `(8, 12, 224, 224)`
- **SAME SHAPE** ✅

**Type:**
- Both: `bfloat16`
- **SAME TYPE** ✅

**Differences:**
- ❌ Transformers has optional dropout operation (disabled in eval mode)
- ✅ TT-NN has no dropout operation

---

### 6. Context Layer Computation

#### Transformers ViTAttention:
```python
context_layer = torch.matmul(attention_probs, value)  # (8, 12, 224, 224) @ (8, 12, 224, 64) -> (8, 12, 224, 64)
```

#### TT-NN vit_attention:
```121:125:workloads/ttnn/vit/ttnn_functional_vit.py
    context_layer = attention_probs @ value
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)
```

**Operators:**
- Transformers: `torch.matmul`, then `view` and `transpose`
- TT-NN: `@` (matmul), `permute`, `to_layout`, `reshape`, `to_layout`

**Shape after matmul:**
- Both: `(8, 12, 224, 64)`
- **SAME SHAPE** ✅

**Shape after reshape:**
- Both: `(8, 224, 768)`
- **SAME SHAPE** ✅

**Differences:**
- ❌ TT-NN has **2 extra layout conversion operations**
- ✅ Final shapes are identical

---

### 7. Output Projection

#### Transformers ViTAttention:
```python
# Inside ViTSelfOutput.forward()
hidden_states = self.dense(context_layer)  # Linear: (8, 224, 768) -> (8, 224, 768)
hidden_states = self.dropout(hidden_states)  # Disabled in eval mode
hidden_states = hidden_states + residual  # Residual connection (done in ViTAttention.forward)
```

#### TT-NN vit_attention:
```127:131:workloads/ttnn/vit/ttnn_functional_vit.py
    self_output = context_layer
    self_output = self_output @ parameters.attention.output.dense.weight
    self_output = self_output + parameters.attention.output.dense.bias

    return self_output
```

**Operators:**
- Transformers: `torch.nn.functional.linear`, optional `dropout`, `+` residual
- TT-NN: `@` (matmul), `+` (add bias)

**Shape:**
- Both: `(8, 224, 768)`
- **SAME SHAPE** ✅

**Type:**
- Both: `bfloat16`
- **SAME TYPE** ✅

**Differences:**
- ❌ Transformers has dropout operation (disabled in eval mode)
- ❌ Transformers includes residual connection in `ViTAttention.forward()`
- ✅ TT-NN doesn't include residual (done in `vit_layer` function)

---

## Summary Table: Operator Count

| Operation Type | Transformers | TT-NN | Difference |
|---------------|--------------|-------|------------|
| **Linear/MatMul** | 4 (Q, K, V, output) | 4 | ✅ Same |
| **Bias Addition** | 4 | 4 | ✅ Same |
| **Reshape/View** | 4 (Q, K, V reshape, context reshape) | 4 | ✅ Same |
| **Transpose/Permute** | 4 (Q, K, V, context) | 4 | ✅ Same |
| **MatMul (attention)** | 2 (Q@K, attn@V) | 2 | ✅ Same |
| **Scale** | 1 (division) | 1 (multiply) | ⚠️ Different op, same result |
| **Softmax** | 1 | 1 | ✅ Same |
| **Layout Conversion** | 0 | **8** | ❌ TT-NN has 8 extra |
| **Dropout** | 1 (disabled in eval) | 0 | ⚠️ Present but disabled |
| **Residual Add** | 1 (in ViTAttention.forward) | 0 (in vit_layer) | ⚠️ Different location |

**Total Core Operations:**
- Transformers: ~17 operations
- TT-NN: ~25 operations (8 extra layout conversions)

---

## Tensor Shapes at Each Stage

| Stage | Transformers Shape | TT-NN Shape | Match? |
|-------|-------------------|------------|--------|
| **Input** | `(8, 224, 768)` | `(8, 224, 768)` | ✅ |
| **After Q projection** | `(8, 224, 768)` | `(8, 224, 768)` | ✅ |
| **After Q reshape/permute** | `(8, 12, 224, 64)` | `(8, 12, 224, 64)` | ✅ |
| **After K projection** | `(8, 224, 768)` | `(8, 224, 768)` | ✅ |
| **After K reshape/permute** | `(8, 12, 64, 224)` | `(8, 12, 64, 224)` | ✅ |
| **After V projection** | `(8, 224, 768)` | `(8, 224, 768)` | ✅ |
| **After V reshape/permute** | `(8, 12, 224, 64)` | `(8, 12, 224, 64)` | ✅ |
| **Attention scores** | `(8, 12, 224, 224)` | `(8, 12, 224, 224)` | ✅ |
| **Attention probs** | `(8, 12, 224, 224)` | `(8, 12, 224, 224)` | ✅ |
| **Context layer** | `(8, 12, 224, 64)` | `(8, 12, 224, 64)` | ✅ |
| **After reshape** | `(8, 224, 768)` | `(8, 224, 768)` | ✅ |
| **After output projection** | `(8, 224, 768)` | `(8, 224, 768)` | ✅ |

**All tensor shapes match at equivalent computation points!** ✅

---

## Tensor Types

| Stage | Transformers Type | TT-NN Type | Match? |
|-------|------------------|------------|--------|
| **All tensors** | `torch.bfloat16` | `ttnn.DataType.BFLOAT16` | ✅ |

**All tensor types match!** ✅

---

## Key Findings

### ✅ What's the SAME:
1. **Tensor shapes** at all equivalent computation points
2. **Tensor types** (both use bfloat16)
3. **Mathematical operations** (same computation)
4. **Final output shape and type**

### ❌ What's DIFFERENT:
1. **Operator sequence**: TT-NN has 8 additional `ttnn.to_layout()` operations
2. **Scale operation**: Transformers uses division (`/`), TT-NN uses multiplication (`*`)
3. **Dropout**: Transformers has dropout operations (disabled in eval), TT-NN doesn't
4. **Residual connection**: Transformers includes it in `ViTAttention.forward()`, TT-NN does it in `vit_layer()`
5. **Framework operations**: `torch.*` vs `ttnn.*` (different backends)

### ⚠️ Important Notes:
1. **Layout conversions are hardware optimizations**: They don't change the mathematical result, only memory layout
2. **Dropout is disabled in eval mode**: So it doesn't affect the computation
3. **Residual connection location**: Different but equivalent - both add the residual eventually
4. **Scale operation**: Division vs multiplication are mathematically equivalent

---

## Conclusion

**Will they result in the same sequence of operators?**
- **NO** - TT-NN has additional layout conversion operations and different operator types

**Will they result in the same tensor shapes?**
- **YES** - All tensor shapes match at equivalent computation points

**Will they result in the same tensor types?**
- **YES** - Both use bfloat16 throughout

**Will they produce the same numerical results?**
- **YES** - The test verifies PCC > 0.9999, confirming mathematical equivalence despite different operator sequences

The differences in operator sequence are due to hardware-specific optimizations (layout management) and framework differences, but the core computation and tensor shapes/types are equivalent.

