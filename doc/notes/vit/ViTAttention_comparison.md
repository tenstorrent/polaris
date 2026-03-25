# Comparison: transformers.models.vit.modeling_vit.ViTAttention vs workloads.ttnn.vit.ttnn_functional_vit.vit_attention

## Overview

This document compares the Hugging Face Transformers `ViTAttention` class with the Tenstorrent TT-NN `vit_attention` function implementation.

## Implementation Structure

### transformers.models.vit.modeling_vit.ViTAttention

**Type:** Class-based (PyTorch `nn.Module`)

**Structure:**
- Object-oriented design
- Contains two sub-modules:
  - `self.attention`: `ViTSelfAttention` - computes Q, K, V and attention scores
  - `self.output`: `ViTSelfOutput` - linear projection + dropout
- Inherits from `nn.Module` with `forward()` method
- Supports attention head pruning
- Uses PyTorch operations (torch.matmul, torch.softmax, etc.)

**Typical Forward Pass:**
```python
def forward(self, hidden_states, head_mask=None, output_attentions=False):
    self_outputs = self.attention(hidden_states, head_mask, output_attentions)
    attention_output = self.output(self_outputs[0], hidden_states)
    # Returns (attention_output, attention_probs if output_attentions)
```

### workloads.ttnn.vit.ttnn_functional_vit.vit_attention

**Type:** Functional implementation

**Structure:**
- Pure function (no class)
- Takes `config`, `hidden_states`, `attention_mask`, and `parameters` as arguments
- Uses TT-NN operations (ttnn.matmul, ttnn.softmax, etc.)
- Optimized for Tenstorrent hardware with explicit layout management

**Function Signature:**
```82:131:workloads/ttnn/vit/ttnn_functional_vit.py
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

    key = hidden_states @ parameters.attention.key.weight
    key = key + parameters.attention.key.bias
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters.attention.value.weight
    value = value + parameters.attention.value.bias
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = ttnn.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = context_layer
    self_output = self_output @ parameters.attention.output.dense.weight
    self_output = self_output + parameters.attention.output.dense.bias

    return self_output
```

## Key Differences

### 1. **Architecture Paradigm**

| Aspect | ViTAttention (Transformers) | vit_attention (TT-NN) |
|--------|----------------------------|----------------------|
| **Type** | Class (nn.Module) | Function |
| **State** | Maintains internal state (weights) | Stateless (weights passed as parameters) |
| **Inheritance** | Inherits from nn.Module | No inheritance |

### 2. **Operation Framework**

| Aspect | ViTAttention (Transformers) | vit_attention (TT-NN) |
|--------|----------------------------|----------------------|
| **Backend** | PyTorch (torch.*) | TT-NN (ttnn.*) |
| **Layout Management** | Automatic (PyTorch handles) | Explicit (ttnn.to_layout calls) |
| **Tensor Operations** | torch.matmul, torch.softmax | ttnn.matmul, ttnn.softmax |

### 3. **Layout and Memory Management**

**TT-NN Implementation:**
- **Explicit layout conversions**: Multiple `ttnn.to_layout()` calls to switch between `ROW_MAJOR_LAYOUT` and `TILE_LAYOUT`
- **Hardware optimization**: TILE_LAYOUT is optimized for Tenstorrent hardware
- **Memory efficiency**: Explicit control over tensor layouts for better memory access patterns

**Transformers Implementation:**
- **Automatic layout**: PyTorch handles memory layout automatically
- **General purpose**: Works on CPU/GPU without explicit layout management

### 4. **Attention Computation Flow**

Both implementations follow the same mathematical flow:

1. **Query, Key, Value Projections**
   - Both compute: `Q = hidden_states @ W_q`, `K = hidden_states @ W_k`, `V = hidden_states @ W_v`
   - Both add bias terms

2. **Reshaping and Permutation**
   - **Transformers**: Typically uses `view()` and `transpose()` operations
   - **TT-NN**: Uses explicit `reshape()` and `permute()` with layout conversions
   - **Key difference**: TT-NN permutes key as `(0, 2, 3, 1)` for efficient matrix multiplication

3. **Attention Scores**
   - Both compute: `attention_scores = Q @ K`
   - Both scale by `1 / sqrt(head_size)`
   - Both apply attention mask if provided

4. **Attention Probabilities**
   - Both apply softmax: `attention_probs = softmax(attention_scores)`

5. **Context Layer**
   - Both compute: `context_layer = attention_probs @ V`
   - Both reshape back to `(batch_size, sequence_size, hidden_size)`

6. **Output Projection**
   - Both apply final linear layer: `output = context_layer @ W_o + bias_o`

### 5. **Parameter Access**

| Aspect | ViTAttention (Transformers) | vit_attention |
|--------|----------------------------|----------------|
| **Parameter Storage** | Internal (self.attention.query, etc.) | External (parameters.attention.query) |
| **Access Pattern** | `self.attention.query.weight` | `parameters.attention.query.weight` |
| **Flexibility** | Less flexible (bound to class) | More flexible (can swap parameters) |

### 6. **Return Values**

| Implementation | Return Value |
|---------------|--------------|
| **ViTAttention** | `(attention_output, attention_probs)` if `output_attentions=True`, else `(attention_output,)` |
| **vit_attention** | `self_output` (single tensor) |

### 7. **Additional Features**

**ViTAttention (Transformers):**
- ✅ Attention head pruning support
- ✅ Optional attention probabilities output
- ✅ Head mask support for masking specific attention heads
- ✅ Dropout in output layer
- ✅ Integration with transformers pipeline

**vit_attention (TT-NN):**
- ✅ Hardware-optimized layout management
- ✅ Explicit tensor shape management
- ✅ Optimized for Tenstorrent accelerators
- ✅ Functional programming style

## Mathematical Equivalence

Both implementations compute the same mathematical operation:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- `Q`, `K`, `V` are query, key, value matrices
- `d_k` is the head dimension (`head_size`)
- The result is projected through an output linear layer

## Test Verification

The test file (`tests/test_ttnn/test_ttnn_orig.py`) verifies equivalence:

```python
def test_vit_attention(device, model_name, batch_size, sequence_size):
    # Create transformers model
    model = transformers.models.vit.modeling_vit.ViTAttention(config).eval()
    torch_output, *_ = model(torch_hidden_states, torch_attention_mask)
    
    # Run TT-NN implementation
    output = ttnn_functional_vit.vit_attention(
        config, hidden_states, attention_mask=attention_mask, parameters=parameters
    )
    
    # Verify with high precision (PCC > 0.9999)
    assert_with_pcc(torch_output, output, 0.9999)
```

This confirms that both implementations produce equivalent results (within numerical precision).

## Use Cases

### When to use ViTAttention (Transformers):
- General-purpose PyTorch workflows
- Research and experimentation
- Integration with Hugging Face ecosystem
- CPU/GPU inference and training
- Need for attention head pruning or attention visualization

### When to use vit_attention (TT-NN):
- Tenstorrent hardware acceleration
- Production inference on Tenstorrent devices
- Maximum performance optimization
- Functional programming style preference
- Fine-grained control over tensor layouts

## Summary

Both implementations are **mathematically equivalent** but differ in:

1. **Design philosophy**: Class-based (Transformers) vs Functional (TT-NN)
2. **Hardware optimization**: General-purpose (Transformers) vs Hardware-specific (TT-NN)
3. **Layout management**: Automatic (Transformers) vs Explicit (TT-NN)
4. **Framework**: PyTorch (Transformers) vs TT-NN (Tenstorrent)

The TT-NN implementation is a hardware-optimized, functional version of the same attention mechanism, designed specifically for efficient execution on Tenstorrent accelerators while maintaining mathematical equivalence with the reference implementation.

