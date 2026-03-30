# Shim Functions to Operators — Final Specification and Implementation Plan

## Final prompt (revised specification)

Shim functions to operators

Study the design of operators or layers in TTNN workloads.
Example workload is workloads/ttnn/vit/run_ttnn_functional_vit.py function vit_attention.
The TTNN front end files are in ttsim/front/ttnn.
The operators are defined in ttsim/ops.
After studying this, plan to convert the tracking mode logic of the tilize function in ttnn_shim as a new operator function, without updating the original function.
This new function is only for new/alternative code paths that want Tilize as a first-class SimOp on the device, with tilize() always unchanged and still using the tracker
This new front end function should create and use ttsim.front.ttnn.tensor.Tensor type used by op.py
This new function should be in a new file ttsim_layout.py under ttsim/ops/desc.
The new tilize operator API should be in ttsim/front/ttnn/ttnn_shim.py next to tilize but as a separate tracking-only function.
The new op's perf_stats should track counters in a format similar to existing sinf functions in ttsim/ops
Follow the same pattern for untilize, tilize_with_padding, untilize_with_padding as well.
Use exact SimOp names Tilize, Untilize, TilizeWithValPadding, UntilizeWithValUnpadding (hardware-aligned; Python API remains ``tilize_with_val_padding`` / ``untilize_with_unpadding_op``)
Current untilize_with_unpadding(input_tensor, output_tensor_end, ...) uses output_tensor_end (end indices). For the new op, should the front-end API be untilize_with_unpadding_op(input_tensor, output_shape, ...) with output_shape = logical output shape (list/tuple), and the descriptor uses that only
TilizeWithValPadding arity Should stay 1→1 with output_padded_shape and pad_value in op.attrs,
The new APIs tilize_op etc should only be importable from ttnn_shim for now.

In case of any ambiguities, ask for clarifications, and do not proceed till all ambiguities are resolved.

---

## Output: Implementation plan

### Scope
- **New op descriptors:** `ttsim/ops/desc/ttsim_layout.py` (shape_inf_funcs + `register_layout_ops`).
- **New front-end APIs:** `ttsim/front/ttnn/ttnn_shim.py` only — `tilize_op`, `untilize_op`, `tilize_with_val_padding_op`, `untilize_with_unpadding_op`, `permute_op`. Tracking-only (create SimOp, add to device); no execution, no `_tracker`.
- **Existing shim functions:** `tilize`, `untilize`, `tilize_with_val_padding`, `untilize_with_unpadding` unchanged (still use tracker and execution).
- **Exports:** New APIs are only importable from `ttnn_shim` (do not add to `ttnn` `__init__.py`).

### 1. `ttsim/ops/desc/ttsim_layout.py` (new file)
- **Imports:** `register_ops` from `.registry`; tile constants (TILE_HEIGHT, TILE_WIDTH = 32) — define locally or import from a small shared helper.
- **tilize_sinf(iTList, oTList, op, **kwargs):** 1 input (ROW_MAJOR). Output same logical shape; padded = last two dims rounded up to tile. Set oTList[0].shape, dtype; op.perf_stats = {inElems, outElems, inBytes, outBytes, instrs} using op.attrs.get('element_size', 2).
- **untilize_sinf:** 1 input (TILE). Output same logical shape; padded = logical. Same perf_stats style.
- **tilize_with_val_padding_sinf:** 1 input. Output padded = op.attrs['output_padded_shape']. perf_stats from shapes.
- **untilize_with_val_unpadding_sinf:** 1 input. Output logical = op.attrs['output_shape']. perf_stats from shapes.
- **register_layout_ops():** register_ops('layout', _optbl) with Tilize, Untilize, TilizeWithValPadding, UntilizeWithValUnpadding (1→1 each), shape_inf_func = the four sinf callables; table columns same as tensor.py/nn.py.

### 2. Op descriptor startup
- In `ttsim/ops/desc/__init__.py`: add `from .ttsim_layout import register_layout_ops` and call `register_layout_ops()` inside `initialize_op_desc()`.

### 3. New APIs in `ttsim/front/ttnn/ttnn_shim.py`
- Use `ttsim.front.ttnn.tensor.Tensor` only. Create output tensors with shape/layout/padded_shape/device; build SimOp; get_perf_counts; update_tensor_counts; device.add_op; return output. No execution, no _tracker.
- **tilize_op(input_tensor, use_multicore=True, element_size=2, memory_config=None):** output logical = input shape, padded = pad_to_tile_shape(logical). SimOp optype 'Tilize'.
- **untilize_op(input_tensor, use_multicore=True, use_pack_untilize=True, element_size=2, memory_config=None):** output ROW_MAJOR, same logical. SimOp optype 'Untilize'.
- **tilize_with_val_padding_op(input_tensor, output_padded_shape, pad_value, use_multicore=True, element_size=2, ...):** attrs output_padded_shape, pad_value. SimOp optype 'TilizeWithValPadding'.
- **untilize_with_unpadding_op(input_tensor, output_shape, use_multicore=True, use_pack_untilize=True, element_size=2, ...):** attrs output_shape (list/tuple). SimOp optype 'UntilizeWithValUnpadding'.
- **permute_op(input_tensor, dims, memory_config=None):** attrs `perm` (list of axis indices). SimOp optype **Permute** (registered in `ttsim/ops/desc/tensor.py`; same shape/perf inference as **Transpose**). Use this when traces should show `Permute` instead of the front-end `ttnn.permute` path, which still lowers to **Transpose** in `op.py`.

### 4. perf_stats format
- Match existing sinf: inElems, outElems, inBytes, outBytes, instrs (dict). Optional num_tiles etc. OK.

### 5. Naming and arity summary
| Op                     | Arity | Front-end API                                              | Main attrs / args |
|------------------------|-------|------------------------------------------------------------|-------------------|
| Tilize                 | 1→1   | tilize_op(input_tensor, ...)                               | use_multicore, element_size |
| Untilize               | 1→1   | untilize_op(input_tensor, ...)                             | use_multicore, use_pack_untilize, element_size |
| TilizeWithValPadding   | 1→1   | tilize_with_val_padding_op(input_tensor, output_padded_shape, pad_value, ...) | output_padded_shape, pad_value, use_multicore, element_size |
| UntilizeWithValUnpadding  | 1→1   | untilize_with_unpadding_op(input_tensor, output_shape, ...) | output_shape, use_multicore, use_pack_untilize, element_size |
| Permute                | 1→1   | permute_op(input_tensor, dims, ...)                         | perm (axis order) |

### 6. What stays unchanged
- Existing tilize, untilize, tilize_with_val_padding, untilize_with_unpadding in ttnn_shim.py: no changes.
- No new exports in ttsim/front/ttnn/__init__.py.
