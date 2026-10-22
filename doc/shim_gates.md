# TTNN shim correlation gates

Reference for the eight opt-in gates in `ttsim/utils/shim_gates.py`.

Each gate turns on one piece of tt-metal behaviour that the TTNN shim does not model
by default. They exist because each one changes the **op sequence or the LUT key** of
every workload that reaches the affected code path, and polaris builds all workload
graphs in one process — so a change that fixes one workload's correlation can silently
move another's.

**All eight default to OFF.** That is the conservative default: an un-opted-in workload
keeps exactly the behaviour it had before the gate existed.

This page is the index. The *semantics* — the two-layer resolution order, why
`Device.__init__` deliberately does not materialise these, and why you must never copy
a `_GATE` value into another module — live in the `ttsim/utils/shim_gates.py` module
docstring, which is the normative source.

---

## The eight gates

| env var | setter / accessor | `Device` attribute |
|---|---|---|
| `TTSIM_PROPAGATE_MEMORY_CONFIG` | `set_memory_config_propagation` / `memory_config_propagation_enabled` | `propagate_memory_config` |
| `TTSIM_TILIZE_BEFORE_MATMUL` | `set_tilize_before_matmul` / `tilize_before_matmul_enabled` | `tilize_before_matmul` |
| `TTSIM_CONV_MATH_FIDELITY_IN_KEY` | `set_conv_math_fidelity_in_key` / `conv_math_fidelity_in_key_enabled` | `conv_math_fidelity_in_key` |
| `TTSIM_PROPAGATE_BCAST_HW_SHAPE` | `set_bcast_hw_shape_propagation` / `bcast_hw_shape_propagation_enabled` | `propagate_bcast_hw_shape` |
| `TTSIM_HALO_OUTPUT_DTYPE_PROMOTION` | `set_halo_output_dtype_promotion` / `halo_output_dtype_promotion_enabled` | `halo_output_dtype_promotion` |
| `TTSIM_CONV_INPUT_RESHARD` | `set_conv_input_reshard` / `conv_input_reshard_enabled` | `conv_input_reshard` |
| `TTSIM_TO_MEMORY_CONFIG_RESHARD` | `set_to_memory_config_reshard` / `to_memory_config_reshard_enabled` | `to_memory_config_reshard` |
| `TTSIM_MOVE_ON_REALLOCATE_HALO_OUTPUT` | `set_move_on_reallocate_halo_output` / `move_on_reallocate_halo_output_enabled` | `move_on_reallocate_halo_output` |

Each env var is read once at module import and accepts `'1'` to enable; anything else
(including unset) is OFF.

### What each one does

- **`memory_config_propagation`** — give conv/matmul outputs the memory config implied
  by their `Conv2dConfig.shard_layout`, so model code can meaningfully ask
  `out.memory_config()`. Without it that call returns `None` and comparisons such as
  ResNet-50's bottleneck `if ds_out.memory_config() != out.memory_config()` cannot be
  evaluated.
- **`tilize_before_matmul`** — emit the ROW_MAJOR → TILE conversion ahead of a conv
  that lowers to a 1×1 matmul.
- **`conv_math_fidelity_in_key`** — carry `compute_config`'s math fidelity into the
  conv / 1×1-matmul op attrs, so it reaches the LUT key.
- **`bcast_hw_shape_propagation`** — propagate NHWC-flat `hw_shape` across elementwise
  binaries. Reached by all three front ends (`bidir_bcast` is the registered sinf for
  19 `ai.onnx` optypes), which is why it is gated rather than unconditional.
- **`halo_output_dtype_promotion`** — model `HaloDeviceOperation`'s output-dtype
  promotion.
- **`conv_input_reshard`** — emit the conv **input** Reshard (+Move) that
  `reshard_if_not_optimal` forces on hardware.
- **`to_memory_config_reshard`** — let `to_memory_config` take tt-metal's *primary*
  sharded→sharded path (`ttnn::reshard`) instead of the `sharded_to_interleaved →
  interleaved_to_sharded` pair that `to_memory_config_op.cpp:279-320` labels a
  *"Workaround"*.
- **`move_on_reallocate_halo_output`** — gate the post-Halo `Move` on
  `reallocate_halo_output` rather than on `deallocate_activation`.

---

## Resolution order

Every `<gate>_enabled(device)` resolves in two layers:

```
attribute absent on the Device  ->  the process-wide default in shim_gates.py,
                                    seeded from TTSIM_* at import
attribute present               ->  that device's explicit override, set only by
                                    set_<gate>(x, device)
```

So `set_<gate>(True, device)` scopes the change to one `Device` and cannot leak into
the next workload built in the same process. `set_<gate>(True)` with no device moves
the process-wide default and affects every workload that has no explicit override.

Import setters and accessors **from `ttsim.utils.shim_gates`**, not through the front
end — that keeps a matched pair arriving from one module, and `ttsim/front/ttnn/op.py`
imports only the accessors it calls itself.

---

## Who opts in today

Only the two ResNet-50 infras, and both turn on all eight, device-scoped:

- `workloads/ttnn/resnet50/bh/resnet50_test_infra_polaris_bh.py:95-134`
- `workloads/ttnn/resnet50/wh/resnet50_test_infra_polaris_wh.py:92-131`

No other in-tree workload sets any gate, so vgg_unet / vit / unet / tt_transformers
all run with all eight OFF.

---

## Two gates with known cross-workload cost

These are the reason the defaults are OFF rather than ON, and the reason a
six-workload sweep is required before changing a default.

**`to_memory_config_reshard`.** Taking the primary path needs *both* shard specs to
evaluate `use_reshard_workaround` (see `_needs_reshard_workaround`), and a workload
that does not propagate memory configs does not have them. With this on for
`vgg_unet`, the HEIGHT→BLOCK transition at its s2 boundary emitted one `Reshard` where
the p100a capture (rows 20-21) round-trips through DRAM as STI+ITS — **two hits became
one miss**.

**`move_on_reallocate_halo_output`.** Changing which flag gates the post-Halo `Move`
moved `vgg_bh_p100a` (bs=1) from **102/102 to 98/102** before it was gated. With the
gate in place vgg is byte-identical to main again.

---

## Running a one-off experiment

The env vars are the cheapest way to test a gate across every workload without editing
a workload file:

```bash
TTSIM_TO_MEMORY_CONFIG_RESHARD=1 python polaris.py ... -s myrun
```

Because the value is read at import, this moves the *process-wide default* — i.e. it
applies to every workload in the run except those with an explicit device override
(today, the two ResNet-50 infras, which set all eight explicitly and are therefore
unaffected by the env vars).

To verify nothing regressed, compare LUT hit counts per workload against the previous
run's `STATS/*-opstats.csv`; `lut_key` is emitted even on a miss, so a diff of the
`lut_key` / `lut_hit_source` columns localises any change to specific ops.

---

## Adding a gate

1. Add the `_GATE = os.getenv('TTSIM_...', '0') == '1'` line plus the `set_` / `_enabled`
   pair to `ttsim/utils/shim_gates.py`, keeping the file a leaf (it imports only `os`,
   which is what broke the `ttsim.ops → ttsim.front` cycle).
2. Add the row to the table above.
3. `tests/test_ttnn/test_shim_gate_defaults.py` parametrises over `GATES` and will
   cover the new gate automatically — default-off, device scoping, setter-returns-previous,
   and the process-wide/override precedence.
4. Run the six-workload sweep before considering any change to a default.
