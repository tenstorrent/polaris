#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for tensor shape parsing, normalization, and
comparison, as well as layout / dtype / memory attribute normalization.

Memory strings are compared via :func:`normalize_memory_tag` so Polaris
``MemoryConfig`` repr and profiler short forms (e.g. ``L1_BLOCK_SHARDED``)
match when semantically identical.

All tools that compare or construct tensor shapes should import from
here rather than maintaining their own parsing or normalization logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ===================================================================
# Layout / Dtype / Memory normalization
# ===================================================================

LAYOUT_NORMALIZATION: Dict[str, str] = {
    "tile_layout": "TILE",
    "tile": "TILE",
    "row_major_layout": "ROW_MAJOR",
    "row_major": "ROW_MAJOR",
}

DTYPE_NORMALIZATION: Dict[str, str] = {
    "float16": "BFLOAT16",
    "bfloat16": "BFLOAT16",
    "float32": "FLOAT32",
    "bfloat8_b": "BFLOAT8_B",
    "int64": "INT64",
    "int32": "INT32",
}


def normalize_attr(value: Optional[str], table: Dict[str, str]) -> Optional[str]:
    """Normalize a single attribute value using a lookup table.

    Returns ``None`` when *value* is ``None``; otherwise looks up the
    lowercased value in *table* and falls back to ``value.upper()``.
    """
    if value is None:
        return None
    return table.get(value.lower().strip(), value.upper().strip())


def normalize_layout(value: Optional[str]) -> Optional[str]:
    """Normalize a layout string (e.g. ``tile_layout`` → ``TILE``)."""
    return normalize_attr(value, LAYOUT_NORMALIZATION)


def normalize_dtype(value: Optional[str]) -> Optional[str]:
    """Normalize a dtype string (e.g. ``bfloat16`` → ``BFLOAT16``)."""
    return normalize_attr(value, DTYPE_NORMALIZATION)


# TensorMemoryLayout.name → suffix segment (matches profiler / ``tensor_memory_str``).
_MEMORY_LAYOUT_SUFFIX: Dict[str, str] = {
    "INTERLEAVED": "INTERLEAVED",
    "HEIGHT_SHARDED": "HEIGHT_SHARDED",
    "BLOCK_SHARDED": "BLOCK_SHARDED",
    "WIDTH_SHARDED": "WIDTH_SHARDED",
}


def canonical_memory_tag_from_enums(buffer_name: str, layout_name: str) -> str:
    """Build tag like ``L1_BLOCK_SHARDED`` or ``DRAM_INTERLEAVED`` from enum names."""
    buf = buffer_name.strip().upper()
    lay = layout_name.strip().upper()
    suffix = _MEMORY_LAYOUT_SUFFIX.get(lay, "INTERLEAVED")
    return f"{buf}_{suffix}"


def normalize_memory_tag(value: Optional[str]) -> Optional[str]:
    """Map Polaris ``MemoryConfig`` repr or profiler short strings to one canonical form.

    Polaris CSV often stores ``str(MemoryConfig)`` (i.e. ``__repr__``). Profiler
    CSV uses values like ``L1_BLOCK_SHARDED`` (sometimes prefixed with
    ``DEV_1_`` before load). After normalization, semantically identical configs
    compare equal.

    Returns ``None`` for missing/empty *value*. Unknown non-``MemoryConfig``
    strings are uppercased with spaces → underscores (legacy behavior).
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    s = re.sub(r"^DEV_\d+_", "", s, flags=re.IGNORECASE)
    if s.startswith("MemoryConfig("):
        ml = re.search(r"TensorMemoryLayout\.(\w+)", s)
        bt = re.search(r"BufferType\.(\w+)", s)
        if ml and bt:
            return canonical_memory_tag_from_enums(bt.group(1), ml.group(1))
    return s.upper().replace(" ", "_")


# ===================================================================
# Shape parsing
# ===================================================================

def parse_shape_string(shape_str: str) -> List[int]:
    """Parse a shape string like ``'1x224x224'`` into a list of ints.

    Returns an empty list when *shape_str* is empty or unparseable.
    """
    if not shape_str or shape_str.strip() == "":
        return []
    try:
        parts = shape_str.split("x")
        return [int(p.strip()) for p in parts if p.strip()]
    except ValueError:
        return []


# ===================================================================
# Shape normalization
# ===================================================================

def normalize_shape(
    shape_list: List[int],
    strip_leading_ones: bool = False,
    strip_singleton_dims: bool = False,
) -> List[int]:
    """Normalize shape by handling 1-valued dimensions.

    Args:
        shape_list: List of dimension sizes.
        strip_leading_ones: Strip *all* leading 1s.
        strip_singleton_dims: Strip *every* 1-valued dim regardless of
            position (e.g. HW ``seq_groups=1`` convention).

    Examples (strip_singleton_dims=True)::

        [8, 1, 197, 768] -> [8, 197, 768]
        [1, 1, 1]        -> [1]

    Examples (strip_leading_ones=True)::

        [1, 1, 1024, 768] -> [1024, 768]

    Examples (default — collapse multiple leading 1s to one)::

        [1, 1, 1, 224, 224] -> [1, 224, 224]
    """
    if not shape_list:
        return []

    if strip_singleton_dims:
        result = [d for d in shape_list if d != 1]
        return result if result else [1]

    leading_ones = 0
    for dim in shape_list:
        if dim == 1:
            leading_ones += 1
        else:
            break

    if leading_ones == len(shape_list):
        return [1]

    if leading_ones > 0:
        if strip_leading_ones:
            return shape_list[leading_ones:]
        elif leading_ones > 1:
            return [1] + shape_list[leading_ones:]

    return shape_list


# ===================================================================
# Rank-4 WZYX promotion (for LUT key construction)
# ===================================================================

def coerce_shape_to_list(shape_like: Any) -> List[int]:
    """Normalize a ``Shape`` object, sequence, or similar to ``list[int]``.

    Handles objects with ``.view()`` (ttsim ``Shape``), ``._shape``, or
    plain iterables.
    """
    if shape_like is None:
        return []
    if hasattr(shape_like, "as_list") and callable(shape_like.as_list):
        return [int(x) for x in shape_like.as_list()]
    if hasattr(shape_like, "view") and callable(getattr(shape_like, "view", None)):
        return [int(x) for x in shape_like.view()]
    if hasattr(shape_like, "_shape"):
        return [int(x) for x in shape_like._shape]
    return [int(x) for x in shape_like]


def promote_to_rank4(dims: Sequence[int]) -> Tuple[int, int, int, int]:
    """Promote an arbitrary-rank shape to rank-4 WZYX by left-padding
    with 1s.

    Uses ``ttsim.ops.tensor.Shape.to_rank(4)`` when available, falling
    back to manual padding.
    """
    try:
        from ttsim.ops.tensor import Shape
        sh = Shape(list(dims))
        r4 = sh.to_rank(4)
        v = r4.as_list() if hasattr(r4, "as_list") else r4.view()
        return (int(v[0]), int(v[1]), int(v[2]), int(v[3]))
    except (ImportError, Exception):
        d = list(dims)
        while len(d) < 4:
            d.insert(0, 1)
        if len(d) > 4:
            extra = 1
            for x in d[: len(d) - 3]:
                extra *= x
            d = [extra] + d[len(d) - 3 :]
        return (int(d[0]), int(d[1]), int(d[2]), int(d[3]))


def reshape_input0_wzyx(w: int, z: int, y: int, x: int) -> Tuple[int, int, int, int]:
    """Master LUT convention for ``reshape`` input_0 logical dims:
    ``(1, 1, w*z*y, x)``.
    """
    return (1, 1, int(w) * int(z) * int(y), int(x))


_TILE_SIZE = 32


def createqkvheads_input0_wzyx(w: int, z: int, y: int, x: int) -> Tuple[int, int, int, int]:
    """Master LUT convention for ``createqkvheads`` input_0 logical dims.

    tt-metal records the input as ``[B, 1, tile_pad(S), H]`` where ``B``
    occupies ``W`` and ``S`` is rounded up to a tile boundary.  The shim
    promotes a 3-D ``[B, S, H]`` input to ``[1, B, S, H]``, so we swap
    ``W`` and ``Z`` and tile-pad ``Y``.
    """
    batch = int(z) if int(w) == 1 else int(w)
    seq_padded = int(y)
    rem = seq_padded % _TILE_SIZE
    if rem != 0:
        seq_padded += _TILE_SIZE - rem
    return (batch, 1, seq_padded, int(x))


# ===================================================================
# Shape comparison
# ===================================================================

def compare_tensor_shapes(
    polaris_shapes: List[str],
    profiler_shapes: List[str],
    strip_leading_ones: bool = False,
    optype: Optional[str] = None,
    strip_singleton_dims: bool = False,
) -> Tuple[bool, str]:
    """Compare two lists of tensor shape strings.

    Returns ``(match, details)`` — ``True`` when all shapes agree after
    normalization.
    """
    from tools.profiling.op_canonical import normalize_polaris_optype

    polaris_normalized = [
        normalize_shape(parse_shape_string(s), strip_leading_ones,
                        strip_singleton_dims=strip_singleton_dims)
        for s in polaris_shapes
    ]
    profiler_normalized = [
        normalize_shape(parse_shape_string(s), strip_leading_ones,
                        strip_singleton_dims=strip_singleton_dims)
        for s in profiler_shapes
    ]

    if len(polaris_normalized) != len(profiler_normalized):
        if optype and normalize_polaris_optype(optype) == "reshape":
            pass  # handled in output comparison
        else:
            return False, f"count mismatch: {len(polaris_normalized)} vs {len(profiler_normalized)}"

    for i, (p_shape, f_shape) in enumerate(zip(polaris_normalized, profiler_normalized)):
        if p_shape != f_shape:
            p_str = "x".join(map(str, p_shape)) if p_shape else "empty"
            f_str = "x".join(map(str, f_shape)) if f_shape else "empty"
            return False, f"tensor {i}: {p_str} vs {f_str}"

    return True, ""


def validate_binary_compatibility(
    polaris_inputs: List[str],
    profiler_inputs: List[str],
    strip_leading_ones: bool = False,
    strip_singleton_dims: bool = False,
) -> Tuple[bool, str]:
    """Special validation for binary operations (add/mul/sub) where
    input representations may differ.

    Handles overlapping-prefix comparison and scalar/untracked operands.
    """
    p_nonempty = [s for s in polaris_inputs if s.strip()]
    f_nonempty = [s for s in profiler_inputs if s.strip()]

    if len(p_nonempty) == len(f_nonempty):
        return compare_tensor_shapes(
            p_nonempty, f_nonempty, strip_leading_ones,
            strip_singleton_dims=strip_singleton_dims,
        )

    overlap = min(len(p_nonempty), len(f_nonempty))
    if overlap > 0:
        match, details = compare_tensor_shapes(
            p_nonempty[:overlap], f_nonempty[:overlap],
            strip_leading_ones, strip_singleton_dims=strip_singleton_dims,
        )
        if match:
            return True, (
                f"binary compatible: matched {overlap} of "
                f"{len(polaris_inputs)} polaris / {len(profiler_inputs)} profiler inputs"
            )
        return False, details

    if overlap == 0 and (len(p_nonempty) == 0 or len(f_nonempty) == 0):
        return True, "binary compatible: one side has only scalar/untracked operands"

    return False, (
        f"binary input count mismatch: "
        f"{len(polaris_inputs)} polaris vs {len(profiler_inputs)} profiler"
    )


def validate_reshape_compatibility(
    polaris_inputs: List[str],
    polaris_outputs: List[str],
    profiler_outputs: List[str],
    strip_leading_ones: bool = False,
    strip_singleton_dims: bool = False,
) -> Tuple[bool, str]:
    """Special validation for reshape operations with different
    representations between Polaris and the profiler.
    """
    if len(polaris_inputs) != 2:
        return False, "polaris doesn't have 2 inputs"

    pol_out_parsed = parse_shape_string(polaris_outputs[0]) if polaris_outputs else []
    prof_out_parsed = parse_shape_string(profiler_outputs[0]) if profiler_outputs else []
    pol_input2_parsed = parse_shape_string(polaris_inputs[1]) if len(polaris_inputs) > 1 else []

    prof_out_normalized = normalize_shape(
        prof_out_parsed, strip_leading_ones,
        strip_singleton_dims=strip_singleton_dims,
    )

    if len(pol_input2_parsed) != 1:
        return False, f"polaris second input not 1-D: {pol_input2_parsed}"

    if len(prof_out_normalized) != 2:
        return False, f"profiler output not 2-D after normalization: {prof_out_normalized}"

    if len(pol_out_parsed) < 3:
        return False, f"polaris output has < 3 dims: {pol_out_parsed}"

    pol_first_three_product = pol_out_parsed[0] * pol_out_parsed[1] * pol_out_parsed[2]
    prof_first_dim = prof_out_normalized[0]

    if pol_first_three_product == prof_first_dim:
        return True, (
            f"reshape compatible: pol {pol_out_parsed[:3]} "
            f"product={pol_first_three_product} matches prof first dim={prof_first_dim}"
        )
    return False, (
        f"product mismatch: pol {pol_out_parsed[:3]} "
        f"product={pol_first_three_product} vs prof first dim={prof_first_dim}"
    )


def compare_tensor_attributes(
    p_layer: Dict[str, Any],
    f_layer: Dict[str, Any],
    direction: str = "input",
) -> Tuple[bool, str]:
    """Compare dtype, layout, and memory for input or output tensors.

    Returns ``(match, details)``.  Attributes that are ``None``/empty
    on either side are silently skipped.
    """
    p_dtypes = p_layer.get(f"{direction}_dtypes", [])
    f_dtypes = f_layer.get(f"{direction}_dtypes", [])
    p_layouts = p_layer.get(f"{direction}_layouts", [])
    f_layouts = f_layer.get(f"{direction}_layouts", [])
    p_mems = p_layer.get(f"{direction}_memories", [])
    f_mems = f_layer.get(f"{direction}_memories", [])

    mismatches: List[str] = []
    n = min(len(p_dtypes), len(f_dtypes))
    for i in range(n):
        pd = normalize_dtype(p_dtypes[i])
        fd = normalize_dtype(f_dtypes[i])
        if pd and fd and pd != fd:
            mismatches.append(f"tensor {i} dtype: {pd} vs {fd}")

    n = min(len(p_layouts), len(f_layouts))
    for i in range(n):
        pl = normalize_layout(p_layouts[i])
        fl = normalize_layout(f_layouts[i])
        if pl and fl and pl != fl:
            mismatches.append(f"tensor {i} layout: {pl} vs {fl}")

    n = min(len(p_mems), len(f_mems))
    for i in range(n):
        pm = p_mems[i]
        fm = f_mems[i]
        npm = normalize_memory_tag(pm)
        nfm = normalize_memory_tag(fm)
        if npm and nfm and npm != nfm:
            mismatches.append(f"tensor {i} memory: {pm} vs {fm}")

    if mismatches:
        return False, "; ".join(mismatches)
    return True, ""


# ===================================================================
# Tensor attribute extraction helpers for LUT key construction
# ===================================================================

def precision_to_master_datatype(precision: Any) -> str:
    """Map an ``op.precision`` value to the master LUT datatype string."""
    if precision is None:
        return "BFLOAT16"
    u = str(precision).upper().replace(" ", "")
    if u in ("BF16",):
        return "BFLOAT16"
    if u in ("FP16",):
        return "FLOAT16"
    if u in ("FP32",):
        return "FLOAT32"
    return u


def _storage_is_numpy_float16(dt: Any) -> bool:
    """True when *dt* is ``numpy.float16``."""
    try:
        import numpy as np
        if isinstance(dt, np.dtype):
            return dt == np.dtype(np.float16)
    except ImportError:
        pass
    nm = getattr(dt, "name", None)
    return isinstance(nm, str) and nm.lower() == "float16"


def tensor_layout_str(t: Any) -> str:
    """Extract layout string from a tensor object for LUT key."""
    lay = getattr(t, "layout", None)
    if lay is None:
        return "TILE"
    name = getattr(lay, "name", str(lay))
    u = name.upper()
    if "TILE" in u:
        return "TILE"
    if "ROW" in u:
        return "ROW_MAJOR"
    return "TILE"


def _memory_config_obj_to_canonical_tag(mc: Any) -> Optional[str]:
    buf = getattr(mc, "buffer_type", None)
    mem_layout = getattr(mc, "memory_layout", None)
    if buf is None or mem_layout is None:
        return None
    buf_name = getattr(buf, "name", str(buf))
    layout_name = getattr(mem_layout, "name", str(mem_layout))
    return canonical_memory_tag_from_enums(buf_name, layout_name)


def tensor_memory_str(t: Any) -> str:
    """Extract memory config string from a tensor object for LUT key.

    Returns profiler-style strings like ``DEV_1_L1_BLOCK_SHARDED`` when
    the tensor carries a rich ``MemoryConfig`` with ``buffer_type`` and
    ``memory_layout`` attributes (matching real tt-metal).
    """
    mc = getattr(t, "memory_config", None)
    if callable(mc):
        try:
            mc = mc()
        except Exception:
            mc = None
    if mc is None:
        mc = getattr(t, "_memory_config", None)
    if mc is None:
        return "DEV_1_DRAM_INTERLEAVED"

    tag = _memory_config_obj_to_canonical_tag(mc)
    if tag is not None:
        return f"DEV_1_{tag}"

    s = str(mc).upper().replace(" ", "_")
    if "L1" in s:
        return "DEV_1_L1_INTERLEAVED"
    return "DEV_1_DRAM_INTERLEAVED"


_KNOWN_TTNN_DTYPES = frozenset({
    "BFLOAT16", "BFLOAT8_B", "BFLOAT4_B", "FLOAT32", "FLOAT16", "INT32",
})


def tensor_datatype(t: Any, op_precision: Any) -> str:
    """Extract datatype string from a tensor object for LUT key.

    Prefers the TTNN logical dtype (``_ttnn_dtype``) when present, since
    numpy storage loses information (e.g. BFLOAT8_B is stored as float32).
    Falls back to ``t.dtype`` and then *op_precision*.
    """
    ttnn_dt = getattr(t, "_ttnn_dtype", None)
    if ttnn_dt is not None:
        name = getattr(ttnn_dt, "name", str(ttnn_dt)).upper()
        if name in _KNOWN_TTNN_DTYPES:
            return name

    dt = getattr(t, "dtype", None)
    if dt is not None:
        name = getattr(dt, "name", str(dt)).upper()
        if "BFLOAT16" in name or name == "BFLOAT16":
            return "BFLOAT16"
        if "FLOAT32" in name or name in ("FLOAT32", "SINGLE"):
            return "FLOAT32"
        if "FLOAT16" in name or name == "FLOAT16":
            if _storage_is_numpy_float16(dt):
                if precision_to_master_datatype(op_precision) == "FLOAT16":
                    return "FLOAT16"
                return "BFLOAT16"
            return "FLOAT16"
        if "INT32" in name:
            return "INT32"
    return precision_to_master_datatype(op_precision)
