#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Operator performance lookup: tt-perf **master** YAML only (``correqn.tt-perf-master``).

Loads via ``tools.perf_lookup.tt_perf_master_loader.load_existing_yaml``. Maps workload ops + tensors to a
logical **9-tuple** (one input), **16-tuple** (two inputs), or **23-tuple** (three inputs) key. Resolves ``single`` (flat
``num_cores`` + stat scalars), ``curve``, and ``hybrid``. For hybrid rows, whether ``curve`` is used
is controlled by :class:`OperatorPerfMap` ``use_hybrid_curve`` (default ``False``: ``single`` only).
See ``doc/tools/perf_lookup/LOOKUP_TABLE_MASTER.md``.

On a **hit** (``msecs`` resolves), **``matrix_pipe_util``** and **``vector_pipe_util``** must resolve
(finite percentages in **[0, 100]** inclusive; **0** allowed). Optional util keys
``mem_util``, ``noc_util``, ``noc_multicast_util``, ``npe_cong_impact_pct`` are validated the same way
when present. Failures raise :class:`OperatorPerfLUTValidationError` (``Device`` re-raises and terminates).

TTNN stores logical BF16 as ``numpy.float16`` (see ``ttsim/front/ttnn/tensor.py``); for LUT keys we
treat that storage as ``BFLOAT16`` unless ``op_precision`` is IEEE FP16 (``fp16`` / ``FLOAT16``).

Master lookup keys always use each tensor's **logical** ``shape`` (rank-4 WZYX after promotion), not
tile-padded extents; ``padded_shape`` is ignored for key construction.

For ``reshape`` master keys, ``input_0`` logical WZYX follows tt-perf convention ``(1, 1, w*z*y, x)``
from the tensor's rank-4 logical ``(w, z, y, x)`` (see :func:`tools.profiling.shape_canonical.reshape_input0_wzyx`).

For ``add``, after a 15-tuple miss, lookup may retry with a key that duplicates the full operand's
WZYX on the broadcast operand when one side is exactly ``(1, 1, 1, X)`` and the other is a
non-trivial rank-4 shape with the same ``X`` (see :func:`build_master_key_tuple_15_add_broadcast_duplicate_full`).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Repo root so ``from tools.perf_lookup…`` works when this file is run as a script
# (``python tools/perf_lookup/lookup_operator_perf.py``); pytest uses ``pythonpath = .``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.perf_lookup.tt_perf_master_loader import load_existing_yaml
from tools.profiling.shape_canonical import (
    coerce_shape_to_list,
    createqkvheads_input0_wzyx,
    promote_to_rank4,
    reshape_input0_wzyx,
    tensor_layout_str,
    tensor_datatype,
    tensor_memory_str,
    precision_to_master_datatype,
)
from tools.perf_lookup.tt_perf_master_schema import (
    MASTER_CURVE_FAMILY_KEY,
    MASTER_CURVE_FAMILY_LINEAR,
    MASTER_CURVE_FAMILY_POWER,
    MASTER_DURATION_MS_KEY,
    MASTER_ENTRY_TYPE_CURVE,
    MASTER_ENTRY_TYPE_HYBRID,
    MASTER_ENTRY_TYPE_KEY,
    MASTER_ENTRY_TYPE_SINGLE,
    MASTER_HYBRID_CURVE_KEY,
    MASTER_HYBRID_SINGLE_KEY,
    MASTER_SINGLE_NUM_CORES_KEY,
    tuple_to_labeled_key_map,
    MATH_FIDELITY_NA,
)

from loguru import logger

_DEFAULT_CORE_COUNT_FALLBACK = 64

# Profiler rows may use an 8-tuple (input_0 only) while the workload graph lists two operands
# (15-tuple). After a full binary key miss, try the first-input 8-tuple for these opcodes.
# ``reshape``: second input is often a small shape/constant tensor, not profiled as input_1.
_BINARY_LUT_FALLBACK_TO_INPUT0_KEY_OPCODES = frozenset({"mul", "reshape"})

# Percentages 0–100 inclusive in master YAML / curve-evaluated stats (not 0–1 fractions).
LUT_OPTIONAL_UTIL_PERCENT_KEYS = frozenset(
    {"mem_util", "noc_util", "noc_multicast_util", "npe_cong_impact_pct"}
)


class OperatorPerfLUTValidationError(ValueError):
    """Raised when a matched LUT row is missing required stats or has invalid utilization percentages."""


def _raise_lut_validation(lut_path: Path, key_t: tuple, detail: str) -> None:
    raise OperatorPerfLUTValidationError(
        f"Operator perf LUT validation failed (file={lut_path}, key={key_t}): {detail}"
    )


def _validate_required_util_percent(
    name: str, raw: Any, lut_path: Path, key_t: tuple
) -> float:
    """Require a finite percentage in [0, 100]. ``raw`` is the resolved scalar from the LUT row."""
    if raw is None:
        _raise_lut_validation(
            lut_path,
            key_t,
            f"required field {name!r} is missing or null; when a LUT row matches, "
            "matrix_pipe_util and vector_pipe_util must both be provided "
            "(percentages 0–100 inclusive; 0 is allowed).",
        )
    if isinstance(raw, bool):
        _raise_lut_validation(
            lut_path, key_t, f"field {name!r} has invalid type bool, value={raw!r}"
        )
    if not isinstance(raw, (int, float)):
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be a number, got {type(raw).__name__}, value={raw!r}",
        )
    v = float(raw)
    if not math.isfinite(v):
        _raise_lut_validation(lut_path, key_t, f"field {name!r} must be finite, got {raw!r}")
    if v < 0.0 or v > 100.0:
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be a percentage in [0, 100] inclusive, got {v!r}",
        )
    return v


def _validate_optional_util_percent(
    name: str, raw: Any, lut_path: Path, key_t: tuple
) -> Optional[float]:
    """If ``raw`` is None, return None; else same rules as required util percent."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        _raise_lut_validation(
            lut_path, key_t, f"field {name!r} has invalid type bool, value={raw!r}"
        )
    if not isinstance(raw, (int, float)):
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be a number or null, got {type(raw).__name__}, value={raw!r}",
        )
    v = float(raw)
    if not math.isfinite(v):
        _raise_lut_validation(
            lut_path, key_t, f"field {name!r} must be finite when present, got {raw!r}"
        )
    if v < 0.0 or v > 100.0:
        _raise_lut_validation(
            lut_path,
            key_t,
            f"field {name!r} must be a percentage in [0, 100] inclusive when present, got {v!r}",
        )
    return v


@dataclass(frozen=True)
class MasterPerfStats:
    """Resolved profiler row for one op (after single/curve/hybrid evaluation).

    ``matrix_pipe_util`` and ``vector_pipe_util`` are percentages in **[0, 100]** (not fractions).
    ``Device.get_exec_stats`` divides by 100 for exec_stats / CSV. Optional ``mem_util`` is the same.
    """

    msecs: float
    matrix_pipe_util: float
    vector_pipe_util: float
    memory_traffic: Optional[float] = None
    mem_util: Optional[float] = None


def _eval_curve_value(family: str, a: float, b: float, core_count: int) -> float:
    """Evaluate regression at ``core_count`` (Core_Count in master spec)."""
    n = float(core_count)
    if family == MASTER_CURVE_FAMILY_LINEAR:
        return a * n + b
    if family == MASTER_CURVE_FAMILY_POWER:
        if n <= 0:
            return b
        return a * (n**b)
    raise ValueError(f"Unknown curve family: {family!r}")


def _resolve_scalar_from_flat(flat: dict, stat_name: str) -> Optional[float]:
    v = flat.get(stat_name)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _resolve_stat_from_flat_single(flat: dict, stat_name: str) -> Optional[float]:
    """Read a stat scalar from a flat single payload (``num_cores`` + ``MASTER_SINGLE_STAT_KEYS``)."""
    return _resolve_scalar_from_flat(flat, stat_name)


def _resolve_stat_from_curve(
    curve_payload: dict, stat_name: str, core_count: int
) -> Optional[float]:
    """Evaluate one stat from a curve mapping (top-level ``curve`` or ``hybrid.curve``)."""
    family = curve_payload.get(MASTER_CURVE_FAMILY_KEY)
    if family not in (MASTER_CURVE_FAMILY_LINEAR, MASTER_CURVE_FAMILY_POWER):
        return None
    sub = curve_payload.get(stat_name)
    if not isinstance(sub, dict):
        return None
    a, b = sub.get("a"), sub.get("b")
    if a is None or b is None:
        return None
    return _eval_curve_value(family, float(a), float(b), core_count)


def _maybe_warn_single_core_mismatch(
    flat: dict, core_count: int, key_t: tuple, lut_path: Path
) -> None:
    nc = flat.get(MASTER_SINGLE_NUM_CORES_KEY)
    if nc is None:
        return
    try:
        n_table = int(nc)
    except (TypeError, ValueError):
        return
    if int(core_count) != n_table:
        logger.debug(
            "Perf lookup single-entry: runtime core_count={} != table num_cores={} "
            "(using table scalars as-is) key={} lut={}",
            core_count,
            n_table,
            key_t,
            lut_path,
        )


def _build_stat_resolver(
    entry_val: dict,
    entry_type: str,
    core_count: int,
    key_t: tuple,
    lut_path: Path,
    *,
    use_hybrid_curve: bool = False,
) -> Optional[Callable[[str], Optional[float]]]:
    """Return ``resolve(stat_name)`` for the loaded entry type, or ``None`` if unsupported."""
    if entry_type == MASTER_ENTRY_TYPE_SINGLE:
        flat = {k: v for k, v in entry_val.items() if k != MASTER_ENTRY_TYPE_KEY}
        _maybe_warn_single_core_mismatch(flat, core_count, key_t, lut_path)

        def resolve(stat: str) -> Optional[float]:
            return _resolve_stat_from_flat_single(flat, stat)

        return resolve

    if entry_type == MASTER_ENTRY_TYPE_CURVE:

        def resolve(stat: str) -> Optional[float]:
            return _resolve_stat_from_curve(entry_val, stat, core_count)

        return resolve

    if entry_type == MASTER_ENTRY_TYPE_HYBRID:
        curve_sub = entry_val.get(MASTER_HYBRID_CURVE_KEY)
        if use_hybrid_curve and isinstance(curve_sub, dict):

            def resolve(stat: str) -> Optional[float]:
                return _resolve_stat_from_curve(curve_sub, stat, core_count)

            return resolve

        single_sub = entry_val.get(MASTER_HYBRID_SINGLE_KEY)
        if isinstance(single_sub, dict):
            _maybe_warn_single_core_mismatch(single_sub, core_count, key_t, lut_path)

            def resolve(stat: str) -> Optional[float]:
                return _resolve_stat_from_flat_single(single_sub, stat)

            return resolve

    return None


def _op_code(op: Any) -> str:
    from tools.profiling.op_canonical import normalize_polaris_optype

    return normalize_polaris_optype(str(getattr(op, "optype", "")).strip())


def _shape_wzyx(tensor: Any) -> Tuple[int, int, int, int]:
    """Rank-4 WZYX shape for LUT key construction."""
    raw = getattr(tensor, "shape", None)
    if raw is None:
        raise ValueError("tensor has no shape")
    raw_list = coerce_shape_to_list(raw)
    return promote_to_rank4(raw_list)


def _input0_wzyx_for_master_key(op: Any, tensor_0: Any) -> Tuple[int, int, int, int]:
    w, z, y, x = _shape_wzyx(tensor_0)
    if _op_code(op) == "reshape":
        return reshape_input0_wzyx(w, z, y, x)
    if _op_code(op) == "createqkvheads":
        return createqkvheads_input0_wzyx(w, z, y, x)
    return (w, z, y, x)


_MATH_FIDELITY_CALLER_CONTROLLED_OPS = frozenset({"layernorm"})


def _op_math_fidelity(op: Any) -> str:
    """Extract math fidelity from SimOp attrs.

    For ops in ``_MATH_FIDELITY_CALLER_CONTROLLED_OPS`` (e.g. layernorm),
    returns the explicit value from attrs when present, or ``HiFi4`` (the
    hardware default) when omitted. For all other ops, returns ``N/A``
    matching the mapper's normalization.
    """
    attrs = getattr(op, "attrs", None)
    if isinstance(attrs, dict):
        mf = attrs.get("math_fidelity")
        if mf is not None:
            return str(mf)
    if _op_code(op) in _MATH_FIDELITY_CALLER_CONTROLLED_OPS:
        return "HiFi4"
    return MATH_FIDELITY_NA


def build_master_key_tuple_8(op: Any, tensor_0: Any) -> Tuple[Any, ...]:
    """Logical 9-tuple (first input + math fidelity); order matches ``KEY_TUPLE_YAML_KEYS[:9]``."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, getattr(op, "precision", None)),
        tensor_memory_str(tensor_0),
        _op_math_fidelity(op),
    )


def build_master_key_tuple_15(
    op: Any,
    tensor_0: Any,
    tensor_1: Any,
) -> Tuple[Any, ...]:
    """Logical 16-tuple matching ``tools.perf_lookup.tt_perf_master_schema.KEY_TUPLE_YAML_KEYS`` order."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    w1, z1, y1, x1 = _shape_wzyx(tensor_1)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, getattr(op, "precision", None)),
        tensor_memory_str(tensor_0),
        _op_math_fidelity(op),
        w1,
        z1,
        y1,
        x1,
        tensor_layout_str(tensor_1),
        tensor_datatype(tensor_1, getattr(op, "precision", None)),
        tensor_memory_str(tensor_1),
    )


def build_master_key_tuple_22(
    op: Any,
    tensor_0: Any,
    tensor_1: Any,
    tensor_2: Any,
) -> Tuple[Any, ...]:
    """Logical 23-tuple matching ``tools.perf_lookup.tt_perf_master_schema.KEY_TUPLE_YAML_KEYS`` order."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    w1, z1, y1, x1 = _shape_wzyx(tensor_1)
    w2, z2, y2, x2 = _shape_wzyx(tensor_2)
    prec = getattr(op, "precision", None)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, prec),
        tensor_memory_str(tensor_0),
        _op_math_fidelity(op),
        w1,
        z1,
        y1,
        x1,
        tensor_layout_str(tensor_1),
        tensor_datatype(tensor_1, prec),
        tensor_memory_str(tensor_1),
        w2,
        z2,
        y2,
        x2,
        tensor_layout_str(tensor_2),
        tensor_datatype(tensor_2, prec),
        tensor_memory_str(tensor_2),
    )


def build_master_key_tuple_15_add_broadcast_duplicate_full(
    op: Any,
    tensor_0: Any,
    tensor_1: Any,
) -> Optional[Tuple[Any, ...]]:
    """
    16-tuple key with the broadcast operand's WZYX replaced by the full operand's WZYX.

    Used when the master row stored both inputs with the full logical shape while the graph has
    ``(1, 1, 1, X)`` on one operand (either ``tensor_0`` or ``tensor_1``). Layout, datatype, and
    memory for each input stay tied to that input's tensor.

    Returns ``None`` unless ``op`` is ``add``, one operand is exactly ``(1,1,1,X)``, the other
    has the same ``X`` and at least one of W, Z, Y greater than 1, and the pattern is unambiguous.
    """
    if _op_code(op) != "add":
        return None
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    w1, z1, y1, x1 = _shape_wzyx(tensor_1)
    b0 = w0 == 1 and z0 == 1 and y0 == 1
    b1 = w1 == 1 and z1 == 1 and y1 == 1
    full0 = not b0
    full1 = not b1
    prec = getattr(op, "precision", None)
    mf = _op_math_fidelity(op)
    lay0, dt0, mem0 = (
        tensor_layout_str(tensor_0),
        tensor_datatype(tensor_0, prec),
        tensor_memory_str(tensor_0),
    )
    lay1, dt1, mem1 = (
        tensor_layout_str(tensor_1),
        tensor_datatype(tensor_1, prec),
        tensor_memory_str(tensor_1),
    )
    if b1 and full0 and x1 == x0:
        wf, zf, yf, xf = w0, z0, y0, x0
        return (
            _op_code(op),
            wf,
            zf,
            yf,
            xf,
            lay0,
            dt0,
            mem0,
            mf,
            wf,
            zf,
            yf,
            xf,
            lay1,
            dt1,
            mem1,
        )
    if b0 and full1 and x0 == x1:
        wf, zf, yf, xf = w1, z1, y1, x1
        return (
            _op_code(op),
            wf,
            zf,
            yf,
            xf,
            lay0,
            dt0,
            mem0,
            mf,
            wf,
            zf,
            yf,
            xf,
            lay1,
            dt1,
            mem1,
        )
    return None


def _wzyx_int_tuple(t4: tuple) -> tuple:
    """Four logical dims as ints (YAML / numpy-safe)."""
    out: list[int] = []
    for x in t4:
        if hasattr(x, "item") and callable(getattr(x, "item", None)):
            out.append(int(x))
        else:
            out.append(int(x))
    return tuple(out)


def _lut_keys_matching_op_and_wzyx(entries: Dict[tuple, dict], key_t: tuple) -> Tuple[tuple, ...]:
    """
    LUT keys with same ``op_code`` and WZYX as ``key_t``
    (input 0 for 8-tuple; inputs 0 and 1 for 15-tuple; inputs 0, 1, and 2 for 22-tuple).

    Layout, datatype, and memory may differ — useful diagnostics when the full key misses.
    """
    n = len(key_t)
    if n not in (9, 16, 23):
        return ()
    oc = key_t[0]
    matched: list[tuple] = []
    if n == 9:
        w0 = _wzyx_int_tuple(key_t[1:5])
        for k in entries:
            if len(k) != 9 or k[0] != oc:
                continue
            if _wzyx_int_tuple(k[1:5]) == w0:
                matched.append(k)
    elif n == 16:
        w0 = _wzyx_int_tuple(key_t[1:5])
        w1 = _wzyx_int_tuple(key_t[9:13])
        for k in entries:
            if len(k) != 16 or k[0] != oc:
                continue
            if _wzyx_int_tuple(k[1:5]) == w0 and _wzyx_int_tuple(k[9:13]) == w1:
                matched.append(k)
    else:
        w0 = _wzyx_int_tuple(key_t[1:5])
        w1 = _wzyx_int_tuple(key_t[9:13])
        w2 = _wzyx_int_tuple(key_t[16:20])
        for k in entries:
            if len(k) != 23 or k[0] != oc:
                continue
            if (
                _wzyx_int_tuple(k[1:5]) == w0
                and _wzyx_int_tuple(k[9:13]) == w1
                and _wzyx_int_tuple(k[16:20]) == w2
            ):
                matched.append(k)
    return tuple(sorted(matched))


def _lut_keys_matching_op_code_only(entries: Dict[tuple, dict], key_t: tuple) -> Tuple[tuple, ...]:
    """
    All LUT keys with the same ``op_code`` and tuple length as ``key_t`` (unary vs binary).

    Listed on full-key miss when no row matches operator type and all key attributes; contrasts
    with :func:`_lut_keys_matching_op_and_wzyx` which also requires WZYX match.
    """
    n = len(key_t)
    if n not in (9, 16, 23):
        return ()
    oc = key_t[0]
    matched = [k for k in entries if len(k) == n and k[0] == oc]
    return tuple(sorted(matched))


def _warn_labeled_lut_key_candidates(
    group_label: str,
    keys: Tuple[tuple, ...],
    *,
    reference_key: Optional[tuple] = None,
) -> None:
    """Log each LUT key tuple for miss diagnostics.

    When ``reference_key`` is set (the constructed lookup key), each candidate is compared
    field-by-field and **only differing** attributes are logged as ``lookup=`` vs ``lut_row=``.
    Otherwise the full labeled mapping is logged for each candidate.
    """
    n = len(keys)
    if n == 0:
        return
    ref_labeled: Optional[Dict[str, Any]] = None
    if reference_key is not None:
        try:
            ref_labeled = tuple_to_labeled_key_map(reference_key)
        except ValueError:
            ref_labeled = None

    for i, k in enumerate(keys):
        logger.warning("LUT {} candidate {}/{}:", group_label, i + 1, n)
        try:
            labeled = tuple_to_labeled_key_map(k)
        except ValueError:
            logger.warning("  raw={!r}", k)
            continue
        if ref_labeled is None:
            for name, val in labeled.items():
                logger.warning("  {}={}", name, val)
            continue
        diffs: list[tuple[str, Any, Any]] = []
        for name, lut_val in labeled.items():
            wanted = ref_labeled.get(name)
            if wanted != lut_val:
                diffs.append((name, wanted, lut_val))
        if diffs:
            logger.warning(
                "  vs lookup: {} attribute(s) differ — {}",
                len(diffs),
                ", ".join(nm for nm, _, _ in diffs),
            )
            for name, wanted, lut_val in diffs:
                logger.warning("    {}: lookup={!r} lut_row={!r}", name, wanted, lut_val)
        else:
            logger.warning(
                "  vs lookup: no differing labeled fields (tuple mismatch may be subtle); "
                "full row: {}",
                labeled,
            )


class OperatorPerfMap:
    """
    Master-format operator performance table: lookup by constructed 9-, 16-, or 23-tuple key
    and core count.
    """

    def __init__(
        self,
        yaml_file: Union[str, Path],
        *,
        use_hybrid_curve: bool = False,
    ):
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_file}")

        self._entries: Dict[tuple, dict] = load_existing_yaml(yaml_path)
        self._source_path = yaml_path
        self._use_hybrid_curve = bool(use_hybrid_curve)

    def __len__(self) -> int:
        return len(self._entries)

    def _stats_from_entry(self, key_t: tuple, entry_val: dict, core_count: int) -> Optional[MasterPerfStats]:
        et = entry_val.get(MASTER_ENTRY_TYPE_KEY)
        if not isinstance(et, str):
            return None
        resolve = _build_stat_resolver(
            entry_val,
            et,
            core_count,
            key_t,
            self._source_path,
            use_hybrid_curve=self._use_hybrid_curve,
        )
        if resolve is None:
            return None

        msecs = resolve(MASTER_DURATION_MS_KEY)
        if msecs is None:
            logger.warning(
                "Perf lookup hit but no msecs for key={} core_count={} (path={})",
                key_t,
                core_count,
                self._source_path,
            )
            return None

        lut_path = self._source_path
        matrix_pipe_util = _validate_required_util_percent(
            "matrix_pipe_util", resolve("matrix_pipe_util"), lut_path, key_t
        )
        vector_pipe_util = _validate_required_util_percent(
            "vector_pipe_util", resolve("vector_pipe_util"), lut_path, key_t
        )

        mem_util = _validate_optional_util_percent(
            "mem_util", resolve("mem_util"), lut_path, key_t
        )
        for opt in LUT_OPTIONAL_UTIL_PERCENT_KEYS:
            if opt == "mem_util":
                continue
            v = resolve(opt)
            if v is not None:
                _validate_optional_util_percent(opt, v, lut_path, key_t)

        return MasterPerfStats(
            msecs=float(msecs),
            matrix_pipe_util=matrix_pipe_util,
            vector_pipe_util=vector_pipe_util,
            memory_traffic=resolve("memory_traffic"),
            mem_util=mem_util,
        )

    def lookup(
        self,
        op: Any,
        wlgraph: Any,
        core_count: int,
    ) -> Optional[MasterPerfStats]:
        """
        Return resolved stats when the table has a matching 9-, 16-, or 23-tuple key, else ``None``.

        ``core_count`` should match profiler Core_Count bucketing (see package config).
        """
        in_list = getattr(op, "inList", [])
        n_in = len(in_list)
        tensors = getattr(wlgraph, "_tensors", {})

        if n_in == 0 or n_in > 3:
            logger.debug(
                "Perf lookup skipped (arity {} not supported for master keys): op={} optype={}",
                n_in,
                getattr(op, "name", "?"),
                getattr(op, "optype", "?"),
            )
            return None

        t0 = t1 = t2 = None
        if n_in == 1:
            t0_name = in_list[0]
            if t0_name not in tensors:
                return None
            t0 = tensors[t0_name]
            if t0.shape is None:
                return None
            try:
                key_t = build_master_key_tuple_8(op, t0)
            except Exception as e:
                logger.debug("Perf lookup key build failed for op {}: {}", getattr(op, "name", "?"), e)
                return None
        elif n_in == 2:
            t0_name, t1_name = in_list[0], in_list[1]
            if t0_name not in tensors or t1_name not in tensors:
                return None
            t0, t1 = tensors[t0_name], tensors[t1_name]
            if t0.shape is None or t1.shape is None:
                return None
            try:
                key_t = build_master_key_tuple_15(op, t0, t1)
            except Exception as e:
                logger.debug("Perf lookup key build failed for op {}: {}", getattr(op, "name", "?"), e)
                return None
        else:
            t0_name, t1_name, t2_name = in_list[0], in_list[1], in_list[2]
            if t0_name not in tensors or t1_name not in tensors or t2_name not in tensors:
                return None
            t0, t1, t2 = tensors[t0_name], tensors[t1_name], tensors[t2_name]
            if t0.shape is None or t1.shape is None or t2.shape is None:
                return None
            try:
                key_t = build_master_key_tuple_22(op, t0, t1, t2)
            except Exception as e:
                logger.debug("Perf lookup key build failed for op {}: {}", getattr(op, "name", "?"), e)
                return None

        entry_val = self._entries.get(key_t)
        lookup_key = key_t

        if entry_val is None and n_in == 2 and _op_code(op) == "add":
            try:
                key_add_bc = build_master_key_tuple_15_add_broadcast_duplicate_full(op, t0, t1)
            except Exception as e:
                logger.debug(
                    "Perf lookup add broadcast key build failed for op {}: {}",
                    getattr(op, "name", "?"),
                    e,
                )
                key_add_bc = None
            if key_add_bc is not None:
                ev_add = self._entries.get(key_add_bc)
                if ev_add is not None:
                    entry_val = ev_add
                    lookup_key = key_add_bc
                    logger.debug(
                        "Perf lookup: 15-tuple miss for add op={!r}; using broadcast-duplicated "
                        "LUT key {} (lut={})",
                        getattr(op, "name", None),
                        key_add_bc,
                        self._source_path,
                    )

        if (
            entry_val is None
            and n_in == 2
            and _op_code(op) in _BINARY_LUT_FALLBACK_TO_INPUT0_KEY_OPCODES
        ):
            try:
                key8 = build_master_key_tuple_8(op, t0)
            except Exception as e:
                logger.debug(
                    "Perf lookup unary fallback key build failed for op {}: {}",
                    getattr(op, "name", "?"),
                    e,
                )
                key8 = None
            if key8 is not None:
                ev8 = self._entries.get(key8)
                if ev8 is not None:
                    entry_val = ev8
                    lookup_key = key8
                    logger.debug(
                        "Perf lookup: 15-tuple miss for {} op={!r}; using unary LUT key {} (lut={})",
                        _op_code(op),
                        getattr(op, "name", None),
                        key8,
                        self._source_path,
                    )

        if entry_val is None:
            same_op_shape_keys = _lut_keys_matching_op_and_wzyx(self._entries, key_t)
            same_op_only_keys = _lut_keys_matching_op_code_only(self._entries, key_t)
            logger.warning(
                "Perf lookup miss (no matching LUT row) opname={!r} optype={!r} arity={} "
                "key={} core_count={} lut={} n_lut_keys_same_op_and_shapes={} "
                "n_lut_keys_same_op_code_only={}",
                getattr(op, "name", None),
                getattr(op, "optype", None),
                n_in,
                key_t,
                core_count,
                self._source_path,
                len(same_op_shape_keys),
                len(same_op_only_keys),
            )
            _warn_labeled_lut_key_candidates(
                "same-op+shape", same_op_shape_keys, reference_key=key_t
            )
            _warn_labeled_lut_key_candidates(
                "same-op-code-only", same_op_only_keys, reference_key=key_t
            )
            logger.warning("------------------------------------------")
            return None

        logger.debug(
            "Perf lookup hit opname={!r} optype={!r} key={} core_count={} lut={} entry={}",
            getattr(op, "name", None),
            getattr(op, "optype", None),
            lookup_key,
            core_count,
            self._source_path,
            entry_val,
        )
        return self._stats_from_entry(lookup_key, entry_val, core_count)


def resolve_operator_lookup_core_count(simcfg_obj: Any, package_model: Any) -> int:
    """
    Core count for master single/curve evaluation.

    Order: ``operator_lookup_core_count`` on package if set; else compute IPGroup
    ``num_units``; else ``_DEFAULT_CORE_COUNT_FALLBACK``.
    """
    raw = getattr(package_model, "operator_lookup_core_count", None)
    if raw is not None:
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass

    try:
        compute_group = package_model.get_ipgroup(iptype="compute")
        nu = int(compute_group.num_units)
        if nu > 0:
            return nu
    except Exception:
        pass

    logger.info(
        "operator_lookup_core_count unset and compute num_units missing; using fallback {}",
        _DEFAULT_CORE_COUNT_FALLBACK,
    )
    return _DEFAULT_CORE_COUNT_FALLBACK
