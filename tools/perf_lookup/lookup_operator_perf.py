#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Operator performance lookup: tt-perf **master** YAML only (``correqn.tt-perf-master``).

Loads via ``tools.perf_lookup.tt_perf_master_loader.load_existing_yaml``. Maps workload ops + tensors to a
logical **8-tuple** (one input) or **15-tuple** (two inputs) key. Resolves ``single`` (flat
``num_cores`` + stat scalars), ``curve``, and ``hybrid``. For hybrid rows, whether ``curve`` is used
is controlled by :class:`OperatorPerfMap` ``use_hybrid_curve`` (default ``False``: ``single`` only).
See ``doc/tools/perf_lookup/LOOKUP_TABLE_MASTER.md``.

TTNN stores logical BF16 as ``numpy.float16`` (see ``ttsim/front/ttnn/tensor.py``); for LUT keys we
treat that storage as ``BFLOAT16`` unless ``op_precision`` is IEEE FP16 (``fp16`` / ``FLOAT16``).

For ``reshape`` master keys, ``input_0`` logical WZYX follows tt-perf convention ``(1, 1, w*z*y, x)``
from the tensor's rank-4 logical ``(w, z, y, x)`` (see :func:`_reshape_input0_lut_wzyx`).
"""

from __future__ import annotations

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
)

from loguru import logger

_DEFAULT_CORE_COUNT_FALLBACK = 64

# Profiler rows may use an 8-tuple (input_0 only) while the workload graph lists two operands
# (15-tuple). After a full binary key miss, try the first-input 8-tuple for these opcodes.
# ``reshape``: second input is often a small shape/constant tensor, not profiled as input_1.
_BINARY_LUT_FALLBACK_TO_INPUT0_KEY_OPCODES = frozenset({"mul", "reshape"})


@dataclass(frozen=True)
class MasterPerfStats:
    """Resolved profiler row for one op (after single/curve/hybrid evaluation)."""

    msecs: float
    memory_traffic: Optional[float] = None
    mem_util: Optional[float] = None
    vector_pipe_util: Optional[float] = None
    matrix_pipe_util: Optional[float] = None


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
    return str(getattr(op, "optype", "")).strip().lower()


def _precision_to_master_datatype(precision: Any) -> str:
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
    """True when ``dt`` is ``numpy.float16`` (TTNN uses it as the container for logical BF16)."""
    try:
        import numpy as np

        if isinstance(dt, np.dtype):
            return dt == np.dtype(np.float16)
    except ImportError:
        pass
    nm = getattr(dt, "name", None)
    return isinstance(nm, str) and nm.lower() == "float16"


def _tensor_layout_str(t: Any) -> str:
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


def _tensor_memory_str(t: Any) -> str:
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
    s = str(mc).upper().replace(" ", "_")
    if "L1" in s:
        return "DEV_1_L1"
    return "DEV_1_DRAM_INTERLEAVED"


def _tensor_datatype(t: Any, op_precision: Any) -> str:
    dt = getattr(t, "dtype", None)
    if dt is not None:
        name = getattr(dt, "name", str(dt)).upper()
        if "BFLOAT16" in name or name == "BFLOAT16":
            return "BFLOAT16"
        if "FLOAT32" in name or name in ("FLOAT32", "SINGLE"):
            return "FLOAT32"
        if "FLOAT16" in name or name == "FLOAT16":
            # numpy float16 is TTNN's storage for logical BF16; keep IEEE FP16 only when op says fp16
            if _storage_is_numpy_float16(dt):
                if _precision_to_master_datatype(op_precision) == "FLOAT16":
                    return "FLOAT16"
                return "BFLOAT16"
            return "FLOAT16"
        if "INT32" in name:
            return "INT32"
    return _precision_to_master_datatype(op_precision)


def _shape_wzyx(tensor: Any) -> Tuple[int, int, int, int]:
    from ttsim.ops.tensor import Shape

    raw = getattr(tensor, "shape", None)
    if raw is None:
        raise ValueError("tensor has no shape")
    sh = Shape(list(raw)) if not isinstance(raw, Shape) else raw
    v = sh.to_rank(4).view()
    return (int(v[0]), int(v[1]), int(v[2]), int(v[3]))


def _reshape_input0_lut_wzyx(w: int, z: int, y: int, x: int) -> Tuple[int, int, int, int]:
    """
    Master LUT convention for ``reshape`` ``input_0_*`` logical dims: ``(1, 1, w*z*y, x)``,
    where ``(w, z, y, x)`` is the tensor's rank-4 logical shape (same basis as :func:`_shape_wzyx`).
    """
    return (1, 1, int(w) * int(z) * int(y), int(x))


def _input0_wzyx_for_master_key(op: Any, tensor_0: Any) -> Tuple[int, int, int, int]:
    w, z, y, x = _shape_wzyx(tensor_0)
    if _op_code(op) == "reshape":
        return _reshape_input0_lut_wzyx(w, z, y, x)
    return (w, z, y, x)


def build_master_key_tuple_8(op: Any, tensor_0: Any) -> Tuple[Any, ...]:
    """Logical 8-tuple (first input only); order matches ``KEY_TUPLE_YAML_KEYS[:8]``."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        _tensor_layout_str(tensor_0),
        _tensor_datatype(tensor_0, getattr(op, "precision", None)),
        _tensor_memory_str(tensor_0),
    )


def build_master_key_tuple_15(op: Any, tensor_0: Any, tensor_1: Any) -> Tuple[Any, ...]:
    """Logical 15-tuple matching ``tools.perf_lookup.tt_perf_master_schema.KEY_TUPLE_YAML_KEYS`` order."""
    w0, z0, y0, x0 = _input0_wzyx_for_master_key(op, tensor_0)
    w1, z1, y1, x1 = _shape_wzyx(tensor_1)
    return (
        _op_code(op),
        w0,
        z0,
        y0,
        x0,
        _tensor_layout_str(tensor_0),
        _tensor_datatype(tensor_0, getattr(op, "precision", None)),
        _tensor_memory_str(tensor_0),
        w1,
        z1,
        y1,
        x1,
        _tensor_layout_str(tensor_1),
        _tensor_datatype(tensor_1, getattr(op, "precision", None)),
        _tensor_memory_str(tensor_1),
    )


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
    LUT keys with same ``op_code`` and WZYX as ``key_t`` (input 0 for 8-tuple; inputs 0 and 1 for 15-tuple).

    Layout, datatype, and memory may differ — useful diagnostics when the full key misses.
    """
    n = len(key_t)
    if n not in (8, 15):
        return ()
    oc = key_t[0]
    matched: list[tuple] = []
    if n == 8:
        w0 = _wzyx_int_tuple(key_t[1:5])
        for k in entries:
            if len(k) != 8 or k[0] != oc:
                continue
            if _wzyx_int_tuple(k[1:5]) == w0:
                matched.append(k)
    else:
        w0 = _wzyx_int_tuple(key_t[1:5])
        w1 = _wzyx_int_tuple(key_t[8:12])
        for k in entries:
            if len(k) != 15 or k[0] != oc:
                continue
            if _wzyx_int_tuple(k[1:5]) == w0 and _wzyx_int_tuple(k[8:12]) == w1:
                matched.append(k)
    return tuple(sorted(matched))


def _lut_keys_matching_op_code_only(entries: Dict[tuple, dict], key_t: tuple) -> Tuple[tuple, ...]:
    """
    All LUT keys with the same ``op_code`` and tuple length as ``key_t`` (unary vs binary).

    Listed on full-key miss when no row matches operator type and all key attributes; contrasts
    with :func:`_lut_keys_matching_op_and_wzyx` which also requires WZYX match.
    """
    n = len(key_t)
    if n not in (8, 15):
        return ()
    oc = key_t[0]
    matched = [k for k in entries if len(k) == n and k[0] == oc]
    return tuple(sorted(matched))


class OperatorPerfMap:
    """
    Master-format operator performance table: lookup by constructed 8- or 15-tuple key
    and core count.
    """

    def __init__(self, yaml_file: Union[str, Path], *, use_hybrid_curve: bool = False):
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

        return MasterPerfStats(
            msecs=float(msecs),
            memory_traffic=resolve("memory_traffic"),
            mem_util=resolve("mem_util"),
            vector_pipe_util=resolve("vector_pipe_util"),
            matrix_pipe_util=resolve("matrix_pipe_util"),
        )

    def lookup(
        self,
        op: Any,
        wlgraph: Any,
        core_count: int,
    ) -> Optional[MasterPerfStats]:
        """
        Return resolved stats when the table has a matching 8- or 15-tuple key, else ``None``.

        ``core_count`` should match profiler Core_Count bucketing (see package config).
        """
        in_list = getattr(op, "inList", [])
        n_in = len(in_list)
        tensors = getattr(wlgraph, "_tensors", {})

        if n_in == 0 or n_in > 2:
            logger.debug(
                "Perf lookup skipped (arity {} not supported for master keys): op={} optype={}",
                n_in,
                getattr(op, "name", "?"),
                getattr(op, "optype", "?"),
            )
            return None

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
        else:
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

        entry_val = self._entries.get(key_t)
        lookup_key = key_t

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
                "key={} core_count={} lut={} lut_keys_same_op_and_shapes={} "
                "lut_keys_same_op_code_only={}",
                getattr(op, "name", None),
                getattr(op, "optype", None),
                n_in,
                key_t,
                core_count,
                self._source_path,
                list(same_op_shape_keys),
                list(same_op_only_keys),
            )
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
