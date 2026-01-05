# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wire-format constants and logical key ↔ labeled YAML mapping for tt-perf master files.

Normative field names: ``doc/YAML_MASTER_FORMAT.md``. Excel pipeline and CLI: ``tools/perf_lookup/tt_perf_mapper.py``.
On-disk ``schema_version`` stays **1** until the first release; bump only then for incompatible layout changes.
``entry_type`` may be ``single``, ``curve``, or ``hybrid`` (matmul). Used by ``tt_perf_master_loader`` and
``tt_perf_mapper`` (serialize, merge, Excel→stats keys).

Shared normalization and scalar checks live here (stdlib only) so loader and mapper stay aligned.
"""

from __future__ import annotations

import math
import numbers

# Labeled record key under ``entries[i]['key']``. Order aligns with Excel ``KEY_TUPLE_COLUMN_NAMES``.
KEY_TUPLE_YAML_KEYS = [
    "op_code",
    "input_0_w_pad_logical",
    "input_0_z_pad_logical",
    "input_0_y_pad_logical",
    "input_0_x_pad_logical",
    "input_0_layout",
    "input_0_datatype",
    "input_0_memory",
    "input_1_w_pad_logical",
    "input_1_z_pad_logical",
    "input_1_y_pad_logical",
    "input_1_x_pad_logical",
    "input_1_layout",
    "input_1_datatype",
    "input_1_memory",
    "input_2_w_pad_logical",
    "input_2_z_pad_logical",
    "input_2_y_pad_logical",
    "input_2_x_pad_logical",
    "input_2_layout",
    "input_2_datatype",
    "input_2_memory",
]

_KEY_TUPLE_YAML_KEYS_SET = frozenset(KEY_TUPLE_YAML_KEYS)

# Device kernel duration in master YAML (milliseconds). Excel uses nanoseconds column title.
MASTER_DURATION_MS_KEY = "msecs"

MASTER_ENTRY_TYPE_KEY = "entry_type"
MASTER_ENTRY_TYPE_SINGLE = "single"
MASTER_ENTRY_TYPE_CURVE = "curve"
MASTER_ENTRY_TYPE_HYBRID = "hybrid"

# Flat ``entry_type: single`` payload: ``num_cores`` plus the stat keys below (same under ``hybrid.single``).
MASTER_SINGLE_NUM_CORES_KEY = "num_cores"
MASTER_SINGLE_STAT_KEYS = (
    MASTER_DURATION_MS_KEY,
    "memory_traffic",
    "mem_util",
    "noc_util",
    "noc_multicast_util",
    "npe_cong_impact_pct",
    "vector_pipe_util",
    "matrix_pipe_util",
)

MASTER_SINGLE_STAT_KEYS_SET = frozenset(MASTER_SINGLE_STAT_KEYS)

# Flat single payload keys that are utilization-like percentages / scores (not duration or traffic).
MASTER_SINGLE_UTIL_STAT_KEYS = frozenset(
    (
        "mem_util",
        "noc_util",
        "noc_multicast_util",
        "npe_cong_impact_pct",
        "vector_pipe_util",
        "matrix_pipe_util",
    )
)

# Each per-stat curve fit object (under ``curve`` / ``hybrid.curve``) uses these keys.
MASTER_CURVE_STAT_ENTRY_KEYS = frozenset(("a", "b", "r2", "equation"))

# Matmul hybrid: ``single`` = flat ``num_cores`` + stats; ``curve`` = ``curve_family`` + per-stat fits.
MASTER_HYBRID_SINGLE_KEY = "single"
MASTER_HYBRID_CURVE_KEY = "curve"

MASTER_CURVE_FAMILY_KEY = "curve_family"
MASTER_CURVE_FAMILY_LINEAR = "linear"
MASTER_CURVE_FAMILY_POWER = "power"

MASTER_YAML_SCHEMA_NAME = "correqn.tt-perf-master"
MASTER_YAML_SCHEMA_NAME_KEY = "schema_name"
# Bump only at first release (or later) when the on-disk layout changes incompatibly.
MASTER_YAML_SCHEMA_VERSION = 1
MASTER_YAML_SCHEMA_VERSION_KEY = "schema_version"
MASTER_YAML_ENTRIES_KEY = "entries"
MASTER_YAML_RECORD_KEY_FIELD = "key"
MASTER_YAML_ENTRY_VALUE_FIELD = "value"


def tuple_to_labeled_key_map(key_t: tuple) -> dict:
    """Serialize logical key tuple to YAML mapping (under ``entries[i]['key']``)."""
    n = len(key_t)
    if n not in (8, 15, 22):
        raise ValueError(f"Key tuple length must be 8, 15, or 22, got {n}")
    return dict(zip(KEY_TUPLE_YAML_KEYS[:n], key_t))


def labeled_key_map_to_tuple(d: dict) -> tuple:
    """Parse YAML labeled mapping back to logical key tuple."""
    if not isinstance(d, dict):
        raise TypeError(f"Labeled key must be a dict, got {type(d)}")
    filtered: dict = {}
    unknown: list = []
    for k, v in d.items():
        ks = k if isinstance(k, str) else str(k)
        if ks in _KEY_TUPLE_YAML_KEYS_SET:
            filtered[ks] = v
        else:
            unknown.append(k)
    if unknown:
        raise ValueError(f"Unknown labeled-key field(s): {unknown}")
    has_i2 = any(str(k).startswith("input_2_") for k in filtered)
    has_i1 = any(str(k).startswith("input_1_") for k in filtered)
    if has_i2:
        names22 = KEY_TUPLE_YAML_KEYS
        missing = [nm for nm in names22 if nm not in filtered]
        if missing:
            raise ValueError(
                "Labeled key with any input_2 field must define all 22 fields; "
                f"missing: {missing}"
            )
        return tuple(filtered[nm] for nm in names22)
    if has_i1:
        names15 = KEY_TUPLE_YAML_KEYS[:15]
        missing = [nm for nm in names15 if nm not in filtered]
        if missing:
            raise ValueError(
                "Labeled key with any input_1 field must define all 15 fields; "
                f"missing: {missing}"
            )
        return tuple(filtered[nm] for nm in names15)
    names8 = KEY_TUPLE_YAML_KEYS[:8]
    missing = [nm for nm in names8 if nm not in filtered]
    if missing:
        raise ValueError(f"Labeled key (8-field) missing: {missing}")
    return tuple(filtered[nm] for nm in names8)


def yaml_labeled_key_to_tuple(k: dict) -> tuple:
    """Parse ``entries[i]['key']`` labeled mapping to logical key tuple."""
    if not isinstance(k, dict):
        raise TypeError(f"Entry key must be a mapping, got {type(k)!r}")
    key_t = labeled_key_map_to_tuple(k)
    n = len(key_t)
    if n not in (8, 15, 22):
        raise ValueError(f"Key tuple length must be 8, 15, or 22, got {n}")
    return key_t


def normalize_num_cores_scalar(nc):
    """Whole-number ``float`` (e.g. ``64.0``) → ``int``; pass through ``int`` and other types."""
    if type(nc) is bool:
        return nc
    if isinstance(nc, float) and nc.is_integer():
        return int(nc)
    return nc


def normalize_flat_single_payload(d: dict) -> dict:
    """Copy flat single / ``hybrid.single`` payload and normalize ``num_cores`` after YAML load."""
    if not isinstance(d, dict):
        raise TypeError(f"flat single payload must be a dict, got {type(d)!r}")
    out = dict(d)
    if MASTER_SINGLE_NUM_CORES_KEY in out:
        out[MASTER_SINGLE_NUM_CORES_KEY] = normalize_num_cores_scalar(
            out[MASTER_SINGLE_NUM_CORES_KEY]
        )
    for uk in MASTER_SINGLE_UTIL_STAT_KEYS:
        if uk not in out:
            continue
        v = out[uk]
        if type(v) is bool:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            out[uk] = 0.0
    return out


def is_real_stat_scalar(x) -> bool:
    """True for finite numeric stats: ``int``/``float``/``Integral``/``Real``, excluding ``bool``.

    Accepts NumPy scalar types (``numpy.floating``, etc.) when present without importing NumPy.
    """
    if type(x) is bool:
        return False
    if isinstance(x, numbers.Integral):
        return True
    if isinstance(x, numbers.Real):
        try:
            fx = float(x)
        except (TypeError, ValueError, OverflowError):
            return False
        return math.isfinite(fx)
    if hasattr(x, "dtype"):
        raw_kind = getattr(x.dtype, "kind", None)
        if isinstance(raw_kind, str) and raw_kind in ("i", "u", "f"):
            try:
                fx = float(x)
            except (TypeError, ValueError, OverflowError):
                return False
            return math.isfinite(fx)
    return False
