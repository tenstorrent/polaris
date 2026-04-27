#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for operation-type normalization.

Every tool that maps a profiler OP CODE or a Polaris ``optype`` to a
canonical name should import from here rather than maintaining its own
mapping tables.

Canonical names are **lowercase fine-grained** identifiers that match
the LUT key vocabulary (e.g. ``add``, ``mul``, ``matmul``,
``layernorm``, ``tilize``, ``tilizewithvalpadding``).

For sequence-level comparison (e.g. ``compare_layers.py``) where
several fine-grained ops should be treated as equivalent, use
:func:`to_comparison_group` which applies a deliberate coarsening on
top of the canonical form.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Profiler OP CODE → canonical Polaris layer type
# ---------------------------------------------------------------------------

# Prefix rules: first match wins (longer/specific prefixes before shorter).
# Applied to the profiler base name *after* stripping "DeviceOperation".
PROFILER_PREFIX_RULES: tuple[tuple[str, str], ...] = (
    ("Matmul", "matmul"),
    ("BinaryNg", "eltwise"),
    ("Multiply", "mul"),
    ("Mul", "mul"),
    ("MUL", "mul"),  # Legacy bare uppercase
    ("Add", "add"),
    ("ADD", "add"),  # Legacy bare uppercase
    ("Subtract", "subtract"),
    ("SUB", "subtract"),  # Legacy bare uppercase
    ("Divide", "divide"),
    ("Softmax", "softmax"),
    ("LayerNorm", "layernorm"),
    ("RMSNorm", "rms_norm"),
    ("NLPCreateQKVHeads", "createqkvheads"),
    ("CreateQKVHeadsDeviceOperation", "createqkvheads"),  # Profiler form → canonical
    ("CreateQKVHeads", "createqkvheads"),
    ("NLPConcatHeads", "concatheads"),
    ("ConcatHeads", "concatheads"),
    ("UntilizeWithValUnpadding", "untilizewithunpadding"),  # Polaris variant with "Val"
    ("UntilizeWithUnpadding", "untilizewithunpadding"),
    ("Untilize", "untilize"),
    ("TilizeWithValPadding", "tilizewithvalpadding"),
    ("Tilize", "tilize"),
    ("ReshapeView", "reshape"),
    ("Reshape", "reshape"),
    ("Transpose", "transpose"),
    ("Permute", "permute"),
    ("Concat", "concat"),
    ("Split", "split"),
    ("Embedding", "embedding"),
    ("Reduce", "reduce"),
    ("Pow", "pow"),
    ("Exp", "exp"),
    ("Sqrt", "sqrt"),
    ("Relu", "relu"),
    ("Gelu", "gelu"),
    ("Binary", "eltwise"),
    ("Fold", "fold"),
)

# BinaryOpType::ENUM (uppercase) → canonical layer type.
BINARY_OP_ENUM_TO_CANONICAL: Dict[str, str] = {
    "ADD": "add",
    "MUL": "mul",
    "MULTIPLY": "mul",
    "SUB": "subtract",
    "SUBTRACT": "subtract",
    "DIV": "divide",
    "DIVIDE": "divide",
    "POW": "pow",
    "EXP": "exp",
    "SQRT": "sqrt",
}

# UnaryOpType::ENUM (uppercase) → canonical layer type.
UNARY_OP_ENUM_TO_CANONICAL: Dict[str, str] = {
    "GELU": "gelu",
    "RELU": "relu",
    "RELU6": "relu",
    "EXP": "exp",
    "SQRT": "sqrt",
    "POW": "pow",
    "LOG": "log",
    "LOG2": "log2",
    "LOG10": "log10",
    "ABS": "abs",
    "NEG": "neg",
    "RECIP": "recip",
    "RSQRT": "rsqrt",
    "SIGMOID": "sigmoid",
    "TANH": "tanh",
    "SILU": "silu",
    "COS": "cos",
    "SIN": "sin",
    "IDENTITY": "identity",
}

# Regex patterns for attribute-based op resolution.
_RE_BINARY_OP_TYPE = re.compile(
    r"binary_op_type['\"]?\s*:\s*['\"]?BinaryOpType::(\w+)",
    re.IGNORECASE,
)
_RE_UNARY_OP_TYPE = re.compile(
    r"UnaryOpType::(\w+)",
    re.IGNORECASE,
)

_UNKNOWN_PROFILER_BASES: set[str] = set()

# ---------------------------------------------------------------------------
# Polaris-side synonyms  (optype field from Polaris CSV → canonical)
# ---------------------------------------------------------------------------

POLARIS_SYNONYMS: Dict[str, str] = {
    "layernormalization": "layernorm",
    "reshapeview": "reshape",
    "nlpcreateqkvheads": "createqkvheads",  # Normalize NLP-prefixed form
    "untilizewithvalunpadding": "untilizewithunpadding",  # Polaris uses "Val", profiler uses plain form
}

# ---------------------------------------------------------------------------
# Comparison groups  (canonical → coarsened group for sequence matching)
# ---------------------------------------------------------------------------

COMPARISON_GROUPS: Dict[str, str] = {
    "add": "binary",
    "binaryng": "binary",
    "mul": "binary",
    "subtract": "binary",
    "sub": "binary",
    "eltwise": "binary",
    "tilizewithvalpadding": "tilize",
}

# ---------------------------------------------------------------------------
# STATS display names  (canonical → PascalCase for STATS format output)
# ---------------------------------------------------------------------------

CANONICAL_TO_STATS_DISPLAY: Dict[str, str] = {
    "matmul": "MatMul",
    "add": "Add",
    "mul": "Mul",
    "subtract": "Subtract",
    "divide": "Divide",
    "softmax": "Softmax",
    "layernorm": "LayerNormalization",
    "rms_norm": "RMSNorm",
    "reshape": "Reshape",
    "transpose": "Transpose",
    "permute": "Permute",
    "tilize": "Tilize",
    "tilizewithvalpadding": "TilizeWithValPadding",
    "untilize": "Untilize",
    "untilizewithunpadding": "UntilizeWithUnpadding",
    "concat": "Concat",
    "split": "Split",
    "embedding": "Embedding",
    "reduce": "Reduce",
    "fold": "Fold",
    "createqkvheads": "NLPCreateQKVHeads",  # Display with NLP prefix
    "concatheads": "ConcatHeads",
    "gelu": "Gelu",
    "relu": "Relu",
    "exp": "Exp",
    "pow": "Pow",
    "sqrt": "Sqrt",
    "eltwise": "Eltwise",
}


# ===================================================================
# Public API
# ===================================================================

def _strip_device_operation_suffix(name: str) -> str:
    if name.endswith("DeviceOperation"):
        return name[: -len("DeviceOperation")]
    return name


def _apply_prefix_rules(base: str) -> str:
    """Map a profiler base name (DeviceOperation stripped) via prefix rules."""
    if not base:
        return "other"
    for prefix, layer_type in PROFILER_PREFIX_RULES:
        if base.startswith(prefix):
            return layer_type
    if base not in _UNKNOWN_PROFILER_BASES:
        _UNKNOWN_PROFILER_BASES.add(base)
        logger.warning(
            "Unknown profiler OP base {!r} (after DeviceOperation strip); "
            "using canonical type {!r}",
            base,
            "other",
        )
    return "other"


def _resolve_binary_attrs(attrs: Any) -> Optional[str]:
    """Extract BinaryOpType from attrs (dict or raw string) → canonical, or None."""
    if attrs is None:
        return None
    if isinstance(attrs, dict):
        bot = attrs.get("binary_op_type")
        if bot is not None:
            enum = str(bot).replace("BinaryOpType::", "").strip().upper()
            return BINARY_OP_ENUM_TO_CANONICAL.get(enum)
        raw_str = str(attrs)
    else:
        raw_str = str(attrs)
    m = _RE_BINARY_OP_TYPE.search(raw_str)
    if m:
        return BINARY_OP_ENUM_TO_CANONICAL.get(m.group(1).upper())
    return None


def _resolve_unary_attrs(attrs: Any) -> Optional[str]:
    """Extract UnaryOpType from attrs (dict or raw string) → canonical, or None."""
    if attrs is None:
        return None
    if isinstance(attrs, dict):
        uot = attrs.get("unary_op_type")
        if uot is not None:
            enum = str(uot).replace("UnaryOpType::", "").strip().upper()
            return UNARY_OP_ENUM_TO_CANONICAL.get(enum)
        op_chain = attrs.get("op_chain")
        if op_chain is not None:
            m = _RE_UNARY_OP_TYPE.search(str(op_chain))
            if m:
                return UNARY_OP_ENUM_TO_CANONICAL.get(m.group(1).upper())
        raw_str = str(attrs)
    else:
        raw_str = str(attrs)
    m = _RE_UNARY_OP_TYPE.search(raw_str)
    if m:
        return UNARY_OP_ENUM_TO_CANONICAL.get(m.group(1).upper())
    return None


def normalize_profiler_opcode(opcode: Any, attrs: Any = None) -> str:
    """Map a profiler OP CODE cell (+ optional ATTRIBUTES) to a canonical
    lowercase Polaris layer type.

    Handles:
    - ``DeviceOperation`` suffix stripping
    - ``BinaryNgDeviceOperation`` → attribute-based resolution (add/mul/…)
    - ``UnaryDeviceOperation`` → attribute-based resolution (gelu/relu/…)
    - PascalCase prefix matching for all other ops
    - Numeric OP CODE cells → ``"numeric"``
    - Empty / NaN → ``""``

    Args:
        opcode: The raw ``OP CODE`` cell value (str, int, float, or None).
        attrs: Parsed attributes dict, raw attributes string, or None.

    Returns:
        Canonical lowercase Polaris layer type string.
    """
    if opcode is None:
        return ""

    # Handle numeric OP CODEs
    if isinstance(opcode, bool):
        s = str(opcode).strip()
    elif isinstance(opcode, int):
        return "numeric"
    elif isinstance(opcode, float):
        if opcode != opcode:  # NaN
            return ""
        if opcode.is_integer():
            return "numeric"
        s = str(opcode).strip()
    else:
        s = str(opcode).strip()

    if not s:
        return ""

    # BinaryNg: resolve from attributes if possible
    base = _strip_device_operation_suffix(s)
    if "BinaryNg" in s or base == "BinaryNg":
        resolved = _resolve_binary_attrs(attrs)
        if resolved is not None:
            return resolved
        logger.warning(
            "OP CODE {!r} has no parseable BinaryOpType in attrs; "
            "classifying from OP CODE base name.",
            s,
        )
        return _apply_prefix_rules(base)

    # Unary: resolve from attributes if possible
    if s.strip() in ("UnaryDeviceOperation",) or base == "Unary":
        resolved = _resolve_unary_attrs(attrs)
        if resolved is not None:
            return resolved
        logger.warning(
            "OP CODE {!r} has no parseable UnaryOpType in attrs; "
            "classifying from OP CODE base name.",
            s,
        )
        return _apply_prefix_rules(base)

    return _apply_prefix_rules(base)


def normalize_polaris_optype(optype: str) -> str:
    """Normalize a Polaris CSV ``optype`` field to the canonical form.

    Applies lowercasing and resolves known synonyms (e.g.
    ``layernormalization`` → ``layernorm``).
    """
    key = optype.lower().strip()
    return POLARIS_SYNONYMS.get(key, key)


def to_comparison_group(canonical: str) -> str:
    """Coarsen a canonical op type for sequence-level matching.

    Maps fine-grained types like ``add``, ``mul``, ``subtract`` to the
    group ``binary``; ``tilizewithvalpadding`` to ``tilize``; etc.
    Types without an explicit group mapping pass through unchanged.
    """
    return COMPARISON_GROUPS.get(canonical, canonical)


def canonical_to_stats_display(canonical: str) -> str:
    """Map a canonical op type to its PascalCase STATS display name.

    Falls back to the canonical name with first letter capitalised if
    no explicit mapping exists.
    """
    return CANONICAL_TO_STATS_DISPLAY.get(canonical, canonical.capitalize())
