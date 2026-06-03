#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generate a Chakra .et trace from a TTSIM workload graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export TTSIM WorkloadGraph ops to a Chakra execution trace (.et)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # BERT-style export (BasicLLM)\n"
            "  python tools/generate_chakra_et_from_ttsim.py \\\n"
            "    --module BasicLLM.py \\\n"
            "    --instance bert \\\n"
            "    --cfg-kv bs=1 --cfg-kv nL=12 --cfg-kv nH=12 --cfg-kv dE=768 --cfg-kv nW=512 \\\n"
            "    --cfg-kv vocab_sz=30522 \\\n"
            "    --out workloads/chakra/bert_ttsim_tp1_dp1_bs1_seq512/bert_ttsim.0.et\n"
            "\n"
            "  # LLAMA-style export (adjust module/instance/cfg for your workload)\n"
            "  python tools/generate_chakra_et_from_ttsim.py \\\n"
            "    --module BasicLLM.py \\\n"
            "    --instance llama \\\n"
            "    --cfg-kv bs=1 --cfg-kv nL=32 --cfg-kv nH=32 --cfg-kv dE=4096 --cfg-kv nW=1 \\\n"
            "    --out workloads/chakra/llama_ttsim/llama.0.et\n"
        ),
    )
    parser.add_argument(
        "--module",
        default="BasicLLM.py",
        help="Python module under workloads/ (default: BasicLLM.py)",
    )
    parser.add_argument(
        "--instance",
        required=True,
        help="TTSIM model instance name passed to the workload class",
    )
    parser.add_argument(
        "--cfg-json",
        default="",
        help='Workload cfg as JSON, e.g. \'{"layers":[3,4,6,3],"img_height":224,...}\'',
    )
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=None,
        help="Optional path to workload config JSON file",
    )
    parser.add_argument(
        "--cfg-kv",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Optional config override (repeatable). VALUE is parsed as JSON first, "
            "then falls back to string."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .et path",
    )
    parser.add_argument(
        "--trace-name",
        default="",
        help="Optional Chakra trace name (default: instance name)",
    )
    args = parser.parse_args()

    def parse_cfg_kv(items: list[str]) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for item in items:
            if "=" not in item:
                raise ValueError(f"Invalid --cfg-kv '{item}', expected KEY=VALUE")
            key, value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid --cfg-kv '{item}', key cannot be empty")
            value = value.strip()
            try:
                cfg[key] = json.loads(value)
            except json.JSONDecodeError:
                cfg[key] = value
        return cfg

    module_path = str(_REPO / "workloads" / args.module)
    cfg: dict[str, Any] = {}
    if args.cfg_file is not None:
        with open(args.cfg_file, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError("--cfg-file must contain a top-level JSON object")
            cfg.update(loaded)
    if args.cfg_json.strip():
        loaded = json.loads(args.cfg_json)
        if not isinstance(loaded, dict):
            raise ValueError("--cfg-json must be a JSON object")
        cfg.update(loaded)
    if args.cfg_kv:
        cfg.update(parse_cfg_kv(args.cfg_kv))

    out_path = (
        args.out
        if args.out is not None
        else _REPO / "workloads" / "chakra" / f"{args.instance}_ttsim" / f"{args.instance}.0.et"
    )
    trace_name = args.trace_name if args.trace_name else args.instance

    from ttsim.front.chakra.ttsim2chakra import build_ttsim_workload_graph, ttsim_graph_to_et

    graph = build_ttsim_workload_graph(module_path, args.instance, cfg)
    count = ttsim_graph_to_et(graph, out_path, trace_name=trace_name)
    print(f"Wrote {out_path} ({count} COMP_NODEs)")


if __name__ == "__main__":
    main()
