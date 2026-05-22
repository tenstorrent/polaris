#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Compute MAC utilization for each workload model by scanning all STATS opstats.json files.

Definition (for each op):
- Actual MACs = value under op['instrs'].get('mac', 0) (falls back to op['instr_count'] if 'mac' missing)
- Potential MACs = compute_capacity_per_cycle * compute_cycles

Where compute_capacity_per_cycle is inferred per-op as:
    compute_capacity_per_cycle = actual_macs / (compute_cycles * DG_COMPUTE_UTIL_CONSTANT)
    with DG_COMPUTE_UTIL_CONSTANT = 0.6 (see polaris/tools/statattr.py and ttsim/back/device.py)

Hence per-op utilization simplifies to DG_COMPUTE_UTIL_CONSTANT when compute is the bottleneck.
However, for memory- or NA-bounded ops (cycles == 0), we consider only compute cycles component.

We will compute:
- Sum(actual_macs) / Sum(potential_macs) across ops in each file
- Aggregate by model directory (parent of STATS) as well

Outputs a concise table.
"""

import argparse
import json
import math
import os
from typing import Dict, Tuple, List


DG_COMPUTE_UTIL_CONSTANT = 0.6

# Cache for device peak IPC lookups: {(model_dir, devname): mapping[(pipe, instr, prec)] -> peak_macs_per_cycle}
_DEV_IPC_CACHE: Dict[Tuple[str, str], Dict[Tuple[str, str, str], float]] = {}


def iter_opstats_json_files(root_dir: str) -> List[str]:
    results: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).upper() == "STATS":
            for fn in filenames:
                if fn.endswith(".json") and fn.endswith("-opstats.json"):
                    results.append(os.path.join(dirpath, fn))
    return sorted(results)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def extract_ops(doc: Dict) -> List[Dict]:
    ops = doc.get("operatorstats")
    if isinstance(ops, list):
        return ops
    # Some CSV-driven or alternative formats may store in another key; fallback to empty
    return []


def get_actual_macs(op: Dict) -> int:
    # Use total instruction count as proxy for MACs per user request
    # This matches conv ops exactly and is close for others; we will filter to MAC-issuing ops
    return int(op.get("instr_count", 0))


def get_total_operations(op: Dict) -> int:
    # Get total operations from instr_count (includes all instruction types, not just MACs)
    return int(op.get("instr_count", 0))


def get_compute_cycles(op: Dict) -> float:
    # prefer explicit compute_cycles
    cc = op.get("compute_cycles")
    if cc is None:
        return 0.0
    try:
        return float(cc)
    except Exception:
        return 0.0


def get_total_cycles(op: Dict) -> float:
    # total cycles field used in STATS CSV is 'cycles'; JSON has it as well
    cyc = op.get("cycles")
    try:
        return float(cyc) if cyc is not None else 0.0
    except Exception:
        return 0.0


def get_model_dir_from_stats_path(stats_path: str) -> str:
    # stats path: .../<model>/STATS/<file>
    return os.path.basename(os.path.dirname(os.path.dirname(stats_path)))


def load_dev_ipc_table(model_root: str, devname: str) -> Dict[Tuple[str, str, str], float]:
    """Load device peak IPC table from CONFIG/<devname>.json under the given model root.

    Returns mapping: (pipe_lower, instr_lower, precision_lower) -> device_peak_macs_per_cycle
    device_peak_macs_per_cycle = ipgroup.num_units * pipe.num_units * pipe.systolic_depth * insn.tpt[precision]
    """
    cache_key = (model_root, devname)
    if cache_key in _DEV_IPC_CACHE:
        return _DEV_IPC_CACHE[cache_key]

    cfg_path = os.path.join(model_root, "CONFIG", f"{devname}.json")
    if not os.path.isfile(cfg_path):
        # Some repos may store configs elsewhere; return empty to skip potential calc
        _DEV_IPC_CACHE[cache_key] = {}
        return _DEV_IPC_CACHE[cache_key]

    cfg = load_json(cfg_path)
    ipgroups = cfg.get("ipgroups", [])
    compute_groups = [g for g in ipgroups if g.get("iptype") == "compute"]
    if not compute_groups:
        _DEV_IPC_CACHE[cache_key] = {}
        return _DEV_IPC_CACHE[cache_key]

    cg = compute_groups[0]
    dev_num_units = int(cg.get("num_units", 1))
    ipobj = cg.get("ipobj", {})
    pipes = ipobj.get("pipes", [])

    table: Dict[Tuple[str, str, str], float] = {}
    for pipe in pipes:
        pipe_name = str(pipe.get("name", "")).lower()
        pipe_units = int(pipe.get("num_units", 1))
        pipe_sd    = int(pipe.get("systolic_depth", 1) or 1)
        for ins in pipe.get("instructions", []):
            instr_name = str(ins.get("name", "")).lower()
            tpt = ins.get("tpt", {}) or {}
            for prec, val in tpt.items():
                try:
                    prec_l = str(prec).lower()
                    v = float(val)
                except Exception:
                    continue
                peak = dev_num_units * pipe_units * pipe_sd * v
                table[(pipe_name, instr_name, prec_l)] = peak

    _DEV_IPC_CACHE[cache_key] = table
    return table


def op_issues_mac(op: Dict) -> bool:
    instrs = op.get("instrs", {}) or {}
    macs = instrs.get("mac", 0)
    try:
        return int(macs) > 0
    except Exception:
        return False


def compute_file_util(path: str, root_dir: str) -> Tuple[int, float, float, float]:
    """Return (num_counted_ops, sum_actual_macs, sum_potential_macs, sum_total_operations).

    We consider only MAC-issuing ops (instrs.mac > 0). Potential MACs = device_peak_macs_per_cycle * total op cycles.
    Device peak MACs/cycle derived from the model's CONFIG/<devname>.json.
    Total operations include all instruction types across all ops.
    """
    doc = load_json(path)
    ops = extract_ops(doc)

    devname = str(doc.get("name", doc.get("devname", "")))
    # get model dir and load device IPC table
    model_dir_name = get_model_dir_from_stats_path(path)
    model_root = os.path.join(root_dir, model_dir_name)
    dev_ipc_tbl = load_dev_ipc_table(model_root, devname)

    counted_ops = 0
    sum_actual = 0.0
    sum_potential = 0.0
    sum_total_ops = 0.0

    # First pass: calculate total operations across ALL ops (not just MAC-issuing ones)
    for op in ops:
        sum_total_ops += get_total_operations(op)

    # Second pass: calculate MAC-specific metrics for MAC-issuing ops only
    for op in ops:
        if not op_issues_mac(op):
            continue
        cycles_total = get_total_cycles(op)
        if cycles_total <= 0:
            # no time spent → no potential capacity contribution
            continue

        actual_macs = get_actual_macs(op)
        op_pipe = str(op.get("pipe", "")).lower()
        op_prec = str(op.get("precision", "")).lower()

        # prefer matrix.mac throughput; if missing, try any mac entry for the pipe
        peak_ipc = 0.0
        for key in [
            (op_pipe, "mac", op_prec),
            (op_pipe, "mac", "int8"),
            (op_pipe, "mac", "bf16"),
            ("matrix", "mac", op_prec),
        ]:
            if key in dev_ipc_tbl:
                peak_ipc = dev_ipc_tbl[key]
                break

        if peak_ipc <= 0:
            # cannot compute potential for this op without device IPC
            continue

        potential_macs = peak_ipc * cycles_total

        counted_ops += 1
        sum_actual += actual_macs
        sum_potential += potential_macs

    return (counted_ops, float(sum_actual), float(sum_potential), float(sum_total_ops))


def compute_file_record(path: str, root_dir: str) -> Dict:
    """Return a dict with metadata and computed utilization for a single file."""
    doc = load_json(path)

    # Meta fields (robust to naming differences)
    devname = doc.get("devname", "NA")
    freq = doc.get("freq_MHz", doc.get("freq_Mhz", None))
    wlgroup = doc.get("wlgroup", doc.get("wlcls", "NA"))
    wlname = doc.get("wlname", "NA")
    wlinstance = doc.get("wlinstance", "NA")
    batch = doc.get("batch", None)

    return {
        "devname": devname,
        "freq_MHz": freq,
        "wlgroup": wlgroup,
        "wlname": wlname,
        "wlinstance": wlinstance,
        "batch": batch,
        "stat_file": os.path.relpath(path, root_dir),
        # The following fields will be filled by caller after compute_file_util
        "sum_actual_MACs": 0,
        "sum_potential_MACs": 0,
        "utilization": 0.0,
        "total_operations": 0,
    }


def compute_op_level_util(path: str, root_dir: str) -> List[Dict]:
    """Return per-op utilization rows for a given opstats file.

    Returns rows with keys: opname, pipe, precision, instr_count, cycles, actual_macs, potential_macs, utilization
    """
    doc = load_json(path)
    ops = extract_ops(doc)

    devname = str(doc.get("name", doc.get("devname", "")))
    model_dir_name = get_model_dir_from_stats_path(path)
    model_root = os.path.join(root_dir, model_dir_name)
    dev_ipc_tbl = load_dev_ipc_table(model_root, devname)

    out_rows: List[Dict] = []
    for op in ops:
        if not op_issues_mac(op):
            continue
        cycles_total = get_total_cycles(op)
        if cycles_total <= 0:
            continue

        actual_macs = get_actual_macs(op)
        op_pipe = str(op.get("pipe", "")).lower()
        op_prec = str(op.get("precision", "")).lower()
        peak_ipc = 0.0
        for key in [
            (op_pipe, "mac", op_prec),
            (op_pipe, "mac", "int8"),
            (op_pipe, "mac", "bf16"),
            ("matrix", "mac", op_prec),
        ]:
            if key in dev_ipc_tbl:
                peak_ipc = dev_ipc_tbl[key]
                break
        if peak_ipc <= 0:
            continue
        potential_macs = peak_ipc * cycles_total
        util = (actual_macs / potential_macs) if potential_macs > 0 else 0.0

        out_rows.append({
            "opname": op.get("opname", ""),
            "pipe": op.get("pipe", ""),
            "precision": op.get("precision", ""),
            "instr_count": int(op.get("instr_count", 0)),
            "total_operations": int(get_total_operations(op)),
            "cycles": float(op.get("cycles", 0.0)),
            "actual_macs": int(actual_macs),
            "potential_macs": int(potential_macs),
            "utilization": util,
        })

    return out_rows


def write_reports(records: List[Dict], root_dir: str) -> None:
    """Write per-model and per-target CSVs under each model's REPORTS directory.

    Layout:
        <root>/<model>/REPORTS/mac_util_<devname>.csv

    Each CSV contains per-file rows and a TOTAL row.
    """
    # Group by model and devname (target)
    by_model: Dict[str, Dict[str, List[Dict]]] = {}
    for rec in records:
        # resolve absolute path from rel stat_file to find model dir
        abs_path = os.path.join(root_dir, rec["stat_file"])  # .../<model>/STATS/<file>
        model_dir = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
        devname = rec.get("devname", "NA")
        if model_dir not in by_model:
            by_model[model_dir] = {}
        if devname not in by_model[model_dir]:
            by_model[model_dir][devname] = []
        by_model[model_dir][devname].append(rec)

    # Write CSVs
    header = [
        "devname","freq_MHz","wlgroup","wlname","wlinstance","batch",
        "stat_file","sum_actual_MACs","sum_potential_MACs","utilization","total_operations"
    ]

    for model, targets in by_model.items():
        reports_dir = os.path.join(root_dir, model, "REPORTS")
        os.makedirs(reports_dir, exist_ok=True)

        for devname, rows in targets.items():
            out_path = os.path.join(reports_dir, f"mac_util_{devname}.csv")
            # aggregate totals per target
            tot_actual = sum(r["sum_actual_MACs"] for r in rows)
            tot_potential = sum(r["sum_potential_MACs"] for r in rows)
            tot_util = (tot_actual / tot_potential) if tot_potential > 0 else 0.0
            tot_operations = sum(r["total_operations"] for r in rows)

            with open(out_path, "w") as f:
                f.write(",".join(header) + "\n")
                for r in sorted(rows, key=lambda x: (str(x.get("freq_MHz")), x["stat_file"])):
                    f.write(
                        f"{r['devname']},{r.get('freq_MHz','')},{r['wlgroup']},{r['wlname']},{r['wlinstance']},{r.get('batch','')},{r['stat_file']},{r['sum_actual_MACs']},{r['sum_potential_MACs']},{r['utilization']:.6f},{r['total_operations']}\n"
                    )
                # TOTAL row
                f.write(
                    f"TOTAL,ALL,,,{model},,{devname},"  # keep columns aligned but mark aggregation
                )
                # the above columns do not map 1:1; pad with empties to align numeric columns
                # Reconstruct a consistent row with blanks for non-numeric
                # Format: devname,freq_MHz,wlgroup,wlname,wlinstance,batch,stat_file,sum_actual,sum_potential,util
                # We'll rewrite the TOTAL line properly:
            with open(out_path, "a") as f:
                f.write(f"TOTAL,ALL,,,,,TOTAL,{tot_actual},{tot_potential},{tot_util:.6f},{tot_operations}\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate MAC utilization across STATS opstats JSONs")
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "rfi_runs"),
        help="Root directory to search under (default: <repo>/rfi_runs)",
    )
    parser.add_argument(
        "--by_model",
        action="store_true",
        help="Aggregate and print per-model summaries (parent folder of STATS)",
    )
    parser.add_argument(
        "--write_reports",
        action="store_true",
        help="Write per-model per-target CSV reports under each model's REPORTS directory",
    )
    parser.add_argument(
        "--write_op_reports",
        action="store_true",
        help="Write per-op MAC utilization CSVs under each model's REPORTS directory",
    )
    args = parser.parse_args()

    files = iter_opstats_json_files(args.root)
    if not files:
        print(f"No STATS opstats JSON files found under: {args.root}")
        return

    # per-file output
    print("Per-file MAC Utilization and Total Operations:")
    print("file,sum_actual_MACs,sum_potential_MACs,utilization,total_operations")

    # group by model (two levels up: .../<model>/STATS/<file>)
    model_aggr: Dict[str, Tuple[float, float, float, int]] = {}  # (actual, potential, total_ops, count)
    records: List[Dict] = []

    for fpath in files:
        num_ops, s_actual, s_potential, s_total_ops = compute_file_util(fpath, args.root)
        util = (s_actual / s_potential) if s_potential > 0 else 0.0
        print(f"{os.path.relpath(fpath, args.root)},{int(s_actual)},{int(s_potential)},{util:.6f},{int(s_total_ops)}")

        # build detailed record for reporting
        rec = compute_file_record(fpath, args.root)
        rec["sum_actual_MACs"] = int(s_actual)
        rec["sum_potential_MACs"] = int(s_potential)
        rec["utilization"] = util
        rec["total_operations"] = int(s_total_ops)
        records.append(rec)

        if args.by_model:
            # model dir is parent of STATS
            model_dir = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
            if model_dir not in model_aggr:
                model_aggr[model_dir] = (0.0, 0.0, 0.0, 0)
            a, p, t, n = model_aggr[model_dir]
            model_aggr[model_dir] = (a + s_actual, p + s_potential, t + s_total_ops, n + 1)

    if args.by_model:
        print("\nPer-model Aggregated MAC Utilization:")
        print("model,sum_actual_MACs,sum_potential_MACs,utilization,total_operations,num_files")
        for model, (a, p, t, n) in sorted(model_aggr.items()):
            util = (a / p) if p > 0 else 0.0
            print(f"{model},{int(a)},{int(p)},{util:.6f},{int(t)},{n}")

    if args.write_reports:
        write_reports(records, args.root)

    if args.write_op_reports:
        # For each file, write per-op CSV next to model reports
        for rec in records:
            rel = rec["stat_file"]  # <model>/STATS/<file>.json
            abs_path = os.path.join(args.root, rel)
            model = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
            dev = rec.get("devname", "NA")
            reports_dir = os.path.join(args.root, model, "REPORTS")
            os.makedirs(reports_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(abs_path))[0]
            out_csv = os.path.join(reports_dir, f"op_mac_util_{dev}_{stem}.csv")
            rows = compute_op_level_util(abs_path, args.root)
            if rows:
                header = ["opname","pipe","precision","instr_count","total_operations","cycles","actual_macs","potential_macs","utilization"]
                with open(out_csv, "w") as f:
                    f.write(",".join(header) + "\n")
                    for r in rows:
                        f.write(f"{r['opname']},{r['pipe']},{r['precision']},{r['instr_count']},{r['total_operations']},{r['cycles']},{r['actual_macs']},{r['potential_macs']},{r['utilization']:.6f}\n")


if __name__ == "__main__":
    main()


