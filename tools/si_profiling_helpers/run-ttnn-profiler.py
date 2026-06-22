#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTNN Profiler Runner

This script runs the TTNN profiler with multiple profiling modes using tracy.
It supports basic profiling, NOC trace collection, and performance counter capture.

Usage:
    python3 run-ttnn-profiler.py --command "<test_command>" \
        --report-name <name> --output-dir <dir> [options]

Examples:
    # Basic profiling only
    python3 run-ttnn-profiler.py \
        --command "test_ttnn_functional_vit.py" \
        --report-name vit_test \
        --output-dir ./profiler_output \
        --basic-only

    # Full profiling (raw + NOC traces + perf counters)
    python3 run-ttnn-profiler.py \
        --command "test_ttnn_functional_vit.py" \
        --report-name vit_full \
        --output-dir ./profiler_output

    # Pytest mode with cleanup
    python3 run-ttnn-profiler.py \
        --command "tests/ttnn/unit_tests/test_model.py::test_forward" \
        --pytest \
        --report-name model_test \
        --output-dir ./profiler_output \
        --cleanup

    # Dry run to see commands without executing
    python3 run-ttnn-profiler.py \
        --command "test.py" \
        --report-name test \
        --output-dir ./out \
        --dryrun

Required Arguments:
    --command       Test script or command to run under the profiler
    --report-name   Name for the profiling report
    --output-dir    Directory to save profiler outputs (must not exist)

Optional Arguments:
    --pytest            Run command with pytest (-m pytest prefix)
    --basic-only        Skip NOC traces and performance counter collection
    --op-support-count  Maximum number of ops tracy will profile (default: 100000)
    --disable-logging   Disable TTNN logging (enable_logging=False in config overrides)
                        Some workloads like VGG have device synchronize calls that
                        fail when enable_logging is true. Use this flag for such cases.
    --show-output       Show stdout/stderr in real-time during command execution.
                        Useful for monitoring progress and debugging hanging commands.
                        By default, output is only captured and saved to results file.
    --cleanup           Remove npe_viz and .logs directories after profiling.
                        These generated directories can be very large; use this flag
                        if their contents are not needed for analysis.
    --dryrun, -n        Show commands without executing them

Output Structure:
    <output-dir>/
        ├── <pass>_results_typescript.txt  # Per-pass output: raw_/perf_/trace_/merge_
        ├── merged_ops_<RUNID>.csv  # 3-CSV merge (full mode only; RUNID = output-dir name)
        ├── raw/                    # Tracy raw profiling output (.logs/, reports/, generated/)
        ├── perf/                   # Tracy performance counter output (full mode only)
        └── trace/                  # Tracy NOC trace output (full mode only)

    Each pass dir is self-contained: TT_METAL_LOGS_PATH and TT_METAL_PROFILER_DIR
    are pinned to it (see run_profiler), so the bulk of tt-metal's 'generated/' output
    lands under the pass dir rather than the cwd. A few paths don't honor those env
    vars (e.g. generated/fabric, generated/test_reports) and may still appear in the
    cwd — small, idempotent, low collision risk (see presets/README.md). This lets two
    runs execute concurrently from the same (e.g. shared/NFS) tt-metal checkout.

Prerequisites:
    - NPE tools must be on PATH (source tt-npe/ENV_SETUP)
    - Tracy profiler module must be available
    - loguru package must be installed

Notes:
    - The script runs up to 3 profiling passes:
      1. Raw profiling with TTNN config overrides
      2. Performance counter collection (unless --basic-only)
      3. NOC trace collection (unless --basic-only)
    - Execution stops if any pass fails
    - All tt-metal artifacts land directly under each pass dir (no cwd/generated);
      concurrent runs from one directory therefore do not collide
    - After a successful full run (raw+perf+trace), the three CSVs are merged into
      merged_ops_<RUNID>.csv on this node, so only it (+ hw_id.json) need be copied off the
      board. The per-board DRAM peak BW is resolved from an internal interim table
      keyed on tt-smi board_type (interim/placeholder values pending the DRAM-BW
      review). Skipped for --basic-only or when the board can't be resolved.
"""

import os
import sys
import argparse
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import threading

from loguru import logger

def is_npe_on_path() -> bool:
    """Check if NPE tools are available on the Python path."""
    return importlib.util.find_spec('npe_analyze_noc_trace_dir') is not None


def run_and_capture(argv: list[str], show_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Execute a command and capture its output.

    Args:
        argv (list[str]): Command and arguments as a list (shell=False).
        show_output (bool): If True, display stdout/stderr in real-time while capturing.
                           If False (default), only capture output silently.

    Returns:
        subprocess.CompletedProcess: Result object with returncode, stdout, stderr
    """
    try:
        process = subprocess.Popen(
            argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        # e.g. tt-smi not installed; surface as a failed run (returncode 127) rather
        # than crashing, so callers can detect it via anyfails() and exit non-zero.
        logger.error(f'command not found: {argv[0]!r} ({e})')
        return subprocess.CompletedProcess(argv, returncode=127, stdout='', stderr=str(e))
    proc_stdout = process.stdout
    proc_stderr = process.stderr
    assert proc_stdout is not None
    assert proc_stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        for line in iter(proc_stdout.readline, ''):
            if line:
                if show_output:
                    print(line, end='')
                    sys.stdout.flush()
                stdout_lines.append(line)
        proc_stdout.close()

    def read_stderr() -> None:
        for line in iter(proc_stderr.readline, ''):
            if line:
                if show_output:
                    print(line, end='', file=sys.stderr)
                    sys.stderr.flush()
                stderr_lines.append(line)
        proc_stderr.close()

    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)

    stdout_thread.start()
    stderr_thread.start()

    process.wait()

    stdout_thread.join()
    stderr_thread.join()

    result = subprocess.CompletedProcess(
        args=argv,
        returncode=process.returncode,
        stdout=''.join(stdout_lines),
        stderr=''.join(stderr_lines),
    )

    if result.returncode == 0:
        logger.info('Command ran successfully.')
    else:
        logger.error('Command failed to run.')

    return result


def setup_environment(report_name: str, enable_logging: bool = True) -> None:
    """Configure TTNN environment variables for profiling.

    Sets up TTNN_CONFIG_OVERRIDES with profiling-friendly settings and
    enables device profiler environment variables. Also sets TT_METAL_HOME
    and PYTHONPATH based on the current working directory.

    Note: This script is expected to be run from the tt-metal home directory.

    Args:
        report_name (str): Name for the profiling report
        enable_logging (bool): Whether to enable TTNN logging. Defaults to True.
            Set to False for workloads like VGG that have device synchronize
            calls which fail when logging is enabled.
    """
    metal_home = os.getcwd()
    os.environ['TT_METAL_HOME'] = metal_home

    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if current_pythonpath:
        os.environ['PYTHONPATH'] = f'{metal_home}:{current_pythonpath}'
    else:
        os.environ['PYTHONPATH'] = metal_home

    environ = {
        'enable_fast_runtime_mode': False,
        'enable_logging': enable_logging,
        'report_name': report_name,
        'enable_graph_report': False,
        'enable_detailed_buffer_report': enable_logging,
        'enable_detailed_tensor_report': False,
        'enable_comparison_mode': False,
    }
    os.environ['TTNN_CONFIG_OVERRIDES'] = json.dumps(environ)

    os.environ['TT_METAL_DEVICE_PROFILER'] = '1'
    os.environ['TT_METAL_PROFILER_SYNC'] = '1'
    # Do NOT enable TT_METAL_PROFILER_CPP_POST_PROCESS here. The C++ post-process path emits the
    # leaner cpp_device_perf_report.csv and explicitly drops device_analysis_types (see tt-metal
    # process_ops_logs.py: "device_analysis_types is not supported when using
    # cpp_device_perf_report.csv; ignoring option"). That analysis is what produces the per-op
    # FPU/SFPU Util Median columns. ops_perf_three_csv_merge.py classifies the perf pass as the
    # "fpu" CSV via those columns, so enabling cpp post-process makes the merge fail with
    # "Expected exactly one fpu CSV, found 0". The legacy device-log parser (import_log_run_stats)
    # is the required path here; the "cpp_device_perf_report.csv not found" warning is expected.
    # NOTE: capture the workload's real command, trace included — do NOT add --disable_trace.
    # With trace on, an op is logged across capture + replay passes (capture = timed; replay =
    # same GLOBAL CALL COUNT, and for some demos empty-duration "shadow" rows). The 3-CSV merge
    # deduplicates these via its op-code-period iteration detection, keeping the refrun faithful
    # to how the model runs on hardware. Disabling trace would change the measured execution
    # path, so it is not used here or in the presets.


def run_profiler(
    command: str,
    output_dir: str,
    pytest_mode: bool,
    report_name: str,
    collect_noc_traces: bool = False,
    collect_perf_counters: bool = False,
    enable_logging: bool = True,
    show_output: bool = False,
    dryrun: bool = False,
    op_support_count: int = 100000,
) -> subprocess.CompletedProcess[str] | None:
    """Run the TTNN profiler with specified options.

    Args:
        command (str): Test script or command to run under the profiler
        output_dir (str): Directory to save profiler output for this pass
        pytest_mode (bool): If True, prefix command with '-m pytest'
        report_name (str): Name for the profiling report
        collect_noc_traces (bool): If True, collect NOC traces
        collect_perf_counters (bool): If True, collect performance counters
        enable_logging (bool): Whether to enable TTNN logging in config overrides
        show_output (bool): If True, display command stdout/stderr in real-time
        dryrun (bool): If True, show commands without executing
        op_support_count (int): Maximum number of ops tracy will profile

    Returns:
        subprocess.CompletedProcess or None: Result object or None if dryrun
    """
    # Use the current interpreter (sys.executable) so tracy runs under the same
    # venv/python as this wrapper, rather than relying on whatever 'python3' is on PATH.
    argv = [sys.executable, '-m', 'tracy', '-p', '-r', '-v',
            f'--op-support-count={op_support_count}', '-o', output_dir]
    if collect_noc_traces:
        argv.append('--collect-noc-traces')
    if collect_perf_counters:
        argv.append('--profiler-capture-perf-counters=fpu')
    if pytest_mode:
        argv.extend(['-m', 'pytest'] + shlex.split(command))
    else:
        argv.extend(shlex.split(command))

    logger.info(f'{argv=}')
    if collect_noc_traces:
        mode = 'trace'
    elif collect_perf_counters:
        mode = 'perf-counter'
    else:
        mode = 'raw'

    logger.info(f'Running command in {mode} mode: {" ".join(argv)}')

    if mode == 'raw':
        setup_environment(report_name, enable_logging)
    else:
        if 'TTNN_CONFIG_OVERRIDES' in os.environ:
            del os.environ['TTNN_CONFIG_OVERRIDES']

    # Pin every tt-metal artifact root to THIS pass's output dir so that nothing
    # is written into the current working directory.
    #
    # Motivation: tt-metal anchors its artifact trees to the cwd by default --
    #   * TT_METAL_LOGS_PATH defaults to the current working directory, and the
    #     'generated/{reports,inspector}/' tree hangs off it (rtoptions);
    #   * TT_METAL_PROFILER_DIR defaults to TT_METAL_HOME/generated/profiler, and
    #     TT_METAL_HOME is itself forced to cwd in setup_environment().
    # So two profiling runs launched from the SAME directory (a shared/NFS
    # tt-metal checkout, or two boards driven from one host) both create
    # cwd/generated and clobber each other's logs -- and the old guard in main()
    # even hard-aborted the second run. Pointing these env vars at the unique
    # per-pass output dir makes each run fully self-contained, so concurrent runs
    # from one directory coexist. This supersedes the previous "let tt-metal
    # create generated/ in cwd, then os.rename it into the output dir afterwards"
    # approach (which only worked for one run at a time).
    os.environ['TT_METAL_LOGS_PATH'] = output_dir
    os.environ['TT_METAL_PROFILER_DIR'] = output_dir

    if dryrun:
        return None

    result = run_and_capture(argv, show_output=show_output)
    if result.returncode == 0:
        logger.info(f'Profiler {mode} mode output saved to: {output_dir} with report name: {report_name}')

    return result


def anyfails(results: list[subprocess.CompletedProcess[str] | None]) -> bool:
    """Check if any profiler run failed."""
    for result in results:
        if result is None:
            continue
        if result.returncode != 0:
            return True
    return False


def cleanup_directories(output_dir: str) -> None:
    """Remove npe_viz and .logs directories and profile_log_device.csv files recursively from output directory."""
    logger.info('Performing cleanup...')
    dirs_to_remove = ['npe_viz', '.logs']
    files_to_remove = ['profile_log_device.csv']
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for dir_name in dirs:
            if dir_name in dirs_to_remove:
                dir_path = os.path.join(root, dir_name)
                logger.info(f'Removing directory: {dir_path}')
                shutil.rmtree(dir_path)
        for file_name in files:
            if file_name in files_to_remove:
                file_path = os.path.join(root, file_name)
                logger.info(f'Removing file: {file_path}')
                os.remove(file_path)
    logger.info('Cleanup completed.')


def write_result_file(output_dir: str, mode: str, res: subprocess.CompletedProcess[str] | None) -> None:
    """Write a single profiling result to <mode>_results_typescript.txt."""
    if res is None:
        return
    output_filename = os.path.join(output_dir, f'{mode}_results_typescript.txt')
    with open(output_filename, 'w') as summary_file:
        summary_file.write(f'Command: {" ".join(res.args)}\n')
        summary_file.write(f'Return code: {res.returncode}\n')
        summary_file.write(f'Stdout:\n{res.stdout}\n')
        summary_file.write(f'Stderr:\n{res.stderr}\n')
        summary_file.write(f'{"="*120}\n')
    logger.info('output saved in {}', output_filename)


# Interim board_type -> DRAM peak bandwidth (GB/s), used to merge a capture on the
# hardware node without a CLI value. These are PLACEHOLDER numbers pending the
# DRAM-BW review resolution: they feed only the util columns of merged_ops.csv (the
# merge structure is unaffected). Keyed by tt-smi board_type PREFIX so revision
# suffixes (e.g. "n150 L") still match. When that review lands, this table should
# move into ops_perf_three_csv_merge.py as a proper per-chip default (answering the
# "set defaults per chip" review comment) and be replaced with validated values.
_INTERIM_BOARD_DRAM_PEAK_BW_GBPS: dict[str, float] = {
    'n150': 288.0,   # Wormhole  (PLACEHOLDER)
    'n300': 288.0,   # Wormhole  (PLACEHOLDER)
    'p100': 448.0,   # Blackhole (PLACEHOLDER)
    'p150': 448.0,   # Blackhole (PLACEHOLDER)
}


def detect_board_type() -> str | None:
    """Best-effort tt-smi board_type (e.g. 'n150', 'p100a'); None if unavailable."""
    try:
        res = subprocess.run(['tt-smi', '-s'], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    m = re.search(r'"board_type"\s*:\s*"([^"]+)"', res.stdout)
    return m.group(1).strip() if m else None


def resolve_interim_dram_peak_bw(board_type: str) -> float | None:
    """Map a tt-smi board_type to the interim DRAM peak BW (GB/s) via prefix match."""
    key = board_type.strip().lower()
    for prefix, bw in _INTERIM_BOARD_DRAM_PEAK_BW_GBPS.items():
        if key.startswith(prefix):
            return bw
    return None


def run_merge(output_dir: str, dram_peak_bw_gbps: float, show_output: bool = False) -> subprocess.CompletedProcess[str] | None:
    """Consolidate the three per-pass CSVs (raw/perf/trace) under output_dir into
    output_dir/merged_ops_<RUNID>.csv (RUNID = output_dir basename), ON THE HARDWARE
    NODE, right after a successful capture.

    The merge tool ships alongside this wrapper (same dir), so it is part of the
    rsynced si_profiling_helpers bundle and its only third-party dep (loguru) is
    already in the tt-metal run env. Merging here means only merged_ops_<RUNID>.csv
    (plus hw_id.json) need be copied off the board, not three raw CSVs. The follow-on
    compare-vs-refrun step needs polaris/ttsim and stays off-device.
    """
    merge_tool = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ops_perf_three_csv_merge.py')
    if not os.path.exists(merge_tool):
        logger.error(f'merge tool not found at {merge_tool}; skipping merge')
        return None
    # Name the merged CSV after the run dir (RUNID = output_dir basename) so per-run
    # merges stay distinct when copied side by side. The merge tool stays generic
    # (its own default is merged_ops.csv); the RUNID-specific name is chosen here,
    # where the run is known and the merge is invoked.
    merged_csv = os.path.join(output_dir, f'merged_ops_{os.path.basename(output_dir)}.csv')
    argv = [sys.executable, merge_tool, '--input-dir', output_dir,
            '--dram-peak-bw-gbps', str(dram_peak_bw_gbps), '--output', merged_csv]
    logger.info(f'merging three CSVs in {output_dir} (--dram-peak-bw-gbps {dram_peak_bw_gbps})')
    result = run_and_capture(argv, show_output=show_output)
    if result.returncode == 0:
        logger.info(f'{os.path.basename(merged_csv)} written under {output_dir}')
    else:
        logger.error('merge step failed (see output above)')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the TTNN profiler')
    parser.add_argument('--command', type=str, required=True, help='Test script or command to run under the profiler')
    parser.add_argument('--pytest', action='store_true', help='Run the profiler in pytest mode')
    parser.add_argument('--report-name', type=str, required=True, help='Name of the profiler report')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the profiler output')
    parser.add_argument(
        '--basic-only', action='store_true', help='Only run basic profiling (skip NOC traces and performance counters)'
    )
    parser.add_argument(
        '--disable-logging',
        action='store_true',
        help='Disable TTNN logging (enable_logging=False). Use for workloads like VGG with device synchronize calls that fail when logging is enabled. Defaults to logging enabled.',
    )
    parser.add_argument(
        '--show-output',
        action='store_true',
        help='Show stdout/stderr in real-time during command execution. Useful for monitoring progress and debugging hanging commands.',
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Remove npe_viz and .logs directories from output directory after run. These directories can be very large; use this flag if their contents are not needed.',
    )
    parser.add_argument('--dryrun', '-n', action='store_true', help='show but do not execute commands')
    parser.add_argument(
        '--op-support-count',
        type=int,
        default=100000,
        help='Maximum number of ops tracy will profile (default: 100000)',
    )
    args = parser.parse_args()

    command = args.command
    output_dir = os.path.realpath(args.output_dir)
    pytest_mode = args.pytest
    report_name = args.report_name
    enable_logging = not args.disable_logging
    show_output = args.show_output
    logger.remove()
    logger.add(sys.stdout, level='INFO')

    if not args.dryrun:
        if os.path.exists(output_dir):
            logger.error(f'Output directory {output_dir} already exists. Please specify a new directory.')
            return 1
        # NOTE: no longer guard against a pre-existing cwd/'generated' dir -- each
        # pass now pins TT_METAL_LOGS_PATH / TT_METAL_PROFILER_DIR to its own
        # output dir (see run_profiler), so tt-metal writes nothing into cwd and
        # concurrent runs from the same directory no longer collide.
        if not args.basic_only and not is_npe_on_path():
            logger.error('NPE not on path; source tt-npe/ENV_SETUP')
            return 1
        os.makedirs(output_dir)

    if any([word in args.command for word in ['tests/', 'pytest']]) and not args.pytest:
        logger.error('possibly using pytest without providing --pytest knob to this script')

    if 'vgg' in args.command.lower() and enable_logging:
        logger.error('possibly running VGG without disabling logging')
        os.system('sleep 5')

    raw_dir = os.path.join(output_dir, 'raw')
    perf_dir = os.path.join(output_dir, 'perf')
    trace_dir = os.path.join(output_dir, 'trace')

    results: list[subprocess.CompletedProcess[str] | None] = []

    if not args.dryrun:
        results.append(run_and_capture(['tt-smi', '-r'], show_output=args.show_output))

    if not anyfails(results):
        logger.info('board reset successful, starting profiling runs')
        result1 = run_profiler(
            command, raw_dir, pytest_mode, report_name,
            collect_noc_traces=False, collect_perf_counters=False,
            enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
            op_support_count=args.op_support_count,
        )
        results.append(result1)
        write_result_file(output_dir, 'raw', result1)

    if not args.basic_only:
        if not anyfails(results):
            result2 = run_profiler(
                command, perf_dir, pytest_mode, report_name,
                collect_noc_traces=False, collect_perf_counters=True,
                enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
                op_support_count=args.op_support_count,
            )
            results.append(result2)
            write_result_file(output_dir, 'perf', result2)
        if not anyfails(results):
            result3 = run_profiler(
                command, trace_dir, pytest_mode, report_name,
                collect_noc_traces=True, collect_perf_counters=False,
                enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
                op_support_count=args.op_support_count,
            )
            results.append(result3)
            write_result_file(output_dir, 'trace', result3)

    if args.dryrun:
        return 0

    # On-device merge: produce merged_ops_<RUNID>.csv as the final artifact of a successful
    # full capture, so only it (+ hw_id.json) need be copied off the board. The
    # per-board DRAM peak BW is resolved from an INTERNAL interim table keyed on the
    # tt-smi board_type -- no CLI value needed. Needs raw+perf+trace, so skipped for
    # --basic-only / any failed pass; skipped (non-fatal) when the board is unknown
    # or tt-smi is unavailable (e.g. off-device) -- merge manually in that case.
    # Runs BEFORE cleanup so the source CSVs still exist.
    if args.basic_only:
        logger.info('--basic-only: skipping merge (needs raw+perf+trace).')
    elif anyfails(results):
        logger.warning('a profiling pass failed; skipping merge.')
    else:
        board = detect_board_type()
        bw = resolve_interim_dram_peak_bw(board) if board else None
        if bw is None:
            logger.warning(
                f'skipping on-device merge: could not resolve DRAM peak BW for board_type={board!r} '
                f'(known prefixes: {sorted(_INTERIM_BOARD_DRAM_PEAK_BW_GBPS)}). Merge manually: '
                f'ops_perf_three_csv_merge.py --input-dir {output_dir} --dram-peak-bw-gbps <gbps>.'
            )
        else:
            logger.info(f'on-device merge: board_type={board} -> interim DRAM peak BW {bw} GB/s (PLACEHOLDER)')
            merge_res = run_merge(output_dir, bw, show_output=show_output)
            results.append(merge_res)
            write_result_file(output_dir, 'merge', merge_res)

    if args.cleanup:
        cleanup_directories(output_dir)

    # Propagate failures (board reset, any profiling pass, or the merge) as a non-zero
    # exit so automation/presets can detect them. Per-pass result files are already
    # written above, so a failed run still leaves its captured output on disk.
    if anyfails(results):
        logger.error('one or more steps failed; exiting non-zero (see per-pass results files).')
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
