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
        ├── generated/              # TTNN generated files
        ├── results_typescript.txt  # Command outputs and results
        ├── raw/                    # Tracy raw profiling output
        ├── perf/                   # Tracy performance counter output (full mode only)
        └── trace/                  # Tracy NOC trace output (full mode only)

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
    - 'generated' directory is automatically moved to output directory
"""

import os
import sys
import argparse
import importlib.util
import json
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
    process = subprocess.Popen(
        argv,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
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
        'enable_detailed_buffer_report': True,
        'enable_detailed_tensor_report': False,
        'enable_comparison_mode': False,
    }
    os.environ['TTNN_CONFIG_OVERRIDES'] = json.dumps(environ)

    os.environ['TT_METAL_DEVICE_PROFILER'] = '1'
    os.environ['TT_METAL_PROFILER_SYNC'] = '1'


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

    Returns:
        subprocess.CompletedProcess or None: Result object or None if dryrun
    """
    argv = ['python3', '-m', 'tracy', '-p', '-r', '-v', '--op-support-count=100000', '-o', output_dir]
    if collect_noc_traces:
        argv.append('--collect-noc-traces')
    if collect_perf_counters:
        argv.append('--profiler-capture-perf-counters=all')
    if pytest_mode:
        argv.extend(['-m', 'pytest', command])
    else:
        argv.append(command)

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
        if os.path.exists('generated'):
            logger.error("Directory named 'generated' already exists. Please remove it before running the profiler.")
            return 1
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

    result1 = run_profiler(
        command, raw_dir, pytest_mode, report_name,
        collect_noc_traces=False, collect_perf_counters=False,
        enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
    )
    results = [result1]

    if not args.basic_only:
        if not anyfails(results):
            result2 = run_profiler(
                command, perf_dir, pytest_mode, report_name,
                collect_noc_traces=False, collect_perf_counters=True,
                enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
            )
            results.append(result2)
        if not anyfails(results):
            result3 = run_profiler(
                command, trace_dir, pytest_mode, report_name,
                collect_noc_traces=True, collect_perf_counters=False,
                enable_logging=enable_logging, show_output=show_output, dryrun=args.dryrun,
            )
            results.append(result3)

    if args.dryrun:
        return 0

    try:
        os.rename('generated', os.path.join(output_dir, 'generated'))
    except Exception:
        pass

    output_filename = os.path.join(output_dir, 'results_typescript.txt')
    with open(output_filename, 'w') as summary_file:
        for res in results:
            if res is not None:
                summary_file.write(f'Command: {" ".join(res.args)}\n')
                summary_file.write(f'Return code: {res.returncode}\n')
                summary_file.write(f'Stdout:\n{res.stdout}\n')
                summary_file.write(f'Stderr:\n{res.stderr}\n')
    logger.info('output saved in {}', output_filename)

    if args.cleanup:
        cleanup_directories(output_dir)
    return 0


if __name__ == '__main__':
    exit(main())
