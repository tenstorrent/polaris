#!/usr/bin/env python
import os
from glob import glob
import argparse
import re
from itertools import product
import subprocess
from typing import Any
from shutil import rmtree

ODIR = '__RUN_TESTS'
LOGD = f'{ODIR}/logs'
OptionalString = str | None


def prepare_commands(condaenvprefix: OptionalString) -> list[str]:
    # Knobs for coverage, pytest (as part of coverage run), mypy are picked up from
    # pyproject.toml. These knobs are NOT replicated here, to avoid inconsistency
    commands = [
        f'{condaenvprefix} coverage run -m pytest && coverage report && coverage html',
        f'{condaenvprefix} mypy ./ ',
    ]
    # NOTE: We do NOT run workloads/*.py, since each such workload is now exercised
    #       by a corresponding test in tests/test_workloads. Any workloads added to
    #       workloads/ directory, which need to be tested, should follow a pattern
    #       to other workloads and their test equivalents

    return commands


def prepare_commands_run_all_tests(condaenvprefix: OptionalString) -> list[str]:
    script     = 'polaris.py'
    wlspec     = 'config/all_workloads.yaml'
    arspec     = 'config/all_archs.yaml'
    wlmspec    = 'config/wl2archmapping.yaml'
    filterarch = 'A100,Q1_A1'
    filterwli  = 'gpt_nano'
    option1 = ["", ' --instr_profile']
    option2 = ["", ' --dump_ttsim_onnx']
    option3 = ["", ' --frequency 1000 1200 100']
    option4 = ["", ' --batchsize 1 8 2']

    ALL_EXPS = product(option1, option2, option3, option4)

    commands = []
    cmd  = f"{condaenvprefix} python {script} -w {wlspec} -a {arspec} -m {wlmspec} "
    cmd += f"--filterarch {filterarch} --filterwli {filterwli} "
    cmd += f"-o {ODIR} "

    for exp_no, exp in enumerate(ALL_EXPS):
        exp_str    = "".join(exp)
        study_name = f"study_{exp_no+1}"
        command    = f"{cmd} -s {study_name} {exp_str} --log_level debug"
        commands.append(command)

    return commands

def prepare_commands_parse_all_wlyaml(condaenvprefix: OptionalString, dryrun: bool = True) -> list[str]:
    script     = 'polaris.py'
    arspec     = 'config/all_archs.yaml'
    wlmspec    = 'config/wl2archmapping.yaml'
    wlspecs    = [f'config/{f}.yaml' for f in ['all_workloads' ]]

    commands = []
    cmd  = f"{condaenvprefix} python {script} -a {arspec} -m {wlmspec} -o {ODIR} -s __dummy"
    if dryrun:
        cmd += ' --dryrun'

    for wlspec in wlspecs:
        command    = f"{cmd} -w {wlspec}"
        commands.append(command)

    return commands


def run_a_job(cmd: str, outputfilename:str)->subprocess.CompletedProcess:
    with open(outputfilename, 'w') as cmdfout:
        print('Command:', cmd, file=cmdfout)
        print('', file=cmdfout)
        cmdfout.flush()
        return subprocess.run(cmd, shell=True, stdout=cmdfout, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', '-e', dest='condaenv', help='Conda environment name or path to be used for tests')
    parser.add_argument('--stop', '-x', action='store_true', help='Stop tests on first failure')
    parser.add_argument('--filter', help='filter commands')
    parser.add_argument('--dryrun', '-n', action='store_true', help='Show but do not execute the commands')
    args = parser.parse_args()

    condaenvprefix: OptionalString = ''
    if args.condaenv is not None:
        condabase: str = os.popen('conda info --base').read().strip()
        condaenvprefix = f'source {condabase}/etc/profile.d/conda.sh && conda activate {args.condaenv} && '

    commands = prepare_commands(condaenvprefix)
    commands.extend(prepare_commands_run_all_tests(condaenvprefix))
    commands.extend(prepare_commands_parse_all_wlyaml(condaenvprefix, dryrun=True))
    commands.extend(prepare_commands_parse_all_wlyaml(condaenvprefix, dryrun=False))
    if args.filter:
        commands = [cmd for cmd in commands if re.search(args.filter, cmd)]

    if not args.dryrun:
        rmtree(ODIR, ignore_errors=True)
        os.makedirs(ODIR, exist_ok=True)
        os.makedirs(LOGD, exist_ok=True)

    num_failures: int = 0
    results: list[dict[str, Any]] = [dict() for _ in range(len(commands))]
    for cmdno, cmd in enumerate(commands):
        study_match = re.search('-s (study_[0-9]+)', cmd)
        if study_match:
            log_file = os.path.join(LOGD, study_match.group(1)+'.log')
        else:
            log_file = os.path.join(LOGD, f'checkin_test_{cmdno+1}.log')
        print(f'#{cmdno+1}/{len(commands)}: {cmd} -> {log_file}')
        if args.dryrun:
            continue
        cmdret     = run_a_job(cmd, log_file)
        results[cmdno] = {'id': f'{cmdno:2d}-{log_file}',
                          'returncode': cmdret.returncode,
                          'status': 'PASS' if cmdret.returncode == 0 else 'FAIL',
                          }
        if cmdret.returncode != 0:
            if args.stop:
                print(f'error: {cmd} failed with exit code {cmdret.returncode}')
                print('checkin tests failed')
                return cmdret.returncode
            num_failures += 1

    if args.dryrun:
        return 0
    mf = max([len(x['id']) for x in results])

    for res in ['PASS', 'FAIL']:
        if res == 'FAIL' and num_failures:
            print('-- Failed commands')
        for cmdno, result in enumerate(results):
            if result['status'] != res:
                continue
            print(f"{result["id"]:{mf}s}  RESULT= {res}")

    if num_failures == 0:
        errorlines = os.popen(f'grep ERROR: {LOGD}/*.log').readlines()
        if errorlines:
            print('Warning: log lines containing ERROR: messages')
            for l in errorlines:
                l = l.rstrip()
                print('\t' + l)
            print('--------------------------------------------------------------------------')
            print('Warning: log lines containing ERROR: messages though runs exit with code 0')
            print('--------------------------------------------------------------------------')

    if num_failures:
        print(f'{num_failures} of {len(commands)} failed')
    else:
        print('checkin tests successful')
    return num_failures

if __name__ == '__main__':
    exit(main())
