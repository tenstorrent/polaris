#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import subprocess
import yaml
from typing import Any
from pathlib import Path

def run_polproj(runcfg_dict: dict[str, Any], runcfgfile_path: Path) -> None:
    """
    Run the polproj script with the given paths.
    """
    with open(runcfgfile_path, 'w') as f:
        yaml.dump(runcfg_dict, f)
    polproj_cmd = ['python', 'polproj.py', '--config', runcfgfile_path.as_posix()]
    ret = subprocess.run(polproj_cmd, check=True)
    assert ret.returncode == 0, f"polproj script {polproj_cmd} failed with return code {ret.returncode}"


def run_projcmp(dir1: Path, dir2: Path, output: Path, extra_args: None|list[str]=None, expected_return: int = 0) -> None:
    """
    Run the compare_projections script with the given directories and output path.
    """
    if extra_args is None:
        extra_args = []
    compare_cmd = ['python', 'tools/compare_projections.py',
                   '--dir1', dir1.as_posix(),
                   '--dir2', dir2.as_posix(),
                   '--output', output.as_posix()
                   ] + extra_args
    ret = subprocess.run(compare_cmd, check=False)
    assert ret.returncode == expected_return, f"compare_projections script {compare_cmd} exited with return code {ret.returncode} != expected {expected_return}"


def test_compare_proj(tmp_path_factory):
    """
    Test the comparison of two project files using the polproj script.
    """
    polproj_path = 'tools/polproj.py'
    arch_cfg = 'config/all_archs.yaml'
    wl_cfg = 'config/mlperf_inference.yaml'
    wlmap_cfg = 'config/wl2archmapping.yaml'
    archs = 'A100'

    tmpdirnames = ['run1', 'run2', 'run3', 'comparison1', 'comparison2', 'comparison3', 'temp']
    tmpdir = {x: tmp_path_factory.mktemp(x) for x in tmpdirnames}
    for tmp in tmpdir.values():
        os.makedirs(tmp, exist_ok=True)

    study_name = 'projcmp'

    runcfg1_dict = {
        'title': 'Test Projection Comparison',
        'study': study_name,
        'odir': tmpdir['run1'].as_posix(),
        'wlspec': wl_cfg,
        'archspec': arch_cfg,
        'wlmapspec': wlmap_cfg,
        'filterarch': ",".join(archs),
        'filterwli': 'bert_large_b1024',
        'filterarch': 'A100'
    }
    run_polproj(runcfg1_dict, tmpdir['temp'] / 'runcfg1.yaml')
    with open(arch_cfg) as fin:
        arch_dict = yaml.safe_load(fin)

    nvidia_entry = next((entry for ndx, entry in enumerate(arch_dict['packages']) if 'nvidia' in entry['name'].lower()), None)
    assert nvidia_entry is not None, "NVIDIA entry not found in architecture configuration"
    a100_entry = next((entry for ndx, entry in enumerate(nvidia_entry['instances']) if 'a100' in entry['name'].lower()), None)
    assert a100_entry is not None, "A100 entry not found in NVIDIA architecture configuration"
    compute_entry = next((entry for ndx, entry in enumerate(a100_entry['ipgroups']) if entry['iptype'].lower() == 'compute'), None)
    assert compute_entry is not None, "Compute entry not found in A100 architecture configuration"
    for override_name in compute_entry['ip_overrides']:
        if override_name.lower().endswith('freq_mhz'):
            compute_entry['ip_overrides'][override_name] -= 100 # Decrease frequency by 100 MHz for testing
    arch_cfg2_path = tmpdir['temp'] / 'all_archs_2.yaml'
    with open(arch_cfg2_path, 'w') as fout:
        yaml.dump(arch_dict, fout)
    runcfg2_dict = {x: y for x, y in runcfg1_dict.items()}
    runcfg2_dict['archspec'] = arch_cfg2_path.as_posix()
    runcfg2_dict['odir'] = tmpdir['run2'].as_posix()
    run_polproj(runcfg2_dict, tmpdir['temp'] / 'runcfg2.yaml')

    runcfg3_dict = {x: y for x, y in runcfg1_dict.items()}
    runcfg3_dict['odir'] = tmpdir['run3'].as_posix()
    run_polproj(runcfg3_dict, tmpdir['temp'] / 'runcfg3.yaml')


    run_projcmp(tmpdir['run1'], tmpdir['run2'], tmpdir['comparison1'], expected_return=1)
    assert (tmpdir['comparison1'] / study_name / 'html').is_dir(), f"HTML comparison directory not created at {tmpdir['comparison1'].stem}"
    run_projcmp(tmpdir['run1'], tmpdir['run2'], tmpdir['comparison2'], extra_args=['--no-html'], expected_return=1)
    assert not (tmpdir['comparison2'] / study_name / 'html').exists(), f"HTML comparison directory not created at {tmpdir['comparison2'].stem}"
    run_projcmp(tmpdir['run1'], tmpdir['run3'], tmpdir['comparison3'], extra_args=['--no-html'], expected_return=0)
    assert not (tmpdir['comparison3'] / study_name / 'html').exists(), f"HTML comparison directory not created at {tmpdir['comparison3'].stem}"
