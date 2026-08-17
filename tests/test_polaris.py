#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest
from tests.common import reset_typespec
import polaris

@pytest.mark.unit
def test_polaris(reset_typespec):
    assert polaris.main(['--odir', '__dummy', '--study', 'dummy', '--wlspec', 'config/mlperf_inference.yaml',
                         '--archspec', 'config/all_archs.yaml', '--wlmapspec',  'config/wl2archmapping.yaml',
                         '--dryrun']) == 0, "Polaris main function should return 0 on success"


@pytest.mark.unit
def test_polaris_disable_fusion_dryrun(reset_typespec):
    assert polaris.main(
        [
            '--odir',
            '__dummy',
            '--study',
            'dummy',
            '--wlspec',
            'config/mlperf_inference.yaml',
            '--archspec',
            'config/all_archs.yaml',
            '--wlmapspec',
            'config/wl2archmapping.yaml',
            '--dryrun',
            '--disable-fusion',
        ]
    ) == 0, "Polaris accepts --disable-fusion with --dryrun"


BASE_ARGS = [
    '--odir', '__dummy', '--study', 'dummy',
    '--wlspec', 'config/mlperf_inference.yaml',
    '--archspec', 'config/all_archs.yaml',
    '--wlmapspec', 'config/wl2archmapping.yaml',
    '--dryrun',
]


@pytest.mark.unit
def test_apply_filter_rejects_unmatched_entry():
    """A filter entry naming nothing raises rather than silently selecting nothing."""
    devices = [('Q1_A1', None), ('Q2_A2', None)]
    with pytest.raises(polaris.SpecSelectionError) as excinfo:
        polaris.apply_filter(devices, 'Q1_A1,NoSuchDevice', lambda x: x[0], 'filterarch')
    msg = str(excinfo.value)
    assert 'NoSuchDevice' in msg, f"error should name the offending entry, got: {msg}"
    assert '--filterarch' in msg, f"error should name the flag, got: {msg}"


@pytest.mark.unit
def test_apply_filter_accepts_valid_entries():
    """Matching stays case-insensitive and tolerates whitespace around entries."""
    devices = [('Q1_A1', None), ('Q2_A2', None)]
    assert polaris.apply_filter(devices, 'q1_a1', lambda x: x[0], 'filterarch') == [('Q1_A1', None)]
    assert polaris.apply_filter(devices, 'Q1_A1, Q2_A2', lambda x: x[0], 'filterarch') == devices
    assert polaris.apply_filter(devices, None, lambda x: x[0], 'filterarch') == devices


@pytest.mark.unit
def test_apply_filter_validates_against_full_domain():
    """An entry valid in the spec but absent from an already-narrowed list is not an error."""
    narrowed = [('Q1_A1', None)]
    domain = {'Q1_A1', 'Q2_A2'}
    assert polaris.apply_filter(narrowed, 'Q2_A2', lambda x: x[0], 'filterarch', domain) == []


@pytest.mark.unit
def test_polaris_rejects_unknown_device(reset_typespec):
    """An arch name absent from the archspec fails the run instead of running nothing."""
    assert polaris.main(BASE_ARGS + ['--filterarch', 'NoSuchDevice']) != 0, \
        "Polaris should fail when --filterarch names a device the archspec does not define"


@pytest.mark.unit
def test_polaris_rejects_unknown_workload_instance(reset_typespec):
    """Same for a workload-instance name absent from the workload spec."""
    assert polaris.main(BASE_ARGS + ['--filterwli', 'no_such_workload_instance']) != 0, \
        "Polaris should fail when --filterwli names an instance the wlspec does not define"


@pytest.mark.unit
def test_polaris_rejects_valid_but_disjoint_filters(reset_typespec):
    """Individually valid filters that share no workload still select nothing -- also a failure."""
    assert polaris.main(BASE_ARGS + ['--filterwl', 'BERT_SQUAD_v1p1', '--filterwli', 'rn50_b1']) != 0, \
        "Polaris should fail when valid filters jointly select no workload"


@pytest.mark.unit
def test_polaris_accepts_valid_filters(reset_typespec):
    """The rejections above must not cost us the ordinary filtered run."""
    assert polaris.main(BASE_ARGS + ['--filterarch', 'Q1_A1', '--filterwli', 'rn50_b1']) == 0, \
        "Polaris should run normally when every filter entry names something real"


@pytest.mark.unit
def test_polaris_reports_empty_spec_rather_than_blaming_filters(reset_typespec, tmp_path):
    """An empty spec and a disjoint filter pair are different mistakes in different files.

    With no filter given there is nothing to blame the filters for, so the message has to
    point at the spec instead — otherwise it sends the reader to their command line when the
    defect is in the YAML.
    """
    empty_spec = tmp_path / 'empty_workloads.yaml'
    empty_spec.write_text('workloads: []\n')

    args = [
        '--odir', str(tmp_path), '--study', 'dummy',
        '--wlspec', str(empty_spec),
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--dryrun',
    ]
    with pytest.raises(polaris.SpecSelectionError) as excinfo:
        polaris.get_workloads(str(empty_spec), polaris.RangeArgument('batchsize', None), None, None, None)
    msg = str(excinfo.value)
    assert 'defines no workload instances' in msg, f"should point at the spec, got: {msg}"
    assert '--filter' not in msg, f"should not blame filters that were never given, got: {msg}"

    assert polaris.main(args) != 0, "an empty spec should still fail the run"


def _run_polaris_cli(extra_args, tmp_path):
    """Invoke polaris.py as a subprocess and return the CompletedProcess."""
    repo_root = Path(polaris.__file__).parent
    argv = [
        sys.executable, str(repo_root / 'polaris.py'),
        '--odir', str(tmp_path), '--study', 'dummy',
        '--wlspec', 'config/mlperf_inference.yaml',
        '--archspec', 'config/all_archs.yaml',
        '--wlmapspec', 'config/wl2archmapping.yaml',
        '--dryrun',
    ] + extra_args
    return subprocess.run(argv, cwd=repo_root, capture_output=True, text=True)


@pytest.mark.unit
def test_polaris_cli_exits_nonzero_on_unknown_filter(tmp_path):
    """The process exit code must carry the failure, not just main()'s return value.

    The in-process tests above cannot catch this: the original defect was that __main__
    computed the status and then discarded it, so main() returned non-zero while the
    process still exited 0. Only a subprocess observes that.
    """
    res = _run_polaris_cli(['--filterarch', 'NoSuchDevice'], tmp_path)
    assert res.returncode != 0, \
        f"polaris.py should exit non-zero on an unknown filter; got {res.returncode}\n{res.stdout}\n{res.stderr}"
    assert 'NoSuchDevice' in (res.stdout + res.stderr), \
        "the error should name the offending filter entry"


@pytest.mark.unit
def test_polaris_cli_exits_zero_on_valid_run(tmp_path):
    """Counterpart to the above, so the check cannot pass by always failing."""
    res = _run_polaris_cli(['--filterarch', 'Q1_A1', '--filterwli', 'rn50_b1'], tmp_path)
    assert res.returncode == 0, \
        f"polaris.py should exit 0 on a valid run; got {res.returncode}\n{res.stdout}\n{res.stderr}"
