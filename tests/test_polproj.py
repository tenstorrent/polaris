#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import json
import logging
import tempfile

import pytest
import yaml
from git import GitCommandError

import polproj


class DummyPolarisRunConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_load_runconfig_reads_yaml_and_returns_config(monkeypatch):
    # Prepare a dummy YAML config
    config_data = {
        'odir': 'output_dir',
        'study': 'study_name',
        'wlspec': 'wlspec.yaml',
        'archspec': 'archspec.yaml',
        'wlmapspec': 'wlmapspec.yaml',
        'log_level': 1,
        'saved_copy': False,
        'githash': None,
        'filterrun': 'inference',
        'dump_ttsim_onnx': False,
        'instr_profile': False,
        'filterapi': None,
        'filterarch': None,
        'filterwl': None,
        'filterwli': None,
        'frequency': None,
        'batchsize': None,
        'knobs': [],
        'enable_memalloc': False,
        'enable_cprofile': False,
        'outputformat': 'json',
        'dump_stats_csv': False,
    }
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as tmpfile:
        yaml.dump(config_data, tmpfile)
        tmpfile_path = tmpfile.name

    # Patch choose_file to just return the file path
    monkeypatch.setattr(polproj, 'choose_file', lambda fname: tmpfile_path)
    # Patch PolarisRunConfig to our dummy class
    monkeypatch.setattr(polproj.runcfgmodel, 'PolarisRunConfig', DummyPolarisRunConfig)

    # Patch logging to avoid clutter
    monkeypatch.setattr(logging, 'info', lambda *a, **k: None)

    result = polproj.load_runconfig('dummy.yaml')
    assert isinstance(result, DummyPolarisRunConfig)
    assert result.kwargs['odir'] == 'output_dir'
    assert result.kwargs['study'] == 'study_name'

    os.remove(tmpfile_path)


def test_load_runconfig_file_not_found(monkeypatch):
    # Patch choose_file to raise FileNotFoundError
    monkeypatch.setattr(polproj, 'choose_file', lambda fname: (_ for _ in ()).throw(FileNotFoundError('not found')))
    with pytest.raises(FileNotFoundError):
        polproj.load_runconfig('does_not_exist.yaml')


def test_determine_commit_returns_none_if_githash_is_none(monkeypatch):
    class DummyHead:
        hexsha = 'abc123'
        commit = None

    class DummyRepo:
        head = DummyHead()

    result = polproj.determine_commit(DummyRepo(), None)  # type: ignore
    assert result is None


def test_determine_commit_returns_none_if_head_matches_githash(monkeypatch):
    class DummyCommit:
        hexsha = 'abc123'

    class DummyHead:
        hexsha = 'abc123'
        commit = DummyCommit()

    class DummyRepo:
        head = DummyHead()

    result = polproj.determine_commit(DummyRepo(), 'abc123')  # type: ignore
    assert result is None


def test_determine_commit_returns_commit_if_githash_differs(monkeypatch):
    class DummyCommit:
        hexsha = 'ghi789'
        # Simulate a commit that is different from the head

    dummy_commit = DummyCommit()

    class DummyHead:
        hexsha = 'abc123'
        commit = dummy_commit

    class DummyRepo:
        head = DummyHead()

        def iter_commits(self, githash, max_count):
            assert githash == 'def456'
            assert max_count == 1
            return [dummy_commit]

    result = polproj.determine_commit(DummyRepo(), 'def456')  # type: ignore
    assert result is dummy_commit


def test_determine_commit_raises_on_gitcommanderror(monkeypatch):
    class DummyCommit:
        hexsha = 'ghi789'

    class DummyHead:
        hexsha = 'abc123'
        commit = DummyCommit()

    class DummyRepo:
        head = DummyHead()

        def iter_commits(self, githash, max_count):
            raise GitCommandError('cmd', 1)

    # Patch logging.error to avoid clutter
    monkeypatch.setattr(logging, 'error', lambda *a, **k: None)

    with pytest.raises(GitCommandError):
        polproj.determine_commit(DummyRepo(), 'def456')  # type: ignore


def test_execute_expands_and_writes_json_and_calls_polaris(monkeypatch, tmp_path):
    # Dummy config and supporting objects
    ODIR_SUFFIX = 'bar'
    STUDY_SUFFIX = 'baz'

    class DummyRunCfg:
        def __init__(self):
            self.odir = str(tmp_path / 'outdir_{{foo}}')
            self.study = 'study_{{foo2}}'

    dummy_runcfg = DummyRunCfg()

    dummy_jtemplate = polproj.JTemplate()
    dummy_jdict = {'foo': ODIR_SUFFIX, 'foo2': STUDY_SUFFIX}

    # Patch polaris.polaris to track call
    called = {}

    def fake_polaris(rcfg):
        called['called'] = True
        called['rcfg'] = rcfg
        return 0

    monkeypatch.setattr(polproj.polaris, 'polaris', fake_polaris)

    # Run
    result = polproj.execute(dummy_runcfg, dummy_jtemplate, dummy_jdict, # type: ignore
                             args=type('Args', (), {})(), passdown_args=[])  # type: ignore

    # Check that odir and study were expanded
    assert dummy_runcfg.odir.endswith(ODIR_SUFFIX)
    assert dummy_runcfg.study.endswith(STUDY_SUFFIX)
    infopath = os.path.join(dummy_runcfg.odir, 'inputs', 'runinfo.json')
    # Check that runinfo.json was written and contains the jdict
    with open(infopath) as f:
        data = json.load(f)
    assert data == dummy_jdict

    # Check that polaris.polaris was called
    assert called['called']
    assert called['rcfg'] is dummy_runcfg

    # Check return value
    assert result == 0


def test_execute_raises_if_passdown_args(monkeypatch):
    class DummyRunCfg:
        odir = 'odir'
        study = 'study'

    class DummyJTemplate:
        def expand(self, s, jdict):
            return s

    dummy_runcfg = DummyRunCfg()
    dummy_jtemplate = DummyJTemplate()
    dummy_jdict: dict[str, polproj.JTemplate] = {}

    with pytest.raises(NotImplementedError):
        polproj.execute(dummy_runcfg, dummy_jtemplate, dummy_jdict,   # type: ignore
                        args=type('Args', (), {})(), passdown_args=['foo'])  # type: ignore


def test_choose_file_returns_absolute_path(tmp_path, monkeypatch):
    # Create a dummy file with absolute path
    abs_file = tmp_path / "absfile.yaml"
    abs_file.write_text("dummy")
    result = polproj.choose_file(str(abs_file))
    assert result == str(abs_file)



def test_choose_file_finds_file_in_current_directory(tmp_path, monkeypatch):
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    fname = "testfile.yaml"
    file_path = tmp_path / fname
    file_path.write_text("dummy")
    result = polproj.choose_file(fname)
    assert result == './' + str(fname)



def test_choose_file_finds_file_in_script_root(monkeypatch, tmp_path):
    # Simulate SCRIPT_ROOT_DIR containing the file
    fname = "rootfile.yaml"
    script_root = tmp_path / "script_root"
    script_root.mkdir()
    file_path = script_root / fname
    file_path.write_text("dummy")

    # Patch SCRIPT_ROOT_DIR in polproj
    monkeypatch.setattr(polproj, "SCRIPT_ROOT_DIR", str(script_root))
    # Ensure file is not in current directory
    assert not os.path.isfile(fname)
    result = polproj.choose_file(fname)
    assert result == str(file_path)


def test_choose_file_raises_if_file_not_found(monkeypatch):
    # Patch SCRIPT_ROOT_DIR to a temp dir with no files
    monkeypatch.setattr(polproj, "SCRIPT_ROOT_DIR", "/tmp/nonexistent_dir")
    with pytest.raises(FileNotFoundError):
        polproj.choose_file("doesnotexist.yaml")


def test_override_runconfig_overrides_odir_and_study(monkeypatch):
    class DummyRunCfg:
        def __init__(self):
            self.odir = "original_odir"
            self.study = "original_study"

    runconfig = DummyRunCfg()
    passdown_args = ['--odir', 'new_odir', '--study', 'new_study']

    # Patch logging.warning to avoid clutter
    monkeypatch.setattr(logging, 'warning', lambda *a, **k: None)

    polproj.override_runconfig(runconfig, passdown_args)  # type: ignore
    assert runconfig.odir == 'new_odir'
    assert runconfig.study == 'new_study'


def test_override_runconfig_missing_odir_value(monkeypatch):
    class DummyRunCfg:
        odir = "original_odir"
        study = "original_study"

    runconfig = DummyRunCfg()
    passdown_args = ['--odir']

    monkeypatch.setattr(logging, 'warning', lambda *a, **k: None)

    with pytest.raises(ValueError) as excinfo:
        polproj.override_runconfig(runconfig, passdown_args)  # type: ignore
    assert 'missing argument for --odir' in str(excinfo.value)


def test_override_runconfig_missing_study_value(monkeypatch):
    class DummyRunCfg:
        odir = "original_odir"
        study = "original_study"

    runconfig = DummyRunCfg()
    passdown_args = ['--study']

    monkeypatch.setattr(logging, 'warning', lambda *a, **k: None)

    with pytest.raises(ValueError) as excinfo:
        polproj.override_runconfig(runconfig, passdown_args)  # type: ignore
    assert 'missing argument for --study' in str(excinfo.value)


def test_override_runconfig_ignores_unrelated_args(monkeypatch):
    class DummyRunCfg:
        def __init__(self):
            self.odir = "original_odir"
            self.study = "original_study"

    runconfig = DummyRunCfg()
    passdown_args = ['--foo', 'bar']

    monkeypatch.setattr(logging, 'warning', lambda *a, **k: None)

    polproj.override_runconfig(runconfig, passdown_args)  # type: ignore
    assert runconfig.odir == "original_odir"
    assert runconfig.study == "original_study"
