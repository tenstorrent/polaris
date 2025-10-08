#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import datetime
import enum
import filecmp
import json
import math
import matplotlib.pyplot as plt
import multiprocessing
import os
import re
import shutil
import subprocess
import typing
import yaml

AlignDir = typing.Literal["left", "right", "<", ">"]
AlignSpec = typing.Tuple[AlignDir, int]  # (width, direction)

GITIGNORE_PREFIX: str = "__"
RTL_DATA_PATH_ROOT: str = f"{GITIGNORE_PREFIX}ext"
OUTPUT_PATH_ROOT: str = f"{GITIGNORE_PREFIX}llk_out"
RTL_DATA_FILE_PREFIX: str = "ext_rtl_test_data_set_"
RTL_DATA_FILE_SUFFIX: str = ".tar.gz"

class Utils:
    @staticmethod
    def get_dirs_with_prefix(prefix: str, root_dir: str) -> list[str]:
        dir_names: list[str] = []
        for pwd, sub_dirs, _ in os.walk(root_dir):
            for sub_dir in sub_dirs:
                if sub_dir.startswith(prefix):
                    dir_names.append(os.path.join(pwd, sub_dir))
        return dir_names

    @staticmethod
    def get_dirs_with_suffix(suffix: str, root_dir: str) -> list[str]:
        dir_names: list[str] = []
        for pwd, _, _ in os.walk(root_dir):
            if pwd.endswith(suffix):
                dir_names.append(pwd)
        return dir_names

    @staticmethod
    def get_dirs_with_name(dir_name: str, root_dir: str) -> list[str]:
        dir_names: list[str] = []
        for pwd, sub_dirs, _ in os.walk(root_dir):
            # TODO: replace for loop with dir_name in sub_dirs
            if dir_name in sub_dirs:
                dir_names.append(os.path.join(pwd, dir_name))
        return dir_names

    @staticmethod
    def get_path_of_only_dir_with_name(dir_name: str, root_dir: str) -> str:
        dir_paths = Utils.get_dirs_with_name(dir_name, root_dir)
        if 1 != len(dir_paths):
            raise Exception(f"- error: expected exactly one dir with name {dir_name} in root {root_dir}, found {len(dir_paths)}")
        return dir_paths[0]

    @staticmethod
    def get_path_of_only_dir_with_suffix(suffix: str, root_dir: str) -> str:
        dir_paths = Utils.get_dirs_with_suffix(suffix, root_dir)
        if 1 != len(dir_paths):
            raise Exception(f"- error: expected exactly one dir with suffix {suffix} in root {root_dir}, found {len(dir_paths)}")
        return dir_paths[0]

    @staticmethod
    def get_path_of_only_dir_with_prefix(prefix: str, root_dir: str) -> str:
        dir_paths = Utils.get_dirs_with_prefix(prefix, root_dir)
        if 1 != len(dir_paths):
            raise Exception(f"- error: expected exactly one dir with prefix {prefix} in root {root_dir}, found {len(dir_paths)}")
        return dir_paths[0]

    @staticmethod
    def get_files_in_dir_with_name(file_name: str, dir_path: str) -> list[str]:
        matching_files: list[str] = []
        for root, _, files in os.walk(dir_path):
            if file_name in files:
                matching_files.append(os.path.join(root, file_name))
        return sorted(matching_files)

    @staticmethod
    def has_only_one_copy_of_file_in_dir(file_name: str, dir_path: str) -> bool:
        files = Utils.get_files_in_dir_with_name(file_name, dir_path)
        return True if ((1 == len(files)) and os.path.basename(files[0]) == file_name) else False

    @staticmethod
    def get_path_of_only_copy_of_file_in_dir(file_name: str, dir_path: str) -> str:
        files = Utils.get_files_in_dir_with_name(file_name, dir_path)
        if 1 != len(files):
            raise Exception(f"- error: expected exactly one copy of file {file_name} in dir {dir_path}, found {len(files)}")
        return files[0]

    @staticmethod
    def rename_with_timestamp(
        path: str,
        *,
        format: str = "%Y%m%d_%H%M%S",      # 20231130_234501
        time_zone: datetime.timezone = datetime.timezone.utc,
    ) -> str:
        """
        Rename `path` to "<stem>_<timestamp><suffix>" and return the new path (as str).
        Example: logfile.txt -> logfile_20231130_234501.txt
        """
        p = os.fspath(path)  # accept str or PathLike

        ts = datetime.datetime.now(time_zone).strftime(format)  # one shot: consistent everywhere
        dir_name, base = os.path.split(p)
        stem, suffix = os.path.splitext(base)

        new_name = f"{stem}_{ts}{suffix}"
        new_path = os.path.join(dir_name, new_name)

        os.rename(p, new_path)
        return new_path

    @staticmethod
    def get_num_dirs_with_keyword(keyword: str, path: str) -> int:
        for _, sub_dirs, _ in os.walk(path):
            if any(sub_dir.startswith(keyword) for sub_dir in sub_dirs):
                kw_dirs: list[str] = []
                for sub_dir in sub_dirs:
                    if sub_dir.startswith(keyword):
                        kw_dirs.append(sub_dir)

                kw_dirs = sorted(kw_dirs)
                if not kw_dirs:
                    raise Exception(f"- error: did not find any directories with keyword {keyword} in path {path}")

                kw_ids = sorted(int(kw_dir.split('_')[-1]) for kw_dir in kw_dirs)
                min_kw_id = min(kw_ids)
                if 0 != min_kw_id:
                    raise Exception(f"- error: min dir id does not start from 0. kw_dirs are as follows: {kw_dirs}")

                if kw_ids == list(range(min(kw_ids), max(kw_ids) + 1)):
                    return len(kw_ids)

        return 0

    @staticmethod
    def get_num_neos(path: str) -> int:
        return Utils.get_num_dirs_with_keyword("neo_", path)

    @staticmethod
    def get_num_threads(path: str) -> int:
        return Utils.get_num_dirs_with_keyword("thread_", path)

    @staticmethod
    def sort_table_by_columns_order(tbl: list[typing.Any], sort_by: list[int] | int, descending: list[bool] | bool | None = None):
        if not tbl:
            return tbl

        if isinstance(sort_by, int):
            assert (descending is None) or isinstance(descending, bool), "If sort_by is int, descending must be None or bool"
            if descending is None:
                descending = False

            descending = [descending]
            sort_by = [sort_by]

        assert isinstance(tbl, list), "tbl must be a list"
        assert isinstance(sort_by, list) and all(isinstance(elem, int) for elem in sort_by), "sort_by must be a list of integers"
        assert isinstance(descending, list) and all(isinstance(elem, bool) for elem in descending), "descending must be a list of booleans"
        assert all(0 <= col < len(tbl[0]) for col in sort_by), "All sort_by values must be valid column indices"
        assert len(sort_by) == len(descending), "All descending values must match sort_by length"


        for c, rev in reversed(list(zip(sort_by, descending))):
            def sort_key(row: typing.Any, col: int = c) -> typing.Any:
                val = row[col]
                return math.inf if val is None else val
            tbl.sort(key = sort_key, reverse = rev)

        # for idx, c in enumerate(reversed(sort_by)):
        #     tbl.sort(key=operator.itemgetter(c), reverse=descending[len(descending) - 1 - idx])

        return tbl

    @staticmethod
    def align_text(text: str, spec: typing.Optional[AlignSpec]) -> str:
        if spec is None:
            return text
        direction, width = spec
        ch = {"left": "<", "right": ">", "<": "<", ">": ">"}.get(direction)
        if ch is None:
            raise ValueError("direction must be 'left'/'right' or '<'/'>'")
        return f"{text:{ch}{width}}"

class TestStatusUtils:
    @staticmethod
    def get_test_classes():
        classes: dict[str, set[str]] = dict()
        classes['datacopy'.upper()] = {'datacopy'}
        classes['eltw'.upper()]     = {'elwmul', 'elwadd', 'elwsub'}
        classes['matmul'.upper()]   = {'matmul'}
        classes['pck'.upper()]      = {'pck'}
        classes['reduce'.upper()]   = {'reduce'}
        classes['sfpu'.upper()]     = {'lrelu', 'tanh', 'sqrt', 'exp', 'recip', 'relu', 'cast'}
        classes['upk'.upper()]      = {'upk'}

        return classes

    @staticmethod
    def get_failure_bins():
        bins = [
            ["attribs expected. Received"],
            ["Check Valid condition Invalid"],
            ["IndexError"],
            ["Timeout", "reached for pipe"],
            ["Timeout", "reached for valid check"],
            ["Too many resources to select from"],
            ["Write Valid condition Invalid"],
        ]

        return sorted(bins)

    @staticmethod
    def get_failure_bins_as_str():
        str_bins: list[str] = list()
        for b in TestStatusUtils.get_failure_bins():
            str_bins.append(" ".join([s for s in b]))

        return str_bins

    @staticmethod
    def get_failure_bin_index(msg: str):
        for idx, b in enumerate(TestStatusUtils.get_failure_bins_as_str()):
            if b == msg:
                return idx

        return len(TestStatusUtils.get_failure_bins_as_str())

class Run:
    """Represents a single test run configuration"""
    def __init__(self,
                 name: str,
                 tests: list[str] | str | None = None,
                 tags: list[str] | str | None = None,
                 parallel: int | None = None,
                 debug: int | None = None,
                 cfg: str | None = None,
                 defCfg: str | None = None,
                 exp: str | None = None,
                 memoryMap: str | None = None,
                 odir: str | None = None,
                 risc_cpi: float | None = None,
                 ttISAFileName: str | None = None):
        if isinstance(tests, str):
            tests = [tests]

        if isinstance(tags, str):
            tags = [tags]

        self.name = str(name).strip() or "unnamed"
        self.tests = tests or []
        self.tags = tags or []
        self.parallel = parallel
        self.debug = debug
        self.cfg = cfg
        self.defCfg = defCfg
        self.exp = exp
        self.memoryMap = memoryMap
        self.odir = odir
        self.risc_cpi = risc_cpi
        self.ttISAFileName = ttISAFileName

    def __str__(self) -> str:
        return f"Run(name={self.name}, tests={self.tests}, tags={self.tags}, parallel={self.parallel}, debug={self.debug}, cfg={self.cfg}, defCfg={self.defCfg}, exp={self.exp}, memoryMap={self.memoryMap}, odir={self.odir}, risc_cpi={self.risc_cpi}, ttISAFileName={self.ttISAFileName})"

    def __repr__(self) -> str:
        return self.__str__()

class RunConfig:
    """Handles loading and managing run configurations from YAML files"""
    def __init__(self, config_file: str | None = None):
        self.runs: list[Run] = []
        if config_file:
            self.load_from_yaml(config_file)

    def append_run(self, run: Run) -> None:
        existing_names = {r.name for r in self.runs}
        if run.name in existing_names:
            base_name = run.name.rstrip("_")
            counter = 1
            while f"{base_name}_{counter}" in existing_names:
                counter += 1
            run.name = f"{base_name}_{counter}"

        self.runs.append(run)

    def load_from_yaml(self, config_file: str) -> None:
        """Load run configurations from a YAML file"""
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        if 'batch' not in config_data:
            raise ValueError("Config file must contain 'batch' section")

        for run_data in config_data['batch']:
            run = Run(
                name=run_data.get('name', "unnamed"),
                tests=run_data.get('test', None),
                tags=run_data.get('tag', None),
                parallel=run_data.get('parallel', None),
                debug=run_data.get('debug', None),
                cfg=run_data.get('cfg', None),
                defCfg=run_data.get('defCfg', None),
                exp=run_data.get('exp', None),
                memoryMap=run_data.get('memoryMap', None),
                odir=run_data.get('odir', None),
                risc_cpi=run_data.get('risc.cpi', None),
                ttISAFileName=run_data.get('ttISAFileName', None)
            )
            self.append_run(run)

    def add_cli_run(self,
                    tests: list[str] | None = None,
                    tags: list[str] | None = None,
                    parallel: int | None = None,
                    debug: int | None = None,
                    cfg: str | None = None,
                    defCfg: str | None = None,
                    exp: str | None = None,
                    memoryMap: str | None = None,
                    odir: str | None = None,
                    risc_cpi: float | None = None,
                    ttISAFileName: str | None = None,
                    name: str | None = None) -> None:
        """Add a run from CLI arguments"""
        cli_run = Run(
            name=name or "baseline",
            tests=tests,
            tags=tags,
            parallel=parallel,
            debug=debug,
            cfg=cfg,
            defCfg=defCfg,
            exp=exp,
            memoryMap=memoryMap,
            odir=odir,
            risc_cpi=risc_cpi,
            ttISAFileName=ttISAFileName
        )
        self.append_run(cli_run)

class InputParams:
    def __init__(self,
                 run_name: str,
                 tags: list[str] | None,
                 tests: list[str] | None,
                 parallel: int | None,
                 cfg: str | None = None,
                 debug: int | None = None,
                 defCfg: str | None = None,
                 exp: str | None = None,
                 force: bool = False,
                 memoryMap: str | None = None,
                 odir: str | None = None,
                 risc_cpi: float | None = None,
                 ttISAFileName: str | None = None,
                 ):

        if risc_cpi is not None and risc_cpi <= 0:
            raise ValueError(f"risc_cpi must be positive, got: {risc_cpi}, run name: {run_name}")

        self.default_model_cfg_file_name_prefix: str = f"ttqs_neo4_"
        self.default_model_cfg_file_name_suffix: str = f".json"
        self.default_model_memory_map_file_name_prefix: str = f"ttqs_memory_map_"
        self.default_model_memory_map_file_name_suffix: str = f".json"
        self.force: bool                          = force or True
        self.gitignore_prefix                     = GITIGNORE_PREFIX
        self.instruction_kind: str                = "ttqs"
        self.isa_file_name: str                   = "assembly.yaml"
        self.max_num_threads_per_neo_core: int    = 4
        self.model_cfg_dir_prefix: str            = f"{GITIGNORE_PREFIX}config_files"
        self.model_inputcfg_file_prefix: str      = "inputcfg_"
        self.model_instruction_sets_dir_path_prefix: str = "ttsim/config/llk/instruction_sets"
        self.model_log_file_end: str              = "Simreport = "
        self.model_log_file_suffix: str           = ".model_test.log"
        self.model_odir_prefix: str               = f"{OUTPUT_PATH_ROOT}/llk"
        self.model_simreport: str                 = "simreport_"
        self.num_processes: int                   = 1
        self.rtl_data_path_prefix: str            = f"{RTL_DATA_PATH_ROOT}/rtl_test_data_set"
        self.rtl_status_file_name: str            = "sim_result.yml"
        self.rtl_tags: list[str]                  = ["jul1", "jul27", "sep23"]
        self.rtl_test_dir_path_suffix: str        = 'rsim/debug'
        self.rtl_test_dir_suffix: str             = '_0'
        self.rtl_tests: list[str]                 = []
        self.start_function                       = "_start"
        self.cfg: str | None                      = cfg or None
        self.debug: int | None                    = debug if debug is not None else 15
        self.experiment: str | None               = exp or None
        self.memoryMap: str | None                = memoryMap or None
        self.odir: str | None                     = odir or None
        self.risc_cpi: float | None               = risc_cpi or None
        self.ttISAFileName: str | None            = ttISAFileName or None
        self.defCfg: str | None                   = defCfg or None
        self.run_name: str                        = run_name or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

        if tags:
            if not all(tag in self.rtl_tags for tag in tags):
                raise ValueError(f"One or more invalid RTL tag(s): {[tag for tag in sorted(tags) if tag not in self.rtl_tags]}, accepted RTL tags: {self.rtl_tags}")
            self.rtl_tags = tags

        for tag in self.rtl_tags:
            rtl_data_path = os.path.join(self.rtl_data_path_prefix, tag)

            if not os.path.isdir(rtl_data_path):
                rtl_data_tar_file_name = RTL_DATA_FILE_PREFIX + tag + RTL_DATA_FILE_SUFFIX
                lfc_downloader = os.path.relpath(os.path.normpath(os.path.join(os.path.dirname(__file__), 'ci/lfc_downloader.sh')))
                if not os.path.isfile(lfc_downloader):
                    raise FileNotFoundError(f"LFC downloader script not found: {lfc_downloader}")
                cmd = f"bash {lfc_downloader} --extract {rtl_data_tar_file_name}"
                print(f"- Downloading RTL test data with command: {cmd}")
                subprocess.run(cmd, shell=True, check=True)

            if not os.path.isdir(rtl_data_path):
                raise ValueError(f"RTL data path {rtl_data_path} not found after download")

        if "ttqs" != self.instruction_kind:
            self.max_num_threads_per_neo_core = 3

        if tests:
            self.rtl_tests = tests

        if isinstance(parallel, int) and parallel > self.num_processes:
            self.num_processes = parallel

        if not self.has_config_files(rtl_tag = None):
            raise ValueError(f"Configuration files not found for one or more RTL tags: {self.rtl_tags}")

    def to_str(self: typing.Self, offset: int = 0) -> str:
        indent = " " * offset
        msg = ""
        msg += f"{indent}debug                                     = {self.debug}\n"
        msg += f"{indent}default_model_cfg_file_name_prefix        = {self.default_model_cfg_file_name_prefix}\n"
        msg += f"{indent}default_model_cfg_file_name_suffix        = {self.default_model_cfg_file_name_suffix}\n"
        msg += f"{indent}default_model_memory_map_file_name_prefix = {self.default_model_memory_map_file_name_prefix}\n"
        msg += f"{indent}default_model_memory_map_file_name_suffix = {self.default_model_memory_map_file_name_suffix}\n"
        msg += f"{indent}force                                     = {self.force}\n"
        msg += f"{indent}instruction_kind                          = {self.instruction_kind}\n"
        msg += f"{indent}isa_file_name                             = {self.isa_file_name}\n"
        msg += f"{indent}max_num_threads_per_neo_core              = {self.max_num_threads_per_neo_core}\n"
        msg += f"{indent}model_cfg_dir_prefix                      = {self.model_cfg_dir_prefix}\n"
        msg += f"{indent}model_inputcfg_file_prefix                = {self.model_inputcfg_file_prefix}\n"
        msg += f"{indent}model_instruction_sets_dir_path_prefix    = {self.model_instruction_sets_dir_path_prefix}\n"
        msg += f"{indent}model_log_file_end                        = {self.model_log_file_end}\n"
        msg += f"{indent}model_log_file_suffix                     = {self.model_log_file_suffix}\n"
        msg += f"{indent}model_odir_prefix                         = {self.model_odir_prefix}\n"
        msg += f"{indent}model_simreport                           = {self.model_simreport}\n"
        msg += f"{indent}run_name                                  = {self.run_name}\n"
        msg += f"{indent}num_processes                             = {self.num_processes}\n"
        msg += f"{indent}rtl_data_path_prefix                      = {self.rtl_data_path_prefix}\n"
        msg += f"{indent}rtl_status_file_name                      = {self.rtl_status_file_name}\n"
        msg += f"{indent}rtl_tags                                  = {self.rtl_tags}\n"
        msg += f"{indent}rtl_test_dir_path_suffix                  = {self.rtl_test_dir_path_suffix}\n"
        msg += f"{indent}rtl_test_dir_suffix                       = {self.rtl_test_dir_suffix}\n"
        msg += f"{indent}rtl_tests                                 = {self.rtl_tests}\n"
        msg += f"{indent}start_function                            = {self.start_function}\n"
        msg += f"{indent}cfg                                       = {self.cfg}\n"
        msg += f"{indent}experiment (exp)                          = {self.experiment}\n"
        msg += f"{indent}memoryMap                                 = {self.memoryMap}\n"
        msg += f"{indent}odir                                      = {self.odir}\n"
        msg += f"{indent}risc_cpi (risc.cpi)                       = {self.risc_cpi}\n"
        msg += f"{indent}ttISAFileName                             = {self.ttISAFileName}\n"
        msg += f"{indent}defCfg                                    = {self.defCfg}\n"

        return msg.rstrip()

    def __str__(self: typing.Self) -> str:
        return self.to_str()

    def __repr__(self: typing.Self) -> str:
        return self.__str__()

    def get_model_root_dir(self: typing.Self) -> str:
        return os.getcwd()

    def get_rtl_data_path_with_tag(self: typing.Self, rtl_tag: str) -> str:
        if rtl_tag not in self.rtl_tags:
            return ""

        path: str = Utils.get_path_of_only_dir_with_name(rtl_tag, self.rtl_data_path_prefix)
        if not os.path.isdir(path):
            raise Exception(f"- error: RTL tag path does not exist: {path}")
        return path

    def get_rtl_test_data_path(self: typing.Self, rtl_tag: str) -> str:
        if rtl_tag not in self.rtl_tags:
            return ""

        path: str = Utils.get_path_of_only_dir_with_suffix(self.rtl_test_dir_path_suffix, self.get_rtl_data_path_with_tag(rtl_tag))
        if not os.path.isdir(path):
            raise Exception(f"- error: RTL test data path does not exist: {path}")
        return path

    def get_rtl_test_names(self: typing.Self, rtl_tag: str | None = None) -> list[str]:
        if self.rtl_tests:
            return self.rtl_tests

        if isinstance(rtl_tag, str) and rtl_tag in self.rtl_tags:
            test_data_path = self.get_rtl_test_data_path(rtl_tag)
            if os.path.isdir(test_data_path) is False:
                raise Exception(f"- error: RTL test data path does not exist for tag: {rtl_tag}")

            test_names: list[str] = []
            with os.scandir(test_data_path) as it:
                for elem in it:
                    if elem.is_dir() and elem.name.endswith(self.rtl_test_dir_suffix):
                        test_names.append(elem.name[:-len(self.rtl_test_dir_suffix)])

            return sorted(test_names)

        return []

    def get_rtl_test_dir_path(self: typing.Self, test_name: str, rtl_tag: str) -> str:
        if rtl_tag not in self.rtl_tags:
            return ""

        test_data_path = self.get_rtl_test_data_path(rtl_tag)
        test_dir = os.path.join(test_data_path, f"{test_name}{self.rtl_test_dir_suffix}")
        if os.path.isdir(test_dir):
            return test_dir

        print(f"- warning: RTL test dir path {test_dir} does not exist, returning empty string")
        if os.path.isdir(test_data_path):
            msg: str = f"  - info: Available tests in {test_data_path}:\n"
            old_val = self.rtl_tests
            self.rtl_tests = []
            available_tests = self.get_rtl_test_names(rtl_tag)
            self.rtl_tests = old_val
            msg += "\n".join([f"    - {idx}. {test}" for idx, test in enumerate(sorted(available_tests))])
            print(msg)

        return ""

    def get_cfg_file_name(self: typing.Self, rtl_tag: str) -> str:
        if rtl_tag not in self.rtl_tags:
            return ""

        cfg_file_name = f"{self.default_model_cfg_file_name_prefix}{rtl_tag}{self.default_model_cfg_file_name_suffix}"
        return cfg_file_name

    def get_memory_map_file_name(self: typing.Self, rtl_tag: str) -> str:
        if rtl_tag not in self.rtl_tags:
            return ""

        memory_map_file_name = f"{self.default_model_memory_map_file_name_prefix}{rtl_tag}{self.default_model_memory_map_file_name_suffix}"
        return memory_map_file_name

    def has_cfg_file(self: typing.Self, rtl_tag: str) -> bool:
        if self.cfg:
            return os.path.isfile(self.cfg)

        if rtl_tag not in self.rtl_tags:
            return False

        cfg_file_name = self.get_cfg_file_name(rtl_tag)
        if Utils.has_only_one_copy_of_file_in_dir(cfg_file_name, self.get_model_root_dir()):
            return True

        print(f"- warning: cfg file {cfg_file_name} not found in RTL data path {self.get_model_root_dir()}")
        return False

    def has_memory_map_file(self: typing.Self, rtl_tag: str) -> bool:
        if self.memoryMap:
            return os.path.isfile(self.memoryMap)

        if rtl_tag not in self.rtl_tags:
            return False

        memory_map_file_name = self.get_memory_map_file_name(rtl_tag)
        if Utils.has_only_one_copy_of_file_in_dir(memory_map_file_name, self.get_model_root_dir()):
            return True

        print(f"- warning: memory map file {memory_map_file_name} not found in RTL data path {self.get_model_root_dir()}")
        return False

    def has_cfg_files(self: typing.Self) -> bool:
        return all(self.has_cfg_file(tag) for tag in self.rtl_tags)

    def has_memory_map_files(self: typing.Self) -> bool:
        return all(self.has_memory_map_file(tag) for tag in self.rtl_tags)

    def has_isa_file(self: typing.Self, rtl_tag: str) -> bool:
        if self.ttISAFileName:
            return os.path.isfile(self.ttISAFileName)

        if rtl_tag not in self.rtl_tags:
            return False

        path = self.get_rtl_data_path_with_tag(rtl_tag)
        if "assembly.yaml" != self.isa_file_name:
            raise Exception(f"- error: expected isa_file_name to be assembly.yaml, received {self.isa_file_name}")

        if Utils.has_only_one_copy_of_file_in_dir(self.isa_file_name, path):
            return True

        return False

    def has_isa_files(self: typing.Self) -> bool:
        return all(self.has_isa_file(tag) for tag in self.rtl_tags)

    def get_isa_file_path(self: typing.Self, rtl_tag: str) -> str:
        if self.ttISAFileName:
            return self.ttISAFileName

        if rtl_tag not in self.rtl_tags:
            return ""

        path = self.get_rtl_data_path_with_tag(rtl_tag)
        if "assembly.yaml" != self.isa_file_name:
            raise Exception(f"- error: expected isa_file_name to be assembly.yaml, received {self.isa_file_name}")

        return Utils.get_path_of_only_copy_of_file_in_dir(self.isa_file_name, path)

    def get_cfg_file_path(self: typing.Self, rtl_tag: str) -> str:
        if self.cfg:
            return self.cfg

        if rtl_tag not in self.rtl_tags:
            return ""

        path = self.get_rtl_data_path_with_tag(rtl_tag)
        cfg_file_name = self.get_cfg_file_name(rtl_tag)

        return Utils.get_path_of_only_copy_of_file_in_dir(cfg_file_name, path)

    def get_memory_map_file_path(self: typing.Self, rtl_tag: str) -> str:
        if self.memoryMap:
            return self.memoryMap

        if rtl_tag not in self.rtl_tags:
            return ""

        path = self.get_rtl_data_path_with_tag(rtl_tag)
        memory_map_file_name = self.get_memory_map_file_name(rtl_tag)

        return Utils.get_path_of_only_copy_of_file_in_dir(memory_map_file_name, path)

    def has_config_files(self: typing.Self, rtl_tag: str | None) -> bool:
        if rtl_tag is None:
            return all(self.has_config_files(tag) for tag in self.rtl_tags)

        if rtl_tag not in self.rtl_tags:
            return False

        has_files = self.has_cfg_file(rtl_tag) and \
            self.has_memory_map_file(rtl_tag) and \
            self.has_isa_file(rtl_tag)

        if not has_files:
            msg = f"- Missing configuration files for RTL tag: {rtl_tag}. has_cfg_file: {self.has_cfg_file(rtl_tag)}, has_memory_map_file: {self.has_memory_map_file(rtl_tag)}, has_isa_file: {self.has_isa_file(rtl_tag)}"
            print(msg)

        return self.has_cfg_file(rtl_tag) and \
            self.has_memory_map_file(rtl_tag) and \
            self.has_isa_file(rtl_tag)

    def __get_inputcfg_odir_path(self: typing.Self, prefix: str, run_name: str, rtl_tag: str) -> str:
        if rtl_tag not in self.rtl_tags:
            return ""

        if not prefix.startswith(self.gitignore_prefix):
            prefix = self.gitignore_prefix + prefix

        path = os.path.join(prefix, run_name, rtl_tag)
        path = os.path.normpath(path)

        os.makedirs(path, exist_ok=True)

        return path

    def get_inputcfg_dir_path(self: typing.Self, rtl_tag: str) -> str:
        return self.__get_inputcfg_odir_path(self.model_cfg_dir_prefix, self.run_name, rtl_tag)

    def get_odir_path(self: typing.Self, rtl_tag: str) -> str:
        if self.odir:
            path = self.odir
            path = os.path.normpath(path)
            os.makedirs(path, exist_ok=True)
            return path
        return self.__get_inputcfg_odir_path(self.model_odir_prefix, self.run_name, rtl_tag)

    def get_isa_dir_path(self: typing.Self) -> str:
        if self.ttISAFileName:
            path = os.path.dirname(os.path.abspath(self.ttISAFileName))
            path = os.path.normpath(path)
            os.makedirs(path, exist_ok=True)
            return path
        path = f"{self.model_instruction_sets_dir_path_prefix}/{self.instruction_kind}"
        path = os.path.normpath(path)
        os.makedirs(path, exist_ok=True)

        return path

    def copy_isa_file_from_rtl_data(self: typing.Self, rtl_tag: str) -> None:
        if rtl_tag not in self.rtl_tags:
            return

        if "assembly.yaml" != self.isa_file_name:
            raise Exception(f"- error: expected isa_file_name to be assembly.yaml, received {self.isa_file_name}")

        isa_dir = self.get_isa_dir_path()
        rtl_isa_path = self.get_isa_file_path(rtl_tag)

        if not os.path.isfile(rtl_isa_path):
            raise Exception(f"  - warning: RTL ISA file does not exist: {rtl_isa_path}")

        isa_file = os.path.join(isa_dir, self.isa_file_name)

        if os.path.isfile(isa_file):
            filecmp.clear_cache()
            if filecmp.cmp(rtl_isa_path, isa_file, shallow = False):
                print(f"  - info: ISA file already exists and is identical, skipping copy: {isa_file}")
                return
            else:
                renamed_path = Utils.rename_with_timestamp(isa_file)
                print(f"  - info: Existing ISA file renamed to: {renamed_path}")

        shutil.copy2(rtl_isa_path, isa_dir)

    def write_run_config_to_inputcfg_dir(self: typing.Self) -> None:
        run_params: dict[str, typing.Any] = dict()
        run_params["name"] = self.run_name
        run_params["tag"] = self.rtl_tags
        run_params["test"] = self.get_rtl_test_names(rtl_tag = None)
        run_params["parallel"] = self.num_processes
        if self.debug:
            run_params["debug"] = self.debug
        if self.cfg:
            run_params["cfg"] = self.cfg
        if self.defCfg:
            run_params["defCfg"] = self.defCfg
        if self.experiment:
            run_params["exp"] = self.experiment
        if self.memoryMap:
            run_params["memoryMap"] = self.memoryMap
        if self.odir:
            run_params["odir"] = self.odir
        if self.risc_cpi:
            run_params["risc.cpi"] = self.risc_cpi
        if self.ttISAFileName:
            run_params["ttISAFileName"] = self.ttISAFileName

        odir = os.path.dirname(self.get_inputcfg_dir_path(self.rtl_tags[0]))
        run_config_file = os.path.join(odir, f"batch_{self.run_name}.yaml")
        print(f"  + Writing batch file to: {run_config_file}")
        with open(run_config_file, 'w') as file:
            yaml.dump({"batch": [run_params]}, file)


class Inputcfg:
    @staticmethod
    def get_file_name_incl_path(test_name: str, tag: str, input_params: InputParams) -> str:
        cfg_dir_incl_path = input_params.get_inputcfg_dir_path(tag)

        if not os.path.isdir(cfg_dir_incl_path):
            os.makedirs(cfg_dir_incl_path, exist_ok = True)

        prefix = input_params.model_inputcfg_file_prefix

        assert "/" not in prefix, f"- error: expected no '/' in {input_params.model_inputcfg_file_prefix}, received {prefix}"

        if not prefix.endswith("_"):
            prefix += "_"

        return os.path.join(cfg_dir_incl_path, f"{prefix}{test_name}.json")

    @staticmethod
    def get_rtl_test_data_path(test_name: str, rtl_tag: str, input_params: InputParams) -> str:
        rtl_test_data_path = os.path.join(input_params.get_rtl_test_data_path(rtl_tag), f"{test_name}{input_params.rtl_test_dir_suffix}")
        return rtl_test_data_path if os.path.isdir(rtl_test_data_path) else ""


    @staticmethod
    def get(test_name: str, rtl_tag: str, input_params: InputParams):
        test_dir_incl_path = Inputcfg.get_rtl_test_data_path(test_name, rtl_tag, input_params)
        if not test_dir_incl_path:
            print(f"  - warning: RTL test data path does not exist for test {test_name} with tag {rtl_tag}")
            return

        input_cfg_dict: dict[str, typing.Any] = dict()
        input_cfg_dict["llkVersionTag"]       = rtl_tag
        input_cfg_dict["debug"]               = input_params.debug
        input_cfg_dict["ttISAFileName"]       = input_params.get_isa_file_path(rtl_tag)
        input_cfg_dict["odir"]                = input_params.get_odir_path(rtl_tag)
        if input_params.cfg:
            input_cfg_dict["cfg"] = input_params.cfg

        if input_params.defCfg:
            input_cfg_dict["defCfg"] = input_params.defCfg

        if input_params.experiment:
            input_cfg_dict["exp"] = input_params.experiment

        if input_params.memoryMap:
            input_cfg_dict["memoryMap"] = input_params.memoryMap

        if input_params.risc_cpi:
            input_cfg_dict["risc.cpi"] = input_params.risc_cpi

        input_cfg_dict["numTCores"] = Utils.get_num_neos(test_dir_incl_path)
        input_cfg_dict["input"] = dict[str, typing.Any]()
        input_cfg: dict[str, typing.Any] = input_cfg_dict["input"]
        input_cfg["syn"] = 0
        input_cfg["name"] = test_name
        num_neos: int = input_cfg_dict["numTCores"]

        for neo_id in range(num_neos):
            tc_key = f"tc{neo_id}"
            input_neo: dict[str, typing.Any] = dict()
            neo_dir = f"neo_{neo_id}"
            for pwd, _, _ in os.walk(test_dir_incl_path):
                if pwd.endswith(neo_dir):
                    input_neo["startFunction"] = input_params.start_function
                    input_neo["numThreads"] = input_params.max_num_threads_per_neo_core
                    for thread_id in range(input_params.max_num_threads_per_neo_core):
                        # key = f"th{thread_id}Path"
                        input_neo[f"th{thread_id}Path"] = ""
                        input_neo[f"th{thread_id}Elf"] = ""

                    num_threads = Utils.get_num_threads(pwd)
                    if num_threads in {0, None}:
                        raise Exception(f"- error: expected at least one thread, received {num_threads}. Path: {pwd}")

                    neo_os_walk = os.walk(pwd)
                    for pwd1, _, files in neo_os_walk:
                        for file in files:
                            if file.endswith(".elf"):
                                # Extract thread number from filename pattern: thread_X.elf
                                match = re.match(r'^thread_(\d+)\.elf$', file)
                                if not match:
                                    raise Exception(f"- error: ELF file does not follow expected pattern 'thread_X.elf'. Found: {file}")

                                thread_id = int(match.group(1))
                                if thread_id >= num_threads:
                                    raise Exception(f"- error: thread id is greater than or equal to number of threads. thread_id: {thread_id}, number of threads: {num_threads}")
                                input_neo[f"th{thread_id}Path"] = os.path.relpath(pwd1)
                                input_neo[f"th{thread_id}Elf"] = file

            input_cfg[tc_key] = {key : input_neo[key] for key in sorted(input_neo.keys())}

        input_cfg_dict["description"] = dict()

        return input_cfg_dict

    @staticmethod
    def write(test_name: str, rtl_tag: str, input_params: InputParams) -> str:
        input_cfg_dict = Inputcfg.get(test_name, rtl_tag, input_params)
        file_name = Inputcfg.get_file_name_incl_path(test_name, rtl_tag, input_params)

        if not os.path.isdir(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name), exist_ok = True)

        with open(file_name, 'w') as file:
            json.dump(input_cfg_dict, file, indent = 2)

        return file_name

class RTLTest:
    @staticmethod
    def prepare(rtl_tag: str, input_params: InputParams):
        if rtl_tag not in input_params.rtl_tags:
            raise Exception(f"RTL tag not found: {rtl_tag}")

        if not input_params.has_config_files(rtl_tag):
            raise Exception(f"Configuration files not found for RTL tag: {rtl_tag}")

        # input_params.copy_isa_file_from_rtl_data(rtl_tag)

    @staticmethod
    def execute(test_id: int, test_name: str, rtl_tag: str, input_params: InputParams, prepare: bool = True):
        if prepare:
            RTLTest.prepare(rtl_tag, input_params)

        inputcfg_file_name = os.path.relpath(Inputcfg.write(test_name, rtl_tag, input_params))
        outdir_incl_path = os.path.relpath(input_params.get_odir_path(rtl_tag))
        log_file_name = os.path.join(outdir_incl_path, f"{test_name}{input_params.model_log_file_suffix}")

        print(f"    + Running RTL test {test_id}. {test_name} for tag {rtl_tag}. The output data (including errors) would be written to: {log_file_name}")

        # Build command as a list of arguments (safer than shell=True)
        cmd_args = [
            "python",
            "ttsim/back/tensix_neo/tneoSim.py",
            "--inputcfg", inputcfg_file_name,
        ]

        # Set up environment with PYTHONPATH
        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        with open(log_file_name, "w") as log_file:
            cmd_args = [str(a) if not isinstance(a, (str, bytes, os.PathLike)) else a for a in cmd_args]
            subprocess.run(
                cmd_args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                check=False  # Don't raise exception on non-zero exit
            )

class RTLTests:
    @staticmethod
    def execute_with_tag(rtl_tag: str, input_params: InputParams):
        rtl_test_names = input_params.get_rtl_test_names(rtl_tag)
        num_processes = min([input_params.num_processes, len(rtl_test_names)])
        print(f"  + RTL tag:                                         {rtl_tag}")
        print(f"    + Number of tests to execute via model:          {len(rtl_test_names)}")
        print(f"    + Number of parallel processes to execute tests: {num_processes}")

        RTLTest.prepare(rtl_tag, input_params)

        with multiprocessing.Pool(processes = num_processes) as pool:
            _ = pool.starmap(RTLTest.execute, [(idx, test_name, rtl_tag, input_params, False) for idx, test_name in enumerate(rtl_test_names, 1)])

    @staticmethod
    def execute(input_params: InputParams):
        for rtl_tag in input_params.rtl_tags:
            RTLTests.execute_with_tag(rtl_tag, input_params)

class SCurveElemIndices(enum.IntEnum):
    test_name = 0
    class_name = enum.auto()
    rtl_passed = enum.auto()
    rtl_num_cycles = enum.auto()
    model_passed = enum.auto()
    model_num_cycles = enum.auto()
    num_cycles_model_by_rtl = enum.auto()

class SCurveElem:
    test_name: str
    class_name: str
    rtl_passed: bool | None
    rtl_num_cycles: int | None
    model_passed: bool | None
    model_num_cycles: int | None
    num_cycles_model_by_rtl: float | None

    def __init__(self, elem: tuple):
        self.test_name = elem[SCurveElemIndices.test_name]
        self.class_name = elem[SCurveElemIndices.class_name]
        self.rtl_passed = elem[SCurveElemIndices.rtl_passed]
        self.rtl_num_cycles = elem[SCurveElemIndices.rtl_num_cycles]
        self.model_passed = elem[SCurveElemIndices.model_passed]
        self.model_num_cycles = elem[SCurveElemIndices.model_num_cycles]
        self.num_cycles_model_by_rtl = elem[SCurveElemIndices.num_cycles_model_by_rtl]

    def get_s_curve_str(self: typing.Self, widths: list[int] | None = None) -> str:
        if widths is None:
            widths = [len(self.test_name), len(self.class_name), 4, 8, 6, 8, 8]
        msg  = f"{self.test_name:<{widths[SCurveElemIndices.test_name]}} | "
        msg += f"{self.class_name:<{widths[SCurveElemIndices.class_name]}} | "
        msg += f"RTL: {'PASS' if self.rtl_passed else 'FAIL'} | "
        msg += f"Cycles: {self.rtl_num_cycles if isinstance(self.rtl_num_cycles, int) else '-':>{widths[SCurveElemIndices.rtl_num_cycles]}} | "
        msg += f"Model: {'PASS' if self.model_passed else 'FAIL'} | "
        msg += f"Cycles: {self.model_num_cycles if isinstance(self.model_num_cycles, int) else '-':>{widths[SCurveElemIndices.model_num_cycles]}} | "
        msg += f"Model/RTL: {f'{self.num_cycles_model_by_rtl:.2f}' if isinstance(self.num_cycles_model_by_rtl, float) else '-':>4}"

        return msg

    def to_tuple(self: typing.Self) -> tuple:
        return (
            self.test_name,
            self.class_name,
            self.rtl_passed,
            self.rtl_num_cycles,
            self.model_passed,
            self.model_num_cycles,
            self.num_cycles_model_by_rtl,
        )

class SCurveUtils:
    @staticmethod
    def sort(data: list[SCurveElem], sort_by: None | int | list[int] = None, descending: None | bool | list[bool] = None):
        if sort_by is None or descending is None:
            assert sort_by is None, "expect sort_by to be None as descending flag is None"
            assert descending is None, "expect descending to be None as sort_by flag is None"
            sort_by    = [SCurveElemIndices.model_passed, SCurveElemIndices.rtl_passed, SCurveElemIndices.num_cycles_model_by_rtl, SCurveElemIndices.test_name]
            descending = [False,                           False,                         True,                                       False]

        if isinstance(sort_by, int) or isinstance(descending, bool):
            assert isinstance(sort_by, int), "expect sort_by to be of type int if descending is bool"
            assert isinstance(descending, bool), "expect descending to be of type bool if sort_by is int"
            sort_by = [sort_by]
            descending = [descending]

        assert isinstance(sort_by, list) and all(isinstance(elem, int) for elem in sort_by), "- error: expect sort_by to a list with each element of type int"
        assert isinstance(descending, list) and all(isinstance(elem, bool) for elem in descending), "- error: expect descending to a list with each element of type bool"

        tbl = [elem.to_tuple() for elem in data if elem is not None]
        tbl = Utils.sort_table_by_columns_order(tbl, sort_by, descending)
        data = [SCurveElem(row) for row in tbl]

        return data

class TestStatus:
    name: str | None = None
    test_class: str | None = None
    rtl_passed: bool | None = None
    rtl_num_cycles: int | None = None
    model_passed: bool | None = None
    model_num_cycles: int | None = None
    num_cycles_model_by_rtl: float | None = None
    rtl_tag: str | None = None
    model_log_file_last_line: str | None = None
    failure_bin: str | None = None

    def __init__(self, name: str, rtl_tag: str, input_params: InputParams) -> None:
        self.name = name
        self.rtl_tag = rtl_tag
        self.set_status(input_params)
        self.set_test_class()
        self.set_failure_bin()

    def set_rtl_test_status(self: typing.Self, input_params: InputParams):
        assert isinstance(self.name, str), "- error: test name is not set"
        assert isinstance(self.rtl_tag, str), "- error: rtl_tag is not set"
        self.rtl_passed = False
        test_dir = input_params.get_rtl_test_dir_path(self.name, self.rtl_tag)
        sim_result_incl_path = Utils.get_path_of_only_copy_of_file_in_dir(input_params.rtl_status_file_name, test_dir)
        if os.path.isfile(sim_result_incl_path):
            with open(sim_result_incl_path) as stream:
                yml = yaml.safe_load(stream)
                if "PASS" == yml['res']:
                    self.rtl_passed = True
                    self.rtl_num_cycles = yml['total-cycles']

    def set_model_test_status(self: typing.Self, input_params: InputParams):
        assert isinstance(self.rtl_tag, str), "- error: rtl_tag is not set"
        self.model_passed = False
        test_dir = input_params.get_odir_path(self.rtl_tag)
        log_file = os.path.join(test_dir, f"{self.name}{input_params.model_log_file_suffix}")
        if os.path.isfile(log_file):
            with open(log_file) as file:
                line = f"File {log_file} is empty"
                for line in file:
                    line = line.strip()
                    if line.startswith("Total Cycles"):
                        self.model_passed = True
                        self.model_num_cycles = int(round(float(line.split("=")[1].strip())))
                self.model_log_file_last_line = line

    def set_status(self: typing.Self, input_params: InputParams):
        self.set_rtl_test_status(input_params)
        self.set_model_test_status(input_params)
        if self.model_passed and self.rtl_passed and isinstance(self.model_num_cycles, int) and isinstance(self.rtl_num_cycles, int):
            if 0 == self.rtl_num_cycles:
                print(f"- Number of RTL cycles for test {self.name} from tag {self.rtl_tag} is {self.rtl_num_cycles}")
                self.num_cycles_model_by_rtl = None
            else:
                self.num_cycles_model_by_rtl = float(self.model_num_cycles) / float(self.rtl_num_cycles)

    def set_test_class(self: typing.Self):
        assert isinstance(self.name, str), "- error: test name is not set"
        classes = TestStatusUtils.get_test_classes()
        test_words = self.name.split("-")
        unclassified_class_prefix = "Unclassified test class: "
        test_class = f"{unclassified_class_prefix}{self.name}"
        for key, values in classes.items():
            if test_class.startswith(unclassified_class_prefix) is False:
                break
            for value in values:
                if value in test_words:
                    test_class = key
                    break

        self.test_class = test_class

    def set_failure_bin(self: typing.Self):
        if not self.model_passed:
            bins = TestStatusUtils.get_failure_bins()
            msg = self.model_log_file_last_line
            if msg is None:
                self.failure_bin = "Unclassified failure: No log file found"
                return
            self.failure_bin = "Unclassified failure: " + msg
            for strings in bins:
                if all(s in msg for s in strings):
                    self.failure_bin = " ".join(strings)
                    break

    def get_s_curve_data(self: typing.Self) -> SCurveElem:
        return SCurveElem((
            self.name,
            self.test_class,
            self.rtl_passed,
            self.rtl_num_cycles,
            self.model_passed,
            self.model_num_cycles,
            self.num_cycles_model_by_rtl
        ))

    def get_s_curve_str(self: typing.Self, widths) -> str:
        s_curve      = self.get_s_curve_data()
        msg  = f"{s_curve.test_name:<{widths[SCurveElemIndices.test_name]}} | "
        msg += f"{s_curve.class_name:<{widths[SCurveElemIndices.class_name]}} | "
        msg += f"RTL: {'PASS' if s_curve.rtl_passed else 'FAIL'} | "
        msg += f"Cycles: {s_curve.rtl_num_cycles if isinstance(s_curve.rtl_num_cycles, int) else '-':>{widths[SCurveElemIndices.rtl_num_cycles]}} | "
        msg += f"Model: {'PASS' if s_curve.model_passed else 'FAIL'} | "
        msg += f"Cycles: {s_curve.model_num_cycles if isinstance(s_curve.model_num_cycles, int) else '-':>{widths[SCurveElemIndices.model_num_cycles]}} | "
        msg += f"Model/RTL: {f'{s_curve.num_cycles_model_by_rtl:.2f}' if isinstance(s_curve.num_cycles_model_by_rtl, float) else '-':>4}"

        return msg

    def __str__(self: typing.Self):
        msg = f"Status: test: {self.name}, rtl_tag = {self.rtl_tag}, rtl_passed = {self.rtl_passed}, rtl_num_cycles = {self.rtl_num_cycles}, model_passed = {self.model_passed}, estimated number of cycles: {self.model_num_cycles}, num_cycles_estimated_by_rtl = {self.num_cycles_model_by_rtl}"
        return msg

    def __repr__(self) -> str:
        return self.__str__()


class SCurve:
    def __init__(self: typing.Self, rtl_tag: str, s_curve: list[SCurveElem]| None = None) -> None:
        self.rtl_tag = rtl_tag
        self.s_curve: list[SCurveElem] = []
        if s_curve is not None:
            for elem in s_curve:
                assert isinstance(elem, SCurveElem), "- error: expect elem to be of type SCurveElem"
                self.s_curve.append(elem)

        self.s_curve = SCurveUtils.sort(self.s_curve)

    def slower_than_RTL_tests(self: typing.Self) -> list[SCurveElem]:
        elems: list[SCurveElem] = []
        for elem in self.s_curve:
            if elem.rtl_passed and elem.model_passed and isinstance(elem.model_num_cycles, int) and isinstance(elem.rtl_num_cycles, int) and (elem.model_num_cycles > elem.rtl_num_cycles):
                elems.append(elem)
        return elems


    def get_num_slower_than_rtl_tests(self: typing.Self) -> int:
        return len(self.slower_than_RTL_tests())

    def geometric_mean_of_model_by_rtl_ratios(self: typing.Self) -> float | None:
        ratios = [elem.num_cycles_model_by_rtl for elem in self.s_curve if (elem.rtl_passed and elem.model_passed and isinstance(elem.num_cycles_model_by_rtl, float))]
        if 0 == len(ratios):
            return None

        return math.e**(sum(math.log(ratio) for ratio in ratios) / len(ratios))

    def get_s_curve_str(self: typing.Self, num_white_chars_at_start: int = 0) -> str:
        prefix = f"{' ' * num_white_chars_at_start}"
        sr_num_max_len = int(math.ceil(math.log10(len(self.s_curve))) if len(self.s_curve) > 0 else 1)
        num_tests = len(self.s_curve)
        widths = [0 for _ in range(SCurveElemIndices.num_cycles_model_by_rtl + 1)]
        widths[SCurveElemIndices.test_name]        = max(len(t.test_name) for t in self.s_curve)
        widths[SCurveElemIndices.class_name]       = max(len(t.class_name) for t in self.s_curve)
        widths[SCurveElemIndices.rtl_num_cycles]   = max(len(str(t.rtl_num_cycles)) for t in self.s_curve)
        widths[SCurveElemIndices.model_num_cycles] = max(len(str(t.model_num_cycles)) for t in self.s_curve)
        msg = ""
        for idx, elem in enumerate(self.s_curve):
            msg += f"{prefix}[{(idx + 1):>{sr_num_max_len}}/{num_tests}] {elem.get_s_curve_str(widths)}\n"
        return msg.rstrip()

    def get_test_class_wise_s_curve(self: typing.Self) -> dict[str, list[SCurveElem]]:
        class_wise_s_curve: dict[str, list[SCurveElem]] = {test_cls : [] for test_cls in sorted({elem.class_name for elem in self.s_curve})}

        for elem in self.s_curve:
            class_wise_s_curve[elem.class_name].append(elem)

        return class_wise_s_curve

    # def failed_tests_on_model(self: typing.Self):
    #     failed_tests = [elem for elem in self.s_curve if not elem.model_passed]
    #     return SCurve(self.rtl_tag, failed_tests)

    # def failed_tests_on_RTL(self: typing.Self):
    #     failed_tests = [elem for elem in self.s_curve if not elem.rtl_passed]
    #     return SCurve(self.rtl_tag, failed_tests)

    # def failed_tests(self: typing.Self, test_type: str):
    #     if "model" == test_type:
    #         return self.failed_tests_on_model()
    #     elif "RTL" == test_type:
    #         return self.failed_tests_on_RTL()
    #     else:
    #         raise ValueError(f"Unknown test type: {test_type}")

class TagStatus:
    rtl_tag: str = ""
    tests: list[str] = list()
    statuses: dict[str, TestStatus] = {}
    s_curve: SCurve = SCurve("", [])

    def __init__(self, rtl_tag: str, input_params: InputParams) -> None:
        self.rtl_tag = rtl_tag
        self.tests = sorted(set(input_params.get_rtl_test_names(rtl_tag)))
        self.set_tests_statuses(input_params)
        self.set_s_curve()

    def set_tests_statuses(self: typing.Self, input_params: InputParams):
        self.statuses = {test_name : TestStatus(test_name, self.rtl_tag, input_params) for test_name in self.tests}

    def set_s_curve(self: typing.Self, sort_by: None | int | list[int] = None, descending: None | bool | list[bool] = None) -> None:
        s_curve_data: list[SCurveElem] = []
        for test_name in self.tests:
            s_curve_data.append(self.statuses[test_name].get_s_curve_data())
        self.s_curve = SCurve(self.rtl_tag, s_curve_data)

    def get_s_curve_str(self: typing.Self, num_white_chars_at_start: int = 0) -> str:
        return self.s_curve.get_s_curve_str(num_white_chars_at_start)

    def num_tests(self: typing.Self):
        return len(self.statuses)

    def num_tests_passed_on_rtl(self: typing.Self):
        return sum(1 for status in self.statuses.values() if status.rtl_passed)

    def num_tests_passed_on_model(self: typing.Self):
        return sum(1 for status in self.statuses.values() if status.model_passed)

    def num_tests_status_to_str(self: typing.Self) -> str:
        num_tests = self.num_tests()
        num_rtl_passed = self.num_tests_passed_on_rtl()
        num_model_passed = self.num_tests_passed_on_model()
        num_rtl_failed = num_tests - num_rtl_passed
        num_model_failed = num_tests - num_model_passed
        num_slower_tests = self.s_curve.get_num_slower_than_rtl_tests()

        assert num_tests == (num_rtl_passed + num_rtl_failed), f"- error: expected number of tests in summary {num_tests} to be equal to number of passed and failed RTL tests for RTL tag {self.rtl_tag}"
        assert num_tests == (num_model_passed + num_model_failed), f"- error: expected number of tests in summary {num_tests} to be equal to number of passed and failed model tests for RTL tag {self.rtl_tag}"
        assert num_slower_tests <= num_tests, f"- error: number of slower tests can not be greater than total number of RTL tests!"

        msg = f"  + Status for RTL tag: {self.rtl_tag}. "
        msg += f"Number of tests: {num_tests}, "
        msg += f"number of tests passing on RTL: {num_rtl_passed} ({(num_rtl_passed / num_tests * 100):.2f} %), "
        msg += f"number of tests passing on model: {num_model_passed} ({(num_model_passed / num_tests * 100):.2f} %), "
        msg += f"number of tests slower than RTL: {num_slower_tests} ({(num_slower_tests / num_tests * 100):.2f} %)"

        return msg.rstrip()

    def get_test_class_wise_failure_bins(self: typing.Self) -> dict[str, dict[str, list[str]]]:
        test_classes_bins_tests: dict[str, dict[str, list[str]]] = dict()
        for test_name, status in self.statuses.items():
            if not status.model_passed:
                assert status.test_class is not None, f"- error: expected test class to be set for test {test_name}"
                assert status.failure_bin is not None, f"- error: expected failure bin to be set for test {test_name}"
                if status.test_class not in test_classes_bins_tests:
                    test_classes_bins_tests[status.test_class] = dict()
                if status.failure_bin not in test_classes_bins_tests[status.test_class]:
                    test_classes_bins_tests[status.test_class][status.failure_bin] = []
                test_classes_bins_tests[status.test_class][status.failure_bin].append(test_name)
        return test_classes_bins_tests

    def get_test_class_wise_failure_bins_str(self: typing.Self, num_white_chars_at_start: int = 0) -> str:
        msg = ""
        failed_tests = self.get_test_class_wise_failure_bins()
        for test_class, failure_bins in failed_tests.items():
            num_failed_tests = sum(len(tests) for tests in failure_bins.values())
            msg += f"{' ' * num_white_chars_at_start}- Test class: {test_class} ({num_failed_tests} tests)\n"
            for failure_bin, tests in failure_bins.items():
                msg += f"{' ' * (num_white_chars_at_start + 2)}- Failure bin: {failure_bin} ({len(tests)} tests)\n"
                for test in tests:
                    msg += f"{' ' * (num_white_chars_at_start + 4)}- Test: {test}\n"
        return msg.rstrip()

    def failed_tests_to_str(self: typing.Self, num_white_chars_at_start: int = 0) -> str:
        msg = ""
        failed_tests_on_rtl = [elem for elem in self.s_curve.s_curve if not elem.rtl_passed]
        if len(failed_tests_on_rtl):
            num_tests = len(failed_tests_on_rtl)
            sr_num_max_len = int(math.ceil(math.log10(num_tests))) if num_tests > 0 else 1
            widths = [0 for _ in range(SCurveElemIndices.num_cycles_model_by_rtl + 1)]
            widths[SCurveElemIndices.test_name]        = max(len(t.test_name) for t in failed_tests_on_rtl)
            widths[SCurveElemIndices.class_name]       = max(len(t.class_name) for t in failed_tests_on_rtl)
            widths[SCurveElemIndices.rtl_num_cycles]   = max(len(str(t.rtl_num_cycles)) for t in failed_tests_on_rtl)
            widths[SCurveElemIndices.model_num_cycles] = max(len(str(t.model_num_cycles)) for t in failed_tests_on_rtl)
            msg += f"{' ' * num_white_chars_at_start}+ Failed tests on RTL:\n"
            for idx, elem in enumerate(failed_tests_on_rtl):
                msg += f"{' ' * (num_white_chars_at_start + 2)}[{(idx + 1):>{sr_num_max_len}}/{num_tests:>{sr_num_max_len}}] {elem.get_s_curve_str(widths=widths)}\n"

        failed_tests_on_model = [elem for elem in self.s_curve.s_curve if not elem.model_passed]
        if len(failed_tests_on_model):
            failed_tests = self.get_test_class_wise_failure_bins()
            num_failed_tests = 0
            for failure_bins in failed_tests.values():
                for tests in failure_bins.values():
                    num_failed_tests += len(tests)

            assert len(failed_tests_on_model) == num_failed_tests, f"- error: number of failed tests mismatch"
            msg += f"{' ' * num_white_chars_at_start}+ Failed tests on model ({num_failed_tests} tests):\n"
            msg += self.get_test_class_wise_failure_bins_str(num_white_chars_at_start + 2)

        return msg.rstrip()

    def get_s_curve_geometric_mean(self: typing.Self) -> float | None:
        return self.s_curve.geometric_mean_of_model_by_rtl_ratios()

    def status_to_str(self: typing.Self) -> str:
        msg = ""
        msg += self.num_tests_status_to_str() + "\n"
        failed_test_msg = self.failed_tests_to_str(num_white_chars_at_start=4)
        if failed_test_msg:
            msg += failed_test_msg + "\n"
        msg += "    + Test class S-Curve:\n"
        msg += self.get_s_curve_str(num_white_chars_at_start=6) + "\n"
        geom_mean = self.get_s_curve_geometric_mean()
        msg += f"    + S-Curve (Model/RTL) geometric mean: {f'{geom_mean:.2f}' if geom_mean is not None else 'N/A'}\n"

        return msg.rstrip()

    def print_status(self: typing.Self):
        print(self.status_to_str())

    def s_curve_plot_data(self: typing.Self) -> tuple[list[str], list[float]]:
        x: list[str] = []
        y: list[float] = []
        for elem in reversed(self.s_curve.s_curve):  # we want the plot to go from min to max
            if isinstance(elem.num_cycles_model_by_rtl, float):
                x.append(elem.test_name)
                y.append(elem.num_cycles_model_by_rtl)

        return (x, y)

    def test_class_wise_s_curve_plot_data(self: typing.Self) -> dict[str, tuple[list[str], list[float]]]:
        result: dict[str, tuple[list[str], list[float]]] = {}
        for elem in reversed(self.s_curve.s_curve):  # we want the plot to go from min to max
            if isinstance(elem.num_cycles_model_by_rtl, float):
                class_name = str(elem.class_name)
                if class_name not in result:
                    result[class_name] = ([], [])
                result[class_name][0].append(elem.test_name)
                result[class_name][1].append(elem.num_cycles_model_by_rtl)
        return result

    def plot_s_curve(self: typing.Self, output_file: str):
        s_curve_data = self.s_curve_plot_data()
        if 0 == len(s_curve_data[0]):
            print(f"  - info: No data available for S-Curve plot for RTL tag {self.rtl_tag}")
            return
        plt.figure(figsize=(10 * len(s_curve_data[0])/40., 6 * len(s_curve_data[0])/40.))
        plt.plot(s_curve_data[0], s_curve_data[1], marker='o')
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Model/RTL = 1')
        plt.title(f"S-Curve for RTL Tag: {self.rtl_tag}")
        plt.xlabel("Tests")
        plt.ylabel("Number of cycles ratio (Model/RTL)")
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        print(f"    + Saving S-Curve plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
        plt.close()

    def plot_class_wise_s_curve(self: typing.Self, output_file: str):
        class_wise_data = self.test_class_wise_s_curve_plot_data()
        num_tests = sum(len(v[0]) for v in class_wise_data.values())
        if 0 == num_tests:
            print(f"  - info: No data available for class-wise S-Curve plot for RTL tag {self.rtl_tag}")
            return
        plt.figure(figsize=(10 * num_tests / 40., 6 * num_tests / 40.))
        for class_name in sorted(class_wise_data.keys()):
            x, y = class_wise_data[class_name]
            plt.plot(x, y, marker='o', label=class_name)
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Model/RTL = 1')
        plt.title(f"Class-wise S-Curve for RTL Tag: {self.rtl_tag}")
        plt.xlabel("Tests")
        plt.ylabel("Number of cycles ratio (Model/RTL)")
        plt.xticks(rotation=90)
        plt.legend(loc='upper left')
        plt.tight_layout()
        print(f"    + Saving test class wise S-Curve plot to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
        plt.close()

    def write_s_curve_to_csv(self: typing.Self, output_file: str):
        """Write S-curve data to CSV file with header"""
        print(f"    + Saving S-Curve data to CSV file: {output_file}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = [
                'Test Name',
                'Test Class',
                'RTL Passed',
                'RTL Cycles',
                'Model Passed',
                'Model Cycles',
                'Model/RTL Ratio'
            ]
            writer.writerow(header)

            # Write data rows
            for elem in self.s_curve.s_curve:
                row = [
                    elem.test_name,
                    elem.class_name,
                    'PASS' if elem.rtl_passed else 'FAIL',
                    elem.rtl_num_cycles if elem.rtl_num_cycles is not None else '',
                    'PASS' if elem.model_passed else 'FAIL',
                    elem.model_num_cycles if elem.model_num_cycles is not None else '',
                    f'{elem.num_cycles_model_by_rtl:.4f}' if elem.num_cycles_model_by_rtl is not None else ''
                ]
                writer.writerow(row)

    def __str__(self: typing.Self):
        return self.status_to_str()

    def __repr__(self) -> str:
        return self.__str__()
class Status:
    @staticmethod
    def get_test_statuses(rtl_tag: str, input_params: InputParams, order_by = "classwise"):
        test_names = input_params.get_rtl_test_names(rtl_tag)
        return {test_name : TestStatus(test_name, rtl_tag, input_params) for test_name in test_names}

    @staticmethod
    def print_status(rtl_tag: str, input_params: InputParams, order_by: str = "classwise-s-curve"):
        print(f"- in function print status. order by: {order_by}")
        test_statuses = Status.get_test_statuses(rtl_tag, input_params)
        max_len_test_name = max(len(test_name) for test_name in test_statuses.keys()) if test_statuses else 0
        max_len_num_tests = int(round(math.ceil(math.log10(len(test_statuses))))) if test_statuses else 1
        msg  = f"  Test Statuses for RTL Tag: {rtl_tag}\n"
        msg += f"    Sr no, Test Name, RTL status, RTL num cycles, Model status, estimated num cycles, num_cycles_estimated_by_RTL"
        if "classwise-s-curve" == order_by:
            classes = TestStatusUtils.get_test_classes()
            for class_name in sorted(classes.keys()):
                msg += f"\n    - Test class: {class_name}"
                class_test_names = [test_name for test_name in test_statuses.keys() if test_statuses[test_name].test_class == class_name]
                for idx, test_name in enumerate(sorted(class_test_names, key = lambda tn: (not test_statuses[tn].model_passed, not test_statuses[tn].rtl_passed, -float(test_statuses[tn].num_cycles_model_by_rtl) if isinstance(test_statuses[tn].num_cycles_model_by_rtl, float) else math.inf, test_statuses[tn].name))):
                    status = test_statuses[test_name]
                    msg += f"\n      [{idx + 1:>{max_len_num_tests}}/{len(class_test_names):>{max_len_num_tests}}] {test_name:<{max_len_test_name}}: "
                    msg += f"{status.rtl_passed}, {status.rtl_num_cycles}, {status.model_passed}, {status.model_num_cycles}"
                    if status.rtl_passed and status.model_passed:
                        msg += f", {status.num_cycles_model_by_rtl:.2f}"
                    else:
                        msg += ", -"
        elif "s-curve" == order_by:
            for idx, test_name in enumerate(sorted(test_statuses.keys(), key = lambda tn: (not test_statuses[tn].model_passed, not test_statuses[tn].rtl_passed, -float(test_statuses[tn].num_cycles_model_by_rtl) if isinstance(test_statuses[tn].num_cycles_model_by_rtl, float) else math.inf, test_statuses[tn].name))):
                status = test_statuses[test_name]
                msg += f"\n  [{idx + 1:>{max_len_num_tests}}/{len(test_statuses):>{max_len_num_tests}}] {test_name:<{max_len_test_name}}: "
                msg += f"{status.rtl_passed}, {status.rtl_num_cycles}, {status.model_passed}, {status.model_num_cycles}"
                if status.rtl_passed and status.model_passed:
                    msg += f", {status.num_cycles_model_by_rtl:.2f}"
                else:
                    msg += ", -"
        else:
            for idx, test_name in enumerate(sorted(test_statuses.keys())):
                status = test_statuses[test_name]
                msg += f"\n  [{idx + 1:>{max_len_num_tests}}/{len(test_statuses):>{max_len_num_tests}}] {test_name:<{max_len_test_name}}: {status.rtl_passed}, {status.rtl_num_cycles}, {status.model_passed}, {status.model_num_cycles}, {status.num_cycles_model_by_rtl:.2f}"
        print(msg)

    @staticmethod
    def print_statuses(input_params: InputParams, order_by: str = "classwise-s-curve"):
        for rtl_tag in input_params.rtl_tags:
            tgs = TagStatus(rtl_tag, input_params)
            tgs.print_status()
            tgs.plot_s_curve(output_file = os.path.join(input_params.get_odir_path(rtl_tag), f"s_curve.png"))
            tgs.plot_class_wise_s_curve(output_file = os.path.join(input_params.get_odir_path(rtl_tag), f"class_wise_s_curve.png"))
            tgs.write_s_curve_to_csv(output_file = os.path.join(input_params.get_odir_path(rtl_tag), f"s_curve.csv"))

        input_params.write_run_config_to_inputcfg_dir()

    @staticmethod
    def get_status(rtl_tag: str, input_params: InputParams):
        test_statuses = Status.get_test_statuses(rtl_tag, input_params)
        return all((status.model_passed and status.rtl_passed) for status in test_statuses.values())

    @staticmethod
    def get_statuses(input_params: InputParams):
        return {rtl_tag : Status.get_status(rtl_tag, input_params) for rtl_tag in input_params.rtl_tags}

def execute_tests_and_get_status(run: Run) -> dict[str, bool]:
    """Execute tests for a single run and return status"""
    input_params = InputParams(
        run_name = run.name,
        tags = run.tags,
        tests = run.tests,
        parallel = run.parallel,
        cfg = run.cfg,
        debug = run.debug,
        defCfg = run.defCfg,
        exp = run.exp,
        memoryMap = run.memoryMap,
        odir = run.odir,
        risc_cpi = run.risc_cpi,
        ttISAFileName = run.ttISAFileName
    )

    print(f"+ Executing run: {run.name}")
    print(f"  + Input parameters:")
    print(input_params.to_str(offset = 4))
    RTLTests.execute(input_params)
    Status.print_statuses(input_params)
    return Status.get_statuses(input_params)

def execute_all_runs(run_config: RunConfig) -> dict[str, dict[str, bool]]:
    """Execute all runs in the configuration and return results"""
    results = {}
    for run in run_config.runs:
        results[run.name] = execute_tests_and_get_status(run)
    return results

if "__main__" == __name__:
    # Set up argument parser
    parser = argparse.ArgumentParser(description = 'Execute RTL tests via model')
    parser.add_argument('--tag', nargs = '+', default = None, help = 'Optional: RTL tags to execute tests with (e.g., feb19 mar18 jul1 jul27 sep23)')
    parser.add_argument('--test', nargs = '+', default = None, help = 'Optional: Specific test names to run (default: run all tests)')
    parser.add_argument('--parallel', '-j', '-np', type = int, default = None, help = 'Optional: Number of parallel processes to use')
    parser.add_argument('--batch-file', type = str, default = None, help = 'Optional: YAML batch file with test runs configuration')
    parser.add_argument('--debug', type = int, default = 15, help = 'Optional: Debug level (default: 15)')
    parser.add_argument('--cfg', type = str, default = None, help = 'Optional: Configuration file path')
    parser.add_argument('--defCfg', type = str, default = None, help = 'Optional: Default configuration file path')
    parser.add_argument('--exp', type = str, default = None, help = 'Optional: Experiment name or identifier')
    parser.add_argument('--memoryMap', type = str, default = None, help = 'Optional: Memory map file path')
    parser.add_argument('--odir', type = str, default = None, help = 'Optional: Output directory path')
    parser.add_argument('--risc.cpi', type = float, default = None, help = 'Optional: RISC CPI (cycles per instruction) value')
    parser.add_argument('--ttISAFileName', type = str, default = None, help = 'Optional: TT ISA file name')
    parser.add_argument('--batch-name', type = str, default = None, help = 'Optional: Name for the run')

    args = parser.parse_args()

    # Create run configuration
    run_config = RunConfig(args.batch_file)

    # If any CLI arguments are provided, create a run from them
    if (args.test or args.tag or args.parallel or args.debug != 15 or args.cfg or args.defCfg or args.exp or args.memoryMap or args.odir or getattr(args, 'risc.cpi', None) or args.ttISAFileName) or not run_config.runs:
        run_config.add_cli_run(
            tests=args.test,
            tags=args.tag,
            parallel=args.parallel,
            debug=args.debug,
            cfg=args.cfg,
            defCfg=args.defCfg,
            exp=args.exp,
            memoryMap=args.memoryMap,
            odir=args.odir,
            risc_cpi=getattr(args, 'risc.cpi', None),
            ttISAFileName=args.ttISAFileName,
            name=args.batch_name
        )

    # Execute all runs
    results = execute_all_runs(run_config)

    # Print overall results summary
    print("+ Overall results summary:")
    for run_name, run_results in results.items():
        print(f"  + Batch '{run_name}': {run_results}")
