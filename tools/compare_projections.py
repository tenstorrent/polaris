#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import logging
from jinja2 import Environment, PackageLoader, select_autoescape, StrictUndefined
from pydantic import BaseModel, model_validator
import math
from enum import Enum
from pathlib import Path
from collections import namedtuple, Counter, defaultdict
from ttsim.utils.common import print_csv
from typing import Tuple, Any, Literal

logfile = open('comparison-log.txt', 'w')

type IndexTuple = Tuple[int|None, int|None]

ATTRIBUTES_TO_SKIP: set[str] = {'type', 'stat_filename'}

class ComparisonStatus(str, Enum):
    Only_in_1 = 'keys_only_in_1'
    Only_in_2 = 'keys_only_in_2'
    TypeMismatch = 'type_mismatch'
    Mismatch = 'mismatch'
    ApproxMatch = 'approx_match'
    Match = 'match'


def flatten_dict(d: dict[str, Any], param_sep: str='_') -> dict[str, Any]:
    """
        Flatten a nested dictionary into a single level dictionary with keys
        concatenated by sep.
        :param d: The dictionary to flatten
        :param param_sep: Separator
        :return: A flattened dictionary
    """
    def _flatten(d: dict[str, Any], parent_key: str, param_sep: str) -> dict[str, Any]:
        items: list[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{param_sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key, param_sep=param_sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(_flatten(item, f"{new_key}{param_sep}{i}", param_sep=param_sep).items())
                    else:
                        items.append((f"{new_key}{param_sep}{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)
    return _flatten(d, parent_key='', param_sep=param_sep)


JobKey = namedtuple('JobKey', ['devname', 'wlcls', 'wlname', 'wlinstance', 'bs'])

def jobkey_2_str(key: JobKey) -> str:
    return f'{key.devname}_{key.wlcls}_{key.wlname}_{key.wlinstance}_b{key.bs}'

def tuple_2_jobkey(key: tuple) -> JobKey:
    return JobKey(*key)

def summary_index(summary_list: list[dict[str, Any]]) -> dict[JobKey, dict]:
    row_dict: dict[JobKey, dict[str, Any]] = {}
    for row in summary_list:
        assert isinstance(row, dict)
        key_attrs = {x: row[x] for x in ['devname', 'wlcls', 'wlname', 'wlinstance', 'bs']}
        k = JobKey(**key_attrs)
        row_dict[k] = row
    return row_dict


def is_value_numeric(v: Any) -> bool:
    return isinstance(v, (int, float))

def status_2_freqcount(values: list[ComparisonStatus]) -> Counter:
    status_count: Counter = Counter()
    for stat in values:
        assert isinstance(stat, ComparisonStatus)
        status_count[stat] += 1
    return status_count

def rollup_status(status_list: list[ComparisonStatus]) -> ComparisonStatus:
    assert isinstance(status_list, list)
    status_count: Counter = status_2_freqcount(status_list)
    # Check the most severe status first
    found_status = [x for x in [ComparisonStatus.Mismatch, ComparisonStatus.TypeMismatch,
                       ComparisonStatus.ApproxMatch, ComparisonStatus.Match]
           if status_count[x] != 0]
    return found_status[0]



def status_2_csvrow(summary: dict[str, Any]) -> dict[str, Any]:
    csv_row = {}
    for k2, v2 in summary.items():
        if v2['type'] == 'non-numeric':
            entry = {
                'Stat': v2['status'].value,
                'Value1': v2['value1'],
                'Value2': v2['value2'],
            }
        elif v2['type'] == 'numeric':
            entry = {
                'Stat': v2['status'].value,
                'Value1': v2['value1'],
                'Value2': v2['value2'],
                'Diff': v2['diff'],
                'Ratio': v2['ratio'],
            }
        else:
            raise NotImplementedError(f'{v2["type"]} not implemented')
        csv_entry = {(k2 + '_' + k3).lower(): v3 for k3, v3 in entry.items()}
        csv_row.update(csv_entry)
    return csv_row


def compare_value(value1: Any, value2: Any, epsilon: float) -> dict[str, Any]:
    result = {}
    result['value1'], result['value2'] = value1, value2
    if is_value_numeric(value1) and is_value_numeric(value2):
        if value1 == value2:
            result['status'] = ComparisonStatus.Match
        elif math.isclose(value1, value2, rel_tol=epsilon):
            result['status'] = ComparisonStatus.ApproxMatch
        else:
            result['status'] = ComparisonStatus.Mismatch
        result['diff'] = value1 - value2
        result['ratio'] = value1 / value2 if value2 != 0 else float('inf')
        result['type'] = 'numeric'
        assert 'status' in result
        return result
    if isinstance(value1, (list, dict)) and isinstance(value2, (list, dict)):
        result['type'] = 'container'
        result['status'] = ComparisonStatus.Match
        return result
    result['type'] = 'non-numeric'
    if value1 == value2:
        result['status'] = ComparisonStatus.Match
    elif type(value1) is not type(value2):
        result['status'] = ComparisonStatus.TypeMismatch
    else:
        result['status'] = ComparisonStatus.Mismatch
    assert 'status' in result
    return result

def next_matching_op(opstats: list[dict[str, Any]], start_index: int, ref_op: dict[str, Any]) -> int | None:
    matching_indices = (
        ndx for ndx in range(start_index, len(opstats)) if opstats[ndx]['optype'] == ref_op['optype']
    )
    assert not isinstance(matching_indices, list)  # it MUST be a generator, we do NOT want to create a list
    try:
        return next(matching_indices)
    except StopIteration:
        return None


def pair_layers(opstatlist_1: list[dict[str, Any]], opstatlist_2: list[dict[str, Any]]) -> list[IndexTuple]:
    pairs: list[IndexTuple] = []
    ndx1 = ndx2 = 0
    while ndx1 < len(opstatlist_1) and ndx2 < len(opstatlist_2):
        next1 = next_matching_op(opstatlist_1, ndx1, opstatlist_2[ndx2])
        next2 = next_matching_op(opstatlist_2, ndx2, opstatlist_1[ndx1])
        if next1 == ndx1 and next2 == ndx2:
            pairs.append((ndx1, ndx2,))
            ndx1, ndx2 = ndx1 + 1, ndx2 + 1
            continue
        if next1 is None:
            pairs.append((None, ndx2,))
            ndx2 += 1
            continue
        if next2 is None:
            pairs.append((ndx1, None,))
            ndx1 += 1
            continue
        logging.error('could not match %d:%s and %d:%s', ndx1, opstatlist_1[ndx1]['optype'],
                      ndx2, opstatlist_2[ndx2]['optype'])
        logging.error('closest match to %d is %s', ndx1, next1)
        logging.error('closest match to %d is %s', ndx2, next2)
        raise NotImplementedError('ambiguous layer matching not implemented')
    return pairs


def compare_operator_stats(opstatlist_1: list, opstatlist_2: list, epsilon: float) -> dict:
    pairs: list[IndexTuple] = pair_layers(opstatlist_1, opstatlist_2)
    op_status: dict[int, dict] = {ndx: {} for ndx in range(len(pairs))}
    result = {
        'elem_status': op_status,
        'rollup_status': ComparisonStatus.Mismatch,
    }
    for pair_no, (ndx1, ndx2) in enumerate(pairs):
        if ndx1 is None:
            op_status[pair_no] = {'rollup_status': ComparisonStatus.Only_in_2}
            continue
        elif ndx2 is None:
            op_status[pair_no] = {'rollup_status': ComparisonStatus.Only_in_1}
            continue
        op_status[pair_no] = compare_dicts(opstatlist_1[ndx1], opstatlist_2[ndx2], epsilon)
    result['rollup_status'] = rollup_status([op_status[ndx]['rollup_status'] for ndx in op_status])
    return result


def compare_dicts(dict1: dict, dict2: dict, epsilon: float) -> dict[str, Any]:
    result = {}
    keys1, keys2 = set(dict1.keys()), set(dict2.keys())
    elem_status: dict[str, Any] = {k: {'status': ComparisonStatus.Mismatch} for k in keys1 | keys2}
    elem_status.update({k: {'status': ComparisonStatus.Only_in_2} for k in keys2 - keys1})
    elem_status.update({k: {'status': ComparisonStatus.Only_in_1} for k in keys1 - keys2})
    for k in keys1 & keys2:
        elem_status[k] = compare_value(dict1[k], dict2[k], epsilon)
    result = {
        'elem_status': elem_status,
        'rollup_status': rollup_status([elem_status[k]['status'] for k in elem_status]),
    }
    return result



def classify_keys(dict_1: dict, dict_2: dict)->dict:
    return {
        'keys_common': dict_1.keys() & dict_2.keys(),
        'keys_only_in_1': dict_1.keys() - dict_2.keys(),
        'keys_only_in_2': dict_2.keys() - dict_1.keys(),
        'keys_all': dict_1.keys() | dict_2.keys(),
    }


class ProjectionRun:
    def __init__(self, path: Path, study: str) -> None:
        self.rootpath = path
        self.study = study
        self.studypath = self.rootpath / self.study

        self.subdirs = {d for d in self.studypath.iterdir() if d.is_dir()}
        if {sd.stem for sd in self.subdirs} != {'CONFIG', 'STATS', 'SUMMARY'}:
            raise RuntimeError(f'{self.rootpath} contains unexpected subdirectories: {self.subdirs}')

        summary_file = self.studypath / 'SUMMARY' / 'study-summary.json'
        self.name = path.name
        tmp_dict = json.load(summary_file.open('r'))
        self.summary_dict = summary_index(tmp_dict['summary'])
        self.runinfo = json.load((self.rootpath / 'inputs' / 'runinfo.json').open('r'))
        self.confignames = {d.stem for d in (self.studypath / 'CONFIG').iterdir() if d.is_file()}

    def classify_keys(self, run2: 'ProjectionRun')->dict:
        return classify_keys(self.summary_dict, run2.summary_dict)

    def statfilename(self, key: JobKey) -> str:
        return self.summary_dict[key]['stat_filename']

    def load_stat(self, key: JobKey) -> dict:
        statfname: str = self.summary_dict[key]['stat_filename']
        statpath: Path = self.rootpath / statfname
        val = json.load(statpath.open('r'))
        flat_operator_stats = []
        for opstat in val['operatorstats']:
            flat_operator_stats.append(flatten_dict(opstat))
        val['operatorstats'] = flat_operator_stats
        return val


class StudyComparison:
    def __init__(self, path1: Path, path2: Path, output_path: Path, epsilon: float, study: str) -> None:
        self.study = study
        self.run1 = ProjectionRun(path1, self.study)
        self.run2 = ProjectionRun(path2, self.study)
        self.epsilon = epsilon
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.keys_classified = self.run1.classify_keys(self.run2)

    def compare_config_dir(self) -> dict[str, Any]:
        configs_common = self.run1.confignames & self.run2.confignames
        configs_only_in_1 = self.run1.confignames - self.run2.confignames
        configs_only_in_2 = self.run2.confignames - self.run1.confignames
        config_status = {cfg: {'status': ComparisonStatus.Mismatch} for cfg in self.run1.confignames | self.run2.confignames}
        config_status.update({cfg: {'status': ComparisonStatus.Only_in_1} for cfg in configs_only_in_1})
        config_status.update({cfg: {'status': ComparisonStatus.Only_in_2} for cfg in configs_only_in_2})
        for cfg in configs_common:
            config1 = flatten_dict(json.load((self.run1.studypath / 'CONFIG' / f'{cfg}.json').open('r')))
            config2 = flatten_dict(json.load((self.run2.studypath / 'CONFIG' / f'{cfg}.json').open('r')))
            config_status[cfg] = compare_dicts(config1, config2, self.epsilon)
        result = {
            'elem_status': config_status,
            'rollup_status': rollup_status([cfgstatus['rollup_status'] for cfgstatus in config_status.values()]),
        }
        return result


    def compare_summary_dir(self) -> dict[str, Any]:
        os.makedirs(self.output_path / self.study, exist_ok=True)
        keys_classified = self.keys_classified
        keys_common = keys_classified['keys_common']
        keys_only_in_1 = keys_classified['keys_only_in_1']
        keys_only_in_2 = keys_classified['keys_only_in_2']
        # Compare the two summaries
        job_status = {}
        job_status.update({jobkey_2_str(k): {'rollup_status': ComparisonStatus.Only_in_1} for k in keys_only_in_1})
        job_status.update({jobkey_2_str(k): {'rollup_status': ComparisonStatus.Only_in_2} for k in keys_only_in_2})
        csv_table: list[dict[str, Any]] = []
        for k in keys_common:
            job_summary = compare_dicts(self.run1.summary_dict[k], self.run2.summary_dict[k], self.epsilon)
            job_status[jobkey_2_str(k)] = job_summary
            csv_row = status_2_csvrow(job_summary['elem_status'])
            csv_table.append(csv_row)

        print_csv(sorted(csv_table[0].keys()), csv_table, self.output_path / self.study / 'summary-comparison.csv')

        summary_status = rollup_status([entry['rollup_status'] for k, entry in job_status.items()])
        return {
            'elem_status': job_status,
            'rollup_status': summary_status,
        }

    def compare_stat_dir(self) -> dict[str, Any]:
        def attr_kind(attrname: str):
            if attrname.startswith('instrs_'):
                return 'instrs'
            if attrname.startswith('inlist_'):
                return 'list'
            if attrname.startswith('outlist_'):
                return 'list'
            if attrname.startswith('attrs_'):
                return 'attrs'
            return 'fixed'
        os.makedirs(self.output_path / self.study / 'csv', exist_ok=True)
        os.makedirs(self.output_path / self.study / 'json', exist_ok=True)
        keys_classified = self.keys_classified
        keys_common = keys_classified['keys_common']
        keys_only_in_1 = keys_classified['keys_only_in_1']
        keys_only_in_2 = keys_classified['keys_only_in_2']
        keys_all = keys_classified['keys_all']
        job_status: dict[str, Any] = {jobkey_2_str(k): {} for k in keys_all}
        result = {
            'elem_status': job_status,
            'rollup_status': ComparisonStatus.Mismatch,
        }
        job_status.update({jobkey_2_str(k): {'rollup_status': ComparisonStatus.Only_in_1} for k in keys_only_in_1})
        job_status.update({jobkey_2_str(k): {'rollup_status': ComparisonStatus.Only_in_2} for k in keys_only_in_2})
        for k in sorted(keys_common):
            kstr: str = jobkey_2_str(k)
            logging.info('Comparing %s i.e. %s', k, kstr)
            csv_table: list[dict[str, Any]] = []
            statdict1, statdict2 = self.run1.load_stat(k), self.run2.load_stat(k)
            jstat = compare_operator_stats(statdict1['operatorstats'], statdict2['operatorstats'], self.epsilon)
            output_filename = self.output_path / self.study / 'json' / os.path.basename(self.run1.statfilename(k))
            job_status[kstr] = {
                'filename': output_filename.as_posix(),
                'rollup_status': jstat['rollup_status'],
            }
            fixed_keys: list[str] = []
            variable_keys: dict[str, list[str]] = defaultdict(list)
            for ndx, row in jstat['elem_status'].items():
                csv_row = status_2_csvrow(row['elem_status'])
                csv_table.append(csv_row)
                if fixed_keys == []:
                    fixed_keys = [x for x in csv_row if attr_kind(x) == 'fixed']
                missing_variable_keys: dict[str, set[str]] = defaultdict(set)
                for x in csv_row:
                    attrk = attr_kind(x)
                    if attrk != 'non-instrs' and x not in variable_keys[attrk]:
                        missing_variable_keys[attrk].add(x)
                for attrk in missing_variable_keys:
                    variable_keys[attrk].extend(missing_variable_keys[attrk])

            all_variable_keys = []
            for attrk in sorted(variable_keys):
                all_variable_keys.extend(sorted(variable_keys[attrk]))
            print_csv(sorted(fixed_keys) + all_variable_keys, csv_table, self.output_path / self.study / 'csv' / f'{kstr}.csv')
            with open(output_filename, 'w') as fout:
                json.dump(jstat, fout, indent=4)
        result['rollup_status'] = rollup_status([job_status[k]['rollup_status'] for k in job_status])
        rollup_csv_table: list[dict[str, Any]] = []
        for jobname, jobentry in sorted(job_status.items()):
            rollup_csv_table.append({
                'Job': jobname,
                'Status': jobentry['rollup_status'].value,
                'Filename': jobentry['filename'],
            })
        print_csv(['Job', 'Status', 'Filename'], rollup_csv_table,
                  filename=self.output_path / self.study / 'stat-comparison.csv')

        return result

    def compare_study(self) -> ComparisonStatus:
        study_path1 = self.run1.studypath
        study_path2 = self.run2.studypath
        output_path_for_study = self.output_path / self.study
        if not study_path1.is_dir() or not study_path2.is_dir():
            raise RuntimeError(f"Study Directory {self.study} not found in both project run directories")
        logging.info("Comparing %s <-> %s", study_path1, study_path2)

        config_result = self.compare_config_dir()

        job_result = self.compare_summary_dir()

        stat_result = self.compare_stat_dir()

        logging.info('status=%s', stat_result['rollup_status'])
        with open(output_path_for_study / 'config-comparison.json', 'w') as fout:
            json.dump(config_result, fout, indent=4)
        with open(output_path_for_study / 'stat-comparison.json', 'w') as fout:
            json.dump(stat_result, fout, indent=4)
        with open(output_path_for_study / 'summary-comparison.json', 'w') as fout:
            json.dump(job_result, fout, indent=4)
        with open(output_path_for_study / 'config-comparison.json', 'w') as fout:
            json.dump(config_result, fout, indent=4)
        return stat_result['rollup_status']


class ProjComparison:
    def __init__(self, path1: Path, path2: Path, output_path: Path, epsilon: float) -> None:
        self.path1 = path1
        self.path2 = path2
        self.epsilon = epsilon
        self.output_path = output_path
        self.__setup__()
        os.makedirs(self.output_path, exist_ok=True)

    def __setup__(self) -> None:
        path1_contents = [f1.stem for f1 in self.path1.iterdir()]
        path2_contents = [f1.stem for f1 in self.path1.iterdir()]
        common_contents = set(path1_contents) & set(path2_contents)
        if 'inputs' not in common_contents:
            raise RuntimeError('inputs directory not found in the project run directories')
        self.studies = common_contents - {'inputs'}

    def compare_studies(self) -> ComparisonStatus:
        results: list[ComparisonStatus] = []
        for study in self.studies:
            study_comp = StudyComparison(self.path1, self.path2, self.output_path, self.epsilon, study)
            results.append(study_comp.compare_study())
        return rollup_status(results)



def main() -> int:
    parser = argparse.ArgumentParser(description="Comparing Performance Reports")
    parser.add_argument('--output',  '-o',  required=True, metavar='<outfile>',       help="Output Directory")
    parser.add_argument('--dir1',    '-d1', required=True, metavar='<perf-report-1>', help="Archbench Report-1 (root directory of projection run)")
    parser.add_argument('--dir2',    '-d2', required=True, metavar='<perf-report-2>', help="Archbench Report-2 (root directory of projection run)")
    parser.add_argument('--epsilon', '-e',  required=False, default=0.05, metavar='<error-bar>',     help="Error Bar(float), default=0.05, i.e. 5%%")

    if len(sys.argv) <= 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s')

    path1 = Path(args.dir1)
    path2 = Path(args.dir2)
    output_path = Path(args.output)
    cmp = ProjComparison(path1, path2, output_path, args.epsilon)
    result = cmp.compare_studies()
    if result == ComparisonStatus.Match:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
