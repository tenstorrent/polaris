#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import statistics

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import json
import logging
import math
from collections import Counter, defaultdict, namedtuple
from enum import Enum
from pathlib import Path
from typing import Any, Tuple

from jinja2 import Environment, PackageLoader, StrictUndefined

import tools.statattr as statattr
from ttsim.utils.common import print_csv

DEBUG = True

type IndexTuple = Tuple[int | None, int | None]

ATTRIBUTES_TO_SKIP: set[str] = {'type', 'stat_filename'}


class ComparisonStatus(str, Enum):
    Only_in_1 = 'keys_only_in_1'
    Only_in_2 = 'keys_only_in_2'
    TypeMismatch = 'type_mismatch'
    Mismatch = 'mismatch'
    ApproxMatch = 'approx_match'
    Match = 'match'


class Jinja2Environment:
    def __init__(self) -> None:
        self.env = Environment(loader=PackageLoader('tools', 'templates'), undefined=StrictUndefined)

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        return template.render(context)


def flatten_dict_as_str(d: dict[str, Any], param_sep: str = '_') -> dict[str, Any]:
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
            new_key = f'{parent_key}{param_sep}{k}' if parent_key else k
            if isinstance(v, (dict, list)):
                items.append((k, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    return _flatten(d, parent_key='', param_sep=param_sep)


JobKey = namedtuple('JobKey', ['devname', 'wlcls', 'wlname', 'wlinstance', 'bs'])


def jobkey_2_str(key: JobKey) -> str:
    return f'{key.devname}_{key.wlcls}_{key.wlname}_{key.wlinstance}_b{key.bs}'


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
        result['ratio'] = value2 / value1 if value1 != 0 else 1 if value2 == 0 else 0  # float('inf')
        result['type'] = 'numeric'
        assert 'status' in result
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
    matching_indices = (ndx for ndx in range(start_index, len(opstats)) if opstats[ndx]['optype'] == ref_op['optype'])
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
        if DEBUG:
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


def classify_keys(dict_1: dict, dict_2: dict) -> dict:
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
            raise AssertionError(f'{self.rootpath} contains unexpected subdirectories: {self.subdirs}')

        summary_file = self.studypath / 'SUMMARY' / 'study-summary.json'
        self.name = path.name
        tmp_dict = json.load(summary_file.open('r'))
        self.summary_dict = summary_index(tmp_dict['summary'])
        self.runinfo = json.load((self.rootpath / 'inputs' / 'runinfo.json').open('r'))
        self.confignames = {d.stem for d in (self.studypath / 'CONFIG').iterdir() if d.is_file()}

    def classify_keys(self, run2: 'ProjectionRun') -> dict:
        return classify_keys(self.summary_dict, run2.summary_dict)

    def statfilename(self, key: JobKey) -> str:
        return self.summary_dict[key]['stat_filename']

    def load_stat(self, key: JobKey) -> dict:
        statfname: str = self.summary_dict[key]['stat_filename']
        statpath: Path = self.rootpath / statfname
        val = json.load(statpath.open('r'))
        flat_operator_stats = []
        for opstat in val['operatorstats']:
            flat_operator_stats.append(flatten_dict_as_str(opstat))
        val['operatorstats'] = flat_operator_stats
        return val


class StudyComparison:
    def __init__(self, path1: Path, path2: Path, output_path: Path, epsilon: float, study: str,
                 jinja2env: Jinja2Environment, flag_generate_html: bool) -> None:
        self.study = study
        self.run1 = ProjectionRun(path1, self.study)
        self.run2 = ProjectionRun(path2, self.study)
        self.epsilon = epsilon
        self.output_path = output_path
        self.jinja2env = jinja2env
        self.flag_generate_html = flag_generate_html
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
            config1 = flatten_dict_as_str(json.load((self.run1.studypath / 'CONFIG' / f'{cfg}.json').open('r')))
            config2 = flatten_dict_as_str(json.load((self.run2.studypath / 'CONFIG' / f'{cfg}.json').open('r')))
            config_status[cfg] = compare_dicts(config1, config2, self.epsilon)
        result = {
            'elem_status': config_status,
            'rollup_status': rollup_status([cfgstatus['rollup_status'] for cfgstatus in config_status.values()]),
        }
        return result

    def generate_cfgsummaries(self, config_result: dict[str, Any], cfg_summary_path: Path) -> None:
        if not self.flag_generate_html:
            return
        status_frequency: Counter = Counter()
        cfgsummary_data = []
        sr = 0
        for cfg, cfgstatus in sorted(config_result['elem_status'].items()):
            sr += 1
            row = [sr, cfg, cfgstatus['rollup_status'].value, '']
            cfgsummary_data.append(row)
            status_frequency[cfgstatus['rollup_status']] += 1
        cfgsummary_data_json = json.dumps(cfgsummary_data, indent=4)

        jdict = {}
        jdict['cfg_compare_result'] = config_result['rollup_status'].value
        jdict['config_dataset'] = cfgsummary_data_json
        jdict['config_columns'] = ['Sr', 'Config', 'Status', 'Link']
        jdict['study_name'] = self.study
        jdict['run1_name'] = self.run1.rootpath.stem
        jdict['run2_name'] = self.run2.rootpath.stem
        jdict['exact_matches'] = status_frequency[ComparisonStatus.Match]
        jdict['approx_matches'] = status_frequency[ComparisonStatus.ApproxMatch]
        jdict['mismatches'] = status_frequency[ComparisonStatus.Mismatch]
        jdict['only_in_1_or_2'] = status_frequency[ComparisonStatus.Only_in_1] + status_frequency[ComparisonStatus.Only_in_2]
        jdict['total'] = len(config_result['elem_status'])
        cfgsummary_html = self.jinja2env.render('template-cfgsummary.html', jdict)
        open(self.output_path / self.study / 'cfgsummary.html', 'w').write(cfgsummary_html)


    def generate_jobsummaries_datatable(self, job_result: dict[str, Any], wl_compare_dtable: Path) -> None:
        if not self.flag_generate_html:
            return
        status_frequency: Counter = Counter()
        jobsummary_data = []
        sr = 0
        job_columns: list[str] = []
        html_table_rows = []
        html_table_header_row_1 = ['<th rowspan="2">Sr</th>', '<th rowspan="2">Job</th>', '<th rowspan="2">Status</th>']
        html_table_header_row_2 = []
        first_key = next((k for k in job_result['elem_status']))
        first_jobstatus = job_result['elem_status'][first_key]
        for col, colentry in first_jobstatus['elem_status'].items():
            # Reduce length by 1 - since we do NOT add the "type" column
            html_table_header_row_1.append(f'<th colspan="{len(first_jobstatus["elem_status"][col]) - 1}">{col}</th>')
            for col2 in colentry:
                html_table_header_row_2.append(f'<th>{col2}</th>')
        for job, jobstatus in sorted(job_result['elem_status'].items()):
            sr += 1
            row = [sr, job, jobstatus['rollup_status'].value]
            status_frequency[jobstatus['rollup_status']] += 1
            elem_status = jobstatus['elem_status']
            if job_columns == []:
                job_columns.extend(['Sr', 'Job', 'Status'])
                for col, colentry in elem_status.items():
                    if col in ATTRIBUTES_TO_SKIP:
                        continue
                    first_flag = True
                    for col2 in colentry:
                        if col2 in ATTRIBUTES_TO_SKIP:
                            continue
                        if first_flag:
                            job_columns.append(col + '&rarr;<br>' + col2)
                            first_flag = False
                        else:
                            job_columns.append(col2)
            for col, colentry in elem_status.items():
                if col in ATTRIBUTES_TO_SKIP:
                    continue
                for col2, col2value in colentry.items():
                    if col2 in ATTRIBUTES_TO_SKIP:
                        continue
                    if isinstance(col2value, ComparisonStatus):
                        row.append(col2value.value)
                    elif isinstance(col2value, float):
                        row.append(f'{col2value:.3f}')
                    else:
                        row.append(str(col2value))
                # row.extend([str(col2value) if not isinstance(col2value, ComparisonStatus) else col2value.value for col2, col2value in colentry.items()])
            jobsummary_data.append(row)
            html_table_rows.append('\n'.join(['<tr>'] + [f'<td>{x}</td>' for x in row] + ['</tr>']))
            assert len(job_columns) == len(row)
        html_header = '\n'.join(['<tr>'] + html_table_header_row_1 + ['</tr>', '<tr>'] + html_table_header_row_2 + ['</tr>'])
        jobsummary_data_json = json.dumps(jobsummary_data, indent=4)

        jdict = {}
        jdict['job_compare_result'] = job_result['rollup_status'].value
        jdict['job_dataset'] = jobsummary_data_json
        jdict['job_html_header'] = html_header
        jdict['job_html_body'] = '\n'.join(html_table_rows)
        jdict['job_columns'] = job_columns
        jdict['study_name'] = self.study
        jdict['run1_name'] = self.run1.rootpath.stem
        jdict['run2_name'] = self.run2.rootpath.stem
        jdict['exact_matches'] = status_frequency[ComparisonStatus.Match]
        jdict['approx_matches'] = status_frequency[ComparisonStatus.ApproxMatch]
        jdict['mismatches'] = status_frequency[ComparisonStatus.Mismatch]
        jdict['only_in_1_or_2'] = status_frequency[ComparisonStatus.Only_in_1] + status_frequency[ComparisonStatus.Only_in_2]
        jdict['total'] = len(job_result['elem_status'])

        jobsummary_html = self.jinja2env.render('template-jobsummary.html', jdict)
        open(wl_compare_dtable, 'w').write(jobsummary_html)

    def generate_jobsummaries_gchart(self, job_result: dict[str, Any], wl_compare_gchart_path: Path) -> None:
        if not self.flag_generate_html:
            return
        attrname_2_desc = {attrdesc.name: attrdesc for ndx, attrdesc in enumerate(sorted(statattr.StatAttributeDescriptors.job_attribute_list, key=lambda x: x.seq))}
        jdict: dict[str, Any] = {}
        jdict['report_title'] = 'Comparison of Projections - Summary'
        jdict['cwd'] = os.path.abspath(os.getcwd())
        jdict['result_final'] = job_result['rollup_status'].value
        jdict['epsilon'] = self.epsilon
        jdict['run1_name'] = self.run1.rootpath.stem
        jdict['run2_name'] = self.run2.rootpath.stem
        jdict['epsilon_str'] = f'{self.epsilon:.3e}'
        jdict['comparison_data'] = json.dumps(job_result, indent=4)
        jdict['linkdir'] = (self.output_path / self.study / 'html').relative_to(self.output_path / self.study).as_posix()
        jdict['attr_desc'] = json.dumps({attr: json.loads(model.model_dump_json()) for attr, model in attrname_2_desc.items()}, indent=4)
        results = [jobstat['rollup_status'] == ComparisonStatus.Match for jobstat in job_result['elem_status'].values()]
        jdict['num_rows'] = len(results)
        jdict['num_matches'] = results.count(True)
        jdict['num_mismatches'] = results.count(False)
        jobsummary_html = self.jinja2env.render('template-cmpperf.html', jdict)
        open(wl_compare_gchart_path, 'w').write(jobsummary_html)

    def generate_comparison_gchart(self,
                                   comparison_result: dict[str, Any],
                                   leftname: str,
                                   rightname: str,
                                   attrlist: statattr.StatAttributeDescriptorList,
                                   filepath:Path) -> None:
        if not self.flag_generate_html:
            return
        attrname_2_desc = {attrdesc.name: attrdesc for ndx, attrdesc in enumerate(sorted(attrlist, key=lambda x: x.seq))}
        jdict: dict[str, Any] = {}
        jdict['report_title'] = 'Comparison of Layerwise Performance - ' + leftname.split('/')[-1]
        jdict['cwd'] = os.path.abspath(os.getcwd())
        jdict['result_final'] = comparison_result['rollup_status'].value
        jdict['epsilon'] = self.epsilon
        jdict['run1_name'] = leftname
        jdict['run2_name'] = rightname
        jdict['epsilon_str'] = f'{self.epsilon:.3e}'
        jdict['comparison_data'] = json.dumps(comparison_result, indent=4)
        jdict['attr_desc'] = json.dumps({attr: json.loads(model.model_dump_json()) for attr, model in attrname_2_desc.items()}, indent=4)
        results = [jobstat['rollup_status'] == ComparisonStatus.Match for jobstat in comparison_result['elem_status'].values()]
        jdict['num_rows'] = len(results)
        jdict['num_matches'] = results.count(True)
        jdict['num_mismatches'] = results.count(False)

        comparison_html = self.jinja2env.render('template-cmpperf.html', jdict)
        open(filepath, 'w').write(comparison_html)

    def generate_topsummary(self, job_result: dict[str, Any],
                            cfg_compare_html: Path,
                            wl_compare_html: Path,
                            output_html_path: Path) -> None:
        if not self.flag_generate_html:
            return
        workload_result_list = [(entry['rollup_status'], entry['elem_status']['tot_cycles']['ratio']) for entry in job_result['elem_status'].values()]
        stat_all = [tuple_entry[1] for tuple_entry in workload_result_list]
        stat_mismatches = [tuple_entry[1] for tuple_entry in workload_result_list
                           if tuple_entry[0] == ComparisonStatus.Mismatch]
        all_geomean = statistics.geometric_mean(stat_all) if stat_all else None
        all_stdev = statistics.stdev(stat_all) if len(stat_all) >= 2 else None
        mismatch_geomean = statistics.geometric_mean(stat_mismatches) if stat_mismatches else None
        mismatch_stdev = statistics.stdev(stat_mismatches) if len(stat_mismatches) >= 2 else None

        final_result = job_result['rollup_status']
        jdict: dict[str, Any] = {}
        jdict['result_final'] = final_result.value
        jdict['run1_name'] = self.run1.rootpath.stem
        jdict['run2_name'] = self.run2.rootpath.stem
        jdict['stats'] = json.dumps({
            'all_geomean': all_geomean,
            'all_stdev': all_stdev,
            'mismatch_geomean': mismatch_geomean,
            'mismatch_stdev': mismatch_stdev,
        }, indent=4)
        jdict['config_comparison_html'] = cfg_compare_html.relative_to(output_html_path.parent, walk_up=True).as_posix()
        jdict['workload_comparison_html'] = wl_compare_html.relative_to(output_html_path.parent, walk_up=True).as_posix()
        topsummary_html = self.jinja2env.render('template-projrun-summary.html', jdict)
        open(output_html_path, 'w').write(topsummary_html)

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
        for ext in ['csv', 'json'] + (['html'] if self.flag_generate_html else []):
            os.makedirs(self.output_path / self.study / ext, exist_ok=True)
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
            logging.info('Comparing %s', kstr)
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
            self.generate_comparison_gchart(jstat,
                                            self.run1.rootpath.stem + '/' + self.study + '/' + kstr,
                                            self.run2.rootpath.stem + '/' + self.study + '/' + kstr,
                                            statattr.StatAttributeDescriptors.op_attribute_list,
                                            self.output_path / self.study / 'html' / f'{kstr}.html')
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
        os.makedirs(output_path_for_study, exist_ok=True)
        if not study_path1.is_dir() or not study_path2.is_dir():
            raise AssertionError(f'Study Directory {self.study} not found in both project run directories')
        logging.info('Comparing %s <-> %s', study_path1, study_path2)

        config_result = self.compare_config_dir()
        config_compare_html = self.output_path / self.study / 'cfgsummary.html'
        self.generate_cfgsummaries(config_result, config_compare_html)

        job_result = self.compare_summary_dir()
        workload_compare_html_gchart = self.output_path / self.study / 'jobsummary.html'
        workload_compare_html_dtable = self.output_path / self.study / 'jobsummary-dt.html'
        self.generate_jobsummaries_gchart(job_result, workload_compare_html_gchart)
        self.generate_jobsummaries_datatable(job_result, workload_compare_html_dtable)

        stat_result = self.compare_stat_dir()
        topsummary_html = self.output_path / self.study / 'summary.html'
        self.generate_topsummary(job_result, config_compare_html, workload_compare_html_gchart, topsummary_html)

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
    def __init__(self, path1: Path, path2: Path,
                 output_path: Path, epsilon: float, jinja2env: Jinja2Environment,
                 flag_generate_html: bool) -> None:
        self.path1 = path1
        self.path2 = path2
        self.epsilon = epsilon
        self.output_path = output_path
        self.jinja2env = jinja2env
        self.flag_generate_html = flag_generate_html
        self.__setup__()
        os.makedirs(self.output_path, exist_ok=True)

    def __setup__(self) -> None:
        path1_contents = [f1.stem for f1 in self.path1.iterdir()]
        path2_contents = [f1.stem for f1 in self.path1.iterdir()]
        common_contents = set(path1_contents) & set(path2_contents)
        if 'inputs' not in common_contents:
            raise AssertionError('inputs directory not found in the project run directories')
        if '.DS_Store' in common_contents:  # pragma: no cover
            common_contents = common_contents - {'.DS_Store'}
        self.studies = common_contents - {'inputs'}

    def compare_studies(self) -> ComparisonStatus:
        results: list[ComparisonStatus] = []
        for study in self.studies:
            study_comp = StudyComparison(self.path1, self.path2, self.output_path, self.epsilon,
                                         study, self.jinja2env, self.flag_generate_html)
            results.append(study_comp.compare_study())
        return rollup_status(results)


def main() -> int:
    parser = argparse.ArgumentParser(description='Comparing Performance Reports')
    parser.add_argument('--output',  '-o',  required=True, metavar='<outfile>', help='Output Directory')
    parser.add_argument('--dir1',    '-d1', required=True, metavar='<perf-report-1>',
                        help='Projection Run-1 (root directory of projection run)')
    parser.add_argument('--dir2',    '-d2', required=True, metavar='<perf-report-2>',
                        help='Projection Run-2 (root directory of projection run)')
    parser.add_argument('--epsilon', '-e',  required=False, default=0.05, metavar='<error-bar>',
                        help='Error Bar(float), default=0.05, i.e. 5%%')
    parser.add_argument('--generate-html', '--html', dest='generate_html', action=argparse.BooleanOptionalAction,
                        default=True, help='Generate HTML output')

    if len(sys.argv) <= 1:  # pragma: no cover
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s')

    path1 = Path(args.dir1)
    path2 = Path(args.dir2)
    output_path = Path(args.output)
    jinja2_env = Jinja2Environment()
    statattr.StatAttributeDescriptors.setup_attribute_descriptors()
    cmp = ProjComparison(path1, path2, output_path, args.epsilon, jinja2_env, args.generate_html)
    result = cmp.compare_studies()
    if result == ComparisonStatus.Match:
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
