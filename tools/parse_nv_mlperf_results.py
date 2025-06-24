#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import re
import json
import csv
import os
import logging

from pathlib import Path
from typing import Dict, List, Any, Union

LOG   = logging.getLogger(__name__)
INFO  = LOG.info
WARN  = LOG.warning
DEBUG = LOG.debug

class FrameStack:
    def __init__(self, strict_lifo: bool = False):
        self._stack      : List = []
        self.strict_lifo : bool = strict_lifo

    def push(self, timestamp: int, interval_name: str, metadata: Dict):
        #if self._stack and interval_name == 'init_start' and interval_name in self._stack[-1]['intervals']:
        #    #fold init_start+ into a single stack frame
        #    self._stack[-1]['intervals'][interval_name]['count'] += 1
        #    pass
        if self._stack and self._stack[-1]['timestamp'] == timestamp:
            frame = self._stack[-1]
            assert interval_name not in frame['intervals'], f"Mulitple {interval_name} INTERVALS in same timestamp"
            frame['intervals'][interval_name] = {'start_time': timestamp, 'metadata': metadata}
            frame['pending_ends'].add(interval_name)
        else:
            frame = {
                'timestamp': timestamp,
                'intervals': {interval_name: {'start_time': timestamp, 'metadata': metadata, 'count': 1}},
                'pending_ends': {interval_name}
            }
            self._stack.append(frame)

    def resolve_end(self, timestamp: int, interval_name: str, metadata: Dict) -> Union[Dict, None]:
        if self.empty():
            raise ValueError(f"INTERVAL_END {interval_name} at {timestamp} with empty stack")

        interval_start_name = interval_name.replace('stop', 'start')
        if self.strict_lifo:
            # Only check top of stack
            frame = self._stack[-1]
            if interval_start_name in frame['intervals']:
                interval = frame['intervals'][interval_start_name]
                interval['end_time'] = timestamp
                interval['end_metadata'] = metadata
                frame['pending_ends'].discard(interval_start_name)
                if not frame['pending_ends']:
                    return self._stack.pop()  # Pop last element
                return None
            raise ValueError(f"No matching INTERVAL_START for {interval_name} at {timestamp}")
        else:
            # Search entire stack
            for i in range(len(self._stack) - 1, -1, -1):
                frame = self._stack[i]
                if interval_start_name in frame['intervals']:
                    interval = frame['intervals'][interval_start_name]
                    interval['end_time'] = timestamp
                    interval['end_metadata'] = metadata
                    frame['pending_ends'].discard(interval_start_name)
                    if not frame['pending_ends']:
                        return self._stack.pop(i)
                    return None
            raise ValueError(f"No matching INTERVAL_START for {interval_name} at {timestamp}")

    def top(self) -> Union[Dict|None]:
        if self.empty():
            return None
        return self._stack[-1]

    def empty(self) -> bool:
        return len(self._stack) == 0

    def size(self) -> int:
        return len(self._stack)

    def __str__(self) -> str:
        return "\n".join([f"Frame(ts={frame['timestamp']}, intervals={frame['intervals']}, pending={frame['pending_ends']})"
                         for frame in self._stack])

class MLPerfTrainingLogParser:
    def __init__(self, strict_lifo: bool = False):
        self.stack               = FrameStack(strict_lifo=strict_lifo)
        self.tree         : Dict = {'intervals': [], 'points': [], 'metadata': {}}
        self.current_node        = self.tree

    def parse_line(self, line: str):
        pattern = r':::MLLOG\s+(\{.*\})'
        match = re.match(pattern, line.strip())
        if not match:
            return

        json_str = match.group(1)
        try:
            log_entry = json.loads(json_str)
        except json.JSONDecodeError:
            WARN(f"Warning: Failed to parse JSON: {json_str}")
            return

        timestamp = log_entry['time_ms'] #/ 1000.0
        event_type = log_entry['event_type']
        key = log_entry['key']
        value = log_entry['value']
        metadata = log_entry.get('metadata', {})

        if event_type == "POINT_IN_TIME":
            if key not in ['cache_clear', 'weights_initialization']:
                point = {'timestamp': timestamp, 'key': key, 'value': value, 'metadata': metadata}
                self.current_node['points'].append(point)
                DEBUG(f"POINT_IN_TIME: {key} = {value} at {timestamp}")

        elif event_type == "INTERVAL_START":
            if key not in ['init_start']:
                self.stack.push(timestamp, key, metadata)
                new_node = {
                        'name': key,
                        'start_time': timestamp,
                        'metadata': metadata,
                        'intervals': [],
                        'points': [],
                        }
                self.current_node['intervals'].append(new_node)
                self.current_node = new_node
                DEBUG(f"INTERVAL_START: {key} at {timestamp}, Stack size: {self.stack.size()}")

        elif event_type == "INTERVAL_END":
            if key not in ['init_stop']:
                resolved_frame = self.stack.resolve_end(timestamp, key, metadata)
                if resolved_frame:
                    for interval_name, interval_data in resolved_frame['intervals'].items():
                        for node in self.current_node['intervals']:
                            if node['name'] == interval_name:
                                node['end_time'] = interval_data.get('end_time')
                                node['end_metadata'] = interval_data.get('end_metadata')
                                break
                    DEBUG(f"INTERVAL_END: {key} at {timestamp}, Popped frame at {resolved_frame['timestamp']}")
                else:
                    self.current_node['end_time'] = timestamp
                    self.current_node['end_metadata'] = metadata
                    DEBUG(f"INTERVAL_END: {key} at {timestamp}, Stack size: {self.stack.size()}")

            if self.stack.size() > 0:
                __tmp_stack_top = self.stack.top() #to help mypy type checking
                assert isinstance(__tmp_stack_top, dict), "self.stack.top needs returned None!!"
                parent_timestamp = __tmp_stack_top['timestamp']
                parent_node = self.tree
                for node in parent_node['intervals']:
                    if node['start_time'] == parent_timestamp:
                        parent_node = node
                        break
                self.current_node = parent_node
            else:
                self.current_node = self.tree

    def process_log(self, log_file: str):
        with open(log_file, 'r') as f:
            for line in f:
                self.parse_line(line)
        return self.get_tree()

    def get_tree(self) -> Dict:
        if self.stack.size() > 0:
            WARN(f"Warning: {self.stack.size()} unresolved intervals remain")
        return self.tree

class MLPerfInferenceLogParser:
    def __init__(self):
        # Simplified tree for POINT_IN_TIME-only logs
        self.tree : Dict = {'points': [], 'metadata': {}}

    def parse_line(self, line: str):
        pattern = r':::MLLOG\s+(\{.*\})'
        match = re.match(pattern, line.strip())
        if not match:
            return

        json_str = match.group(1)
        try:
            log_entry = json.loads(json_str)
        except json.JSONDecodeError:
            WARN(f"Warning: Failed to parse JSON: {json_str}")
            return

        key = log_entry['key']
        if 'loadgen' in key or key == 'loaded_qsl_set': # == 'loadgen_file_sha1': #don't add sha because the value fields are very big
            return

        timestamp = log_entry['time_ms']
        event_type = log_entry['event_type']
        value = log_entry['value']
        metadata = log_entry.get('metadata', {})
        if isinstance(value, str) and '\r' in value:
            value = value.strip('\r')

        # Assert that event_type is always POINT_IN_TIME
        assert event_type == "POINT_IN_TIME", f"Unexpected event_type '{event_type}' in inference log at {timestamp} ms; expected 'POINT_IN_TIME'"

        # Add POINT_IN_TIME event to the tree
        point = {'timestamp': timestamp, 'key': key, 'value': value, 'metadata': metadata}
        self.tree['points'].append(point)
        DEBUG(f"POINT_IN_TIME: {key} = {value} at {timestamp}")

    def process_log(self, log_file: str):
        with open(log_file, 'r') as f:
            for line in f:
                self.parse_line(line)
        return self.get_tree()

    def get_tree(self) -> Dict:
        return self.tree

def flatten_tree_to_csv(tree: Dict, outfilename: str):
    """Flatten the tree and export POINT_IN_TIME stats to a CSV file, expanding dict values."""
    # List to store flattened stats
    flattened_stats = []

    def flatten_value(base_path: str, value: Any, timestamp: float, metadata: Dict):
        """Recursively flatten a value (scalar or dict) into stats."""
        if isinstance(value, dict):
            # If value is a dict, expand each key
            for k, v in value.items():
                new_path = f"{base_path}.{k}"
                flatten_value(new_path, v, timestamp, metadata)
        else:
            # Scalar value, add to stats
            flattened_stats.append({
                'Path': base_path,
                'Timestamp': timestamp,
                'Value': value,
                'Metadata': str(metadata)
            })

    def walk(node: Dict, path: str = ""):
        # Build the current path
        current_path = f"{path}{node.get('name', 'root')}" if path else "root"

        # Process POINT_IN_TIME events at this node
        for point in node.get('points', []):
            stat_path = f"{current_path}.{point['key']}"
            flatten_value(stat_path, point['value'], point['timestamp'], point['metadata'])

        # Recursively process nested intervals
        for sub_node in node.get('intervals', []):
            walk(sub_node, f"{current_path}.")

    # Walk the tree starting from the root
    walk(tree)

    # Write to CSV
    with open(outfilename, 'w') as csvfile:
        fieldnames = ['Path', 'Timestamp', 'Value', 'Metadata']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in flattened_stats:
            writer.writerow(stat)

    INFO(f"Flattened stats written to {outfilename}")

def walk_resultsdir(rdir, is_training):
    rpath = Path(rdir)
    wlTbl = {}
    if is_training:
        for log_file in rpath.rglob('*.txt'):
            log_file_fields = log_file.parts
            f1, f4 = log_file_fields[-1], log_file_fields[-4]
            if f4 == 'results' and f1.startswith('result_'):
                sys_cfg = log_file_fields[-3]
                wlname  = log_file_fields[-2]
                run_id  = f1.replace('.txt','')
                assert sys_cfg in systemsTbl, f"Unable to find {sys_cfg} in systemsTbl"

                wlkey = (wlname, 'n/a', run_id, sys_cfg)
                assert wlkey not in wlTbl, f"{wlkey} not unique!!"
                wlTbl[wlkey] = log_file
    else:
        for log_file in rpath.rglob('*.txt'):
            log_file_fields = log_file.parts
            f7 = log_file_fields[-7]
            f3 = log_file_fields[-3]
            f2 = log_file_fields[-2]
            f1 = log_file_fields[-1]
            if f7 == 'results' and f3 == 'performance' and f2.startswith('run_') \
               and f1 == 'mlperf_log_detail.txt':
                   run_id  = f2
                   sys_cfg = log_file_fields[-6]
                   wlname  = log_file_fields[-5]
                   source  = log_file_fields[-4]
                   assert source in ['Offline', 'Server', 'SingleStream', 'MultiStream'], f"BAD SOURCE {source}"
                   assert sys_cfg in systemsTbl, f"Unable to find {sys_cfg} in systemsTbl"
                   wlkey = (wlname, source, run_id, sys_cfg)
                   assert wlkey not in wlTbl, f"{wlkey} not unique!!"
                   wlTbl[wlkey] = log_file

    wlFields = ['WLNAME', 'SOURCE', 'RUNID', 'SYSCFG']
    wlFields.append('LOG')
    return wlFields, wlTbl

def walk_systemsdir(sdir):
    cfgTbl = {}
    for cfgF in Path(sdir).glob('*.json'):
        cfgName = os.path.basename(cfgF).replace(".json","")
        with open(cfgF, 'r') as cf:
            cfgTbl[cfgName] = json.load(cf)
            cfgTbl[cfgName]['_cfgName'] = cfgName

    fieldnames = set()
    for cfgName in cfgTbl:
        for fname in cfgTbl[cfgName]:
            fieldnames.add(fname)
    return sorted(fieldnames), cfgTbl

def dump_workloads_info(csvfname, wl_fields, wl_tbl):
    with open(csvfname, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=wl_fields)
        writer.writeheader()
        for k,v in wl_tbl.items():
            rec = {x0: x1 for x0, x1 in zip(wl_fields, k)}
            rec['LOG'] = v
            writer.writerow(rec)
    return

def dump_systems_info(csvfname, systems_fields, systems_tbl):
    with open(csvfname, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=systems_fields)
        writer.writeheader()
        for v in systems_tbl.values():
            writer.writerow(v)

def generate_big_resultfile(csv_files: Dict, filemetadata: List, output_file: str) -> None:
    """
    Combine all CSV files in input_dir into a single CSV, adding path components as columns.

    Args:
        csv_files (Dict): CSV files.
        filemetadata (List): File Meta Data
        output_file (str): Path to the output combined CSV file.
    """
    all_rows = []
    csv_fields = ['Timestamp', 'Path', 'Value', 'Metadata']
    for metafields, csv_file in csv_files.items():
        INFO(f'Processing csv... {csv_file}')
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            __tmp_fieldnames = reader.fieldnames or [] #for mypy type checking
            assert set(__tmp_fieldnames) == set(csv_fields), f"{reader.fieldnames} != {csv_fields}"
            for row_num, row in enumerate(reader):
                row.update({x: y for x,y in zip(filemetadata, metafields)})
                all_rows.append(row)

    outfieldnames = filemetadata + csv_fields

    # Step 3: Write combined CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=outfieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    return

if __name__ == '__main__':
    logging_levels = [ 'debug', 'info', 'warning', 'error', 'critical' ]
    cmdlineparser = argparse.ArgumentParser('parse_mlperf_logs')
    cmdlineparser.add_argument('--inputdir',  '-i', required=True, default=None, help='MLPerf Log Dir')
    cmdlineparser.add_argument('--outputdir', '-o', required=True, default=None, help='Output Results Dir')
    cmdlineparser.add_argument('--training',  '-t', action='store_true', default=False, help='training vs inference logs')
    cmdlineparser.add_argument('--log_level', '-l', type=str, default='warn',  help="set logging level", choices=logging_levels)
    args, passdown_args = cmdlineparser.parse_known_args()

    #set logging level...
    numeric_level = getattr(logging, args.log_level.upper(), None)
    assert isinstance(numeric_level, int), f'Invalid log level: {args.log}'
    logging_format = "%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s"
    logging.basicConfig(level=numeric_level, format=logging_format)

    systemsdir = os.path.join(args.inputdir, "systems")
    resultsdir = os.path.join(args.inputdir, "results")
    assert os.path.isdir(args.inputdir), f'Invalid Directory {args.inputdir}'
    assert os.path.isdir(systemsdir), f'Invalid MLPerf Systems Directory {systemsdir}'
    assert os.path.isdir(resultsdir), f'Invalid MLPerf Results Directory {resultsdir}'

    os.makedirs(args.outputdir, exist_ok=True)
    systemsCsvFile = os.path.join(args.outputdir, 'systems.csv')
    wlinfoCsvFile  = os.path.join(args.outputdir, 'wlinfo.csv')
    statsCsvFile   = os.path.join(args.outputdir, 'final_results.csv')
    statsDir       = os.path.join(args.outputdir, 'STAT')

    systemsFields,  systemsTbl   = walk_systemsdir(systemsdir)
    workloadFields, workloadsTbl = walk_resultsdir(resultsdir, args.training)

    dump_systems_info(systemsCsvFile, systemsFields, systemsTbl)
    dump_workloads_info(wlinfoCsvFile, workloadFields, workloadsTbl)

    outcsvfiles = {}
    for wlcount, (wlkey, wllogfile) in enumerate(workloadsTbl.items()):
        if args.training:
            wlname, _, run_id, sys_cfg = wlkey
            strict_lifo_match = False if wlname in ['ssd', 'gpt3', 'dlrm_dcnv2', 'stable_diffusion'] else True
            logParser1 = MLPerfTrainingLogParser(strict_lifo=strict_lifo_match) #logParser1 for mypy typing checks
            log_tree  = logParser1.process_log(wllogfile)
            outdir = "/".join((statsDir, wlname, sys_cfg))
            outfilecsv = "/".join((outdir, run_id + '.csv'))
        else:
            wlname, source, run_id, sys_cfg = wlkey
            logParser2 = MLPerfInferenceLogParser() #logParser2 for mypy typing checks
            log_tree  = logParser2.process_log(wllogfile)
            outdir = "/".join((statsDir, wlname, source, sys_cfg))
            outfilecsv = "/".join((outdir, run_id + '.csv'))

        INFO(f'..Processing log for {wlkey}')
        os.makedirs(outdir, exist_ok=True)
        systemCfg = systemsTbl[sys_cfg]
        flatten_tree_to_csv(log_tree, outfilecsv)
        outcsvfiles[wlkey] = outfilecsv

    generate_big_resultfile(outcsvfiles, workloadFields[0:-1], statsCsvFile)
