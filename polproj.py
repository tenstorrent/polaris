#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import argparse
import logging
import yaml
import ttsim.config.runcfgmodel as runcfgmodel
from git import Repo
import datetime
import jinja2
from jinja2 import Template
import subprocess
from typing import Any, Union



SCRIPT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Type Annotations
type JDICT_TYPE = dict[str, Template]
type RUNCFG_TYPE = runcfgmodel.PolarisRunConfig


def choose_file(fname: str) -> str:
    if os.path.isabs(fname):
        return fname
    for tmp in ['.', SCRIPT_ROOT_DIR]:
        fpath: str = os.path.join(tmp, fname)
        if os.path.isfile(fpath):
            return fpath
    raise FileNotFoundError(fname)


def load_runconfig(cfgfile: str) -> RUNCFG_TYPE:
    rundict = yaml.load(open(choose_file(cfgfile)), Loader=yaml.CLoader)
    logging.info('validating %s', cfgfile)
    return runcfgmodel.PolarisRunConfig(**rundict)


class JTemplate:
    def __init__(self) -> None:
        self.jlogger = jinja2.make_logging_undefined(logger=logging.getLogger(), base=jinja2.Undefined)

        self.templates: JDICT_TYPE = dict()

    def expand(self, s: str, obj: JDICT_TYPE) -> str:
        try:
            template = self.templates[s]
        except KeyError:
            self.templates[s] = Template(s, undefined=self.jlogger)
            template = self.templates[s]
        return template.render(obj)


def timestamp(tm: datetime.datetime) -> str:
    return tm.strftime('%y%m%d-%H%M%S')


# TODO: P0 Use polaris directly, instead of through separate process
def execute(runcfg: RUNCFG_TYPE, jtemplate: JTemplate, jdict: JDICT_TYPE,
            args: argparse.Namespace, diagnostic: bool = True) -> int:
    outputdir = jtemplate.expand(runcfg.output, jdict)
    command_words = ['python', 'polaris.py',
                     '--odir', jtemplate.expand(runcfg.output, jdict),
                     '--study', jtemplate.expand(runcfg.study, jdict),
                     ]
    if runcfg.runtype == 'inference':
        rt_inference, rt_training = (True, False)
    else:
        rt_inference, rt_training = (False, True)
    command_words.extend(['--inference', str(rt_inference).lower(), '--training', str(rt_training).lower()])
    command_words.extend(['--archspec', runcfg.archspec])
    if args.dryrun:
        command_words.extend(['--dryrun'])
    if runcfg.dump_ttsim_onnx:
        command_words.extend(['--dump_ttsim_onnx'])
    if runcfg.instr_profile:
        command_words.extend(['--instr_profile'])

    if runcfg.filter is not None:
        command_words.extend(['--filter', runcfg.filter])
    if runcfg.filterapi is not None:
        command_words.extend(['--filterwlg', runcfg.filterapi])
    if runcfg.filterarch is not None:
        command_words.extend(['--filterarch', runcfg.filterarch])
    if runcfg.filterwl is not None:
        command_words.extend(['--filterwl', runcfg.filterwl])
    if runcfg.filterwli is not None:
        command_words.extend(['--filterwli', runcfg.filterwli])

    command_words.extend(['--log_level', str(runcfg.loglevel)])

    command_words.extend(['--wlspec', runcfg.wlspec])
    command_words.extend(['--archspec', runcfg.archspec])
    command_words.extend(['--wlmapspec', runcfg.wlmapspec])
    if runcfg.frequency is not None:
        command_words.extend(['--frequency'] + [str(_tmp) for _tmp in runcfg.frequency])
    if runcfg.batchsize is not None:
        command_words.extend(['--batchsize'] + [str(_tmp) for _tmp in runcfg.batchsize])

    logging.info('executing %s', command_words)
    logging.info('executing %s', " ".join(command_words))
    ret = subprocess.run(command_words)
    if ret.returncode != 0 and diagnostic:
        logging.error('command "%s" failed with exit code %d' % (command_words, ret.returncode))
    return ret.returncode


def jinja_variables(rootrepo: Repo) -> dict[str, Any]:
    head = rootrepo.head.commit
    commit_time = head.committed_datetime
    author_time = head.authored_datetime
    tag = rootrepo.tags[0]
    return {
        'gitroot': rootrepo.working_tree_dir,
        'githash': head.hexsha,
        'gitbranch': rootrepo.active_branch.name,
        'tags': [tag.name for tag in rootrepo.tags if tag.commit == head],
        'commit_tstamp': timestamp(commit_time),
        'author_tstamp': timestamp(author_time),
        'tstamp': timestamp(datetime.datetime.now()),
        'commit_year': commit_time.year,
        'commit_month': commit_time.month,

    }

def determine_commits(ab_repo: Repo, githash: runcfgmodel.TYPE_GITHASH) -> Union[list[str], None]:
    if githash is None:
        return None
    raise NotImplementedError('githash support')

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('polproj.py')
    parser.add_argument('--config', '-c', required=True, help='Run configuration yaml file')
    parser.add_argument('--dryrun', '-n', action='store_true', help='Show the commands, but do not execute')

    if not sys.argv[1:]:
        parser.print_help()
        exit(0)
    args = parser.parse_args()
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s')
    args: argparse.Namespace = get_args()
    runconfig : RUNCFG_TYPE = load_runconfig(args.config)

    repodir : str = os.getcwd()
    rootrepo : Repo = Repo(repodir)

    commits = determine_commits(rootrepo, runconfig.githash)

    if commits is None:
        jinjadict = jinja_variables(rootrepo)
        mytemplate = JTemplate()
        ret = execute(runconfig, mytemplate, jinjadict, args)
        return ret
    raise NotImplementedError('githash not supported yet')


if __name__ == '__main__':
    exit(main())
