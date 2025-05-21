#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import argparse
import copy
import datetime
import json
import logging
import shutil
import subprocess
from typing import Any, Tuple, Union

import jinja2
import yaml
from git import GitCommandError, Repo
from jinja2 import Template

import ttsim.config.runcfgmodel as runcfgmodel

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
            args: argparse.Namespace, passdown_args: list[str], diagnostic: bool = True) -> int:
    command_words = ['python', 'polaris.py',
                     '--odir', jtemplate.expand(runcfg.odir, jdict),
                     '--study', jtemplate.expand(runcfg.study, jdict),
                     ]
    if runcfg.filterrun == 'inference':
        rt_inference, rt_training = (True, False)
    else:
        rt_inference, rt_training = (False, True)
    command_words.extend(['--inference', str(rt_inference).lower(), '--training', str(rt_training).lower()])
    if args.dryrun:
        command_words.extend(['--dryrun'])
    if runcfg.dump_ttsim_onnx:
        command_words.extend(['--dump_ttsim_onnx'])
    if runcfg.instr_profile:
        command_words.extend(['--instr_profile'])

    if runcfg.filterapi is not None:
        command_words.extend(['--filterwlg', runcfg.filterapi])
    if runcfg.filterarch is not None:
        command_words.extend(['--filterarch', runcfg.filterarch])
    if runcfg.filterwl is not None:
        command_words.extend(['--filterwl', runcfg.filterwl])
    if runcfg.filterwli is not None:
        command_words.extend(['--filterwli', runcfg.filterwli])

    command_words.extend(['--log_level', str(runcfg.log_level)])

    command_words.extend(['--wlspec', runcfg.wlspec])
    command_words.extend(['--archspec', runcfg.archspec])
    command_words.extend(['--wlmapspec', runcfg.wlmapspec])
    if runcfg.frequency is not None:
        command_words.extend(['--frequency'] + [str(_tmp) for _tmp in runcfg.frequency])
    if runcfg.batchsize is not None:
        command_words.extend(['--batchsize'] + [str(_tmp) for _tmp in runcfg.batchsize])
    if runcfg.knobs:
        command_words.extend(runcfg.knobs)
    if runcfg.enable_memalloc:
        command_words.extend(['--enable_memalloc'])
    if runcfg.enable_cprofile:
        command_words.extend(['--enable_cprofile'])
    command_words.extend(['--outputformat', runcfg.outputformat])
    if runcfg.dumpstatscsv:
        command_words.extend(['--dumpstatscsv'])
    command_words.extend(passdown_args)

    infopath = os.path.join(runcfg.odir, 'inputs', 'runinfo.json')
    os.makedirs(os.path.dirname(infopath), exist_ok=True)
    with open(infopath, 'w') as fout:
        json.dump(jdict, fout, indent=4)

    logging.info('executing %s', command_words)
    logging.info('executing %s', ' '.join(command_words))
    ret = subprocess.run(command_words)
    if ret.returncode != 0 and diagnostic:
        logging.error('command "%s" failed with exit code %d' % (command_words, ret.returncode))
    return ret.returncode


def jinja_variables(rootrepo: Repo) -> dict[str, Any]:
    head = rootrepo.head.commit
    print(f'{head=}')
    commit_time = head.committed_datetime
    author_time = head.authored_datetime
    is_detached = False
    try:
        rootrepo.active_branch
    except TypeError:
        is_detached = True
    return {
        'giturl': rootrepo.remote('origin').url,
        'githash': head.hexsha,
        'gitdirty': rootrepo.is_dirty(),
        'gituntracked': rootrepo.is_dirty(untracked_files=True),
        'gitroot': rootrepo.working_tree_dir if not is_detached else '',
        'gitbranch': rootrepo.active_branch.name if not is_detached else '',
        'author': head.author.email,
        'author_tstamp': timestamp(author_time),
        'committer': head.committer.email,
        'commit_tstamp': timestamp(commit_time),
        'tstamp': timestamp(datetime.datetime.now()),
        'commit_year': commit_time.year,
        'commit_month': f'{commit_time.month:02}',
        'commit_message': head.message,
    }


def determine_commit(rootrepo: Repo, githash: runcfgmodel.TYPE_GITHASH) -> Union[list[str], None]:
    head = rootrepo.head.commit
    if githash is None:
        return None
    if head.hexsha == githash:  # Current head is the same as the required githash, so do nothing
        return None
    try:
        commits = list(rootrepo.iter_commits(githash, max_count=1))
    except GitCommandError as _e:
        logging.error('invalid githash "%s"', githash)
        raise
    return commits[0]


def get_args() -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser('polproj.py')
    parser.add_argument('--config', '-c', required=True, help='Run configuration yaml file')
    parser.add_argument('--no-dirty', dest='no_dirty', action='store_true',
                        help='fail if repository is dirty (changes and/or untracked files)')
    parser.add_argument('--dryrun', '-n', action='store_true', help='Show the commands, but do not execute')

    if not sys.argv[1:]:
        parser.print_help()
        exit(0)
    args, passdown_args = parser.parse_known_args()
    return args, passdown_args


def prepare_for_run(runconfig: RUNCFG_TYPE, jtemplate: JTemplate, jdict: dict[str, Any], 
                    args: argparse.Namespace, passdown_args: list[str]) -> RUNCFG_TYPE:
    outdir = jtemplate.expand(runconfig.odir, jdict)
    study = jtemplate.expand(runconfig.study, jdict)
    runconfig.odir = outdir
    runconfig.study = study
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    studydir = os.path.join(outdir, study)
    if not os.path.isdir(studydir):
        os.makedirs(studydir)
    inputsdir = os.path.join(outdir, 'inputs')
    if not os.path.isdir(inputsdir):
        os.makedirs(inputsdir)
    shutil.copyfile(runconfig.wlspec, os.path.join(inputsdir, os.path.basename(runconfig.wlspec)))
    shutil.copyfile(runconfig.archspec, os.path.join(inputsdir, os.path.basename(runconfig.archspec)))
    shutil.copyfile(runconfig.wlmapspec, os.path.join(inputsdir, os.path.basename(runconfig.wlmapspec)))
    runconfig.saved_copy = True
    runconfig.githash = jdict['githash']
    with open(os.path.join(inputsdir, os.path.basename(args.config)), 'w') as fout:
        yaml.dump(runconfig.model_dump(), fout, indent=4, Dumper=yaml.CDumper)
    with open(os.path.join(inputsdir, 'rerun.sh'), 'w') as fout:
        fout.write('#!/bin/bash\n')
        fout.write('set -e\n')
        fout.write('set -x\n')
        fout.write('python polproj.py ' + 
                   f'--config {inputsdir}/{os.path.basename(args.config)}' +
                   ' '.join(passdown_args) + 
                   '\n')
    return runconfig


def override_runconfig(runconfig: RUNCFG_TYPE, passdown_args: list[str]) -> None:
    for ndx, arg in enumerate(passdown_args):
        if arg == '--odir':
            if ndx + 1 >= len(passdown_args):
                raise ValueError('missing argument for --odir')
            runconfig.odir = passdown_args[ndx + 1]
            logging.warning('overriding output directory with %s', runconfig.odir)
        elif arg == '--study':
            if ndx + 1 >= len(passdown_args):
                raise ValueError('missing argument for --study')
            runconfig.study = passdown_args[ndx + 1]
            logging.warning('overriding study name with %s', runconfig.study)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s')
    args: argparse.Namespace
    passdown_args: list[str]
    args, passdown_args = get_args()
    runconfig: RUNCFG_TYPE = load_runconfig(args.config)

    repodir: str = os.getcwd()
    rootrepo: Repo = Repo(repodir)

    if args.no_dirty:
        if rootrepo.is_dirty():
            logging.error('Repository is dirty (changes and/or untracked files), please commit or stash changes before running')
            return 1
        logging.info('Repository is clean')

    commit = determine_commit(rootrepo, runconfig.githash)

    override_runconfig(runconfig, passdown_args)

    if commit is not None:
        if rootrepo.is_dirty():
            logging.error('Repository is dirty (trying to checkout %s), please commit or stash changes before running', 
                        commit)
            return 1
        rootrepo.git.checkout(commit)
        logging.warning('checked out requested git hash %s', commit)

    jinjadict = jinja_variables(rootrepo)
    mytemplate = JTemplate()
    prepared_config: RUNCFG_TYPE
    if runconfig.saved_copy:
        prepared_config = runconfig
    else:
        prepared_config = copy.copy(runconfig)
        prepare_for_run(prepared_config, mytemplate, jinjadict, args, passdown_args)
    ret = execute(prepared_config, mytemplate, jinjadict, args, passdown_args)
    return ret


if __name__ == '__main__':
    exit(main())
