#!/usr/bin/env python
"""
The provided code is a Python script designed to check and validate SPDX
(Software Package Data Exchange) headers in source code files. These headers
typically include license and copyright information, ensuring compliance with
licensing requirements. The script supports multiple programming languages,
such as Python, JavaScript, HTML, CSS, and YAML, and provides functionality
to classify, parse, and validate files based on their extensions and content.

It also integrates with Git to exclude ignored files.
"""
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse
import fnmatch
import re
from enum import Enum
from typing import Union

import yaml
from loguru import logger
from pydantic import BaseModel


class SPDXHeaderStatus(Enum):
    """
    Enum to represent the status of a file.
    """

    ST_OK = 'ok'
    ST_MISSING = 'missing'
    ST_INCORRECT = 'incorrect'
    ST_ILLFORMED = 'illformed'


# The script uses the LANGUAGES dictionary to map file extensions to programming
# languages and the LANG_2_SYNTAX dictionary to define the comment syntax for
# each language. The ext_2_lang function determines the language of a
# file based on its extension
LANGUAGES = {
    'python': ['.py'],
    'javascript': ['.js', '.mjs', '.cjs'],
    'html': ['.html', '.htm'],
    'css': ['.css'],
    'yaml': ['.yaml', '.yml'],
}

LANG_2_SYNTAX = {
    'python': {'comment': '#'},
    'javascript': {'comment': '//'},
    'html': {'comment': '<!--', 'end_comment': '-->'},
    'css': {'comment': '/*', 'end_comment': '*/'},
    'yaml': {'comment': '#'},
}

type IgnorePattern = Union[None, re.Pattern[str]]


SPDX_LICENSE = re.compile('SPDX-License-Identifier:\\s+(?P<license_text>.*)')
SPDX_COPYRIGHT = re.compile('SPDX-FileCopyrightText:\\s+(?P<copyright_text>.*)')
COPYRIGHT_REGEX = re.compile('(?P<cprt_string>©|[(][cC][)])\\s+(?P<cprt_years>\\d{4}(-\\d{4})?)\\s+(?P<cprt_holder>.*)')
TEXT_COPYRIGHT_REGEX = re.compile('Copyright ' + COPYRIGHT_REGEX.pattern)


def ext_2_lang(ext: str) -> str:
    """
    Convert file extension to language name.
    """
    ext = ext.lower()
    for lang, exts in LANGUAGES.items():
        if ext in exts:
            return lang
    return 'unknown'


def create_args() -> argparse.ArgumentParser:
    """
    Create and return an argument parser for the script.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='check spdx headers for license and copyright.')
    parser.add_argument('--gitignore', action=argparse.BooleanOptionalAction,
        default=True, help='ignore files in .gitignore')
    parser.add_argument('--ignorelist', '-i', type=str, help='file with list of files to ignore')
    parser.add_argument('--allowed-licenses', '-a', dest='allowed_licenses', type=str, nargs='*',
                        default=['Apache-2.0'],
                        help='list of allowed licenses')
    parser.add_argument('--allowed-copyright', '-c', dest='allowed_copyright', type=str,
                        default='Tenstorrent AI ULC',
                        help='allowed copyright')
    parser.add_argument('--loglevel', '-l', type=lambda x: x.upper(), choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='WARNING', help='set log level')
    parser.add_argument('--dryrun', '-n', action=argparse.BooleanOptionalAction,
                        default=False, help='dryrun')
    return parser


class IgnoreFileModel(BaseModel):
    """
    Model to hold ignore and warning patterns from the ignore specification file.
    """

    ignore: list[str] = []  # File-patterns that should be ignored
    warning: list[str] = []  # File-patterns that should not cause an error, but should be logged as warnings


def collect_git_status_files(gitignore_flag: bool) -> dict[str, str]:
    """
    Collect files in the current directory that are ignored by git.
    """
    if not gitignore_flag:
        return {}
    if not os.path.exists('.git'):
        logger.error('No .git directory found in the current directory.')
        raise FileNotFoundError('.git directory not found')
    files_status: dict[str, str] = {}
    with os.popen('git status --porcelain -uall --ignored --untracked-files=all') as fin:
        for line in fin:
            line = line.strip()
            status_indicator, filename = line[:2], line[3:]
            if filename[0] == '"' and filename[-1] == '"':
                filename = filename[1:-1]
            if status_indicator == '!!':
                status = 'ignored'
            elif status_indicator == '??':
                status = 'untracked'
            else:
                status = 'active'
            files_status[filename] = status
    return files_status


def collect_all_files(dirname: str) -> list[str]:
    """
    Collect all files in the current directory and its subdirectories.
    """
    files: list[str] = []
    filename: str
    root: str
    _dirs: list[str]
    filenames: list[str]
    for root, _dirs, filenames in os.walk(dirname):
        # Skip .git directory; Only the startswith condition will wrongly match .gitHub directory
        if root == dirname + '/.git' or root.startswith(dirname + '/.git/'):
            continue
        for filename in filenames:
            files.append(os.path.join(root, filename).replace(dirname + '/', ''))
    return sorted(files)


def classify_file(filename: str) -> tuple[str, str]:
    ext = os.path.splitext(filename)[1]
    try:
        lang = ext_2_lang(ext)
    except Exception as e:
        logger.warning(f'Error classifying file {filename}: {e}')
        lang = 'unknown'
    return ext, lang


def analyze_file(filename: str, allowed_licenses: list[str], allowed_copyright: str) -> tuple[SPDXHeaderStatus, SPDXHeaderStatus]:
    """
    Analyze a file to determine its extension and language.
    """
    ext, lang = classify_file(filename)
    license_status: SPDXHeaderStatus = SPDXHeaderStatus.ST_MISSING
    copyright_status: SPDXHeaderStatus = SPDXHeaderStatus.ST_MISSING
    if filename == 'LICENSE':
        # LICENSE file is always a special case. It will only have copyright text
        # as a special case, since the license "txt" file does not have a
        # SPDX-License-Identifier line
        result = parse_copyright_in_txt(filename, allowed_copyright)
        license_status = SPDXHeaderStatus.ST_OK
        copyright_status = result
    elif lang == 'unknown':
        logger.error(f'File {filename} has unknown extension {ext}. Skipping.')
    else:
        parser = LanguageParser(lang, allowed_licenses, allowed_copyright)
        parser_result = parser.parse(filename)
        license_status = parser_result['license']
        copyright_status = parser_result['copyright']
    return license_status, copyright_status


class LanguageParser:
    """
    Class to parse files based on their language.
    """

    def __init__(self, lang: str, allowed_licenses: list[str], allowed_copyright: str):
        self.lang = lang
        self.allowed_licenses = allowed_licenses
        self.allowed_copyright = allowed_copyright
        self.syntax = LANG_2_SYNTAX.get(lang, {})
        self.comment_syntax = self.syntax.get('comment', '')
        self.license_re = re.compile(self.comment_syntax + r'(?P<optional_space>\s*)' + SPDX_LICENSE.pattern)
        end_comment = self.syntax.get('end_comment', '')
        self.copyright_re = re.compile(self.comment_syntax + r'(?P<optional_space>\s*)' + SPDX_COPYRIGHT.pattern + r'(?P<suffix>' + end_comment + r')?')
        self.license_status: SPDXHeaderStatus = SPDXHeaderStatus.ST_MISSING

    def parse(self, filename: str) -> dict[str, SPDXHeaderStatus]:
        """
        Parse a file for SPDX headers.
        """
        result: dict[str, SPDXHeaderStatus] = {x: SPDXHeaderStatus.ST_MISSING for x in ['license', 'copyright']}
        with open(filename) as f:
            if len(contents := f.read()) == 0:
                for x in result:
                    result[x] = SPDXHeaderStatus.ST_OK
                return result
            for line in contents.splitlines():
                if (line := line.rstrip()).startswith(self.comment_syntax):
                    if (comment_parse_result := self.parse_comment(line)) is None:
                        continue
                    comment_type, comment_status = comment_parse_result
                    if comment_type == 'license':
                        result['license'] = comment_status
                    elif comment_type == 'copyright':
                        result['copyright'] = comment_status
                if all ([entry != SPDXHeaderStatus.ST_MISSING for entry in result.values()]): # ['license']['status'] != FileSPDXStatus.ST_MISSING and result['copyright']['status'] != FileSPDXStatus.ST_MISSING:
                    # If both license and copyright are found, we can stop parsing
                    break
        return result

    def parse_license(self, license_match) -> SPDXHeaderStatus:
        """
        Parse a license line for license information.
        """
        # license_text = license_match.group('license_text')
        if license_match.group('license_text') in self.allowed_licenses:
            return SPDXHeaderStatus.ST_OK
        else:
            return SPDXHeaderStatus.ST_INCORRECT

    def parse_copyright(self, copyright_match) -> SPDXHeaderStatus:
        """
        Parse a copyright line for copyright information.
        """
        copyright_text = copyright_match.group('copyright_text')
        if not (copyright_parts_match := COPYRIGHT_REGEX.search(copyright_text)):
            return SPDXHeaderStatus.ST_ILLFORMED
        if copyright_parts_match.group('cprt_holder') == self.allowed_copyright:
            return SPDXHeaderStatus.ST_OK
        else:
            return SPDXHeaderStatus.ST_INCORRECT

    def parse_comment(self, line) -> tuple[str, SPDXHeaderStatus] | None:
        """
        Parse a comment line for license and copyright information.
        """
        if license_match := self.license_re.search(line):
            return 'license', self.parse_license(license_match)
        if copyright_match := self.copyright_re.search(line):
            return 'copyright', self.parse_copyright(copyright_match)
        return None


def get_ignore_patterns(ignore_pattern_list: list[str]) -> Union[None, re.Pattern[str]]:
    """
    Read ignore patterns from a file.
    """
    ignore_patterns = [fnmatch.translate(pat) for pat in ignore_pattern_list]
    ignore_re = re.compile('|'.join(ignore_patterns))
    logger.debug(f'Ignoring files matching patterns: {ignore_patterns}')
    return ignore_re


def parse_copyright_in_txt(filename: str, allowed_copyright: str) -> SPDXHeaderStatus:
    """
    Parse a text file for copyright information.
    """
    with open(filename) as f:
        for line in f:
            if not (copyright_match := TEXT_COPYRIGHT_REGEX.search(line.strip())):
                continue
            if copyright_match.group('cprt_holder').strip() == allowed_copyright:
                return SPDXHeaderStatus.ST_OK
            else:
                return SPDXHeaderStatus.ST_INCORRECT
    return SPDXHeaderStatus.ST_MISSING


def get_active_files(gitignore_flag: bool, ignore_spec: IgnoreFileModel) -> list[str]:
    """
    Get a list of active files in the current directory, excluding those that match
    the .gitignore patterns and any additional ignore patterns specified in the ignore_spec.
    """
    git_status: dict[str, str] = collect_git_status_files(gitignore_flag)
    all_files: list[str] = collect_all_files('.')

    active_files: list[str] = []
    ignore_re: IgnorePattern = get_ignore_patterns(ignore_spec.ignore)
    for f in all_files:
        if git_status.get(f, '') == 'ignored':
            logger.debug(f'File {f} matches .gitignore pattern. Skipping.')
            continue
        if ignore_re is not None and ignore_re.search(f):
            logger.debug(f'File {f} matches ignore pattern. Skipping.')
            continue
        active_files.append(f)
    return active_files


def main() -> int:
    args: argparse.Namespace = create_args().parse_args()
    logger.remove()
    logger.add(sys.stdout, format='{level}:{message}', level=args.loglevel)

    ignore_spec = IgnoreFileModel(**yaml.safe_load(open(args.ignorelist))) if args.ignorelist else IgnoreFileModel()

    active_files = get_active_files(args.gitignore, ignore_spec)
    warn_re: IgnorePattern = get_ignore_patterns(ignore_spec.warning) if args.ignorelist else None

    num_errors = 0
    for fname in active_files:
        license_status, copyright_status = analyze_file(fname, args.allowed_licenses, args.allowed_copyright)
        warn_flag = warn_re is not None and warn_re.search(fname)
        if license_status == SPDXHeaderStatus.ST_OK:
            logger.info(f'{fname}: License {license_status.value}')
        elif warn_flag:
            logger.warning(f'{fname}: License {license_status.value}')
        else:
            num_errors += 1
            logger.error(f'{fname}: License {license_status.value}')
        if copyright_status == SPDXHeaderStatus.ST_OK:
            logger.info(f'{fname}: Copyright {copyright_status.value}')
        elif warn_flag:
            logger.warning(f'{fname}: Copyright {copyright_status.value}')
        else:
            num_errors += 1
            logger.error(f'{fname}: Copyright {copyright_status.value}')
    return 0 if num_errors == 0 else 1


if __name__ == '__main__':
    exit(main())
