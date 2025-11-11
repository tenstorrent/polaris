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
    'shell': ['.sh'],
    'javascript': ['.js', '.mjs', '.cjs'],
    'html': ['.html', '.htm'],
    'css': ['.css'],
    'yaml': ['.yaml', '.yml'],
}

LANG_2_SYNTAX = {
    'python': {'comment': '#'},
    'shell': {'comment': '#'},
    'javascript': {'comment': '//'},
    'html': {'comment': '<!--', 'end_comment': '-->'},
    'css': {'comment': '/*', 'end_comment': '*/'},
    'yaml': {'comment': '#'},
}

type IgnorePattern = Union[None, re.Pattern[str]]

SPDX_LICENSE_PREFIX = 'SPDX-License-Identifier:'
SPDX_COPYRIGHT_PREFIX = 'SPDX-FileCopyrightText:'
SPDX_LICENSE = re.compile(SPDX_LICENSE_PREFIX + '\\s+(?P<license_text>.*)')
SPDX_COPYRIGHT = re.compile(SPDX_COPYRIGHT_PREFIX + '\\s+(?P<copyright_text>.*)')
COPYRIGHT_REGEX = re.compile('(?P<cprt_string>©|[(][cC][)])\\s+(?P<cprt_years>\\d{4}(-\\d{4})?)\\s+(?P<cprt_holder>.*)')

# Valid SPDX license identifiers - maintained as a hardcoded list
VALID_SPDX_LICENSES = [
    'Apache-2.0',
    # Add more valid SPDX identifiers as needed
]


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
    parser.add_argument('--config', '-c', type=str,
                        default='.github/spdxchecker-config.yml',
                        help='SPDX checker configuration file path')
    parser.add_argument('--allowed-licenses', '-a', dest='allowed_licenses', type=str, nargs='*',
                        default=None,
                        help='list of allowed licenses (overrides config file)')
    parser.add_argument('--allowed-copyrights', dest='allowed_copyrights', type=str, nargs='*',
                        default=None,
                        help='list of allowed copyright holders (overrides config file)')
    parser.add_argument('--validate-spdx-licenses', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='validate licenses against known SPDX identifiers (default: True)')
    parser.add_argument('--loglevel', '-l', type=lambda x: x.upper(), choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='set log level')
    parser.add_argument('--dryrun', '-n', action=argparse.BooleanOptionalAction,
                        default=False, help='dryrun')
    return parser


class ConfigFileModel(BaseModel):
    """
    Model to hold SPDX checker configuration including ignore patterns,
    allowed licenses, and allowed copyright holders.
    """

    ignore: list[str] = []  # File-patterns that should be ignored
    warning: list[str] = []  # File-patterns that should not cause an error, but should be logged as warnings
    allowed_licenses: list[str] = []  # Allowed SPDX license identifiers
    allowed_copyrights: list[str] = []  # Allowed copyright holders

def validate_config(config: ConfigFileModel, validate_spdx: bool = True) -> None:
    """
    Validate the configuration file for correctness.
    Checks that licenses are valid SPDX identifiers if validation is enabled.
    Raises ValueError if invalid licenses are found (when validation is enabled).
    
    Args:
        config: Configuration model to validate
        validate_spdx: If True, validate licenses against known SPDX identifiers (default: True)
    """
    # Validate licenses against known SPDX identifiers (strict validation)
    if validate_spdx and config.allowed_licenses:
        invalid_licenses = [lic for lic in config.allowed_licenses if lic not in VALID_SPDX_LICENSES]
        if invalid_licenses:
            logger.error(f'Configuration contains invalid SPDX license identifiers: {invalid_licenses}')
            logger.error(f'Valid SPDX identifiers are: {VALID_SPDX_LICENSES}')
            raise ValueError(f'Invalid SPDX license identifiers in configuration: {invalid_licenses}')

    # Check that at least one license is specified
    if not config.allowed_licenses:
        logger.warning('No allowed licenses specified in configuration')

    # Check that at least one copyright holder is specified
    if not config.allowed_copyrights:
        logger.warning('No allowed copyright holders specified in configuration')


def load_config(config_path: str, validate_spdx: bool = True) -> ConfigFileModel:
    """
    Load and validate configuration from file.
    
    Args:
        config_path: Path to the configuration file
        validate_spdx: If True, validate licenses against known SPDX identifiers (default: True)
    """
    if not os.path.exists(config_path):
        logger.error(f'Configuration file not found: {config_path}')
        raise FileNotFoundError(f'Configuration file not found: {config_path}')

    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            config_data = {}

        config = ConfigFileModel(**config_data)
        validate_config(config, validate_spdx)
        return config
    except yaml.YAMLError as e:
        logger.error(f'Invalid YAML in configuration file {config_path}: {e}')
        raise
    except Exception as e:
        logger.error(f'Error loading configuration file {config_path}: {e}')
        raise


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


def analyze_file(filename: str, allowed_licenses: list[str], allowed_copyrights: list[str],
                 warn_flag: bool = False) -> tuple[SPDXHeaderStatus, SPDXHeaderStatus]:
    """
    Analyze a file to determine its extension and language.
    """
    ext, lang = classify_file(filename)
    license_status: SPDXHeaderStatus = SPDXHeaderStatus.ST_MISSING
    copyright_status: SPDXHeaderStatus = SPDXHeaderStatus.ST_MISSING
    if filename == 'LICENSE':
        # LICENSE file is always a special case. It should NOT have copyright text
        # and does not need SPDX-License-Identifier line as it is the license text itself
        license_status = SPDXHeaderStatus.ST_OK
        copyright_status = SPDXHeaderStatus.ST_OK
    elif lang == 'unknown':
        logger.error(f'File {filename} has unknown extension {ext}. Skipping.')
    else:
        parser = LanguageParser(lang, allowed_licenses, allowed_copyrights, warn_flag)
        parser_result = parser.parse(filename)
        license_status = parser_result['license']
        copyright_status = parser_result['copyright']
    return license_status, copyright_status


class LanguageParser:
    """
    Class to parse files based on their language.
    """

    def __init__(self, lang: str, allowed_licenses: list[str], allowed_copyrights: list[str],
                 warn_flag: bool = False):
        self.lang = lang
        self.allowed_licenses = allowed_licenses
        self.allowed_copyrights = allowed_copyrights
        self.syntax = LANG_2_SYNTAX.get(lang, {})
        self.comment_syntax = self.syntax.get('comment', '')
        end_comment = self.syntax.get('end_comment', '')
        self.license_re = re.compile(self.comment_syntax + r'(?P<optional_space>\s*)' + SPDX_LICENSE.pattern + r'(?P<optional_space2>\s*)(?P<suffix>' + end_comment + r')')
        self.copyright_re = re.compile(self.comment_syntax + r'(?P<optional_space>\s*)' + SPDX_COPYRIGHT.pattern + r'(?P<optional_space2>\s*)(?P<suffix>' + end_comment + r')')
        self.license_status: SPDXHeaderStatus = SPDXHeaderStatus.ST_MISSING
        self.warn_flag = warn_flag
        self.parsing: str | None = None

    def parse(self, filename: str) -> dict[str, SPDXHeaderStatus]:
        """
        Parse a file for SPDX headers.
        """
        self.parsing = filename
        result: dict[str, SPDXHeaderStatus] = {x: SPDXHeaderStatus.ST_MISSING for x in ['license', 'copyright']}
        with open(filename) as f:
            if len(contents := f.read()) == 0:
                for x in result:
                    result[x] = SPDXHeaderStatus.ST_OK
                self.parsing = None
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
        self.parsing = None
        return result

    def parse_license(self, license_match) -> SPDXHeaderStatus:
        """
        Parse a license line for license information.
        """
        if (license_text := license_match.group('license_text').strip()) in self.allowed_licenses:
            return SPDXHeaderStatus.ST_OK
        else:
            if self.warn_flag:
                logger.warning(f'{self.parsing}: wrong license text: "{license_text}", allowed licenses: {self.allowed_licenses}')
            else:
                logger.error(f'{self.parsing}: wrong license text: "{license_text}", allowed licenses: {self.allowed_licenses}')
            return SPDXHeaderStatus.ST_INCORRECT

    def parse_copyright(self, copyright_match) -> SPDXHeaderStatus:
        """
        Parse a copyright line for copyright information.
        Supports multiple copyright holders through exact string matching.
        """
        copyright_text = copyright_match.group('copyright_text')
        if not (copyright_parts_match := COPYRIGHT_REGEX.search(copyright_text)):
            logger.debug(f'{self.parsing}: Copyright line is ill-formed: "{copyright_text}"')
            return SPDXHeaderStatus.ST_ILLFORMED

        cprt_holder = copyright_parts_match.group('cprt_holder').strip()

        # Check if copyright holder matches any of the allowed copyright holders (exact match)
        if cprt_holder in self.allowed_copyrights:
            return SPDXHeaderStatus.ST_OK
        else:
            # Log mismatch at appropriate level based on warn_flag
            if self.warn_flag:
                logger.warning(f'{self.parsing}: wrong copyright holder: "{cprt_holder}", allowed copyrights: {self.allowed_copyrights}')
            else:
                logger.error(f'{self.parsing}: wrong copyright holder: "{cprt_holder}", allowed copyrights: {self.allowed_copyrights}')
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
    if ignore_patterns == []:
        ignore_re = None
    else:
        ignore_re = re.compile('|'.join(ignore_patterns))
    logger.debug(f'Ignoring files matching patterns: {ignore_patterns=} {ignore_re=}')
    return ignore_re



def get_active_files(gitignore_flag: bool, config: ConfigFileModel) -> list[str]:
    """
    Get a list of active files in the current directory, excluding those that match
    the .gitignore patterns and any additional ignore patterns specified in the config.
    """
    git_status: dict[str, str] = collect_git_status_files(gitignore_flag)
    all_files: list[str] = collect_all_files('.')

    active_files: list[str] = []
    ignore_re: IgnorePattern = get_ignore_patterns(config.ignore)
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
    logger.add(sys.stdout,
        format='<level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>',
        level=args.loglevel)

    # Load configuration from file
    logger.info(f'Using configuration file: {args.config}')
    config = load_config(args.config, args.validate_spdx_licenses)

    # Log validation mode
    if args.validate_spdx_licenses:
        logger.info('SPDX license validation is enabled')
    else:
        logger.info('SPDX license validation is disabled')

    # Command-line arguments override config file settings
    allowed_licenses = args.allowed_licenses if args.allowed_licenses is not None else config.allowed_licenses
    allowed_copyrights = args.allowed_copyrights if args.allowed_copyrights is not None else config.allowed_copyrights

    # Log overrides if applied
    if args.allowed_licenses is not None:
        logger.info(f'Allowed licenses overridden by command-line arguments: {allowed_licenses}')
    else:
        logger.info(f'Allowed licenses from config file: {allowed_licenses}')

    if args.allowed_copyrights is not None:
        logger.info(f'Allowed copyrights overridden by command-line arguments: {allowed_copyrights}')
    else:
        logger.info(f'Allowed copyrights from config file: {allowed_copyrights}')

    # Ensure we have at least one license and copyright configured
    if not allowed_licenses:
        logger.error('No allowed licenses specified. Please configure allowed_licenses in config file or use --allowed-licenses')
        return 1

    if not allowed_copyrights:
        logger.error('No allowed copyright holders specified. Please configure allowed_copyrights in config file or use --allowed-copyrights')
        return 1

    active_files = get_active_files(args.gitignore, config)
    warn_re: IgnorePattern = get_ignore_patterns(config.warning)
    num_errors = 0
    for fname in active_files:
        warn_flag: bool = warn_re is not None and warn_re.search(fname) is not None
        license_status, copyright_status = analyze_file(fname, allowed_licenses, allowed_copyrights, warn_flag)
        status_message = f'License: {license_status.value}, Copyright: {copyright_status.value}'
        if license_status == SPDXHeaderStatus.ST_OK and copyright_status == SPDXHeaderStatus.ST_OK:
            logger.debug(f'{fname}: {status_message}')
        elif warn_flag:
            logger.warning(f'{fname}: {status_message}')
        else:
            logger.error(f'{fname}: {status_message}')
            num_errors += 1
    if num_errors:
        logger.error('Valid license lines : "<comment> {prefix} <license>", where license is one of {licenses}',
                     prefix=SPDX_LICENSE_PREFIX, licenses=allowed_licenses)
        logger.error('Valid copyright line: "<comment> {prefix} <copyright>", where copyright is one of {copyrights}',
                     prefix=SPDX_COPYRIGHT_PREFIX, copyrights=allowed_copyrights)
    else:
        logger.info('All files have valid SPDX headers.')
    return 0 if num_errors == 0 else 1


if __name__ == '__main__':
    exit(main())
