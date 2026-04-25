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
COPYRIGHT_REGEX_PARTS = ('Copyright', '©', '[(][cC][)]')
COPYRIGHT_REGEX = re.compile(f'(?P<cprt_string>{"|".join(COPYRIGHT_REGEX_PARTS)})\\s+(?P<cprt_years>\\d{{4}}(-\\d{{4}})?)\\s+(?P<cprt_holder>.*)')
COPYRIGHT_PREFIX_STRINGS = ('Copyright', '©', '(c)', '(C)')
COPYRIGHT_PREFIXES = tuple(s + ' ' for s in COPYRIGHT_PREFIX_STRINGS)

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
    # Normalize dirname to use forward slashes
    dirname_normalized = dirname.replace('\\', '/')
    for root, _dirs, filenames in os.walk(dirname):
        # Normalize root path
        root_normalized = root.replace('\\', '/')
        # Skip .git directory; Only the startswith condition will wrongly match .gitHub directory
        if root_normalized == dirname_normalized + '/.git' or root_normalized.startswith(dirname_normalized + '/.git/'):
            continue
        for filename in filenames:
            # Build relative path with forward slashes
            full_path = os.path.join(root, filename).replace('\\', '/')
            rel_path = os.path.relpath(full_path, dirname).replace(os.sep, '/')
            files.append(rel_path)
    return sorted(files)


def classify_file(filename: str) -> tuple[str, str]:
    ext = os.path.splitext(filename)[1]
    try:
        lang = ext_2_lang(ext)
    except Exception as e:
        logger.warning(f'Error classifying file {filename}: {e}')
        lang = 'unknown'
    return ext, lang


def is_text_file(filename: str) -> tuple[bool, str | None]:
    """
    Check if a file appears to be text.
    Reads only a small prefix once, then attempts to decode that sample with
    UTF-8 and common fallbacks. Returns (is_text, encoding) where encoding
    is the successful encoding or None if not text.
    Uses binary sniffing to reject files with NUL bytes.
    """
    # Only accept strict UTF-8 encodings for unknown extensions to avoid
    # misclassifying binary files (latin-1/cp1252 decode any byte sequence)
    encodings = ['utf-8', 'utf-8-sig']
    sample_size = 16384  # Read first 16KB for detection

    try:
        with open(filename, 'rb') as f:
            sample = f.read(sample_size)
    except Exception as e:
        logger.debug(f'Error reading file {filename}: {e}')
        return False, None

    # Empty files are considered text
    if not sample:
        return True, 'utf-8'

    # Binary sniff: reject files with NUL bytes
    if b'\x00' in sample:
        return False, None

    # Try to decode with allowed encodings
    for encoding in encodings:
        try:
            sample.decode(encoding)
            return True, encoding
        except (UnicodeDecodeError, ValueError):
            continue

    return False, None


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
    elif filename.endswith('Dockerfile') or '/Dockerfile' in filename:
        # Dockerfile is a special case, skip SPDX check
        license_status = SPDXHeaderStatus.ST_OK
        copyright_status = SPDXHeaderStatus.ST_OK
    elif lang == 'unknown':
        # Try to parse as text file with multiple comment syntaxes
        is_text, encoding = is_text_file(filename)
        if is_text and encoding:
            logger.debug(f'File {filename} has unknown extension {ext}. Attempting multi-syntax parsing.')
            multi_parser = MultiSyntaxParser(allowed_licenses, allowed_copyrights, warn_flag)
            try:
                parser_result = multi_parser.parse(filename, encoding)
                license_status = parser_result['license']
                copyright_status = parser_result['copyright']
            except UnicodeDecodeError as exc:
                logger.error(
                    f'File {filename} was identified as text but could not be decoded during parsing: {exc}. '
                    'Skipping.'
                )
        else:
            logger.error(f'File {filename} has unknown extension {ext}. Skipping.')
    else:
        lang_parser = LanguageParser(lang, allowed_licenses, allowed_copyrights, warn_flag)
        parser_result = lang_parser.parse(filename)
        license_status = parser_result['license']
        copyright_status = parser_result['copyright']
    return license_status, copyright_status


class MultiSyntaxParser:
    """
    Parser that tries multiple comment syntaxes for files with unknown extensions.
    Attempts to detect SPDX headers using all known comment syntaxes.
    """

    def __init__(self, allowed_licenses: list[str], allowed_copyrights: list[str],
                 warn_flag: bool = False):
        self.allowed_licenses = allowed_licenses
        self.allowed_copyrights = allowed_copyrights
        self.warn_flag = warn_flag
        self.parsing: str | None = None
        self.found_licenses: list[tuple[str, SPDXHeaderStatus]] = []
        self.found_copyrights: list[tuple[str, SPDXHeaderStatus]] = []

    def parse(self, filename: str, encoding: str = 'utf-8') -> dict[str, SPDXHeaderStatus]:
        """
        Parse a file trying all known comment syntaxes.
        Aggregates all SPDX header matches found across syntaxes and returns
        the overall status for each header type.
        Args:
            filename: Path to the file to parse
            encoding: Text encoding to use (should match what is_text_file() detected)
        """
        self.parsing = filename
        self.found_licenses = []
        self.found_copyrights = []

        try:
            with open(filename, encoding=encoding) as f:
                contents = f.read()
                if len(contents) == 0:
                    # Empty files with unknown extensions should fail validation
                    # since they don't contain required SPDX headers
                    self.parsing = None
                    return {'license': SPDXHeaderStatus.ST_MISSING, 'copyright': SPDXHeaderStatus.ST_MISSING}

                # Try each known comment syntax once, de-duplicating languages
                # that share the same comment delimiters.
                seen_syntaxes = set()
                for syntax in LANG_2_SYNTAX.values():
                    syntax_key = (syntax.get('comment'), syntax.get('end_comment'))
                    if syntax_key in seen_syntaxes:
                        continue
                    seen_syntaxes.add(syntax_key)
                    self._parse_with_syntax(contents, syntax)

        except (IOError, OSError) as e:
            logger.debug(f'Error reading file {filename}: {e}')
            self.parsing = None
            return {'license': SPDXHeaderStatus.ST_MISSING, 'copyright': SPDXHeaderStatus.ST_MISSING}

        # Determine overall status based on all found entries
        result = {
            'license': self._determine_overall_status(self.found_licenses),
            'copyright': self._determine_overall_status(self.found_copyrights)
        }

        self.parsing = None
        return result

    def _parse_with_syntax(self, contents: str, syntax: dict[str, str]) -> None:
        """
        Parse file contents using a specific comment syntax.
        """
        comment_syntax = syntax.get('comment', '')
        end_comment = syntax.get('end_comment', '')
        
        # Escape special regex characters in comment syntax
        escaped_comment = re.escape(comment_syntax)
        escaped_end_comment = re.escape(end_comment)
        
        # Build regex patterns for this syntax
        license_re = re.compile(escaped_comment + r'(?P<optional_space>\s*)' + SPDX_LICENSE.pattern + r'(?P<optional_space2>\s*)(?P<suffix>' + escaped_end_comment + r')')
        copyright_re = re.compile(escaped_comment + r'(?P<optional_space>\s*)' + SPDX_COPYRIGHT.pattern + r'(?P<optional_space2>\s*)(?P<suffix>' + escaped_end_comment + r')')

        # Parse all lines
        for line in contents.splitlines():
            line = line.rstrip()
            if not line.startswith(comment_syntax):
                continue

            # Try license pattern
            if license_match := license_re.search(line):
                license_text = license_match.group('license_text').strip()
                status = self._check_license(license_text)
                self.found_licenses.append((license_text, status))

            # Try copyright pattern
            if copyright_match := copyright_re.search(line):
                copyright_text = copyright_match.group('copyright_text')
                status = self._check_copyright(copyright_text)
                self.found_copyrights.append((copyright_text, status))

    def _check_license(self, license_text: str) -> SPDXHeaderStatus:
        """Check if license text is in allowed list."""
        if license_text in self.allowed_licenses:
            return SPDXHeaderStatus.ST_OK
        else:
            if self.warn_flag:
                logger.warning(f'{self.parsing}: wrong license text: "{license_text}", allowed licenses: {self.allowed_licenses}')
            else:
                logger.error(f'{self.parsing}: wrong license text: "{license_text}", allowed licenses: {self.allowed_licenses}')
            return SPDXHeaderStatus.ST_INCORRECT

    def _check_copyright(self, copyright_text: str) -> SPDXHeaderStatus:
        """Check if copyright text matches allowed patterns."""
        if not (copyright_parts_match := COPYRIGHT_REGEX.search(copyright_text)):
            # Fallback: accept line without year if full text or holder part matches allowed copyright
            cprt_holder_fallback = copyright_text.strip()
            if cprt_holder_fallback in self.allowed_copyrights:
                logger.debug(f'{self.parsing}: Copyright line does not match expected format but holder matches allowed list: "{copyright_text}"')
                return SPDXHeaderStatus.ST_OK
            # Strip leading copyright symbols
            for prefix in COPYRIGHT_PREFIXES:
                if cprt_holder_fallback.startswith(prefix):
                    if cprt_holder_fallback[len(prefix):].strip() in self.allowed_copyrights:
                        logger.debug(f'{self.parsing}: Copyright line does not match expected format but holder matches allowed list after stripping prefix: "{copyright_text}"')
                        return SPDXHeaderStatus.ST_OK
                    break
            logger.debug(f'{self.parsing}: Copyright line is ill-formed: "{copyright_text}"')
            return SPDXHeaderStatus.ST_ILLFORMED

        cprt_holder = copyright_parts_match.group('cprt_holder').strip()

        if cprt_holder in self.allowed_copyrights:
            return SPDXHeaderStatus.ST_OK
        else:
            if self.warn_flag:
                logger.warning(f'{self.parsing}: wrong copyright holder: "{cprt_holder}", allowed copyrights: {self.allowed_copyrights}')
            else:
                logger.error(f'{self.parsing}: wrong copyright holder: "{cprt_holder}", allowed copyrights: {self.allowed_copyrights}')
            return SPDXHeaderStatus.ST_INCORRECT

    def _determine_overall_status(self, found_entries: list[tuple[str, SPDXHeaderStatus]]) -> SPDXHeaderStatus:
        """
        Determine the overall status based on all found entries.
        If no entries are found, status is MISSING.
        If any entry has an error status, the overall status reflects the error.
        If all entries are OK, the overall status is OK.
        """
        if not found_entries:
            return SPDXHeaderStatus.ST_MISSING

        # Check if all entries are OK
        if all(status == SPDXHeaderStatus.ST_OK for _, status in found_entries):
            return SPDXHeaderStatus.ST_OK

        # If there are any errors, prioritize the most severe error status
        error_statuses = [status for _, status in found_entries if status != SPDXHeaderStatus.ST_OK]
        if SPDXHeaderStatus.ST_INCORRECT in error_statuses:
            return SPDXHeaderStatus.ST_INCORRECT
        elif SPDXHeaderStatus.ST_ILLFORMED in error_statuses:
            return SPDXHeaderStatus.ST_ILLFORMED
        raise AssertionError("Unexpected error status fallback in _determine_overall_status")


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
        self.found_licenses: list[tuple[str, SPDXHeaderStatus]] = []
        self.found_copyrights: list[tuple[str, SPDXHeaderStatus]] = []

    def parse(self, filename: str) -> dict[str, SPDXHeaderStatus]:
        """
        Parse a file for SPDX headers.
        """
        self.parsing = filename
        self.found_licenses = []
        self.found_copyrights = []

        with open(filename) as f:
            if len(contents := f.read()) == 0:
                self.parsing = None
                return {'license': SPDXHeaderStatus.ST_OK, 'copyright': SPDXHeaderStatus.ST_OK}

            # Parse all lines to collect all license and copyright entries
            for line in contents.splitlines():
                if (line := line.rstrip()).startswith(self.comment_syntax):
                    if (comment_parse_result := self.parse_comment(line)) is None:
                        continue
                    comment_type, comment_status, content = comment_parse_result
                    if comment_type == 'license':
                        self.found_licenses.append((content, comment_status))
                    elif comment_type == 'copyright':
                        self.found_copyrights.append((content, comment_status))

        # Determine overall status based on all found entries
        result = {
            'license': self._determine_overall_status(self.found_licenses),
            'copyright': self._determine_overall_status(self.found_copyrights)
        }

        self.parsing = None
        return result

    def _determine_overall_status(self, found_entries: list[tuple[str, SPDXHeaderStatus]]) -> SPDXHeaderStatus:
        """
        Determine the overall status based on all found entries.
        If no entries are found, status is MISSING.
        If any entry has an error status, the overall status reflects the error.
        If all entries are OK, the overall status is OK.
        """
        if not found_entries:
            return SPDXHeaderStatus.ST_MISSING

        # Check if all entries are OK
        if all(status == SPDXHeaderStatus.ST_OK for _, status in found_entries):
            return SPDXHeaderStatus.ST_OK

        # If there are any errors, prioritize the most severe error status
        # Error prioritization: MISSING is handled above (empty check).
        # Among found entries, INCORRECT is prioritized over ILLFORMED.
        # OK is not an error status and is excluded from error_statuses.
        # Increasing severity: OK < ILLFORMED < INCORRECT
        error_statuses = [status for _, status in found_entries if status != SPDXHeaderStatus.ST_OK]
        if SPDXHeaderStatus.ST_INCORRECT in error_statuses:
            return SPDXHeaderStatus.ST_INCORRECT
        elif SPDXHeaderStatus.ST_ILLFORMED in error_statuses:
            return SPDXHeaderStatus.ST_ILLFORMED
        raise AssertionError("Unexpected error status fallback in _determine_overall_status")


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
            # Fallback: accept line without year if full text or holder part matches allowed copyright
            cprt_holder_fallback = copyright_text.strip()
            if cprt_holder_fallback in self.allowed_copyrights:
                logger.debug(f'{self.parsing}: Copyright line does not match expected format but holder matches allowed list: "{copyright_text}"')
                return SPDXHeaderStatus.ST_OK
            # Strip leading copyright symbols so "Copyright Holder" etc. can match "Holder" in allowed list
            for prefix in COPYRIGHT_PREFIXES:
                if cprt_holder_fallback.startswith(prefix):
                    if cprt_holder_fallback[len(prefix):].strip() in self.allowed_copyrights:
                        logger.debug(f'{self.parsing}: Copyright line does not match expected format but holder matches allowed list after stripping prefix: "{copyright_text}"')
                        return SPDXHeaderStatus.ST_OK
                    break  # If one prefix is at the beginning, we should not check the other prefix as it won't match
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

    def parse_comment(self, line) -> tuple[str, SPDXHeaderStatus, str] | None:
        """
        Parse a comment line for license and copyright information.
        """
        if license_match := self.license_re.search(line):
            license_text = license_match.group('license_text').strip()
            return 'license', self.parse_license(license_match), license_text
        if copyright_match := self.copyright_re.search(line):
            copyright_text = copyright_match.group('copyright_text')
            return 'copyright', self.parse_copyright(copyright_match), copyright_text
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
