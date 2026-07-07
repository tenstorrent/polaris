#!/usr/bin/env python
"""
Guard against unpinned dependencies in the conda environment definition files.

An unpinned dependency (one with no version specifier) lets a fresh `conda env
create` solve to whatever is newest at the time. That silently drifts the
toolchain and can redden CI on unchanged code — e.g. a newer Pillow tightening
`Image.fromarray`'s inline type stubs, or a newer type-stub / ruff release
introducing rules the code has never seen. See issue #475.

This checker parses each environment YAML's `dependencies:` list (both conda
entries and the nested `pip:` block) and fails if any entry lacks a version
pin. Channels, comments, and the `pip:` key itself are ignored.
"""
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys

import yaml
from loguru import logger

# A dependency counts as pinned only with an EXACT version specifier: conda
# 'name=ver' (optionally a '=build' suffix) or pip 'name==ver'. Range/inequality
# operators (>=, <=, >, <, !=, ~=) still let the toolchain drift, so they do NOT
# count — defeating them is the whole point of the guard (issue #475).
#
# Note: the env files are not expected to use '@' / URL / VCS (PEP 508 direct)
# references — they carry no exact version specifier, so they are deliberately
# flagged as unpinned; if one is ever added it surfaces for explicit human
# review rather than being silently accepted.
# Matched as whole operators (not bare characters) so a PEP 440 epoch pin like
# 'pkg==1!2.0' is not mistaken for the '!=' range operator.
_RANGE_OPERATORS = ('>=', '<=', '!=', '~=', '<', '>')

DEFAULT_ENV_FILES = ['environment.yaml', 'envdev.yaml']


def _is_pinned(spec: str) -> bool:
    """True only for an exact pin (conda '=', pip '=='). Range/inequality
    specifiers and wildcard pins are treated as unpinned, since they still
    permit the resolved version to drift."""
    # Drop any PEP 508 environment marker (everything after ';'): its operators
    # (e.g. python_version<'3.12') constrain the environment, not the package
    # version, so they must not be mistaken for a range spec on the package.
    spec = spec.split(';', 1)[0].strip()
    if '*' in spec:  # wildcard pin, e.g. 'pkg=1.2.*' — still allows patch drift
        return False
    if any(op in spec for op in _RANGE_OPERATORS):
        return False
    if '=' not in spec:
        return False
    # reject a bare operator with no version part, e.g. 'pkg=' / 'pkg=='
    return spec.rsplit('=', 1)[1].strip() != ''


def find_unpinned(env_file: str) -> list[str]:
    """Return the list of unpinned dependency specs in one environment YAML."""
    with open(env_file, encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}

    unpinned: list[str] = []
    for dep in doc.get('dependencies', []):
        if isinstance(dep, str):
            # conda dependency, e.g. 'numpy=2.2.3' or (unpinned) 'gitpython'
            if not _is_pinned(dep):
                unpinned.append(dep)
        elif isinstance(dep, dict):
            # the nested pip block: {'pip': ['pytest==8.3.4', 'types-requests', ...]}
            # an empty block ('- pip:' with no items) yields {'pip': None} -> treat as []
            for pip_spec in (dep.get('pip') or []):
                if isinstance(pip_spec, str) and not _is_pinned(pip_spec):
                    unpinned.append(f'pip: {pip_spec}')
    return unpinned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'env_files',
        nargs='*',
        default=DEFAULT_ENV_FILES,
        help=f'environment YAML files to check (default: {" ".join(DEFAULT_ENV_FILES)})',
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format='<level>{level: <8}</level> | <level>{message}</level>')

    total = 0
    for env_file in args.env_files:
        unpinned = find_unpinned(env_file)
        if unpinned:
            total += len(unpinned)
            logger.error(f'{env_file}: {len(unpinned)} unpinned dependency(ies):')
            for spec in unpinned:
                logger.error(f'    - {spec}')
        else:
            logger.info(f'{env_file}: all dependencies pinned')

    if total:
        logger.error(
            f'{total} unpinned dependency(ies) found. Pin each to its current '
            'installed version (see issue #475) to prevent CI-breaking env drift.'
        )
        return 1
    logger.success('all environment dependencies are pinned')
    return 0


if __name__ == '__main__':
    sys.exit(main())
