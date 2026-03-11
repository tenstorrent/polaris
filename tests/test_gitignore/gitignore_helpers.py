#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Helpers for running git commands and .gitignore pattern matching (used by .gitignore tests)."""
from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path
from typing import Iterable


class GitignoreSpec:
    """
    Best-effort matcher for .gitignore patterns for unit testing.

    This class does not implement full git ignore semantics (e.g. directory-only
    rules, negation inside already-ignored parent directories, or **). Use
    actual git commands (e.g. git check-ignore) for authoritative behavior.
    """

    def __init__(self, patterns: Iterable[str]) -> None:
        """Initialize with list of gitignore pattern lines."""
        self.patterns: list[str] = []
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern and not pattern.startswith('#'):
                self.patterns.append(pattern)

    def match_file(self, filepath: str) -> bool:
        """
        Check if a file matches any ignore pattern.

        Returns True if file should be ignored, False otherwise.
        Handles negation patterns (starting with !) correctly.
        """
        filepath = filepath.rstrip('/')
        ignored = False

        for pattern in self.patterns:
            if pattern.startswith('!'):
                # Negation pattern - if file matches, it should NOT be ignored
                neg_pattern = pattern[1:]
                if self._matches_pattern(filepath, neg_pattern):
                    ignored = False
            else:
                # Regular ignore pattern
                if self._matches_pattern(filepath, pattern):
                    ignored = True

        return ignored

    def _matches_pattern(self, filepath: str, pattern: str) -> bool:
        """Check if filepath matches a single pattern."""
        # Normalize: strip trailing / so e.g. __pycache__/ matches path components
        pattern = pattern.rstrip('/')

        # Simple pattern matching
        # Handle patterns like __* which match files starting with __
        if pattern.startswith('__') and pattern.endswith('*'):
            # Match files starting with __
            filename = Path(filepath).name
            if filename.startswith('__'):
                return True

        # Exact filename match anywhere in path
        if '/' not in pattern and '*' not in pattern:
            filename = Path(filepath).name
            if filename == pattern:
                return True

        # Use fnmatch for wildcard patterns
        if fnmatch.fnmatch(filepath, pattern):
            return True
        if fnmatch.fnmatch(filepath, '*/' + pattern):
            return True

        # Check each path component
        parts = filepath.split('/')
        for part in parts:
            if fnmatch.fnmatch(part, pattern):
                return True

        return False


def git_check_ignore(repo_path: Path | str, file_path: str) -> bool:
    """Check if a file is ignored by git. Returns True if ignored."""
    result = subprocess.run(
        ["git", "check-ignore", file_path],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def git_ls_files(repo_path: Path | str, *args: str) -> list[str]:
    """Run git ls-files with given arguments. Returns list of paths (or empty list)."""
    result = subprocess.run(
        ["git", "ls-files"] + list(args),
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def git_status_porcelain(repo_path: Path | str) -> str:
    """Get git status output in porcelain format."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout
