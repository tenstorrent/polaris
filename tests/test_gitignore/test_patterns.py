#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for .gitignore pattern matching logic.

Tests verify that the .gitignore patterns correctly:
- Ignore directories matching __*/ pattern
- Ignore files matching __* pattern
- Explicitly allow __init__.py and __main__.py files
"""

import pytest


@pytest.mark.unit
class TestGitignorePythonFileExceptions:
    """Test that __init__.py and __main__.py are explicitly allowed."""

    def test_init_py_not_ignored(self, gitignore_spec):
        """__init__.py should NOT be ignored."""
        assert not gitignore_spec.match_file("__init__.py")
        assert not gitignore_spec.match_file("ttsim/__init__.py")
        assert not gitignore_spec.match_file("tests/helpers/__init__.py")

    def test_main_py_not_ignored(self, gitignore_spec):
        """__main__.py should NOT be ignored."""
        assert not gitignore_spec.match_file("__main__.py")
        assert not gitignore_spec.match_file("ttsim/__main__.py")
        assert not gitignore_spec.match_file("tools/parser/__main__.py")

    def test_nested_init_py_not_ignored(self, gitignore_spec):
        """__init__.py in deeply nested directories should NOT be ignored."""
        assert not gitignore_spec.match_file("a/b/c/d/__init__.py")
        assert not gitignore_spec.match_file("ttsim/front/llk/__init__.py")

    def test_nested_main_py_not_ignored(self, gitignore_spec):
        """__main__.py in deeply nested directories should NOT be ignored."""
        assert not gitignore_spec.match_file("a/b/c/d/__main__.py")
        assert not gitignore_spec.match_file("workloads/ttnn/__main__.py")


@pytest.mark.unit
class TestGitignoreTemporaryFilePatterns:
    """Test that temporary files and directories starting with __ are ignored."""

    def test_pycache_directory_ignored(self, gitignore_spec):
        """__pycache__/ directories should be ignored."""
        assert gitignore_spec.match_file("__pycache__/")
        assert gitignore_spec.match_file("ttsim/__pycache__/")
        assert gitignore_spec.match_file("tests/__pycache__/some_file.pyc")

    def test_dunder_directories_ignored(self, gitignore_spec):
        """Directories starting with __ should be ignored."""
        assert gitignore_spec.match_file("__temp/")
        assert gitignore_spec.match_file("__output/")
        assert gitignore_spec.match_file("__htmlcov/")
        assert gitignore_spec.match_file("__ci/")

    def test_dunder_files_ignored(self, gitignore_spec):
        """Files starting with __ (except __init__.py and __main__.py) should be ignored."""
        assert gitignore_spec.match_file("__temp.txt")
        assert gitignore_spec.match_file("__output.log")
        assert gitignore_spec.match_file("__cache.dat")
        assert gitignore_spec.match_file("ttsim/__temp_file.txt")

    def test_dunder_files_in_subdirs_ignored(self, gitignore_spec):
        """Files starting with __ in subdirectories should be ignored."""
        assert gitignore_spec.match_file("config/__temp.yaml")
        assert gitignore_spec.match_file("tools/__output.json")


@pytest.mark.unit
class TestGitignoreEdgeCases:
    """Test edge cases for the gitignore patterns."""

    def test_init_py_variations_ignored(self, gitignore_spec):
        """Files similar to __init__.py but not exact should be ignored."""
        # These should be ignored (match __* pattern, not exception)
        assert gitignore_spec.match_file("__init__.pyc")
        assert gitignore_spec.match_file("__init__.pyo")
        assert gitignore_spec.match_file("__init__.py.bak")
        assert gitignore_spec.match_file("__init__.py~")
        assert gitignore_spec.match_file("__initpy")

    def test_main_py_variations_ignored(self, gitignore_spec):
        """Files similar to __main__.py but not exact should be ignored."""
        # These should be ignored (match __* pattern, not exception)
        assert gitignore_spec.match_file("__main__.pyc")
        assert gitignore_spec.match_file("__main__.pyo")
        assert gitignore_spec.match_file("__main__.py.bak")
        assert gitignore_spec.match_file("__mainpy")

    def test_files_with_embedded_dunder_not_ignored(self, gitignore_spec):
        """Files with __ in the middle should NOT be ignored."""
        assert not gitignore_spec.match_file("my__file.py")
        assert not gitignore_spec.match_file("test__init__.py")
        assert not gitignore_spec.match_file("module__main__.py")

    def test_pypackages_directory_ignored(self, gitignore_spec):
        """__pypackages__/ (PEP 582) should be ignored."""
        assert gitignore_spec.match_file("__pypackages__/")
        assert gitignore_spec.match_file("__pypackages__/lib/")

    def test_regular_python_files_not_ignored(self, gitignore_spec):
        """Regular Python files should NOT be ignored."""
        assert not gitignore_spec.match_file("polaris.py")
        assert not gitignore_spec.match_file("ttsim/utils/common.py")
        assert not gitignore_spec.match_file("tests/test_polaris.py")


@pytest.mark.unit
class TestGitignorePatternPrecedence:
    """Test that pattern precedence works correctly."""

    def test_exception_overrides_wildcard(self, gitignore_spec):
        """Exception patterns (!) should override earlier wildcard patterns."""
        # __* pattern would match __init__.py, but !__init__.py overrides it
        assert not gitignore_spec.match_file("__init__.py")
        # __* pattern would match __main__.py, but !__main__.py overrides it
        assert not gitignore_spec.match_file("__main__.py")
        # __* pattern matches other __ files
        assert gitignore_spec.match_file("__other.py")

    def test_directory_vs_file_patterns(self, gitignore_spec):
        """Test distinction between directory and file patterns."""
        # __*/ matches directories
        assert gitignore_spec.match_file("__temp/")
        # __* matches both files and directories
        assert gitignore_spec.match_file("__temp")
        assert gitignore_spec.match_file("__temp.txt")

    def test_exception_in_ignored_directory(self, gitignore_spec):
        """Document GitignoreSpec behavior for exception paths inside ignored directories."""
        # The directory __temp/ itself is ignored
        assert gitignore_spec.match_file("__temp/")

        # GitignoreSpec does not model "parent ignored => don't descend". It still
        # applies negation to any path, so __init__.py inside __temp/ is not ignored
        # by the spec. Real git would ignore it (git doesn't descend into ignored dirs).
        assert not gitignore_spec.match_file("__temp/__init__.py")
