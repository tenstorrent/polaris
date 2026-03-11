#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for .gitignore behavior with actual git commands.

Tests verify that git correctly tracks and ignores files according to the
.gitignore patterns, specifically testing the exception rules for __init__.py
and __main__.py files.
"""

import subprocess

import pytest

from tests.test_gitignore.gitignore_helpers import git_check_ignore, git_ls_files, git_status_porcelain


@pytest.mark.integration
class TestGitignoreWithGitCommands:
    """Test .gitignore behavior using actual git commands."""

    def test_init_py_tracked_by_git(self, git_test_repo):
        """__init__.py files should be tracked by git."""
        # Create __init__.py in various locations
        (git_test_repo / "ttsim").mkdir()
        init_file = git_test_repo / "ttsim" / "__init__.py"
        init_file.write_text("# Package init\n")

        # Check git status - most important check is that it's not ignored
        assert not git_check_ignore(git_test_repo, "ttsim/__init__.py")

        # File should appear in git status (as part of untracked directory or individual file)
        status = git_status_porcelain(git_test_repo)
        # Git may show either "?? ttsim/" or "?? ttsim/__init__.py" depending on state
        assert "ttsim" in status, "ttsim directory or files should appear in git status"

    def test_main_py_tracked_by_git(self, git_test_repo):
        """__main__.py files should be tracked by git."""
        # Create __main__.py
        main_file = git_test_repo / "__main__.py"
        main_file.write_text("# Entry point\n")

        # Check git status
        assert not git_check_ignore(git_test_repo, "__main__.py")

        # File should appear in git status as untracked
        status = git_status_porcelain(git_test_repo)
        assert "__main__.py" in status

    def test_pycache_ignored_by_git(self, git_test_repo):
        """__pycache__/ directories should be ignored by git."""
        # Create __pycache__ directory
        pycache = git_test_repo / "__pycache__"
        pycache.mkdir()
        (pycache / "module.cpython-39.pyc").write_text("compiled")

        # Should be ignored
        assert git_check_ignore(git_test_repo, "__pycache__")

        # Should not appear in git status
        status = git_status_porcelain(git_test_repo)
        assert "__pycache__" not in status

    def test_dunder_temp_files_ignored(self, git_test_repo):
        """Files starting with __ should be ignored (except __init__.py and __main__.py)."""
        # Create various __ files
        (git_test_repo / "__temp.txt").write_text("temp")
        (git_test_repo / "__output.log").write_text("output")
        (git_test_repo / "__cache.dat").write_text("cache")

        # All should be ignored
        assert git_check_ignore(git_test_repo, "__temp.txt")
        assert git_check_ignore(git_test_repo, "__output.log")
        assert git_check_ignore(git_test_repo, "__cache.dat")

        # Should not appear in git status
        status = git_status_porcelain(git_test_repo)
        assert "__temp.txt" not in status
        assert "__output.log" not in status
        assert "__cache.dat" not in status

    def test_dunder_directories_ignored(self, git_test_repo):
        """Directories starting with __ should be ignored."""
        # Create various __ directories
        (git_test_repo / "__htmlcov").mkdir()
        (git_test_repo / "__htmlcov" / "index.html").write_text("<html></html>")

        (git_test_repo / "__temp").mkdir()
        (git_test_repo / "__temp" / "data.txt").write_text("data")

        # Directories should be ignored
        assert git_check_ignore(git_test_repo, "__htmlcov")
        assert git_check_ignore(git_test_repo, "__temp")

        # Should not appear in git status
        status = git_status_porcelain(git_test_repo)
        assert "__htmlcov" not in status
        assert "__temp" not in status

    def test_git_add_all_respects_gitignore(self, git_test_repo):
        """git add . should only add non-ignored files."""
        # Create a mix of files
        (git_test_repo / "module.py").write_text("# Regular module\n")
        (git_test_repo / "__init__.py").write_text("# Package init\n")
        (git_test_repo / "__temp.txt").write_text("temp")

        pycache = git_test_repo / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_text("compiled")

        # Add all files
        subprocess.run(["git", "add", "."], cwd=git_test_repo, check=True)

        # Check what was staged
        staged_files = git_ls_files(git_test_repo, "--cached")

        # Should include .gitignore, module.py, __init__.py
        assert ".gitignore" in staged_files
        assert "module.py" in staged_files
        assert "__init__.py" in staged_files

        # Should NOT include __temp.txt or __pycache__ contents
        assert not any("__temp.txt" in f for f in staged_files)
        assert not any("__pycache__" in f for f in staged_files)

    def test_nested_package_structure(self, git_test_repo):
        """Test __init__.py in nested package structures."""
        # Create nested package structure
        pkg = git_test_repo / "ttsim" / "utils" / "helpers"
        pkg.mkdir(parents=True)

        # Create __init__.py at each level
        (git_test_repo / "ttsim" / "__init__.py").write_text("")
        (git_test_repo / "ttsim" / "utils" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")

        # All should be tracked
        assert not git_check_ignore(git_test_repo, "ttsim/__init__.py")
        assert not git_check_ignore(git_test_repo, "ttsim/utils/__init__.py")
        assert not git_check_ignore(git_test_repo, "ttsim/utils/helpers/__init__.py")

        # Add and verify
        subprocess.run(["git", "add", "."], cwd=git_test_repo, check=True)
        staged_files = git_ls_files(git_test_repo, "--cached")

        assert "ttsim/__init__.py" in staged_files
        assert "ttsim/utils/__init__.py" in staged_files
        assert "ttsim/utils/helpers/__init__.py" in staged_files


@pytest.mark.integration
class TestGitignoreEdgeCasesWithGit:
    """Test edge cases with actual git behavior."""

    def test_init_py_variations_with_git(self, git_test_repo):
        """Test that variations of __init__.py are properly ignored."""
        # Create variations
        (git_test_repo / "__init__.py").write_text("")  # Should be tracked
        (git_test_repo / "__init__.pyc").write_text("")  # Should be ignored
        (git_test_repo / "__init__.py.bak").write_text("")  # Should be ignored

        # Check ignore status
        assert not git_check_ignore(git_test_repo, "__init__.py")
        assert git_check_ignore(git_test_repo, "__init__.pyc")
        assert git_check_ignore(git_test_repo, "__init__.py.bak")

    def test_main_py_variations_with_git(self, git_test_repo):
        """Test that variations of __main__.py are properly ignored."""
        # Create variations
        (git_test_repo / "__main__.py").write_text("")  # Should be tracked
        (git_test_repo / "__main__.pyc").write_text("")  # Should be ignored
        (git_test_repo / "__main__.py~").write_text("")  # Should be ignored

        # Check ignore status
        assert not git_check_ignore(git_test_repo, "__main__.py")
        assert git_check_ignore(git_test_repo, "__main__.pyc")
        assert git_check_ignore(git_test_repo, "__main__.py~")

    def test_init_in_ignored_directory(self, git_test_repo):
        """Test that __init__.py inside an ignored directory is still ignored."""
        # Create ignored directory
        temp_dir = git_test_repo / "__temp_output"
        temp_dir.mkdir()
        (temp_dir / "__init__.py").write_text("")
        (temp_dir / "data.txt").write_text("")

        # The directory itself is ignored
        assert git_check_ignore(git_test_repo, "__temp_output")

        # Files inside ignored directory are also ignored
        # (git doesn't descend into ignored directories)
        assert git_check_ignore(git_test_repo, "__temp_output/__init__.py")

    def test_regular_files_not_affected(self, git_test_repo):
        """Test that regular Python files are tracked normally."""
        # Create regular files
        (git_test_repo / "polaris.py").write_text("# Main script\n")
        (git_test_repo / "config.py").write_text("# Config\n")

        # Should not be ignored
        assert not git_check_ignore(git_test_repo, "polaris.py")
        assert not git_check_ignore(git_test_repo, "config.py")

        # Should appear in status
        status = git_status_porcelain(git_test_repo)
        assert "polaris.py" in status
        assert "config.py" in status

    def test_files_with_embedded_dunder_tracked(self, git_test_repo):
        """Files with __ in the middle (not at start) should be tracked."""
        # Create files with __ in middle
        (git_test_repo / "my__file.py").write_text("")
        (git_test_repo / "test__utils.py").write_text("")

        # Should not be ignored
        assert not git_check_ignore(git_test_repo, "my__file.py")
        assert not git_check_ignore(git_test_repo, "test__utils.py")


@pytest.mark.integration
def test_gitignore_spec_matches_git_for_sample_paths(git_test_repo, gitignore_spec):
    """GitignoreSpec should agree with git check-ignore for a small set of paths."""
    sample_paths = [
        "__init__.py",
        "__main__.py",
        "ttsim/__init__.py",
        "__pycache__/module.pyc",
        "__temp.txt",
        "__htmlcov/index.html",
    ]
    for path in sample_paths:
        git_ignored = git_check_ignore(git_test_repo, path)
        spec_ignored = gitignore_spec.match_file(path)
        assert spec_ignored == git_ignored, (
            f"GitignoreSpec disagrees with git for {path!r}: "
            f"spec={spec_ignored}, git={git_ignored}"
        )
