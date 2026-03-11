#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Edge case tests for .gitignore behavior.

Tests complex scenarios including:
- Deeply nested structures
- Symlinks to __init__.py and __main__.py
- Mixed scenarios with tracked and ignored files
- Real project structure scenarios
"""

import os
import subprocess

import pytest

from tests.test_gitignore.gitignore_helpers import git_check_ignore, git_ls_files


@pytest.mark.integration
class TestDeeplyNestedStructures:
    """Test __init__.py and __main__.py in deeply nested structures."""

    def test_deeply_nested_init_files(self, git_test_repo):
        """Test __init__.py files in very deep directory hierarchies."""
        # Create a deep package structure
        deep_path = git_test_repo / "a" / "b" / "c" / "d" / "e" / "f"
        deep_path.mkdir(parents=True)

        # Create __init__.py at each level
        init_files = []
        current = git_test_repo / "a"
        for level in ["b", "c", "d", "e", "f"]:
            init_file = current / "__init__.py"
            init_file.write_text(f"# Level: {current}\n")
            init_files.append(init_file.relative_to(git_test_repo))
            current = current / level
            current.mkdir(exist_ok=True)

        # Final level
        final_init = deep_path / "__init__.py"
        final_init.write_text("# Final level\n")
        init_files.append(final_init.relative_to(git_test_repo))

        # All __init__.py files should be tracked
        for init_file in init_files:
            assert not git_check_ignore(git_test_repo, str(init_file)), \
                f"{init_file} should not be ignored"

        # Add all and verify
        subprocess.run(["git", "add", "."], cwd=git_test_repo, check=True)
        staged = git_ls_files(git_test_repo, "--cached")

        for init_file in init_files:
            assert str(init_file) in staged, f"{init_file} should be staged"

    def test_main_py_at_various_depths(self, git_test_repo):
        """Test __main__.py at root and various nesting levels."""
        # Root level
        (git_test_repo / "__main__.py").write_text("# Root entry\n")

        # Package level
        pkg = git_test_repo / "myapp"
        pkg.mkdir()
        (pkg / "__main__.py").write_text("# Package entry\n")

        # Subpackage level
        subpkg = pkg / "commands"
        subpkg.mkdir()
        (subpkg / "__main__.py").write_text("# Subpackage entry\n")

        # All should be tracked
        assert not git_check_ignore(git_test_repo, "__main__.py")
        assert not git_check_ignore(git_test_repo, "myapp/__main__.py")
        assert not git_check_ignore(git_test_repo, "myapp/commands/__main__.py")

    def test_mixed_tracked_and_ignored_files(self, git_test_repo):
        """Test directory with mix of tracked and ignored files."""
        pkg = git_test_repo / "mypackage"
        pkg.mkdir()

        # Create various files
        (pkg / "__init__.py").write_text("")  # Tracked
        (pkg / "module.py").write_text("")  # Tracked
        (pkg / "__temp.py").write_text("")  # Ignored
        (pkg / "__cache.dat").write_text("")  # Ignored

        pycache = pkg / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_text("")  # Ignored

        # Check each file
        assert not git_check_ignore(git_test_repo, "mypackage/__init__.py")
        assert not git_check_ignore(git_test_repo, "mypackage/module.py")
        assert git_check_ignore(git_test_repo, "mypackage/__temp.py")
        assert git_check_ignore(git_test_repo, "mypackage/__cache.dat")
        assert git_check_ignore(git_test_repo, "mypackage/__pycache__")

        # Add all and verify only tracked files are added
        subprocess.run(["git", "add", "mypackage"], cwd=git_test_repo, check=True)
        staged = git_ls_files(git_test_repo, "--cached")

        assert "mypackage/__init__.py" in staged
        assert "mypackage/module.py" in staged
        assert not any("__temp.py" in f for f in staged)
        assert not any("__cache.dat" in f for f in staged)
        assert not any("__pycache__" in f for f in staged)


@pytest.mark.integration
class TestSymlinkScenarios:
    """Test behavior with symlinks to __init__.py and __main__.py."""

    @pytest.mark.skipif(os.name == 'nt', reason="Symlinks may require admin on Windows")
    def test_symlink_to_init_py(self, git_test_repo):
        """Test symlink to __init__.py file."""
        # Create original file
        src_dir = git_test_repo / "src"
        src_dir.mkdir()
        orig_init = src_dir / "__init__.py"
        orig_init.write_text("# Original init\n")

        # Create symlink
        link_dir = git_test_repo / "link"
        link_dir.mkdir()
        link_init = link_dir / "__init__.py"
        link_init.symlink_to(orig_init)

        # Both should be tracked
        assert not git_check_ignore(git_test_repo, "src/__init__.py")
        # Git follows symlinks in some cases, but check-ignore checks the link itself
        assert not git_check_ignore(git_test_repo, "link/__init__.py")

    @pytest.mark.skipif(os.name == 'nt', reason="Symlinks may require admin on Windows")
    def test_symlink_to_main_py(self, git_test_repo):
        """Test symlink to __main__.py file."""
        # Create original file
        orig_main = git_test_repo / "original_main.py"
        orig_main.write_text("# Original main\n")

        # Create symlink named __main__.py
        link_main = git_test_repo / "__main__.py"
        link_main.symlink_to(orig_main)

        # Both should be tracked
        assert not git_check_ignore(git_test_repo, "original_main.py")
        assert not git_check_ignore(git_test_repo, "__main__.py")


@pytest.mark.integration
class TestRealProjectStructures:
    """Test scenarios matching real project structures."""

    def test_ttsim_package_structure(self, git_test_repo):
        """Simulate the ttsim/ package structure from the project."""
        # Create ttsim package with subpackages
        ttsim = git_test_repo / "ttsim"
        ttsim.mkdir()
        (ttsim / "__init__.py").write_text("")

        # Create subpackages
        for subpkg in ["front", "back", "ops", "utils", "stats"]:
            subdir = ttsim / subpkg
            subdir.mkdir()
            (subdir / "__init__.py").write_text("")

            # Add a regular module
            (subdir / f"{subpkg}_module.py").write_text("")

            # Add __pycache__
            pycache = subdir / "__pycache__"
            pycache.mkdir()
            (pycache / f"{subpkg}_module.pyc").write_text("")

        # Add all
        subprocess.run(["git", "add", "."], cwd=git_test_repo, check=True)
        staged = git_ls_files(git_test_repo, "--cached")

        # All __init__.py should be tracked
        assert "ttsim/__init__.py" in staged
        for subpkg in ["front", "back", "ops", "utils", "stats"]:
            assert f"ttsim/{subpkg}/__init__.py" in staged
            assert f"ttsim/{subpkg}/{subpkg}_module.py" in staged
            # __pycache__ should not be tracked
            assert not any("__pycache__" in f for f in staged)

    def test_tests_helpers_structure(self, git_test_repo):
        """Simulate the tests/helpers/ structure from the project."""
        # tests/ directory (no __init__.py)
        tests = git_test_repo / "tests"
        tests.mkdir()

        # tests/helpers/ package (has __init__.py)
        helpers = tests / "helpers"
        helpers.mkdir()
        (helpers / "__init__.py").write_text("# Test helpers package\n")
        (helpers / "lfc_helper.py").write_text("# LFC helper\n")

        # tests/ direct test files
        (tests / "test_polaris.py").write_text("")
        (tests / "conftest.py").write_text("")

        # Add __pycache__ in tests
        pycache = tests / "__pycache__"
        pycache.mkdir()
        (pycache / "conftest.pyc").write_text("")

        # Add all
        subprocess.run(["git", "add", "."], cwd=git_test_repo, check=True)
        staged = git_ls_files(git_test_repo, "--cached")

        # tests/helpers/__init__.py should be tracked
        assert "tests/helpers/__init__.py" in staged
        assert "tests/helpers/lfc_helper.py" in staged

        # tests/ files should be tracked
        assert "tests/test_polaris.py" in staged
        assert "tests/conftest.py" in staged

        # __pycache__ should not be tracked
        assert not any("__pycache__" in f for f in staged)

    def test_htmlcov_and_ci_directories_ignored(self, git_test_repo):
        """Test that __htmlcov and __ci directories are ignored as expected."""
        # These are common output directories in the project
        htmlcov = git_test_repo / "__htmlcov"
        htmlcov.mkdir()
        (htmlcov / "index.html").write_text("<html></html>")
        (htmlcov / "coverage.json").write_text("{}")

        ci = git_test_repo / "__ci"
        ci.mkdir()
        (ci / "results.json").write_text("{}")

        # Both directories should be ignored
        assert git_check_ignore(git_test_repo, "__htmlcov")
        assert git_check_ignore(git_test_repo, "__ci")

        # Contents should not appear in status
        subprocess.run(["git", "add", "."], cwd=git_test_repo, check=True)
        staged = git_ls_files(git_test_repo, "--cached")

        assert not any("__htmlcov" in f for f in staged)
        assert not any("__ci" in f for f in staged)


@pytest.mark.unit
class TestPatternDocumentation:
    """Test to document the .gitignore pattern behavior."""

    def test_gitignore_has_required_patterns(self, gitignore_content):
        """Verify .gitignore contains the expected patterns."""
        content = gitignore_content
        assert "__*/" in content, "Should have pattern to ignore __ directories"
        assert "__*" in content, "Should have pattern to ignore __ files"
        assert "!__init__.py" in content, "Should have exception for __init__.py"
        assert "!__main__.py" in content, "Should have exception for __main__.py"
        lines = content.split("\n")
        dunder_wildcard_line = next(i for i, line in enumerate(lines) if line.strip() == "__*")
        init_exception_line = next(i for i, line in enumerate(lines) if line.strip() == "!__init__.py")
        main_exception_line = next(i for i, line in enumerate(lines) if line.strip() == "!__main__.py")
        assert init_exception_line > dunder_wildcard_line, \
            "!__init__.py exception must come after __* pattern"
        assert main_exception_line > dunder_wildcard_line, \
            "!__main__.py exception must come after __* pattern"

    def test_gitignore_pattern_comments(self, gitignore_content):
        """Verify .gitignore has at least one comment in the block with the exception patterns."""
        lines = gitignore_content.split("\n")
        dunder_wildcard_line = next(i for i, line in enumerate(lines) if line.strip() == "__*")
        main_exception_line = next(i for i, line in enumerate(lines) if line.strip() == "!__main__.py")
        block = lines[dunder_wildcard_line : main_exception_line + 1]
        has_comment = any(line.strip().startswith("#") for line in block)
        assert has_comment, "Should have a comment in the block containing __* and exception patterns"
