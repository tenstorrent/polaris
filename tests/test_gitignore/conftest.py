#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess

import pytest

from tests.test_gitignore.gitignore_helpers import GitignoreSpec


@pytest.fixture
def gitignore_spec(gitignore_content):
    """Build GitignoreSpec from the project's .gitignore content (single read)."""
    return GitignoreSpec(gitignore_content.splitlines())


@pytest.fixture
def git_test_repo(tmp_path, project_root):
    """Create a temporary git repository with the project's .gitignore."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    gitignore_src = project_root / ".gitignore"
    gitignore_dst = repo / ".gitignore"
    shutil.copy(gitignore_src, gitignore_dst)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"],
                   cwd=repo, check=True, capture_output=True)
    return repo


@pytest.fixture
def gitignore_content(project_root):
    """
    Fixture providing the content of the project's .gitignore file.

    Returns the .gitignore content as a string for pattern testing.
    """
    gitignore_path = project_root / ".gitignore"
    with open(gitignore_path, "r", encoding="utf-8") as f:
        return f.read()
