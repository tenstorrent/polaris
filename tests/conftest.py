#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def session_temp_directory(tmp_path_factory):
    dname = tmp_path_factory.mktemp("outputs")
    return dname


@pytest.fixture
def project_root():
    """Return the project (repository) root as a Path."""
    return Path(__file__).resolve().parent.parent

