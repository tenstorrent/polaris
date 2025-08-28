#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttsim.config.wl2archmap import get_wlmapspec_from_yaml

# Initialize the WL2ArchTypeSpec singleton for all tests
@pytest.fixture(scope="session", autouse=True)
def initialize_singleton():
    """Initialize the WL2ArchTypeSpec singleton for all tests."""
    try:
        get_wlmapspec_from_yaml('config/wl2archmapping.yaml')
    except Exception as e:
        # If config file is not found, skip initialization
        # This allows tests to run even if config is missing
        pass

@pytest.fixture(scope="session")
def session_temp_directory(tmp_path_factory):
    dname = tmp_path_factory.mktemp("outputs")
    return dname
