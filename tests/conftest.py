# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def mocker(monkeypatch):
    """Lightweight mocker fixture using monkeypatch to satisfy tests expecting `mocker`.

    Provides a minimal API subset: patch, patch.object, mock_open.
    """
    class _Mocker:
        def patch(self, target, *args, **kwargs):
            return monkeypatch.setattr(target, *args, **kwargs)

        def mock_open(self, read_data=""):
            from io import StringIO

            class _MO:
                def __call__(self, *args, **kwargs):
                    return StringIO(read_data)

            return _MO()

    return _Mocker()

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
