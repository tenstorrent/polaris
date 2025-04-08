#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import pytest

@pytest.fixture(scope="session")
def session_temp_directory(tmp_path_factory):
    dname = tmp_path_factory.mktemp("outputs")
    return dname