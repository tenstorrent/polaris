#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
from ttsim.front.ttnn.device import Device, ARCH, open_device, close_device

@pytest.fixture(scope="session")
def device(tmp_path_factory):
    device = open_device(device_id=0)
    device.architecture = ARCH.WORMHOLE_B0
    yield device

    close_device(device)
