#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration for Deformable DETR reference tests.
"""
import pytest
from workloads.Deformable_DETR.reference.tests.test_matcher_comparision import TestReport


@pytest.fixture
def report():
    """Provide a TestReport instance for matcher comparison tests."""
    return TestReport()
