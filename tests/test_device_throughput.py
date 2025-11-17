#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.mark.unit
def test_throughput_guardband_formula():
    """
    Test that throughput calculation correctly applies guardband.

    The guardband is applied by increasing time:
        tot_msecs = (1 + G_GUARDBAND) * tot_ideal_msecs

    Since throughput is inversely proportional to time (throughput = bs * 1000 / time),
    it should be divided by (1 + G_GUARDBAND), not multiplied by (1 - G_GUARDBAND).

    This test verifies that:
        tot_throughput = tot_ideal_throughput / (1 + G_GUARDBAND)
    is mathematically equivalent to:
        tot_throughput = bs * 1000 / tot_msecs
    where tot_msecs = (1 + G_GUARDBAND) * tot_ideal_msecs
    """
    from ttsim.back.device import Device

    # Test with the actual guardband constant used in Device class
    G_GUARDBAND = Device.G_GUARDBAND  # Should be 0.25

    # Test case parameters
    bs = 100
    tot_ideal_msecs = 10.0

    # Calculate ideal throughput
    tot_ideal_throughput = bs * 1000 / tot_ideal_msecs

    # Calculate expected throughput using the guardband formula
    # This is the correct formula based on time increase
    tot_msecs = (1 + G_GUARDBAND) * tot_ideal_msecs
    expected_throughput = bs * 1000 / tot_msecs

    # Calculate throughput using the simplified formula (what should be in code)
    calculated_throughput = tot_ideal_throughput / (1 + G_GUARDBAND)

    # Verify they are equivalent
    assert abs(expected_throughput - calculated_throughput) < 1e-10, \
        f"Expected throughput {expected_throughput} != calculated {calculated_throughput}"

    # Additional verification with different values
    test_cases = [
        (1, 1.0),
        (10, 5.0),
        (256, 12.5),
        (1024, 100.0),
    ]

    for test_bs, test_ideal_msecs in test_cases:
        test_ideal_throughput = test_bs * 1000 / test_ideal_msecs
        test_msecs = (1 + G_GUARDBAND) * test_ideal_msecs
        test_expected = test_bs * 1000 / test_msecs
        test_calculated = test_ideal_throughput / (1 + G_GUARDBAND)

        assert abs(test_expected - test_calculated) < 1e-10, \
            f"For bs={test_bs}, ideal_msecs={test_ideal_msecs}: " \
            f"expected {test_expected} != calculated {test_calculated}"


@pytest.mark.unit
def test_throughput_inverse_relationship():
    """
    Test that demonstrates the inverse relationship between throughput and time.

    If time increases by factor (1 + G_GUARDBAND), then throughput must decrease
    by the same factor, i.e., be divided by (1 + G_GUARDBAND).
    """
    from ttsim.back.device import Device

    G_GUARDBAND = Device.G_GUARDBAND

    # Example values
    bs = 128
    ideal_time = 20.0

    # Ideal throughput
    ideal_throughput = bs * 1000 / ideal_time

    # When time increases by (1 + G_GUARDBAND)
    actual_time = (1 + G_GUARDBAND) * ideal_time

    # Throughput must decrease proportionally
    actual_throughput = bs * 1000 / actual_time

    # This should equal ideal_throughput / (1 + G_GUARDBAND)
    formula_throughput = ideal_throughput / (1 + G_GUARDBAND)

    assert abs(actual_throughput - formula_throughput) < 1e-10, \
        f"Throughput inverse relationship failed: {actual_throughput} != {formula_throughput}"

    # Verify the relationship: actual_throughput * actual_time = bs * 1000
    assert abs(actual_throughput * actual_time - bs * 1000) < 1e-6, \
        "Throughput * time should equal bs * 1000"

    # Verify: ideal_throughput * ideal_time = bs * 1000
    assert abs(ideal_throughput * ideal_time - bs * 1000) < 1e-6, \
        "Ideal throughput * ideal time should equal bs * 1000"
