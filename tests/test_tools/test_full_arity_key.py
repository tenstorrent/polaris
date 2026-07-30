# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-arity LUT key tests (schema v5, issue #478).

The master key now covers ALL populated inputs (input_3..input_7), so >3-input ops
(PagedFusedUpdateCache, ScaledDotProductAttention) key on every operand. Arity 1/2/3 keys
stay byte-identical to the historical 9/16/23-tuples.
"""
import pytest

from tools.perf_lookup.tt_perf_master_schema import (
    HALO_KEY_TUPLE_YAML_KEYS,
    ITS_KEY_TUPLE_YAML_KEYS,
    MAX_KEY_INPUTS,
    STANDARD_KEY_TUPLE_LENGTHS,
    tuple_to_labeled_key_map,
    yaml_labeled_key_to_tuple,
)


def _synth_key(n_inputs: int) -> tuple:
    """A synthetic standard-op key tuple with n_inputs operands (length 9 + 7*(n-1))."""
    key = ["someop", 1, 1, 32, 128, "TILE", "BFLOAT16", "DEV_0_L1_INTERLEAVED", "N/A"]
    for _ in range(1, n_inputs):
        key += [1, 1, 32, 128, "TILE", "BFLOAT8_B", "DEV_0_DRAM_INTERLEAVED"]
    return tuple(key)


@pytest.mark.unit
def test_standard_key_lengths():
    assert sorted(STANDARD_KEY_TUPLE_LENGTHS) == [9, 16, 23, 30, 37, 44, 51, 58]
    assert MAX_KEY_INPUTS == 8


@pytest.mark.unit
@pytest.mark.parametrize("n_inputs", [1, 2, 3, 4, 5, 6, 7, 8])
def test_key_roundtrip_all_arities(n_inputs):
    """tuple -> labeled dict -> tuple round-trips for every arity, including >3."""
    kt = _synth_key(n_inputs)
    assert len(kt) == 9 + 7 * (n_inputs - 1)
    labeled = tuple_to_labeled_key_map(kt)
    back = yaml_labeled_key_to_tuple(labeled)
    assert back == kt


@pytest.mark.unit
def test_noncontiguous_input_slots_rejected():
    """A key with input_2 but no input_1 (a gap) is rejected on parse-back."""
    kt = _synth_key(3)
    labeled = tuple_to_labeled_key_map(kt)
    # drop all input_1_* fields -> input_0 + input_2 with a gap
    gapped = {k: v for k, v in labeled.items() if not str(k).startswith("input_1_")}
    with pytest.raises(ValueError, match="contiguous"):
        yaml_labeled_key_to_tuple(gapped)


@pytest.mark.unit
def test_generic_builder_matches_fixed_builders_for_small_arity():
    """build_master_key_tuple_n == build_master_key_tuple_8/15/22 for arity 1/2/3."""
    import ttsim.front.ttnn as ttnn
    from ttsim.front.ttnn.device import close_device, open_device, set_default_device
    from tools.perf_lookup.lookup_operator_perf import (
        build_master_key_tuple_8,
        build_master_key_tuple_15,
        build_master_key_tuple_22,
        build_master_key_tuple_n,
    )

    dev = open_device()
    set_default_device(dev)
    try:
        t0 = ttnn.zeros([1, 1, 32, 128], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        t1 = ttnn.zeros([1, 1, 128, 256], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
        t2 = ttnn.zeros([1, 1, 256, 512], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)

        class _Op:
            optype = "MatMul"
            attrs: dict = {}
            precision = None
            inList: list = []
            name = "op"

        op = _Op()
        assert build_master_key_tuple_n(op, [t0]) == build_master_key_tuple_8(op, t0)
        assert build_master_key_tuple_n(op, [t0, t1]) == build_master_key_tuple_15(op, t0, t1)
        assert build_master_key_tuple_n(op, [t0, t1, t2]) == build_master_key_tuple_22(op, t0, t1, t2)
        # arity 5 -> 37-tuple
        five = build_master_key_tuple_n(op, [t0, t1, t2, t1, t2])
        assert len(five) == 37
    finally:
        close_device(dev)


@pytest.mark.unit
def test_halo_v4_key_length_accepted():
    """halo v4 keys (16 = v3 + is_transpose) must survive yaml_labeled_key_to_tuple.

    16 is accepted only because it coincides with the 2-input standard length — there is no
    explicit halo-v4 entry in the accepted set (see yaml_labeled_key_to_tuple). This pins the
    coincidence so a change to MAX_KEY_INPUTS or the standard-length formula cannot silently
    start rejecting every halo v4 key.
    """
    assert len(HALO_KEY_TUPLE_YAML_KEYS) == 16, 'halo v4 key is the 9 std fields + halo extras + is_transpose'
    assert len(ITS_KEY_TUPLE_YAML_KEYS) == 10
    assert 16 in STANDARD_KEY_TUPLE_LENGTHS, 'halo v4 acceptance rides on 16 being a standard length'

    halo_key = dict(zip(HALO_KEY_TUPLE_YAML_KEYS, _synth_key(2)))
    assert len(yaml_labeled_key_to_tuple(halo_key)) == 16
