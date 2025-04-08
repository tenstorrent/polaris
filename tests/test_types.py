#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import pytest
from itertools import product
import ttsim.utils.types as types


def test_types():
    assert types.socnodetype2str(types.SOCNodeType.CORE) == 'C'
    assert types.str2df('float16') == types.DataFormat.FP16A
    assert types.str2df(['float16', 'float16_b']) == [types.DataFormat.FP16A, types.DataFormat.FP16B]
    assert types.str2mf('lofi') == types.MathFidelity.LoFi
    assert types.get_sim_dtype('BOOL') == types.SimDataType.BOOL
    assert types.get_sim_dtype('wrongtype') == types.SimDataType.UNKNOWN
    assert types.get_bpe(types.SimDataType.INT64) == 8
