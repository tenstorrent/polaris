#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from itertools import product
import ttsim.utils.common as common
import ttsim.utils.prime_factorization as pf

@pytest.mark.unit
def test_util_convert_units():
    dump_golden: bool = False
    V0 = ['T', 'G', 'M', 'K', ' ']
    V1 = ['HZ', 'B', 'FLOPS', 'OPS']
    V2 = [(f1 + t, f2 + t) for f1,f2 in product(V0, V0) for t in V1]
    if dump_golden:
        golden = dict()
    else:
        golden = {
            ('THZ', 'THZ', 1): 1, ('TB', 'TB', 1): 1, ('TFLOPS', 'TFLOPS', 1): 1, ('TOPS', 'TOPS', 1): 1, ('THZ', 'GHZ', 1): 1000,
            ('TB', 'GB', 1): 1024, ('TFLOPS', 'GFLOPS', 1): 1000, ('TOPS', 'GOPS', 1): 1000, ('THZ', 'MHZ', 1): 1000000,
            ('TB', 'MB', 1): 1048576, ('TFLOPS', 'MFLOPS', 1): 1000000, ('TOPS', 'MOPS', 1): 1000000, ('THZ', 'KHZ', 1): 1000000000,
            ('TB', 'KB', 1): 1073741824, ('TFLOPS', 'KFLOPS', 1): 1000000000, ('TOPS', 'KOPS', 1): 1000000000, ('THZ', ' HZ', 1): 1000000000000,
            ('TB', ' B', 1): 1099511627776, ('TFLOPS', ' FLOPS', 1): 1000000000000, ('TOPS', ' OPS', 1): 1000000000000, ('GHZ', 'THZ', 1): 0.001,
            ('GB', 'TB', 1): 0.0009765625, ('GFLOPS', 'TFLOPS', 1): 0.001, ('GOPS', 'TOPS', 1): 0.001, ('GHZ', 'GHZ', 1): 1, ('GB', 'GB', 1): 1,
            ('GFLOPS', 'GFLOPS', 1): 1, ('GOPS', 'GOPS', 1): 1, ('GHZ', 'MHZ', 1): 1000, ('GB', 'MB', 1): 1024, ('GFLOPS', 'MFLOPS', 1): 1000,
            ('GOPS', 'MOPS', 1): 1000, ('GHZ', 'KHZ', 1): 1000000, ('GB', 'KB', 1): 1048576, ('GFLOPS', 'KFLOPS', 1): 1000000,
            ('GOPS', 'KOPS', 1): 1000000, ('GHZ', ' HZ', 1): 1000000000, ('GB', ' B', 1): 1073741824, ('GFLOPS', ' FLOPS', 1): 1000000000,
            ('GOPS', ' OPS', 1): 1000000000, ('MHZ', 'THZ', 1): 1e-06, ('MB', 'TB', 1): 9.5367431640625e-07, ('MFLOPS', 'TFLOPS', 1): 1e-06,
            ('MOPS', 'TOPS', 1): 1e-06, ('MHZ', 'GHZ', 1): 0.001, ('MB', 'GB', 1): 0.0009765625, ('MFLOPS', 'GFLOPS', 1): 0.001,
            ('MOPS', 'GOPS', 1): 0.001, ('MHZ', 'MHZ', 1): 1, ('MB', 'MB', 1): 1, ('MFLOPS', 'MFLOPS', 1): 1, ('MOPS', 'MOPS', 1): 1,
            ('MHZ', 'KHZ', 1): 1000, ('MB', 'KB', 1): 1024, ('MFLOPS', 'KFLOPS', 1): 1000, ('MOPS', 'KOPS', 1): 1000, ('MHZ', ' HZ', 1): 1000000,
            ('MB', ' B', 1): 1048576, ('MFLOPS', ' FLOPS', 1): 1000000, ('MOPS', ' OPS', 1): 1000000, ('KHZ', 'THZ', 1): 1e-09,
            ('KB', 'TB', 1): 9.313225746154785e-10, ('KFLOPS', 'TFLOPS', 1): 1e-09, ('KOPS', 'TOPS', 1): 1e-09, ('KHZ', 'GHZ', 1): 1e-06,
            ('KB', 'GB', 1): 9.5367431640625e-07, ('KFLOPS', 'GFLOPS', 1): 1e-06, ('KOPS', 'GOPS', 1): 1e-06, ('KHZ', 'MHZ', 1): 0.001,
            ('KB', 'MB', 1): 0.0009765625, ('KFLOPS', 'MFLOPS', 1): 0.001, ('KOPS', 'MOPS', 1): 0.001, ('KHZ', 'KHZ', 1): 1, ('KB', 'KB', 1): 1,
            ('KFLOPS', 'KFLOPS', 1): 1, ('KOPS', 'KOPS', 1): 1, ('KHZ', ' HZ', 1): 1000, ('KB', ' B', 1): 1024, ('KFLOPS', ' FLOPS', 1): 1000,
            ('KOPS', ' OPS', 1): 1000, (' HZ', 'THZ', 1): 1e-12, (' B', 'TB', 1): 9.094947017729282e-13, (' FLOPS', 'TFLOPS', 1): 1e-12,
            (' OPS', 'TOPS', 1): 1e-12, (' HZ', 'GHZ', 1): 1e-09, (' B', 'GB', 1): 9.313225746154785e-10, (' FLOPS', 'GFLOPS', 1): 1e-09,
            (' OPS', 'GOPS', 1): 1e-09, (' HZ', 'MHZ', 1): 1e-06, (' B', 'MB', 1): 9.5367431640625e-07, (' FLOPS', 'MFLOPS', 1): 1e-06,
            (' OPS', 'MOPS', 1): 1e-06, (' HZ', 'KHZ', 1): 0.001, (' B', 'KB', 1): 0.0009765625, (' FLOPS', 'KFLOPS', 1): 0.001,
            (' OPS', 'KOPS', 1): 0.001, (' HZ', ' HZ', 1): 1, (' B', ' B', 1): 1, (' FLOPS', ' FLOPS', 1): 1, (' OPS', ' OPS', 1): 1
        }
    for (_f, _t) in V2:
        v = 1
        cv = common.convert_units(v, _f, _t)
        src = (_f, _t, v)
        if dump_golden:
            golden[src] = cv
            continue
        assert cv == golden[src], f'converting {v} {_f} to {_t}: expected {golden[src]} {_t}, actual {cv} {_t}'

    with pytest.raises(AssertionError, match='Z should be one of'):
        common.convert_units(100, 'Z', 'X')


    if dump_golden:
        print(golden)

@pytest.mark.unit
def test_dict2obj():
    input_dict = {'a': 1, 'b': 2, 'c': 3, 'd': {'a_1': 100, 'b_1': 200}}
    output_obj = common.dict2obj(input_dict)
    assert output_obj.a == 1
    assert output_obj.b == 2
    assert output_obj.c == 3
    assert output_obj.d.a_1 == 100
    assert output_obj.d.b_1 == 200


@pytest.mark.unit
def test_str_to_bool():
    assert common.str_to_bool(True)
    assert not common.str_to_bool(False)
    assert not common.str_to_bool(0)
    assert common.str_to_bool(1)
    assert not common.str_to_bool(0.0)
    assert common.str_to_bool(0.5)
    for s in ['true', 't', 'yes', 'y', 'on', 'enable', '1']:
        assert common.str_to_bool(s)
    for s in ['false', 'f', 'no', 'n', 'off', 'disable', '0']:
        assert not common.str_to_bool(s)
    with pytest.raises(ValueError, match='expecting boolean value'):
        common.str_to_bool('invalid-str')

@pytest.mark.unit
def test_ttsim_functional_instance():
    common.get_ttsim_functional_instance('workloads/ResNet@basicresnet.py', '',
                                         {'bs': 1, 'layers': [3,4,6,3],  'num_classes': 1000, 'num_channels': 3})


@pytest.mark.unit
def test_writers(tmp_path_factory):
    odir = tmp_path_factory.mktemp('test-writers')
    os.makedirs(odir, exist_ok=True)
    csvfile = str(odir / 'output1.csv')
    jsonfile = str(odir / 'output2.json')
    print(csvfile, type(csvfile), jsonfile)
    common.print_csv(['id', 'value'],
                     [
                        {'id': '1', 'value': 'value1'},
                        {'id': '2', 'value': 'value2'}
                     ],
                     csvfile)
    common.print_json({
                          'key1': 'value1',
                          'key2': 'value2'
                      },
                      jsonfile
    )

@pytest.mark.unit
@pytest.mark.parametrize("N, check_primes, verbose",[(int(1e5), False, False)])
def test_prime_factorization(N: int, check_primes: bool, verbose: bool) -> None:
    SMALL_PRIMES = pf.sieve_of_eratosthenes(N) if check_primes else []

    for n in range(1,N):
        f = pf.PrimeFactorization(n)

        if verbose:
            print(n, '=', f)

        if check_primes:
            for p,e in f.factors.items():
                assert p in SMALL_PRIMES, f"p={p} is not prime!!"
    return

