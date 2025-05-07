#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from ttsim.utils.common import parse_csv, parse_xlsx
from ttsim.config import get_arspec_from_yaml

# 'fp64' uses vector clk in Nvidia, but we use matrix clk
# which creates a small delta, we ignore that for now!!
FLOAT_PRECISIONS = ['bf16', 'fp16', 'tf32', 'fp32']
INT_PRECISIONS   = ['int4', 'int8', 'int32']
PRECISIONS       = INT_PRECISIONS + FLOAT_PRECISIONS

def fmt(x):
    return f"{x:.2f}" if isinstance(x, (float, int)) else x

def check_uniq_keys_in_list(L):
    TBL = {}
    for p,v in L:
        if p not in TBL: TBL[p] = 0
        TBL[p] += 1
    return all([TBL[p] == 1 for p in TBL])

def cmp_exact(val, refval):
    return val == refval

def cmp_aprox(val, refval, eps=1e-2):
    if refval == 0.0:
        chk = val == refval
    else:
        rel = val/refval
        chk = rel >= 1.0 - eps and rel <= 1.0 + eps
    return chk

def cmp_tflops(TBL0, TBL1, DESC):
    res = []
    for p in PRECISIONS:
        v, rv = TBL0[p], TBL1[p]
        if v is None and rv is None:
            res.append((p, True, v, rv))
        elif v is not None and rv is not None:
            tflops_chk = cmp_aprox(v, rv)
            if not tflops_chk:
                res.append((p, False,v,rv))
                #print(f"{DESC} TFLOPS MISMATCH for {p}, {v:.2f} != {rv:.2f}")
            else:
                res.append((p, True,v,rv))
        elif v is None:
            if rv != 0.0:
                res.append((p, False,v,rv))
                #print(f"{DESC} TFLOPS MISMATCH for {p}, {v} != {rv:.2f}")
            else:
                res.append((p, True,v,rv))
        else:
            if v != 0.0:
                res.append((p, False,v,rv))
                #print(f"{DESC} TFLOPS MISMATCH for {p}, {v:.2f} != {rv}")
            else:
                res.append((p, True,v,rv))
    return all([s for _,s,_,_ in res]), res

def get_ref_stats(refcsv):
    if refcsv.endswith('xlsx'):
        if '@' in refcsv:
            xlsx_sheetname, xlsx_filename = refcsv.split('@')
        else:
            xlsx_sheetname, xlsx_filename = None, refcsv
        rows, cols = parse_xlsx(xlsx_filename, xlsx_sheetname)
    else:
        rows, cols = parse_csv(refcsv)

    #structure of XLSX
    # hdr  row : CONFIG, cfg1, cfg2, ..., cfgN
    # data rows: STAT,   v1,   v2,   ..., vN
    ref_stats = {c: {} for c in cols[1:]} #type: ignore
    for r in rows:
        statname = r['CONFIG']
        for c in cols[1:]:
            ref_stats[c][statname] = r[c]

    return ref_stats

def check_arch_stats(archyaml, refcsv):
    ref_stats = get_ref_stats(refcsv)
    ref_exact_stat_names = [
            'Memory Freq (MHz)',
            'Memory Capacity (GB)',
            'Memory',
            'SMCount',
            'Memory Units',
            'Dev Ramp Penalty',
            'BoostFreq1 (MHz)',
            'BoostFreq2 (MHz)',
            ]
    ref_aprox_stat_names = [
            'Memory Bandwidth (GB/s)',
            ]

    ipblocks, packages = get_arspec_from_yaml(archyaml)
    for pn, package in packages.items():
        pmemory  = package.get_ipgroup('memory')
        pcompute = package.get_ipgroup('compute')
        if pn in ref_stats:
            EV0 = [ package.mem_frequency(), package.mem_size(), pmemory.ipname.upper(),
                   pcompute.num_units,
                   pmemory.num_units,
                   pcompute.ramp_penalty,
                   pcompute.ipobj.get_pipe('matrix').frequency(),
                   pcompute.ipobj.get_pipe('vector').frequency(),
                   ]
            EV1 = [ref_stats[pn][s] for s in ref_exact_stat_names]
            EN0 = len(EV0)
            EN1 = len(EV1)
            assert EN0 == EN1, f"len(exact_stat_vals)[{EN0}] != len(exact_ref_stat_vals)[{EN1}]"
            E_CMP = [cmp_exact(EV0[i], EV1[i]) for i in range(EN0)]
            E_CHK = all(E_CMP)

            AV0 = [ package.peak_bandwidth(), ]
            AV1 = [ref_stats[pn][s] for s in ref_aprox_stat_names]
            AN0 = len(AV0)
            AN1 = len(AV1)
            assert AN0 == AN1, f"len(aprox_stat_vals)[{AN0}] != len(aprox_ref_stat_vals)[{AN1}]"
            A_CMP = [cmp_aprox(AV0[i], AV1[i]) for i in range(AN0)]
            A_CHK = all(A_CMP)

            print("Checking Package.... ", pn)
            if not E_CHK:
                E_ERR = [(ref_exact_stat_names[i],EV0[i], EV1[i]) for i,x in enumerate(E_CMP) if not x]
                for n,v,rv in E_ERR:
                    print(f"  EXACT STAT MISMATCH {n}, val={fmt(v)}, ref={fmt(rv)}")
                assert False

            if not A_CHK:
                A_ERR = [(ref_aprox_stat_names[i],AV0[i], AV1[i]) for i,y in enumerate(A_CMP) if not y]
                for n,v,rv in A_ERR:
                    print(f"  APROX STAT MISMATCH {n}, val={fmt(v)}, ref={fmt(rv)}")
                assert False

            #Get Ref Compute TFLOPS
            MN1 = [(prec, f"TensorCore {prec.upper()} TFLOPS") for prec in FLOAT_PRECISIONS] + \
                  [(prec, f"TensorCore {prec.upper()} TOPS") for prec in INT_PRECISIONS]

            VN1 = [(prec, f"CudaCore {prec.upper()} TFLOPS") for prec in FLOAT_PRECISIONS] + \
                  [(prec, f"CudaCore {prec.upper()} TOPS") for prec in INT_PRECISIONS]
            MV1 = {p: ref_stats[pn].get(s, None) for p,s in MN1}
            VV1 = {p: ref_stats[pn].get(s, None) for p,s in VN1}

            #Get Arch Compute TFLOPS
            Mpipe = pcompute.ipobj.get_pipe('matrix')
            Vpipe = pcompute.ipobj.get_pipe('vector')
            ML0 = [(prec, xx.tpt.get(prec, None)) for prec in PRECISIONS for xx in Mpipe.instructions if xx.name == 'mac']
            VL0 = [(prec, xx.tpt.get(prec, None)) for prec in PRECISIONS for xx in Vpipe.instructions if xx.name == 'mac']
            assert check_uniq_keys_in_list(ML0), f"ML0 Error"
            assert check_uniq_keys_in_list(VL0), f"VL0 Error"
            MV0 = {p: package.peak_flops('matrix', 'mac', p, mul_factor=2) if s is not None else s for p,s in ML0}
            VV0 = {p: package.peak_flops('vector', 'mac', p, mul_factor=2) if s is not None else s for p,s in VL0}

            #Compare
            MpipeCheck, MRes = cmp_tflops(MV0, MV1, "MATRIX")
            VpipeCheck, VRes = cmp_tflops(VV0, VV1, "VECTOR")

            if not MpipeCheck:
                for p, c, v, rv in MRes:
                    if not c:
                        print(f"  MATRIX TFLOPS MISMATCH: {p}, val={fmt(v)}, ref={fmt(rv)}")
                assert False

            if not VpipeCheck:
                for p, c, v, rv in VRes:
                    if not c:
                        print(f"  VECTOR TFLOPS MISMATCH: {p}, val={fmt(v)}, ref={fmt(rv)}")
                assert False

def test_device():
    print()
    check_arch_stats('config/all_archs.yaml','GPU@config/Nvidia.xlsx')
