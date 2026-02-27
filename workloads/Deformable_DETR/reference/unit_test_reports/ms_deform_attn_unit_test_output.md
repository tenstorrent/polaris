
═════════════════════════════════════════════════════════════════
MS DEFORM ATTN CORE UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 4 items

workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_func_unit.py::test_shape_inference 
MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93mstandard_multi_level[0m (4 levels, typical multi-scale — baseline shape)
├─ INPUT: value[2,3343,8,32] sampling[2,100,8,4,4,2] attn[2,100,8,4,4]
├─ SHAPE: Expected=[2,100,256] | Actual=[2,100,256] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93msingle_level[0m (L=1, single feature level — minimal level loop)
├─ INPUT: value[1,1024,4,16] sampling[1,50,4,1,2,2] attn[1,50,4,1,2]
├─ SHAPE: Expected=[1,50,64] | Actual=[1,50,64] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93mmany_points[0m (P=8 sampling points — more interpolation)
├─ INPUT: value[1,320,4,16] sampling[1,64,4,2,8,2] attn[1,64,4,2,8]
├─ SHAPE: Expected=[1,64,64] | Actual=[1,64,64] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93mlarge_batch[0m (N=4 batch size — batch scalability)
├─ INPUT: value[4,525,8,32] sampling[4,80,8,3,4,2] attn[4,80,8,3,4]
├─ SHAPE: Expected=[4,80,256] | Actual=[4,80,256] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93msingle_query[0m (Lq=1 — minimal query count)
├─ INPUT: value[2,80,4,16] sampling[2,1,4,2,4,2] attn[2,1,4,2,4]
├─ SHAPE: Expected=[2,1,64] | Actual=[2,1,64] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93mmany_heads[0m (M=16 attention heads — head scalability)
├─ INPUT: value[1,320,16,8] sampling[1,50,16,2,4,2] attn[1,50,16,2,4]
├─ SHAPE: Expected=[1,50,128] | Actual=[1,50,128] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93mnon_square_spatial[0m (Non-square spatial dims (H!=W) — asymmetric grids)
├─ INPUT: value[1,160,4,16] sampling[1,30,4,2,4,2] attn[1,30,4,2,4]
├─ SHAPE: Expected=[1,30,64] | Actual=[1,30,64] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93mminimal_config[0m (Tiny dims for quick smoke test)
├─ INPUT: value[1,4,1,4] sampling[1,2,1,1,1,2] attn[1,2,1,1,1]
├─ SHAPE: Expected=[1,2,4] | Actual=[1,2,4] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93msingle_element_pooling[0m (P=1, single sampling point — minimal interpolation)
├─ INPUT: value[1,16,2,8] sampling[1,10,2,1,1,2] attn[1,10,2,1,1]
├─ SHAPE: Expected=[1,10,16] | Actual=[1,10,16] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid config (all dims minimal) — degenerate case)
├─ INPUT: value[1,1,1,4] sampling[1,1,1,1,1,2] attn[1,1,1,1,1]
├─ SHAPE: Expected=[1,1,4] | Actual=[1,1,4] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_func_unit.py::test_e2e_numerical 
MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mstandard[0m (Multi-level typical dims — baseline correctness)
├─ INPUT: value[2,3343,8,32] sampling[2,100,8,4,4,2] attn[2,100,8,4,4]
├─ INPUT value[0:5]: [0.496714, -0.138264, 0.647689, 1.523030, -0.234153]
├─ INPUT sampling[0:5]: [0.267198, 0.677165, 0.876217, 0.261169, 0.857704]
├─ INPUT attn[0:5]: [0.134004, 0.575127, 0.062931, 0.227938, 0.228665]
├─ SHAPE: PT=[2,100,256] | TT=[2,100,256] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.25e-06, mean_diff=1.15e-07 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.507730, 0.123630, 0.629666, -0.327337, 0.654623]
├─ TT OUTPUT[0:5]: [-0.507730, 0.123630, 0.629666, -0.327337, 0.654623]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0-2.0) — baseline test)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ INPUT sampling[0:5]: [0.674177, 0.861255, 0.599754, 0.995571, 0.294672]
├─ INPUT attn[0:5]: [0.116668, 0.487893, 0.153057, 0.242382, 0.218033]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=7.15e-07, mean_diff=1.18e-07 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [2.515529, 2.185426, 2.217101, 2.208278, 2.442133]
├─ TT OUTPUT[0:5]: [2.515529, 2.185426, 2.217101, 2.208278, 2.442133]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [-1.834842, -1.104796, -1.744640, -1.360501, -1.359311]
├─ INPUT sampling[0:5]: [0.011429, 0.577764, 0.509227, 0.807720, 0.449382]
├─ INPUT attn[0:5]: [0.117331, 0.322556, 0.142667, 0.417447, 0.592899]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=7.15e-07, mean_diff=1.15e-07 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-2.967896, -2.837913, -3.113606, -2.518616, -2.861939]
├─ TT OUTPUT[0:5]: [-2.967896, -2.837913, -3.113605, -2.518616, -2.861939]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — zero feature maps edge case)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT sampling[0:5]: [0.989012, 0.549545, 0.281447, 0.077290, 0.444469]
├─ INPUT attn[0:5]: [0.353644, 0.299405, 0.099080, 0.247871, 0.059899]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative — real-world distribution)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [1.169752, 2.462391, 1.643800, -1.598457, 0.824106]
├─ INPUT sampling[0:5]: [0.313337, 0.730103, 0.825690, 0.572840, 0.106986]
├─ INPUT attn[0:5]: [0.080688, 0.518309, 0.227415, 0.173588, 0.266025]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=6.40e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.692807, -0.225032, -0.288613, 1.358294, -0.326578]
├─ TT OUTPUT[0:5]: [-1.692806, -0.225032, -0.288613, 1.358294, -0.326578]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001]
├─ INPUT sampling[0:5]: [0.480416, 0.727508, 0.576044, 0.614486, 0.760458]
├─ INPUT attn[0:5]: [0.500444, 0.160897, 0.329089, 0.009571, 0.228276]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.27e-13, mean_diff=3.83e-14 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
├─ TT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [17490.271484, 891573.250000, 284861.187500, 298976.406250, 792034.250000]
├─ INPUT sampling[0:5]: [0.410095, 0.067034, 0.587016, 0.879863, 0.509672]
├─ INPUT attn[0:5]: [0.337907, 0.251291, 0.120565, 0.290237, 0.388449]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.50e-01, mean_diff=3.70e-02 → [92m✓ PASS (tol: rtol=0.001, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [875114.375000, 938587.500000, 893268.125000, 750162.375000, 950978.875000]
├─ TT OUTPUT[0:5]: [875114.375000, 938587.437500, 893268.125000, 750162.375000, 950978.875000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93msingle_level[0m (L=1, single feature level — minimal level loop)
├─ INPUT: value[1,1024,4,16] sampling[1,50,4,1,2,2] attn[1,50,4,1,2]
├─ INPUT value[0:5]: [-1.043159, -0.820856, 0.665146, 1.822627, -1.441583]
├─ INPUT sampling[0:5]: [0.277605, 0.465664, 0.809017, 0.168974, 0.813884]
├─ INPUT attn[0:5]: [0.433086, 0.566914, 0.222482, 0.777518, 0.754408]
├─ SHAPE: PT=[1,50,64] | TT=[1,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=1.73e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.001767, 0.986160, -0.642988, 0.722283, -0.011570]
├─ TT OUTPUT[0:5]: [0.001767, 0.986160, -0.642988, 0.722283, -0.011570]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mmany_points[0m (P=8 sampling points — more interpolation)
├─ INPUT: value[1,320,4,16] sampling[1,64,4,2,8,2] attn[1,64,4,2,8]
├─ INPUT value[0:5]: [-1.560352, -0.030978, -0.620928, -1.464581, 1.411946]
├─ INPUT sampling[0:5]: [0.302386, 0.560740, 0.330474, 0.104734, 0.457218]
├─ INPUT attn[0:5]: [0.114242, 0.218645, 0.205612, 0.005862, 0.185692]
├─ SHAPE: PT=[1,64,64] | TT=[1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=2.46e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.009823, -0.348366, -0.405002, -0.357803, 0.285534]
├─ TT OUTPUT[0:5]: [-0.009823, -0.348366, -0.405002, -0.357803, 0.285534]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mlarge_batch[0m (N=4 batch size — batch scalability)
├─ INPUT: value[4,525,8,32] sampling[4,80,8,3,4,2] attn[4,80,8,3,4]
├─ INPUT value[0:5]: [-0.290503, 0.112128, 1.250795, -1.360890, 0.099933]
├─ INPUT sampling[0:5]: [0.425312, 0.970344, 0.960152, 0.488511, 0.039182]
├─ INPUT attn[0:5]: [0.161523, 0.337211, 0.296645, 0.204621, 0.216636]
├─ SHAPE: PT=[4,80,256] | TT=[4,80,256] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.91e-06, mean_diff=9.29e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.644840, 0.765168, 0.170880, 0.449732, 0.765179]
├─ TT OUTPUT[0:5]: [0.644840, 0.765169, 0.170880, 0.449732, 0.765179]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mmany_heads[0m (M=16 attention heads — head scalability)
├─ INPUT: value[1,320,16,8] sampling[1,50,16,2,4,2] attn[1,50,16,2,4]
├─ INPUT value[0:5]: [0.519476, -1.268750, 0.240420, -0.803957, 0.017344]
├─ INPUT sampling[0:5]: [0.090514, 0.526550, 0.367943, 0.888221, 0.877011]
├─ INPUT attn[0:5]: [0.189085, 0.259260, 0.296135, 0.255520, 0.387178]
├─ SHAPE: PT=[1,50,128] | TT=[1,50,128] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=3.58e-07, mean_diff=3.21e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.354342, 0.766523, -0.050726, 0.610562, 0.236416]
├─ TT OUTPUT[0:5]: [-0.354342, 0.766523, -0.050726, 0.610562, 0.236416]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93msingle_query[0m (Lq=1 — minimal query count)
├─ INPUT: value[2,80,4,16] sampling[2,1,4,2,4,2] attn[2,1,4,2,4]
├─ INPUT value[0:5]: [0.205865, 1.166762, -2.072640, -0.632687, 0.997126]
├─ INPUT sampling[0:5]: [0.208827, 0.889028, 0.330314, 0.370895, 0.366108]
├─ INPUT attn[0:5]: [0.046826, 0.040621, 0.847271, 0.065283, 0.286695]
├─ SHAPE: PT=[2,1,64] | TT=[2,1,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=3.64e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.198628, 0.472577, -0.273686, 0.108083, 0.820938]
├─ TT OUTPUT[0:5]: [-0.198628, 0.472577, -0.273686, 0.108083, 0.820938]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93muniform_attention[0m (Equal attention weights — no weighting bias)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [-1.852211, -1.080999, 0.078564, -1.354136, 0.626148]
├─ INPUT sampling[0:5]: [0.393802, 0.650219, 0.523307, 0.864704, 0.981569]
├─ INPUT attn[0:5]: [0.125000, 0.125000, 0.125000, 0.125000, 0.125000]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.19e-07, mean_diff=1.39e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.023750, -0.021358, -0.030756, 0.139451, 0.415119]
├─ TT OUTPUT[0:5]: [-0.023750, -0.021358, -0.030756, 0.139451, 0.415119]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mconcentrated_attention[0m (One-hot attention (all weight on first point))
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [-1.623731, -0.101784, -1.809791, 0.262654, 0.259953]
├─ INPUT sampling[0:5]: [0.431403, 0.574050, 0.221250, 0.716210, 0.475500]
├─ INPUT attn[0:5]: [1.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=1.68e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.113498, 1.050956, 0.268554, -0.155340, -0.454083]
├─ TT OUTPUT[0:5]: [-0.113498, 1.050956, 0.268554, -0.155340, -0.454083]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mboundary_coords[0m (Sampling near 0.0 and 1.0 — grid edge behavior)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [-1.037643, 0.593658, 1.102681, -0.512178, -0.265420]
├─ INPUT sampling[0:5]: [0.500000, 0.750000, 0.250000, 1.000000, 0.750000]
├─ INPUT attn[0:5]: [0.316390, 0.390996, 0.172041, 0.120573, 0.274949]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=1.57e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.396689, -0.015501, 0.014716, -0.355559, -0.163312]
├─ TT OUTPUT[0:5]: [-0.396689, -0.015501, 0.014716, -0.355559, -0.163312]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mcenter_coords[0m (All locations at 0.5 — center-only sampling)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [-0.131052, -0.200646, -1.690123, -0.794418, 1.528002]
├─ INPUT sampling[0:5]: [0.500000, 0.500000, 0.500000, 0.500000, 0.500000]
├─ INPUT attn[0:5]: [0.147401, 0.214099, 0.398155, 0.240344, 0.099996]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=3.58e-07, mean_diff=3.53e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.467051, 0.204189, 0.026655, -1.399529, 1.028497]
├─ TT OUTPUT[0:5]: [1.467051, 0.204189, 0.026655, -1.399530, 1.028497]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mcorner_coords[0m (All at 0.0 or 1.0 — extreme corner positions)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]
├─ INPUT value[0:5]: [-0.760186, -2.101584, -0.809757, -0.007517, -1.704361]
├─ INPUT sampling[0:5]: [0.000000, 0.000000, 1.000000, 1.000000, 1.000000]
├─ INPUT attn[0:5]: [0.299219, 0.199619, 0.176018, 0.325144, 0.317514]
├─ SHAPE: PT=[2,50,64] | TT=[2,50,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.19e-07, mean_diff=8.96e-09 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.011282, -0.292511, 0.180377, -0.259270, -0.102999]
├─ TT OUTPUT[0:5]: [-0.011282, -0.292511, 0.180377, -0.259270, -0.102999]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93msingle_element_pooling[0m (P=1, single sampling point — minimal interpolation)
├─ INPUT: value[1,16,2,8] sampling[1,10,2,1,1,2] attn[1,10,2,1,1]
├─ INPUT value[0:5]: [-1.669284, 0.563218, 0.420383, -1.601359, 0.632337]
├─ INPUT sampling[0:5]: [0.246291, 0.828703, 0.845454, 0.441380, 0.669421]
├─ INPUT attn[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: PT=[1,10,16] | TT=[1,10,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.19e-07, mean_diff=1.75e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.300562, -0.405697, -0.182414, 0.026230, 0.343335]
├─ TT OUTPUT[0:5]: [-0.300562, -0.405697, -0.182414, 0.026230, 0.343335]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mE2E_Numerical[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid config (all dims minimal) — degenerate case)
├─ INPUT: value[1,4,1,4] sampling[1,1,1,1,1,2] attn[1,1,1,1,1]
├─ INPUT value[0:5]: [-0.921771, -0.586318, 1.163999, -1.241724, -1.985230]
├─ INPUT sampling[0:5]: [0.581434, 0.839191]
├─ INPUT attn[0:5]: [1.000000]
├─ SHAPE: PT=[1,1,4] | TT=[1,1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=7.45e-09, mean_diff=1.86e-09 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.242719, -0.287044, -0.628677, -0.080735]
├─ TT OUTPUT[0:5]: [1.242719, -0.287044, -0.628677, -0.080735]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_func_unit.py::test_per_level_grid_sample 
MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93mstandard_4_levels[0m (4 levels — per-level grid_sample check)
├─ INPUT: value[2,3343,8,32] sampling[2,100,8,4,4,2] L=4 levels
├─ INPUT value[0:5]: [0.129740, 0.902362, 1.005804, 0.471890, -0.326213]
├─ INPUT sampling[0:5]: [0.194748, 0.385913, 0.563946, 0.119553, 0.029821]
├─ INPUT attn[0:5]: [0.166032, 0.386575, 0.209710, 0.237684, 0.198851]
├─ SHAPE: 4 levels checked — worst max_diff=8.88e-06 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=8.88e-06, mean_diff=4.55e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.018932, 0.611523, 0.250192, 0.847094, -0.017220]
├─ TT OUTPUT[0:5]: [0.018932, 0.611523, 0.250192, 0.847094, -0.017220]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93msingle_level[0m (L=1, single feature level — minimal level loop)
├─ INPUT: value[1,1024,4,16] sampling[1,50,4,1,2,2] L=1 levels
├─ INPUT value[0:5]: [-0.833154, 1.237691, -2.475731, -1.038631, 0.994645]
├─ INPUT sampling[0:5]: [0.972891, 0.859921, 0.564467, 0.249019, 0.426448]
├─ INPUT attn[0:5]: [0.014387, 0.985613, 0.737697, 0.262303, 0.813239]
├─ SHAPE: 1 levels checked — worst max_diff=2.38e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=1.80e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.156823, -0.075361, -0.039904, 0.154837, -0.160338]
├─ TT OUTPUT[0:5]: [-0.156823, -0.075361, -0.039904, 0.154837, -0.160338]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93mtwo_levels_asymmetric[0m (2 levels with very different spatial sizes)
├─ INPUT: value[2,1040,4,16] sampling[2,64,4,2,4,2] L=2 levels
├─ INPUT value[0:5]: [-1.298571, -0.092539, 0.070074, 1.855052, 1.370260]
├─ INPUT sampling[0:5]: [0.558301, 0.421189, 0.782076, 0.610531, 0.955816]
├─ INPUT attn[0:5]: [0.305776, 0.344110, 0.120838, 0.229276, 0.305855]
├─ SHAPE: 2 levels checked — worst max_diff=2.38e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=1.78e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.103294, 0.305755, -0.878443, -0.413167, -1.155927]
├─ TT OUTPUT[0:5]: [0.103294, 0.305755, -0.878443, -0.413167, -1.155927]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93mnegative_values[0m (Negative feature values with standard geometry)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels
├─ INPUT value[0:5]: [-1.283042, -1.002030, -1.530579, -1.080361, -1.652258]
├─ INPUT sampling[0:5]: [0.565911, 0.714777, 0.732933, 0.124525, 0.997036]
├─ INPUT attn[0:5]: [0.038694, 0.503343, 0.050244, 0.407719, 0.289141]
├─ SHAPE: 2 levels checked — worst max_diff=2.38e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=3.46e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-2.810215, -2.960869, -3.170809, -2.763809, -3.196222]
├─ TT OUTPUT[0:5]: [-2.810216, -2.960869, -3.170808, -2.763809, -3.196222]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93msmall_values[0m (Very small feature values with standard geometry)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels
├─ INPUT value[0:5]: [0.000000, 0.000001, 0.000000, 0.000000, 0.000000]
├─ INPUT sampling[0:5]: [0.985452, 0.797251, 0.876362, 0.570412, 0.309836]
├─ INPUT attn[0:5]: [0.090761, 0.284484, 0.292354, 0.332401, 0.223938]
├─ SHAPE: 2 levels checked — worst max_diff=1.14e-13 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.14e-13, mean_diff=1.18e-14 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
├─ TT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93mzeros_values[0m (All-zero feature values — zero output edge case)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels
├─ INPUT value[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT sampling[0:5]: [0.366696, 0.632981, 0.030831, 0.691256, 0.937811]
├─ INPUT attn[0:5]: [0.655148, 0.059963, 0.160685, 0.124204, 0.190865]
├─ SHAPE: 2 levels checked — worst max_diff=0.00e+00 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93mmixed_values[0m (Mix of positive/negative feature values)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels
├─ INPUT value[0:5]: [-2.172171, 0.455375, 3.775336, 2.687807, 0.269445]
├─ INPUT sampling[0:5]: [0.210195, 0.442968, 0.046006, 0.681472, 0.848801]
├─ INPUT attn[0:5]: [0.238729, 0.308520, 0.442936, 0.009815, 0.404104]
├─ SHAPE: 2 levels checked — worst max_diff=4.77e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=3.51e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.769352, 0.063549, -1.921472, 2.578893, 0.885959]
├─ TT OUTPUT[0:5]: [0.769352, 0.063549, -1.921472, 2.578893, 0.885959]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93mlarge_values[0m (Very large feature values — overflow risk)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels
├─ INPUT value[0:5]: [881307.562500, 979703.625000, 864582.500000, 838072.187500, 909790.312500]
├─ INPUT sampling[0:5]: [0.601214, 0.241093, 0.925680, 0.917536, 0.010038]
├─ INPUT attn[0:5]: [0.064494, 0.319389, 0.379562, 0.236555, 0.029153]
├─ SHAPE: 2 levels checked — worst max_diff=1.25e-01 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.25e-01, mean_diff=1.21e-02 → [92m✓ PASS (tol: rtol=0.001, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [863501.250000, 841981.687500, 1036102.875000, 963337.375000, 1044595.312500]
├─ TT OUTPUT[0:5]: [863501.250000, 841981.625000, 1036102.750000, 963337.375000, 1044595.375000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93msingle_element_pooling[0m (P=1, single sampling point — minimal interpolation)
├─ INPUT: value[1,16,2,8] sampling[1,10,2,1,1,2] L=1 levels
├─ INPUT value[0:5]: [-0.231383, 0.390613, 0.358297, 0.566244, 0.166234]
├─ INPUT sampling[0:5]: [0.867835, 0.418072, 0.922803, 0.727356, 0.121228]
├─ INPUT attn[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: 1 levels checked — worst max_diff=2.38e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=2.12e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.229940, -0.357867, -0.100184, 1.138430, -1.114634]
├─ TT OUTPUT[0:5]: [0.229940, -0.357867, -0.100184, 1.138430, -1.114634]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPerLevelGridSample[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid config (all dims minimal) — degenerate case)
├─ INPUT: value[1,4,1,4] sampling[1,1,1,1,1,2] L=1 levels
├─ INPUT value[0:5]: [-0.636738, 0.531559, 0.990208, -0.624134, 1.467781]
├─ INPUT sampling[0:5]: [0.034794, 0.784137]
├─ INPUT attn[0:5]: [1.000000]
├─ SHAPE: 1 levels checked — worst max_diff=5.96e-08 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.96e-08, mean_diff=3.07e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.719860, 0.994087, -0.051970, 0.299630]
├─ TT OUTPUT[0:5]: [0.719860, 0.994087, -0.051970, 0.299630]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_func_unit.py::test_intermediate_steps 
MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93mstandard_config[0m (Standard config — full intermediate comparison)
├─ INPUT: value[2,3343,8,32] sampling[2,100,8,4,4,2] 6 steps
├─ INPUT value[0:5]: [-0.357519, 0.148448, 0.993531, 1.838968, -0.744026]
├─ INPUT sampling[0:5]: [0.888653, 0.224453, 0.213538, 0.600574, 0.721727]
├─ INPUT attn[0:5]: [0.211679, 0.247400, 0.350225, 0.190695, 0.171880]
├─ SHAPE: 6 steps checked — worst max_diff=7.78e-06 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=7.78e-06, mean_diff=4.12e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.787104, -1.190204, -2.153804, 0.400656, -0.105699]
├─ TT OUTPUT[0:5]: [-0.787104, -1.190204, -2.153804, 0.400656, -0.105699]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93msmall_config[0m (Tiny dims for quick smoke test)
├─ INPUT: value[1,20,2,8] sampling[1,10,2,2,2,2] 6 steps
├─ INPUT value[0:5]: [0.632876, -0.656156, -1.627301, 1.129801, -1.211659]
├─ INPUT sampling[0:5]: [0.768442, 0.258594, 0.290348, 0.973731, 0.688250]
├─ INPUT attn[0:5]: [0.489972, 0.510028, 0.619067, 0.380933, 0.372255]
├─ SHAPE: 6 steps checked — worst max_diff=2.38e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=2.71e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.696509, 0.111555, -0.674418, -0.244389, -0.932417]
├─ TT OUTPUT[0:5]: [-0.696509, 0.111555, -0.674418, -0.244389, -0.932417]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93mnegative_values[0m (Negative feature values with standard geometry)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps
├─ INPUT value[0:5]: [-1.290678, -1.099277, -1.249007, -1.396347, -1.072394]
├─ INPUT sampling[0:5]: [0.165644, 0.717805, 0.246591, 0.419160, 0.617434]
├─ INPUT attn[0:5]: [0.114063, 0.081521, 0.327779, 0.476637, 0.178663]
├─ SHAPE: 6 steps checked — worst max_diff=4.77e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=1.19e-07 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-3.020215, -3.361649, -2.816862, -3.338176, -3.100731]
├─ TT OUTPUT[0:5]: [-3.020215, -3.361649, -2.816862, -3.338176, -3.100731]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93mlarge_values[0m (Very large feature values — overflow risk)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps
├─ INPUT value[0:5]: [906853.875000, 112530.007812, 100300.132812, 76207.679688, 942521.937500]
├─ INPUT sampling[0:5]: [0.318519, 0.505626, 0.767378, 0.564579, 0.230446]
├─ INPUT attn[0:5]: [0.512174, 0.238287, 0.010214, 0.239325, 0.462410]
├─ SHAPE: 6 steps checked — worst max_diff=2.50e-01 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.50e-01, mean_diff=3.79e-02 → [92m✓ PASS (tol: rtol=0.001, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [1056071.500000, 1280265.000000, 1030785.312500, 1025686.375000, 948792.125000]
├─ TT OUTPUT[0:5]: [1056071.500000, 1280264.875000, 1030785.312500, 1025686.375000, 948792.062500]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93mzeros_values[0m (All-zero feature values — zero output edge case)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps
├─ INPUT value[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT sampling[0:5]: [0.479649, 0.912466, 0.690418, 0.589381, 0.006957]
├─ INPUT attn[0:5]: [0.326881, 0.151054, 0.219168, 0.302898, 0.460478]
├─ SHAPE: 6 steps checked — worst max_diff=0.00e+00 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93mmixed_values[0m (Mix of positive/negative feature values)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps
├─ INPUT value[0:5]: [1.487826, -0.488540, -2.810998, 3.018626, 0.330088]
├─ INPUT sampling[0:5]: [0.200542, 0.558721, 0.138921, 0.102533, 0.596504]
├─ INPUT attn[0:5]: [0.016252, 0.497891, 0.020780, 0.465077, 0.380309]
├─ SHAPE: 6 steps checked — worst max_diff=5.96e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.96e-07, mean_diff=6.27e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-2.275722, -1.508327, -1.546454, 1.342775, 0.545717]
├─ TT OUTPUT[0:5]: [-2.275722, -1.508327, -1.546454, 1.342775, 0.545718]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93msmall_values[0m (Very small feature values with standard geometry)
├─ INPUT: value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps
├─ INPUT value[0:5]: [0.000001, 0.000000, 0.000000, 0.000001, 0.000001]
├─ INPUT sampling[0:5]: [0.601067, 0.241265, 0.747372, 0.294572, 0.228762]
├─ INPUT attn[0:5]: [0.174159, 0.423282, 0.043682, 0.358877, 0.168210]
├─ SHAPE: 6 steps checked — worst max_diff=3.41e-13 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=3.41e-13, mean_diff=3.80e-14 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
├─ TT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93msingle_element_pooling[0m (P=1, single sampling point — minimal interpolation)
├─ INPUT: value[1,16,2,8] sampling[1,10,2,1,1,2] 6 steps
├─ INPUT value[0:5]: [1.427614, 0.080751, 0.191511, 0.608525, -0.582141]
├─ INPUT sampling[0:5]: [0.979673, 0.328587, 0.922652, 0.674105, 0.034768]
├─ INPUT attn[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: 6 steps checked — worst max_diff=1.19e-07 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.19e-07, mean_diff=1.16e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.361245, -0.360597, -0.159358, 0.022807, -0.148236]
├─ TT OUTPUT[0:5]: [0.361245, -0.360597, -0.159358, 0.022807, -0.148236]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mIntermediateSteps[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid config (all dims minimal) — degenerate case)
├─ INPUT: value[1,4,1,4] sampling[1,1,1,1,1,2] 6 steps
├─ INPUT value[0:5]: [-0.700122, -0.878386, 1.273181, -1.531156, 1.363569]
├─ INPUT sampling[0:5]: [0.130578, 0.634538]
├─ INPUT attn[0:5]: [1.000000]
├─ SHAPE: 6 steps checked — worst max_diff=2.98e-08 → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.98e-08, mean_diff=1.12e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.244143, 0.581252, -0.384707, -0.260498]
├─ TT OUTPUT[0:5]: [0.244143, 0.581252, -0.384707, -0.260498]
└─ RESULT: [92m✓ PASS[0m
PASSED

============================== slowest durations ===============================
0.23s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_func_unit.py::test_e2e_numerical
0.12s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_func_unit.py::test_intermediate_steps
0.12s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_func_unit.py::test_per_level_grid_sample

(9 durations < 0.005s hidden.  Use -vv to show these durations.)
============================== 4 passed in 0.51s ===============================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                      SHAPE       NUMERICAL   TOTAL
ShapeInference              10/10       N/A         [92m✓ PASS[0m
E2E_Numerical               19/19       19/19       [92m✓ PASS[0m
PerLevelGridSample          10/10       10/10       [92m✓ PASS[0m
IntermediateSteps           9/9         9/9         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
TOTAL                       48/48          38/38       [92m✓ PASS[0m
═════════════════════════════════════════════════════════════════
