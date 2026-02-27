
═════════════════════════════════════════════════════════════════
SEGMENTATION UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

====================================== test session starts ======================================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 6 items

polaris/workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_helper_functions 
MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [4, 8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[4,8] | TTSim=[4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374540, -1000.000000, -1000.000000, -1000.000000, 1.156019]
├─ TT OUTPUT[0:5]: [1.374540, -1000.000000, -1000.000000, -1000.000000, 1.156019]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 3, 4, 4]
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: PyTorch=[2,3,4,4] | TTSim=[2,3,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 1.609066, 1.133391, 0.000000, 1.327139]
├─ TT OUTPUT[0:5]: [0.000000, 1.609066, 1.133391, 0.000000, 1.327139]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [4, 8]
├─ INPUT input[0:5]: [-1.834842, -1.104796, -1.744640, -1.360501, -1.359311]
├─ SHAPE: PyTorch=[4,8] | TTSim=[4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.834842, -1.104796, -1000.000000, -1000.000000, -1000.000000]
├─ TT OUTPUT[0:5]: [-1.834842, -1.104796, -1000.000000, -1000.000000, -1000.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division edge case)
├─ INPUT: [4, 8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[4,8] | TTSim=[4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1000.000000, -1000.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [-1000.000000, -1000.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [4, 8]
├─ INPUT input[0:5]: [1.169752, 2.462391, 1.643800, -1.598457, 0.824106]
├─ SHAPE: PyTorch=[4,8] | TTSim=[4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.169752, 2.462391, 1.643800, -1000.000000, 0.824106]
├─ TT OUTPUT[0:5]: [1.169752, 2.462391, 1.643800, -1000.000000, 0.824106]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [4, 8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001]
├─ SHAPE: PyTorch=[4,8] | TTSim=[4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, -1000.000000, -1000.000000, -1000.000000, -1000.000000]
├─ TT OUTPUT[0:5]: [0.000000, -1000.000000, -1000.000000, -1000.000000, -1000.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [4, 8]
├─ INPUT input[0:5]: [17490.271484, 891573.250000, 284861.187500, 298976.406250, 792034.250000]
├─ SHAPE: PyTorch=[4,8] | TTSim=[4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.001, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [17490.271484, -1000.000000, 284861.187500, -1000.000000, 792034.250000]
├─ TT OUTPUT[0:5]: [17490.271484, -1000.000000, 284861.187500, -1000.000000, 792034.250000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 3, 4, 4]
├─ INPUT input[0:5]: [1.300964, 1.247062, 1.926335, 1.891603, 1.683277]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.300964, 1.300964, 1.247062, 1.247062, 1.926335]
├─ TT OUTPUT[0:5]: [1.300964, 1.300964, 1.247062, 1.247062, 1.926335]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 8, 4, 4]
├─ INPUT input[0:5]: [1.494602, 1.228083, 1.255474, 1.396330, 1.377315]
├─ SHAPE: PyTorch=[1,8,16,16] | TTSim=[1,8,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.494602, 1.494602, 1.494602, 1.494602, 1.228083]
├─ TT OUTPUT[0:5]: [1.494602, 1.494602, 1.494602, 1.494602, 1.228083]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [2, 3, 4, 4]
├─ INPUT input[0:5]: [-1.675731, -1.044712, -1.343304, -1.644020, -1.284213]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.675731, -1.675731, -1.044712, -1.044712, -1.343304]
├─ TT OUTPUT[0:5]: [-1.675731, -1.675731, -1.044712, -1.044712, -1.343304]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division edge case)
├─ INPUT: [2, 3, 4, 4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [2, 3, 4, 4]
├─ INPUT input[0:5]: [0.411730, 2.333524, -4.145279, -1.265374, 1.994253]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.411730, 0.411730, 2.333524, 2.333524, -4.145279]
├─ TT OUTPUT[0:5]: [0.411730, 0.411730, 2.333524, 2.333524, -4.145279]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [2, 3, 4, 4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000001, 0.000000]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [2, 3, 4, 4]
├─ INPUT input[0:5]: [93108.289062, 971655.875000, 483860.000000, 242522.703125, 531123.812500]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.001, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [93108.289062, 93108.289062, 971655.875000, 971655.875000, 483860.000000]
├─ TT OUTPUT[0:5]: [93108.289062, 93108.289062, 971655.875000, 971655.875000, 483860.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 4, 2, 2]
├─ INPUT input[0:5]: [1.984192, 1.333412, 1.673702, 1.196390, 1.354446]
├─ SHAPE: PyTorch=[1,4,4,4] | TTSim=[1,4,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.984192, 1.984192, 1.333412, 1.333412, 1.984192]
├─ TT OUTPUT[0:5]: [1.984192, 1.984192, 1.333412, 1.333412, 1.984192]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 4, 8, 8]
├─ INPUT input[0:5]: [1.087350, 1.230477, 1.411061, 1.310783, 1.565956]
├─ SHAPE: PyTorch=[2,8,8,8] | TTSim=[2,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.96e-08, mean_diff=1.04e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.038402, -0.030852, -0.018305, -0.031466, -0.022230]
├─ TT OUTPUT[0:5]: [0.038402, -0.030852, -0.018305, -0.031466, -0.022230]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 64, 8, 8]
├─ INPUT input[0:5]: [1.365106, 1.451206, 1.496060, 1.075622, 1.571762]
├─ SHAPE: PyTorch=[2,64,8,8] | TTSim=[2,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.79e-07, mean_diff=2.50e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.460384, -0.361515, -0.387830, -0.314252, -0.479949]
├─ TT OUTPUT[0:5]: [-0.460384, -0.361515, -0.387830, -0.314252, -0.479949]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [2, 4, 8, 8]
├─ INPUT input[0:5]: [-1.924035, -1.157871, -1.866915, -1.084157, -1.573574]
├─ SHAPE: PyTorch=[2,8,8,8] | TTSim=[2,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.19e-07, mean_diff=1.45e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.001609, -0.000155, -0.061005, 0.021151, -0.036896]
├─ TT OUTPUT[0:5]: [0.001609, -0.000155, -0.061005, 0.021151, -0.036896]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division edge case)
├─ INPUT: [2, 4, 8, 8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,8,8,8] | TTSim=[2,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.009561, -0.009561, -0.009561, -0.009561, -0.009561]
├─ TT OUTPUT[0:5]: [-0.009561, -0.009561, -0.009561, -0.009561, -0.009561]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [2, 4, 8, 8]
├─ INPUT input[0:5]: [-0.862991, 0.878751, -0.002524, 2.122246, 0.406730]
├─ SHAPE: PyTorch=[2,8,8,8] | TTSim=[2,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.79e-07, mean_diff=1.76e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.158569, 0.107051, -0.124483, 0.388849, 0.141957]
├─ TT OUTPUT[0:5]: [0.158569, 0.107051, -0.124483, 0.388849, 0.141957]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [2, 4, 8, 8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000001, 0.000000, 0.000001]
├─ SHAPE: PyTorch=[2,8,8,8] | TTSim=[2,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.003728, 0.003728, 0.003728, 0.003728, 0.003728]
├─ TT OUTPUT[0:5]: [0.003728, 0.003728, 0.003728, 0.003728, 0.003728]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [2, 4, 8, 8]
├─ INPUT input[0:5]: [553941.000000, 396247.906250, 48126.773438, 525672.562500, 153038.578125]
├─ SHAPE: PyTorch=[2,8,8,8] | TTSim=[2,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=6.25e-02, mean_diff=4.98e-03 → [92m✓ PASS (tol: rtol=0.001, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [-11309.625977, -21893.412109, -5574.290527, 26585.097656, -11424.587891]
├─ TT OUTPUT[0:5]: [-11309.627930, -21893.410156, -5574.293945, 26585.097656, -11424.583984]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mHelperFunctions[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 4, 3, 3]
├─ INPUT input[0:5]: [1.379099, 1.567098, 1.595592, 1.449859, 1.457020]
├─ SHAPE: PyTorch=[1,8,3,3] | TTSim=[1,8,3,3] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=8.94e-08, mean_diff=1.11e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.123742, 0.208825, 0.153611, 0.055725, 0.027563]
├─ TT OUTPUT[0:5]: [0.123742, 0.208825, 0.153611, 0.055725, 0.027563]
└─ RESULT: [92m✓ PASS[0m
PASSED
polaris/workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_mhattention_map 
MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [0.137454, 0.195071, 0.173199, 0.159866, 0.115602]
├─ INPUT k[0:5]: [0.193273, 0.186606, 0.104522, 0.102637, 0.137646]
├─ SHAPE: PyTorch=[1,4,8,8,8] | TTSim=[1,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.66e-10, mean_diff=9.82e-11 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.002022, 0.002004, 0.002008, 0.002020, 0.002003]
├─ TT OUTPUT[0:5]: [0.002022, 0.002004, 0.002008, 0.002020, 0.002003]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: q=[2, 4, 128] k=[2, 128, 8, 8]
├─ INPUT q[0:5]: [0.111505, 0.160907, 0.113339, 0.124059, 0.132714]
├─ INPUT k[0:5]: [0.148012, 0.117399, 0.113870, 0.191657, 0.118169]
├─ SHAPE: PyTorch=[2,4,8,8,8] | TTSim=[2,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.66e-10, mean_diff=1.07e-10 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.001931, 0.001923, 0.001939, 0.001922, 0.001935]
├─ TT OUTPUT[0:5]: [0.001931, 0.001923, 0.001939, 0.001922, 0.001935]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [0.183484, 0.110480, 0.174464, 0.136050, 0.135931]
├─ INPUT k[0:5]: [0.172480, 0.184405, 0.144260, 0.124295, 0.180611]
├─ SHAPE: PyTorch=[1,4,8,8,8] | TTSim=[1,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.66e-10, mean_diff=9.03e-11 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.002093, 0.002097, 0.002091]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.002093, 0.002097, 0.002091]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [-0.198901, -0.154954, -0.128145, -0.107729, -0.144447]
├─ INPUT k[0:5]: [-0.179686, -0.110648, -0.174996, -0.191654, -0.115229]
├─ SHAPE: PyTorch=[1,4,8,8,8] | TTSim=[1,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=6.98e-10, mean_diff=1.59e-10 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.001991, 0.001989, 0.001993, 0.001999, 0.001989]
├─ TT OUTPUT[0:5]: [0.001991, 0.001989, 0.001993, 0.001999, 0.001989]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division edge case)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT k[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,4,8,8,8] | TTSim=[1,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.01, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [0.001953, 0.001953, 0.001953, 0.001953, 0.001953]
├─ TT OUTPUT[0:5]: [0.001953, 0.001953, 0.001953, 0.001953, 0.001953]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [-0.169602, 0.261181, 0.184842, 0.128082, -0.210947]
├─ INPUT k[0:5]: [0.097370, 0.196714, -0.113386, -0.220388, -0.150153]
├─ SHAPE: PyTorch=[1,4,8,8,8] | TTSim=[1,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.66e-10, mean_diff=1.23e-10 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.001979, 0.002110, 0.001981, 0.001971, 0.001961]
├─ TT OUTPUT[0:5]: [0.001979, 0.002110, 0.001981, 0.001971, 0.001961]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT k[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,4,8,8,8] | TTSim=[1,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.01, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [0.001953, 0.001953, 0.001953, 0.001953, 0.001953]
├─ TT OUTPUT[0:5]: [0.001953, 0.001953, 0.001953, 0.001953, 0.001953]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [30096.443359, 24706.183594, 92633.515625, 89160.343750, 68327.679688]
├─ INPUT k[0:5]: [24321.074219, 4369.121582, 44715.750000, 67356.015625, 42770.609375]
├─ SHAPE: PyTorch=[1,4,8,8,8] | TTSim=[1,4,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.01, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 4, 4]
├─ INPUT q[0:5]: [0.149460, 0.122808, 0.125547, 0.139633, 0.137732]
├─ INPUT k[0:5]: [0.163343, 0.132264, 0.134368, 0.148858, 0.178490]
├─ SHAPE: PyTorch=[1,4,8,4,4] | TTSim=[1,4,8,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.86e-09, mean_diff=4.99e-10 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.007894, 0.007913, 0.007948, 0.007957, 0.008014]
├─ TT OUTPUT[0:5]: [0.007894, 0.007913, 0.007948, 0.007957, 0.008014]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: q=[1, 1, 128] k=[1, 128, 8, 8]
├─ INPUT q[0:5]: [0.167573, 0.104471, 0.134330, 0.164402, 0.128421]
├─ INPUT k[0:5]: [0.159704, 0.182638, 0.198278, 0.174947, 0.144113]
├─ SHAPE: PyTorch=[1,1,8,8,8] | TTSim=[1,1,8,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.66e-10, mean_diff=1.06e-10 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.002011, 0.002006, 0.002008, 0.001995, 0.002021]
├─ TT OUTPUT[0:5]: [0.002011, 0.002006, 0.002008, 0.001995, 0.002021]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMHAttentionMap[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: q=[1, 4, 128] k=[1, 128, 8, 4]
├─ INPUT q[0:5]: [0.182311, 0.102612, 0.121077, 0.161842, 0.109828]
├─ INPUT k[0:5]: [0.179594, 0.191523, 0.191276, 0.163035, 0.143432]
├─ SHAPE: PyTorch=[1,4,8,8,4] | TTSim=[1,4,8,8,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=9.31e-10, mean_diff=2.19e-10 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.003869, 0.003887, 0.003901, 0.003896, 0.003860]
├─ TT OUTPUT[0:5]: [0.003869, 0.003887, 0.003901, 0.003896, 0.003860]
└─ RESULT: [92m✓ PASS[0m
PASSED
polaris/workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_maskhead_smallconv 
MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: x=[1,128,8,8] bbox=[1,2,8,8,8]
├─ INPUT x[0:5]: [0.137454, 0.195071, 0.173199, 0.159866, 0.115602]
├─ SHAPE: PyTorch=[2,1,64,64] | TTSim=[2,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.41e-06, mean_diff=5.53e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [0.296576, 0.899959, 0.931204, 1.141358, 1.012982]
├─ TT OUTPUT[0:5]: [0.296576, 0.899960, 0.931204, 1.141357, 1.012982]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: x=[2,128,8,8] bbox=[2,2,8,8,8]
├─ INPUT x[0:5]: [0.111505, 0.160907, 0.113339, 0.124059, 0.132714]
├─ SHAPE: PyTorch=[4,1,64,64] | TTSim=[4,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.36e-06, mean_diff=7.58e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [-0.508734, -0.451531, -0.106726, 0.030066, -0.080515]
├─ TT OUTPUT[0:5]: [-0.508735, -0.451531, -0.106726, 0.030066, -0.080515]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: x=[1,128,8,8] bbox=[1,2,8,8,8]
├─ INPUT x[0:5]: [-0.183484, -0.110480, -0.174464, -0.136050, -0.135931]
├─ SHAPE: PyTorch=[2,1,64,64] | TTSim=[2,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=6.20e-06, mean_diff=7.74e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [0.293193, -0.284888, -1.051550, -0.656367, -0.572925]
├─ TT OUTPUT[0:5]: [0.293193, -0.284888, -1.051550, -0.656367, -0.572926]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division edge case)
├─ INPUT: x=[1,128,8,8] bbox=[1,2,8,8,8]
├─ INPUT x[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,1,64,64] | TTSim=[2,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.80e-06, mean_diff=4.00e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [-0.276332, -0.284760, -0.313273, -0.417176, -0.205953]
├─ TT OUTPUT[0:5]: [-0.276332, -0.284760, -0.313273, -0.417177, -0.205953]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: x=[1,128,8,8] bbox=[1,2,8,8,8]
├─ INPUT x[0:5]: [0.116975, 0.246239, 0.164380, -0.159846, 0.082411]
├─ SHAPE: PyTorch=[2,1,64,64] | TTSim=[2,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.97e-06, mean_diff=3.48e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [-0.318976, -0.316454, -0.442345, -0.346609, -0.230193]
├─ TT OUTPUT[0:5]: [-0.318976, -0.316454, -0.442345, -0.346609, -0.230194]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: x=[1,128,8,8] bbox=[1,2,8,8,8]
├─ INPUT x[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,1,64,64] | TTSim=[2,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-06, mean_diff=3.35e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [0.300875, 0.517971, 0.375127, 0.134444, 0.322707]
├─ TT OUTPUT[0:5]: [0.300875, 0.517971, 0.375127, 0.134444, 0.322706]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: x=[1,128,8,8] bbox=[1,2,8,8,8]
├─ INPUT x[0:5]: [1749.027222, 89157.328125, 28486.119141, 29897.640625, 79203.429688]
├─ SHAPE: PyTorch=[2,1,64,64] | TTSim=[2,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.88e-06, mean_diff=3.09e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [-0.047646, 0.457433, -0.146887, 0.704054, -0.054737]
├─ TT OUTPUT[0:5]: [-0.047646, 0.457433, -0.146887, 0.704054, -0.054737]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: x=[1,128,4,4] bbox=[1,2,8,4,4]
├─ INPUT x[0:5]: [0.130096, 0.124706, 0.192634, 0.189160, 0.168328]
├─ SHAPE: PyTorch=[2,1,32,32] | TTSim=[2,1,32,32] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=3.34e-06, mean_diff=6.03e-07 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [-0.202064, -0.037285, 0.072440, 0.074973, 0.184521]
├─ TT OUTPUT[0:5]: [-0.202064, -0.037285, 0.072439, 0.074973, 0.184521]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMaskHeadSmallConv[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: x=[1,128,8,8] bbox=[1,1,8,8,8]
├─ INPUT x[0:5]: [0.149460, 0.122808, 0.125547, 0.139633, 0.137732]
├─ SHAPE: PyTorch=[1,1,64,64] | TTSim=[1,1,64,64] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=7.75e-06, mean_diff=1.08e-06 → [92m✓ PASS (tol: rtol=0.01, atol=0.001)[0m
├─ PT OUTPUT[0:5]: [-0.200552, -0.621062, -0.534593, -0.514427, -0.585355]
├─ TT OUTPUT[0:5]: [-0.200552, -0.621062, -0.534593, -0.514427, -0.585355]
└─ RESULT: [92m✓ PASS[0m
PASSED
polaris/workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_detrsegm_shapes 
MODULE: [1mDETRsegm[0m
├─ EDGE CASE: [93mbaseline[0m (DETRsegm B=1 Q=4 16×16)
├─ INPUT: features=[1,256,16,16]
├─ INPUT features[0:5]: [-0.034433, -0.053072, -0.167742, -0.028395, -0.011712]
├─ SHAPE: TTSim=[1,4,2,2] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mDETRsegm[0m
├─ EDGE CASE: [93mbatch[0m (DETRsegm B=2 Q=4 16×16)
├─ INPUT: features=[2,256,16,16]
├─ INPUT features[0:5]: [-0.081541, -0.076305, -0.107817, 0.028865, 0.170967]
├─ SHAPE: TTSim=[2,4,2,2] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mDETRsegm[0m
├─ EDGE CASE: [93mscale[0m (DETRsegm Q=100 16×16)
├─ INPUT: features=[1,256,16,16]
├─ INPUT features[0:5]: [0.072590, 0.078277, 0.000272, -0.023251, -0.114489]
├─ SHAPE: TTSim=[1,100,2,2] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mDETRsegm[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size - degenerate/boundary case)
├─ INPUT: features=[1,256,8,8]
├─ INPUT features[0:5]: [0.081107, -0.060739, 0.162190, -0.062260, 0.033460]
├─ SHAPE: TTSim=[1,4,1,1] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
polaris/workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_reshape_squeeze 
MODULE: [1mReshape+Squeeze[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [8, 1, 4, 4]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[2,4,4,4] | TTSim=[2,4,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mReshape+Squeeze[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 1, 4, 4]
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: PyTorch=[1,1,4,4] | TTSim=[1,1,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mReshape+Squeeze[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [8, 1, 4, 4]
├─ INPUT input[0:5]: [-1.834842, -1.104796, -1.744640, -1.360501, -1.359311]
├─ SHAPE: PyTorch=[2,4,4,4] | TTSim=[2,4,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.834842, -1.104796, -1.744640, -1.360501, -1.359311]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mReshape+Squeeze[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division edge case)
├─ INPUT: [8, 1, 4, 4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,4,4,4] | TTSim=[2,4,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mReshape+Squeeze[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [8, 1, 4, 4]
├─ INPUT input[0:5]: [1.169752, 2.462391, 1.643800, -1.598457, 0.824106]
├─ SHAPE: PyTorch=[2,4,4,4] | TTSim=[2,4,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.169752, 2.462391, 1.643800, -1.598457, 0.824106]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mReshape+Squeeze[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [8, 1, 4, 4]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001]
├─ SHAPE: PyTorch=[2,4,4,4] | TTSim=[2,4,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mReshape+Squeeze[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [8, 1, 4, 4]
├─ INPUT input[0:5]: [17490.271484, 891573.250000, 284861.187500, 298976.406250, 792034.250000]
├─ SHAPE: PyTorch=[2,4,4,4] | TTSim=[2,4,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [17490.271484, 891573.250000, 284861.187500, 298976.406250, 792034.250000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
polaris/workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_parameter_counts 
MODULE: [1mMHAttentionMap params hdim=256[0m
├─ PyTorch params: 131,584
├─ TTSim  params:  131,584
└─ RESULT: [92m✓ MATCH[0m

MODULE: [1mMaskHeadSmallConv params dims=264[0m
├─ PyTorch params: 1,202,073
├─ TTSim  params:  1,202,073
└─ RESULT: [92m✓ MATCH[0m

MODULE: [1mDETRsegm total params[0m
├─ PyTorch params: 1,333,657
├─ TTSim  params:  1,333,657
└─ RESULT: [92m✓ MATCH[0m

MODULE: [1mMHA params hdim=128[0m
├─ PyTorch params: 33,024
├─ TTSim  params:  33,024
└─ RESULT: [92m✓ MATCH[0m

MODULE: [1mMaskHead params dims=136[0m
├─ PyTorch params: 280,697
├─ TTSim  params:  280,697
└─ RESULT: [92m✓ MATCH[0m
PASSED

======================================= warnings summary ========================================
workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_mhattention_map
  /home/aughag/Videos/TensTorrent/polaris/ttsim/ops/desc/data_compute.py:186: RuntimeWarning: invalid value encountered in multiply
    return iTList[0].data * iTList[1].data

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================================= slowest durations =======================================
210.63s call     workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_maskhead_smallconv
3.98s call     workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_mhattention_map
0.43s call     workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_helper_functions
0.08s call     workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_detrsegm_shapes
0.08s call     workloads/Deformable_DETR/unit_tests/test_segmentation_unit.py::test_parameter_counts

(13 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== 6 passed, 1 warning in 215.22s (0:03:35) ============================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                          SHAPE       NUMERICAL   TOTAL
HelperFunctions                 23/23       23/23       [92m✓ PASS[0m
MHAttentionMap                  11/11       11/11       [92m✓ PASS[0m
MaskHeadSmallConv               9/9         9/9         [92m✓ PASS[0m
DETRsegm                        4/4         N/A         [92m✓ PASS[0m
Reshape+Squeeze                 7/7         7/7         [92m✓ PASS[0m
ParamCount                      5/5         N/A         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
TOTAL                           59/59          50/50       [92m✓ PASS[0m
═════════════════════════════════════════════════════════════════
