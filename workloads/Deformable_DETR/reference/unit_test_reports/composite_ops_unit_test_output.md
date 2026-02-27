
═════════════════════════════════════════════════════════════════
COMPOSITE OPS UNIT TEST SUITE — TTSim SimOpHandle pipeline
═════════════════════════════════════════════════════════════════

============================================== test session starts ===============================================
platform win32 -- Python 3.13.2, pytest-8.3.4, pluggy-1.6.0 -- C:\Users\Akandala\AppData\Local\miniforge3\envs\polarisdev\python.exe
cachedir: .pytest_cache
metadata: {'Python': '3.13.2', 'Platform': 'Windows-11-10.0.26100-SP0', 'Packages': {'pytest': '8.3.4', 'pluggy': '1.6.0'}, 'Plugins': {'hydra-core': '1.3.2', 'cov': '6.3.0', 'json-report': '1.5.0', 'metadata': '3.1.1', 'mock': '3.15.1', 'xdist': '3.8.0'}}
rootdir: C:\Users\Akandala\Desktop\Projects\2026\Tenstorrent\polaris
configfile: pyproject.toml
plugins: hydra-core-1.3.2, cov-6.3.0, json-report-1.5.0, metadata-3.1.1, mock-3.15.1, xdist-3.8.0
collecting ... collected 60 items

workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[positive] 
OP: [1mConv2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: X=[1,2,4,4] W=[2,2,3,3] K=3
├─ INPUT X[0:5]: [1.771270, 1.074045, 1.358466, 1.115869, 1.863103]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.96e-08, mean_diff=2.61e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.426961, 0.410136, 0.431454, 0.419554, 0.384351, 0.365865, 0.383672, 0.373604]
├─ TT  [0:5]: [0.426961, 0.410136, 0.431454, 0.419554, 0.384351, 0.365865, 0.383672, 0.373604]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[negative] 
OP: [1mConv2d[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: X=[1,2,4,4] W=[2,2,3,3] K=3
├─ INPUT X[0:5]: [-1.771270, -1.074045, -1.358466, -1.115869, -1.863103]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.96e-08, mean_diff=2.61e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.392824, 0.375998, 0.397317, 0.385417, 0.349770, 0.331285, 0.349091, 0.339024]
├─ TT  [0:5]: [0.392824, 0.375998, 0.397317, 0.385417, 0.349770, 0.331285, 0.349091, 0.339024]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[zeros] 
OP: [1mConv2d[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: X=[1,2,4,4] W=[2,2,3,3] K=3
├─ INPUT X[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[mixed] 
OP: [1mConv2d[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: X=[1,2,4,4] W=[2,2,3,3] K=3
├─ INPUT X[0:5]: [0.723272, -1.290239, 0.722791, 3.076073, -0.071652]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.96e-08, mean_diff=1.82e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.127367, -0.137305, 0.019335, -0.117431, 0.001470, 0.031685, -0.116114, 0.359292]
├─ TT  [0:5]: [0.127367, -0.137305, 0.019335, -0.117431, 0.001470, 0.031685, -0.116114, 0.359292]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[small] 
OP: [1mConv2d[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: X=[1,2,4,4] W=[2,2,3,3] K=3
├─ INPUT X[0:5]: [0.000001, 0.000000, 0.000000, 0.000000, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[large] 
OP: [1mConv2d[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: X=[1,2,4,4] W=[2,2,3,3] K=3
├─ INPUT X[0:5]: [771270.312500, 74044.648438, 358465.718750, 115869.062500, 863103.437500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=8.19e+03, mean_diff=2.05e+03 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [42827046912.000000, 39332184064.000000, 49825349632.000000, 45291802624.000000, 31476049920.000000, 26321469440.000000, 33302413312.000000, 30601437184.000000]
├─ TT  [0:5]: [42827046912.000000, 39332175872.000000, 49825353728.000000, 45291802624.000000, 31476049920.000000, 26321469440.000000, 33302409216.000000, 30601437184.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[minimum_input] 
OP: [1mConv2d[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: X=[1,1,3,3] W=[1,1,1,1] K=1
├─ INPUT X[0:5]: [-0.234153, -0.234137, 1.579213, 0.767435, -0.469474]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.013714, 0.013714, 0.025459, 0.020201, 0.012190, 0.018744, 0.012229, 0.012214, 0.016797]
├─ TT  [0:5]: [0.013714, 0.013714, 0.025459, 0.020201, 0.012190, 0.018744, 0.012229, 0.012214, 0.016797]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d_with_padding PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d_no_bias PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[positive] 
OP: [1mMaxPool2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: X=[1,3,8,8] K=2 stride=2
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[negative] 
OP: [1mMaxPool2d[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: X=[1,3,8,8] K=2 stride=2
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[zeros] 
OP: [1mMaxPool2d[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: X=[1,3,8,8] K=2 stride=2
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[mixed] 
OP: [1mMaxPool2d[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: X=[1,3,8,8] K=2 stride=2
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[small] 
OP: [1mMaxPool2d[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: X=[1,3,8,8] K=2 stride=2
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[large] 
OP: [1mMaxPool2d[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: X=[1,3,8,8] K=2 stride=2
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[minimum_input] 
OP: [1mMaxPool2d[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: X=[1,1,2,2] K=2 stride=2
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d_with_padding PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d[positive] 
OP: [1mBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: X=[1,4,8,8] C=4
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d[negative] 
OP: [1mBatchNorm2d[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: X=[1,4,8,8] C=4
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d[zeros] 
OP: [1mBatchNorm2d[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: X=[1,4,8,8] C=4
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d[mixed] 
OP: [1mBatchNorm2d[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: X=[1,4,8,8] C=4
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d[small] 
OP: [1mBatchNorm2d[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: X=[1,4,8,8] C=4
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d[large] 
OP: [1mBatchNorm2d[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: X=[1,4,8,8] C=4
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d[minimum_input] 
OP: [1mBatchNorm2d[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: X=[1,1,1,1] C=1
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m
├─ INFO: perf_stats populated: True
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_batchnorm2d_param_count PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm[positive] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: X=[2,8,16] norm_size=16
├─ INPUT X[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.305196, 1.443177, 0.779481, 0.374881, -0.968289, -0.968362, -1.265468, 1.186650, 0.382335, 0.706893]
├─ TT  [0:5]: [-0.305196, 1.443177, 0.779481, 0.374881, -0.968289, -0.968362, -1.265468, 1.186650, 0.382335, 0.706893]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm[negative] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: X=[2,8,16] norm_size=16
├─ INPUT X[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.305196, -1.443177, -0.779481, -0.374881, 0.968289, 0.968362, 1.265468, -1.186650, -0.382335, -0.706893]
├─ TT  [0:5]: [0.305196, -1.443177, -0.779481, -0.374881, 0.968289, 0.968362, 1.265468, -1.186650, -0.382335, -0.706893]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm[zeros] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: X=[2,8,16] norm_size=16
├─ INPUT X[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm[mixed] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: X=[2,8,16] norm_size=16
├─ INPUT X[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.555353, -0.119996, 0.715926, 1.646920, -0.221982, -0.221964, 1.706675, 0.843285, -0.472264, 0.604114]
├─ TT  [0:5]: [0.555353, -0.119996, 0.715926, 1.646920, -0.221982, -0.221964, 1.706675, 0.843285, -0.472264, 0.604114]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm[small] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: X=[2,8,16] norm_size=16
├─ INPUT X[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.000032, 0.000150, 0.000081, 0.000039, -0.000101, -0.000101, -0.000132, 0.000124, 0.000040, 0.000074]
├─ TT  [0:5]: [-0.000032, 0.000150, 0.000081, 0.000039, -0.000101, -0.000101, -0.000132, 0.000124, 0.000040, 0.000074]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm[large] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: X=[2,8,16] norm_size=16
├─ INPUT X[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.305210, 1.443244, 0.779517, 0.374898, -0.968334, -0.968407, -1.265527, 1.186705, 0.382352, 0.706925]
├─ TT  [0:5]: [-0.305210, 1.443244, 0.779517, 0.374898, -0.968334, -0.968407, -1.265527, 1.186705, 0.382352, 0.706925]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm[minimum_input] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: X=[1,1,4] norm_size=4
├─ INPUT X[0:5]: [0.496714, -0.138264, 0.647689, 1.523030]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.228693, -1.299775, 0.025971, 1.502497]
├─ TT  [0:5]: [-0.228693, -1.299775, 0.025971, 1.502497]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_layernorm_param_count PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_groupnorm[positive] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: X=[1,8,4,4] groups=4
├─ INPUT X[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.221979, 1.876112, 1.079660, 0.594130, -1.017708, -1.017795, -1.374330, 1.568273, 0.603075, 0.992552]
├─ TT  [0:5]: [-0.221979, 1.876112, 1.079660, 0.594130, -1.017708, -1.017795, -1.374330, 1.568273, 0.603075, 0.992552]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_groupnorm[negative] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: X=[1,8,4,4] groups=4
├─ INPUT X[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.221979, -1.876112, -1.079660, -0.594130, 1.017708, 1.017795, 1.374330, -1.568273, -0.603075, -0.992552]
├─ TT  [0:5]: [0.221979, -1.876112, -1.079660, -0.594130, 1.017708, 1.017795, 1.374330, -1.568273, -0.603075, -0.992552]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_groupnorm[zeros] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: X=[1,8,4,4] groups=4
├─ INPUT X[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_groupnorm[mixed] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: X=[1,8,4,4] groups=4
├─ INPUT X[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.680941, -0.001028, 0.843088, 1.783207, -0.104013, -0.103995, 1.843548, 0.971696, -0.356748, 0.730180]
├─ TT  [0:5]: [0.680941, -0.001028, 0.843088, 1.783207, -0.104013, -0.103995, 1.843548, 0.971696, -0.356748, 0.730180]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_groupnorm[small] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: X=[1,8,4,4] groups=4
├─ INPUT X[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.000019, 0.000163, 0.000094, 0.000052, -0.000088, -0.000088, -0.000119, 0.000136, 0.000052, 0.000086]
├─ TT  [0:5]: [-0.000019, 0.000163, 0.000094, 0.000052, -0.000088, -0.000088, -0.000119, 0.000136, 0.000052, 0.000086]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_groupnorm[large] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: X=[1,8,4,4] groups=4
├─ INPUT X[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.221994, 1.876237, 1.079732, 0.594169, -1.017775, -1.017862, -1.374421, 1.568378, 0.603115, 0.992618]
├─ TT  [0:5]: [-0.221994, 1.876237, 1.079732, 0.594169, -1.017775, -1.017862, -1.374421, 1.568378, 0.603115, 0.992618]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_groupnorm[minimum_input] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: X=[1,4,1,1] groups=4
├─ INPUT X[0:5]: [0.496714, -0.138264, 0.647689, 1.523030]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize[positive] 
OP: [1mResize[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: X=[1,2,4,4] scale=2.0
├─ INPUT X[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.374540, 1.950714, 1.950714, 1.731994, 1.731994, 1.598659, 1.598659, 1.374540, 1.374540]
├─ TT  [0:5]: [1.374540, 1.374540, 1.950714, 1.950714, 1.731994, 1.731994, 1.598659, 1.598659, 1.374540, 1.374540]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize[negative] 
OP: [1mResize[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: X=[1,2,4,4] scale=2.0
├─ INPUT X[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.374540, -1.950714, -1.950714, -1.731994, -1.731994, -1.598659, -1.598659, -1.374540, -1.374540]
├─ TT  [0:5]: [-1.374540, -1.374540, -1.950714, -1.950714, -1.731994, -1.731994, -1.598659, -1.598659, -1.374540, -1.374540]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize[zeros] 
OP: [1mResize[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: X=[1,2,4,4] scale=2.0
├─ INPUT X[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize[mixed] 
OP: [1mResize[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: X=[1,2,4,4] scale=2.0
├─ INPUT X[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, 0.993428, -0.276529, -0.276529, 1.295377, 1.295377, 3.046060, 3.046060, 0.993428, 0.993428]
├─ TT  [0:5]: [0.993428, 0.993428, -0.276529, -0.276529, 1.295377, 1.295377, 3.046060, 3.046060, 0.993428, 0.993428]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize[small] 
OP: [1mResize[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: X=[1,2,4,4] scale=2.0
├─ INPUT X[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize[large] 
OP: [1mResize[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: X=[1,2,4,4] scale=2.0
├─ INPUT X[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 374540.125000, 950714.312500, 950714.312500, 731993.937500, 731993.937500, 598658.500000, 598658.500000, 374540.125000, 374540.125000]
├─ TT  [0:5]: [374540.125000, 374540.125000, 950714.312500, 950714.312500, 731993.937500, 731993.937500, 598658.500000, 598658.500000, 374540.125000, 374540.125000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize[minimum_input] 
OP: [1mResize[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: X=[1,1,2,2] scale=2.0
├─ INPUT X[0:5]: [0.496714, -0.138264, 0.647689, 1.523030]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714, 0.496714, -0.138264, -0.138264, 0.496714, 0.496714, -0.138264, -0.138264, 0.647689, 0.647689]
├─ TT  [0:5]: [0.496714, 0.496714, -0.138264, -0.138264, 0.496714, 0.496714, -0.138264, -0.138264, 0.647689, 0.647689]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_resize_scale_list PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_identity[positive] 
OP: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT X[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_identity[negative] 
OP: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT X[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_identity[zeros] 
OP: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT X[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_identity[mixed] 
OP: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT X[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_identity[small] 
OP: [1mDropout(p=0)[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT X[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_identity[large] 
OP: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT X[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_identity[minimum_input] 
OP: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: [1,1]
├─ INPUT X[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_param_structure PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_dropout_training_mode PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_pipeline_sim_op_created PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_pipeline_output_tensor_linked PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_pipeline_precision_set PASSED

=============================================== slowest durations ================================================
0.05s call     workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_maxpool2d[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[minimum_input]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_composite_ops_unit.py::test_conv2d[zeros]

(175 durations < 0.005s hidden.  Use -vv to show these durations.)
=============================================== 60 passed in 0.32s ===============================================

═════════════════════════════════════════════════════════════════
COMPOSITE OPS UNIT TEST SUMMARY
═════════════════════════════════════════════════════════════════
OP                      SHAPE     NUMERICAL   RESULT
Conv2d                  9/9        9/9           [92m✓ PASS[0m
MaxPool2d               8/8        8/8           [92m✓ PASS[0m
BatchNorm2d             8/8        8/8           [92m✓ PASS[0m
LayerNorm               8/8        8/8           [92m✓ PASS[0m
GroupNorm               7/7        7/7           [92m✓ PASS[0m
Resize                  8/8        8/8           [92m✓ PASS[0m
Dropout                 9/9        9/9           [92m✓ PASS[0m
Pipeline                3/3        3/3           [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
OVERALL: [92m✓ ALL PASS[0m
═════════════════════════════════════════════════════════════════
