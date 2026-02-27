
═════════════════════════════════════════════════════════════════
FUNCTIONAL OPS UNIT TEST SUITE — TTSim data_compute
═════════════════════════════════════════════════════════════════

============================================== test session starts ===============================================
platform win32 -- Python 3.13.2, pytest-8.3.4, pluggy-1.6.0 -- C:\Users\Akandala\AppData\Local\miniforge3\envs\polarisdev\python.exe
cachedir: .pytest_cache
metadata: {'Python': '3.13.2', 'Platform': 'Windows-11-10.0.26100-SP0', 'Packages': {'pytest': '8.3.4', 'pluggy': '1.6.0'}, 'Plugins': {'hydra-core': '1.3.2', 'cov': '6.3.0', 'json-report': '1.5.0', 'metadata': '3.1.1', 'mock': '3.15.1', 'xdist': '3.8.0'}}
rootdir: C:\Users\Akandala\Desktop\Projects\2026\Tenstorrent\polaris
configfile: pyproject.toml
plugins: hydra-core-1.3.2, cov-6.3.0, json-report-1.5.0, metadata-3.1.1, mock-3.15.1, xdist-3.8.0
collecting ... collected 158 items

workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[positive] 
OP: [1mRelu[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[negative] 
OP: [1mRelu[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[zeros] 
OP: [1mRelu[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[mixed] 
OP: [1mRelu[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, 0.000000, 1.295377, 3.046060, 0.000000, 0.000000, 3.158426, 1.534869, 0.000000, 1.085120]
├─ TT  [0:5]: [0.993428, 0.000000, 1.295377, 3.046060, 0.000000, 0.000000, 3.158426, 1.534869, 0.000000, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[small] 
OP: [1mRelu[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[large] 
OP: [1mRelu[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[minimum_input] 
OP: [1mRelu[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[positive] 
OP: [1mSigmoid[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.798113, 0.875525, 0.849667, 0.831831, 0.760608, 0.760604, 0.742324, 0.866015, 0.832174, 0.846586]
├─ TT  [0:5]: [0.798113, 0.875525, 0.849667, 0.831831, 0.760608, 0.760604, 0.742324, 0.866015, 0.832174, 0.846586]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[negative] 
OP: [1mSigmoid[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.201887, 0.124475, 0.150333, 0.168169, 0.239391, 0.239396, 0.257676, 0.133985, 0.167826, 0.153414]
├─ TT  [0:5]: [0.201887, 0.124475, 0.150333, 0.168169, 0.239391, 0.239396, 0.257676, 0.133985, 0.167826, 0.153414]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[zeros] 
OP: [1mSigmoid[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000]
├─ TT  [0:5]: [0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[mixed] 
OP: [1mSigmoid[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.729765, 0.431305, 0.785056, 0.954612, 0.385017, 0.385025, 0.959239, 0.822718, 0.281113, 0.747462]
├─ TT  [0:5]: [0.729765, 0.431305, 0.785056, 0.954612, 0.385017, 0.385025, 0.959239, 0.822718, 0.281113, 0.747462]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[small] 
OP: [1mSigmoid[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000]
├─ TT  [0:5]: [0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[large] 
OP: [1mSigmoid[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ TT  [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[minimum_input] 
OP: [1mSigmoid[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.621687]
├─ TT  [0:5]: [0.621687]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[positive] 
OP: [1mSoftmax[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.106031, 0.188652, 0.151590, 0.132668, 0.085218, 0.085215, 0.077267, 0.173359, 0.135635, 0.150947]
├─ TT  [0:5]: [0.106031, 0.188652, 0.151590, 0.132668, 0.085218, 0.085215, 0.077267, 0.173359, 0.135635, 0.150947]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[negative] 
OP: [1mSoftmax[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.132747, 0.074610, 0.092851, 0.106094, 0.165169, 0.165173, 0.182164, 0.081192, 0.103251, 0.092778]
├─ TT  [0:5]: [0.132747, 0.074610, 0.092851, 0.106094, 0.165169, 0.165173, 0.182164, 0.081192, 0.103251, 0.092778]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[zeros] 
OP: [1mSoftmax[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000]
├─ TT  [0:5]: [0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[mixed] 
OP: [1mSoftmax[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.046908, 0.013174, 0.063442, 0.365335, 0.010875, 0.010875, 0.408781, 0.080610, 0.063673, 0.481944]
├─ TT  [0:5]: [0.046908, 0.013174, 0.063442, 0.365335, 0.010875, 0.010875, 0.408781, 0.080610, 0.063673, 0.481944]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[small] 
OP: [1mSoftmax[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000]
├─ TT  [0:5]: [0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[large] 
OP: [1mSoftmax[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[minimum_input] 
OP: [1mSoftmax[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.000000]
├─ TT  [0:5]: [1.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_log[positive] 
OP: [1mLog[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.318119, 0.668196, 0.549273, 0.469165, 0.144982, 0.144961, 0.056460, 0.623892, 0.470700, 0.535366]
├─ TT  [0:5]: [0.318119, 0.668196, 0.549273, 0.469165, 0.144982, 0.144961, 0.056460, 0.623892, 0.470700, 0.535366]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_log[small] 
OP: [1mLog[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-14.560920, -13.766041, -13.999441, -14.174104, -15.178016, -15.178110, -15.660142, -13.849919, -14.170594, -14.028614]
├─ TT  [0:5]: [-14.560920, -13.766041, -13.999441, -14.174104, -15.178016, -15.178110, -15.660142, -13.849919, -14.170594, -14.028614]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_log[large] 
OP: [1mLog[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [12.833454, 13.764969, 13.503528, 13.302446, 11.957731, 11.957576, 10.969639, 13.671844, 13.306541, 13.470302]
├─ TT  [0:5]: [12.833454, 13.764969, 13.503528, 13.302446, 11.957731, 11.957576, 10.969639, 13.671844, 13.306541, 13.470302]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_log[minimum_input] 
OP: [1mLog[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.699740]
├─ TT  [0:5]: [-0.699740]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[positive] 
OP: [1mCos[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.194999, -0.370844, -0.160500, -0.027859, 0.402987, 0.403009, 0.490543, -0.291103, -0.030314, -0.136846]
├─ TT  [0:5]: [0.194999, -0.370844, -0.160500, -0.027859, 0.402987, 0.403009, 0.490543, -0.291103, -0.030314, -0.136846]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[negative] 
OP: [1mCos[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.194999, -0.370844, -0.160500, -0.027859, 0.402987, 0.403009, 0.490543, -0.291103, -0.030314, -0.136846]
├─ TT  [0:5]: [0.194999, -0.370844, -0.160500, -0.027859, 0.402987, 0.403009, 0.490543, -0.291103, -0.030314, -0.136846]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[zeros] 
OP: [1mCos[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ TT  [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[mixed] 
OP: [1mCos[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.545820, 0.962009, 0.271950, -0.995440, 0.892334, 0.892349, -0.999858, 0.035919, 0.590637, 0.466807]
├─ TT  [0:5]: [0.545820, 0.962009, 0.271950, -0.995440, 0.892334, 0.892349, -0.999858, 0.035919, 0.590637, 0.466807]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[small] 
OP: [1mCos[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ TT  [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[large] 
OP: [1mCos[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.851917, 0.738796, -0.957561, -0.967795, 0.647679, -0.298583, -0.273957, 0.237220, -0.887027, -0.052358]
├─ TT  [0:5]: [0.851917, 0.738796, -0.957561, -0.967795, 0.647679, -0.298583, -0.273957, 0.237220, -0.887027, -0.052358]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[minimum_input] 
OP: [1mCos[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.879153]
├─ TT  [0:5]: [0.879153]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[positive] 
OP: [1mSin[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.980803, 0.928695, 0.987036, 0.999612, 0.915206, 0.915196, 0.871417, 0.956692, 0.999540, 0.990592]
├─ TT  [0:5]: [0.980803, 0.928695, 0.987036, 0.999612, 0.915206, 0.915196, 0.871417, 0.956692, 0.999540, 0.990592]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[negative] 
OP: [1mSin[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.980803, -0.928695, -0.987036, -0.999612, -0.915206, -0.915196, -0.871417, -0.956692, -0.999540, -0.990592]
├─ TT  [0:5]: [-0.980803, -0.928695, -0.987036, -0.999612, -0.915206, -0.915196, -0.871417, -0.956692, -0.999540, -0.990592]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[zeros] 
OP: [1mSin[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[mixed] 
OP: [1mSin[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.837902, -0.273018, 0.962311, 0.095388, -0.451376, -0.451347, -0.016832, 0.999355, -0.806938, 0.884359]
├─ TT  [0:5]: [0.837902, -0.273018, 0.962311, 0.095388, -0.451376, -0.451347, -0.016832, 0.999355, -0.806938, 0.884359]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[small] 
OP: [1mSin[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[large] 
OP: [1mSin[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.523677, -0.673929, 0.288231, 0.251738, 0.761914, 0.954384, 0.961742, 0.971456, 0.461718, 0.998628]
├─ TT  [0:5]: [-0.523677, -0.673929, 0.288231, 0.251738, 0.761914, 0.954384, 0.961742, 0.971456, 0.461718, 0.998628]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[minimum_input] 
OP: [1mSin[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.476539]
├─ TT  [0:5]: [0.476539]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[positive] 
OP: [1mIdentity[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[negative] 
OP: [1mIdentity[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[zeros] 
OP: [1mIdentity[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[mixed] 
OP: [1mIdentity[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[small] 
OP: [1mIdentity[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[large] 
OP: [1mIdentity[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[minimum_input] 
OP: [1mIdentity[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_inverse_sigmoid[positive] 
OP: [1mInverseSigmoid[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.798113, 0.875525, 0.849667, 0.831831, 0.760608]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156018, 1.155995, 1.058083, 1.866177, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156018, 1.155995, 1.058083, 1.866177, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_inverse_sigmoid[small] 
OP: [1mInverseSigmoid[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.500000, 0.500000, 0.500000, 0.500000, 0.500000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_inverse_sigmoid[mixed] 
OP: [1mInverseSigmoid[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.729765, 0.431305, 0.785056, 0.954612, 0.385017]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046059, -0.468307, -0.468274, 3.158425, 1.534870, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046059, -0.468307, -0.468274, 3.158425, 1.534870, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_inverse_sigmoid[minimum_input] 
OP: [1mInverseSigmoid[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.621687]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[positive] 
OP: [1mGlu[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.045487, 1.483721, 1.285701, 1.384463, 1.380245, 1.316433, 0.781029, 1.508083, 1.087306, 1.155135]
├─ TT  [0:5]: [1.045487, 1.483721, 1.285701, 1.384463, 1.380245, 1.316433, 0.781029, 1.508083, 1.087306, 1.155135]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[negative] 
OP: [1mGlu[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.329053, -0.466993, -0.446293, -0.214196, -0.220870, -0.391640, -0.239556, -0.461827, -0.216937, -0.369621]
├─ TT  [0:5]: [-0.329053, -0.466993, -0.446293, -0.214196, -0.220870, -0.391640, -0.239556, -0.461827, -0.216937, -0.369621]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[zeros] 
OP: [1mGlu[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[mixed] 
OP: [1mGlu[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.382487, -0.106470, 1.242577, 2.506047, -0.580903, 0.023135, -0.028522, -0.228360, -1.923107, 0.244479]
├─ TT  [0:5]: [0.382487, -0.106470, 1.242577, 2.506047, -0.580903, 0.023135, -0.028522, -0.228360, -1.923107, 0.244479]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[small] 
OP: [1mGlu[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[large] 
OP: [1mGlu[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 601115.000000, 708072.625000, 20584.494141, 969909.875000, 304242.250000, 524756.437500]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 601115.000000, 708072.625000, 20584.494141, 969909.875000, 304242.250000, 524756.437500]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[minimum_input] 
OP: [1mGlu[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,2]
├─ INPUT input[0:5]: [0.496714, -0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.231215]
├─ TT  [0:5]: [0.231215]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[positive] 
OP: [1mTanh[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.879723, 0.960375, 0.939291, 0.921466, 0.819738, 0.819730, 0.784929, 0.953246, 0.921836, 0.936411]
├─ TT  [0:5]: [0.879723, 0.960375, 0.939291, 0.921466, 0.819738, 0.819730, 0.784929, 0.953246, 0.921836, 0.936411]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[negative] 
OP: [1mTanh[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.879723, -0.960375, -0.939291, -0.921466, -0.819738, -0.819730, -0.784929, -0.953246, -0.921836, -0.936411]
├─ TT  [0:5]: [-0.879723, -0.960375, -0.939291, -0.921466, -0.819738, -0.819730, -0.784929, -0.953246, -0.921836, -0.936411]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[zeros] 
OP: [1mTanh[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[mixed] 
OP: [1mTanh[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.758820, -0.269689, 0.860528, 0.995489, -0.436830, -0.436804, 0.996395, 0.911254, -0.734739, 0.795090]
├─ TT  [0:5]: [0.758820, -0.269689, 0.860528, 0.995489, -0.436830, -0.436804, 0.996395, 0.911254, -0.734739, 0.795090]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[small] 
OP: [1mTanh[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[large] 
OP: [1mTanh[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ TT  [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[minimum_input] 
OP: [1mTanh[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.459529]
├─ TT  [0:5]: [0.459529]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[positive] 
OP: [1mExp[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [3.953258, 7.033710, 5.651912, 4.946393, 3.177258, 3.177182, 2.880845, 6.463533, 4.958558, 5.518315]
├─ TT  [0:5]: [3.953258, 7.033710, 5.651912, 4.946393, 3.177258, 3.177182, 2.880845, 6.463533, 4.958558, 5.518315]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[negative] 
OP: [1mExp[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.252956, 0.142172, 0.176931, 0.202168, 0.314737, 0.314744, 0.347120, 0.154714, 0.201672, 0.181215]
├─ TT  [0:5]: [0.252956, 0.142172, 0.176931, 0.202168, 0.314737, 0.314744, 0.347120, 0.154714, 0.201672, 0.181215]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[zeros] 
OP: [1mExp[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ TT  [0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[mixed] 
OP: [1mExp[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2.700476, 0.758412, 3.652373, 21.032307, 0.626061, 0.626082, 23.533514, 4.640720, 0.391039, 2.959795]
├─ TT  [0:5]: [2.700476, 0.758412, 3.652373, 21.032307, 0.626061, 0.626082, 23.533514, 4.640720, 0.391039, 2.959795]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[small] 
OP: [1mExp[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.000000, 1.000001, 1.000001, 1.000001, 1.000000, 1.000000, 1.000000, 1.000001, 1.000001, 1.000001]
├─ TT  [0:5]: [1.000000, 1.000001, 1.000001, 1.000001, 1.000000, 1.000000, 1.000000, 1.000001, 1.000001, 1.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[minimum_input] 
OP: [1mExp[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.643313]
├─ TT  [0:5]: [1.643313]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sqrt[positive] 
OP: [1mSqrt[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.172408, 1.396680, 1.316052, 1.264381, 1.075183, 1.075172, 1.028632, 1.366081, 1.265352, 1.306933]
├─ TT  [0:5]: [1.172408, 1.396680, 1.316052, 1.264381, 1.075183, 1.075172, 1.028632, 1.366081, 1.265352, 1.306933]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sqrt[zeros] 
OP: [1mSqrt[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sqrt[small] 
OP: [1mSqrt[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000612, 0.000975, 0.000856, 0.000774, 0.000395, 0.000395, 0.000241, 0.000931, 0.000775, 0.000841]
├─ TT  [0:5]: [0.000612, 0.000975, 0.000856, 0.000774, 0.000395, 0.000395, 0.000241, 0.000931, 0.000775, 0.000841]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sqrt[large] 
OP: [1mSqrt[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,4,8]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [611.996826, 975.045776, 855.566467, 773.730225, 394.991943, 394.961395, 241.005417, 930.685852, 775.316040, 841.470520]
├─ TT  [0:5]: [611.996826, 975.045776, 855.566467, 773.730225, 394.991943, 394.961395, 241.005417, 930.685852, 775.316040, 841.470520]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sqrt[minimum_input] 
OP: [1mSqrt[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.704780]
├─ TT  [0:5]: [0.704780]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[positive] 
OP: [1mAdd[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.280935, 1.542696, 1.140924, 1.802197, 1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2.655475, 3.493410, 2.872918, 3.400856, 2.230569, 3.142881, 2.830328, 3.064892, 2.606637, 3.523534]
├─ TT  [0:5]: [2.655475, 3.493410, 2.872918, 3.400856, 2.230569, 3.142881, 2.830328, 3.064892, 2.606637, 3.523534]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[negative] 
OP: [1mAdd[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.280935, -1.542696, -1.140924, -1.802197, -1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-2.655475, -3.493410, -2.872918, -3.400856, -2.230569, -3.142881, -2.830328, -3.064892, -2.606637, -3.523534]
├─ TT  [0:5]: [-2.655475, -3.493410, -2.872918, -3.400856, -2.230569, -3.142881, -2.830328, -3.064892, -2.606637, -3.523534]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[zeros] 
OP: [1mAdd[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[mixed] 
OP: [1mAdd[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [1.625052, 2.712480, -0.144020, 2.007066, 0.723272]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2.618480, 2.435951, 1.151357, 5.053125, 0.254965, -1.758513, 3.881217, 4.610943, -1.010601, 4.214407]
├─ TT  [0:5]: [2.618480, 2.435951, 1.151357, 5.053125, 0.254965, -1.758513, 3.881217, 4.610943, -1.010601, 4.214407]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[small] 
OP: [1mAdd[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002]
├─ TT  [0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[large] 
OP: [1mAdd[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [280934.500000, 542696.062500, 140924.234375, 802197.000000, 74550.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [655474.625000, 1493410.375000, 872918.187500, 1400855.500000, 230569.281250, 1142881.500000, 830328.375000, 1064891.750000, 606637.125000, 1523534.000000]
├─ TT  [0:5]: [655474.625000, 1493410.375000, 872918.187500, 1400855.500000, 230569.281250, 1142881.500000, 830328.375000, 1064891.750000, 606637.125000, 1523534.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[minimum_input] 
OP: [1mAdd[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1] B=[1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.358450]
├─ TT  [0:5]: [0.358450]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[positive] 
OP: [1mSub[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.280935, 1.542696, 1.140924, 1.802197, 1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.093606, 0.408018, 0.591070, -0.203538, 0.081468, -0.830892, -0.714161, 0.667460, 0.595593, -0.107389]
├─ TT  [0:5]: [0.093606, 0.408018, 0.591070, -0.203538, 0.081468, -0.830892, -0.714161, 0.667460, 0.595593, -0.107389]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[negative] 
OP: [1mSub[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.280935, -1.542696, -1.140924, -1.802197, -1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.093606, -0.408018, -0.591070, 0.203538, -0.081468, 0.830892, 0.714161, -0.667460, -0.595593, 0.107389]
├─ TT  [0:5]: [-0.093606, -0.408018, -0.591070, 0.203538, -0.081468, 0.830892, 0.714161, -0.667460, -0.595593, 0.107389]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[zeros] 
OP: [1mSub[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[mixed] 
OP: [1mSub[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [1.625052, 2.712480, -0.144020, 2.007066, 0.723272]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.631623, -2.989009, 1.439397, 1.038994, -1.191579, 0.821966, 2.435634, -1.541204, -0.867297, -2.044167]
├─ TT  [0:5]: [-0.631623, -2.989009, 1.439397, 1.038994, -1.191579, 0.821966, 2.435634, -1.541204, -0.867297, -2.044167]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[small] 
OP: [1mSub[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000001, -0.000000, 0.000000, -0.000001, -0.000001, 0.000001, 0.000001, -0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000001, -0.000000, 0.000000, -0.000001, -0.000001, 0.000001, 0.000001, -0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[large] 
OP: [1mSub[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [280934.500000, 542696.062500, 140924.234375, 802197.000000, 74550.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [93605.625000, 408018.250000, 591069.687500, -203538.500000, 81468.000000, -830892.437500, -714161.125000, 667460.437500, 595592.875000, -107388.812500]
├─ TT  [0:5]: [93605.625000, 408018.250000, 591069.687500, -203538.500000, 81468.000000, -830892.437500, -714161.125000, 667460.437500, 595592.875000, -107388.812500]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[minimum_input] 
OP: [1mSub[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1] B=[1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.634978]
├─ TT  [0:5]: [0.634978]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[positive] 
OP: [1mMul[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.280935, 1.542696, 1.140924, 1.802197, 1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.760696, 3.009359, 1.976074, 2.881098, 1.242200, 2.296830, 1.875183, 2.237015, 1.609957, 3.100940]
├─ TT  [0:5]: [1.760696, 3.009359, 1.976074, 2.881098, 1.242200, 2.296830, 1.875183, 2.237015, 1.609957, 3.100940]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[negative] 
OP: [1mMul[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.280935, -1.542696, -1.140924, -1.802197, -1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.760696, 3.009359, 1.976074, 2.881098, 1.242200, 2.296830, 1.875183, 2.237015, 1.609957, 3.100940]
├─ TT  [0:5]: [1.760696, 3.009359, 1.976074, 2.881098, 1.242200, 2.296830, 1.875183, 2.237015, 1.609957, 3.100940]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[zeros] 
OP: [1mMul[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[mixed] 
OP: [1mMul[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [1.625052, 2.712480, -0.144020, 2.007066, 0.723272]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.614372, -0.750078, -0.186561, 6.113642, -0.338713, 0.604185, 2.282882, 4.721371, 0.067278, 3.395653]
├─ TT  [0:5]: [1.614372, -0.750078, -0.186561, 6.113642, -0.338713, 0.604185, 2.282882, 4.721371, 0.067278, 3.395653]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[small] 
OP: [1mMul[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[large] 
OP: [1mMul[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [280934.500000, 542696.062500, 140924.234375, 802197.000000, 74550.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [105221242880.000000, 515948904448.000000, 103155687424.000000, 480242040832.000000, 11631289344.000000, 153948946432.000000, 44854763520.000000, 172122783744.000000, 3319427584.000000, 577405911040.000000]
├─ TT  [0:5]: [105221242880.000000, 515948904448.000000, 103155687424.000000, 480242040832.000000, 11631289344.000000, 153948946432.000000, 44854763520.000000, 172122783744.000000, 3319427584.000000, 577405911040.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[minimum_input] 
OP: [1mMul[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1] B=[1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.068678]
├─ TT  [0:5]: [-0.068678]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[positive] 
OP: [1mDiv[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.280935, 1.542696, 1.140924, 1.802197, 1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.073076, 1.264484, 1.518062, 0.887061, 1.075816, 0.581812, 0.597030, 1.556813, 1.592322, 0.940848]
├─ TT  [0:5]: [1.073076, 1.264484, 1.518062, 0.887061, 1.075816, 0.581812, 0.597030, 1.556813, 1.592322, 0.940848]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[negative] 
OP: [1mDiv[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.280935, -1.542696, -1.140924, -1.802197, -1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.073076, 1.264484, 1.518062, 0.887061, 1.075816, 0.581812, 0.597030, 1.556813, 1.592322, 0.940848]
├─ TT  [0:5]: [1.073076, 1.264484, 1.518062, 0.887061, 1.075816, 0.581812, 0.597030, 1.556813, 1.592322, 0.940848]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[mixed] 
OP: [1mDiv[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [1.625052, 2.712480, -0.144020, 2.007066, 0.723272]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.611321, -0.101947, -8.994411, 1.517668, -0.647484, 0.362936, 4.369762, 0.498970, 13.104279, 0.346763]
├─ TT  [0:5]: [0.611321, -0.101947, -8.994411, 1.517668, -0.647484, 0.362936, 4.369762, 0.498970, 13.104279, 0.346763]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[small] 
OP: [1mDiv[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.333194, 1.751836, 5.194237, 0.746274, 2.092788, 0.158067, 0.075214, 4.358871, 108.855896, 0.868309]
├─ TT  [0:5]: [1.333194, 1.751836, 5.194237, 0.746274, 2.092788, 0.158067, 0.075214, 4.358871, 108.855896, 0.868309]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[large] 
OP: [1mDiv[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [280934.500000, 542696.062500, 140924.234375, 802197.000000, 74550.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.333194, 1.751836, 5.194237, 0.746274, 2.092788, 0.158067, 0.075214, 4.358871, 108.855896, 0.868309]
├─ TT  [0:5]: [1.333194, 1.751836, 5.194237, 0.746274, 2.092788, 0.158067, 0.075214, 4.358871, 108.855896, 0.868309]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[minimum_input] 
OP: [1mDiv[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1] B=[1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-3.592498]
├─ TT  [0:5]: [-3.592498]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[positive] 
OP: [1mPow[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [0.013904, 1.021495, 0.834822, 0.444216, 0.239731]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.004433, 1.978934, 1.581772, 1.231719, 1.035368, 1.102833, 1.112347, 1.496737, 1.629690, 2.122818]
├─ TT  [0:5]: [1.004433, 1.978934, 1.581772, 1.231719, 1.035368, 1.102833, 1.112347, 1.496737, 1.629690, 2.122818]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[zeros] 
OP: [1mPow[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.749080, 1.901429, 1.463988, 1.197317, 0.312037]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000006, 0.000000, 0.000000, 0.000000, 0.006542, 0.006548, 0.153755, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000006, 0.000000, 0.000000, 0.000000, 0.006542, 0.006548, 0.153755, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[small] 
OP: [1mPow[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.013904, 1.021495, 0.834822, 0.444216, 0.239731]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.816720, 0.000001, 0.000008, 0.001843, 0.026288, 0.000035, 0.000000, 0.000129, 0.000000, 0.000000]
├─ TT  [0:5]: [0.816720, 0.000001, 0.000008, 0.001843, 0.026288, 0.000035, 0.000000, 0.000129, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[large] 
OP: [1mPow[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,4,8] B=[2,4,8]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [0.013904, 1.021495, 0.834822, 0.444216, 0.239731]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.195351, 1278044.500000, 78671.937500, 368.394592, 17.577778, 3210.294678, 964150720.000000, 6888.165039, 991146.312500, 168047200.000000]
├─ TT  [0:5]: [1.195351, 1278044.500000, 78671.937500, 368.394592, 17.577778, 3210.294678, 964150720.000000, 6888.165039, 991146.312500, 168047200.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[minimum_input] 
OP: [1mPow[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1] B=[1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [1.463988]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.359008]
├─ TT  [0:5]: [0.359008]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[positive] 
OP: [1mMatMul[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,4,8] B=[2,8,3]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.280935, 1.542696, 1.140924, 1.802197, 1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [18.466875, 16.769924, 17.839781, 18.253908, 16.421791, 17.597250, 16.957441, 15.162651, 16.160816, 17.636454]
├─ TT  [0:5]: [18.466875, 16.769924, 17.839781, 18.253908, 16.421791, 17.597250, 16.957441, 15.162651, 16.160816, 17.636454]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[negative] 
OP: [1mMatMul[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,4,8] B=[2,8,3]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.280935, -1.542696, -1.140924, -1.802197, -1.074551]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [18.466875, 16.769924, 17.839781, 18.253908, 16.421791, 17.597250, 16.957441, 15.162651, 16.160816, 17.636454]
├─ TT  [0:5]: [18.466875, 16.769924, 17.839781, 18.253908, 16.421791, 17.597250, 16.957441, 15.162651, 16.160816, 17.636454]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[zeros] 
OP: [1mMatMul[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,4,8] B=[2,8,3]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[mixed] 
OP: [1mMatMul[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,4,8] B=[2,8,3]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [1.625052, 2.712480, -0.144020, 2.007066, 0.723272]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [21.102942, -9.458479, 0.609675, 3.294251, 3.176115, -0.534850, -6.617993, -2.738224, -6.908641, -5.801263]
├─ TT  [0:5]: [21.102942, -9.458479, 0.609675, 3.294251, 3.176115, -0.534850, -6.617993, -2.738224, -6.908641, -5.801263]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[small] 
OP: [1mMatMul[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,4,8] B=[2,8,3]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[large] 
OP: [1mMatMul[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,4,8] B=[2,8,3]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [280934.500000, 542696.062500, 140924.234375, 802197.000000, 74550.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2360636080128.000000, 1624613453824.000000, 2154956193792.000000, 2330155548672.000000, 1458966495232.000000, 2094912241664.000000, 1781355380736.000000, 947493863424.000000, 1406146838528.000000, 2050307915776.000000]
├─ TT  [0:5]: [2360636080128.000000, 1624613453824.000000, 2154956193792.000000, 2330155548672.000000, 1458966495232.000000, 2094912241664.000000, 1781355380736.000000, 947493863424.000000, 1406146838528.000000, 2050307915776.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[minimum_input] 
OP: [1mMatMul[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1,1] B=[1,1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.068678]
├─ TT  [0:5]: [-0.068678]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul_broadcast 
OP: [1mMatMul_broadcast[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[1,4,8] B=[2,8,3]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.065052, 1.948886, 1.965632, 1.808397, 1.304614]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [19.131413, 17.941816, 18.363554, 18.177942, 17.766687, 18.335014, 17.383881, 16.848135, 16.905920, 18.042988]
├─ TT  [0:5]: [19.131413, 17.941816, 18.363554, 18.177942, 17.766687, 18.335014, 17.383881, 16.848135, 16.905920, 18.042988]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[positive] 
OP: [1mEinsum[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[1,4,2,3] B=[1,2,3,3,3]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [7.594700, 9.595114, 7.356866, 7.505783, 6.601958, 7.356594, 7.697854, 5.849869, 7.791928, 6.739999]
├─ TT  [0:5]: [7.594700, 9.595114, 7.356866, 7.505783, 6.601958, 7.356594, 7.697854, 5.849869, 7.791928, 6.739999]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[negative] 
OP: [1mEinsum[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[1,4,2,3] B=[1,2,3,3,3]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [7.594700, 9.595114, 7.356866, 7.505783, 6.601958, 7.356594, 7.697854, 5.849869, 7.791928, 6.739999]
├─ TT  [0:5]: [7.594700, 9.595114, 7.356866, 7.505783, 6.601958, 7.356594, 7.697854, 5.849869, 7.791928, 6.739999]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[zeros] 
OP: [1mEinsum[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[1,4,2,3] B=[1,2,3,3,3]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[mixed] 
OP: [1mEinsum[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[1,4,2,3] B=[1,2,3,3,3]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.796252, -1.014613, -5.442150, -1.233995, -1.302975, 2.893753, -0.414150, -1.295807, 0.718028, -1.292981]
├─ TT  [0:5]: [-0.796252, -1.014613, -5.442150, -1.233995, -1.302975, 2.893753, -0.414150, -1.295807, 0.718028, -1.292981]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[small] 
OP: [1mEinsum[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[1,4,2,3] B=[1,2,3,3,3]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[large] 
OP: [1mEinsum[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[1,4,2,3] B=[1,2,3,3,3]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1098107781120.000000, 1877737078784.000000, 1032766160896.000000, 967164362752.000000, 542911889408.000000, 1048594219008.000000, 1046197829632.000000, 315203977216.000000, 1204866252800.000000, 678629736448.000000]
├─ TT  [0:5]: [1098107781120.000000, 1877737078784.000000, 1032766160896.000000, 967164362752.000000, 542911889408.000000, 1048594219008.000000, 1046197829632.000000, 315203977216.000000, 1204866252800.000000, 678629736448.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[minimum_input] 
OP: [1mEinsum[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1,1,1,1] B=[1,1,1,1,1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.068678]
├─ TT  [0:5]: [-0.068678]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[positive] 
OP: [1mCdist[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: x1=[5,4] x2=[3,4]
├─ INPUT x1[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT x2[0:5]: [1.611853, 1.139494, 1.292145, 1.366362, 1.456070]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.980727, 0.569686, 1.031479, 0.716000, 0.793606, 0.994123, 0.872589, 0.516426, 1.192208, 0.315611]
├─ TT  [0:5]: [0.980727, 0.569686, 1.031479, 0.716000, 0.793606, 0.994123, 0.872589, 0.516426, 1.192208, 0.315611]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[negative] 
OP: [1mCdist[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: x1=[5,4] x2=[3,4]
├─ INPUT x1[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT x2[0:5]: [-1.611853, -1.139494, -1.292145, -1.366362, -1.456070]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.980727, 0.569686, 1.031479, 0.716000, 0.793606, 0.994123, 0.872589, 0.516426, 1.192208, 0.315611]
├─ TT  [0:5]: [0.980727, 0.569686, 1.031479, 0.716000, 0.793606, 0.994123, 0.872589, 0.516426, 1.192208, 0.315611]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[zeros] 
OP: [1mCdist[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: x1=[5,4] x2=[3,4]
├─ INPUT x1[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT x2[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[mixed] 
OP: [1mCdist[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: x1=[5,4] x2=[3,4]
├─ INPUT x1[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT x2[0:5]: [2.931298, -0.451553, 0.135056, -2.849496, -1.088766]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [6.315845, 4.773931, 3.404184, 6.318308, 5.593852, 4.927847, 4.706023, 2.343231, 4.941849, 5.762562]
├─ TT  [0:5]: [6.315845, 4.773931, 3.404184, 6.318308, 5.593852, 4.927847, 4.706023, 2.343231, 4.941849, 5.762562]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[small] 
OP: [1mCdist[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: x1=[5,4] x2=[3,4]
├─ INPUT x1[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT x2[0:5]: [0.000001, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000000]
├─ TT  [0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[large] 
OP: [1mCdist[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: x1=[5,4] x2=[3,4]
├─ INPUT x1[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT x2[0:5]: [611852.875000, 139493.875000, 292144.656250, 366361.843750, 456069.968750]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [980726.812500, 569686.062500, 1031479.062500, 716000.000000, 793606.250000, 994123.062500, 872589.375000, 516426.250000, 1192207.875000, 315610.656250]
├─ TT  [0:5]: [980726.812500, 569686.062500, 1031479.062500, 716000.000000, 793606.250000, 994123.062500, 872589.375000, 516426.250000, 1192207.875000, 315610.656250]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[minimum_input] 
OP: [1mCdist[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: x1=[1,1] x2=[1,1]
├─ INPUT x1[0:5]: [0.496714]
├─ INPUT x2[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.634978]
├─ TT  [0:5]: [0.634978]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist_l1 PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[positive] 
OP: [1mTile[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3]
├─ INPUT data[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.374540, 1.950714, 1.731994, 1.374540, 1.950714, 1.731994, 1.598659]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.374540, 1.950714, 1.731994, 1.374540, 1.950714, 1.731994, 1.598659]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[negative] 
OP: [1mTile[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3]
├─ INPUT data[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.374540, -1.950714, -1.731994, -1.374540, -1.950714, -1.731994, -1.598659]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.374540, -1.950714, -1.731994, -1.374540, -1.950714, -1.731994, -1.598659]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[zeros] 
OP: [1mTile[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3]
├─ INPUT data[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[mixed] 
OP: [1mTile[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3]
├─ INPUT data[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 0.993428, -0.276529, 1.295377, 0.993428, -0.276529, 1.295377, 3.046060]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 0.993428, -0.276529, 1.295377, 0.993428, -0.276529, 1.295377, 3.046060]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[small] 
OP: [1mTile[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3]
├─ INPUT data[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[large] 
OP: [1mTile[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3]
├─ INPUT data[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 374540.125000, 950714.312500, 731993.937500, 374540.125000, 950714.312500, 731993.937500, 598658.500000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 374540.125000, 950714.312500, 731993.937500, 374540.125000, 950714.312500, 731993.937500, 598658.500000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[minimum_input] 
OP: [1mTile[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT data[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714, 0.496714, 0.496714]
├─ TT  [0:5]: [0.496714, 0.496714, 0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[positive] 
OP: [1mConcat[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,3,4] B=[2,5,4]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[negative] 
OP: [1mConcat[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,3,4] B=[2,5,4]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[zeros] 
OP: [1mConcat[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,3,4] B=[2,5,4]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[mixed] 
OP: [1mConcat[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,3,4] B=[2,5,4]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[small] 
OP: [1mConcat[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,3,4] B=[2,5,4]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[large] 
OP: [1mConcat[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,3,4] B=[2,5,4]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[minimum_input] 
OP: [1mConcat[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1,1] B=[1,1]
├─ INPUT A[0:5]: [0.496714]
├─ INPUT B[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714, -0.138264]
├─ TT  [0:5]: [0.496714, -0.138264]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[positive] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [1,8,4,4]
├─ INPUT x[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.221979, 1.876112, 1.079660, 0.594130, -1.017708, -1.017795, -1.374330, 1.568273, 0.603075, 0.992552]
├─ TT  [0:5]: [-0.221979, 1.876112, 1.079660, 0.594130, -1.017708, -1.017795, -1.374330, 1.568273, 0.603075, 0.992552]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[negative] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [1,8,4,4]
├─ INPUT x[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.221979, -1.876112, -1.079660, -0.594130, 1.017708, 1.017795, 1.374330, -1.568273, -0.603075, -0.992552]
├─ TT  [0:5]: [0.221979, -1.876112, -1.079660, -0.594130, 1.017708, 1.017795, 1.374330, -1.568273, -0.603075, -0.992552]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[zeros] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [1,8,4,4]
├─ INPUT x[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[mixed] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [1,8,4,4]
├─ INPUT x[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.680941, -0.001028, 0.843088, 1.783207, -0.104013, -0.103995, 1.843548, 0.971696, -0.356748, 0.730180]
├─ TT  [0:5]: [0.680941, -0.001028, 0.843088, 1.783207, -0.104013, -0.103995, 1.843548, 0.971696, -0.356748, 0.730180]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[small] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [1,8,4,4]
├─ INPUT x[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.000019, 0.000163, 0.000094, 0.000052, -0.000088, -0.000088, -0.000119, 0.000136, 0.000052, 0.000086]
├─ TT  [0:5]: [-0.000019, 0.000163, 0.000094, 0.000052, -0.000088, -0.000088, -0.000119, 0.000136, 0.000052, 0.000086]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[large] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [1,8,4,4]
├─ INPUT x[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.221994, 1.876237, 1.079732, 0.594169, -1.017775, -1.017862, -1.374421, 1.568378, 0.603115, 0.992618]
├─ TT  [0:5]: [-0.221994, 1.876237, 1.079732, 0.594169, -1.017775, -1.017862, -1.374421, 1.568378, 0.603115, 0.992618]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[minimum_input] 
OP: [1mGroupNorm[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,4,1,1]
├─ INPUT x[0:5]: [0.496714, -0.138264, 0.647689, 1.523030]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[positive] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,8,16]
├─ INPUT x[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.305196, 1.443177, 0.779481, 0.374881, -0.968289, -0.968362, -1.265468, 1.186650, 0.382335, 0.706893]
├─ TT  [0:5]: [-0.305196, 1.443177, 0.779481, 0.374881, -0.968289, -0.968362, -1.265468, 1.186650, 0.382335, 0.706893]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[negative] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,8,16]
├─ INPUT x[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.305196, -1.443177, -0.779481, -0.374881, 0.968289, 0.968362, 1.265468, -1.186650, -0.382335, -0.706893]
├─ TT  [0:5]: [0.305196, -1.443177, -0.779481, -0.374881, 0.968289, 0.968362, 1.265468, -1.186650, -0.382335, -0.706893]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[zeros] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,8,16]
├─ INPUT x[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[mixed] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,8,16]
├─ INPUT x[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.555353, -0.119996, 0.715926, 1.646920, -0.221982, -0.221964, 1.706675, 0.843285, -0.472264, 0.604114]
├─ TT  [0:5]: [0.555353, -0.119996, 0.715926, 1.646920, -0.221982, -0.221964, 1.706675, 0.843285, -0.472264, 0.604114]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[small] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,8,16]
├─ INPUT x[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.000032, 0.000150, 0.000081, 0.000039, -0.000101, -0.000101, -0.000132, 0.000124, 0.000040, 0.000074]
├─ TT  [0:5]: [-0.000032, 0.000150, 0.000081, 0.000039, -0.000101, -0.000101, -0.000132, 0.000124, 0.000040, 0.000074]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[large] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,8,16]
├─ INPUT x[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.305210, 1.443244, 0.779517, 0.374898, -0.968334, -0.968407, -1.265527, 1.186705, 0.382352, 0.706925]
├─ TT  [0:5]: [-0.305210, 1.443244, 0.779517, 0.374898, -0.968334, -0.968407, -1.265527, 1.186705, 0.382352, 0.706925]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[minimum_input] 
OP: [1mLayerNorm[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,1,4]
├─ INPUT x[0:5]: [0.496714, -0.138264, 0.647689, 1.523030]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.228693, -1.299775, 0.025971, 1.502497]
├─ TT  [0:5]: [-0.228693, -1.299775, 0.025971, 1.502497]
└─ RESULT: [92m✓ PASS[0m
PASSED

=============================================== slowest durations ================================================
0.18s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[minimum_input]
0.17s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[mixed]
0.15s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[negative]
0.11s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[mixed]
0.10s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[positive]
0.10s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[zeros]
0.08s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[positive]
0.08s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[large]
0.07s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[positive]
0.06s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[mixed]
0.06s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[small]
0.06s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[minimum_input]
0.06s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[zeros]
0.06s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[mixed]
0.05s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[small]
0.05s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[negative]
0.04s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[small]
0.04s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[minimum_input]
0.04s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[large]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[large]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[small]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[positive]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[minimum_input]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[zeros]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[positive]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[zeros]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[large]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_glu[small]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[negative]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[large]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul_broadcast
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[small]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[minimum_input]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[large]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[minimum_input]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[mixed]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[small]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sqrt[zeros]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[large]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[negative]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[mixed]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[large]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[positive]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[positive]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[mixed]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[mixed]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_inverse_sigmoid[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sigmoid[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[minimum_input]
0.01s teardown workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[mixed]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_relu[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[small]
0.01s teardown workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_log[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[positive]
0.01s setup    workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[large]
0.01s setup    workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cdist[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[mixed]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sin[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_softmax[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[small]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_identity[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_pow[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_add[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[minimum_input]
0.01s teardown workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tile[minimum_input]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_mul[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_cos[negative]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_concat[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_div[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_sub[minimum_input]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_layernorm[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_groupnorm[zeros]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_matmul[mixed]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_einsum[minimum_input]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_tanh[minimum_input]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_functional_ops_unit.py::test_exp[zeros]

(378 durations < 0.005s hidden.  Use -vv to show these durations.)
============================================== 158 passed in 4.26s ===============================================

═════════════════════════════════════════════════════════════════
FUNCTIONAL OPS UNIT TEST SUMMARY
═════════════════════════════════════════════════════════════════
OP                      SHAPE     NUMERICAL   RESULT
Relu                    7/7        7/7           [92m✓ PASS[0m
Sigmoid                 7/7        7/7           [92m✓ PASS[0m
Softmax                 7/7        7/7           [92m✓ PASS[0m
Log                     4/4        4/4           [92m✓ PASS[0m
Cos                     7/7        7/7           [92m✓ PASS[0m
Sin                     7/7        7/7           [92m✓ PASS[0m
Identity                7/7        7/7           [92m✓ PASS[0m
InverseSigmoid          4/4        4/4           [92m✓ PASS[0m
Glu                     7/7        7/7           [92m✓ PASS[0m
Tanh                    7/7        7/7           [92m✓ PASS[0m
Exp                     6/6        6/6           [92m✓ PASS[0m
Sqrt                    5/5        5/5           [92m✓ PASS[0m
Add                     7/7        7/7           [92m✓ PASS[0m
Sub                     7/7        7/7           [92m✓ PASS[0m
Mul                     7/7        7/7           [92m✓ PASS[0m
Div                     6/6        6/6           [92m✓ PASS[0m
Pow                     5/5        5/5           [92m✓ PASS[0m
MatMul                  7/7        7/7           [92m✓ PASS[0m
MatMul_broadcast        1/1        1/1           [92m✓ PASS[0m
Einsum                  7/7        7/7           [92m✓ PASS[0m
Cdist                   7/7        7/7           [92m✓ PASS[0m
Tile                    7/7        7/7           [92m✓ PASS[0m
Concat                  7/7        7/7           [92m✓ PASS[0m
GroupNorm               7/7        7/7           [92m✓ PASS[0m
LayerNorm               7/7        7/7           [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
OVERALL: [92m✓ ALL PASS[0m
═════════════════════════════════════════════════════════════════
