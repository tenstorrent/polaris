
═════════════════════════════════════════════════════════════════
TENSOR OPS UNIT TEST SUITE — TTSim tensor operations
═════════════════════════════════════════════════════════════════

================================ test session starts =================================
platform win32 -- Python 3.13.2, pytest-8.3.4, pluggy-1.6.0 -- C:\Users\Akandala\AppData\Local\miniforge3\envs\polarisdev\python.exe
cachedir: .pytest_cache
metadata: {'Python': '3.13.2', 'Platform': 'Windows-11-10.0.26100-SP0', 'Packages': {'pytest': '8.3.4', 'pluggy': '1.6.0'}, 'Plugins': {'hydra-core': '1.3.2', 'cov': '6.3.0', 'json-report': '1.5.0', 'metadata': '3.1.1', 'mock': '3.15.1', 'xdist': '3.8.0'}}
rootdir: C:\Users\Akandala\Desktop\Projects\2026\Tenstorrent\polaris
configfile: pyproject.toml
plugins: hydra-core-1.3.2, cov-6.3.0, json-report-1.5.0, metadata-3.1.1, mock-3.15.1, xdist-3.8.0
collecting ... collected 145 items

workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[positive] 
OP: [1mview[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4]→[6,4]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[negative] 
OP: [1mview[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4]→[6,4]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[zeros] 
OP: [1mview[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4]→[6,4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[mixed] 
OP: [1mview[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4]→[6,4]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[small] 
OP: [1mview[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4]→[6,4]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[large] 
OP: [1mview[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4]→[6,4]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[minimum_input] 
OP: [1mview[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]→[1,1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view_infer_dim 
OP: [1mview(-1,4)[0m
├─ EDGE CASE: [93minfer_dim[0m (Infer dimension with -1)
├─ INPUT: [2,3,4]→[-1,4]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_reshape_alias PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose[positive] 
OP: [1mtranspose[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4] dims=(0,2)
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.832443, 1.156019, 1.304242, 1.601115, 1.611853, 1.950714, 1.212339, 1.155995, 1.524756]
├─ TT  [0:5]: [1.374540, 1.832443, 1.156019, 1.304242, 1.601115, 1.611853, 1.950714, 1.212339, 1.155995, 1.524756]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose[negative] 
OP: [1mtranspose[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4] dims=(0,2)
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.832443, -1.156019, -1.304242, -1.601115, -1.611853, -1.950714, -1.212339, -1.155995, -1.524756]
├─ TT  [0:5]: [-1.374540, -1.832443, -1.156019, -1.304242, -1.601115, -1.611853, -1.950714, -1.212339, -1.155995, -1.524756]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose[zeros] 
OP: [1mtranspose[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4] dims=(0,2)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose[mixed] 
OP: [1mtranspose[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4] dims=(0,2)
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, 0.483925, -0.468307, -2.025662, -0.938949, 2.931298, -0.276529, -3.826560, -0.468274, 0.628495]
├─ TT  [0:5]: [0.993428, 0.483925, -0.468307, -2.025662, -0.938949, 2.931298, -0.276529, -3.826560, -0.468274, 0.628495]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose[small] 
OP: [1mtranspose[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4] dims=(0,2)
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose[large] 
OP: [1mtranspose[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4] dims=(0,2)
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 832442.625000, 156018.640625, 304242.250000, 601115.000000, 611852.875000, 950714.312500, 212339.125000, 155994.515625, 524756.437500]
├─ TT  [0:5]: [374540.125000, 832442.625000, 156018.640625, 304242.250000, 601115.000000, 611852.875000, 950714.312500, 212339.125000, 155994.515625, 524756.437500]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose[minimum_input] 
OP: [1mtranspose[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,1] dims=(0,1)
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose_negative_dim PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_permute[positive] 
OP: [1mpermute[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4] perm=[2, 0, 1]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.156019, 1.601115, 1.832443, 1.304242, 1.611853, 1.950714, 1.155995, 1.708073, 1.212339]
├─ TT  [0:5]: [1.374540, 1.156019, 1.601115, 1.832443, 1.304242, 1.611853, 1.950714, 1.155995, 1.708073, 1.212339]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_permute[negative] 
OP: [1mpermute[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4] perm=[2, 0, 1]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.156019, -1.601115, -1.832443, -1.304242, -1.611853, -1.950714, -1.155995, -1.708073, -1.212339]
├─ TT  [0:5]: [-1.374540, -1.156019, -1.601115, -1.832443, -1.304242, -1.611853, -1.950714, -1.155995, -1.708073, -1.212339]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_permute[zeros] 
OP: [1mpermute[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4] perm=[2, 0, 1]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_permute[mixed] 
OP: [1mpermute[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4] perm=[2, 0, 1]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.468307, -0.938949, 0.483925, -2.025662, 2.931298, -0.276529, -0.468274, 1.085120, -3.826560]
├─ TT  [0:5]: [0.993428, -0.468307, -0.938949, 0.483925, -2.025662, 2.931298, -0.276529, -0.468274, 1.085120, -3.826560]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_permute[small] 
OP: [1mpermute[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4] perm=[2, 0, 1]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_permute[large] 
OP: [1mpermute[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4] perm=[2, 0, 1]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 156018.640625, 601115.000000, 832442.625000, 304242.250000, 611852.875000, 950714.312500, 155994.515625, 708072.625000, 212339.125000]
├─ TT  [0:5]: [374540.125000, 156018.640625, 601115.000000, 832442.625000, 304242.250000, 611852.875000, 950714.312500, 155994.515625, 708072.625000, 212339.125000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_permute[minimum_input] 
OP: [1mpermute[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,1,1] perm=[2, 0, 1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze[positive] 
OP: [1munsqueeze[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4] dim=1
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze[negative] 
OP: [1munsqueeze[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4] dim=1
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze[zeros] 
OP: [1munsqueeze[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4] dim=1
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze[mixed] 
OP: [1munsqueeze[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4] dim=1
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze[small] 
OP: [1munsqueeze[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4] dim=1
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze[large] 
OP: [1munsqueeze[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4] dim=1
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze[minimum_input] 
OP: [1munsqueeze[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1] dim=0
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_unsqueeze_negative_dim PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze[positive] 
OP: [1msqueeze[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,1,4] dim=1
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze[negative] 
OP: [1msqueeze[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,1,4] dim=1
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze[zeros] 
OP: [1msqueeze[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,1,4] dim=1
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze[mixed] 
OP: [1msqueeze[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,1,4] dim=1
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze[small] 
OP: [1msqueeze[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,1,4] dim=1
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze[large] 
OP: [1msqueeze[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,1,4] dim=1
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze[minimum_input] 
OP: [1msqueeze[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,1] dim=0
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[positive] 
OP: [1mflatten[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4] dims=(1,2)
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[negative] 
OP: [1mflatten[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4] dims=(1,2)
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[zeros] 
OP: [1mflatten[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4] dims=(1,2)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[mixed] 
OP: [1mflatten[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4] dims=(1,2)
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[small] 
OP: [1mflatten[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4] dims=(1,2)
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[large] 
OP: [1mflatten[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4] dims=(1,2)
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[minimum_input] 
OP: [1mflatten[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,1,1] dims=(0,2)
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten_default_end PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat[positive] 
OP: [1mrepeat[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3] sizes=(1, 2)
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.598659]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.598659]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat[negative] 
OP: [1mrepeat[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3] sizes=(1, 2)
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.598659]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.598659]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat[zeros] 
OP: [1mrepeat[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3] sizes=(1, 2)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat[mixed] 
OP: [1mrepeat[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3] sizes=(1, 2)
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.046060]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.046060]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat[small] 
OP: [1mrepeat[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3] sizes=(1, 2)
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat[large] 
OP: [1mrepeat[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3] sizes=(1, 2)
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 598658.500000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 598658.500000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat[minimum_input] 
OP: [1mrepeat[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1] sizes=(3,)
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714, 0.496714, 0.496714]
├─ TT  [0:5]: [0.496714, 0.496714, 0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_add[positive] 
OP: [1m__add__[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2.830610, 3.735890, 2.931668, 3.112893, 2.748433, 2.202445, 2.665628, 3.036700, 2.666167, 3.656958]
├─ TT  [0:5]: [2.830610, 3.735890, 2.931668, 3.112893, 2.748433, 2.202445, 2.665628, 3.036700, 2.666167, 3.656958]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_add[negative] 
OP: [1m__add__[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-2.830610, -3.735890, -2.931668, -3.112893, -2.748433, -2.202445, -2.665628, -3.036700, -2.666167, -3.656958]
├─ TT  [0:5]: [-2.830610, -3.735890, -2.931668, -3.112893, -2.748433, -2.202445, -2.665628, -3.036700, -2.666167, -3.656958]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_add[zeros] 
OP: [1m__add__[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_add[mixed] 
OP: [1m__add__[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.095337, -0.054683, -1.006610, 3.797456, -1.669584, -1.051661, 1.955012, 5.239426, -0.965943, -1.030302]
├─ TT  [0:5]: [-0.095337, -0.054683, -1.006610, 3.797456, -1.669584, -1.051661, 1.955012, 5.239426, -0.965943, -1.030302]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_add[small] 
OP: [1m__add__[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000001, 0.000002, 0.000001, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001, 0.000002]
├─ TT  [0:5]: [0.000001, 0.000002, 0.000001, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001, 0.000002]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_add[large] 
OP: [1m__add__[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [830610.125000, 1735890.250000, 931667.750000, 1112893.000000, 748433.187500, 202444.937500, 665628.437500, 1036700.250000, 666166.625000, 1656958.250000]
├─ TT  [0:5]: [830610.125000, 1735890.250000, 931667.750000, 1112893.000000, 748433.187500, 202444.937500, 665628.437500, 1036700.250000, 666166.625000, 1656958.250000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_add[minimum_input] 
OP: [1m__add__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sub[positive] 
OP: [1m__sub__[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.081530, 0.165538, 0.532320, 0.084424, -0.436396, 0.109544, -0.549461, 0.695652, 0.536063, -0.240813]
├─ TT  [0:5]: [-0.081530, 0.165538, 0.532320, 0.084424, -0.436396, 0.109544, -0.549461, 0.695652, 0.536063, -0.240813]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sub[negative] 
OP: [1m__sub__[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.081530, -0.165538, -0.532320, -0.084424, 0.436396, -0.109544, 0.549461, -0.695652, -0.536063, 0.240813]
├─ TT  [0:5]: [0.081530, -0.165538, -0.532320, -0.084424, 0.436396, -0.109544, 0.549461, -0.695652, -0.536063, 0.240813]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sub[zeros] 
OP: [1m__sub__[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sub[mixed] 
OP: [1m__sub__[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2.082194, -0.498374, 3.597364, 2.294663, 0.732971, 0.115114, 4.361839, -2.169687, -0.911954, 3.200542]
├─ TT  [0:5]: [2.082194, -0.498374, 3.597364, 2.294663, 0.732971, 0.115114, 4.361839, -2.169687, -0.911954, 3.200542]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sub[small] 
OP: [1m__sub__[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.000000, 0.000000, 0.000001, 0.000000, -0.000000, 0.000000, -0.000001, 0.000001, 0.000001, -0.000000]
├─ TT  [0:5]: [-0.000000, 0.000000, 0.000001, 0.000000, -0.000000, 0.000000, -0.000001, 0.000001, 0.000001, -0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sub[large] 
OP: [1m__sub__[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-81529.843750, 165538.312500, 532320.125000, 84424.062500, -436395.937500, 109544.101562, -549461.187500, 695652.000000, 536063.375000, -240812.937500]
├─ TT  [0:5]: [-81529.843750, 165538.312500, 532320.125000, 84424.062500, -436395.937500, 109544.101562, -549461.187500, 695652.000000, 536063.375000, -240812.937500]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sub[minimum_input] 
OP: [1m__sub__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_mul[positive] 
OP: [1m__mul__[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2.001426, 3.482368, 2.077828, 2.420744, 1.840861, 1.209691, 1.700917, 2.184404, 1.705270, 3.328838]
├─ TT  [0:5]: [2.001426, 3.482368, 2.077828, 2.420744, 1.840861, 1.209691, 1.700917, 2.184404, 1.705270, 3.328838]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_mul[negative] 
OP: [1m__mul__[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [2.001426, 3.482368, 2.077828, 2.420744, 1.840861, 1.209691, 1.700917, 2.184404, 1.705270, 3.328838]
├─ TT  [0:5]: [2.001426, 3.482368, 2.077828, 2.420744, 1.840861, 1.209691, 1.700917, 2.184404, 1.705270, 3.328838]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_mul[zeros] 
OP: [1m__mul__[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_mul[mixed] 
OP: [1m__mul__[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.081610, -0.061347, -2.981941, 2.288797, 0.562566, 0.273185, -3.800891, 5.686010, 0.025346, -2.295487]
├─ TT  [0:5]: [-1.081610, -0.061347, -2.981941, 2.288797, 0.562566, 0.273185, -3.800891, 5.686010, 0.025346, -2.295487]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_mul[small] 
OP: [1m__mul__[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_mul[large] 
OP: [1m__mul__[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [170816503808.000000, 746478043136.000000, 146159992832.000000, 307850805248.000000, 92427714560.000000, 7246009856.000000, 35288399872.000000, 147703922688.000000, 39103488000.000000, 671879921664.000000]
├─ TT  [0:5]: [170816503808.000000, 746478043136.000000, 146159992832.000000, 307850805248.000000, 92427714560.000000, 7246009856.000000, 35288399872.000000, 147703922688.000000, 39103488000.000000, 671879921664.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_mul[minimum_input] 
OP: [1m__mul__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_div[positive] 
OP: [1m__div__[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT B[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.944007, 1.092729, 1.443721, 1.055754, 0.725953, 1.104682, 0.658199, 1.594308, 1.503322, 0.876436]
├─ TT  [0:5]: [0.944007, 1.092729, 1.443721, 1.055754, 0.725953, 1.104682, 0.658199, 1.594308, 1.503322, 0.876436]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_div[negative] 
OP: [1m__div__[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT B[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.944007, 1.092729, 1.443721, 1.055754, 0.725953, 1.104682, 0.658199, 1.594308, 1.503322, 0.876436]
├─ TT  [0:5]: [0.944007, 1.092729, 1.443721, 1.055754, 0.725953, 1.104682, 0.658199, 1.594308, 1.503322, 0.876436]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_div[mixed] 
OP: [1m__div__[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT B[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-0.912435, -1.246494, -0.562721, 4.053867, 0.389841, 0.802681, -2.624556, 0.414319, 34.783028, -0.512957]
├─ TT  [0:5]: [-0.912435, -1.246494, -0.562721, 4.053867, 0.389841, 0.802681, -2.624556, 0.414319, 34.783028, -0.512957]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_div[small] 
OP: [1m__div__[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT B[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.821234, 1.210830, 3.665949, 1.164174, 0.263361, 0.000000, 0.095604, 5.079494, 0.000001, 0.746215]
├─ TT  [0:5]: [0.821234, 1.210830, 3.665949, 1.164174, 0.263361, 0.000000, 0.095604, 5.079494, 0.000001, 0.746215]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_div[large] 
OP: [1m__div__[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.821234, 1.210830, 3.665949, 1.164174, 0.263361, 3.358302, 0.095604, 5.079493, 9.240588, 0.746215]
├─ TT  [0:5]: [0.821234, 1.210830, 3.665949, 1.164174, 0.263361, 3.358302, 0.095604, 5.079493, 9.240588, 0.746215]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_div[minimum_input] 
OP: [1m__div__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_pow[positive] 
OP: [1m__pow__[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [1.375540, 1.951714, 1.732994, 1.599659, 1.157019]
├─ INPUT B[0:5]: [1.556070, 1.885176, 1.299674, 1.614234, 1.692415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.642382, 3.527653, 2.043423, 2.134754, 1.279965, 1.181969, 1.102985, 2.210790, 1.731724, 2.998470]
├─ TT  [0:5]: [1.642382, 3.527653, 2.043423, 2.134754, 1.279965, 1.181969, 1.102985, 2.210790, 1.731724, 2.998470]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_pow[small] 
OP: [1m__pow__[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [0.001000, 0.001001, 0.001001, 0.001001, 0.001000]
├─ INPUT B[0:5]: [0.100000, 0.100001, 0.100000, 0.100001, 0.100001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.501204, 0.501232, 0.501223, 0.501215, 0.501193, 0.501195, 0.501188, 0.501230, 0.501217, 0.501219]
├─ TT  [0:5]: [0.501204, 0.501232, 0.501223, 0.501215, 0.501193, 0.501195, 0.501188, 0.501230, 0.501217, 0.501219]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_pow[large] 
OP: [1m__pow__[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: A=[2,3,4] B=[2,3,4]
├─ INPUT A[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT B[0:5]: [2.068750, 0.100000, 1.881250, 0.537500, 0.662500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [338980077568.000000, 3.961002, 107796545536.000000, 1274.189209, 2757.271484, 467.286133, 22243.291016, 21.675112, 6139167744.000000, 5318075392.000000]
├─ TT  [0:5]: [338980077568.000000, 3.961002, 107796545536.000000, 1274.189209, 2757.271484, 467.286133, 22243.291016, 21.675112, 6139167744.000000, 5318075392.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_pow[minimum_input] 
OP: [1m__pow__[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: A=[1] B=[1]
├─ INPUT A[0:5]: [0.497714]
├─ INPUT B[0:5]: [0.238264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.846840]
├─ TT  [0:5]: [0.846840]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_neg[positive] 
OP: [1m__neg__[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_neg[negative] 
OP: [1m__neg__[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_neg[zeros] 
OP: [1m__neg__[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_neg[mixed] 
OP: [1m__neg__[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_neg[small] 
OP: [1m__neg__[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_neg[large] 
OP: [1m__neg__[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_neg[minimum_input] 
OP: [1m__neg__[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_matmul[positive] 
OP: [1m__matmul__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_matmul[negative] 
OP: [1m__matmul__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_matmul[zeros] 
OP: [1m__matmul__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_matmul[mixed] 
OP: [1m__matmul__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_matmul[small] 
OP: [1m__matmul__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_matmul[minimum_input] 
OP: [1m__matmul__[0m
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
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_cos[positive] 
OP: [1mcos[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_cos[negative] 
OP: [1mcos[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_cos[zeros] 
OP: [1mcos[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_cos[mixed] 
OP: [1mcos[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_cos[small] 
OP: [1mcos[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_cos[large] 
OP: [1mcos[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_cos[minimum_input] 
OP: [1mcos[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sin[positive] 
OP: [1msin[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sin[negative] 
OP: [1msin[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sin[zeros] 
OP: [1msin[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sin[mixed] 
OP: [1msin[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sin[small] 
OP: [1msin[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sin[large] 
OP: [1msin[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_sin[minimum_input] 
OP: [1msin[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.496714]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: [96m⊘ SKIPPED (sinf does not compute data)[0m)
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_softmax[positive] 
OP: [1msoftmax[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.183146, 0.325857, 0.261841, 0.229156, 0.202388, 0.202384, 0.183507, 0.411721, 0.242808, 0.270218]
├─ TT  [0:5]: [0.183146, 0.325857, 0.261841, 0.229156, 0.202388, 0.202384, 0.183507, 0.411721, 0.242808, 0.270218]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_softmax[negative] 
OP: [1msoftmax[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.326721, 0.183631, 0.228526, 0.261122, 0.278204, 0.278211, 0.306829, 0.136756, 0.228461, 0.205287]
├─ TT  [0:5]: [0.326721, 0.183631, 0.228526, 0.261122, 0.278204, 0.278211, 0.306829, 0.136756, 0.228461, 0.205287]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_softmax[zeros] 
OP: [1msoftmax[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000]
├─ TT  [0:5]: [0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_softmax[mixed] 
OP: [1msoftmax[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.095954, 0.026948, 0.129776, 0.747322, 0.021276, 0.021276, 0.799742, 0.157706, 0.094440, 0.714820]
├─ TT  [0:5]: [0.095954, 0.026948, 0.129776, 0.747322, 0.021276, 0.021276, 0.799742, 0.157706, 0.094440, 0.714820]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_softmax[small] 
OP: [1msoftmax[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000]
├─ TT  [0:5]: [0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_softmax[large] 
OP: [1msoftmax[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4]
├─ INPUT input[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_tensor_softmax[minimum_input] 
OP: [1msoftmax[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1,4]
├─ INPUT input[0:5]: [0.496714, -0.138264, 0.647689, 1.523030]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.182359, 0.096641, 0.212078, 0.508922]
├─ TT  [0:5]: [0.182359, 0.096641, 0.212078, 0.508922]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat[positive] 
OP: [1mcat[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: [2,3,4] + [2,5,4] dim=1
├─ INPUT t0[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT t1[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat[negative] 
OP: [1mcat[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: [2,3,4] + [2,5,4] dim=1
├─ INPUT t0[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT t1[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat[zeros] 
OP: [1mcat[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: [2,3,4] + [2,5,4] dim=1
├─ INPUT t0[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT t1[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat[mixed] 
OP: [1mcat[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: [2,3,4] + [2,5,4] dim=1
├─ INPUT t0[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT t1[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat[small] 
OP: [1mcat[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: [2,3,4] + [2,5,4] dim=1
├─ INPUT t0[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT t1[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat[large] 
OP: [1mcat[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: [2,3,4] + [2,5,4] dim=1
├─ INPUT t0[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT t1[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat[minimum_input] 
OP: [1mcat[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: [1] + [1] dim=0
├─ INPUT t0[0:5]: [0.496714]
├─ INPUT t1[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714, -0.138264]
├─ TT  [0:5]: [0.496714, -0.138264]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_cat_three_tensors PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack[positive] 
OP: [1mstack[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0) — baseline)
├─ INPUT: 3x[2,3,4] dim=1
├─ INPUT t0[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT t1[0:5]: [1.456070, 1.785176, 1.199674, 1.514234, 1.592415]
├─ INPUT t2[0:5]: [1.546710, 1.184855, 1.969585, 1.775133, 1.939499]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack[negative] 
OP: [1mstack[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) — sign handling)
├─ INPUT: 3x[2,3,4] dim=1
├─ INPUT t0[0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019]
├─ INPUT t1[0:5]: [-1.456070, -1.785176, -1.199674, -1.514234, -1.592415]
├─ INPUT t2[0:5]: [-1.546710, -1.184855, -1.969585, -1.775133, -1.939499]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack[zeros] 
OP: [1mstack[0m
├─ EDGE CASE: [93mzeros[0m (All zeros — edge case for division / log)
├─ INPUT: 3x[2,3,4] dim=1
├─ INPUT t0[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT t1[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT t2[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack[mixed] 
OP: [1mstack[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative — real-world distribution)
├─ INPUT: 3x[2,3,4] dim=1
├─ INPUT t0[0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307]
├─ INPUT t1[0:5]: [-1.088766, 0.221845, -2.301987, 0.751396, -1.201277]
├─ INPUT t2[0:5]: [0.687237, -3.526080, 0.648168, -0.770165, -1.353844]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack[small] 
OP: [1mstack[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) — precision near zero)
├─ INPUT: 3x[2,3,4] dim=1
├─ INPUT t0[0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000]
├─ INPUT t1[0:5]: [0.000000, 0.000001, 0.000000, 0.000001, 0.000001]
├─ INPUT t2[0:5]: [0.000001, 0.000000, 0.000001, 0.000001, 0.000001]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack[large] 
OP: [1mstack[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) — overflow handling)
├─ INPUT: 3x[2,3,4] dim=1
├─ INPUT t0[0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625]
├─ INPUT t1[0:5]: [456069.968750, 785176.000000, 199673.781250, 514234.437500, 592414.562500]
├─ INPUT t2[0:5]: [546710.250000, 184854.468750, 969584.625000, 775132.812500, 939498.937500]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack[minimum_input] 
OP: [1mstack[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size (1 element))
├─ INPUT: 2x[1] dim=0
├─ INPUT t0[0:5]: [0.496714]
├─ INPUT t1[0:5]: [-0.138264]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ REF [0:5]: [0.496714, -0.138264]
├─ TT  [0:5]: [0.496714, -0.138264]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack_dim0 PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack_dim_neg1 PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view_incompatible_shape_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view_two_neg1_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_transpose_oob_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_squeeze_oob_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten_invalid_range_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_repeat_wrong_ndim_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack_empty_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_stack_shape_mismatch_raises PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_no_link_module_raises PASSED

================================= slowest durations ==================================
0.04s call     workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_view[positive]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_tensor_ops_unit.py::test_flatten[large]

(433 durations < 0.005s hidden.  Use -vv to show these durations.)
================================ 145 passed in 0.51s =================================

═════════════════════════════════════════════════════════════════
TENSOR OPS UNIT TEST SUMMARY
═════════════════════════════════════════════════════════════════
OP                      SHAPE     NUMERICAL   RESULT
view                    7/7        7/7         [92m✓ PASS[0m
view_infer              1/1        1/1         [92m✓ PASS[0m
reshape                 1/1        1/1         [92m✓ PASS[0m
transpose               8/8        8/8         [92m✓ PASS[0m
permute                 7/7        7/7         [92m✓ PASS[0m
unsqueeze               8/8        8/8         [92m✓ PASS[0m
squeeze                 7/7        7/7         [92m✓ PASS[0m
flatten                 8/8        8/8         [92m✓ PASS[0m
repeat                  7/7        7/7         [92m✓ PASS[0m
__add__                 7/7        7/7         [92m✓ PASS[0m
__sub__                 7/7        7/7         [92m✓ PASS[0m
__mul__                 7/7        7/7         [92m✓ PASS[0m
__div__                 6/6        6/6         [92m✓ PASS[0m
__pow__                 4/4        4/4         [92m✓ PASS[0m
__neg__                 7/7        N/A         [92m✓ PASS[0m
__matmul__              6/6        6/6         [92m✓ PASS[0m
cos                     7/7        N/A         [92m✓ PASS[0m
sin                     7/7        N/A         [92m✓ PASS[0m
softmax                 7/7        7/7         [92m✓ PASS[0m
cat                     8/8        8/8         [92m✓ PASS[0m
stack                   9/9        9/9         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
OVERALL: [92m✓ ALL PASS[0m
═════════════════════════════════════════════════════════════════
