
═════════════════════════════════════════════════════════════════
NN MODULES UNIT TEST SUITE — SimNN
═════════════════════════════════════════════════════════════════

============================================== test session starts ===============================================
platform win32 -- Python 3.13.2, pytest-8.3.4, pluggy-1.6.0 -- C:\Users\Akandala\AppData\Local\miniforge3\envs\polarisdev\python.exe
cachedir: .pytest_cache
metadata: {'Python': '3.13.2', 'Platform': 'Windows-11-10.0.26100-SP0', 'Packages': {'pytest': '8.3.4', 'pluggy': '1.6.0'}, 'Plugins': {'hydra-core': '1.3.2', 'cov': '6.3.0', 'json-report': '1.5.0', 'metadata': '3.1.1', 'mock': '3.15.1', 'xdist': '3.8.0'}}
rootdir: C:\Users\Akandala\Desktop\Projects\2026\Tenstorrent\polaris
configfile: pyproject.toml
plugins: hydra-core-1.3.2, cov-6.3.0, json-report-1.5.0, metadata-3.1.1, mock-3.15.1, xdist-3.8.0
collecting ... collected 29 items

workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_module_tensor_registration 
MODULE: [1mModule[0m
├─ EDGE CASE: [93mtensor_reg[0m (Register SimTensor via __setattr__)
├─ INPUT: N/A
├─ SHAPE: [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_module_op_registration FAILED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_module_submodule_registration 
MODULE: [1mModule[0m
├─ EDGE CASE: [93msubmod_reg[0m (Register sub-Module via __setattr__)
├─ INPUT: N/A
├─ SHAPE: [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_module_link_op2module 
MODULE: [1mModule[0m
├─ EDGE CASE: [93mlink_op2mod[0m (link_op2module propagation)
├─ INPUT: N/A
├─ SHAPE: [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_modulelist_basics 
MODULE: [1mModuleList[0m
├─ EDGE CASE: [93mbasics[0m (len / index / iter / immutability)
├─ INPUT: N/A
├─ SHAPE: [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_modulelist_empty_rejected 
MODULE: [1mModuleList[0m
├─ EDGE CASE: [93mempty_reject[0m (Empty ModuleList assert)
├─ INPUT: N/A
├─ SHAPE: [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[positive] 
MODULE: [1mLinear[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0))
├─ INPUT: [2,6,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [-0.018541, -0.541951, -0.479817, 0.046163, -0.123459, -0.316877, -0.486986, 0.031988, -0.054199, -0.376446]
├─ TT  [0:5]: [-0.018541, -0.541951, -0.479817, 0.046163, -0.123459, -0.316877, -0.486986, 0.031988, -0.054199, -0.376446]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[negative] 
MODULE: [1mLinear[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0))
├─ INPUT: [2,6,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.056265, 0.545443, 0.484968, -0.047652, 0.161183, 0.320369, 0.492137, -0.033477, 0.091923, 0.379937]
├─ TT  [0:5]: [0.056265, 0.545443, 0.484968, -0.047652, 0.161183, 0.320369, 0.492137, -0.033477, 0.091923, 0.379937]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[zeros] 
MODULE: [1mLinear[0m
├─ EDGE CASE: [93mzeros[0m (All zeros)
├─ INPUT: [2,6,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [-0.000135, -0.010577, 0.008225, -0.012208, -0.000135, -0.010577, 0.008225, -0.012208, -0.000135, -0.010577]
├─ TT  [0:5]: [-0.000135, -0.010577, 0.008225, -0.012208, -0.000135, -0.010577, 0.008225, -0.012208, -0.000135, -0.010577]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[mixed] 
MODULE: [1mLinear[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative)
├─ INPUT: [2,6,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.465833, 0.050548, -0.099138, -0.564912, -0.512126, 0.528089, -0.868433, 0.196106, -0.328719, 0.001668]
├─ TT  [0:5]: [0.465833, 0.050548, -0.099138, -0.564912, -0.512126, 0.528089, -0.868433, 0.196106, -0.328719, 0.001668]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[small] 
MODULE: [1mLinear[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6))
├─ INPUT: [2,6,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.018862, 0.001746, 0.002575, -0.000744, 0.018862, 0.001746, 0.002575, -0.000744, 0.018862, 0.001746]
├─ TT  [0:5]: [0.018862, 0.001746, 0.002575, -0.000744, 0.018862, 0.001746, 0.002575, -0.000744, 0.018862, 0.001746]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[large] 
MODULE: [1mLinear[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6))
├─ INPUT: [2,6,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [39328.664062, -255004.828125, -137045.250000, 48265.898438, -65588.945312, -29930.966797, -144215.171875, 34091.183594, 3671.259521, -89499.468750]
├─ TT  [0:5]: [39328.664062, -255004.828125, -137045.250000, 48265.898438, -65588.945312, -29930.966797, -144215.171875, 34091.183594, 3671.259521, -89499.468750]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[minimum_input] 
MODULE: [1mLinear[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: [1,1,1]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [-0.000391]
├─ TT  [0:5]: [-0.000391]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear_no_bias 
MODULE: [1mLinear_no_bias[0m
├─ EDGE CASE: [93mmixed[0m (No bias term)
├─ INPUT: [2,6,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.464837, 0.055583, -0.083631, -0.565597, -0.513122, 0.533123, -0.852926, 0.195420, -0.329716, 0.006702]
├─ TT  [0:5]: [0.464837, 0.055583, -0.083631, -0.565597, -0.513122, 0.533123, -0.852926, 0.195420, -0.329716, 0.006702]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[positive] 
MODULE: [1mGroupNorm[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0))
├─ INPUT: [1,8,4,4]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [-0.221979, 1.876112, 1.079660, 0.594130, -1.017708, -1.017795, -1.374330, 1.568273, 0.603075, 0.992552]
├─ TT  [0:5]: [-0.221979, 1.876112, 1.079660, 0.594130, -1.017708, -1.017795, -1.374330, 1.568273, 0.603075, 0.992552]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[negative] 
MODULE: [1mGroupNorm[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0))
├─ INPUT: [1,8,4,4]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.221979, -1.876112, -1.079660, -0.594130, 1.017708, 1.017795, 1.374330, -1.568273, -0.603075, -0.992552]
├─ TT  [0:5]: [0.221979, -1.876112, -1.079660, -0.594130, 1.017708, 1.017795, 1.374330, -1.568273, -0.603075, -0.992552]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[zeros] 
MODULE: [1mGroupNorm[0m
├─ EDGE CASE: [93mzeros[0m (All zeros)
├─ INPUT: [1,8,4,4]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[mixed] 
MODULE: [1mGroupNorm[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative)
├─ INPUT: [1,8,4,4]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.680941, -0.001028, 0.843088, 1.783207, -0.104013, -0.103995, 1.843548, 0.971696, -0.356748, 0.730180]
├─ TT  [0:5]: [0.680941, -0.001028, 0.843088, 1.783207, -0.104013, -0.103995, 1.843548, 0.971696, -0.356748, 0.730180]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[small] 
MODULE: [1mGroupNorm[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6))
├─ INPUT: [1,8,4,4]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [-0.000019, 0.000163, 0.000094, 0.000052, -0.000088, -0.000088, -0.000119, 0.000136, 0.000052, 0.000086]
├─ TT  [0:5]: [-0.000019, 0.000163, 0.000094, 0.000052, -0.000088, -0.000088, -0.000119, 0.000136, 0.000052, 0.000086]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[large] 
MODULE: [1mGroupNorm[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6))
├─ INPUT: [1,8,4,4]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [-0.221994, 1.876237, 1.079732, 0.594169, -1.017775, -1.017862, -1.374421, 1.568378, 0.603115, 0.992618]
├─ TT  [0:5]: [-0.221994, 1.876237, 1.079732, 0.594169, -1.017775, -1.017862, -1.374421, 1.568378, 0.603115, 0.992618]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[minimum_input] 
MODULE: [1mGroupNorm[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: [1,4,1,1]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[positive] 
MODULE: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 – 2.0))
├─ INPUT: [2,4,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
├─ TT  [0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019, 1.155995, 1.058084, 1.866176, 1.601115, 1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[negative] 
MODULE: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0))
├─ INPUT: [2,4,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
├─ TT  [0:5]: [-1.374540, -1.950714, -1.731994, -1.598659, -1.156019, -1.155995, -1.058084, -1.866176, -1.601115, -1.708073]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[zeros] 
MODULE: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mzeros[0m (All zeros)
├─ INPUT: [2,4,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT  [0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[mixed] 
MODULE: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive / negative)
├─ INPUT: [2,4,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
├─ TT  [0:5]: [0.993428, -0.276529, 1.295377, 3.046060, -0.468307, -0.468274, 3.158426, 1.534869, -0.938949, 1.085120]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[small] 
MODULE: [1mDropout(p=0)[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6))
├─ INPUT: [2,4,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT  [0:5]: [0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[large] 
MODULE: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6))
├─ INPUT: [2,4,8]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
├─ TT  [0:5]: [374540.125000, 950714.312500, 731993.937500, 598658.500000, 156018.640625, 155994.515625, 58083.613281, 866176.125000, 601115.000000, 708072.625000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[minimum_input] 
MODULE: [1mDropout(p=0)[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size)
├─ INPUT: [1,1]
├─ SHAPE: [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ REF [0:5]: [0.496714]
├─ TT  [0:5]: [0.496714]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_param_count 
MODULE: [1mDropout[0m
├─ EDGE CASE: [93mparam_count[0m (Zero params)
├─ INPUT: N/A
├─ SHAPE: [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED

==================================================== FAILURES ====================================================
__________________________________________ test_module_op_registration ___________________________________________
workloads\Deformable_DETR\unit_tests\test_ops\test_nn_modules_unit.py:192: in test_module_op_registration
    assert "dummy_op.relu" in m._op_hndls
E   AssertionError: assert 'dummy_op.relu' in {'relu': <ttsim.front.functional.op.SimOpHandle object at 0x0000021C9659DA90>}
E    +  where {'relu': <ttsim.front.functional.op.SimOpHandle object at 0x0000021C9659DA90>} = <workloads.Deformable_DETR.unit_tests.test_ops.test_nn_modules_unit.test_module_op_registration.<locals>.DummyModule object at 0x0000021C9659D940>._op_hndls
=============================================== slowest durations ================================================
0.12s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[mixed]
0.07s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[minimum_input]
0.06s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[large]
0.04s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_module_tensor_registration
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[small]
0.03s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[mixed]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_dropout_identity[negative]
0.02s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[mixed]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_linear[large]
0.01s call     workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[positive]
0.01s setup    workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_groupnorm_module[mixed]

(75 durations < 0.005s hidden.  Use -vv to show these durations.)
============================================ short test summary info =============================================
FAILED workloads/Deformable_DETR/unit_tests/test_ops/test_nn_modules_unit.py::test_module_op_registration - Asse...
========================================== 1 failed, 28 passed in 1.49s ==========================================

═════════════════════════════════════════════════════════════════
NN MODULES UNIT TEST SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                  PASSED    TOTAL     RESULT
Module                  3         3         [92m✓ PASS[0m
ModuleList              2         2         [92m✓ PASS[0m
Linear                  8         8         [92m✓ PASS[0m
GroupNorm               7         7         [92m✓ PASS[0m
Dropout                 8         8         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
OVERALL: [92m✓ ALL PASS[0m
═════════════════════════════════════════════════════════════════
