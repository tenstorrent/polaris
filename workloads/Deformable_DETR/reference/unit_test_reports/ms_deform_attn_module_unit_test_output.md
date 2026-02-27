
=================================================================
MSDeformAttn MODULE UNIT TEST SUITE - PyTorch vs TTSim
=================================================================

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 7 items

workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_value_projection 
MODULE: [1mValueProjection[0m
|- EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
|- INPUT: [2, 340, 256]
|- INPUT input_flatten[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
|- SHAPE: PyTorch=[2,340,256] | TTSim=[2,340,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.91e-06, mean_diff=2.07e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.306117, -0.383856, 0.743524, -0.551626, -0.106879]
|- TT OUTPUT[0:5]: [-0.306118, -0.383857, 0.743524, -0.551626, -0.106879]
|_ RESULT: [92mV PASS[0m

MODULE: [1mValueProjection[0m
|- EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
|- INPUT: [2, 340, 256]
|- INPUT input_flatten[0:5]: [-1.115055, -1.609066, -1.133391, -1.240590, -1.327139]
|- SHAPE: PyTorch=[2,340,256] | TTSim=[2,340,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.91e-06, mean_diff=2.07e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.129690, -0.312756, -0.878697, -0.149183, 0.855688]
|- TT OUTPUT[0:5]: [0.129689, -0.312756, -0.878696, -0.149183, 0.855687]
|_ RESULT: [92mV PASS[0m

MODULE: [1mValueProjection[0m
|- EDGE CASE: [93mzeros[0m (All zeros - tests zero input behavior)
|- INPUT: [2, 340, 256]
|- INPUT input_flatten[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- SHAPE: PyTorch=[2,340,256] | TTSim=[2,340,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mValueProjection[0m
|- EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
|- INPUT: [2, 340, 256]
|- INPUT input_flatten[0:5]: [0.052750, 0.520643, -0.790291, -0.408602, -2.543265]
|- SHAPE: PyTorch=[2,340,256] | TTSim=[2,340,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=2.38e-06, mean_diff=2.32e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [3.773111, 1.827650, -0.464630, -3.194379, 1.235718]
|- TT OUTPUT[0:5]: [3.773111, 1.827650, -0.464629, -3.194378, 1.235717]
|_ RESULT: [92mV PASS[0m

MODULE: [1mValueProjection[0m
|- EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
|- INPUT: [2, 340, 256]
|- INPUT input_flatten[0:5]: [0.000001, 0.000001, 0.000000, 0.000001, 0.000000]
|- SHAPE: PyTorch=[2,340,256] | TTSim=[2,340,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=9.09e-13, mean_diff=7.25e-14 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.000000, 0.000000, 0.000001, -0.000000, 0.000000]
|- TT OUTPUT[0:5]: [-0.000000, 0.000000, 0.000001, -0.000000, 0.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mValueProjection[0m
|- EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
|- INPUT: [2, 340, 256]
|- INPUT input_flatten[0:5]: [113488.476562, 974483.062500, 728734.625000, 351467.812500, 707605.125000]
|- SHAPE: PyTorch=[2,340,256] | TTSim=[2,340,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=7.50e-01, mean_diff=7.24e-02 -> [91m[1mX FAIL[0m
|- PT OUTPUT[0:5]: [137587.203125, -251811.453125, 596442.437500, 48665.589844, -272271.906250]
|- TT OUTPUT[0:5]: [137587.234375, -251811.453125, 596442.375000, 48665.613281, -272271.812500]
|- FAILURE REASON: [91m[1mmax_diff=7.50e-01 exceeds atol=0.01[0m
|_ RESULT: [91m[1mX FAIL[0m
FAILED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_sampling_offsets 
MODULE: [1mSamplingOffsets[0m
|- EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|- TT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingOffsets[0m
|- EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [-1.115055, -1.609066, -1.133391, -1.240590, -1.327139]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|- TT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingOffsets[0m
|- EDGE CASE: [93mzeros[0m (All zeros - tests zero input behavior)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|- TT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingOffsets[0m
|- EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [0.052750, 0.520643, -0.790291, -0.408602, -2.543265]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|- TT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingOffsets[0m
|- EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [0.000001, 0.000001, 0.000000, 0.000001, 0.000000]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|- TT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingOffsets[0m
|- EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [113488.476562, 974483.062500, 728734.625000, 351467.812500, 707605.125000]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.001, atol=0.01)[0m
|- PT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|- TT OUTPUT[0:5]: [1.000000, 0.000000, 2.000000, 0.000000, 3.000000]
|_ RESULT: [92mV PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_attention_weights_softmax 
MODULE: [1mAttnWeights+Softmax[0m
|- EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
|- SHAPE: PyTorch=[2,50,8,4,4] | TTSim=[2,50,8,4,4] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|- TT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mAttnWeights+Softmax[0m
|- EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [-1.115055, -1.609066, -1.133391, -1.240590, -1.327139]
|- SHAPE: PyTorch=[2,50,8,4,4] | TTSim=[2,50,8,4,4] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|- TT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mAttnWeights+Softmax[0m
|- EDGE CASE: [93mzeros[0m (All zeros - tests zero input behavior)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- SHAPE: PyTorch=[2,50,8,4,4] | TTSim=[2,50,8,4,4] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|- TT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mAttnWeights+Softmax[0m
|- EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [0.052750, 0.520643, -0.790291, -0.408602, -2.543265]
|- SHAPE: PyTorch=[2,50,8,4,4] | TTSim=[2,50,8,4,4] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|- TT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mAttnWeights+Softmax[0m
|- EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [0.000001, 0.000001, 0.000000, 0.000001, 0.000000]
|- SHAPE: PyTorch=[2,50,8,4,4] | TTSim=[2,50,8,4,4] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|- TT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mAttnWeights+Softmax[0m
|- EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
|- INPUT: [2, 50, 256]
|- INPUT query[0:5]: [113488.476562, 974483.062500, 728734.625000, 351467.812500, 707605.125000]
|- SHAPE: PyTorch=[2,50,8,4,4] | TTSim=[2,50,8,4,4] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.001, atol=0.01)[0m
|- PT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|- TT OUTPUT[0:5]: [0.062500, 0.062500, 0.062500, 0.062500, 0.062500]
|_ RESULT: [92mV PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_sampling_locations 
MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
|- INPUT ref_points[0:5]: [0.126038, 0.609059, 0.607082, 0.260183, 0.727180]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.19e-07, mean_diff=1.17e-08 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.188538, 0.609059, 0.251038, 0.609059, 0.313538]
|- TT OUTPUT[0:5]: [0.188538, 0.609059, 0.251038, 0.609059, 0.313538]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [-1.115055, -1.609066, -1.133391, -1.240590, -1.327139]
|- INPUT ref_points[0:5]: [0.452140, 0.548791, 0.734503, 0.152027, 0.849251]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.19e-07, mean_diff=1.15e-08 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.514640, 0.548791, 0.577140, 0.548791, 0.639640]
|- TT OUTPUT[0:5]: [0.514640, 0.548791, 0.577140, 0.548791, 0.639640]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mzeros[0m (All zeros - tests zero input behavior)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- INPUT ref_points[0:5]: [0.834842, 0.104796, 0.744640, 0.360501, 0.359311]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.19e-07, mean_diff=1.17e-08 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.897342, 0.104796, 0.959842, 0.104796, 1.022342]
|- TT OUTPUT[0:5]: [0.897342, 0.104796, 0.959842, 0.104796, 1.022342]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [0.052750, 0.520643, -0.790291, -0.408602, -2.543265]
|- INPUT ref_points[0:5]: [0.270833, 0.246739, 0.549037, 0.001210, 0.628241]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.19e-07, mean_diff=1.21e-08 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.333333, 0.246739, 0.395833, 0.246739, 0.458333]
|- TT OUTPUT[0:5]: [0.333333, 0.246739, 0.395833, 0.246739, 0.458333]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [0.000001, 0.000001, 0.000000, 0.000001, 0.000000]
|- INPUT ref_points[0:5]: [0.510847, 0.683504, 0.522954, 0.108034, 0.832471]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.19e-07, mean_diff=1.20e-08 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.573347, 0.683504, 0.635847, 0.683504, 0.698347]
|- TT OUTPUT[0:5]: [0.573347, 0.683504, 0.635847, 0.683504, 0.698347]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [113488.476562, 974483.062500, 728734.625000, 351467.812500, 707605.125000]
|- INPUT ref_points[0:5]: [0.164591, 0.372574, 0.813491, 0.295974, 0.496157]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=1.19e-07, mean_diff=1.20e-08 -> [92mV PASS (tol: rtol=0.001, atol=0.01)[0m
|- PT OUTPUT[0:5]: [0.227091, 0.372574, 0.289591, 0.372574, 0.352091]
|- TT OUTPUT[0:5]: [0.227091, 0.372574, 0.289591, 0.372574, 0.352091]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mboundary_coords[0m (Sampling near 0.0 and 1.0 - grid edge behavior)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [-0.993631, -1.063401, -0.638076, 1.061589, -0.157404]
|- INPUT ref_points[0:5]: [0.010000, 0.010000, 0.990000, 0.990000, 0.010000]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.072500, 0.010000, 0.135000, 0.010000, 0.197500]
|- TT OUTPUT[0:5]: [0.072500, 0.010000, 0.135000, 0.010000, 0.197500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mcenter_coords[0m (All locations at 0.5 - center-only sampling)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [-1.043159, -0.820856, 0.665146, 1.822627, -1.441583]
|- INPUT ref_points[0:5]: [0.500000, 0.500000, 0.500000, 0.500000, 0.500000]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=2.38e-08, mean_diff=1.90e-09 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.562500, 0.500000, 0.625000, 0.500000, 0.687500]
|- TT OUTPUT[0:5]: [0.562500, 0.500000, 0.625000, 0.500000, 0.687500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93mcorner_coords[0m (All at 0.0 or 1.0 - extreme corner positions)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 2]
|- INPUT query[0:5]: [-1.560352, -0.030978, -0.620928, -1.464581, 1.411946]
|- INPUT ref_points[0:5]: [0.000000, 1.000000, 0.000000, 1.000000, 1.000000]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=2.98e-08, mean_diff=1.78e-09 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.062500, 1.000000, 0.125000, 1.000000, 0.187500]
|- TT OUTPUT[0:5]: [0.062500, 1.000000, 0.125000, 1.000000, 0.187500]
|_ RESULT: [92mV PASS[0m

MODULE: [1mSamplingLocations[0m
|- EDGE CASE: [93m4d_ref_points[0m (4D reference points (x, y, w, h) - box-based references)
|- INPUT: query=[2, 50, 256] ref_pts=[2, 50, 4, 4]
|- INPUT query[0:5]: [1.675731, 1.044712, 1.343304, 1.644020, 1.284213]
|- INPUT ref_points[0:5]: [0.394936, 0.927054, 0.892394, 0.547227, 0.459062]
|- SHAPE: PyTorch=[2,50,8,4,4,2] | TTSim=[2,50,8,4,4,2] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.506485, 0.927054, 0.618034, 0.927054, 0.729584]
|- TT OUTPUT[0:5]: [0.506485, 0.927054, 0.618034, 0.927054, 0.729584]
|_ RESULT: [92mV PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_e2e_2d_ref_points 
MODULE: [1mMSDeformAttn E2E (2D)[0m
|- EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256]
|- INPUT query[0:5]: [1.412827, 1.231669, 1.969363, 1.990798, 1.276504]
|- INPUT input_flatten[0:5]: [1.486709, 1.970928, 1.093409, 1.655735, 1.928102]
|- INPUT ref_points[0:5]: [0.210327, 0.407464, 0.988499, 0.106760, 0.231883]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=9.54e-07, mean_diff=1.26e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.756537, -0.302089, -1.331972, -0.259613, -1.289268]
|- TT OUTPUT[0:5]: [-0.756537, -0.302089, -1.331972, -0.259613, -1.289268]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (2D)[0m
|- EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256]
|- INPUT query[0:5]: [-1.335220, -1.776061, -1.798844, -1.377926, -1.256579]
|- INPUT input_flatten[0:5]: [-1.112715, -1.870849, -1.347485, -1.218080, -1.076603]
|- INPUT ref_points[0:5]: [0.767307, 0.408024, 0.520757, 0.313529, 0.609734]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=9.54e-07, mean_diff=1.21e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.746862, -0.179131, 0.863702, -1.120666, 0.259016]
|- TT OUTPUT[0:5]: [0.746862, -0.179131, 0.863702, -1.120666, 0.259016]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (2D)[0m
|- EDGE CASE: [93mzeros[0m (All zeros - tests zero input behavior)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256]
|- INPUT query[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- INPUT input_flatten[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- INPUT ref_points[0:5]: [0.820353, 0.034136, 0.423412, 0.005252, 0.186777]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (2D)[0m
|- EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256]
|- INPUT query[0:5]: [0.607635, 1.757827, -0.788944, 0.880530, 2.894395]
|- INPUT input_flatten[0:5]: [1.284893, -0.522131, 1.827970, -0.679520, 3.283894]
|- INPUT ref_points[0:5]: [0.702969, 0.763942, 0.595591, 0.078363, 0.814485]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=5.36e-07, mean_diff=8.23e-08 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.105273, 0.212990, 0.256677, 0.387729, 0.097998]
|- TT OUTPUT[0:5]: [-0.105273, 0.212990, 0.256677, 0.387729, 0.097998]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (2D)[0m
|- EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256]
|- INPUT query[0:5]: [0.000000, 0.000000, 0.000001, 0.000000, 0.000000]
|- INPUT input_flatten[0:5]: [0.000001, 0.000000, 0.000001, 0.000001, 0.000001]
|- INPUT ref_points[0:5]: [0.226985, 0.875929, 0.849945, 0.000314, 0.341303]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=3.41e-13, mean_diff=4.22e-14 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.000000, 0.000000, -0.000000, 0.000000, 0.000000]
|- TT OUTPUT[0:5]: [-0.000000, 0.000000, -0.000000, 0.000000, 0.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (2D)[0m
|- EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256]
|- INPUT query[0:5]: [529815.000000, 979481.687500, 358202.968750, 200275.156250, 770302.625000]
|- INPUT input_flatten[0:5]: [738644.312500, 96725.320312, 86424.828125, 473048.687500, 769339.187500]
|- INPUT ref_points[0:5]: [0.275152, 0.683625, 0.006303, 0.315750, 0.903095]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=3.44e-01, mean_diff=4.24e-02 -> [91m[1mX FAIL[0m
|- PT OUTPUT[0:5]: [-216136.671875, 189185.078125, -81051.500000, 282128.781250, -507460.125000]
|- TT OUTPUT[0:5]: [-216136.593750, 189185.015625, -81051.445312, 282128.718750, -507460.125000]
|- FAILURE REASON: [91m[1mmax_diff=3.44e-01 exceeds atol=0.01[0m
|_ RESULT: [91m[1mX FAIL[0m

MODULE: [1mMSDeformAttn E2E (2D)[0m
|- EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
|- INPUT: query=[1, 10, 128] flat=[1, 20, 128]
|- INPUT query[0:5]: [1.723209, 1.611448, 1.677061, 1.296519, 1.107935]
|- INPUT input_flatten[0:5]: [1.641469, 1.011989, 1.031547, 1.676645, 1.636197]
|- INPUT ref_points[0:5]: [0.960143, 0.895561, 0.161336, 0.295521, 0.539111]
|- SHAPE: PyTorch=[1,10,128] | TTSim=[1,10,128] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=5.96e-07, mean_diff=9.90e-08 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.108576, -0.407355, -0.276170, -0.270224, 0.187681]
|- TT OUTPUT[0:5]: [-0.108576, -0.407355, -0.276170, -0.270224, 0.187681]
|_ RESULT: [92mV PASS[0m
FAILED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_e2e_4d_ref_points 
MODULE: [1mMSDeformAttn E2E (4D)[0m
|- EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256] ref_dim=4
|- INPUT query[0:5]: [1.412827, 1.231669, 1.969363, 1.990798, 1.276504]
|- INPUT input_flatten[0:5]: [1.341614, 1.966492, 1.826253, 1.581450, 1.091093]
|- INPUT ref_points_4d[0:5]: [0.210327, 0.407464, 0.900000, 0.106760, 0.231883]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=2.15e-06, mean_diff=2.07e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-1.382908, -0.401412, -1.791454, -0.198896, -1.759205]
|- TT OUTPUT[0:5]: [-1.382908, -0.401412, -1.791454, -0.198897, -1.759205]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (4D)[0m
|- EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256] ref_dim=4
|- INPUT query[0:5]: [-1.335220, -1.776061, -1.798844, -1.377926, -1.256579]
|- INPUT input_flatten[0:5]: [-1.449079, -1.808722, -1.426916, -1.354097, -1.286406]
|- INPUT ref_points_4d[0:5]: [0.767307, 0.408024, 0.520757, 0.313529, 0.609734]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=2.15e-06, mean_diff=1.94e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [1.161554, -0.316069, 1.201040, -1.931637, 0.485394]
|- TT OUTPUT[0:5]: [1.161554, -0.316069, 1.201040, -1.931636, 0.485394]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (4D)[0m
|- EDGE CASE: [93mzeros[0m (All zeros - tests zero input behavior)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256] ref_dim=4
|- INPUT query[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- INPUT input_flatten[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- INPUT ref_points_4d[0:5]: [0.820353, 0.034136, 0.423412, 0.100000, 0.186777]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|- TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (4D)[0m
|- EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256] ref_dim=4
|- INPUT query[0:5]: [0.607635, 1.757827, -0.788944, 0.880530, 2.894395]
|- INPUT input_flatten[0:5]: [-1.036414, 4.069465, -2.844192, -0.436692, -3.230360]
|- INPUT ref_points_4d[0:5]: [0.702969, 0.763942, 0.595591, 0.100000, 0.814485]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=8.34e-07, mean_diff=1.16e-07 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.002377, -0.297923, -0.099813, -0.664856, 0.345065]
|- TT OUTPUT[0:5]: [-0.002377, -0.297924, -0.099813, -0.664856, 0.345065]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (4D)[0m
|- EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256] ref_dim=4
|- INPUT query[0:5]: [0.000000, 0.000000, 0.000001, 0.000000, 0.000000]
|- INPUT input_flatten[0:5]: [0.000001, 0.000001, 0.000000, 0.000001, 0.000001]
|- INPUT ref_points_4d[0:5]: [0.226985, 0.875929, 0.849945, 0.100000, 0.341303]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=5.68e-13, mean_diff=6.69e-14 -> [92mV PASS (tol: rtol=0.0001, atol=1e-05)[0m
|- PT OUTPUT[0:5]: [-0.000000, 0.000000, -0.000000, 0.000000, 0.000001]
|- TT OUTPUT[0:5]: [-0.000000, 0.000000, -0.000000, 0.000000, 0.000001]
|_ RESULT: [92mV PASS[0m

MODULE: [1mMSDeformAttn E2E (4D)[0m
|- EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256] ref_dim=4
|- INPUT query[0:5]: [529815.000000, 979481.687500, 358202.968750, 200275.156250, 770302.625000]
|- INPUT input_flatten[0:5]: [364869.468750, 803436.125000, 658322.250000, 45927.187500, 237650.609375]
|- INPUT ref_points_4d[0:5]: [0.275152, 0.683625, 0.100000, 0.315750, 0.903095]
|- SHAPE: PyTorch=[2,50,256] | TTSim=[2,50,256] -> [92mV MATCH[0m
|- NUMERICAL: max_diff=5.62e-01, mean_diff=6.69e-02 -> [91m[1mX FAIL[0m
|- PT OUTPUT[0:5]: [-280014.031250, 385583.000000, -18482.103516, 210852.718750, -1037573.187500]
|- TT OUTPUT[0:5]: [-280013.968750, 385583.000000, -18482.263672, 210852.671875, -1037573.062500]
|- FAILURE REASON: [91m[1mmax_diff=5.62e-01 exceeds atol=0.01[0m
|_ RESULT: [91m[1mX FAIL[0m
FAILED
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_shape_inference 
MODULE: [1mShapeInference[0m
|- EDGE CASE: [93mstandard[0m (Standard multi-level configuration - baseline correctness)
|- INPUT: query=[2, 100, 256] flat=[2, 3343, 256]
|- SHAPE: Expected=[2,100,256] | Actual=[2,100,256] -> [92mV MATCH[0m
|_ RESULT: [92mV PASS[0m

MODULE: [1mShapeInference[0m
|- EDGE CASE: [93msmall_config[0m (Small configuration for quick test)
|- INPUT: query=[1, 50, 128] flat=[1, 1280, 128]
|- SHAPE: Expected=[1,50,128] | Actual=[1,50,128] -> [92mV MATCH[0m
|_ RESULT: [92mV PASS[0m

MODULE: [1mShapeInference[0m
|- EDGE CASE: [93msingle_level[0m (Single feature level - minimal level loop)
|- INPUT: query=[2, 100, 256] flat=[2, 2500, 256]
|- SHAPE: Expected=[2,100,256] | Actual=[2,100,256] -> [92mV MATCH[0m
|_ RESULT: [92mV PASS[0m

MODULE: [1mShapeInference[0m
|- EDGE CASE: [93mmany_heads[0m (Many attention heads - head scalability)
|- INPUT: query=[2, 50, 256] flat=[2, 340, 256]
|- SHAPE: Expected=[2,50,256] | Actual=[2,50,256] -> [92mV MATCH[0m
|_ RESULT: [92mV PASS[0m

MODULE: [1mShapeInference[0m
|- EDGE CASE: [93msingle_query[0m (Single query - minimal query count)
|- INPUT: query=[1, 1, 256] flat=[1, 85, 256]
|- SHAPE: Expected=[1,1,256] | Actual=[1,1,256] -> [92mV MATCH[0m
|_ RESULT: [92mV PASS[0m

MODULE: [1mShapeInference[0m
|- EDGE CASE: [93mlarge_batch[0m (Large batch size - batch scalability)
|- INPUT: query=[4, 50, 256] flat=[4, 340, 256]
|- SHAPE: Expected=[4,50,256] | Actual=[4,50,256] -> [92mV MATCH[0m
|_ RESULT: [92mV PASS[0m

MODULE: [1mShapeInference[0m
|- EDGE CASE: [93mminimum_input[0m (Smallest valid input size - degenerate/boundary case)
|- INPUT: query=[1, 1, 64] flat=[1, 4, 64]
|- SHAPE: Expected=[1,1,64] | Actual=[1,1,64] -> [92mV MATCH[0m
|_ RESULT: [92mV PASS[0m
PASSED

=================================== FAILURES ===================================
____________________________ test_value_projection _____________________________
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py:685: in test_value_projection
    assert passed == len(
E   AssertionError: ValueProjection: 5/6 passed
E   assert 5 == 6
E    +  where 6 = len([('Baseline positive', 'positive', 'baseline'), ('Negative values', 'negative', 'edge_value'), ('Zero values', 'zeros', 'edge_value'), ('Mixed pos/neg', 'mixed', 'edge_value'), ('Very small (1e-6)', 'small', 'edge_value'), ('Very large (1e6)', 'large', 'edge_value')])
____________________________ test_e2e_2d_ref_points ____________________________
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py:1623: in test_e2e_2d_ref_points
    assert passed == len(
E   AssertionError: MSDeformAttn E2E (2D): 6/7 passed
E   assert 6 == 7
E    +  where 7 = len([('Baseline positive', 'positive', 256, 4, 8, 4, ...), ('Negative values', 'negative', 256, 4, 8, 4, ...), ('Zero values', 'zeros', 256, 4, 8, 4, ...), ('Mixed pos/neg', 'mixed', 256, 4, 8, 4, ...), ('Very small (1e-6)', 'small', 256, 4, 8, 4, ...), ('Very large (1e6)', 'large', 256, 4, 8, 4, ...), ...])
____________________________ test_e2e_4d_ref_points ____________________________
workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py:1846: in test_e2e_4d_ref_points
    assert passed == len(
E   AssertionError: MSDeformAttn E2E (4D): 5/6 passed
E   assert 5 == 6
E    +  where 6 = len([('Baseline positive 4D', 'positive', 'baseline'), ('Negative values 4D', 'negative', 'edge_value'), ('Zero values 4D', 'zeros', 'edge_value'), ('Mixed pos/neg 4D', 'mixed', 'edge_value'), ('Very small (1e-6) 4D', 'small', 'edge_value'), ('Very large (1e6) 4D', 'large', 'edge_value')])
============================== slowest durations ===============================
0.52s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_e2e_2d_ref_points
0.50s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_e2e_4d_ref_points
0.29s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_value_projection
0.10s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_sampling_locations
0.05s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_sampling_offsets
0.04s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_attention_weights_softmax
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_shape_inference

(14 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ============================
FAILED workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_value_projection
FAILED workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_e2e_2d_ref_points
FAILED workloads/Deformable_DETR/reference/unit_tests/test_ms_deform_attn_module_unit.py::test_e2e_4d_ref_points
========================= 3 failed, 4 passed in 1.59s ==========================

=================================================================
SUMMARY
=================================================================
MODULE                             SHAPE       NUMERICAL   TOTAL
ValueProjection                    6/6         5/6         [91m[1mX FAIL[0m
SamplingOffsets                    6/6         6/6         [92mV PASS[0m
AttnWeights+Softmax                6/6         6/6         [92mV PASS[0m
SamplingLocations                  10/10       10/10       [92mV PASS[0m
MSDeformAttn E2E (2D)              7/7         6/7         [91m[1mX FAIL[0m
MSDeformAttn E2E (4D)              6/6         5/6         [91m[1mX FAIL[0m
ShapeInference                     7/7         N/A         [92mV PASS[0m
-----------------------------------------------------------------
TOTAL                              48/48          38/41       [91m[1mX FAIL[0m

[91m[1mFAILED TESTS:[0m
- ValueProjection | large values | max_diff=7.50e-01 > atol=0.01
- MSDeformAttn E2E (2D) | large values | max_diff=3.44e-01 > atol=0.01
- MSDeformAttn E2E (4D) | large values | max_diff=5.62e-01 > atol=0.01
=================================================================
