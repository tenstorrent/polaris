
═════════════════════════════════════════════════════════════════
MISC UTILITIES UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 7 items

workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_nested_tensor 
MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 3, 56, 56]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: tensor: PT=[2,3,56,56] | TT=[2,3,56,56], mask: PT=[2,56,56] | TT=[2,56,56] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ TT OUTPUT[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 32, 32]
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: tensor: PT=[1,3,32,32] | TT=[1,3,32,32], mask: PT=[1,32,32] | TT=[1,32,32] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ TT OUTPUT[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 64, 16, 16]
├─ INPUT input[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ SHAPE: tensor: PT=[2,64,16,16] | TT=[2,64,16,16], mask: PT=[2,16,16] | TT=[2,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ TT OUTPUT[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [2, 3, 16, 16]
├─ INPUT input[0:5]: [-1.989012, -1.549545, -1.281447, -1.077290, -1.444469]
├─ SHAPE: tensor: PT=[2,3,16,16] | TT=[2,3,16,16], mask: PT=[2,16,16] | TT=[2,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.989012, -1.549545, -1.281447, -1.077290, -1.444469]
├─ TT OUTPUT[0:5]: [-1.989012, -1.549545, -1.281447, -1.077290, -1.444469]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-value propagation)
├─ INPUT: [2, 3, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: tensor: PT=[2,3,16,16] | TT=[2,3,16,16], mask: PT=[2,16,16] | TT=[2,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [2, 3, 16, 16]
├─ INPUT input[0:5]: [-1.696019, 2.611813, 1.848416, 1.280824, -2.109474]
├─ SHAPE: tensor: PT=[2,3,16,16] | TT=[2,3,16,16], mask: PT=[2,16,16] | TT=[2,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.696019, 2.611813, 1.848416, 1.280824, -2.109474]
├─ TT OUTPUT[0:5]: [-1.696019, 2.611813, 1.848416, 1.280824, -2.109474]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [2, 3, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000000, 0.000000, 0.000001]
├─ SHAPE: tensor: PT=[2,3,16,16] | TT=[2,3,16,16], mask: PT=[2,16,16] | TT=[2,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000001, 0.000000, 0.000000, 0.000001]
├─ TT OUTPUT[0:5]: [0.000000, 0.000001, 0.000000, 0.000000, 0.000001]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [2, 3, 16, 16]
├─ INPUT input[0:5]: [300964.437500, 247061.828125, 926335.125000, 891603.437500, 683276.750000]
├─ SHAPE: tensor: PT=[2,3,16,16] | TT=[2,3,16,16], mask: PT=[2,16,16] | TT=[2,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [300964.437500, 247061.828125, 926335.125000, 891603.437500, 683276.750000]
├─ TT OUTPUT[0:5]: [300964.437500, 247061.828125, 926335.125000, 891603.437500, 683276.750000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 1, 1, 1]
├─ INPUT input[0:5]: [1.494602]
├─ SHAPE: tensor: PT=[1,1,1,1] | TT=[1,1,1,1], mask: PT=[1,1,1] | TT=[1,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.494602]
├─ TT OUTPUT[0:5]: [1.494602]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mNestedTensor[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [2, 3, 1, 1]
├─ INPUT input[0:5]: [-0.581006, 0.224256, 2.501590, -2.721780, 0.199866]
├─ SHAPE: tensor: PT=[2,3,1,1] | TT=[2,3,1,1], mask: PT=[2,1,1] | TT=[2,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.581006, 0.224256, 2.501590, -2.721780, 0.199866]
├─ TT OUTPUT[0:5]: [-0.581006, 0.224256, 2.501590, -2.721780, 0.199866]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_interpolate_nearest 
MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 3, 10, 10] → (20, 20)
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[2,3,20,20] | TTSim=[2,3,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374540, 1.374540, 1.950714, 1.950714, 1.731994]
├─ TT OUTPUT[0:5]: [1.374540, 1.374540, 1.950714, 1.950714, 1.731994]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 8, 8] → (24, 24)
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: PyTorch=[1,3,24,24] | TTSim=[1,3,24,24] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.115055, 1.115055, 1.115055, 1.609066, 1.609066]
├─ TT OUTPUT[0:5]: [1.115055, 1.115055, 1.115055, 1.609066, 1.609066]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 8, 10] → (16, 20)
├─ INPUT input[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ SHAPE: PyTorch=[1,3,16,20] | TTSim=[1,3,16,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.834842, 1.834842, 1.104796, 1.104796, 1.744640]
├─ TT OUTPUT[0:5]: [1.834842, 1.834842, 1.104796, 1.104796, 1.744640]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 64, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
├─ SHAPE: PyTorch=[1,64,16,16] | TTSim=[1,64,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.989012, 1.989012, 1.549545, 1.549545, 1.281447]
├─ TT OUTPUT[0:5]: [1.989012, 1.989012, 1.549545, 1.549545, 1.281447]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [2, 3, 10, 10] → (20, 20)
├─ INPUT input[0:5]: [-1.783832, -1.634834, -1.249043, -1.758076, -1.313077]
├─ SHAPE: PyTorch=[2,3,20,20] | TTSim=[2,3,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.783832, -1.783832, -1.634834, -1.634834, -1.249043]
├─ TT OUTPUT[0:5]: [-1.783832, -1.783832, -1.634834, -1.634834, -1.249043]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-value propagation)
├─ INPUT: [2, 3, 10, 10] → (20, 20)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,3,20,20] | TTSim=[2,3,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [2, 3, 10, 10] → (20, 20)
├─ INPUT input[0:5]: [-1.987263, -2.126802, -1.276152, 2.123177, -0.314808]
├─ SHAPE: PyTorch=[2,3,20,20] | TTSim=[2,3,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.987263, -1.987263, -2.126802, -2.126802, -1.276152]
├─ TT OUTPUT[0:5]: [-1.987263, -1.987263, -2.126802, -2.126802, -1.276152]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [2, 3, 10, 10] → (20, 20)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ SHAPE: PyTorch=[2,3,20,20] | TTSim=[2,3,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000001]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000001]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [2, 3, 10, 10] → (20, 20)
├─ INPUT input[0:5]: [494601.625000, 228083.109375, 255473.906250, 396329.906250, 377315.093750]
├─ SHAPE: PyTorch=[2,3,20,20] | TTSim=[2,3,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [494601.625000, 494601.625000, 228083.109375, 228083.109375, 255473.906250]
├─ TT OUTPUT[0:5]: [494601.625000, 494601.625000, 228083.109375, 228083.109375, 255473.906250]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 1, 1, 1] → (2, 2)
├─ INPUT input[0:5]: [1.675731]
├─ SHAPE: PyTorch=[1,1,2,2] | TTSim=[1,1,2,2] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.675731, 1.675731, 1.675731, 1.675731]
├─ TT OUTPUT[0:5]: [1.675731, 1.675731, 1.675731, 1.675731]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (nearest)[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 1, 1, 1] → (1, 1)
├─ INPUT input[0:5]: [1.038952]
├─ SHAPE: PyTorch=[1,1,1,1] | TTSim=[1,1,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.038952]
├─ TT OUTPUT[0:5]: [1.038952]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_interpolate_bilinear 
MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[1,3,16,16] | TTSim=[1,3,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=2.51e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374540, 1.518584, 1.806671, 1.896034, 1.786674]
├─ TT OUTPUT[0:5]: [1.374540, 1.518584, 1.806671, 1.896034, 1.786674]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 4, 4] → (12, 12)
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: PyTorch=[1,3,12,12] | TTSim=[1,3,12,12] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=4.80e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.115055, 1.115055, 1.279725, 1.444396, 1.609066]
├─ TT OUTPUT[0:5]: [1.115055, 1.115055, 1.279725, 1.444396, 1.609066]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 6, 8] → (12, 16)
├─ INPUT input[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ SHAPE: PyTorch=[1,3,12,16] | TTSim=[1,3,12,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=2.24e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.834842, 1.652331, 1.287308, 1.264757, 1.584679]
├─ TT OUTPUT[0:5]: [1.834842, 1.652331, 1.287308, 1.264757, 1.584679]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 3, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
├─ SHAPE: PyTorch=[2,3,16,16] | TTSim=[2,3,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=2.81e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.989012, 1.879145, 1.659411, 1.482520, 1.348472]
├─ TT OUTPUT[0:5]: [1.989012, 1.879145, 1.659411, 1.482520, 1.348472]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [1, 3, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [-1.783832, -1.634834, -1.249043, -1.758076, -1.313077]
├─ SHAPE: PyTorch=[1,3,16,16] | TTSim=[1,3,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=2.39e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.783832, -1.746583, -1.672083, -1.538386, -1.345491]
├─ TT OUTPUT[0:5]: [-1.783832, -1.746583, -1.672083, -1.538386, -1.345491]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-value propagation)
├─ INPUT: [1, 3, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,3,16,16] | TTSim=[1,3,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 3, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [-1.987263, -2.126802, -1.276152, 2.123177, -0.314808]
├─ SHAPE: PyTorch=[1,3,16,16] | TTSim=[1,3,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=2.63e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.987263, -2.022147, -2.091917, -1.914139, -1.488814]
├─ TT OUTPUT[0:5]: [-1.987263, -2.022147, -2.091917, -1.914139, -1.488814]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [1, 3, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ SHAPE: PyTorch=[1,3,16,16] | TTSim=[1,3,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.14e-13, mean_diff=9.99e-15 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000001]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000001]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [1, 3, 8, 8] → (16, 16)
├─ INPUT input[0:5]: [494601.625000, 228083.109375, 255473.906250, 396329.906250, 377315.093750]
├─ SHAPE: PyTorch=[1,3,16,16] | TTSim=[1,3,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.25e-01, mean_diff=8.56e-03 → [92m✓ PASS (tol: rtol=0.001, atol=0.01)[0m
├─ PT OUTPUT[0:5]: [494601.625000, 427972.000000, 294712.750000, 234930.812500, 248626.218750]
├─ TT OUTPUT[0:5]: [494601.625000, 427972.000000, 294712.750000, 234930.812500, 248626.218750]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (bilinear)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 1, 2, 2] → (4, 4)
├─ INPUT input[0:5]: [1.675731, 1.044712, 1.343304, 1.644020]
├─ SHAPE: PyTorch=[1,1,4,4] | TTSim=[1,1,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.19e-07, mean_diff=3.73e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.675731, 1.517977, 1.202467, 1.044712, 1.592624]
├─ TT OUTPUT[0:5]: [1.675731, 1.517977, 1.202467, 1.044712, 1.592624]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_interpolate_downsample 
MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 3, 32, 32] → (8, 8)
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374540, 1.156019, 1.601115, 1.832443, 1.304242]
├─ TT OUTPUT[0:5]: [1.374540, 1.156019, 1.601115, 1.832443, 1.304242]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 16, 16] → (8, 8)
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: PyTorch=[1,3,8,8] | TTSim=[1,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.115055, 1.133391, 1.327139, 1.666090, 1.029014]
├─ TT OUTPUT[0:5]: [1.115055, 1.133391, 1.327139, 1.666090, 1.029014]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 3, 24, 32] → (6, 8)
├─ INPUT input[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ SHAPE: PyTorch=[1,3,6,8] | TTSim=[1,3,6,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.834842, 1.359311, 1.509902, 1.427652, 1.943351]
├─ TT OUTPUT[0:5]: [1.834842, 1.359311, 1.509902, 1.427652, 1.943351]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [2, 3, 32, 32] → (8, 8)
├─ INPUT input[0:5]: [-1.989012, -1.549545, -1.281447, -1.077290, -1.444469]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.989012, -1.444469, -1.115951, -1.990722, -1.976003]
├─ TT OUTPUT[0:5]: [-1.989012, -1.444469, -1.115951, -1.990722, -1.976003]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-value propagation)
├─ INPUT: [2, 3, 32, 32] → (8, 8)
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [2, 3, 32, 32] → (8, 8)
├─ INPUT input[0:5]: [-1.696019, 2.611813, 1.848416, 1.280824, -2.109474]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.696019, -2.109474, -0.589301, -1.864681, 1.439066]
├─ TT OUTPUT[0:5]: [-1.696019, -2.109474, -0.589301, -1.864681, 1.439066]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [2, 3, 32, 32] → (8, 8)
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000000, 0.000000, 0.000001]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [2, 3, 32, 32] → (8, 8)
├─ INPUT input[0:5]: [300964.437500, 247061.828125, 926335.125000, 891603.437500, 683276.750000]
├─ SHAPE: PyTorch=[2,3,8,8] | TTSim=[2,3,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [300964.437500, 683276.750000, 769754.500000, 442212.781250, 962167.562500]
├─ TT OUTPUT[0:5]: [300964.437500, 683276.750000, 769754.500000, 442212.781250, 962167.562500]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 1, 2, 2] → (1, 1)
├─ INPUT input[0:5]: [1.494602, 1.228083, 1.255474, 1.396330]
├─ SHAPE: PyTorch=[1,1,1,1] | TTSim=[1,1,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.494602]
├─ TT OUTPUT[0:5]: [1.494602]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minterpolate (downsample)[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 3, 4, 4] → (4, 4)
├─ INPUT input[0:5]: [-0.581006, 0.224256, 2.501590, -2.721780, 0.199866]
├─ SHAPE: PyTorch=[1,3,4,4] | TTSim=[1,3,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.581006, 0.224256, 2.501590, -2.721780, 0.199866]
├─ TT OUTPUT[0:5]: [-0.581006, 0.224256, 2.501590, -2.721780, 0.199866]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_nested_tensor_from_tensor_list 
MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [3, 10, 15] + [3, 12, 18] + [3, 8, 20]
├─ INPUT tensor0[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT tensor1[0:5]: [1.777147, 1.558404, 1.424222, 1.906354, 1.111197]
├─ INPUT tensor2[0:5]: [1.779584, 1.350125, 1.057843, 1.969103, 1.883786]
├─ SHAPE: tensor: PT=[3,3,12,20] | TT=[3,3,12,20], mask: PT=[3,12,20] | TT=[3,12,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ TT OUTPUT[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [3, 16, 16] + [3, 16, 16]
├─ INPUT tensor0[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ INPUT tensor1[0:5]: [1.880242, 1.599169, 1.325142, 1.392751, 1.226588]
├─ SHAPE: tensor: PT=[2,3,16,16] | TT=[2,3,16,16], mask: PT=[2,16,16] | TT=[2,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ TT OUTPUT[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [3, 8, 12] + [3, 10, 10] + [3, 6, 14] + [3, 12, 8]
├─ INPUT tensor0[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ INPUT tensor1[0:5]: [1.828604, 1.499271, 1.283350, 1.189092, 1.646965]
├─ INPUT tensor2[0:5]: [1.728743, 1.409898, 1.067315, 1.097431, 1.193316]
├─ SHAPE: tensor: PT=[4,3,12,14] | TT=[4,3,12,14], mask: PT=[4,12,14] | TT=[4,12,14] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ TT OUTPUT[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 10, 10] + [1, 8, 12]
├─ INPUT tensor0[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
├─ INPUT tensor1[0:5]: [1.523749, 1.297884, 1.570986, 1.573988, 1.044422]
├─ SHAPE: tensor: PT=[2,1,10,12] | TT=[2,1,10,12], mask: PT=[2,10,12] | TT=[2,10,12] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
├─ TT OUTPUT[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [3, 10, 15] + [3, 12, 18]
├─ INPUT tensor0[0:5]: [-1.783832, -1.634834, -1.249043, -1.758076, -1.313077]
├─ INPUT tensor1[0:5]: [-1.647633, -1.794958, -1.560003, -1.516853, -1.803617]
├─ SHAPE: tensor: PT=[2,3,12,18] | TT=[2,3,12,18], mask: PT=[2,12,18] | TT=[2,12,18] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.783832, -1.634834, -1.249043, -1.758076, -1.313077]
├─ TT OUTPUT[0:5]: [-1.783832, -1.634834, -1.249043, -1.758076, -1.313077]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-value propagation)
├─ INPUT: [3, 10, 15] + [3, 12, 18]
├─ INPUT tensor0[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT tensor1[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: tensor: PT=[2,3,12,18] | TT=[2,3,12,18], mask: PT=[2,12,18] | TT=[2,12,18] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [3, 10, 15] + [3, 12, 18]
├─ INPUT tensor0[0:5]: [-1.987263, -2.126802, -1.276152, 2.123177, -0.314808]
├─ INPUT tensor1[0:5]: [-2.507509, -2.905108, -0.852859, 0.269001, -2.978230]
├─ SHAPE: tensor: PT=[2,3,12,18] | TT=[2,3,12,18], mask: PT=[2,12,18] | TT=[2,12,18] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.987263, -2.126802, -1.276152, 2.123177, -0.314808]
├─ TT OUTPUT[0:5]: [-1.987263, -2.126802, -1.276152, 2.123177, -0.314808]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [3, 10, 15] + [3, 12, 18]
├─ INPUT tensor0[0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ INPUT tensor1[0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ SHAPE: tensor: PT=[2,3,12,18] | TT=[2,3,12,18], mask: PT=[2,12,18] | TT=[2,12,18] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000001, 0.000001, 0.000001]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [3, 10, 15] + [3, 12, 18]
├─ INPUT tensor0[0:5]: [494601.625000, 228083.109375, 255473.906250, 396329.906250, 377315.093750]
├─ INPUT tensor1[0:5]: [648482.125000, 32361.767578, 270590.218750, 55003.882812, 453784.031250]
├─ SHAPE: tensor: PT=[2,3,12,18] | TT=[2,3,12,18], mask: PT=[2,12,18] | TT=[2,12,18] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [494601.625000, 228083.109375, 255473.906250, 396329.906250, 377315.093750]
├─ TT OUTPUT[0:5]: [494601.625000, 228083.109375, 255473.906250, 396329.906250, 377315.093750]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 1, 1] + [1, 1, 1]
├─ INPUT tensor0[0:5]: [1.675731]
├─ INPUT tensor1[0:5]: [1.044712]
├─ SHAPE: tensor: PT=[2,1,1,1] | TT=[2,1,1,1], mask: PT=[2,1,1] | TT=[2,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.675731, 1.044712]
├─ TT OUTPUT[0:5]: [1.675731, 1.044712]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mnested_tensor_from_tensor_list[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 1, 1] + [1, 2, 3]
├─ INPUT tensor0[0:5]: [1.038952]
├─ INPUT tensor1[0:5]: [-2.537501, 0.480840, -1.607915, 0.034688, 0.788788]
├─ SHAPE: tensor: PT=[2,1,2,3] | TT=[2,1,2,3], mask: PT=[2,2,3] | TT=[2,2,3] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.038952, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [1.038952, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_inverse_sigmoid 
MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mmid_range[0m (Values around 0.5 - tests logit near 0)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [0.474908, 0.590143, 0.546399, 0.519732, 0.431204]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.98e-08, mean_diff=8.41e-09 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.100452, 0.364556, 0.186131, 0.078968, -0.276942]
├─ TT OUTPUT[0:5]: [-0.100452, 0.364556, 0.186131, 0.078968, -0.276942]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93muniform[0m (Uniform [0,1] values - tests full sigmoid range)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [0.115055, 0.609067, 0.133391, 0.240590, 0.327139]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=6.17e-09 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-2.040120, 0.443390, -1.871304, -1.149450, -0.721153]
├─ TT OUTPUT[0:5]: [-2.040120, 0.443390, -1.871304, -1.149450, -0.721153]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93muniform[0m (Uniform [0,1] values - tests full sigmoid range)
├─ INPUT: [4, 8, 16]
├─ INPUT input[0:5]: [0.834842, 0.104796, 0.744640, 0.360501, 0.359311]
├─ SHAPE: PyTorch=[4,8,16] | TTSim=[4,8,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=9.48e-09 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.620341, -2.145035, 1.070229, -0.573191, -0.578357]
├─ TT OUTPUT[0:5]: [1.620341, -2.145035, 1.070229, -0.573191, -0.578357]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mnear_zero[0m (Values near 0 (0.01-0.1) - tests logit near -inf)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [0.099011, 0.059459, 0.035330, 0.016956, 0.050002]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-07, mean_diff=1.49e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-2.208261, -2.761168, -3.307046, -4.060029, -2.944391]
├─ TT OUTPUT[0:5]: [-2.208261, -2.761168, -3.307046, -4.060029, -2.944391]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mnear_one[0m (Values near 1 (0.9-0.99) - tests logit near +inf)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [0.970545, 0.957135, 0.922414, 0.968227, 0.928177]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=2.61e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [3.494990, 3.105889, 2.475605, 3.416843, 2.559016]
├─ TT OUTPUT[0:5]: [3.494990, 3.105889, 2.475605, 3.416843, 2.559016]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mboundary[0m (Values at exact 0 and 1 - tests clamping)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [0.000000, 1.000000, 0.000000, 1.000000, 0.000000]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-11.512925, 11.512925, -11.512925, 11.512925, -11.512925]
├─ TT OUTPUT[0:5]: [-11.512925, 11.512925, -11.512925, 11.512925, -11.512925]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-value propagation)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-11.512925, -11.512925, -11.512925, -11.512925, -11.512925]
├─ TT OUTPUT[0:5]: [-11.512925, -11.512925, -11.512925, -11.512925, -11.512925]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [-1.300964, -1.247062, -1.926335, -1.891603, -1.683277]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-11.512925, -11.512925, -11.512925, -11.512925, -11.512925]
├─ TT OUTPUT[0:5]: [-11.512925, -11.512925, -11.512925, -11.512925, -11.512925]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [2, 4, 8]
├─ INPUT input[0:5]: [494601.625000, 228083.109375, 255473.906250, 396329.906250, 377315.093750]
├─ SHAPE: PyTorch=[2,4,8] | TTSim=[2,4,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [11.512925, 11.512925, 11.512925, 11.512925, 11.512925]
├─ TT OUTPUT[0:5]: [11.512925, 11.512925, 11.512925, 11.512925, 11.512925]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93mmid_range[0m (Values around 0.5 - tests logit near 0)
├─ INPUT: [1]
├─ INPUT input[0:5]: [0.535146]
├─ SHAPE: PyTorch=[1] | TTSim=[1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.49e-08, mean_diff=1.49e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.140817]
├─ TT OUTPUT[0:5]: [0.140817]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1minverse_sigmoid[0m
├─ EDGE CASE: [93muniform[0m (Uniform [0,1] values - tests full sigmoid range)
├─ INPUT: [1, 1, 1]
├─ INPUT input[0:5]: [0.823110]
├─ SHAPE: PyTorch=[1,1,1] | TTSim=[1,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.537564]
├─ TT OUTPUT[0:5]: [1.537564]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_shape_inference 
MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mbaseline[0m (interpolate: interpolate nearest 2x up)
├─ INPUT: {'shape': [2, 3, 10, 10], 'size': (20, 20), 'mode': 'nearest'}
├─ SHAPE: expected=[2,3,20,20] | got=[2,3,20,20], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mbaseline[0m (interpolate: interpolate bilinear 2x up)
├─ INPUT: {'shape': [1, 3, 8, 8], 'size': (16, 16), 'mode': 'bilinear'}
├─ SHAPE: expected=[1,3,16,16] | got=[1,3,16,16], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mbaseline[0m (interpolate: interpolate 4x downsample)
├─ INPUT: {'shape': [2, 3, 32, 32], 'size': (8, 8), 'mode': 'nearest'}
├─ SHAPE: expected=[2,3,8,8] | got=[2,3,8,8], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mbaseline[0m (interpolate: interpolate scale_factor=2)
├─ INPUT: {'shape': [1, 3, 10, 10], 'scale_factor': 2.0, 'mode': 'nearest'}
├─ SHAPE: expected=[1,3,20,20] | got=[1,3,20,20], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mbaseline[0m (interpolate: interpolate non-square)
├─ INPUT: {'shape': [1, 64, 6, 8], 'size': (12, 16), 'mode': 'nearest'}
├─ SHAPE: expected=[1,64,12,16] | got=[1,64,12,16], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mbaseline[0m (nested_tensor_from_tensor_list: nested_tensor 3 tensors)
├─ INPUT: {'tensor_shapes': [(3, 10, 15), (3, 12, 18), (3, 8, 20)]}
├─ SHAPE: expected=[3,3,12,20] | got=[3,3,12,20], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mminimum_input[0m (interpolate: interpolate minimum 1x1→2x2)
├─ INPUT: {'shape': [1, 1, 1, 1], 'size': (2, 2), 'mode': 'nearest'}
├─ SHAPE: expected=[1,1,2,2] | got=[1,1,2,2], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShape Inference[0m
├─ EDGE CASE: [93mminimum_input[0m (nested_tensor_from_tensor_list: nested_tensor minimum 1x1)
├─ INPUT: {'tensor_shapes': [(1, 1, 1), (1, 2, 3)]}
├─ SHAPE: expected=[2,1,2,3] | got=[2,1,2,3], data=None: True → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED

============================== slowest durations ===============================
0.04s call     workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_interpolate_bilinear
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_nested_tensor
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_interpolate_nearest
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_nested_tensor_from_tensor_list
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_inverse_sigmoid
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_misc_unit.py::test_interpolate_downsample

(15 durations < 0.005s hidden.  Use -vv to show these durations.)
============================== 7 passed in 0.25s ===============================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                             SHAPE       NUMERICAL   TOTAL
NestedTensor                       10/10       10/10       [92m✓ PASS[0m
interpolate (nearest)              11/11       11/11       [92m✓ PASS[0m
interpolate (bilinear)             10/10       10/10       [92m✓ PASS[0m
interpolate (downsample)           10/10       10/10       [92m✓ PASS[0m
nested_tensor_from_list            11/11       11/11       [92m✓ PASS[0m
inverse_sigmoid                    11/11       11/11       [92m✓ PASS[0m
Shape Inference                    8/8         N/A         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
TOTAL                              71/71          63/63       [92m✓ PASS[0m
═════════════════════════════════════════════════════════════════
