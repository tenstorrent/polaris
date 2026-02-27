
═════════════════════════════════════════════════════════════════
POSITION ENCODING UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 6 items

workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_position_embedding_sine 
MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mbaseline[0m (Standard configuration with random input)
├─ INPUT: [2, 256, 28, 28]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,256,28,28] | TTSim=[2,256,28,28] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.111964, 0.111964, 0.111964, 0.111964, 0.111964]
├─ TT OUTPUT[0:5]: [0.111964, 0.111964, 0.111964, 0.111964, 0.111964]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mbaseline[0m (Standard configuration with random input)
├─ INPUT: [1, 64, 16, 16]
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,64,16,16] | TTSim=[1,64,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
├─ TT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mbatch[0m (Multi-batch test)
├─ INPUT: [4, 256, 28, 28]
├─ INPUT input[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[4,256,28,28] | TTSim=[4,256,28,28] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.111964, 0.111964, 0.111964, 0.111964, 0.111964]
├─ TT OUTPUT[0:5]: [0.111964, 0.111964, 0.111964, 0.111964, 0.111964]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mnon_square[0m (Non-square spatial dimensions)
├─ INPUT: [2, 128, 14, 28]
├─ INPUT input[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,128,14,28] | TTSim=[2,128,14,28] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.222521, 0.222521, 0.222521, 0.222521, 0.222521]
├─ TT OUTPUT[0:5]: [0.222521, 0.222521, 0.222521, 0.222521, 0.222521]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mnon_square[0m (Non-square spatial dimensions)
├─ INPUT: [1, 64, 32, 8]
├─ INPUT input[0:5]: [1.783832, 1.634834, 1.249043, 1.758076, 1.313077]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,32,8] | TTSim=[1,128,32,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.098017, 0.098017, 0.098017, 0.098017, 0.098017]
├─ TT OUTPUT[0:5]: [0.098017, 0.098017, 0.098017, 0.098017, 0.098017]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [-1.113488, -1.974483, -1.728735, -1.351468, -1.707605]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
├─ TT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-mask cumsum edge case)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
├─ TT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [-2.086318, -1.641711, 1.330292, 3.645254, -2.883167]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
├─ TT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
├─ TT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [675731.437500, 44712.183594, 343303.687500, 644019.750000, 284213.000000]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
├─ TT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size - degenerate/boundary case)
├─ INPUT: [1, 64, 1, 1]
├─ INPUT input[0:5]: [1.823110, 1.026118, 1.210771, 1.618422, 1.098284]
├─ INPUT mask[0:5]: [0.000000]
├─ SHAPE: PyTorch=[1,64,1,1] | TTSim=[1,64,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000003, -1.000000, 0.980883, -0.194600, 0.837918]
├─ TT OUTPUT[0:5]: [0.000003, -1.000000, 0.980883, -0.194600, 0.837918]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93medge_shape[0m (Sine 1xW strip)
├─ INPUT: [1, 64, 1, 16]
├─ INPUT input[0:5]: [1.846662, 1.561166, 1.454875, 1.352175, 1.585851]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,64,1,16] | TTSim=[1,64,1,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000003, 0.000003, 0.000003, 0.000003, 0.000003]
├─ TT OUTPUT[0:5]: [0.000003, 0.000003, 0.000003, 0.000003, 0.000003]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93medge_shape[0m (Sine Hx1 strip)
├─ INPUT: [1, 64, 16, 1]
├─ INPUT input[0:5]: [1.420183, 1.363240, 1.184877, 1.518283, 1.008605]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,64,16,1] | TTSim=[1,64,16,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.555570, 0.831470, 0.980785, 0.980785]
├─ TT OUTPUT[0:5]: [0.195090, 0.555570, 0.831470, 0.980785, 0.980785]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93munnormalised[0m (Unnormalised position embedding (no scale))
├─ INPUT: [1, 64, 16, 16]
├─ INPUT input[0:5]: [1.093108, 1.971656, 1.483860, 1.242523, 1.531124]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,64,16,16] | TTSim=[1,64,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.841471, 0.841471, 0.841471, 0.841471, 0.841471]
├─ TT OUTPUT[0:5]: [0.841471, 0.841471, 0.841471, 0.841471, 0.841471]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mparam_sweep[0m (Sine D=256 T=10000)
├─ INPUT: [1, 128, 12, 12]
├─ INPUT input[0:5]: [1.984192, 1.333412, 1.673702, 1.196390, 1.354446]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,512,12,12] | TTSim=[1,512,12,12] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.258819, 0.258819, 0.258819, 0.258819, 0.258819]
├─ TT OUTPUT[0:5]: [0.258819, 0.258819, 0.258819, 0.258819, 0.258819]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mparam_sweep[0m (Sine D=64 T=100)
├─ INPUT: [1, 128, 12, 12]
├─ INPUT input[0:5]: [1.087350, 1.230477, 1.411061, 1.310783, 1.565956]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,12,12] | TTSim=[1,128,12,12] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.258819, 0.258819, 0.258819, 0.258819, 0.258819]
├─ TT OUTPUT[0:5]: [0.258819, 0.258819, 0.258819, 0.258819, 0.258819]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingSine[0m
├─ EDGE CASE: [93mparam_sweep[0m (Sine D=64 T=20000)
├─ INPUT: [1, 128, 12, 12]
├─ INPUT input[0:5]: [1.365106, 1.451206, 1.496060, 1.075622, 1.571762]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,12,12] | TTSim=[1,128,12,12] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.258819, 0.258819, 0.258819, 0.258819, 0.258819]
├─ TT OUTPUT[0:5]: [0.258819, 0.258819, 0.258819, 0.258819, 0.258819]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_position_embedding_learned 
MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mbaseline[0m (Standard configuration with random input)
├─ INPUT: [2, 256, 28, 28]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[2,256,28,28] | TTSim=[2,256,28,28] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.908265, 0.193723, 0.658249, 0.830067, 0.102959]
├─ TT OUTPUT[0:5]: [0.908265, 0.193723, 0.658249, 0.830067, 0.102959]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mbaseline[0m (Standard configuration with random input)
├─ INPUT: [1, 64, 10, 10]
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: PyTorch=[1,64,10,10] | TTSim=[1,64,10,10] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.125928, 0.923728, 0.745954, 0.255958, 0.060664]
├─ TT OUTPUT[0:5]: [0.125928, 0.923728, 0.745954, 0.255958, 0.060664]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mnon_square[0m (Non-square spatial dimensions)
├─ INPUT: [2, 128, 14, 28]
├─ INPUT input[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ SHAPE: PyTorch=[2,128,14,28] | TTSim=[2,128,14,28] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.126526, 0.403979, 0.666558, 0.697680, 0.506819]
├─ TT OUTPUT[0:5]: [0.126526, 0.403979, 0.666558, 0.697680, 0.506819]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mbatch[0m (Multi-batch test)
├─ INPUT: [4, 256, 28, 28]
├─ INPUT input[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
├─ SHAPE: PyTorch=[4,256,28,28] | TTSim=[4,256,28,28] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.576294, 0.441611, 0.265361, 0.233555, 0.036517]
├─ TT OUTPUT[0:5]: [0.576294, 0.441611, 0.265361, 0.233555, 0.036517]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mparam_sweep[0m (Learned 20x20 D=256)
├─ INPUT: [1, 512, 20, 20]
├─ INPUT input[0:5]: [1.783832, 1.634834, 1.249043, 1.758076, 1.313077]
├─ SHAPE: PyTorch=[1,512,20,20] | TTSim=[1,512,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.672881, 0.731578, 0.726486, 0.146807, 0.289830]
├─ TT OUTPUT[0:5]: [0.672881, 0.731578, 0.726486, 0.146807, 0.289830]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [-1.113488, -1.974483, -1.728735, -1.351468, -1.707605]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.871868, 0.059559, 0.204412, 0.386751, 0.898868]
├─ TT OUTPUT[0:5]: [0.871868, 0.059559, 0.204412, 0.386751, 0.898868]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-mask cumsum edge case)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.003351, 0.259137, 0.551184, 0.143520, 0.416025]
├─ TT OUTPUT[0:5]: [0.003351, 0.259137, 0.551184, 0.143520, 0.416025]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [-2.086318, -1.641711, 1.330292, 3.645254, -2.883167]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.302319, 0.421887, 0.671263, 0.117893, 0.109669]
├─ TT OUTPUT[0:5]: [0.302319, 0.421887, 0.671263, 0.117893, 0.109669]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.279060, 0.342881, 0.630847, 0.629719, 0.032477]
├─ TT OUTPUT[0:5]: [0.279060, 0.342881, 0.630847, 0.629719, 0.032477]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [675731.437500, 44712.183594, 343303.687500, 644019.750000, 284213.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.083631, 0.673881, 0.466032, 0.576406, 0.972970]
├─ TT OUTPUT[0:5]: [0.083631, 0.673881, 0.466032, 0.576406, 0.972970]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionEmbeddingLearned[0m
├─ EDGE CASE: [93mminimum_input[0m (Smallest valid input size - degenerate/boundary case)
├─ INPUT: [1, 64, 4, 4]
├─ INPUT input[0:5]: [1.823110, 1.026118, 1.210771, 1.618422, 1.098284]
├─ SHAPE: PyTorch=[1,64,4,4] | TTSim=[1,64,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.239545, 0.264373, 0.456213, 0.259196, 0.239545]
├─ TT OUTPUT[0:5]: [0.239545, 0.264373, 0.456213, 0.259196, 0.239545]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_sine_mask_variations 
MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mno_mask[0m (No masked positions — uniform cumsum)
├─ INPUT: [2, 128, 20, 20]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,128,20,20] | TTSim=[2,128,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.156434, 0.156434, 0.156434, 0.156434, 0.156434]
├─ TT OUTPUT[0:5]: [0.156434, 0.156434, 0.156434, 0.156434, 0.156434]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mtop_rows[0m (Top rows masked — shifts y cumsum)
├─ INPUT: [2, 128, 20, 20]
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ INPUT mask[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: PyTorch=[2,128,20,20] | TTSim=[2,128,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.207912, -0.207912, -0.207912, -0.207912, -0.207912]
├─ TT OUTPUT[0:5]: [-0.207912, -0.207912, -0.207912, -0.207912, -0.207912]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mleft_cols[0m (Left columns masked — shifts x cumsum)
├─ INPUT: [2, 128, 20, 20]
├─ INPUT input[0:5]: [1.834842, 1.104796, 1.744640, 1.360501, 1.359311]
├─ INPUT mask[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: PyTorch=[2,128,20,20] | TTSim=[2,128,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.096261, -0.096261, -0.096261, -0.096261, -0.096261]
├─ TT OUTPUT[0:5]: [-0.096261, -0.096261, -0.096261, -0.096261, -0.096261]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mcheckerboard[0m (Alternating masked pixels — irregular cumsum)
├─ INPUT: [2, 128, 20, 20]
├─ INPUT input[0:5]: [1.989012, 1.549545, 1.281447, 1.077290, 1.444469]
├─ INPUT mask[0:5]: [1.000000, 0.000000, 1.000000, 0.000000, 1.000000]
├─ SHAPE: PyTorch=[2,128,20,20] | TTSim=[2,128,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.309017, 0.309017, -0.309017, 0.309017, -0.309017]
├─ TT OUTPUT[0:5]: [-0.309017, 0.309017, -0.309017, 0.309017, -0.309017]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mbottom_right[0m (Bottom-right quadrant masked)
├─ INPUT: [2, 128, 20, 20]
├─ INPUT input[0:5]: [1.783832, 1.634834, 1.249043, 1.758076, 1.313077]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[2,128,20,20] | TTSim=[2,128,20,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.156434, 0.156434, 0.156434, 0.156434, 0.156434]
├─ TT OUTPUT[0:5]: [0.156434, 0.156434, 0.156434, 0.156434, 0.156434]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [-1.113488, -1.974483, -1.728735, -1.351468, -1.707605]
├─ INPUT mask[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.258819, -0.258819, -0.258819, -0.258819, -0.258819]
├─ TT OUTPUT[0:5]: [-0.258819, -0.258819, -0.258819, -0.258819, -0.258819]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests zero-mask cumsum edge case)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT mask[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.258819, -0.258819, -0.258819, -0.258819, -0.258819]
├─ TT OUTPUT[0:5]: [-0.258819, -0.258819, -0.258819, -0.258819, -0.258819]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [-2.086318, -1.641711, 1.330292, 3.645254, -2.883167]
├─ INPUT mask[0:5]: [1.000000, 0.000000, 1.000000, 0.000000, 1.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.382683, 0.382683, -0.382683, 0.382683, -0.382683]
├─ TT OUTPUT[0:5]: [-0.382683, 0.382683, -0.382683, 0.382683, -0.382683]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ INPUT mask[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 1.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.096261, -0.096261, -0.096261, -0.096261, -0.096261]
├─ TT OUTPUT[0:5]: [-0.096261, -0.096261, -0.096261, -0.096261, -0.096261]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [1, 128, 16, 16]
├─ INPUT input[0:5]: [675731.437500, 44712.183594, 343303.687500, 644019.750000, 284213.000000]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,16,16] | TTSim=[1,128,16,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
├─ TT OUTPUT[0:5]: [0.195090, 0.195090, 0.195090, 0.195090, 0.195090]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineMaskVariations[0m
├─ EDGE CASE: [93mtop_rows[0m (Top rows masked — shifts y cumsum)
├─ INPUT: [1, 64, 4, 4]
├─ INPUT input[0:5]: [1.823110, 1.026118, 1.210771, 1.618422, 1.098284]
├─ INPUT mask[0:5]: [1.000000, 1.000000, 1.000000, 1.000000, 0.000000]
├─ SHAPE: PyTorch=[1,128,4,4] | TTSim=[1,128,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.866025, -0.866025, -0.866025, -0.866025, 0.866025]
├─ TT OUTPUT[0:5]: [-0.866025, -0.866025, -0.866025, -0.866025, 0.866025]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_sine_component_analysis 
MODULE: [1mSineComponentAnalysis[0m
├─ EDGE CASE: [93mbaseline[0m (Components 24x24 standard)
├─ INPUT: [2, 256, 24, 24]
├─ SHAPE: PyTorch=[2,256,24,24] | TTSim=[2,256,24,24] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ Y-COMPONENT [0:128]: [92m✓ MATCH[0m
├─ X-COMPONENT [128:256]: [92m✓ MATCH[0m
├─ Y-POS COLUMN INVARIANT: [92m✓ YES[0m
├─ X-POS ROW INVARIANT: [92m✓ YES[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineComponentAnalysis[0m
├─ EDGE CASE: [93mbaseline[0m (Components 12x12 small)
├─ INPUT: [1, 128, 12, 12]
├─ SHAPE: PyTorch=[1,128,12,12] | TTSim=[1,128,12,12] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ Y-COMPONENT [0:64]: [92m✓ MATCH[0m
├─ X-COMPONENT [64:128]: [92m✓ MATCH[0m
├─ Y-POS COLUMN INVARIANT: [92m✓ YES[0m
├─ X-POS ROW INVARIANT: [92m✓ YES[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineComponentAnalysis[0m
├─ EDGE CASE: [93mnon_square[0m (Components 8x16 non-square)
├─ INPUT: [2, 128, 8, 16]
├─ SHAPE: PyTorch=[2,128,8,16] | TTSim=[2,128,8,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ Y-COMPONENT [0:64]: [92m✓ MATCH[0m
├─ X-COMPONENT [64:128]: [92m✓ MATCH[0m
├─ Y-POS COLUMN INVARIANT: [92m✓ YES[0m
├─ X-POS ROW INVARIANT: [92m✓ YES[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mSineComponentAnalysis[0m
├─ EDGE CASE: [93mminimum_input[0m (Components 4x4 minimum)
├─ INPUT: [1, 64, 4, 4]
├─ SHAPE: PyTorch=[1,64,4,4] | TTSim=[1,64,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS[0m
├─ Y-COMPONENT [0:32]: [92m✓ MATCH[0m
├─ X-COMPONENT [32:64]: [92m✓ MATCH[0m
├─ Y-POS COLUMN INVARIANT: [92m✓ YES[0m
├─ X-POS ROW INVARIANT: [92m✓ YES[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_build_position_encoding 
MODULE: [1mbuild_position_encoding[0m
├─ CONFIG: position_embedding='sine', hidden_dim=256
├─ TYPE: PT=PositionEmbeddingSine | TT=PositionEmbeddingSine → [92m✓ MATCH[0m
├─ SHAPE: PyTorch=[2,256,14,14] | TTSim=[2,256,14,14] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00 → [92m✓ PASS[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mbuild_position_encoding[0m
├─ CONFIG: position_embedding='learned', hidden_dim=256
├─ TYPE: PT=PositionEmbeddingLearned | TT=PositionEmbeddingLearned → [92m✓ MATCH[0m
├─ SHAPE: PyTorch=[2,256,14,14] | TTSim=[2,256,14,14] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mbuild_position_encoding[0m
├─ CONFIG: position_embedding='v2', hidden_dim=256
├─ TYPE: PT=PositionEmbeddingSine | TT=PositionEmbeddingSine → [92m✓ MATCH[0m
├─ SHAPE: PyTorch=[2,256,14,14] | TTSim=[2,256,14,14] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00 → [92m✓ PASS[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mbuild_position_encoding[0m
├─ CONFIG: position_embedding='v3', hidden_dim=256
├─ TYPE: PT=PositionEmbeddingLearned | TT=PositionEmbeddingLearned → [92m✓ MATCH[0m
├─ SHAPE: PyTorch=[2,256,14,14] | TTSim=[2,256,14,14] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mbuild_position_encoding[0m
├─ CONFIG: position_embedding='v2', hidden_dim=128
├─ TYPE: PT=PositionEmbeddingSine | TT=PositionEmbeddingSine → [92m✓ MATCH[0m
├─ SHAPE: PyTorch=[2,128,14,14] | TTSim=[2,128,14,14] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00 → [92m✓ PASS[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mbuild_position_encoding[0m
├─ CONFIG: position_embedding='sine', hidden_dim=512
├─ TYPE: PT=PositionEmbeddingSine | TT=PositionEmbeddingSine → [92m✓ MATCH[0m
├─ SHAPE: PyTorch=[2,512,14,14] | TTSim=[2,512,14,14] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00 → [92m✓ PASS[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mbuild_position_encoding[0m
├─ CONFIG: position_embedding='banana', hidden_dim=256
├─ PT RAISES: [92m✓ YES[0m
├─ TT RAISES: [92m✓ YES[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_shape_inference 
MODULE: [1mShapeInference (sine)[0m
├─ EDGE CASE: [93mbaseline[0m (Shape sine 28x28)
├─ INPUT: [2, 256, 28, 28]
├─ SHAPE: Expected=[2,256,28,28] | TTSim=[2,256,28,28] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference (sine)[0m
├─ EDGE CASE: [93mbaseline[0m (Shape sine 14x14)
├─ INPUT: [1, 128, 14, 14]
├─ SHAPE: Expected=[1,128,14,14] | TTSim=[1,128,14,14] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference (sine)[0m
├─ EDGE CASE: [93mnon_square[0m (Shape sine 7x11)
├─ INPUT: [2, 256, 7, 11]
├─ SHAPE: Expected=[2,256,7,11] | TTSim=[2,256,7,11] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference (sine)[0m
├─ EDGE CASE: [93mminimum_input[0m (Shape sine 1x1 minimum)
├─ INPUT: [1, 64, 1, 1]
├─ SHAPE: Expected=[1,64,1,1] | TTSim=[1,64,1,1] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference (learned)[0m
├─ EDGE CASE: [93mbaseline[0m (Shape learned 28x28)
├─ INPUT: [2, 256, 28, 28]
├─ SHAPE: Expected=[2,256,28,28] | TTSim=[2,256,28,28] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference (learned)[0m
├─ EDGE CASE: [93mbaseline[0m (Shape learned 10x10)
├─ INPUT: [1, 128, 10, 10]
├─ SHAPE: Expected=[1,128,10,10] | TTSim=[1,128,10,10] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mShapeInference (learned)[0m
├─ EDGE CASE: [93mminimum_input[0m (Shape learned 4x4 min)
├─ INPUT: [1, 64, 4, 4]
├─ SHAPE: Expected=[1,64,4,4] | TTSim=[1,64,4,4] → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED

============================== slowest durations ===============================
0.09s call     workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_position_embedding_sine
0.07s call     workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_position_embedding_learned
0.04s call     workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_sine_mask_variations
0.02s call     workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_sine_component_analysis
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_build_position_encoding
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_position_encoding_unit.py::test_shape_inference

(12 durations < 0.005s hidden.  Use -vv to show these durations.)
============================== 6 passed in 0.26s ===============================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                          SHAPE       NUMERICAL   TOTAL
PositionEmbeddingSine           17/17       17/17       [92m✓ PASS[0m
PosEmbLearned                   11/11       11/11       [92m✓ PASS[0m
SineMaskVariations              11/11       11/11       [92m✓ PASS[0m
SineComponents                  4/4         4/4         [92m✓ PASS[0m
build_factory                   7/7         N/A         [92m✓ PASS[0m
ShapeInference                  7/7         N/A         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
TOTAL                           57/57          43/43       [92m✓ PASS[0m
═════════════════════════════════════════════════════════════════
