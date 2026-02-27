
═════════════════════════════════════════════════════════════════
BACKBONE UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 5 items

workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_frozen_batchnorm2d 
MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[1,64,8,8] | TTSim=[1,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.374533, 1.950705, 1.731985, 1.598651, 1.156013]
├─ TT OUTPUT[0:5]: [1.374533, 1.950705, 1.731985, 1.598651, 1.156013]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 64, 8, 8]
├─ INPUT input[0:5]: [1.013682, 1.081599, 1.258592, 1.027852, 1.631375]
├─ SHAPE: PyTorch=[2,64,8,8] | TTSim=[2,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.013677, 1.081593, 1.258585, 1.027847, 1.631367]
├─ TT OUTPUT[0:5]: [1.013677, 1.081593, 1.258585, 1.027847, 1.631367]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 128, 8, 8]
├─ INPUT input[0:5]: [1.081390, 1.011433, 1.427451, 1.410338, 1.988390]
├─ SHAPE: PyTorch=[1,128,8,8] | TTSim=[1,128,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.081385, 1.011428, 1.427443, 1.410331, 1.988380]
├─ TT OUTPUT[0:5]: [1.081385, 1.011428, 1.427443, 1.410331, 1.988380]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 64, 8, 16]
├─ INPUT input[0:5]: [1.845242, 1.087856, 1.857931, 1.483690, 1.837147]
├─ SHAPE: PyTorch=[1,64,8,16] | TTSim=[1,64,8,16] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.845233, 1.087851, 1.857921, 1.483683, 1.837138]
├─ TT OUTPUT[0:5]: [1.845233, 1.087851, 1.857921, 1.483683, 1.837138]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [-1.763316, -1.302599, -1.985766, -1.500817, -1.516539]
├─ SHAPE: PyTorch=[1,64,8,8] | TTSim=[1,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.763308, -1.302593, -1.985756, -1.500810, -1.516532]
├─ TT OUTPUT[0:5]: [-1.763308, -1.302593, -1.985756, -1.500810, -1.516532]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division by variance edge case)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,64,8,8] | TTSim=[1,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [0.366471, 0.499350, 2.112050, 2.614618, 0.672562]
├─ SHAPE: PyTorch=[1,64,8,8] | TTSim=[1,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.366469, 0.499348, 2.112039, 2.614605, 0.672558]
├─ TT OUTPUT[0:5]: [0.366469, 0.499348, 2.112039, 2.614605, 0.672558]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [0.000001, 0.000001, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,64,8,8] | TTSim=[1,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000001, 0.000001, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000001, 0.000001, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [729752.875000, 684619.000000, 53347.726562, 237818.500000, 679957.187500]
├─ SHAPE: PyTorch=[1,64,8,8] | TTSim=[1,64,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [729749.250000, 684615.562500, 53347.460938, 237817.312500, 679953.812500]
├─ TT OUTPUT[0:5]: [729749.250000, 684615.562500, 53347.460938, 237817.312500, 679953.812500]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 64, 1, 1]
├─ INPUT input[0:5]: [1.062239, 1.834653, 1.274939, 1.495903, 1.381065]
├─ SHAPE: PyTorch=[1,64,1,1] | TTSim=[1,64,1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.062234, 1.834643, 1.274932, 1.495896, 1.381058]
├─ TT OUTPUT[0:5]: [1.062234, 1.834643, 1.274932, 1.495896, 1.381058]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 1, 8, 8]
├─ INPUT input[0:5]: [1.880137, 1.571914, 1.952304, 1.818964, 1.415634]
├─ SHAPE: PyTorch=[1,1,8,8] | TTSim=[1,1,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.880127, 1.571906, 1.952294, 1.818955, 1.415627]
├─ TT OUTPUT[0:5]: [1.880127, 1.571906, 1.952294, 1.818955, 1.415627]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFrozenBatchNorm2d[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 64, 32, 32]
├─ INPUT input[0:5]: [1.495237, 1.802099, 1.933900, 1.269761, 1.130194]
├─ SHAPE: PyTorch=[1,64,32,32] | TTSim=[1,64,32,32] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.495229, 1.802090, 1.933890, 1.269755, 1.130188]
├─ TT OUTPUT[0:5]: [1.495229, 1.802090, 1.933890, 1.269755, 1.130188]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_resnet_bottleneck 
MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [1.374540, 1.950714, 1.731994, 1.598659, 1.156019]
├─ SHAPE: PyTorch=[1,256,8,8] | TTSim=[1,256,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.15e-06, mean_diff=3.82e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [2, 64, 8, 8]
├─ INPUT input[0:5]: [1.115055, 1.609066, 1.133391, 1.240590, 1.327139]
├─ SHAPE: PyTorch=[2,256,8,8] | TTSim=[2,256,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.67e-06, mean_diff=3.94e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93mnegative[0m (All negative values (-2.0 to -1.0) - tests sign handling)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [-1.834842, -1.104796, -1.744640, -1.360501, -1.359311]
├─ SHAPE: PyTorch=[1,256,8,8] | TTSim=[1,256,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-06, mean_diff=1.67e-07 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.651776, 1.651156, 1.364823, 1.920387, 1.738803]
├─ TT OUTPUT[0:5]: [1.651776, 1.651156, 1.364823, 1.920387, 1.738803]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93mzeros[0m (All zeros - tests division by variance edge case)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=[1,256,8,8] | TTSim=[1,256,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=4.77e-07, mean_diff=3.47e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.469240, 0.447957, 0.447957, 0.447957, 0.447957]
├─ TT OUTPUT[0:5]: [0.469240, 0.447957, 0.447957, 0.447957, 0.447957]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93mmixed[0m (Mix of positive/negative values - tests real-world distribution)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [1.169752, 2.462391, 1.643800, -1.598457, 0.824106]
├─ SHAPE: PyTorch=[1,256,8,8] | TTSim=[1,256,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.86e-06, mean_diff=9.86e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [1.899840, 0.000000, 0.000000, 0.671133, 0.690961]
├─ TT OUTPUT[0:5]: [1.899840, 0.000000, 0.000000, 0.671133, 0.690960]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93msmall[0m (Very small values (~1e-6) - tests numerical precision near zero)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [0.000000, 0.000001, 0.000001, 0.000000, 0.000001]
├─ SHAPE: PyTorch=[1,256,8,8] | TTSim=[1,256,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.96e-07, mean_diff=3.73e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.469240, 0.447957, 0.447957, 0.447957, 0.447957]
├─ TT OUTPUT[0:5]: [0.469240, 0.447957, 0.447957, 0.447957, 0.447957]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93mlarge[0m (Very large values (~1e6) - tests numerical overflow handling)
├─ INPUT: [1, 64, 8, 8]
├─ INPUT input[0:5]: [17490.271484, 891573.250000, 284861.187500, 298976.406250, 792034.250000]
├─ SHAPE: PyTorch=[1,256,8,8] | TTSim=[1,256,8,8] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=5.00e-01, mean_diff=1.25e-02 → [91m[1m✗ FAIL[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ FAILURE REASON: [91m[1mmax_diff=5.00e-01 exceeds atol=0.01[0m
└─ RESULT: [91m[1m✗ FAIL[0m

MODULE: [1mResNetBottleneck[0m
├─ EDGE CASE: [93mpositive[0m (Standard positive values (1.0 - 2.0) - baseline test)
├─ INPUT: [1, 64, 4, 4]
├─ INPUT input[0:5]: [1.300964, 1.247062, 1.926335, 1.891603, 1.683277]
├─ SHAPE: PyTorch=[1,256,4,4] | TTSim=[1,256,4,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=2.38e-06, mean_diff=4.05e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
FAILED
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_backbone_shapes 
MODULE: [1mBackbone[0m
├─ EDGE CASE: [93mbaseline[0m (ResNet50 32x32 batch=1)
├─ INPUT: [1, 3, 32, 32]
├─ SHAPE: 3/3 layers match → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mBackbone[0m
├─ EDGE CASE: [93mscale[0m (ResNet50 64x64 batch=1)
├─ INPUT: [1, 3, 64, 64]
├─ SHAPE: 3/3 layers match → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mBackbone[0m
├─ EDGE CASE: [93mbatch[0m (ResNet50 32x32 batch=2)
├─ INPUT: [2, 3, 32, 32]
├─ SHAPE: 3/3 layers match → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mBackbone[0m
├─ EDGE CASE: [93mminimum_input[0m (ResNet50 16x16 minimum)
├─ INPUT: [1, 3, 16, 16]
├─ SHAPE: 3/3 layers match → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_shapes 
MODULE: [1mJoiner[0m
├─ EDGE CASE: [93mbaseline[0m (Joiner 32x32 batch=1)
├─ INPUT: [1, 3, 32, 32]
├─ SHAPE: 3/3 feats + 3/3 pos → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mJoiner[0m
├─ EDGE CASE: [93mscale[0m (Joiner 64x64 batch=1)
├─ INPUT: [1, 3, 64, 64]
├─ SHAPE: 3/3 feats + 3/3 pos → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mJoiner[0m
├─ EDGE CASE: [93mbatch[0m (Joiner 32x32 batch=2)
├─ INPUT: [2, 3, 32, 32]
├─ SHAPE: 3/3 feats + 3/3 pos → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mJoiner[0m
├─ EDGE CASE: [93mminimum_input[0m (Joiner 16x16 minimum)
├─ INPUT: [1, 3, 16, 16]
├─ SHAPE: 3/3 feats + 3/3 pos → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_pos_enc_numerical 
MODULE: [1mPositionalEncoding[0m
├─ EDGE CASE: [93mbaseline[0m (Pos-enc 32x32 batch=1)
├─ INPUT: [1, 3, 32, 32]
├─ INPUT input[0:5]: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=3 layers | TTSim=3 layers → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.707107, 0.707107, 0.707107, 0.707107, 0.707107]
├─ TT OUTPUT[0:5]: [0.707107, 0.707107, 0.707107, 0.707107, 0.707107]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionalEncoding[0m
├─ EDGE CASE: [93mscale[0m (Pos-enc 64x64 batch=1)
├─ INPUT: [1, 3, 64, 64]
├─ INPUT input[0:5]: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=3 layers | TTSim=3 layers → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.382683, 0.382683, 0.382683, 0.382683, 0.382683]
├─ TT OUTPUT[0:5]: [0.382683, 0.382683, 0.382683, 0.382683, 0.382683]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionalEncoding[0m
├─ EDGE CASE: [93mbatch[0m (Pos-enc 32x32 batch=2)
├─ INPUT: [2, 3, 32, 32]
├─ INPUT input[0:5]: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=3 layers | TTSim=3 layers → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.707107, 0.707107, 0.707107, 0.707107, 0.707107]
├─ TT OUTPUT[0:5]: [0.707107, 0.707107, 0.707107, 0.707107, 0.707107]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mPositionalEncoding[0m
├─ EDGE CASE: [93mminimum_input[0m (Pos-enc 8x8 minimum)
├─ INPUT: [1, 3, 8, 8]
├─ INPUT input[0:5]: [1.926915, 1.487284, 0.900717, -2.105521, 0.678418]
├─ INPUT mask[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ SHAPE: PyTorch=3 layers | TTSim=3 layers → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.000003, -1.000000, 0.408754, -0.912645, 0.707343]
├─ TT OUTPUT[0:5]: [0.000003, -1.000000, 0.408754, -0.912645, 0.707343]
└─ RESULT: [92m✓ PASS[0m
PASSED

=================================== FAILURES ===================================
____________________________ test_resnet_bottleneck ____________________________
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py:923: in test_resnet_bottleneck
    assert passed == len(
E   AssertionError: ResNetBottleneck: 7/8 passed
E   assert 7 == 8
E    +  where 8 = len([('layer1[0] 64->256 stride=1', 1, 64, 256, 8, 8, ...), ('layer1[0] batch=2', 2, 64, 256, 8, 8, ...), ('layer1[0] negative input', 1, 64, 256, 8, 8, ...), ('layer1[0] zero values', 1, 64, 256, 8, 8, ...), ('layer1[0] mixed input', 1, 64, 256, 8, 8, ...), ('layer1[0] small values', 1, 64, 256, 8, 8, ...), ...])
=============================== warnings summary ===============================
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_resnet_bottleneck
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_backbone_shapes
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_shapes
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_pos_enc_numerical
  /home/aughag/.local/lib/python3.13/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
    warnings.warn(

workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_resnet_bottleneck
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_backbone_shapes
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_shapes
workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_pos_enc_numerical
  /home/aughag/.local/lib/python3.13/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
    warnings.warn(msg)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================== slowest durations ===============================
13.85s call     workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_resnet_bottleneck
8.55s call     workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_backbone_shapes
8.50s call     workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_shapes
8.42s call     workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_joiner_pos_enc_numerical
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_frozen_batchnorm2d

(10 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ============================
FAILED workloads/Deformable_DETR/reference/unit_tests/test_backbone_unit.py::test_resnet_bottleneck
=================== 1 failed, 4 passed, 8 warnings in 39.39s ===================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                      SHAPE       NUMERICAL   TOTAL
FrozenBatchNorm2d           12/12       12/12       [92m✓ PASS[0m
ResNetBottleneck            8/8         7/8         [91m[1m✗ FAIL[0m
Backbone                    4/4         N/A         [92m✓ PASS[0m
Joiner                      4/4         N/A         [92m✓ PASS[0m
PositionalEncoding          4/4         4/4         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
TOTAL                       32/32          23/24       [91m[1m✗ FAIL[0m

[91m[1mFAILED TESTS:[0m
- ResNetBottleneck | large values | max_diff=5.00e-01 > atol=0.01
═════════════════════════════════════════════════════════════════
