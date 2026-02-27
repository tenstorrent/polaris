# MS Deform Attn Core Unit Test Report
**PyTorch vs TTSim Comparison** | **48/48 passed** | PASS
Generated: 2026-02-20 14:49:12 | Exit Code: 0

---

## Summary

| Module | Passed | Total | Status |
|--------|--------|-------|--------|
| ShapeInference | 10 | 10 | PASS |
| E2E_Numerical | 19 | 19 | PASS |
| PerLevelGridSample | 10 | 10 | PASS |
| IntermediateSteps | 9 | 9 | PASS |

**Total: 48/48 tests passed**

---

## ShapeInference (10/10 PASS)
*Shape inference mode (data=None) — verifies output shape [N, Lq, M*D]*

| # | Test Case | Edge Case | Input | Expected | Actual | Result |
|:--|:----------|:----------|:------|:---------|:-------|:-------|
| 0 | Standard multi-level | standard_multi_level | `value[2,3343,8,32] sampling[2,100,8,4,4,2] attn[2,100,8,4,4]` | `[2, 100, 256]` | `[2, 100, 256]` | ✅ PASS |
| 1 | Single level L=1 | single_level | `value[1,1024,4,16] sampling[1,50,4,1,2,2] attn[1,50,4,1,2]` | `[1, 50, 64]` | `[1, 50, 64]` | ✅ PASS |
| 2 | Many points P=8 | many_points | `value[1,320,4,16] sampling[1,64,4,2,8,2] attn[1,64,4,2,8]` | `[1, 64, 64]` | `[1, 64, 64]` | ✅ PASS |
| 3 | Large batch N=4 | large_batch | `value[4,525,8,32] sampling[4,80,8,3,4,2] attn[4,80,8,3,4]` | `[4, 80, 256]` | `[4, 80, 256]` | ✅ PASS |
| 4 | Single query Lq=1 | single_query | `value[2,80,4,16] sampling[2,1,4,2,4,2] attn[2,1,4,2,4]` | `[2, 1, 64]` | `[2, 1, 64]` | ✅ PASS |
| 5 | Many heads M=16 | many_heads | `value[1,320,16,8] sampling[1,50,16,2,4,2] attn[1,50,16,2,4]` | `[1, 50, 128]` | `[1, 50, 128]` | ✅ PASS |
| 6 | Non-square spatial | non_square_spatial | `value[1,160,4,16] sampling[1,30,4,2,4,2] attn[1,30,4,2,4]` | `[1, 30, 64]` | `[1, 30, 64]` | ✅ PASS |
| 7 | Minimal config | minimal_config | `value[1,4,1,4] sampling[1,2,1,1,1,2] attn[1,2,1,1,1]` | `[1, 2, 4]` | `[1, 2, 4]` | ✅ PASS |
| 8 | Single element pool P=1 | single_element_pooling | `value[1,16,2,8] sampling[1,10,2,1,1,2] attn[1,10,2,1,1]` | `[1, 10, 16]` | `[1, 10, 16]` | ✅ PASS |
| 9 | Minimum input size | minimum_input | `value[1,1,1,4] sampling[1,1,1,1,1,2] attn[1,1,1,1,1]` | `[1, 1, 4]` | `[1, 1, 4]` | ✅ PASS |

---

### 🟢 TEST[0] Standard multi-level

**Edge Case:** `standard_multi_level` — 4 levels, typical multi-scale — baseline shape

**Input:** `value[2,3343,8,32] sampling[2,100,8,4,4,2] attn[2,100,8,4,4]`

**Expected:** `[2, 100, 256]` → **Actual:** `[2, 100, 256]`


---

### 🟢 TEST[1] Single level L=1

**Edge Case:** `single_level` — L=1, single feature level — minimal level loop

**Input:** `value[1,1024,4,16] sampling[1,50,4,1,2,2] attn[1,50,4,1,2]`

**Expected:** `[1, 50, 64]` → **Actual:** `[1, 50, 64]`


---

### 🟢 TEST[2] Many points P=8

**Edge Case:** `many_points` — P=8 sampling points — more interpolation

**Input:** `value[1,320,4,16] sampling[1,64,4,2,8,2] attn[1,64,4,2,8]`

**Expected:** `[1, 64, 64]` → **Actual:** `[1, 64, 64]`


---

### 🟢 TEST[3] Large batch N=4

**Edge Case:** `large_batch` — N=4 batch size — batch scalability

**Input:** `value[4,525,8,32] sampling[4,80,8,3,4,2] attn[4,80,8,3,4]`

**Expected:** `[4, 80, 256]` → **Actual:** `[4, 80, 256]`


---

### 🟢 TEST[4] Single query Lq=1

**Edge Case:** `single_query` — Lq=1 — minimal query count

**Input:** `value[2,80,4,16] sampling[2,1,4,2,4,2] attn[2,1,4,2,4]`

**Expected:** `[2, 1, 64]` → **Actual:** `[2, 1, 64]`


---

### 🟢 TEST[5] Many heads M=16

**Edge Case:** `many_heads` — M=16 attention heads — head scalability

**Input:** `value[1,320,16,8] sampling[1,50,16,2,4,2] attn[1,50,16,2,4]`

**Expected:** `[1, 50, 128]` → **Actual:** `[1, 50, 128]`


---

### 🟢 TEST[6] Non-square spatial

**Edge Case:** `non_square_spatial` — Non-square spatial dims (H!=W) — asymmetric grids

**Input:** `value[1,160,4,16] sampling[1,30,4,2,4,2] attn[1,30,4,2,4]`

**Expected:** `[1, 30, 64]` → **Actual:** `[1, 30, 64]`


---

### 🟢 TEST[7] Minimal config

**Edge Case:** `minimal_config` — Tiny dims for quick smoke test

**Input:** `value[1,4,1,4] sampling[1,2,1,1,1,2] attn[1,2,1,1,1]`

**Expected:** `[1, 2, 4]` → **Actual:** `[1, 2, 4]`


---

### 🟢 TEST[8] Single element pool P=1

**Edge Case:** `single_element_pooling` — P=1, single sampling point — minimal interpolation

**Input:** `value[1,16,2,8] sampling[1,10,2,1,1,2] attn[1,10,2,1,1]`

**Expected:** `[1, 10, 16]` → **Actual:** `[1, 10, 16]`


---

### 🟢 TEST[9] Minimum input size

**Edge Case:** `minimum_input` — Smallest valid config (all dims minimal) — degenerate case

**Input:** `value[1,1,1,4] sampling[1,1,1,1,1,2] attn[1,1,1,1,1]`

**Expected:** `[1, 1, 4]` → **Actual:** `[1, 1, 4]`


---

## E2E_Numerical (19/19 PASS)
*End-to-end numerical validation — full ms_deform_attn_core pipeline*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Standard 4 levels | standard | `value[2,3343,8,32] sampling[2,100,8,4,4,2] attn[2,100,8,4,4]` | `[2, 100, 256]` | 5.25e-06 | 1.15e-07 | ✅ PASS |
| 1 | Positive values | positive | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 7.15e-07 | 1.18e-07 | ✅ PASS |
| 2 | Negative values | negative | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 7.15e-07 | 1.15e-07 | ✅ PASS |
| 3 | Zero values | zeros | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Mixed positive/negative | mixed | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 4.77e-07 | 6.40e-08 | ✅ PASS |
| 5 | Small values (~1e-6) | small | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 2.27e-13 | 3.83e-14 | ✅ PASS |
| 6 | Large values (~1e6) | large | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 2.50e-01 | 3.70e-02 | ✅ PASS |
| 7 | Single level L=1 | single_level | `value[1,1024,4,16] sampling[1,50,4,1,2,2] attn[1,50,4,1,2]` | `[1, 50, 64]` | 4.77e-07 | 1.73e-08 | ✅ PASS |
| 8 | Many sampling points P=8 | many_points | `value[1,320,4,16] sampling[1,64,4,2,8,2] attn[1,64,4,2,8]` | `[1, 64, 64]` | 2.38e-07 | 2.46e-08 | ✅ PASS |
| 9 | Large batch N=4 | large_batch | `value[4,525,8,32] sampling[4,80,8,3,4,2] attn[4,80,8,3,4]` | `[4, 80, 256]` | 1.91e-06 | 9.29e-08 | ✅ PASS |
| 10 | Many heads M=16 | many_heads | `value[1,320,16,8] sampling[1,50,16,2,4,2] attn[1,50,16,2,4]` | `[1, 50, 128]` | 3.58e-07 | 3.21e-08 | ✅ PASS |
| 11 | Single query Lq=1 | single_query | `value[2,80,4,16] sampling[2,1,4,2,4,2] attn[2,1,4,2,4]` | `[2, 1, 64]` | 2.38e-07 | 3.64e-08 | ✅ PASS |
| 12 | Uniform attention | uniform_attention | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 1.19e-07 | 1.39e-08 | ✅ PASS |
| 13 | Concentrated attention | concentrated_attention | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 2.38e-07 | 1.68e-08 | ✅ PASS |
| 14 | Boundary coordinates | boundary_coords | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 2.38e-07 | 1.57e-08 | ✅ PASS |
| 15 | Center coordinates | center_coords | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 3.58e-07 | 3.53e-08 | ✅ PASS |
| 16 | Corner coordinates | corner_coords | `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` | `[2, 50, 64]` | 1.19e-07 | 8.96e-09 | ✅ PASS |
| 17 | Single element pool P=1 | single_element_pooling | `value[1,16,2,8] sampling[1,10,2,1,1,2] attn[1,10,2,1,1]` | `[1, 10, 16]` | 1.19e-07 | 1.75e-08 | ✅ PASS |
| 18 | Minimum input size | minimum_input | `value[1,4,1,4] sampling[1,1,1,1,1,2] attn[1,1,1,1,1]` | `[1, 1, 4]` | 7.45e-09 | 1.86e-09 | ✅ PASS |

---

### 🟢 TEST[0] Standard 4 levels

**Edge Case:** `standard` — Multi-level typical dims — baseline correctness

**Input:** `value[2,3343,8,32] sampling[2,100,8,4,4,2] attn[2,100,8,4,4]` → **Output Shape:** `[2, 100, 256]`

**Input Float Samples [0:10]:**
- value:    `[0.496714, -0.138264, 0.647689, 1.523030, -0.234153, -0.234137, 1.579213, 0.767435, -0.469474, 0.542560]`
- sampling: `[0.267198, 0.677165, 0.876217, 0.261169, 0.857704, 0.379366, 0.514186, 0.024252, 0.364474, 0.772173]`
- attn:     `[0.134004, 0.575127, 0.062931, 0.227938, 0.228665, 0.262807, 0.161113, 0.347416, 0.078998, 0.318972]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.507730, 0.123630, 0.629666, -0.327337, 0.654623, 1.186651, 0.127434, -0.237678, -0.720122, 0.465857]`
- TTSim:   `[-0.507730, 0.123630, 0.629666, -0.327337, 0.654623, 1.186651, 0.127434, -0.237678, -0.720122, 0.465857]`


---

### 🟢 TEST[1] Positive values

**Edge Case:** `positive` — Standard positive values (1.0-2.0) — baseline test

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[1.115055, 1.609066, 1.133391, 1.240590, 1.327139, 1.859138, 1.666090, 1.541162, 1.029014, 1.733748]`
- sampling: `[0.674177, 0.861255, 0.599754, 0.995571, 0.294672, 0.399368, 0.160924, 0.068172, 0.357294, 0.748442]`
- attn:     `[0.116668, 0.487893, 0.153057, 0.242382, 0.218033, 0.225691, 0.263977, 0.292299, 0.544291, 0.080235]`

**Output Float Samples [0:10]:**
- PyTorch: `[2.515529, 2.185426, 2.217101, 2.208278, 2.442133, 2.219783, 2.266051, 2.466883, 2.445711, 2.433266]`
- TTSim:   `[2.515529, 2.185426, 2.217101, 2.208278, 2.442133, 2.219783, 2.266051, 2.466883, 2.445712, 2.433266]`


---

### 🟢 TEST[2] Negative values

**Edge Case:** `negative` — All negative values (-2.0 to -1.0) — sign handling

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[-1.834842, -1.104796, -1.744640, -1.360501, -1.359311, -1.609238, -1.393780, -1.409073, -1.509902, -1.710148]`
- sampling: `[0.011429, 0.577764, 0.509227, 0.807720, 0.449382, 0.343308, 0.861999, 0.082127, 0.810972, 0.111794]`
- attn:     `[0.117331, 0.322556, 0.142667, 0.417447, 0.592899, 0.189754, 0.031412, 0.185935, 0.170457, 0.281866]`

**Output Float Samples [0:10]:**
- PyTorch: `[-2.967896, -2.837913, -3.113606, -2.518616, -2.861939, -2.795045, -2.692098, -2.752330, -3.056627, -2.745742]`
- TTSim:   `[-2.967896, -2.837913, -3.113605, -2.518616, -2.861939, -2.795045, -2.692097, -2.752330, -3.056627, -2.745742]`


---

### 🟢 TEST[3] Zero values

**Edge Case:** `zeros` — All zeros — zero feature maps edge case

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`
- sampling: `[0.989012, 0.549545, 0.281447, 0.077290, 0.444469, 0.472808, 0.048522, 0.163324, 0.115951, 0.627392]`
- attn:     `[0.353644, 0.299405, 0.099080, 0.247871, 0.059899, 0.329838, 0.378469, 0.231794, 0.173966, 0.385682]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`
- TTSim:   `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`


---

### 🟢 TEST[4] Mixed positive/negative

**Edge Case:** `mixed` — Mix of positive/negative — real-world distribution

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[1.169752, 2.462391, 1.643800, -1.598457, 0.824106, -0.352313, -0.146344, -1.131333, -0.186930, 1.714602]`
- sampling: `[0.313337, 0.730103, 0.825690, 0.572840, 0.106986, 0.489541, 0.548977, 0.711404, 0.984290, 0.127837]`
- attn:     `[0.080688, 0.518309, 0.227415, 0.173588, 0.266025, 0.239276, 0.231022, 0.263676, 0.268637, 0.141487]`

**Output Float Samples [0:10]:**
- PyTorch: `[-1.692807, -0.225032, -0.288613, 1.358294, -0.326578, -1.260612, 0.678547, -1.941075, -1.229549, 0.269974]`
- TTSim:   `[-1.692806, -0.225032, -0.288613, 1.358294, -0.326578, -1.260612, 0.678547, -1.941075, -1.229549, 0.269974]`


---

### 🟢 TEST[5] Small values (~1e-6)

**Edge Case:** `small` — Very small values (~1e-6) — precision near zero

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[0.000000, 0.000001, 0.000001, 0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000001, 0.000000]`
- sampling: `[0.480416, 0.727508, 0.576044, 0.614486, 0.760458, 0.713848, 0.900581, 0.441424, 0.409692, 0.867338]`
- attn:     `[0.500444, 0.160897, 0.329089, 0.009571, 0.228276, 0.168618, 0.597699, 0.005408, 0.212775, 0.242781]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]`
- TTSim:   `[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]`


---

### 🟢 TEST[6] Large values (~1e6)

**Edge Case:** `large` — Very large values (~1e6) — overflow handling

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]` (relaxed)

**Input Float Samples [0:10]:**
- value:    `[17490.271484, 891573.250000, 284861.187500, 298976.406250, 792034.250000, 324470.625000, 864710.375000, 447512.625000, 548229.937500, 357171.968750]`
- sampling: `[0.410095, 0.067034, 0.587016, 0.879863, 0.509672, 0.301575, 0.386909, 0.007973, 0.403197, 0.036870]`
- attn:     `[0.337907, 0.251291, 0.120565, 0.290237, 0.388449, 0.136370, 0.151680, 0.323501, 0.491172, 0.034858]`

**Output Float Samples [0:10]:**
- PyTorch: `[875114.375000, 938587.500000, 893268.125000, 750162.375000, 950978.875000, 999974.062500, 1275060.125000, 722869.375000, 857092.500000, 795076.125000]`
- TTSim:   `[875114.375000, 938587.437500, 893268.125000, 750162.375000, 950978.875000, 999974.062500, 1275060.000000, 722869.375000, 857092.625000, 795076.187500]`


---

### 🟢 TEST[7] Single level L=1

**Edge Case:** `single_level` — L=1, single feature level — minimal level loop

**Input:** `value[1,1024,4,16] sampling[1,50,4,1,2,2] attn[1,50,4,1,2]` → **Output Shape:** `[1, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[-1.043159, -0.820856, 0.665146, 1.822627, -1.441583, 0.233808, 0.339619, 0.231214, -0.009926, 1.803848]`
- sampling: `[0.277605, 0.465664, 0.809017, 0.168974, 0.813884, 0.636371, 0.982073, 0.271863, 0.625454, 0.358271]`
- attn:     `[0.433086, 0.566914, 0.222482, 0.777518, 0.754408, 0.245592, 0.990160, 0.009840, 0.640874, 0.359126]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.001767, 0.986160, -0.642988, 0.722283, -0.011570, -0.475600, -0.474106, 0.401154, 0.739161, 0.053460]`
- TTSim:   `[0.001767, 0.986160, -0.642988, 0.722283, -0.011570, -0.475600, -0.474106, 0.401154, 0.739161, 0.053460]`


---

### 🟢 TEST[8] Many sampling points P=8

**Edge Case:** `many_points` — P=8 sampling points — more interpolation

**Input:** `value[1,320,4,16] sampling[1,64,4,2,8,2] attn[1,64,4,2,8]` → **Output Shape:** `[1, 64, 64]`

**Input Float Samples [0:10]:**
- value:    `[-1.560352, -0.030978, -0.620928, -1.464581, 1.411946, -0.476732, -0.780469, 1.070268, -1.282293, -1.327479]`
- sampling: `[0.302386, 0.560740, 0.330474, 0.104734, 0.457218, 0.350384, 0.881509, 0.396933, 0.941724, 0.667537]`
- attn:     `[0.114242, 0.218645, 0.205612, 0.005862, 0.185692, 0.023997, 0.190134, 0.055815, 0.069097, 0.100740]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.009823, -0.348366, -0.405002, -0.357803, 0.285534, 0.037974, -0.286297, 0.208190, 0.647991, 0.719326]`
- TTSim:   `[-0.009823, -0.348366, -0.405002, -0.357803, 0.285534, 0.037974, -0.286297, 0.208190, 0.647991, 0.719326]`


---

### 🟢 TEST[9] Large batch N=4

**Edge Case:** `large_batch` — N=4 batch size — batch scalability

**Input:** `value[4,525,8,32] sampling[4,80,8,3,4,2] attn[4,80,8,3,4]` → **Output Shape:** `[4, 80, 256]`

**Input Float Samples [0:10]:**
- value:    `[-0.290503, 0.112128, 1.250795, -1.360890, 0.099933, -0.047991, -0.356230, -1.088559, -0.351018, 2.588400]`
- sampling: `[0.425312, 0.970344, 0.960152, 0.488511, 0.039182, 0.534441, 0.565897, 0.173665, 0.942299, 0.835709]`
- attn:     `[0.161523, 0.337211, 0.296645, 0.204621, 0.216636, 0.173586, 0.310179, 0.299598, 0.214710, 0.151572]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.644840, 0.765168, 0.170880, 0.449732, 0.765179, -0.798986, 0.494030, -0.238844, -0.154013, -0.636700]`
- TTSim:   `[0.644840, 0.765169, 0.170880, 0.449732, 0.765179, -0.798986, 0.494030, -0.238844, -0.154012, -0.636700]`


---

### 🟢 TEST[10] Many heads M=16

**Edge Case:** `many_heads` — M=16 attention heads — head scalability

**Input:** `value[1,320,16,8] sampling[1,50,16,2,4,2] attn[1,50,16,2,4]` → **Output Shape:** `[1, 50, 128]`

**Input Float Samples [0:10]:**
- value:    `[0.519476, -1.268750, 0.240420, -0.803957, 0.017344, 0.394394, 1.279132, 0.659736, -0.455381, 1.465789]`
- sampling: `[0.090514, 0.526550, 0.367943, 0.888221, 0.877011, 0.522913, 0.030619, 0.803944, 0.455841, 0.588454]`
- attn:     `[0.189085, 0.259260, 0.296135, 0.255520, 0.387178, 0.238601, 0.059013, 0.315208, 0.198518, 0.251806]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.354342, 0.766523, -0.050726, 0.610562, 0.236416, -0.499009, -0.074317, 0.867693, 0.328497, -0.131652]`
- TTSim:   `[-0.354342, 0.766523, -0.050726, 0.610562, 0.236416, -0.499009, -0.074317, 0.867693, 0.328497, -0.131652]`


---

### 🟢 TEST[11] Single query Lq=1

**Edge Case:** `single_query` — Lq=1 — minimal query count

**Input:** `value[2,80,4,16] sampling[2,1,4,2,4,2] attn[2,1,4,2,4]` → **Output Shape:** `[2, 1, 64]`

**Input Float Samples [0:10]:**
- value:    `[0.205865, 1.166762, -2.072640, -0.632687, 0.997126, 2.394554, -0.541234, 0.752705, -2.150352, -0.305441]`
- sampling: `[0.208827, 0.889028, 0.330314, 0.370895, 0.366108, 0.759958, 0.393132, 0.801804, 0.570480, 0.589200]`
- attn:     `[0.046826, 0.040621, 0.847271, 0.065283, 0.286695, 0.165947, 0.271696, 0.275663, 0.283173, 0.014307]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.198628, 0.472577, -0.273686, 0.108083, 0.820938, 0.349537, 1.236966, 0.568144, 0.645639, -0.383288]`
- TTSim:   `[-0.198628, 0.472577, -0.273686, 0.108083, 0.820938, 0.349537, 1.236966, 0.568144, 0.645639, -0.383289]`


---

### 🟢 TEST[12] Uniform attention

**Edge Case:** `uniform_attention` — Equal attention weights — no weighting bias

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[-1.852211, -1.080999, 0.078564, -1.354136, 0.626148, 0.733388, -1.470978, -1.675420, 0.583028, -0.677461]`
- sampling: `[0.393802, 0.650219, 0.523307, 0.864704, 0.981569, 0.026597, 0.144080, 0.152644, 0.271902, 0.608124]`
- attn:     `[0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000, 0.125000]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.023750, -0.021358, -0.030756, 0.139451, 0.415119, -0.309649, 0.462284, 0.267733, 0.164797, -0.215028]`
- TTSim:   `[-0.023750, -0.021358, -0.030756, 0.139451, 0.415119, -0.309649, 0.462284, 0.267733, 0.164798, -0.215028]`


---

### 🟢 TEST[13] Concentrated attention

**Edge Case:** `concentrated_attention` — One-hot attention (all weight on first point)

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[-1.623731, -0.101784, -1.809791, 0.262654, 0.259953, -0.381086, -0.002290, 0.341615, 0.897572, -0.361100]`
- sampling: `[0.431403, 0.574050, 0.221250, 0.716210, 0.475500, 0.821750, 0.888244, 0.281694, 0.016085, 0.165161]`
- attn:     `[1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.113498, 1.050956, 0.268554, -0.155340, -0.454083, -0.335566, -0.857609, 0.524427, 0.479698, -0.236139]`
- TTSim:   `[-0.113498, 1.050956, 0.268554, -0.155340, -0.454083, -0.335566, -0.857609, 0.524427, 0.479698, -0.236139]`


---

### 🟢 TEST[14] Boundary coordinates

**Edge Case:** `boundary_coords` — Sampling near 0.0 and 1.0 — grid edge behavior

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[-1.037643, 0.593658, 1.102681, -0.512178, -0.265420, -1.617006, -0.271514, 0.945554, -0.626993, -0.265947]`
- sampling: `[0.500000, 0.750000, 0.250000, 1.000000, 0.750000, 0.250000, 0.500000, 0.500000, 0.500000, 1.000000]`
- attn:     `[0.316390, 0.390996, 0.172041, 0.120573, 0.274949, 0.326619, 0.071843, 0.326589, 0.433792, 0.252472]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.396689, -0.015501, 0.014716, -0.355559, -0.163312, 0.046376, -0.323291, -0.154297, -0.280857, -0.232485]`
- TTSim:   `[-0.396689, -0.015501, 0.014716, -0.355559, -0.163312, 0.046376, -0.323291, -0.154297, -0.280857, -0.232485]`


---

### 🟢 TEST[15] Center coordinates

**Edge Case:** `center_coords` — All locations at 0.5 — center-only sampling

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[-0.131052, -0.200646, -1.690123, -0.794418, 1.528002, 2.236405, -2.611512, 0.766009, 1.746168, -1.689139]`
- sampling: `[0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000, 0.500000]`
- attn:     `[0.147401, 0.214099, 0.398155, 0.240344, 0.099996, 0.082552, 0.445153, 0.372299, 0.065054, 0.357053]`

**Output Float Samples [0:10]:**
- PyTorch: `[1.467051, 0.204189, 0.026655, -1.399529, 1.028497, -0.919075, -1.802052, -0.627874, 0.574719, 0.915984]`
- TTSim:   `[1.467051, 0.204189, 0.026655, -1.399530, 1.028497, -0.919075, -1.802052, -0.627874, 0.574719, 0.915984]`


---

### 🟢 TEST[16] Corner coordinates

**Edge Case:** `corner_coords` — All at 0.0 or 1.0 — extreme corner positions

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] attn[2,50,4,2,4]` → **Output Shape:** `[2, 50, 64]`

**Input Float Samples [0:10]:**
- value:    `[-0.760186, -2.101584, -0.809757, -0.007517, -1.704361, 0.581404, -0.969144, -1.480901, -0.523343, 1.267065]`
- sampling: `[0.000000, 0.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]`
- attn:     `[0.299219, 0.199619, 0.176018, 0.325144, 0.317514, 0.250177, 0.222291, 0.210018, 0.228301, 0.213468]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.011282, -0.292511, 0.180377, -0.259270, -0.102999, 0.135088, -0.217338, 0.313015, 0.156214, 0.096049]`
- TTSim:   `[-0.011282, -0.292511, 0.180377, -0.259270, -0.102999, 0.135088, -0.217338, 0.313015, 0.156214, 0.096049]`


---

### 🟢 TEST[17] Single element pool P=1

**Edge Case:** `single_element_pooling` — P=1, single sampling point — minimal interpolation

**Input:** `value[1,16,2,8] sampling[1,10,2,1,1,2] attn[1,10,2,1,1]` → **Output Shape:** `[1, 10, 16]`

**Input Float Samples [0:10]:**
- value:    `[-1.669284, 0.563218, 0.420383, -1.601359, 0.632337, -1.192067, -1.235049, 1.772303, -1.122322, -2.194668]`
- sampling: `[0.246291, 0.828703, 0.845454, 0.441380, 0.669421, 0.843783, 0.571851, 0.041961, 0.772670, 0.101534]`
- attn:     `[1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.300562, -0.405697, -0.182414, 0.026230, 0.343335, 0.070368, -0.271301, -0.694446, 0.037562, 0.170904]`
- TTSim:   `[-0.300562, -0.405697, -0.182414, 0.026230, 0.343335, 0.070368, -0.271301, -0.694446, 0.037562, 0.170904]`


---

### 🟢 TEST[18] Minimum input size

**Edge Case:** `minimum_input` — Smallest valid config (all dims minimal) — degenerate case

**Input:** `value[1,4,1,4] sampling[1,1,1,1,1,2] attn[1,1,1,1,1]` → **Output Shape:** `[1, 1, 4]`

**Input Float Samples [0:10]:**
- value:    `[-0.921771, -0.586318, 1.163999, -1.241724, -1.985230, 1.306709, 0.737807, 0.379111, 0.000989, -1.105035]`
- sampling: `[0.581434, 0.839191]`
- attn:     `[1.000000]`

**Output Float Samples [0:10]:**
- PyTorch: `[1.242719, -0.287044, -0.628677, -0.080735]`
- TTSim:   `[1.242719, -0.287044, -0.628677, -0.080735]`


---

## PerLevelGridSample (10/10 PASS)
*Per-level grid_sample output — isolates interpolation accuracy per feature level*

| # | Test Case | Edge Case | Input | Worst Max Diff | Worst Mean Diff | Result |
|:--|:----------|:----------|:------|:---------------|:----------------|:-------|
| 0 | Standard 4 levels | standard_4_levels | `value[2,3343,8,32] sampling[2,100,8,4,4,2] L=4 levels` | 8.88e-06 | 4.55e-08 | ✅ PASS |
| 1 | Single level | single_level | `value[1,1024,4,16] sampling[1,50,4,1,2,2] L=1 levels` | 2.38e-07 | 1.80e-08 | ✅ PASS |
| 2 | Two levels asymmetric | two_levels_asymmetric | `value[2,1040,4,16] sampling[2,64,4,2,4,2] L=2 levels` | 2.38e-07 | 1.78e-08 | ✅ PASS |
| 3 | Negative values | negative_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels` | 2.38e-07 | 3.46e-08 | ✅ PASS |
| 4 | Small values (~1e-6) | small_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels` | 1.14e-13 | 1.18e-14 | ✅ PASS |
| 5 | Zero values | zeros_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Mixed positive/negative | mixed_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels` | 4.77e-07 | 3.51e-08 | ✅ PASS |
| 7 | Large values (~1e6) | large_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels` | 1.25e-01 | 1.21e-02 | ✅ PASS |
| 8 | Single element pool P=1 | single_element_pooling | `value[1,16,2,8] sampling[1,10,2,1,1,2] L=1 levels` | 2.38e-07 | 2.12e-08 | ✅ PASS |
| 9 | Minimum input size | minimum_input | `value[1,4,1,4] sampling[1,1,1,1,1,2] L=1 levels` | 5.96e-08 | 3.07e-08 | ✅ PASS |

---

### 🟢 TEST[0] Standard 4 levels

**Edge Case:** `standard_4_levels` — 4 levels — per-level grid_sample check

**Input:** `value[2,3343,8,32] sampling[2,100,8,4,4,2] L=4 levels`

**Per-Level Results:**
  - Level 0: shape=`[16, 32, 100, 4]`, max_diff=8.88e-06, mean_diff=4.55e-08 ✅
  - Level 1: shape=`[16, 32, 100, 4]`, max_diff=4.53e-06, mean_diff=4.54e-08 ✅
  - Level 2: shape=`[16, 32, 100, 4]`, max_diff=2.15e-06, mean_diff=4.14e-08 ✅
  - Level 3: shape=`[16, 32, 100, 4]`, max_diff=1.31e-06, mean_diff=3.66e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[0.129740, 0.902362, 1.005804, 0.471890, -0.326213, -0.262650, -0.368877, 1.478741, 1.299381, -0.525526]`
- sampling: `[0.194748, 0.385913, 0.563946, 0.119553, 0.029821, 0.898832, 0.849841, 0.254641, 0.158878, 0.876961]`
- attn:     `[0.166032, 0.386575, 0.209710, 0.237684, 0.198851, 0.382153, 0.272026, 0.146971, 0.230696, 0.014836]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.018932, 0.611523, 0.250192, 0.847094, -0.017220, -0.152772, 0.463894, -0.759078, -0.099546, -1.278480]`
- TTSim:   `[0.018932, 0.611523, 0.250192, 0.847094, -0.017220, -0.152772, 0.463894, -0.759078, -0.099546, -1.278481]`


---

### 🟢 TEST[1] Single level

**Edge Case:** `single_level` — L=1, single feature level — minimal level loop

**Input:** `value[1,1024,4,16] sampling[1,50,4,1,2,2] L=1 levels`

**Per-Level Results:**
  - Level 0: shape=`[4, 16, 50, 2]`, max_diff=2.38e-07, mean_diff=1.80e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[-0.833154, 1.237691, -2.475731, -1.038631, 0.994645, 0.178225, -0.489080, -0.442089, -0.251608, -0.468647]`
- sampling: `[0.972891, 0.859921, 0.564467, 0.249019, 0.426448, 0.227939, 0.529626, 0.081214, 0.880467, 0.939299]`
- attn:     `[0.014387, 0.985613, 0.737697, 0.262303, 0.813239, 0.186761, 0.514086, 0.485914, 0.908884, 0.091116]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.156823, -0.075361, -0.039904, 0.154837, -0.160338, -0.825157, 0.259009, 0.083343, 0.949817, -0.948571]`
- TTSim:   `[-0.156823, -0.075361, -0.039904, 0.154837, -0.160338, -0.825157, 0.259009, 0.083343, 0.949817, -0.948570]`


---

### 🟢 TEST[2] Two levels asymmetric

**Edge Case:** `two_levels_asymmetric` — 2 levels with very different spatial sizes

**Input:** `value[2,1040,4,16] sampling[2,64,4,2,4,2] L=2 levels`

**Per-Level Results:**
  - Level 0: shape=`[8, 16, 64, 4]`, max_diff=2.38e-07, mean_diff=1.78e-08 ✅
  - Level 1: shape=`[8, 16, 64, 4]`, max_diff=2.38e-07, mean_diff=1.49e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[-1.298571, -0.092539, 0.070074, 1.855052, 1.370260, -0.182580, -1.170023, 1.027954, -0.834468, 0.187793]`
- sampling: `[0.558301, 0.421189, 0.782076, 0.610531, 0.955816, 0.008113, 0.899970, 0.295461, 0.088143, 0.809468]`
- attn:     `[0.305776, 0.344110, 0.120838, 0.229276, 0.305855, 0.279860, 0.324322, 0.089963, 0.349527, 0.378332]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.103294, 0.305755, -0.878443, -0.413167, -1.155927, 0.544295, -0.710742, -0.262321, 0.394104, 0.550351]`
- TTSim:   `[0.103294, 0.305755, -0.878443, -0.413167, -1.155927, 0.544295, -0.710742, -0.262321, 0.394104, 0.550351]`


---

### 🟢 TEST[3] Negative values

**Edge Case:** `negative_values` — Negative feature values with standard geometry

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels`

**Per-Level Results:**
  - Level 0: shape=`[8, 16, 50, 4]`, max_diff=2.38e-07, mean_diff=3.46e-08 ✅
  - Level 1: shape=`[8, 16, 50, 4]`, max_diff=2.38e-07, mean_diff=3.27e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[-1.283042, -1.002030, -1.530579, -1.080361, -1.652258, -1.770739, -1.598140, -1.326684, -1.697576, -1.495345]`
- sampling: `[0.565911, 0.714777, 0.732933, 0.124525, 0.997036, 0.582632, 0.852942, 0.572596, 0.823582, 0.065542]`
- attn:     `[0.038694, 0.503343, 0.050244, 0.407719, 0.289141, 0.290526, 0.155729, 0.264604, 0.382543, 0.314875]`

**Output Float Samples [0:10]:**
- PyTorch: `[-2.810215, -2.960869, -3.170809, -2.763809, -3.196222, -3.001483, -2.656764, -2.765560, -3.077836, -3.125887]`
- TTSim:   `[-2.810216, -2.960869, -3.170808, -2.763809, -3.196222, -3.001483, -2.656764, -2.765560, -3.077836, -3.125887]`


---

### 🟢 TEST[4] Small values (~1e-6)

**Edge Case:** `small_values` — Very small feature values with standard geometry

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels`

**Per-Level Results:**
  - Level 0: shape=`[8, 16, 50, 4]`, max_diff=1.14e-13, mean_diff=1.18e-14 ✅
  - Level 1: shape=`[8, 16, 50, 4]`, max_diff=1.14e-13, mean_diff=1.13e-14 ✅

**Input Float Samples [0:10]:**
- value:    `[0.000000, 0.000001, 0.000000, 0.000000, 0.000000, 0.000001, 0.000000, 0.000000, 0.000001, 0.000001]`
- sampling: `[0.985452, 0.797251, 0.876362, 0.570412, 0.309836, 0.937740, 0.038580, 0.261507, 0.800154, 0.770735]`
- attn:     `[0.090761, 0.284484, 0.292354, 0.332401, 0.223938, 0.375096, 0.092753, 0.308213, 0.305542, 0.148147]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]`
- TTSim:   `[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]`


---

### 🟢 TEST[5] Zero values

**Edge Case:** `zeros_values` — All-zero feature values — zero output edge case

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels`

**Per-Level Results:**
  - Level 0: shape=`[8, 16, 50, 4]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Level 1: shape=`[8, 16, 50, 4]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅

**Input Float Samples [0:10]:**
- value:    `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`
- sampling: `[0.366696, 0.632981, 0.030831, 0.691256, 0.937811, 0.941416, 0.386083, 0.074266, 0.485595, 0.570020]`
- attn:     `[0.655148, 0.059963, 0.160685, 0.124204, 0.190865, 0.245037, 0.343173, 0.220925, 0.009811, 0.430382]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`
- TTSim:   `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`


---

### 🟢 TEST[6] Mixed positive/negative

**Edge Case:** `mixed_values` — Mix of positive/negative feature values

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels`

**Per-Level Results:**
  - Level 0: shape=`[8, 16, 50, 4]`, max_diff=4.77e-07, mean_diff=3.51e-08 ✅
  - Level 1: shape=`[8, 16, 50, 4]`, max_diff=4.77e-07, mean_diff=3.33e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[-2.172171, 0.455375, 3.775336, 2.687807, 0.269445, -0.177499, 1.083030, 3.423449, -2.865374, 1.393151]`
- sampling: `[0.210195, 0.442968, 0.046006, 0.681472, 0.848801, 0.105253, 0.455342, 0.817189, 0.586975, 0.548972]`
- attn:     `[0.238729, 0.308520, 0.442936, 0.009815, 0.404104, 0.059313, 0.255888, 0.280695, 0.061861, 0.477962]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.769352, 0.063549, -1.921472, 2.578893, 0.885959, -0.749560, 0.206846, -2.126499, -2.391582, 1.653580]`
- TTSim:   `[0.769352, 0.063549, -1.921472, 2.578893, 0.885959, -0.749560, 0.206846, -2.126499, -2.391582, 1.653580]`


---

### 🟢 TEST[7] Large values (~1e6)

**Edge Case:** `large_values` — Very large feature values — overflow risk

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] L=2 levels` (relaxed)

**Per-Level Results:**
  - Level 0: shape=`[8, 16, 50, 4]`, max_diff=1.25e-01, mean_diff=1.21e-02 ✅
  - Level 1: shape=`[8, 16, 50, 4]`, max_diff=1.25e-01, mean_diff=1.13e-02 ✅

**Input Float Samples [0:10]:**
- value:    `[881307.562500, 979703.625000, 864582.500000, 838072.187500, 909790.312500, 860982.000000, 470752.375000, 441234.437500, 600155.750000, 69804.960938]`
- sampling: `[0.601214, 0.241093, 0.925680, 0.917536, 0.010038, 0.683613, 0.817621, 0.027995, 0.014155, 0.818237]`
- attn:     `[0.064494, 0.319389, 0.379562, 0.236555, 0.029153, 0.292026, 0.367303, 0.311518, 0.373793, 0.122692]`

**Output Float Samples [0:10]:**
- PyTorch: `[863501.250000, 841981.687500, 1036102.875000, 963337.375000, 1044595.312500, 1044535.500000, 1148098.375000, 1006699.625000, 944818.875000, 1024391.812500]`
- TTSim:   `[863501.250000, 841981.625000, 1036102.750000, 963337.375000, 1044595.375000, 1044535.375000, 1148098.500000, 1006699.625000, 944818.875000, 1024391.812500]`


---

### 🟢 TEST[8] Single element pool P=1

**Edge Case:** `single_element_pooling` — P=1, single sampling point — minimal interpolation

**Input:** `value[1,16,2,8] sampling[1,10,2,1,1,2] L=1 levels`

**Per-Level Results:**
  - Level 0: shape=`[2, 8, 10, 1]`, max_diff=2.38e-07, mean_diff=2.12e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[-0.231383, 0.390613, 0.358297, 0.566244, 0.166234, 1.913412, 0.302903, -0.830826, -0.060142, -0.094294]`
- sampling: `[0.867835, 0.418072, 0.922803, 0.727356, 0.121228, 0.745881, 0.597340, 0.195523, 0.422198, 0.167045]`
- attn:     `[1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.229940, -0.357867, -0.100184, 1.138430, -1.114634, 1.570107, -1.044895, 1.257637, -0.186515, 1.106222]`
- TTSim:   `[0.229940, -0.357867, -0.100184, 1.138430, -1.114634, 1.570107, -1.044895, 1.257637, -0.186515, 1.106222]`


---

### 🟢 TEST[9] Minimum input size

**Edge Case:** `minimum_input` — Smallest valid config (all dims minimal) — degenerate case

**Input:** `value[1,4,1,4] sampling[1,1,1,1,1,2] L=1 levels`

**Per-Level Results:**
  - Level 0: shape=`[1, 4, 1, 1]`, max_diff=5.96e-08, mean_diff=3.07e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[-0.636738, 0.531559, 0.990208, -0.624134, 1.467781, 0.405013, 1.298174, -2.613633, 1.356434, 1.873161]`
- sampling: `[0.034794, 0.784137]`
- attn:     `[1.000000]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.719860, 0.994087, -0.051970, 0.299630]`
- TTSim:   `[0.719860, 0.994087, -0.051970, 0.299630]`


---

## IntermediateSteps (9/9 PASS)
*Step-by-step intermediate comparison — pin-points first diverging step*

| # | Test Case | Edge Case | Input | Worst Max Diff | Worst Mean Diff | Result |
|:--|:----------|:----------|:------|:---------------|:----------------|:-------|
| 0 | Standard config | standard_config | `value[2,3343,8,32] sampling[2,100,8,4,4,2] 6 steps` | 7.78e-06 | 4.12e-08 | ✅ PASS |
| 1 | Small config | small_config | `value[1,20,2,8] sampling[1,10,2,2,2,2] 6 steps` | 2.38e-07 | 2.71e-08 | ✅ PASS |
| 2 | Negative values | negative_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps` | 4.77e-07 | 1.19e-07 | ✅ PASS |
| 3 | Large values (~1e6) | large_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps` | 2.50e-01 | 3.79e-02 | ✅ PASS |
| 4 | Zero values | zeros_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Mixed values | mixed_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps` | 5.96e-07 | 6.27e-08 | ✅ PASS |
| 6 | Small values (~1e-6) | small_values | `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps` | 3.41e-13 | 3.80e-14 | ✅ PASS |
| 7 | Single element pool P=1 | single_element_pooling | `value[1,16,2,8] sampling[1,10,2,1,1,2] 6 steps` | 1.19e-07 | 1.16e-08 | ✅ PASS |
| 8 | Minimum input size | minimum_input | `value[1,4,1,4] sampling[1,1,1,1,1,2] 6 steps` | 2.98e-08 | 1.12e-08 | ✅ PASS |

---

### 🟢 TEST[0] Standard config

**Edge Case:** `standard_config` — Standard config — full intermediate comparison

**Input:** `value[2,3343,8,32] sampling[2,100,8,4,4,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[2, 100, 8, 4, 4, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[16, 32, 100, 4, 4]`, max_diff=7.78e-06, mean_diff=4.12e-08 ✅
  - Flattened Stacked: shape=`[16, 32, 100, 16]`, max_diff=7.78e-06, mean_diff=4.12e-08 ✅
  - Attention Reshaped: shape=`[16, 1, 100, 16]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[16, 32, 100]`, max_diff=2.62e-06, mean_diff=1.06e-07 ✅
  - Final Output: shape=`[2, 100, 256]`, max_diff=2.62e-06, mean_diff=1.06e-07 ✅

**Input Float Samples [0:10]:**
- value:    `[-0.357519, 0.148448, 0.993531, 1.838968, -0.744026, -0.309339, 1.064354, 0.393791, -0.723268, -0.439088]`
- sampling: `[0.888653, 0.224453, 0.213538, 0.600574, 0.721727, 0.788731, 0.609828, 0.442966, 0.283160, 0.957038]`
- attn:     `[0.211679, 0.247400, 0.350225, 0.190695, 0.171880, 0.293883, 0.299987, 0.234250, 0.257578, 0.209795]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.787104, -1.190204, -2.153804, 0.400656, -0.105699, -1.162257, -0.520941, -0.158356, -0.528519, 0.422479]`
- TTSim:   `[-0.787104, -1.190204, -2.153804, 0.400656, -0.105699, -1.162257, -0.520941, -0.158356, -0.528519, 0.422479]`


---

### 🟢 TEST[1] Small config

**Edge Case:** `small_config` — Tiny dims for quick smoke test

**Input:** `value[1,20,2,8] sampling[1,10,2,2,2,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[1, 10, 2, 2, 2, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[2, 8, 10, 2, 2]`, max_diff=1.19e-07, mean_diff=1.70e-08 ✅
  - Flattened Stacked: shape=`[2, 8, 10, 4]`, max_diff=1.19e-07, mean_diff=1.70e-08 ✅
  - Attention Reshaped: shape=`[2, 1, 10, 4]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[2, 8, 10]`, max_diff=2.38e-07, mean_diff=2.71e-08 ✅
  - Final Output: shape=`[1, 10, 16]`, max_diff=2.38e-07, mean_diff=2.71e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[0.632876, -0.656156, -1.627301, 1.129801, -1.211659, -2.142316, -0.894444, 1.141073, 0.242753, -0.281128]`
- sampling: `[0.768442, 0.258594, 0.290348, 0.973731, 0.688250, 0.864945, 0.913037, 0.511163, 0.582180, 0.898000]`
- attn:     `[0.489972, 0.510028, 0.619067, 0.380933, 0.372255, 0.627745, 0.400716, 0.599284, 0.493662, 0.506338]`

**Output Float Samples [0:10]:**
- PyTorch: `[-0.696509, 0.111555, -0.674418, -0.244389, -0.932417, -0.060517, -0.526953, -1.114504, -0.463791, 0.677459]`
- TTSim:   `[-0.696509, 0.111555, -0.674418, -0.244389, -0.932417, -0.060517, -0.526953, -1.114504, -0.463791, 0.677459]`


---

### 🟢 TEST[2] Negative values

**Edge Case:** `negative_values` — Negative feature values with standard geometry

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[2, 50, 4, 2, 4, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[8, 16, 50, 2, 4]`, max_diff=2.38e-07, mean_diff=3.41e-08 ✅
  - Flattened Stacked: shape=`[8, 16, 50, 8]`, max_diff=2.38e-07, mean_diff=3.41e-08 ✅
  - Attention Reshaped: shape=`[8, 1, 50, 8]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[8, 16, 50]`, max_diff=4.77e-07, mean_diff=1.19e-07 ✅
  - Final Output: shape=`[2, 50, 64]`, max_diff=4.77e-07, mean_diff=1.19e-07 ✅

**Input Float Samples [0:10]:**
- value:    `[-1.290678, -1.099277, -1.249007, -1.396347, -1.072394, -1.217645, -1.990484, -1.631038, -1.647354, -1.511369]`
- sampling: `[0.165644, 0.717805, 0.246591, 0.419160, 0.617434, 0.228055, 0.740349, 0.544515, 0.616638, 0.843905]`
- attn:     `[0.114063, 0.081521, 0.327779, 0.476637, 0.178663, 0.029988, 0.514073, 0.277276, 0.395694, 0.333557]`

**Output Float Samples [0:10]:**
- PyTorch: `[-3.020215, -3.361649, -2.816862, -3.338176, -3.100731, -2.923933, -2.824472, -3.039901, -2.978840, -2.855113]`
- TTSim:   `[-3.020215, -3.361649, -2.816862, -3.338176, -3.100731, -2.923933, -2.824472, -3.039901, -2.978840, -2.855113]`


---

### 🟢 TEST[3] Large values (~1e6)

**Edge Case:** `large_values` — Very large feature values — overflow risk

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps` (relaxed)

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[2, 50, 4, 2, 4, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[8, 16, 50, 2, 4]`, max_diff=1.25e-01, mean_diff=1.15e-02 ✅
  - Flattened Stacked: shape=`[8, 16, 50, 8]`, max_diff=1.25e-01, mean_diff=1.15e-02 ✅
  - Attention Reshaped: shape=`[8, 1, 50, 8]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[8, 16, 50]`, max_diff=2.50e-01, mean_diff=3.79e-02 ✅
  - Final Output: shape=`[2, 50, 64]`, max_diff=2.50e-01, mean_diff=3.79e-02 ✅

**Input Float Samples [0:10]:**
- value:    `[906853.875000, 112530.007812, 100300.132812, 76207.679688, 942521.937500, 849352.500000, 951836.312500, 181182.671875, 901289.375000, 385884.375000]`
- sampling: `[0.318519, 0.505626, 0.767378, 0.564579, 0.230446, 0.336614, 0.712559, 0.565179, 0.924362, 0.490196]`
- attn:     `[0.512174, 0.238287, 0.010214, 0.239325, 0.462410, 0.067910, 0.086499, 0.383181, 0.092200, 0.523518]`

**Output Float Samples [0:10]:**
- PyTorch: `[1056071.500000, 1280265.000000, 1030785.312500, 1025686.375000, 948792.125000, 923640.312500, 765240.687500, 1253728.875000, 689915.875000, 1258035.250000]`
- TTSim:   `[1056071.500000, 1280264.875000, 1030785.312500, 1025686.375000, 948792.062500, 923640.312500, 765240.750000, 1253728.750000, 689915.875000, 1258035.250000]`


---

### 🟢 TEST[4] Zero values

**Edge Case:** `zeros_values` — All-zero feature values — zero output edge case

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[2, 50, 4, 2, 4, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[8, 16, 50, 2, 4]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Flattened Stacked: shape=`[8, 16, 50, 8]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Attention Reshaped: shape=`[8, 1, 50, 8]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[8, 16, 50]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Final Output: shape=`[2, 50, 64]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅

**Input Float Samples [0:10]:**
- value:    `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`
- sampling: `[0.479649, 0.912466, 0.690418, 0.589381, 0.006957, 0.410549, 0.575440, 0.463360, 0.064091, 0.313986]`
- attn:     `[0.326881, 0.151054, 0.219168, 0.302898, 0.460478, 0.083043, 0.028543, 0.427936, 0.346593, 0.133924]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`
- TTSim:   `[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`


---

### 🟢 TEST[5] Mixed values

**Edge Case:** `mixed_values` — Mix of positive/negative feature values

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[2, 50, 4, 2, 4, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[8, 16, 50, 2, 4]`, max_diff=4.77e-07, mean_diff=3.38e-08 ✅
  - Flattened Stacked: shape=`[8, 16, 50, 8]`, max_diff=4.77e-07, mean_diff=3.38e-08 ✅
  - Attention Reshaped: shape=`[8, 1, 50, 8]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[8, 16, 50]`, max_diff=5.96e-07, mean_diff=6.27e-08 ✅
  - Final Output: shape=`[2, 50, 64]`, max_diff=5.96e-07, mean_diff=6.27e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[1.487826, -0.488540, -2.810998, 3.018626, 0.330088, 2.268781, -1.278978, 2.456594, -1.981991, -1.259241]`
- sampling: `[0.200542, 0.558721, 0.138921, 0.102533, 0.596504, 0.996806, 0.243583, 0.334414, 0.118297, 0.874810]`
- attn:     `[0.016252, 0.497891, 0.020780, 0.465077, 0.380309, 0.041980, 0.347310, 0.230401, 0.184946, 0.632010]`

**Output Float Samples [0:10]:**
- PyTorch: `[-2.275722, -1.508327, -1.546454, 1.342775, 0.545717, -1.358645, 1.860170, -1.223813, -0.042730, -0.488851]`
- TTSim:   `[-2.275722, -1.508327, -1.546454, 1.342775, 0.545718, -1.358645, 1.860170, -1.223813, -0.042730, -0.488851]`


---

### 🟢 TEST[6] Small values (~1e-6)

**Edge Case:** `small_values` — Very small feature values with standard geometry

**Input:** `value[2,320,4,16] sampling[2,50,4,2,4,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[2, 50, 4, 2, 4, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[8, 16, 50, 2, 4]`, max_diff=1.14e-13, mean_diff=1.15e-14 ✅
  - Flattened Stacked: shape=`[8, 16, 50, 8]`, max_diff=1.14e-13, mean_diff=1.15e-14 ✅
  - Attention Reshaped: shape=`[8, 1, 50, 8]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[8, 16, 50]`, max_diff=3.41e-13, mean_diff=3.80e-14 ✅
  - Final Output: shape=`[2, 50, 64]`, max_diff=3.41e-13, mean_diff=3.80e-14 ✅

**Input Float Samples [0:10]:**
- value:    `[0.000001, 0.000000, 0.000000, 0.000001, 0.000001, 0.000001, 0.000000, 0.000001, 0.000000, 0.000000]`
- sampling: `[0.601067, 0.241265, 0.747372, 0.294572, 0.228762, 0.665620, 0.248820, 0.714246, 0.404371, 0.983856]`
- attn:     `[0.174159, 0.423282, 0.043682, 0.358877, 0.168210, 0.389518, 0.140617, 0.301656, 0.271114, 0.309564]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]`
- TTSim:   `[0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001]`


---

### 🟢 TEST[7] Single element pool P=1

**Edge Case:** `single_element_pooling` — P=1, single sampling point — minimal interpolation

**Input:** `value[1,16,2,8] sampling[1,10,2,1,1,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[1, 10, 2, 1, 1, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[2, 8, 10, 1, 1]`, max_diff=1.19e-07, mean_diff=1.16e-08 ✅
  - Flattened Stacked: shape=`[2, 8, 10, 1]`, max_diff=1.19e-07, mean_diff=1.16e-08 ✅
  - Attention Reshaped: shape=`[2, 1, 10, 1]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[2, 8, 10]`, max_diff=1.19e-07, mean_diff=1.16e-08 ✅
  - Final Output: shape=`[1, 10, 16]`, max_diff=1.19e-07, mean_diff=1.16e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[1.427614, 0.080751, 0.191511, 0.608525, -0.582141, 1.751322, -0.196018, 0.715930, -3.126014, -0.441438]`
- sampling: `[0.979673, 0.328587, 0.922652, 0.674105, 0.034768, 0.759011, 0.541098, 0.301172, 0.465036, 0.216507]`
- attn:     `[1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.361245, -0.360597, -0.159358, 0.022807, -0.148236, 0.249082, 0.108746, 0.387670, 0.407241, -0.541546]`
- TTSim:   `[0.361245, -0.360597, -0.159358, 0.022807, -0.148236, 0.249082, 0.108746, 0.387670, 0.407241, -0.541546]`


---

### 🟢 TEST[8] Minimum input size

**Edge Case:** `minimum_input` — Smallest valid config (all dims minimal) — degenerate case

**Input:** `value[1,4,1,4] sampling[1,1,1,1,1,2] 6 steps`

**Step-by-Step Results:**
  - Sampling Grids [0,1]→[-1,1]: shape=`[1, 1, 1, 1, 1, 2]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Stacked Grid Sample Outputs: shape=`[1, 4, 1, 1, 1]`, max_diff=2.98e-08, mean_diff=1.12e-08 ✅
  - Flattened Stacked: shape=`[1, 4, 1, 1]`, max_diff=2.98e-08, mean_diff=1.12e-08 ✅
  - Attention Reshaped: shape=`[1, 1, 1, 1]`, max_diff=0.00e+00, mean_diff=0.00e+00 ✅
  - Weighted Sum: shape=`[1, 4, 1]`, max_diff=2.98e-08, mean_diff=1.12e-08 ✅
  - Final Output: shape=`[1, 1, 4]`, max_diff=2.98e-08, mean_diff=1.12e-08 ✅

**Input Float Samples [0:10]:**
- value:    `[-0.700122, -0.878386, 1.273181, -1.531156, 1.363569, 0.900524, -1.798084, 0.868980, 0.627284, 1.256685]`
- sampling: `[0.130578, 0.634538]`
- attn:     `[1.000000]`

**Output Float Samples [0:10]:**
- PyTorch: `[0.244143, 0.581252, -0.384707, -0.260498]`
- TTSim:   `[0.244143, 0.581252, -0.384707, -0.260498]`


---

## Configuration
- Tolerance: rtol=0.0001, atol=1e-05
- Relaxed Tolerance (large values): rtol=0.001, atol=0.01
- Random Seed: 42
