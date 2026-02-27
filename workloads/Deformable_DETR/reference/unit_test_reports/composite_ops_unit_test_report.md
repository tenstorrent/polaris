# Composite Ops Unit Test Report
**TTSim SimOpHandle pipeline validation** | **60/60 passed** | PASS
Generated: 2026-02-17 14:58:35 | Exit: 0

---

## Summary

| Op | Passed | Total | Status |
|:---|-------:|------:|:-------|
| Conv2d | 9 | 9 | PASS |
| MaxPool2d | 8 | 8 | PASS |
| BatchNorm2d | 8 | 8 | PASS |
| LayerNorm | 8 | 8 | PASS |
| GroupNorm | 7 | 7 | PASS |
| Resize | 8 | 8 | PASS |
| Dropout | 9 | 9 | PASS |
| Pipeline | 3 | 3 | PASS |

**Total: 60/60**

---

## Conv2d (9/9 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | OK | OK | 5.96e-08 | 2.61e-08 |
| negative | OK | OK | 5.96e-08 | 2.61e-08 |
| zeros | OK | OK | 0.00e+00 | 0.00e+00 |
| mixed | OK | OK | 5.96e-08 | 1.82e-08 |
| small | OK | OK | 0.00e+00 | 0.00e+00 |
| large | OK | OK | 8.19e+03 | 2.05e+03 |
| minimum_input | OK | OK | 0.00e+00 | 0.00e+00 |
| padding | OK | OK | 1.49e-08 | 4.12e-09 |
| no_bias | OK | OK | 7.45e-09 | 3.73e-09 |

## MaxPool2d (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | OK | OK | 0.00e+00 | 0.00e+00 |
| negative | OK | OK | 0.00e+00 | 0.00e+00 |
| zeros | OK | OK | 0.00e+00 | 0.00e+00 |
| mixed | OK | OK | 0.00e+00 | 0.00e+00 |
| small | OK | OK | 0.00e+00 | 0.00e+00 |
| large | OK | OK | 0.00e+00 | 0.00e+00 |
| minimum_input | OK | OK | 0.00e+00 | 0.00e+00 |
| padding | OK | OK | 0.00e+00 | 0.00e+00 |

## BatchNorm2d (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | OK | OK | 0.00e+00 | 0.00e+00 |
| negative | OK | OK | 0.00e+00 | 0.00e+00 |
| zeros | OK | OK | 0.00e+00 | 0.00e+00 |
| mixed | OK | OK | 0.00e+00 | 0.00e+00 |
| small | OK | OK | 0.00e+00 | 0.00e+00 |
| large | OK | OK | 0.00e+00 | 0.00e+00 |
| minimum_input | OK | OK | 0.00e+00 | 0.00e+00 |
| param_count | OK | OK | 0.00e+00 | 0.00e+00 |

## LayerNorm (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | OK | OK | 0.00e+00 | 0.00e+00 |
| negative | OK | OK | 0.00e+00 | 0.00e+00 |
| zeros | OK | OK | 0.00e+00 | 0.00e+00 |
| mixed | OK | OK | 0.00e+00 | 0.00e+00 |
| small | OK | OK | 0.00e+00 | 0.00e+00 |
| large | OK | OK | 0.00e+00 | 0.00e+00 |
| minimum_input | OK | OK | 0.00e+00 | 0.00e+00 |
| param_count | OK | OK | 0.00e+00 | 0.00e+00 |

## GroupNorm (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | OK | OK | 0.00e+00 | 0.00e+00 |
| negative | OK | OK | 0.00e+00 | 0.00e+00 |
| zeros | OK | OK | 0.00e+00 | 0.00e+00 |
| mixed | OK | OK | 0.00e+00 | 0.00e+00 |
| small | OK | OK | 0.00e+00 | 0.00e+00 |
| large | OK | OK | 0.00e+00 | 0.00e+00 |
| minimum_input | OK | OK | 0.00e+00 | 0.00e+00 |

## Resize (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | OK | OK | 0.00e+00 | 0.00e+00 |
| negative | OK | OK | 0.00e+00 | 0.00e+00 |
| zeros | OK | OK | 0.00e+00 | 0.00e+00 |
| mixed | OK | OK | 0.00e+00 | 0.00e+00 |
| small | OK | OK | 0.00e+00 | 0.00e+00 |
| large | OK | OK | 0.00e+00 | 0.00e+00 |
| minimum_input | OK | OK | 0.00e+00 | 0.00e+00 |
| scale_list | OK | OK | 0.00e+00 | 0.00e+00 |

## Dropout (9/9 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | OK | OK | 0.00e+00 | 0.00e+00 |
| negative | OK | OK | 0.00e+00 | 0.00e+00 |
| zeros | OK | OK | 0.00e+00 | 0.00e+00 |
| mixed | OK | OK | 0.00e+00 | 0.00e+00 |
| small | OK | OK | 0.00e+00 | 0.00e+00 |
| large | OK | OK | 0.00e+00 | 0.00e+00 |
| minimum_input | OK | OK | 0.00e+00 | 0.00e+00 |
| param_structure | OK | OK | 0.00e+00 | 0.00e+00 |
| training | OK | OK | 0.00e+00 | 0.00e+00 |

## Pipeline (3/3 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| sim_op_created | OK | OK | 0.00e+00 | 0.00e+00 |
| output_linked | OK | OK | 0.00e+00 | 0.00e+00 |
| precision_set | OK | OK | 0.00e+00 | 0.00e+00 |

---
Tolerance: rtol=0.0001, atol=1e-05 | Seed: 42