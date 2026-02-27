# NN Modules Unit Test Report
**SimNN module validation** | **28/28 passed** | PASS
Generated: 2026-02-17 14:57:52 | Exit: 1

---

## Summary

| Module | Passed | Total | Status |
|:-------|-------:|------:|:-------|
| Module | 3 | 3 | PASS |
| ModuleList | 2 | 2 | PASS |
| Linear | 8 | 8 | PASS |
| GroupNorm | 7 | 7 | PASS |
| Dropout | 8 | 8 | PASS |

**Total: 28/28**

---

## Module (3/3 PASS)

| Test | Passed | Max Diff |
|:-----|:-------|:---------|
| tensor_reg | ✅ | N/A |
| submod_reg | ✅ | N/A |
| link_op2mod | ✅ | N/A |

## ModuleList (2/2 PASS)

| Test | Passed | Max Diff |
|:-----|:-------|:---------|
| basics | ✅ | N/A |
| empty_reject | ✅ | N/A |

## Linear (8/8 PASS)

| Test | Passed | Max Diff |
|:-----|:-------|:---------|
| positive | ✅ | 0.00e+00 |
| negative | ✅ | 0.00e+00 |
| zeros | ✅ | 0.00e+00 |
| mixed | ✅ | 0.00e+00 |
| small | ✅ | 0.00e+00 |
| large | ✅ | 0.00e+00 |
| minimum_input | ✅ | 0.00e+00 |
| no_bias | ✅ | 0.00e+00 |

## GroupNorm (7/7 PASS)

| Test | Passed | Max Diff |
|:-----|:-------|:---------|
| positive | ✅ | 0.00e+00 |
| negative | ✅ | 0.00e+00 |
| zeros | ✅ | 0.00e+00 |
| mixed | ✅ | 0.00e+00 |
| small | ✅ | 0.00e+00 |
| large | ✅ | 0.00e+00 |
| minimum_input | ✅ | 0.00e+00 |

## Dropout (8/8 PASS)

| Test | Passed | Max Diff |
|:-----|:-------|:---------|
| positive | ✅ | 0.00e+00 |
| negative | ✅ | 0.00e+00 |
| zeros | ✅ | 0.00e+00 |
| mixed | ✅ | 0.00e+00 |
| small | ✅ | 0.00e+00 |
| large | ✅ | 0.00e+00 |
| minimum_input | ✅ | 0.00e+00 |
| param_count | ✅ | N/A |

---
Tolerance: rtol=0.0001, atol=1e-05 | Seed: 42