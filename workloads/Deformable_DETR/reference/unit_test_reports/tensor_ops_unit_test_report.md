# Tensor Ops Unit Test Report
**TTSim tensor operation validation** | **136/136 passed** | PASS
Generated: 2026-02-17 14:24:18 | Exit: 0

---

## Summary

| Op | Passed | Total | Status |
|:---|-------:|------:|:-------|
| view | 7 | 7 | PASS |
| view_infer | 1 | 1 | PASS |
| reshape | 1 | 1 | PASS |
| transpose | 8 | 8 | PASS |
| permute | 7 | 7 | PASS |
| unsqueeze | 8 | 8 | PASS |
| squeeze | 7 | 7 | PASS |
| flatten | 8 | 8 | PASS |
| repeat | 7 | 7 | PASS |
| __add__ | 7 | 7 | PASS |
| __sub__ | 7 | 7 | PASS |
| __mul__ | 7 | 7 | PASS |
| __div__ | 6 | 6 | PASS |
| __pow__ | 4 | 4 | PASS |
| __neg__ | 7 | 7 | PASS |
| __matmul__ | 6 | 6 | PASS |
| cos | 7 | 7 | PASS |
| sin | 7 | 7 | PASS |
| softmax | 7 | 7 | PASS |
| cat | 8 | 8 | PASS |
| stack | 9 | 9 | PASS |

**Total: 136/136**

---

## view (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## view_infer (1/1 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| infer_dim | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## reshape (1/1 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| alias | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## transpose (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| neg_dim | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## permute (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## unsqueeze (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| neg_dim | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## squeeze (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## flatten (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| default_end | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## repeat (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## __add__ (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## __sub__ (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## __mul__ (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## __div__ (6/6 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## __pow__ (4/4 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## __neg__ (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |

## __matmul__ (6/6 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## cos (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |

## sin (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ⊘ | 0.00e+00 | 0.00e+00 |

## softmax (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## cat (8/8 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| 3_tensors | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## stack (9/9 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| dim0 | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| dim_neg1 | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

---
Tolerance: rtol=0.0001, atol=1e-05 | Seed: 42