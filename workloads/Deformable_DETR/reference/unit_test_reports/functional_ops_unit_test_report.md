# Functional Ops Unit Test Report
**TTSim data_compute validation** | **157/157 passed** | PASS
Generated: 2026-02-17 14:59:06 | Exit: 0

---

## Summary

| Op | Passed | Total | Status |
|:---|-------:|------:|:-------|
| Relu | 7 | 7 | PASS |
| Sigmoid | 7 | 7 | PASS |
| Softmax | 7 | 7 | PASS |
| Log | 4 | 4 | PASS |
| Cos | 7 | 7 | PASS |
| Sin | 7 | 7 | PASS |
| Identity | 7 | 7 | PASS |
| InverseSigmoid | 4 | 4 | PASS |
| Glu | 7 | 7 | PASS |
| Tanh | 7 | 7 | PASS |
| Exp | 6 | 6 | PASS |
| Sqrt | 5 | 5 | PASS |
| Add | 7 | 7 | PASS |
| Sub | 7 | 7 | PASS |
| Mul | 7 | 7 | PASS |
| Div | 6 | 6 | PASS |
| Pow | 5 | 5 | PASS |
| MatMul | 7 | 7 | PASS |
| MatMul_broadcast | 1 | 1 | PASS |
| Einsum | 7 | 7 | PASS |
| Cdist | 7 | 7 | PASS |
| Tile | 7 | 7 | PASS |
| Concat | 7 | 7 | PASS |
| GroupNorm | 7 | 7 | PASS |
| LayerNorm | 7 | 7 | PASS |

**Total: 157/157**

---

## Relu (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Sigmoid (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Softmax (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Log (4/4 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Cos (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Sin (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Identity (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## InverseSigmoid (4/4 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Glu (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Tanh (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Exp (6/6 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Sqrt (5/5 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Add (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Sub (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Mul (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Div (6/6 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Pow (5/5 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## MatMul (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## MatMul_broadcast (1/1 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Einsum (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Cdist (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Tile (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## Concat (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## GroupNorm (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

## LayerNorm (7/7 PASS)

| Edge Case | Shape | Numerical | Max Diff | Mean Diff |
|:----------|:------|:----------|:---------|:----------|
| positive | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| negative | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| zeros | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| mixed | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| small | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| large | ✅ | ✅ | 0.00e+00 | 0.00e+00 |
| minimum_input | ✅ | ✅ | 0.00e+00 | 0.00e+00 |

---
Tolerance: rtol=0.0001, atol=1e-05 | Seed: 42