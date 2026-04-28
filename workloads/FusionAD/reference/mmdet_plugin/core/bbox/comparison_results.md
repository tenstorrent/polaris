# Comparison Results (heads) -- 2026-04-15 09:21:10

## test_bbox_util.py  --  PASS

### stdout

```
======================================================================
bbox/util.py — PyTorch vs TTSIM Validation
======================================================================

======================================================================
TEST 1: normalize_bbox shape — no velocity (7→8)
======================================================================
  [PASS] PyTorch shape
  [PASS] TTSIM  shape

======================================================================
TEST 2: normalize_bbox shape — with velocity (9→10)
======================================================================
  [PASS] PyTorch shape
  [PASS] TTSIM  shape

======================================================================
TEST 3: denormalize_bbox shape — no velocity (8→7)
======================================================================
  [PASS] PyTorch shape
  [PASS] TTSIM  shape

======================================================================
TEST 4: denormalize_bbox shape — with velocity (10→9)
======================================================================
  [PASS] PyTorch shape
  [PASS] TTSIM  shape

======================================================================
TEST 5: normalize_bbox numerical — no velocity
======================================================================
  [PASS] Arrays close (atol=1e-5)
  [PASS] cx pass-through
  [PASS] cy pass-through
  [PASS] log(w) match
  [PASS] log(l) match
  [PASS] cz pass-through
  [PASS] log(h) match
  [PASS] sin(rot) match
  [PASS] cos(rot) match

======================================================================
TEST 6: normalize_bbox numerical — with velocity
======================================================================
  [PASS] Arrays close (atol=1e-5)
  [PASS] vx pass-through
  [PASS] vy pass-through

======================================================================
TEST 7: denormalize_bbox numerical — no velocity
======================================================================
  [PASS] Arrays close (atol=1e-4)
  [PASS] cx match
  [PASS] cy match
  [PASS] cz match
  [PASS] w (exp) match
  [PASS] l (exp) match
  [PASS] h (exp) match
  [PASS] rot (atan2) match

======================================================================
TEST 8: denormalize_bbox numerical — with velocity
======================================================================
  [PASS] Arrays close (atol=1e-4)
  [PASS] vx match
  [PASS] vy match

======================================================================
TEST 9: Round-trip consistency — no velocity
======================================================================
  [PASS] PT round-trip (atol=1e-5)
  [PASS] TTSIM normalize shape
  [PASS] TTSIM denormalize shape
  [PASS] TTSIM round-trip (atol=1e-4)

======================================================================
TEST 10: Round-trip consistency — with velocity
======================================================================
  [PASS] PT round-trip (atol=1e-5)
  [PASS] TTSIM denormalize shape
  [PASS] TTSIM round-trip (atol=1e-4)

======================================================================
TEST 11: Various batch sizes
======================================================================
  [PASS] batch=1  shape [1,7]→[1,8]
  [PASS] batch=16  shape [16,7]→[16,8]
  [PASS] batch=64  shape [64,7]→[64,8]

======================================================================
SUMMARY: 41 passed, 0 failed out of 41 checks
======================================================================
```

---

## test_util.py  --  PASS

### stdout

```
================================================================================
TEST 1: NormalizeBbox TTSim Module – 9-dim (with velocity)
================================================================================

  NormalizeBbox (9-dim):
    PyTorch shape: (3, 10)
    TTSim   shape: (3, 10)
    Max diff: 2.980232e-08, Mean diff: 2.235174e-09
    [OK] Match (atol=1e-05)

[OK] TEST 1 PASSED

================================================================================
TEST 2: NormalizeBbox TTSim Module – 7-dim (no velocity)
================================================================================

  NormalizeBbox (7-dim):
    PyTorch shape: (3, 8)
    TTSim   shape: (3, 8)
    Max diff: 2.980232e-08, Mean diff: 2.793968e-09
    [OK] Match (atol=1e-05)

[OK] TEST 2 PASSED

================================================================================
TEST 3: DenormalizeBbox TTSim Module – 10-dim (with velocity)
================================================================================

  DenormalizeBbox (10-dim):
    PyTorch shape: (3, 9)
    TTSim   shape: (3, 9)
    Max diff: 5.960464e-08, Mean diff: 6.622738e-09
    [OK] Match (atol=1e-05)

[OK] TEST 3 PASSED

================================================================================
TEST 4: DenormalizeBbox TTSim Module – 8-dim (no velocity)
================================================================================

  DenormalizeBbox (8-dim):
    PyTorch shape: (3, 7)
    TTSim   shape: (3, 7)
    Max diff: 5.960464e-08, Mean diff: 8.514950e-09
    [OK] Match (atol=1e-05)

[OK] TEST 4 PASSED

================================================================================
TEST 5: Round-trip (TTSim NormalizeBbox → DenormalizeBbox)
================================================================================

  Round-trip 9-dim (TTSim):
    PyTorch shape: (3, 9)
    TTSim   shape: (3, 9)
    Max diff: 5.960464e-08, Mean diff: 6.070843e-09
    [OK] Match (atol=1e-05)

  Round-trip 7-dim (TTSim):
    PyTorch shape: (3, 7)
    TTSim   shape: (3, 7)
    Max diff: 5.960464e-08, Mean diff: 7.805371e-09
    [OK] Match (atol=1e-05)

[OK] TEST 5 PASSED

================================================================================
TEST 6: Numpy Convenience Functions vs PyTorch
================================================================================

  normalize_bbox_np:
    PyTorch shape: (3, 10)
    TTSim   shape: (3, 10)
    Max diff: 2.980232e-08, Mean diff: 2.235174e-09
    [OK] Match (atol=1e-05)

  denormalize_bbox_np:
    PyTorch shape: (3, 9)
    TTSim   shape: (3, 9)
    Max diff: 5.960464e-08, Mean diff: 6.622738e-09
    [OK] Match (atol=1e-05)

[OK] TEST 6 PASSED

================================================================================
TEST 7: Batch Dimensions (TTSim modules)
================================================================================

  NormalizeBbox (4, 9):
    PyTorch shape: (4, 10)
    TTSim   shape: (4, 10)
    Max diff: 2.980232e-08, Mean diff: 9.778887e-10
    [OK] Match (atol=1e-05)

  DenormalizeBbox (4, 9):
    PyTorch shape: (4, 9)
    TTSim   shape: (4, 9)
    Max diff: 5.960464e-08, Mean diff: 6.001857e-09
    [OK] Match (atol=1e-05)

  NormalizeBbox (2, 5, 9):
    PyTorch shape: (2, 5, 10)
    TTSim   shape: (2, 5, 10)
    Max diff: 1.192093e-07, Mean diff: 2.668239e-09
    [OK] Match (atol=1e-05)

  DenormalizeBbox (2, 5, 9):
    PyTorch shape: (2, 5, 9)
    TTSim   shape: (2, 5, 9)
    Max diff: 1.192093e-07, Mean diff: 7.285012e-09
    [OK] Match (atol=1e-05)

[OK] TEST 7 PASSED

================================================================================
TEST 8: atan2 Edge Cases (all quadrants)
================================================================================

  atan2 quadrant test:
    PyTorch shape: (8, 9)
    TTSim   shape: (8, 9)
    Max diff: 4.768372e-07, Mean diff: 2.152390e-08
    [OK] Match (atol=0.0001)

[OK] TEST 8 PASSED

================================================================================
TEST SUMMARY
================================================================================
  TEST 1: NormalizeBbox TTSim – 9-dim (with velocity)............ [OK] PASSED
  TEST 2: NormalizeBbox TTSim – 7-dim (no velocity).............. [OK] PASSED
  TEST 3: DenormalizeBbox TTSim – 10-dim (with velocity)......... [OK] PASSED
  TEST 4: DenormalizeBbox TTSim – 8-dim (no velocity)............ [OK] PASSED
  TEST 5: Round-trip (TTSim NormalizeBbox → DenormalizeBbox)..... [OK] PASSED
  TEST 6: Numpy Convenience Functions vs PyTorch................. [OK] PASSED
  TEST 7: Batch Dimensions (TTSim modules)....................... [OK] PASSED
  TEST 8: atan2 Edge Cases (all quadrants)....................... [OK] PASSED

Total: 8/8 tests passed

================================================================================
All tests passed!
================================================================================
```

---


