# Comparison Results (track_head_plugin) -- 2026-04-15 10:28:05

## test_modules.py  --  PASS

### stdout

```
================================================================================
TEST 1: MemoryBank forward (no mask)
================================================================================

  MemoryBank output:
    PyTorch shape: [10, 256]
    TTSim   shape: [10, 256]
    Max diff: 7.152557e-07, Mean diff: 1.165014e-07
    [OK] Match (atol=0.0001)

================================================================================
TEST 2: MemoryBank forward (with mask)
================================================================================

  MemoryBank with mask:
    PyTorch shape: [10, 256]
    TTSim   shape: [10, 256]
    Max diff: 5.960464e-07, Mean diff: 1.182938e-07
    [OK] Match (atol=0.0001)

================================================================================
TEST 3: QueryInteractionModule forward (no pos update)
================================================================================

  QIM output (no pos):
    PyTorch shape: [10, 512]
    TTSim   shape: [10, 512]
    Max diff: 4.768372e-07, Mean diff: 4.573395e-08
    [OK] Match (atol=0.0001)

================================================================================
TEST 4: QueryInteractionModule forward (with pos update)
================================================================================

  QIM output (with pos):
    PyTorch shape: [10, 512]
    TTSim   shape: [10, 512]
    Max diff: 7.152557e-07, Mean diff: 8.987370e-08
    [OK] Match (atol=0.0001)

================================================================================
TEST 5: MemoryBank analytical_param_count
================================================================================
  Total MemoryBank params: 461,568

  PyTorch param count: 395,776
  TTSim  param count: 461,568
  [OK] param count is 461,568

================================================================================
TEST 6: QIM analytical_param_count
================================================================================
  Total QueryInteractionModule params: 527,872
  Total QueryInteractionModule params: 659,968

  QIM (no pos):   527,872
  QIM (with pos): 659,968
  [OK] Param count difference makes sense

================================================================================
SUMMARY: 6 passed, 0 failed out of 6
================================================================================
[OK] All tests passed.
```

---


