# Comparison Results (bevformer) — 2026-03-21 00:12:39

## test_detr_loss.py  —  PASS

### stdout

```

================================================================================
DETR Losses Comparison: PyTorch vs ttsim
================================================================================

================================================================================
TEST 1: LinesL1Loss (Smooth L1, beta=0.5)
================================================================================
PyTorch LinesL1Loss (Smooth): 0.044957
ttsim LinesL1Loss (Smooth):   0.044957
Difference: 0.00000000
[PASS] PASS: LinesL1Loss (Smooth) matches!

================================================================================
TEST 2: LinesL1Loss (Standard L1, beta=0)
================================================================================
PyTorch LinesL1Loss (L1): 0.056769
ttsim LinesL1Loss (L1):   0.056769
Difference: 0.00000001
[PASS] PASS: LinesL1Loss (L1) matches!

================================================================================
TEST 3: MasksLoss (BCE)
================================================================================
PyTorch MasksLoss: 0.810909
ttsim MasksLoss:   0.810909
Difference: 0.00000000
[PASS] PASS: MasksLoss matches!

================================================================================
TEST 4: LenLoss (Cross Entropy)
================================================================================
PyTorch LenLoss: 3.442037
ttsim LenLoss:   3.442037
Difference: 0.00000000
[PASS] PASS: LenLoss matches!

================================================================================
TEST 5: Weighted Reduction with avg_factor
================================================================================
PyTorch Weighted Loss: 1.820282
ttsim Weighted Loss:   1.820282
Difference: 0.00000000
[PASS] PASS: Weighted reduction matches!

================================================================================
SUMMARY
================================================================================
[PASS] PASS: LinesL1Loss (Smooth)
[PASS] PASS: LinesL1Loss (L1)
[PASS] PASS: MasksLoss (BCE)
[PASS] PASS: LenLoss (CE)
[PASS] PASS: Weighted Reduction

Total: 5/5 tests passed

[PASS] All tests passed! PyTorch and ttsim implementations match perfectly!
```

### stderr

```
C:\Users\SaSagar\AppData\Local\miniforge3\envs\polaris\Lib\site-packages\requests\__init__.py:109: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (None)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
```

---

## test_seg_loss.py  —  PASS

### stdout

```

================================================================================
Segmentation Losses Comparison: PyTorch vs ttsim
================================================================================

================================================================================
TEST 1: MaskFocalLoss
================================================================================
PyTorch MaskFocalLoss: 0.174577
[WARN]  ttsim output shape: (1, 1, 32, 32), expected scalar
ttsim MaskFocalLoss:   0.174577
Difference: 0.00000001
[PASS] PASS: MaskFocalLoss matches!

================================================================================
TEST 2: MaskDiceLoss
================================================================================
PyTorch MaskDiceLoss: 0.368535
ttsim MaskDiceLoss:   0.368535
Difference: 0.00000000
[PASS] PASS: MaskDiceLoss matches!

================================================================================
TEST 3: Perfect Prediction (Edge Case)
================================================================================
PyTorch DiceLoss (perfect): 0.000000
ttsim DiceLoss (perfect):   0.000000
Difference: 0.00000006
[PASS] PASS: Perfect prediction gives low loss!

================================================================================
SUMMARY
================================================================================
[PASS] PASS: MaskFocalLoss
[PASS] PASS: MaskDiceLoss
[PASS] PASS: Perfect Prediction

Total: 3/3 tests passed

[PASS] All tests passed! PyTorch and ttsim implementations match perfectly!
```

### stderr

```
C:\Users\SaSagar\AppData\Local\miniforge3\envs\polaris\Lib\site-packages\requests\__init__.py:109: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (None)/charset_normalizer (3.4.4) doesn't match a supported version!
  warnings.warn(
```

---


