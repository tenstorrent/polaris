# Comparison Results (backbones) -- 2026-04-15 10:30:19

## test_sparse_encoder_hd.py  --  PASS

### stdout

```
======================================================================
sparse_encoder_hd.py — PyTorch vs TTSIM Validation
======================================================================

======================================================================
TEST 1: SparseConvModule shape — stride=1
======================================================================
  [PASS] output shape
  [PASS] spatial preserved
  [PASS] PT shape matches

======================================================================
TEST 2: SparseConvModule shape — stride=2
======================================================================
  [PASS] output shape
  [PASS] spatial halved
  [PASS] PT shape matches

======================================================================
TEST 3: SparseBasicBlockTTSIM shape
======================================================================
  [PASS] output shape
  [PASS] channels preserved
  [PASS] PT shape matches

======================================================================
TEST 4: SparseConvModule numerical — stride=1
======================================================================
  [PASS] TTSIM shape == PT shape
  [PASS] TTSIM data populated (shape-only mode)

======================================================================
TEST 5: SparseConvModule numerical — stride=2
======================================================================
  [PASS] TTSIM shape == PT shape
  [PASS] TTSIM data populated (shape-only mode)

======================================================================
TEST 6: SparseBasicBlockTTSIM numerical
======================================================================
  [PASS] TTSIM shape == PT shape
  [PASS] TTSIM data populated (shape-only mode)

======================================================================
TEST 7: Full SparseEncoderHD shape — conv_module mode
======================================================================
  [PASS] output dims == 4 (B,C,H,W)
  [PASS] batch preserved
  [PASS] output channels
  [PASS] spatial H reduced
  [PASS] spatial W reduced

======================================================================
TEST 8: Full SparseEncoderHD shape — basicblock mode
======================================================================
  [PASS] output dims == 4
  [PASS] batch preserved
  [PASS] output channels

======================================================================
TEST 9: Full SparseEncoderHD shape — keep_depth=True
======================================================================
  [PASS] output dims == 5
  [PASS] batch preserved
  [PASS] output channels

======================================================================
TEST 10: Config & attribute preservation
======================================================================
  [PASS] sparse_shape
  [PASS] in_channels
  [PASS] base_channels
  [PASS] output_channels
  [PASS] encoder_channels
  [PASS] encoder_strides
  [PASS] stage_num
  [PASS] keep_depth default
  [PASS] param count > 0

======================================================================
TEST 11: Various sparse_shape sizes
======================================================================
  [PASS] default HD: ndim==4
  [PASS] default HD: batch==1
  [PASS] default HD: C==128
  [PASS] square 128: ndim==4
  [PASS] square 128: batch==2
  [PASS] square 128: C==128
  [PASS] small 64: ndim==4
  [PASS] small 64: batch==1
  [PASS] small 64: C==128

======================================================================
SUMMARY: 44 passed, 0 failed out of 44 checks
======================================================================
```

---

## test_vovnet.py  --  PASS

### stdout

```
======================================================================
vovnet.py — PyTorch vs TTSIM Validation
======================================================================

======================================================================
TEST 1: Conv3x3Block shape
======================================================================
  [PASS] TTSIM shape
  [PASS] PT shape
  [PASS] shapes match

======================================================================
TEST 2: Conv1x1Block shape
======================================================================
  [PASS] TTSIM shape
  [PASS] PT shape
  [PASS] shapes match

======================================================================
TEST 3: DWConv3x3Block shape
======================================================================
  [PASS] TTSIM shape
  [PASS] PT shape
  [PASS] shapes match

======================================================================
TEST 4: Hsigmoid numerical
======================================================================
  [PASS] TTSIM shape == PT shape
  [PASS] numerical close (atol=1e-5)

======================================================================
TEST 5: eSEModule shape
======================================================================
  [PASS] TTSIM shape
  [PASS] PT shape
  [PASS] shapes match

======================================================================
TEST 6: _OSA_module shape (no depthwise)
======================================================================
  [PASS] TTSIM shape
  [PASS] PT shape
  [PASS] shapes match

======================================================================
TEST 7: _OSA_module shape (depthwise)
======================================================================
  [PASS] TTSIM shape
  [PASS] isReduced flag
  [PASS] PT shape
  [PASS] shapes match

======================================================================
TEST 8: _OSA_stage shape (with pool)
======================================================================
  [PASS] TTSIM spatial halved (pool)
  [PASS] TTSIM channels
  [PASS] has_pool flag
  [PASS] PT channels
  [PASS] PT spatial == TTSIM spatial

======================================================================
TEST 9: _OSA_stage shape (no pool, stage2)
======================================================================
  [PASS] TTSIM spatial preserved (no pool)
  [PASS] TTSIM channels
  [PASS] has_pool flag
  [PASS] PT spatial preserved
  [PASS] shapes match

======================================================================
TEST 10: Full VoVNet shape — V-19-slim-eSE
======================================================================
  [PASS] same output keys
  [PASS] stage2 shapes match
  [PASS] stage2 channels == 112
  [PASS] stage3 shapes match
  [PASS] stage3 channels == 256
  [PASS] stage4 shapes match
  [PASS] stage4 channels == 384
  [PASS] stage5 shapes match
  [PASS] stage5 channels == 512

======================================================================
TEST 11: Full VoVNet shape — V-99-eSE
======================================================================
  [PASS] stage2 shapes match
  [PASS] stage2 channels == 256
  [PASS] stage3 shapes match
  [PASS] stage3 channels == 512
  [PASS] stage4 shapes match
  [PASS] stage4 channels == 768
  [PASS] stage5 shapes match
  [PASS] stage5 channels == 1024

======================================================================
TEST 12: Hsigmoid numerical — PyTorch vs TTSIM
======================================================================
  [PASS] values close (atol=1e-5)
  [PASS] x=-10 → 0
  [PASS] x=0 → 0.5
  [PASS] x=10 → 1

======================================================================
TEST 13: Config & stride verification
======================================================================
  [PASS] stem channels
  [PASS] stage2 channels
  [PASS] stage3 channels
  [PASS] stage4 channels
  [PASS] stage5 channels
  [PASS] stem stride
  [PASS] stage2 stride
  [PASS] stage3 stride
  [PASS] stage4 stride
  [PASS] stage5 stride
  [PASS] 4 stage_names
  [PASS] stage names correct

======================================================================
TEST 14: Various input sizes
======================================================================
  [PASS] 224x224: shapes match
  [PASS] 320x320: shapes match
  [PASS] 128x128: shapes match

======================================================================
TEST 15: Conv3x3Block numerical — shared weights
======================================================================
  Conv3x3Block forward:
    PyTorch shape: [2, 32, 16, 16]
    TTSim   shape: [2, 32, 16, 16]
    Max diff: 4.768372e-07, Mean diff: 2.727982e-08
    [OK] Match (atol=0.0001)
  [PASS] numerical match

======================================================================
TEST 16: eSEModule numerical — shared weights
======================================================================
  eSEModule forward:
    PyTorch shape: [2, 32, 8, 8]
    TTSim   shape: [2, 32, 8, 8]
    Max diff: 1.192093e-07, Mean diff: 6.243823e-10
    [OK] Match (atol=0.0001)
  [PASS] numerical match

======================================================================
TEST 17: _OSA_module numerical — shared weights
======================================================================
  OSA module forward:
    PyTorch shape: [1, 64, 8, 8]
    TTSim   shape: [1, 64, 8, 8]
    Max diff: 1.490116e-07, Mean diff: 1.372985e-08
    [OK] Match (atol=0.0001)
  [PASS] numerical match

======================================================================
SUMMARY: 70 passed, 0 failed out of 70 checks
======================================================================
```

---


