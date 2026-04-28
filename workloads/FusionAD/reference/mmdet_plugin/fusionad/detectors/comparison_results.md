# Comparison Results (detectors) -- 2026-04-15 09:45:51

## test_fusionad_e2e.py  --  PASS

### stdout

```
================================================================================
TEST 1: pop_elem_in_result - TTSim vs PyTorch numerical match
================================================================================
  Remaining keys: ['labels', 'pred_boxes', 'scores', 'track_query_embeddings']
  'track_query_embeddings': shape (5, 128) - match OK
  'scores': shape (5,) - match OK
  'pred_boxes': shape (5, 10) - match OK
  'labels': shape (5,) - match OK

[OK] TEST 1

================================================================================
TEST 2: pop_elem_in_result - edge cases (empty, no-match, multi-suffix)
================================================================================
  Empty dict -> empty result: OK
  No matching keys -> all keys survived: OK
  Pop non-existent key (no error): OK
  Multiple query/embedding suffixes removed: OK

[OK] TEST 2

================================================================================
TEST 3: FusionAD construction - attributes, sub-modules, inherited shapes
================================================================================
  query_embedding: callable OK (type=SimOpHandle)
  reference_points: callable OK (type=Linear)
  All attributes and sub-modules present: OK

[OK] TEST 3

================================================================================
TEST 4: FusionAD properties - with_*_head toggling (None vs dummy)
================================================================================
  All-None: with_seg=False, with_motion=False, with_occ=False, with_plan=False: OK
  Dummy heads: with_seg=True, with_motion=True, with_occ=False, with_plan=False: OK
  Head references correct: OK

[OK] TEST 4

================================================================================
TEST 5: FusionAD inheritance - MRO, inherited methods
================================================================================
  FusionADTrack in MRO: OK
  instance MRO contains FusionADTrack: OK
  instance MRO contains Module: OK
  All inherited methods present: OK

[OK] TEST 5

================================================================================
TEST 6: __call__ input validation - TypeError on non-list img_metas
================================================================================
  Caught expected TypeError: img_metas must be a list, but got <class 'str'>

[OK] TEST 6

================================================================================
TEST 7: ego_info tensor - F._from_data vs torch.from_numpy comparison
================================================================================

  ego_info:
    PyTorch shape: (18,)
    TTSim   shape: (18,)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-07)

[OK] TEST 7

================================================================================
TEST 8: can_bus delta - 2-frame numerical comparison
================================================================================

  Frame 1 delta pos:
    PyTorch shape: (3,)
    TTSim   shape: (3,)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-07)

  Frame 2 delta pos:
    PyTorch shape: (3,)
    TTSim   shape: (3,)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-07)

  Frame 1 delta angle: TTSim=45.0, PT=45.0 - OK
  Frame 2 delta angle: TTSim=5.0, PT=5.0 - OK

[OK] TEST 8

================================================================================
TEST 9: Post-processing pipeline - dict assembly numerical match
================================================================================
  Remaining keys after post-processing: ['labels', 'pred_boxes', 'scores']
  'scores': shape (5,) - match OK
  'pred_boxes': shape (5, 10) - match OK
  'labels': shape (5,) - match OK
  Result assembly with token: OK

[OK] TEST 9

================================================================================
TEST 10: Embedding + Linear forward pass - PT vs TT numerical match
================================================================================

  Embedding output:
    PyTorch shape: (3, 128)
    TTSim   shape: (3, 128)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-05)

  Linear output:
    PyTorch shape: (3, 3)
    TTSim   shape: (3, 3)
    Max diff: 1.192093e-07, Mean diff: 1.986821e-08
    [OK] Match (atol=0.0001)

[OK] TEST 10

================================================================================
TEST 11: Polaris-mode construction (use_lidar=True) - backbone + BEV encoder + LiDAR
================================================================================
  img_backbone: ResNetBackbone OK
  img_neck: FPN OK
  pts_backbone: SparseEncoderHD OK
  bev_encoder: BEVFormerEncoder OK
  ResNet DCN (stage_with_dcn): checked
  score_thresh=0.4, filter_score_thresh=0.35
  query_interact.update_query_pos=True: OK
  planning_head BEV adapter (3 blocks): OK
  All backbones, BEV encoder, task heads, DCN, QIM, adapter present: OK

[OK] TEST 11

================================================================================
TEST 12: create_input_tensors — img + voxels shapes (use_lidar=True)
================================================================================
  img shape: Shape([12, 3, 128, 256]) OK
  voxels shape: Shape([2, 5, 41, 32, 32]) OK
  img tensor set_module OK
  voxels tensor set_module OK

[OK] TEST 12

================================================================================
TEST 13: Polaris-mode construction (use_lidar=False) — no voxels, no pts_backbone
================================================================================
  pts_backbone: None OK
  bev_encoder layers: BEVFormerLayer (no LiDAR) OK
  No voxels in input_tensors: OK
  img shape: Shape([6, 3, 128, 256]) OK

[OK] TEST 13

================================================================================
SUMMARY: 13 passed, 0 failed, 13 total
================================================================================
```

### stderr

```
C:\Users\SaSagar\Downloads\TensTorrent\polaris\workloads\FusionAD\reference\mmdet_plugin\fusionad\detectors\..\..\..\..\..\..\workloads\FusionAD\projects\mmdet_plugin\fusionad\modules\custom_base_transformer_layer.py:374: UserWarning: The arguments `feedforward_channels` in BaseTransformerLayer has been deprecated, now you should set `feedforward_channels` and other FFN related arguments to a dict named `ffn_cfgs`. 
  warnings.warn(
C:\Users\SaSagar\Downloads\TensTorrent\polaris\workloads\FusionAD\reference\mmdet_plugin\fusionad\detectors\..\..\..\..\..\..\workloads\FusionAD\projects\mmdet_plugin\fusionad\modules\custom_base_transformer_layer.py:374: UserWarning: The arguments `ffn_dropout` in BaseTransformerLayer has been deprecated, now you should set `ffn_drop` and other FFN related arguments to a dict named `ffn_cfgs`. 
  warnings.warn(
```

---

## test_fusionad_track.py  --  PASS

### stdout

```
================================================================================
TEST 1: Instances container — set/get, slicing, cat
================================================================================
  set/get fields: OK
  __getitem__: OK
  cat: OK

[OK] TEST 1

================================================================================
TEST 2: RuntimeTrackerBase — construction / defaults
================================================================================
  Attributes: OK
  update (no-op): OK

[OK] TEST 2

================================================================================
TEST 3: Embedding — TTSim F.Embedding vs nn.Embedding
================================================================================

  Embedding full lookup:
    PyTorch shape: (17, 128)
    TTSim   shape: (17, 128)
    Max diff: 0.000000e+00, Mean diff: 0.000000e+00
    [OK] Match (atol=1e-06)

[OK] TEST 3

================================================================================
TEST 4: Linear — TTSim SimNN.Linear vs nn.Linear
================================================================================

  Linear forward:
    PyTorch shape: (4, 3)
    TTSim   shape: (4, 3)
    Max diff: 1.490116e-07, Mean diff: 5.215406e-08
    [OK] Match (atol=1e-05)

[OK] TEST 4

================================================================================
TEST 5: velo_update math — PyTorch vs numpy
================================================================================

  velo_update output:
    PyTorch shape: (8, 3)
    TTSim   shape: (8, 3)
    Max diff: 3.576279e-07, Mean diff: 5.246450e-08
    [OK] Match (atol=0.001)

[OK] TEST 5

================================================================================
TEST 6: FusionADTrack construction — attributes & sub-modules
================================================================================
  query_interact.update_query_pos=False (default): OK
  query_interact.update_query_pos=True (explicit): OK
  All attributes and sub-modules present: OK

[OK] TEST 6

================================================================================
TEST 7: _generate_empty_tracks — shapes and fields
================================================================================
  All fields present with correct shapes: OK

[OK] TEST 7

================================================================================
TEST 8: Sigmoid — TTSim vs PyTorch
================================================================================

  Sigmoid output:
    PyTorch shape: (4, 10)
    TTSim   shape: (4, 10)
    Max diff: 5.960464e-08, Mean diff: 8.195639e-09
    [OK] Match (atol=1e-06)

[OK] TEST 8

================================================================================
TEST 9: InverseSigmoid — TTSim vs PyTorch
================================================================================

  InverseSigmoid output:
    PyTorch shape: (4, 10)
    TTSim   shape: (4, 10)
    Max diff: 2.384186e-07, Mean diff: 9.685754e-09
    [OK] Match (atol=1e-05)

[OK] TEST 9

================================================================================
TEST 10: _copy_tracks_for_loss — field presence
================================================================================
  field 'obj_idxes': present
  field 'matched_gt_idxes': present
  field 'disappear_time': present
  field 'scores': present
  field 'track_scores': present
  field 'pred_boxes': present
  field 'iou': present
  field 'pred_logits': present
  field 'save_period': present
  All copied fields present with correct shapes: OK

[OK] TEST 10

================================================================================
SUMMARY: 10 passed, 0 failed, 10 total
================================================================================
```

---


