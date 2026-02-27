
================================================================================
  DeformableDETR — Comprehensive Shape Validation
  PyTorch vs TTSim (Components + End-to-End)
================================================================================

================================================================================
  Test 1: MLP (Multi-Layer Perceptron)
================================================================================

  Config: bbox_head  input=(2, 100, 256)  MLP(256,256,4,3)
  ✓ PASS MLP.bbox_head output:  PyTorch [2, 100, 4]  vs  TTSim [2, 100, 4]

  Config: class_head  input=(2, 100, 256)  MLP(256,256,91,1)
  ✓ PASS MLP.class_head output:  PyTorch [2, 100, 91]  vs  TTSim [2, 100, 91]

  Config: deep  input=(4, 50, 128)  MLP(128,512,4,5)
  ✓ PASS MLP.deep output:  PyTorch [4, 50, 4]  vs  TTSim [4, 50, 4]

  Config: 2D_input  input=(8, 256)  MLP(256,256,4,3)
  ✓ PASS MLP.2D_input output:  PyTorch [8, 4]  vs  TTSim [8, 4]

  Config: 4D_input  input=(6, 2, 100, 256)  MLP(256,256,4,3)
  ✓ PASS MLP.4D_input output:  PyTorch [6, 2, 100, 4]  vs  TTSim [6, 2, 100, 4]

================================================================================
  Test 2: PostProcess (top-k detection output)
================================================================================

  Config: standard  bs=2 queries=100 classes=91
  ✓ PASS   img0 scores:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img0 labels:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img0 boxes:  PyTorch [100, 4]  vs  TTSim [100, 4]
  ✓ PASS   img1 scores:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img1 labels:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img1 boxes:  PyTorch [100, 4]  vs  TTSim [100, 4]

  Config: single  bs=1 queries=50 classes=20
  ✓ PASS   img0 scores:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img0 labels:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img0 boxes:  PyTorch [100, 4]  vs  TTSim [100, 4]

  Config: large_cls  bs=2 queries=300 classes=250
  ✓ PASS   img0 scores:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img0 labels:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img0 boxes:  PyTorch [100, 4]  vs  TTSim [100, 4]
  ✓ PASS   img1 scores:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img1 labels:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS   img1 boxes:  PyTorch [100, 4]  vs  TTSim [100, 4]

================================================================================
  Test 3: sigmoid_focal_loss
================================================================================

  Config: small  N=4 C=10
  ✓ PASS focal_loss.small:  PyTorch 0.568407  vs  TTSim 0.568407  (diff=2.77e-09)

  Config: medium  N=16 C=91
  ✓ PASS focal_loss.medium:  PyTorch 0.521136  vs  TTSim 0.521136  (diff=1.93e-08)

  Config: large  N=64 C=250
  ✓ PASS focal_loss.large:  PyTorch 0.525316  vs  TTSim 0.525316  (diff=1.96e-08)

================================================================================
  Test 4: dice_loss
================================================================================

  Config: small  N=4 HW=100
  ✓ PASS dice_loss.small:  PyTorch 0.502764  vs  TTSim 0.502764  (diff=4.11e-08)

  Config: medium  N=8 HW=400
  ✓ PASS dice_loss.medium:  PyTorch 0.506081  vs  TTSim 0.506081  (diff=5.24e-09)

================================================================================
  Test 5: inverse_sigmoid
================================================================================

  Config: uniform  shape=(2, 100, 2)
  ✓ PASS inv_sigmoid.uniform output:  PyTorch [2, 100, 2]  vs  TTSim [2, 100, 2]
    ✓ numerical max_diff=1.19e-07

  Config: 3d  shape=(4, 50, 4)
  ✓ PASS inv_sigmoid.3d output:  PyTorch [4, 50, 4]  vs  TTSim [4, 50, 4]
    ✓ numerical max_diff=1.19e-07

  Config: 1d  shape=(256,)
  ✓ PASS inv_sigmoid.1d output:  PyTorch [256]  vs  TTSim [256]
    ✓ numerical max_diff=1.19e-07

================================================================================
  Test 6: SetCriterion (loss computation)
================================================================================

  PyTorch loss keys: ['cardinality_error', 'class_error', 'loss_bbox', 'loss_ce', 'loss_giou']
  TTSim loss keys:   ['cardinality_error', 'class_error', 'loss_bbox', 'loss_ce', 'loss_giou']
  ✓ PASS loss[cardinality_error]:  PyTorch 5.500000  vs  TTSim 5.500000  (diff=0.00e+00)
  ✓ PASS loss[class_error]:  PyTorch 80.000000  vs  TTSim 80.000000  (diff=0.00e+00)
  ✓ PASS loss[loss_bbox]:  PyTorch 0.621418  vs  TTSim 0.621418  (diff=7.15e-08)
  ✓ PASS loss[loss_ce]:  PyTorch 5.559540  vs  TTSim 5.559540  (diff=1.40e-07)
  ✓ PASS loss[loss_giou]:  PyTorch 1.022536  vs  TTSim 1.022536  (diff=9.54e-08)

================================================================================
  Test 7: PostProcessSegm (mask post-processing)
================================================================================
  ✓ PASS img0 mask:  PyTorch [5, 1, 200, 300]  vs  TTSim [5, 1, 200, 300]
  ✓ PASS img1 mask:  PyTorch [5, 1, 150, 250]  vs  TTSim [5, 1, 150, 250]

================================================================================
  Test 8: SetCriterion with auxiliary outputs
================================================================================

  PyTorch: 13 loss keys
  TTSim:   13 loss keys
  ✓ PASS  Loss key sets match (13 keys)
  ✓ aux key 'loss_ce_0': PT=True TT=True

================================================================================
  Test 9: PostProcess numerical comparison
================================================================================
  ✓ img0 scores max_diff=5.96e-08
  ✓ img0 labels match=True
  ✓ img0 boxes max_diff=0.00
  ✓ img1 scores max_diff=5.96e-08
  ✓ img1 labels match=True
  ✓ img1 boxes max_diff=0.00

================================================================================
  Test 10: DeformableTransformer (Output Shapes)
================================================================================
  ✓ PASS hs output:  PyTorch [1, 1, 10, 64]  vs  TTSim [1, 1, 10, 64]
  ✓ PASS reference_points:  PyTorch [1, 10, 2]  vs  TTSim [1, 10, 2]
  ✓ PASS hs expected dims:  PyTorch [1, 1, 10, 64]  vs  TTSim [1, 1, 10, 64]
  ✓ PASS ref expected dims:  PyTorch [1, 10, 2]  vs  TTSim [1, 10, 2]

================================================================================
  Test 11: MLP (Bbox Head — Standalone Shape)
================================================================================
  ✓ PASS MLP output:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]
  ✓ PASS MLP expected [B,Q,4]:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]

================================================================================
  Test 12: inverse_sigmoid (Standalone Shape)
================================================================================
  ✓ PASS inv_sigmoid.2d:  PyTorch [1, 10, 2]  vs  TTSim [1, 10, 2]
  ✓ PASS inv_sigmoid.2d == input shape:  PyTorch [1, 10, 2]  vs  TTSim [1, 10, 2]
  ✓ PASS inv_sigmoid.4d:  PyTorch [2, 5, 10, 4]  vs  TTSim [2, 5, 10, 4]
  ✓ PASS inv_sigmoid.4d == input shape:  PyTorch [2, 5, 10, 4]  vs  TTSim [2, 5, 10, 4]
  ✓ PASS inv_sigmoid.1d:  PyTorch [100]  vs  TTSim [100]
  ✓ PASS inv_sigmoid.1d == input shape:  PyTorch [100]  vs  TTSim [100]

================================================================================
  Test 13: Class + Bbox Head Output Shapes
================================================================================
  ✓ PASS class_embed output:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS bbox_embed output:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]
  ✓ PASS class expected [B,Q,num_classes]:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS bbox expected [B,Q,4]:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]

================================================================================
  Test 14: Full DeformableDETR Shapes (1 enc + 1 dec)
================================================================================
  ✓ PASS pred_logits:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS pred_boxes:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]
  ✓ PASS pred_logits expected [B,Q,nc]:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS pred_boxes expected [B,Q,4]:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]

  aux_outputs present: PT=True  TT=False
  PT aux count: 0

================================================================================
  Test 15: Full DeformableDETR Shapes (3 enc + 3 dec)
================================================================================
  ✓ PASS pred_logits:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS pred_boxes:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]
  ✓ PASS pred_logits expected [B,Q,nc]:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS pred_boxes expected [B,Q,4]:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]

  aux_outputs count: PyTorch=2  TTSim=2  expected=2
  ✓ PASS  aux_outputs count matches (2)
  ✓ PASS aux[0].pred_logits:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS aux[0].pred_boxes:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]
  ✓ PASS aux[1].pred_logits:  PyTorch [1, 10, 10]  vs  TTSim [1, 10, 10]
  ✓ PASS aux[1].pred_boxes:  PyTorch [1, 10, 4]  vs  TTSim [1, 10, 4]

================================================================================
  SUMMARY
================================================================================

  Part A — Component Shape Tests:
    ✓  01_MLP
    ✓  02_PostProcess
    ✓  03_sigmoid_focal_loss
    ✓  04_dice_loss
    ✓  05_inverse_sigmoid
    ✓  06_SetCriterion
    ✓  07_PostProcessSegm
    ✓  08_SetCriterion_aux
    ✓  09_PostProcess_num

  Part B — End-to-End Pipeline Shape Tests:
    ✓  10_Transformer
    ✓  11_MLP_standalone
    ✓  12_inverse_sigmoid
    ✓  13_Heads
    ✓  14_Full_DETR_1x1
    ✓  15_Full_DETR_3x3

  Total checks: 67  |  Passed: 67  |  Failed: 0

  OVERALL: ALL PASSED ✓


*Report saved to: /home/aughag/Videos/TensTorrent/polaris/workloads/Deformable_DETR/reference/reports/deformable_DETR_shape_validation.md*
