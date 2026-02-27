
═════════════════════════════════════════════════════════════════
HUNGARIAN MATCHER UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 5 items

workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_focal_loss_cost 
MODULE: [1mFocalLossCost[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical inputs — baseline correctness)
├─ INPUT: logits[10,91] tgt_ids[5]
├─ SHAPE: PT=[10,5] | TT=[10,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.19e-07, mean_diff=1.77e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.421150, 0.077295, -0.835299, -0.006908, -0.209603]
├─ TT OUTPUT[0:5]: [-0.421150, 0.077295, -0.835299, -0.006908, -0.209603]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFocalLossCost[0m
├─ EDGE CASE: [93mextreme_logits[0m (Very large/small logits — sigmoid saturation)
├─ INPUT: logits[10,91] tgt_ids[5]
├─ SHAPE: PT=[10,5] | TT=[10,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=9.54e-07, mean_diff=8.64e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.750837, -3.351773, 1.480311, -9.151949, -5.177003]
├─ TT OUTPUT[0:5]: [0.750837, -3.351773, 1.480311, -9.151949, -5.177003]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFocalLossCost[0m
├─ EDGE CASE: [93muniform_probs[0m (Uniform probability (0.5) — degenerate focal loss)
├─ INPUT: logits[10,91] tgt_ids[5]
├─ SHAPE: PT=[10,5] | TT=[10,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.086643, -0.086643, -0.086643, -0.086643, -0.086643]
├─ TT OUTPUT[0:5]: [-0.086643, -0.086643, -0.086643, -0.086643, -0.086643]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFocalLossCost[0m
├─ EDGE CASE: [93msingle_target[0m (Single GT object — minimal matching)
├─ INPUT: logits[10,91] tgt_ids[1]
├─ SHAPE: PT=[10,1] | TT=[10,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=8.94e-08, mean_diff=1.97e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.042509, 0.089306, -0.404774, -0.157413, 0.017822]
├─ TT OUTPUT[0:5]: [0.042509, 0.089306, -0.404774, -0.157413, 0.017822]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFocalLossCost[0m
├─ EDGE CASE: [93mmany_classes[0m (Large class space — scalability check)
├─ INPUT: logits[10,200] tgt_ids[5]
├─ SHAPE: PT=[10,5] | TT=[10,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=1.04e-07, mean_diff=1.58e-08 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.079662, -0.266633, 0.152675, 0.238550, -0.085263]
├─ TT OUTPUT[0:5]: [0.079662, -0.266633, 0.152675, 0.238550, -0.085263]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mFocalLossCost[0m
├─ EDGE CASE: [93msparse_logits[0m (Mostly-zero logits — sparse activation pattern)
├─ INPUT: logits[10,91] tgt_ids[5]
├─ SHAPE: PT=[10,5] | TT=[10,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=7.45e-08, mean_diff=3.58e-09 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.086643, 0.230809, -0.086643, -0.086643, -0.086643]
├─ TT OUTPUT[0:5]: [-0.086643, 0.230809, -0.086643, -0.086643, -0.086643]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_l1_bbox_cost 
MODULE: [1mL1BboxCost[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical inputs — baseline correctness)
├─ INPUT: pred[10,4] tgt[5,4]
├─ SHAPE: PT=[10,5] | TT=[10,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-06)[0m
├─ PT OUTPUT[0:5]: [1.716307, 0.902825, 1.352095, 1.078155, 2.000874]
├─ TT OUTPUT[0:5]: [1.716307, 0.902825, 1.352095, 1.078155, 2.000874]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mL1BboxCost[0m
├─ EDGE CASE: [93midentical_boxes[0m (Identical pred/tgt boxes — zero distance)
├─ INPUT: pred[5,4] tgt[5,4]
├─ SHAPE: PT=[5,5] | TT=[5,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-06)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mL1BboxCost[0m
├─ EDGE CASE: [93mdistant_boxes[0m (Widely separated boxes — large L1 distances)
├─ INPUT: pred[5,4] tgt[5,4]
├─ SHAPE: PT=[5,5] | TT=[5,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-06)[0m
├─ PT OUTPUT[0:5]: [3.606376, 3.589005, 3.495424, 3.639874, 3.571229]
├─ TT OUTPUT[0:5]: [3.606376, 3.589005, 3.495424, 3.639874, 3.571229]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mL1BboxCost[0m
├─ EDGE CASE: [93msingle_pair[0m (Single query + single GT — 1×1 cost matrix)
├─ INPUT: pred[1,4] tgt[1,4]
├─ SHAPE: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-06)[0m
├─ PT OUTPUT[0:5]: [0.940239]
├─ TT OUTPUT[0:5]: [0.940239]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mL1BboxCost[0m
├─ EDGE CASE: [93mmany_boxes[0m (Large query/GT count — scalability)
├─ INPUT: pred[50,4] tgt[20,4]
├─ SHAPE: PT=[50,20] | TT=[50,20] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-06)[0m
├─ PT OUTPUT[0:5]: [0.606226, 0.200061, 0.915898, 1.238966, 1.071800]
├─ TT OUTPUT[0:5]: [0.606226, 0.200061, 0.915898, 1.238966, 1.071800]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mL1BboxCost[0m
├─ EDGE CASE: [93mzero_size_boxes[0m (Degenerate zero-area boxes)
├─ INPUT: pred[5,4] tgt[5,4]
├─ SHAPE: PT=[5,5] | TT=[5,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-06)[0m
├─ PT OUTPUT[0:5]: [2.787025, 1.926322, 2.084340, 1.201447, 2.052060]
├─ TT OUTPUT[0:5]: [2.787025, 1.926322, 2.084340, 1.201447, 2.052060]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_giou_cost 
MODULE: [1mGIoUCost[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical inputs — baseline correctness)
├─ INPUT: pred[10,4] tgt[5,4]
├─ SHAPE: PT=[10,5] | TT=[10,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.279498, -0.153933, 0.295379, 0.032200, 0.471136]
├─ TT OUTPUT[0:5]: [0.279498, -0.153933, 0.295379, 0.032200, 0.471136]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mGIoUCost[0m
├─ EDGE CASE: [93midentical_boxes[0m (Identical pred/tgt boxes — zero distance)
├─ INPUT: pred[5,4] tgt[5,4]
├─ SHAPE: PT=[5,5] | TT=[5,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-1.000000, -1.000000, -1.000000, -1.000000, -1.000000]
├─ TT OUTPUT[0:5]: [-1.000000, -1.000000, -1.000000, -1.000000, -1.000000]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mGIoUCost[0m
├─ EDGE CASE: [93mno_overlap[0m (Non-overlapping boxes — GIoU < 0)
├─ INPUT: pred[5,4] tgt[5,4]
├─ SHAPE: PT=[5,5] | TT=[5,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.990912, 0.991040, 0.990843, 0.990727, 0.991389]
├─ TT OUTPUT[0:5]: [0.990912, 0.991040, 0.990843, 0.990727, 0.991389]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mGIoUCost[0m
├─ EDGE CASE: [93menclosed[0m (One box fully inside another — containment)
├─ INPUT: pred[5,4] tgt[5,4]
├─ SHAPE: PT=[5,5] | TT=[5,5] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [-0.015625, -0.015625, -0.015625, -0.015625, -0.015625]
├─ TT OUTPUT[0:5]: [-0.015625, -0.015625, -0.015625, -0.015625, -0.015625]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mGIoUCost[0m
├─ EDGE CASE: [93msingle_pair[0m (Single query + single GT — 1×1 cost matrix)
├─ INPUT: pred[1,4] tgt[1,4]
├─ SHAPE: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.506308]
├─ TT OUTPUT[0:5]: [0.506308]
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mGIoUCost[0m
├─ EDGE CASE: [93mmany_boxes[0m (Large query/GT count — scalability)
├─ INPUT: pred[30,4] tgt[15,4]
├─ SHAPE: PT=[30,15] | TT=[30,15] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0.0001, atol=1e-05)[0m
├─ PT OUTPUT[0:5]: [0.389696, 0.319005, 0.565900, -0.232988, 0.417704]
├─ TT OUTPUT[0:5]: [0.389696, 0.319005, 0.565900, -0.232988, 0.417704]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_matcher_shapes 
MODULE: [1mMatcherShape[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical inputs — baseline correctness)
├─ INPUT: bs=2 nq=10 nc=91 gt=[3, 5]
├─ SHAPE: img0:PT=3,TT=3 | img1:PT=5,TT=5 → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherShape[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical inputs — baseline correctness)
├─ INPUT: bs=2 nq=10 nc=91 gt=[4, 3]
├─ SHAPE: img0:PT=4,TT=4 | img1:PT=3,TT=3 → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherShape[0m
├─ EDGE CASE: [93mmany_gt[0m (Many ground-truth objects per image)
├─ INPUT: bs=2 nq=10 nc=91 gt=[8, 9]
├─ SHAPE: img0:PT=8,TT=8 | img1:PT=9,TT=9 → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherShape[0m
├─ EDGE CASE: [93mfew_gt[0m (Very few GT objects (1 per image))
├─ INPUT: bs=2 nq=10 nc=91 gt=[1, 1]
├─ SHAPE: img0:PT=1,TT=1 | img1:PT=1,TT=1 → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherShape[0m
├─ EDGE CASE: [93mequal_qgt[0m (num_queries == num_gt — square cost matrix)
├─ INPUT: bs=2 nq=5 nc=91 gt=[5, 5]
├─ SHAPE: img0:PT=5,TT=5 | img1:PT=5,TT=5 → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherShape[0m
├─ EDGE CASE: [93munbalanced_gt[0m (Different GT counts per image in batch)
├─ INPUT: bs=2 nq=10 nc=91 gt=[1, 8]
├─ SHAPE: img0:PT=1,TT=1 | img1:PT=8,TT=8 → [92m✓ MATCH[0m
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_matcher_e2e 
MODULE: [1mMatcherE2E[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical inputs — baseline correctness)
├─ INPUT: bs=2 nq=10 nc=91 gt=[3, 5] w=[2.0,5.0,2.0]
├─ SHAPE: img0:✓(n=3) | img1:✓(n=5) → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0, atol=0)[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherE2E[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical inputs — baseline correctness)
├─ INPUT: bs=2 nq=10 nc=91 gt=[4, 3] w=[2.0,5.0,2.0]
├─ SHAPE: img0:✓(n=4) | img1:✓(n=3) → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0, atol=0)[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherE2E[0m
├─ EDGE CASE: [93mmany_gt[0m (Many ground-truth objects per image)
├─ INPUT: bs=2 nq=10 nc=91 gt=[8, 9] w=[2.0,5.0,2.0]
├─ SHAPE: img0:✓(n=8) | img1:✓(n=9) → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0, atol=0)[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherE2E[0m
├─ EDGE CASE: [93mfew_gt[0m (Very few GT objects (1 per image))
├─ INPUT: bs=2 nq=10 nc=91 gt=[1, 1] w=[2.0,5.0,2.0]
├─ SHAPE: img0:✓(n=1) | img1:✓(n=1) → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0, atol=0)[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherE2E[0m
├─ EDGE CASE: [93mdifferent_wts[0m (Non-default cost weights)
├─ INPUT: bs=2 nq=10 nc=91 gt=[3, 5] w=[5.0,0.0,0.0]
├─ SHAPE: img0:✓(n=3) | img1:✓(n=5) → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0, atol=0)[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherE2E[0m
├─ EDGE CASE: [93mdifferent_wts[0m (Non-default cost weights)
├─ INPUT: bs=2 nq=10 nc=91 gt=[3, 5] w=[0.0,5.0,0.0]
├─ SHAPE: img0:✓(n=3) | img1:✓(n=5) → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0, atol=0)[0m
└─ RESULT: [92m✓ PASS[0m

MODULE: [1mMatcherE2E[0m
├─ EDGE CASE: [93mdifferent_wts[0m (Non-default cost weights)
├─ INPUT: bs=2 nq=10 nc=91 gt=[3, 5] w=[0.0,0.0,5.0]
├─ SHAPE: img0:✓(n=3) | img1:✓(n=5) → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=0, atol=0)[0m
└─ RESULT: [92m✓ PASS[0m
PASSED

============================== slowest durations ===============================
0.02s call     workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_matcher_e2e
0.02s call     workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_matcher_shapes
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_focal_loss_cost
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_l1_bbox_cost
0.01s call     workloads/Deformable_DETR/reference/unit_tests/test_matcher_unit.py::test_giou_cost

(10 durations < 0.005s hidden.  Use -vv to show these durations.)
============================== 5 passed in 0.10s ===============================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
MODULE                      SHAPE       NUMERICAL   TOTAL
FocalLossCost               6/6         6/6         [92m✓ PASS[0m
L1BboxCost                  6/6         6/6         [92m✓ PASS[0m
GIoUCost                    6/6         6/6         [92m✓ PASS[0m
MatcherShape                6/6         N/A         [92m✓ PASS[0m
MatcherE2E                  7/7         7/7         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
TOTAL                       31/31          25/25       [92m✓ PASS[0m
═════════════════════════════════════════════════════════════════
