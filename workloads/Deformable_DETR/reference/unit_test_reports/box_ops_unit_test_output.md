
═════════════════════════════════════════════════════════════════
BOX OPERATIONS UNIT TEST SUITE - PyTorch vs TTSim
═════════════════════════════════════════════════════════════════

============================= test session starts ==============================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /usr/bin/python3.13
cachedir: .pytest_cache
rootdir: /home/aughag/Videos/TensTorrent/polaris
configfile: pyproject.toml
collecting ... collected 6 items

workloads/Deformable_DETR/reference/unit_tests/test_box_ops_unit.py::test_box_cxcywh_to_xyxy 
FUNCTION: [1mbox_cxcywh_to_xyxy[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical boxes — baseline correctness)
├─ INPUT: [3,4]
├─ SHAPE: PT=[3,4] | TT=[3,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.300000, 0.200000, 0.700000, 0.800000, 0.100000]
├─ TT OUTPUT[0:5]: [0.300000, 0.200000, 0.700000, 0.800000, 0.100000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_cxcywh_to_xyxy[0m
├─ EDGE CASE: [93msingle[0m (Single box / minimal valid input)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1,4] | TT=[1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 1.000000, 1.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 1.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_cxcywh_to_xyxy[0m
├─ EDGE CASE: [93mmany[0m (Large batch of boxes — scalability check)
├─ INPUT: [20,4]
├─ SHAPE: PT=[20,4] | TT=[20,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.008543, 0.651385, 0.740537, 1.250044, 0.126977]
├─ TT OUTPUT[0:5]: [0.008543, 0.651385, 0.740537, 1.250044, 0.126977]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_cxcywh_to_xyxy[0m
├─ EDGE CASE: [93mzero_size[0m (Zero-width or zero-height — degenerate geometry)
├─ INPUT: [2,4]
├─ SHAPE: PT=[2,4] | TT=[2,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.500000, 0.300000, 0.500000, 0.700000, 0.200000]
├─ TT OUTPUT[0:5]: [0.500000, 0.300000, 0.500000, 0.700000, 0.200000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_cxcywh_to_xyxy[0m
├─ EDGE CASE: [93mlarge_coords[0m (Large coordinates (~1e4) — overflow / precision)
├─ INPUT: [2,4]
├─ SHAPE: PT=[2,4] | TT=[2,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [4000.000000, 3500.000000, 6000.000000, 6500.000000, 9949.000000]
├─ TT OUTPUT[0:5]: [4000.000000, 3500.000000, 6000.000000, 6500.000000, 9949.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_cxcywh_to_xyxy[0m
├─ EDGE CASE: [93msmall_coords[0m (Very small coordinates (~1e-6) — underflow / precision)
├─ INPUT: [2,4]
├─ SHAPE: PT=[2,4] | TT=[2,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.499999]
├─ TT OUTPUT[0:5]: [0.000001, 0.000001, 0.000001, 0.000001, 0.499999]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_box_ops_unit.py::test_box_xyxy_to_cxcywh 
FUNCTION: [1mbox_xyxy_to_cxcywh[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical boxes — baseline correctness)
├─ INPUT: [3,4]
├─ SHAPE: PT=[3,4] | TT=[3,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.300000, 0.450000, 0.400000, 0.500000, 0.550000]
├─ TT OUTPUT[0:5]: [0.300000, 0.450000, 0.400000, 0.500000, 0.550000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_xyxy_to_cxcywh[0m
├─ EDGE CASE: [93msingle[0m (Single box / minimal valid input)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1,4] | TT=[1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.500000, 0.500000, 1.000000, 1.000000]
├─ TT OUTPUT[0:5]: [0.500000, 0.500000, 1.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_xyxy_to_cxcywh[0m
├─ EDGE CASE: [93mmany[0m (Large batch of boxes — scalability check)
├─ INPUT: [20,4]
├─ SHAPE: PT=[20,4] | TT=[20,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.486599, 0.841354, 0.224118, -0.218720, 0.107039]
├─ TT OUTPUT[0:5]: [0.486599, 0.841354, 0.224118, -0.218720, 0.107039]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_xyxy_to_cxcywh[0m
├─ EDGE CASE: [93mzero_size[0m (Zero-width or zero-height — degenerate geometry)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1,4] | TT=[1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.500000, 0.500000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.500000, 0.500000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_xyxy_to_cxcywh[0m
├─ EDGE CASE: [93mlarge_coords[0m (Large coordinates (~1e4) — overflow / precision)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1,4] | TT=[1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [2550.000000, 4100.000000, 4900.000000, 7800.000000]
├─ TT OUTPUT[0:5]: [2550.000000, 4100.000000, 4900.000000, 7800.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_xyxy_to_cxcywh[0m
├─ EDGE CASE: [93msmall_coords[0m (Very small coordinates (~1e-6) — underflow / precision)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1,4] | TT=[1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_box_ops_unit.py::test_box_area 
FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical boxes — baseline correctness)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1] | TT=[1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [1.000000]
├─ TT OUTPUT[0:5]: [1.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical boxes — baseline correctness)
├─ INPUT: [4,4]
├─ SHAPE: PT=[4] | TT=[4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [1.000000, 0.200000, 0.300000, 0.250000]
├─ TT OUTPUT[0:5]: [1.000000, 0.200000, 0.300000, 0.250000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93msingle[0m (Single box / minimal valid input)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1] | TT=[1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.040000]
├─ TT OUTPUT[0:5]: [0.040000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93mzero_size[0m (Zero-width or zero-height — degenerate geometry)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1] | TT=[1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000]
├─ TT OUTPUT[0:5]: [0.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93mzero_size[0m (Zero-width or zero-height — degenerate geometry)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1] | TT=[1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000]
├─ TT OUTPUT[0:5]: [0.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93mlarge_coords[0m (Large coordinates (~1e4) — overflow / precision)
├─ INPUT: [2,4]
├─ SHAPE: PT=[2] | TT=[2] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [100000000.000000, 24990000.000000]
├─ TT OUTPUT[0:5]: [100000000.000000, 24990000.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93msmall_coords[0m (Very small coordinates (~1e-6) — underflow / precision)
├─ INPUT: [1,4]
├─ SHAPE: PT=[1] | TT=[1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000]
├─ TT OUTPUT[0:5]: [0.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_area[0m
├─ EDGE CASE: [93mmany[0m (Large batch of boxes — scalability check)
├─ INPUT: [50,4]
├─ SHAPE: PT=[50] | TT=[50] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.005000, 0.039968, 0.056562, 0.077516, 0.004403]
├─ TT OUTPUT[0:5]: [0.005000, 0.039968, 0.056562, 0.077516, 0.004403]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_box_ops_unit.py::test_box_iou 
FUNCTION: [1mbox_iou[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical boxes — baseline correctness)
├─ INPUT: boxes1=[2,4], boxes2=[3,4]
├─ SHAPE: IoU: PT=[2,3] | TT=[2,3] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [1.000000, 0.020408, 0.000000, 0.219512, 0.219512]
├─ TT OUTPUT[0:5]: [1.000000, 0.020408, 0.000000, 0.219512, 0.219512]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_iou[0m
├─ EDGE CASE: [93midentical[0m (Identical boxes — IoU = 1.0 special case)
├─ INPUT: boxes1=[2,4], boxes2=[2,4]
├─ SHAPE: IoU: PT=[2,2] | TT=[2,2] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [1.000000, 0.173077, 0.173077, 1.000000]
├─ TT OUTPUT[0:5]: [1.000000, 0.173077, 0.173077, 1.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_iou[0m
├─ EDGE CASE: [93mno_overlap[0m (Disjoint boxes — zero intersection)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: IoU: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000]
├─ TT OUTPUT[0:5]: [0.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_iou[0m
├─ EDGE CASE: [93msingle[0m (Single box / minimal valid input)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: IoU: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.142857]
├─ TT OUTPUT[0:5]: [0.142857]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_iou[0m
├─ EDGE CASE: [93menclosed[0m (One box fully inside another — containment)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: IoU: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.360000]
├─ TT OUTPUT[0:5]: [0.360000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mbox_iou[0m
├─ EDGE CASE: [93mlarge_coords[0m (Large coordinates (~1e4) — overflow / precision)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: IoU: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.083333]
├─ TT OUTPUT[0:5]: [0.083333]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_box_ops_unit.py::test_generalized_box_iou 
FUNCTION: [1mgeneralized_box_iou[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical boxes — baseline correctness)
├─ INPUT: boxes1=[2,4], boxes2=[3,4]
├─ SHAPE: PT=[2,3] | TT=[2,3] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [1.000000, -0.079365, -0.710000, -0.590000, -0.253047]
├─ TT OUTPUT[0:5]: [1.000000, -0.079365, -0.710000, -0.590000, -0.253047]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mgeneralized_box_iou[0m
├─ EDGE CASE: [93midentical[0m (Identical boxes — IoU = 1.0 special case)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [1.000000]
├─ TT OUTPUT[0:5]: [1.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mgeneralized_box_iou[0m
├─ EDGE CASE: [93mno_overlap[0m (Disjoint boxes — zero intersection)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [-0.980000]
├─ TT OUTPUT[0:5]: [-0.980000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mgeneralized_box_iou[0m
├─ EDGE CASE: [93menclosed[0m (One box fully inside another — containment)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.160000]
├─ TT OUTPUT[0:5]: [0.160000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mgeneralized_box_iou[0m
├─ EDGE CASE: [93msingle[0m (Single box / minimal valid input)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [-0.136790]
├─ TT OUTPUT[0:5]: [-0.136790]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mgeneralized_box_iou[0m
├─ EDGE CASE: [93mlarge_coords[0m (Large coordinates (~1e4) — overflow / precision)
├─ INPUT: boxes1=[1,4], boxes2=[1,4]
├─ SHAPE: PT=[1,1] | TT=[1,1] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [-0.166667]
├─ TT OUTPUT[0:5]: [-0.166667]
└─ RESULT: [92m✓ PASS[0m
PASSED
workloads/Deformable_DETR/reference/unit_tests/test_box_ops_unit.py::test_masks_to_boxes 
FUNCTION: [1mmasks_to_boxes[0m
├─ EDGE CASE: [93mstandard[0m (Well-formed typical boxes — baseline correctness)
├─ INPUT: [3,10,10]
├─ SHAPE: PT=[3,4] | TT=[3,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [3.000000, 2.000000, 7.000000, 4.000000, 1.000000]
├─ TT OUTPUT[0:5]: [3.000000, 2.000000, 7.000000, 4.000000, 1.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mmasks_to_boxes[0m
├─ EDGE CASE: [93mfull_mask[0m (Entire image True — maximum bounding box)
├─ INPUT: [1,8,8]
├─ SHAPE: PT=[1,4] | TT=[1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 7.000000, 7.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 7.000000, 7.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mmasks_to_boxes[0m
├─ EDGE CASE: [93msingle_pixel[0m (Single True pixel — point bounding box)
├─ INPUT: [1,16,16]
├─ SHAPE: PT=[1,4] | TT=[1,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [7.000000, 7.000000, 7.000000, 7.000000]
├─ TT OUTPUT[0:5]: [7.000000, 7.000000, 7.000000, 7.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mmasks_to_boxes[0m
├─ EDGE CASE: [93msingle_pixel[0m (Single True pixel — point bounding box)
├─ INPUT: [2,10,10]
├─ SHAPE: PT=[2,4] | TT=[2,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 9.000000]
├─ TT OUTPUT[0:5]: [0.000000, 0.000000, 0.000000, 0.000000, 9.000000]
└─ RESULT: [92m✓ PASS[0m

FUNCTION: [1mmasks_to_boxes[0m
├─ EDGE CASE: [93mlarge_coords[0m (Large coordinates (~1e4) — overflow / precision)
├─ INPUT: [2,64,64]
├─ SHAPE: PT=[2,4] | TT=[2,4] → [92m✓ MATCH[0m
├─ NUMERICAL: max_diff=0.00e+00, mean_diff=0.00e+00 → [92m✓ PASS (tol: rtol=1e-05, atol=1e-07)[0m
├─ PT OUTPUT[0:5]: [5.000000, 10.000000, 54.000000, 49.000000, 0.000000]
├─ TT OUTPUT[0:5]: [5.000000, 10.000000, 54.000000, 49.000000, 0.000000]
└─ RESULT: [92m✓ PASS[0m
PASSED

=============================== warnings summary ===============================
workloads/Deformable_DETR/reference/unit_tests/test_box_ops_unit.py::test_masks_to_boxes
  /home/aughag/.local/lib/python3.13/site-packages/torch/functional.py:505: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at /pytorch/aten/src/ATen/native/TensorShape.cpp:4381.)
    return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================== slowest durations ===============================

(18 durations < 0.005s hidden.  Use -vv to show these durations.)
========================= 6 passed, 1 warning in 0.05s =========================

═════════════════════════════════════════════════════════════════
SUMMARY
═════════════════════════════════════════════════════════════════
FUNCTION                    SHAPE       NUMERICAL   TOTAL
box_cxcywh_to_xyxy          6/6         6/6         [92m✓ PASS[0m
box_xyxy_to_cxcywh          6/6         6/6         [92m✓ PASS[0m
box_area                    8/8         8/8         [92m✓ PASS[0m
box_iou                     6/6         6/6         [92m✓ PASS[0m
generalized_box_iou         6/6         6/6         [92m✓ PASS[0m
masks_to_boxes              5/5         5/5         [92m✓ PASS[0m
─────────────────────────────────────────────────────────────────
TOTAL                       37/37          37/37       [92m✓ PASS[0m
═════════════════════════════════════════════════════════════════
