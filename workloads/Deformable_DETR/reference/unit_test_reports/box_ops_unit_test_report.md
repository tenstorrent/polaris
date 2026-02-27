# Box Operations Unit Test Report
**PyTorch vs TTSim Comparison** | **37/37 passed** | PASS
Generated: 2026-02-20 14:48:54 | Exit Code: 0

---

## Summary

| Function | Passed | Total | Status |
|----------|--------|-------|--------|
| box_cxcywh_to_xyxy | 6 | 6 | PASS |
| box_xyxy_to_cxcywh | 6 | 6 | PASS |
| box_area | 8 | 8 | PASS |
| box_iou | 6 | 6 | PASS |
| generalized_box_iou | 6 | 6 | PASS |
| masks_to_boxes | 5 | 5 | PASS |

**Total: 37/37 tests passed**

---

## box_cxcywh_to_xyxy (6/6 PASS)
*Convert boxes from (cx,cy,w,h) to (x0,y0,x1,y1)*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Standard 3 boxes | standard | `[3, 4]` | `[3, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Single box | single | `[1, 4]` | `[1, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Many boxes (20) | many | `[20, 4]` | `[20, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Zero-width box (w=0) | zero_size | `[2, 4]` | `[2, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Large coordinates (~1e4) | large_coords | `[2, 4]` | `[2, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Small coordinates (~1e-6) | small_coords | `[2, 4]` | `[2, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## box_xyxy_to_cxcywh (6/6 PASS)
*Convert boxes from (x0,y0,x1,y1) to (cx,cy,w,h)*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Standard 3 boxes | standard | `[3, 4]` | `[3, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Single box | single | `[1, 4]` | `[1, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Many boxes (20) | many | `[20, 4]` | `[20, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Zero-area point box | zero_size | `[1, 4]` | `[1, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Large coordinates | large_coords | `[1, 4]` | `[1, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Small coordinates | small_coords | `[1, 4]` | `[1, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## box_area (8/8 PASS)
*Compute area of boxes in (x0,y0,x1,y1) format: area=(x1-x0)*(y1-y0)*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Unit square | standard | `[1, 4]` | `[1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Multiple boxes | standard | `[4, 4]` | `[4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Single tiny box | single | `[1, 4]` | `[1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Zero-area box (point) | zero_size | `[1, 4]` | `[1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Zero-width box | zero_size | `[1, 4]` | `[1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Large coordinate boxes | large_coords | `[2, 4]` | `[2]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 6 | Small coordinate boxes | small_coords | `[1, 4]` | `[1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 7 | Many boxes (50) | many | `[50, 4]` | `[50]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## box_iou (6/6 PASS)
*Pairwise Intersection-over-Union and union area*

| # | Test Case | Edge Case | Expected | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:---------|:-------|:---------|:----------|:-------|
| 0 | Standard overlap | standard | `[2,3]` | `[2, 3]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Identical boxes (IoU=1) | identical | `[2,2]` | `[2, 2]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | No overlap (disjoint) | no_overlap | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Single box pair | single | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Enclosed box | enclosed | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Large coordinate boxes | large_coords | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## generalized_box_iou (6/6 PASS)
*Generalized IoU: GIoU = IoU - (enclosing - union) / enclosing*

| # | Test Case | Edge Case | Expected | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:---------|:-------|:---------|:----------|:-------|
| 0 | Standard overlap | standard | `[2,3]` | `[2, 3]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Identical boxes (GIoU=1) | identical | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | No overlap — negative GIoU | no_overlap | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Enclosed box | enclosed | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Single box pair | single | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Large coordinate boxes | large_coords | `[1,1]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## masks_to_boxes (5/5 PASS)
*Extract xyxy bounding boxes from binary masks*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | 3 rectangle masks (10×10) | standard | `[3, 10, 10]` | `[3, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Full mask 8×8 | full_mask | `[1, 8, 8]` | `[1, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Single pixel [7,7] | single_pixel | `[1, 16, 16]` | `[1, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Corner pixels [0,0] & [9,9] | single_pixel | `[2, 10, 10]` | `[2, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Large 64×64 masks | large_coords | `[2, 64, 64]` | `[2, 4]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## Configuration
- Tolerance: rtol=1e-05, atol=1e-07
- Random Seed: 42
