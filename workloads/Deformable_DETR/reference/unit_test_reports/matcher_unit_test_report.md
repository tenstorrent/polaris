# HungarianMatcher Unit Test Report
**PyTorch vs TTSim Comparison** | **31/31 passed** | PASS
Generated: 2026-02-20 14:49:01 | Exit Code: 0

---

## Summary

| Module | Passed | Total | Status |
|--------|--------|-------|--------|
| FocalLossCost | 6 | 6 | PASS |
| L1BboxCost | 6 | 6 | PASS |
| GIoUCost | 6 | 6 | PASS |
| MatcherShape | 6 | 6 | PASS |
| MatcherE2E | 7 | 7 | PASS |

**Total: 31/31 tests passed**

---

## FocalLossCost (6/6 PASS)
*Focal loss cost component — α(1−p)^γ · (−log p) classification cost*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Standard 10q × 91c × 5gt | standard | `logits[10,91] tgt_ids[5]` | `[10, 5]` | 1.19e-07 | 1.77e-08 | ✅ PASS |
| 1 | Extreme logits ±10 | extreme_logits | `logits[10,91] tgt_ids[5]` | `[10, 5]` | 9.54e-07 | 8.64e-08 | ✅ PASS |
| 2 | Uniform probs (logit=0) | uniform_probs | `logits[10,91] tgt_ids[5]` | `[10, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Single target | single_target | `logits[10,91] tgt_ids[1]` | `[10, 1]` | 8.94e-08 | 1.97e-08 | ✅ PASS |
| 4 | Many classes (200) | many_classes | `logits[10,200] tgt_ids[5]` | `[10, 5]` | 1.04e-07 | 1.58e-08 | ✅ PASS |
| 5 | Sparse logits (mostly zero) | sparse_logits | `logits[10,91] tgt_ids[5]` | `[10, 5]` | 7.45e-08 | 3.58e-09 | ✅ PASS |

---

## L1BboxCost (6/6 PASS)
*L1 pairwise bounding box distance cost — torch.cdist(p=1)*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Standard 10q × 5gt | standard | `pred[10,4] tgt[5,4]` | `[10, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Identical boxes | identical_boxes | `pred[5,4] tgt[5,4]` | `[5, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Distant boxes | distant_boxes | `pred[5,4] tgt[5,4]` | `[5, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Single pair (1q × 1gt) | single_pair | `pred[1,4] tgt[1,4]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Many boxes (50q × 20gt) | many_boxes | `pred[50,4] tgt[20,4]` | `[50, 20]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Zero size boxes (w=0) | zero_size_boxes | `pred[5,4] tgt[5,4]` | `[5, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## GIoUCost (6/6 PASS)
*Generalized IoU cost component — −GIoU(xyxy(pred), xyxy(tgt))*

| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |
|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|
| 0 | Standard 10q × 5gt | standard | `pred[10,4] tgt[5,4]` | `[10, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 1 | Identical boxes | identical_boxes | `pred[5,4] tgt[5,4]` | `[5, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 2 | Non-overlapping boxes | no_overlap | `pred[5,4] tgt[5,4]` | `[5, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 3 | Enclosed boxes | enclosed | `pred[5,4] tgt[5,4]` | `[5, 5]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 4 | Single pair (1q × 1gt) | single_pair | `pred[1,4] tgt[1,4]` | `[1, 1]` | 0.00e+00 | 0.00e+00 | ✅ PASS |
| 5 | Many boxes (30q × 15gt) | many_boxes | `pred[30,4] tgt[15,4]` | `[30, 15]` | 0.00e+00 | 0.00e+00 | ✅ PASS |

---

## MatcherShape (6/6 PASS)
*HungarianMatcher output structure — list of (pred_idx, gt_idx) tuples*

| # | Test Case | Edge Case | Input | Match Counts | Result |
|:--|:----------|:----------|:------|:-------------|:-------|
| 0 | Standard bs=2 nq=10 | standard | `bs=2 nq=10 nc=91 gt=[3, 5]` | img0:PT=3,TT=3 | img1:PT=5,TT=5 | ✅ PASS |
| 1 | Small batch bs=2 | standard | `bs=2 nq=10 nc=91 gt=[4, 3]` | img0:PT=4,TT=4 | img1:PT=3,TT=3 | ✅ PASS |
| 2 | Many GT per image | many_gt | `bs=2 nq=10 nc=91 gt=[8, 9]` | img0:PT=8,TT=8 | img1:PT=9,TT=9 | ✅ PASS |
| 3 | Single GT per image | few_gt | `bs=2 nq=10 nc=91 gt=[1, 1]` | img0:PT=1,TT=1 | img1:PT=1,TT=1 | ✅ PASS |
| 4 | Equal queries & GT | equal_qgt | `bs=2 nq=5 nc=91 gt=[5, 5]` | img0:PT=5,TT=5 | img1:PT=5,TT=5 | ✅ PASS |
| 5 | Unbalanced GT counts | unbalanced_gt | `bs=2 nq=10 nc=91 gt=[1, 8]` | img0:PT=1,TT=1 | img1:PT=8,TT=8 | ✅ PASS |

---

## MatcherE2E (7/7 PASS)
*Full HungarianMatcher end-to-end — exact index assignment validation*

| # | Test Case | Edge Case | Input | Index Match | Result |
|:--|:----------|:----------|:------|:------------|:-------|
| 0 | Standard bs=2 nq=10 | standard | `bs=2 nq=10 nc=91 gt=[3, 5] w=[2.0,5.0,2.0]` | img0:✓(n=3) | img1:✓(n=5) | ✅ PASS |
| 1 | Small batch bs=2 | standard | `bs=2 nq=10 nc=91 gt=[4, 3] w=[2.0,5.0,2.0]` | img0:✓(n=4) | img1:✓(n=3) | ✅ PASS |
| 2 | Many GT (8,9) | many_gt | `bs=2 nq=10 nc=91 gt=[8, 9] w=[2.0,5.0,2.0]` | img0:✓(n=8) | img1:✓(n=9) | ✅ PASS |
| 3 | Few GT (1 each) | few_gt | `bs=2 nq=10 nc=91 gt=[1, 1] w=[2.0,5.0,2.0]` | img0:✓(n=1) | img1:✓(n=1) | ✅ PASS |
| 4 | Class-only weights | different_wts | `bs=2 nq=10 nc=91 gt=[3, 5] w=[5.0,0.0,0.0]` | img0:✓(n=3) | img1:✓(n=5) | ✅ PASS |
| 5 | Bbox-only weights | different_wts | `bs=2 nq=10 nc=91 gt=[3, 5] w=[0.0,5.0,0.0]` | img0:✓(n=3) | img1:✓(n=5) | ✅ PASS |
| 6 | GIoU-only weights | different_wts | `bs=2 nq=10 nc=91 gt=[3, 5] w=[0.0,0.0,5.0]` | img0:✓(n=3) | img1:✓(n=5) | ✅ PASS |

---

## Configuration
- Focal Loss Tolerance: rtol=0.0001, atol=1e-05
- L1 Bbox Tolerance: rtol=1e-05, atol=1e-06
- GIoU Tolerance: rtol=0.0001, atol=1e-05
- End-to-End: exact index match (np.array_equal)
- Random Seed: 42
