# Comparison Results (bevformer) — 2026-04-27 09:29:45

## test_base_mapper.py  —  PASS

### stdout

```

================================================================================
BaseMapper Comparison: PyTorch vs ttsim
================================================================================

================================================================================
TEST 1: Property Methods
================================================================================
PyTorch with_neck: True
ttsim with_neck:   True
Properties match: True
[PASS] PASS: Properties match!

================================================================================
TEST 2: extract_feat()
================================================================================
  Synced conv.weight -> conv_op.param, shape: (64, 3, 3, 3)
  Synced neck.weight -> neck_op.param, shape: (64, 64, 1, 1)
  Synced head.weight -> head_op.param, shape: (10, 64, 1, 1)
PyTorch feat shape: torch.Size([2, 64, 32, 32])
PyTorch feat mean: -0.004951
ttsim feat shape: (2, 64, 32, 32)
ttsim feat mean: -0.004951
Max difference: 0.00000024
[PASS] PASS: extract_feat matches!

================================================================================
TEST 3: forward_train()
================================================================================
  Synced conv.weight -> conv_op.param, shape: (64, 3, 3, 3)
  Synced neck.weight -> neck_op.param, shape: (64, 64, 1, 1)
  Synced head.weight -> head_op.param, shape: (10, 64, 1, 1)
PyTorch loss: 1.014594
PyTorch log_vars: {'loss': 1.0145939588546753}
ttsim loss: 1.014594
ttsim log_vars: {'loss': 1.0145938396453857}
Difference: 0.00000012
[PASS] PASS: forward_train matches!

================================================================================
TEST 4: train_step()
================================================================================
  Synced conv.weight -> conv_op.param, shape: (64, 3, 3, 3)
  Synced neck.weight -> neck_op.param, shape: (64, 64, 1, 1)
  Synced head.weight -> head_op.param, shape: (10, 64, 1, 1)
PyTorch train_step loss: 1.014483
ttsim train_step loss: 1.014483
Difference: 0.00000000
[PASS] PASS: train_step matches!

================================================================================
SUMMARY
================================================================================
[PASS] PASS: Properties
[PASS] PASS: extract_feat
[PASS] PASS: forward_train
[PASS] PASS: train_step

Total: 4/4 tests passed

[PASS] All tests passed! PyTorch and ttsim implementations match perfectly!
```

---

## test_maptracker.py  —  PASS

### stdout

```

================================================================================
MapTracker Layer-by-Layer Validation - TTSim Complete Test Suite
================================================================================

Running: Layer-by-Layer Validation with TTSim

MapTracker Architecture Layer-by-Layer Validation
--------------------------------------------------------------------------------

  ===========================================================================
  MAPTRACKER COMPLETE ARCHITECTURE VALIDATION
  ===========================================================================
  ResNet-50 backbone + FPN + faithful downstream stages
  Data flows through each layer with proper shapes and operations
  All computations use TTSim compute functions for validation
  ===========================================================================


  ===========================================================================
  STAGE 0: IMAGE BACKBONE + FPN (ResNet-50 Bottleneck)
  ===========================================================================
  ResNet-50 backbone with Bottleneck blocks (1 block per stage for speed)
  Extracts multi-scale features from 2 camera views

  Layer 1: Multi-Camera Image Input
  ---------------------------------------------------------------------------
    Input shape: torch.Size([1, 2, 3, 28, 50]) [B, num_cams, C, H, W]
    Number of cameras: 2
    Image size: 28x50

  Layer 2: Backbone Stem (Conv7x7/2 + BN + ReLU + MaxPool3x3/2)
  ---------------------------------------------------------------------------
    Conv7x7/2: [2, 3, 28, 50] -> [2, 8, 14, 25]
    BN + ReLU + MaxPool3x3/2 -> torch.Size([2, 8, 7, 13])
  PASS Stem: Match! Max diff: 1.79e-07, Mean diff: 1.10e-08

  Layer 3: ResNet-50 Stages 1-4 (Bottleneck blocks)
  ---------------------------------------------------------------------------
  PASS Bottleneck: Match! Max diff: 1.79e-07, Mean diff: 1.33e-08
    Stage 1: 8->8 ch, stride=1, 1 block(s) -> torch.Size([2, 8, 7, 13])
  PASS Bottleneck: Match! Max diff: 5.96e-08, Mean diff: 4.60e-09
    Stage 2: 8->16 ch, stride=2, 1 block(s) -> torch.Size([2, 16, 4, 7])
  PASS Bottleneck: Match! Max diff: 2.98e-08, Mean diff: 3.48e-09
    Stage 3: 16->32 ch, stride=2, 1 block(s) -> torch.Size([2, 32, 2, 4])
  PASS Bottleneck: Match! Max diff: 2.98e-08, Mean diff: 2.19e-09
    Stage 4: 32->64 ch, stride=2, 1 block(s) -> torch.Size([2, 64, 1, 2])
    out_indices=(1,2,3): C3=torch.Size([2, 16, 4, 7]), C4=torch.Size([2, 32, 2, 4]), C5=torch.Size([2, 64, 1, 2])

  Layer 4: FPN (Lateral 1x1 + Top-Down + 3x3 Output Convs)
  ---------------------------------------------------------------------------
    Lateral C5: torch.Size([2, 64, 1, 2]) -> 1x1 conv -> torch.Size([2, 32, 1, 2])
    Lateral C4: torch.Size([2, 32, 2, 4]) -> 1x1 conv -> torch.Size([2, 32, 2, 4])
    Lateral C3: torch.Size([2, 16, 4, 7]) -> 1x1 conv -> torch.Size([2, 32, 4, 7])
  ✓ FPN Fusion: Match! Max difference: 4.66e-09
  ✓ FPN Fusion: Match! Max difference: 5.59e-09
  PASS FPN P3 output conv: Match! Max diff: 5.59e-09, Mean diff: 5.84e-10
  PASS FPN P4 output conv: Match! Max diff: 1.63e-09, Mean diff: 3.90e-10
  PASS FPN P5 output conv: Match! Max diff: 5.82e-10, Mean diff: 1.47e-10
    FPN Level 0 (P3): torch.Size([2, 32, 4, 7])
    FPN Level 1 (P4): torch.Size([2, 32, 2, 4])
    FPN Level 2 (P5): torch.Size([2, 32, 1, 2])

  Layer 5: BEV Query Init + Positional Encoding
  ---------------------------------------------------------------------------
    BEV queries: torch.Size([1, 200, 32]) [B, bev_h*bev_w, embed_dims]
    Positional encoding: torch.Size([1, 200, 32])
  PASS BEV Queries+Pos: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00

  ───────────────────────────────────────────────────────────────────────────
  ENCODER LAYER 1/2
  ───────────────────────────────────────────────────────────────────────────

  ├─ Layer 6: Temporal Self-Attention
     * Step 1: Q, K, V projections (K,V from current+prev BEV)
  PASS        Q projection: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00
     * Step 2: Temporal multi-head attention (8 heads)
     * Step 3: Output projection
  PASS        Output Projection: Match! Max diff: 3.06e-10, Mean diff: 6.05e-11
     * Step 4: Residual connection [1, 200, 32]
  PASS        Residual: Match! Max diff: 1.19e-07, Mean diff: 8.96e-11

  ├─ Layer 7: LayerNorm
  PASS      Normalized: Match! Max diff: 4.77e-07, Mean diff: 3.99e-08

  ├─ Layer 8: Spatial Cross-Attention (Deformable)
     * Step 1: 3D ref pts projected to camera views
     * Step 2: Sample from 1 FPN level x 2 cameras
     * Step 3: Output projection (32 -> 32)
  PASS        Projection: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00
     * Step 4: Residual connection
  PASS      Final Output: Match! Max diff: 4.77e-07, Mean diff: 3.99e-08

  ├─ Layer 9: LayerNorm
  PASS      Normalized: Match! Max diff: 4.77e-07, Mean diff: 5.10e-08

  ├─ Layer 10: FFN (Linear-ReLU-Linear)
     * Step 1: Linear expand (32 -> 128)
  ✓        Linear1: Match! Max difference: 4.47e-08
     * Step 2: ReLU activation
  ✓        ReLU: Match! Max difference: 4.47e-08
     * Step 3: Linear contract (128 -> 32)
  ✓        Linear2: Match! Max difference: 3.73e-09
     * Residual connection
  PASS      Final Output: Match! Max diff: 4.77e-07, Mean diff: 5.12e-08

  └─ Layer 11: LayerNorm (Final)
  PASS      Final Normalized: Match! Max diff: 4.77e-07, Mean diff: 5.64e-08

  ───────────────────────────────────────────────────────────────────────────
  ENCODER LAYER 2/2
  ───────────────────────────────────────────────────────────────────────────

  ├─ Layer 12: Temporal Self-Attention
     * Step 1: Q, K, V projections (K,V from current+prev BEV)
  PASS        Q projection: Match! Max diff: 4.10e-08, Mean diff: 5.31e-09
     * Step 2: Temporal multi-head attention (8 heads)
     * Step 3: Output projection
  PASS        Output Projection: Match! Max diff: 3.78e-10, Mean diff: 6.90e-11
     * Step 4: Residual connection [1, 200, 32]
  PASS        Residual: Match! Max diff: 7.15e-07, Mean diff: 5.65e-08

  ├─ Layer 13: LayerNorm
  PASS      Normalized: Match! Max diff: 4.77e-07, Mean diff: 6.03e-08

  ├─ Layer 14: Spatial Cross-Attention (Deformable)
     * Step 1: 3D ref pts projected to camera views
     * Step 2: Sample from 1 FPN level x 2 cameras
     * Step 3: Output projection (32 -> 32)
  PASS        Projection: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00
     * Step 4: Residual connection
  PASS      Final Output: Match! Max diff: 4.77e-07, Mean diff: 6.02e-08

  ├─ Layer 15: LayerNorm
  PASS      Normalized: Match! Max diff: 9.54e-07, Mean diff: 6.68e-08

  ├─ Layer 16: FFN (Linear-ReLU-Linear)
     * Step 1: Linear expand (32 -> 128)
  ✓        Linear1: Match! Max difference: 5.96e-08
     * Step 2: ReLU activation
  ✓        ReLU: Match! Max difference: 4.47e-08
     * Step 3: Linear contract (128 -> 32)
  ✓        Linear2: Match! Max difference: 5.59e-09
     * Residual connection
  PASS      Final Output: Match! Max diff: 9.54e-07, Mean diff: 6.70e-08

  └─ Layer 17: LayerNorm (Final)
  PASS      Final Normalized: Match! Max diff: 9.54e-07, Mean diff: 7.60e-08

  ENCODER COMPLETE: 2 layers processed
  Final BEV features: torch.Size([1, 200, 32]) [B, bev_h*bev_w, 32]

  ===========================================================================
  STAGE 1: BEV INPUT (output from Image Backbone + BEV Encoder)
  ===========================================================================

  Layer 18: BEV Feature Input (connected from Stage 0 encoder)
  ---------------------------------------------------------------------------
    Encoder output: [1, 200, 32] (from Stage 0)
    Reshape + Permute -> torch.Size([1, 32, 20, 10]) [B, C, H, W]
    BEV grid: 20 x 10 = 200 cells
  PASS BEV Input (from encoder): Match! Max diff: 9.54e-07, Mean diff: 7.60e-08

  ===========================================================================
  STAGE 2: BEV BACKBONE (BEVFormerBackbone)
  ===========================================================================

  Layer 19: BEV Embedding Lookup
  ---------------------------------------------------------------------------
    Embedding table: [200, 32]
    BEV queries: torch.Size([200, 32])
  PASS BEV Embedding: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00

  Layer 20: Reshape [bs, H*W, C] -> [bs, C, H, W]
  ---------------------------------------------------------------------------
    Step 1: Unsqueeze + Tile -> torch.Size([1, 200, 32])
  PASS Expanded queries: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00
    Step 2: Reshape to [1, 20, 10, 32] + Permute -> torch.Size([1, 32, 20, 10])
  PASS BEV spatial: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00

  Layer 21: History Warping (GridSample)
  ---------------------------------------------------------------------------
    History frames: 2, each [1, 32, 20, 10]
    Stack -> [2, 1, 32, 20, 10]
    Permute -> [1, 2, 32, 20, 10]
    Flatten -> [2, 32, 20, 10]
    GridSample -> [2, 32, 20, 10]
    Reshape -> torch.Size([1, 2, 32, 20, 10])
  PASS GridSample Warping: Match! Max diff: 2.38e-07, Mean diff: 1.72e-08

  ===========================================================================
  STAGE 3: SEGMENTATION HEAD (MapSegHead)
  ===========================================================================

  Layer 22: Conv2d(3x3, no bias) + ReLU
  ---------------------------------------------------------------------------
    Conv2d: [1, 32, 20, 10] -> [1, 32, 20, 10]
    ReLU applied
  PASS SegHead Conv_in + ReLU: Match! Max diff: 3.87e-07, Mean diff: 2.39e-08

  Layer 23: Upsample(2x) + Conv2d(3x3) + bias + ReLU
  ---------------------------------------------------------------------------
    Upsample: [1, 32, 20, 10] -> [1, 32, 40, 20]
    Conv2d(3x3) + bias + ReLU
  PASS SegHead Upsample+Conv: Match! Max diff: 5.96e-08, Mean diff: 3.73e-09

  Layer 24: Conv2d(1x1) + bias -> Segmentation Predictions
  ---------------------------------------------------------------------------
    Conv2d(1x1): [1, 32, 40, 20] -> [1, 3, 40, 20]
  PASS SegHead Predictions: Match! Max diff: 1.86e-09, Mean diff: 3.32e-10

  Layer 25: Downsample(0.5x) -> seg_feats
  ---------------------------------------------------------------------------
    Bilinear downsample: [1, 32, 40, 20] -> torch.Size([1, 32, 20, 10])
  PASS SegHead Features: Match! Max diff: 9.31e-08, Mean diff: 4.85e-09

  ===========================================================================
  STAGE 4: VECTOR MAP DETECTION HEAD (MapDetectorHead)
  ===========================================================================

  Layer 26: Input Projection Conv2d(1x1) + bias
  ---------------------------------------------------------------------------
    Conv2d(1x1): [1, 32, 20, 10] -> [1, 32, 20, 10]
  PASS Input Projection: Match! Max diff: 5.22e-08, Mean diff: 6.50e-09

  Layer 27: BEV Sine Positional Embedding + Add
  ---------------------------------------------------------------------------
    Positional encoding: (1, 32, 20, 10)
    Add to projected BEV -> torch.Size([1, 32, 20, 10])
  PASS BEV + PosEmbed: Match! Max diff: 1.19e-07, Mean diff: 6.66e-09

  Layer 28: Flatten BEV for Decoder
  ---------------------------------------------------------------------------
    Reshape [1, 32, 20, 10] -> [1, 32, 200]
    Permute -> [1, 200, 32]
    Permute -> [200, 1, 32] (sequence-first for attention)
  PASS BEV Flattened: Match! Max diff: 1.19e-07, Mean diff: 6.66e-09

  Layer 29: Query Initialization (Embedding + Ref Points)
  ---------------------------------------------------------------------------
    Query embedding: [10, 32] -> [1, 10, 32]
    Reference points Linear(32->10) + Sigmoid -> Reshape
    Reference points: torch.Size([1, 10, 5, 2])
  PASS Query Embedding: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00
  PASS Reference Points: Match! Max diff: 5.96e-08, Mean diff: 5.96e-10

  ===========================================================================
  STAGE 5: TRANSFORMER DECODER (2 layers)
  ===========================================================================
  Each layer: Self-Attn -> Norm -> BEV Cross-Attn -> Norm -> Memory Cross-Attn -> Norm -> FFN -> Norm
  Then: RegressionBranch + ClassificationBranch


  ───────────────────────────────────────────────────────────────────────────
  DECODER LAYER 1/2
  ───────────────────────────────────────────────────────────────────────────

  |-- Layer 30: Self-Attention (MultiheadAttention)
     * Q,K,V projections: Linear(32->32) × 3
     * Reshape -> 8 heads x 4 head_dim
     * MatMul(Q, K^T) / sqrt(4) -> Softmax -> MatMul(attn, V)
     * Output projection + residual
  PASS      Self-Attn Output: Match! Max diff: 4.66e-10, Mean diff: 1.46e-12

  |-- Layer 31: LayerNorm
  PASS      LayerNorm: Match! Max diff: 2.38e-07, Mean diff: 3.09e-08

  |-- Layer 32: Cross-Attention (Deformable Attention to BEV)
     * Value projection: Linear(32->32)
     * Sampling offsets: Linear(32->64)
     * Attention weights: Linear(32->32) + Softmax
     * Deformable sampling from BEV [200, 32]
     * Output projection + residual
  PASS      Deformable Attn Output: Match! Max diff: 2.38e-07, Mean diff: 3.11e-08

  |-- Layer 33: LayerNorm
  PASS      LayerNorm: Match! Max diff: 2.38e-07, Mean diff: 4.95e-08

  |-- Layer 34: Memory Cross-Attention (Frame 0: ACTIVE)
     * Pre-populated memory: 1 track(s), mem_len=10
     * Q from decoder: [1, 10, 32]
     * K = memory + relative_PE: [1, 10, 32]
     * V from memory: [1, 10, 32]
     * MHA per-track: Q,K,V proj -> 8 heads -> Softmax -> Output proj
     * Fusion: query_memory + query_bev (additive)
  PASS      MemAttn Output: Match! Max diff: 2.38e-07, Mean diff: 4.95e-08

  |-- Layer 35: LayerNorm
  PASS      LayerNorm: Match! Max diff: 4.77e-07, Mean diff: 5.37e-08

  |-- Layer 36: FFN (Linear-ReLU-Linear + Residual)
     * Expansion: 32 -> 128 (ReLU) -> 32
  PASS      FFN Output: Match! Max diff: 4.77e-07, Mean diff: 5.39e-08

  |-- Layer 37: LayerNorm (Final)
  PASS      Final LayerNorm: Match! Max diff: 4.77e-07, Mean diff: 6.24e-08

  |-- Layer 38: RegressionBranch
     * Linear(32->64) -> LN -> ReLU
     * Linear(64->64) -> LN -> ReLU
     * Linear(64->10) -> Sigmoid
     * Reference points: torch.Size([1, 10, 5, 2])
  PASS      Regression Branch: Match! Max diff: 5.96e-08, Mean diff: 3.58e-09

  |-- Layer 39: ClassificationBranch
     * Linear(32 -> 3)
  PASS      Classification Branch: Match! Max diff: 1.49e-08, Mean diff: 5.72e-09

  ───────────────────────────────────────────────────────────────────────────
  DECODER LAYER 2/2
  ───────────────────────────────────────────────────────────────────────────

  |-- Layer 40: Self-Attention (MultiheadAttention)
     * Q,K,V projections: Linear(32->32) × 3
     * Reshape -> 8 heads x 4 head_dim
     * MatMul(Q, K^T) / sqrt(4) -> Softmax -> MatMul(attn, V)
     * Output projection + residual
  PASS      Self-Attn Output: Match! Max diff: 4.77e-07, Mean diff: 6.26e-08

  |-- Layer 41: LayerNorm
  PASS      LayerNorm: Match! Max diff: 4.77e-07, Mean diff: 8.53e-08

  |-- Layer 42: Cross-Attention (Deformable Attention to BEV)
     * Value projection: Linear(32->32)
     * Sampling offsets: Linear(32->64)
     * Attention weights: Linear(32->32) + Softmax
     * Deformable sampling from BEV [200, 32]
     * Output projection + residual
  PASS      Deformable Attn Output: Match! Max diff: 4.77e-07, Mean diff: 8.40e-08

  |-- Layer 43: LayerNorm
  PASS      LayerNorm: Match! Max diff: 4.77e-07, Mean diff: 8.78e-08

  |-- Layer 44: Memory Cross-Attention (Frame 0: ACTIVE)
     * Pre-populated memory: 1 track(s), mem_len=10
     * Q from decoder: [1, 10, 32]
     * K = memory + relative_PE: [1, 10, 32]
     * V from memory: [1, 10, 32]
     * MHA per-track: Q,K,V proj -> 8 heads -> Softmax -> Output proj
     * Fusion: query_memory + query_bev (additive)
  PASS      MemAttn Output: Match! Max diff: 4.77e-07, Mean diff: 8.79e-08

  |-- Layer 45: LayerNorm
  PASS      LayerNorm: Match! Max diff: 4.77e-07, Mean diff: 8.97e-08

  |-- Layer 46: FFN (Linear-ReLU-Linear + Residual)
     * Expansion: 32 -> 128 (ReLU) -> 32
  PASS      FFN Output: Match! Max diff: 4.77e-07, Mean diff: 9.02e-08

  |-- Layer 47: LayerNorm (Final)
  PASS      Final LayerNorm: Match! Max diff: 4.77e-07, Mean diff: 9.79e-08

  |-- Layer 48: RegressionBranch
     * Linear(32->64) -> LN -> ReLU
     * Linear(64->64) -> LN -> ReLU
     * Linear(64->10) -> Sigmoid
     * Reference points: torch.Size([1, 10, 5, 2])
  PASS      Regression Branch: Match! Max diff: 5.96e-08, Mean diff: 4.77e-09

  |-- Layer 49: ClassificationBranch
     * Linear(32 -> 3)
  PASS      Classification Branch: Match! Max diff: 1.49e-08, Mean diff: 5.06e-09

  ===========================================================================
  DECODER COMPLETE: 2 layers processed
  Final query features: torch.Size([10, 1, 32])
  PASS All Cls Scores (stacked): Match! Max diff: 1.49e-08, Mean diff: 5.39e-09
  PASS All Reg Points (stacked): Match! Max diff: 5.96e-08, Mean diff: 4.17e-09

  ===========================================================================
  STAGE 6: QUERY PROPAGATION (MotionMLP)
  ===========================================================================

  Layer 50: Embedder (Sin/Cos Positional Encoding)
  ---------------------------------------------------------------------------
    Pose: [10, 7] -> [10, 147]
    10 frequency bands: sin/cos encoding
  PASS Embedder Output: Match! Max diff: 5.96e-08, Mean diff: 6.99e-09

  Layer 51: MotionMLP (Concat -> Linear -> LN -> ReLU -> Linear + Residual)
  ---------------------------------------------------------------------------
    Concat: [10, 32] + [10, 147] -> [10, 179]
    Linear(179->64) -> LN -> ReLU
    Linear(64->32) + Residual
  PASS MotionMLP Output: Match! Max diff: 1.19e-07, Mean diff: 7.49e-09

  ===========================================================================
  STAGE 7: POST-PROCESSING
  ===========================================================================

  Layer 52: Sigmoid + ArgMax -> Final Predictions
  ---------------------------------------------------------------------------
    Cls scores: (1, 10, 3) -> Sigmoid -> Max
    Scores: torch.Size([1, 10]), Labels: torch.Size([1, 10])
  PASS Scores (sigmoid+max): Match! Max diff: 0.00e+00, Mean diff: 0.00e+00
  PASS Labels (argmax): Match! Max diff: 0.00e+00, Mean diff: 0.00e+00
    Seg preds: [3, 40, 20] -> ArgMax -> Semantic mask
  PASS Semantic Mask: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00

  ===========================================================================
  STAGE 8: FINAL OUTPUT
  ===========================================================================

  Layer 53: Final Model Output
  ---------------------------------------------------------------------------
    Output Dictionary:
    |-- 'vectors': (1, 10, 5, 2) (sigmoid reference points)
    |-- 'scores': torch.Size([1, 10]) (sigmoid max cls)
    |-- 'labels': torch.Size([1, 10]) (argmax cls)
    |-- 'semantic_mask': (40, 20)
    |-- 'seg_preds': torch.Size([1, 3, 40, 20])
    '-- 'hs_embeds': torch.Size([10, 1, 32]) (decoder output)

  ===========================================================================
  STAGE 9: MULTI-FRAME TEMPORAL CONSISTENCY
  ===========================================================================
  Running 3 frames: Frame 0 (done above) -> Frame 1 -> Frame 2
  Tests: BEV warp grid from ego-motion, GridSample, Memory Cross-Attn,
         Query Propagation via MotionMLP, Decoder with active memory


  Layer 54: PositionalEncoding1D Table Generation
  ---------------------------------------------------------------------------
    inv_freq: 1/(10000^(k/32)) for 16 frequencies
    sin_inp: outer(positions, inv_freq) -> [100, 16]
    Interleave sin/cos -> [100, 32] -> slice to [100, 32]
  PASS PE Table: Match! Max diff: 0.00e+00, Mean diff: 0.00e+00

  ───────────────────────────────────────────────────────────────────────────
  FRAME 1/2
  ───────────────────────────────────────────────────────────────────────────

  Layer 55: Ego-Motion Warp Grid (Frame 1)
  ---------------------------------------------------------------------------
    Ego translation: prev=[100. 200.] -> curr=[105. 200.]
    Curr->Prev transform applied to BEV plane [20, 10, 4]
    Warp grid: [20, 10, 2] (normalized to [-1,1])
  PASS Warp Grid: Match! Max diff: 2.98e-08, Mean diff: 1.68e-09

  Layer 56: History BEV Warping (GridSample, Frame 1)
  ---------------------------------------------------------------------------
    Previous BEV: [1, 32, 20, 10]
    Warp grid: [1, 20, 10, 2]
    Warped BEV: torch.Size([1, 32, 20, 10])
  PASS Warped BEV: Match! Max diff: 1.06e-06, Mean diff: 3.51e-08

  Layer 57: BEV Query Fusion (Frame 1)
  ---------------------------------------------------------------------------
    Prop BEV: [200, 1, 32]
    Valid mask: sum > 0 -> replace BEV queries with warped features
    Fused queries: torch.Size([200, 1, 32])
  PASS BEV Query Fusion: Match! Max diff: 1.06e-06, Mean diff: 1.34e-08

  Layer 58: SegHead (Frame 1)
  ---------------------------------------------------------------------------
    BEV [1,32,20,10] -> Conv->ReLU->Up->Conv->ReLU->Conv
    Seg preds: torch.Size([1, 3, 40, 20])
  PASS SegHead Preds: Match! Max diff: 2.27e-09, Mean diff: 3.28e-10

  Layer 59: Query Propagation via MotionMLP (Frame 1)
  ---------------------------------------------------------------------------
    Pose: quat=[0,0,0,1] + delta_t=[5. 0. 0.]
    Embedder: [10, 7] -> [10, 147]
    MLP: [10, 179] -> [10, 32]
  PASS MotionMLP Propagation: Match! Max diff: 2.38e-07, Mean diff: 6.33e-09

  Running Decoder for Frame 1 (Memory Cross-Attn ACTIVE)...

  |-- Layer 60: Memory Cross-Attention (ACTIVE, Decoder L1, Frame 1)
     + Relative Temporal PE added to memory keys
     * Q from decoder: [1, 10, 32]
     * K = memory + relative_PE(gap=1): [1, 10, 32]
     * V from memory (prev frame): [1, 10, 32]
     * MHA: Q,K,V proj -> 8 heads -> Softmax -> Output proj
  PASS      MemAttn Output: Match! Max diff: 2.38e-07, Mean diff: 4.81e-08

  |-- Layer 61: Memory Cross-Attention (ACTIVE, Decoder L2, Frame 1)
     + Relative Temporal PE added to memory keys
     * Q from decoder: [1, 10, 32]
     * K = memory + relative_PE(gap=1): [1, 10, 32]
     * V from memory (prev frame): [1, 10, 32]
     * MHA: Q,K,V proj -> 8 heads -> Softmax -> Output proj
  PASS      MemAttn Output: Match! Max diff: 4.77e-07, Mean diff: 7.99e-08

  Layer 62: Frame 1 Decoder Output Validation
  ---------------------------------------------------------------------------
  PASS Decoder Output F1: Match! Max diff: 5.96e-07, Mean diff: 9.60e-08
  PASS Cls Scores F1: Match! Max diff: 2.98e-08, Mean diff: 7.20e-09
  PASS Reg Points F1: Match! Max diff: 5.96e-08, Mean diff: 3.87e-09
    Decoder queries: torch.Size([10, 1, 32])
    Cls: (2, 1, 10, 3), Reg: (2, 1, 10, 5, 2)

  ───────────────────────────────────────────────────────────────────────────
  FRAME 2/2
  ───────────────────────────────────────────────────────────────────────────

  Layer 63: Ego-Motion Warp Grid (Frame 2)
  ---------------------------------------------------------------------------
    Ego translation: prev=[105. 200.] -> curr=[110. 200.]
    Curr->Prev transform applied to BEV plane [20, 10, 4]
    Warp grid: [20, 10, 2] (normalized to [-1,1])
  PASS Warp Grid: Match! Max diff: 2.98e-08, Mean diff: 1.68e-09

  Layer 64: History BEV Warping (GridSample, Frame 2)
  ---------------------------------------------------------------------------
    Previous BEV: [1, 32, 20, 10]
    Warp grid: [1, 20, 10, 2]
    Warped BEV: torch.Size([1, 32, 20, 10])
  PASS Warped BEV: Match! Max diff: 9.83e-07, Mean diff: 3.50e-08

  Layer 65: BEV Query Fusion (Frame 2)
  ---------------------------------------------------------------------------
    Prop BEV: [200, 1, 32]
    Valid mask: sum > 0 -> replace BEV queries with warped features
    Fused queries: torch.Size([200, 1, 32])
  PASS BEV Query Fusion: Match! Max diff: 8.94e-07, Mean diff: 1.47e-08

  Layer 66: SegHead (Frame 2)
  ---------------------------------------------------------------------------
    BEV [1,32,20,10] -> Conv->ReLU->Up->Conv->ReLU->Conv
    Seg preds: torch.Size([1, 3, 40, 20])
  PASS SegHead Preds: Match! Max diff: 1.86e-09, Mean diff: 3.08e-10

  Layer 67: Query Propagation via MotionMLP (Frame 2)
  ---------------------------------------------------------------------------
    Pose: quat=[0,0,0,1] + delta_t=[5. 0. 0.]
    Embedder: [10, 7] -> [10, 147]
    MLP: [10, 179] -> [10, 32]
  PASS MotionMLP Propagation: Match! Max diff: 1.19e-07, Mean diff: 5.63e-09

  Running Decoder for Frame 2 (Memory Cross-Attn ACTIVE)...

  |-- Layer 68: Memory Cross-Attention (ACTIVE, Decoder L1, Frame 2)
     + Relative Temporal PE added to memory keys
     * Q from decoder: [1, 10, 32]
     * K = memory + relative_PE(gap=1): [1, 10, 32]
     * V from memory (prev frame): [1, 10, 32]
     * MHA: Q,K,V proj -> 8 heads -> Softmax -> Output proj
  PASS      MemAttn Output: Match! Max diff: 2.38e-07, Mean diff: 4.19e-08

  |-- Layer 69: Memory Cross-Attention (ACTIVE, Decoder L2, Frame 2)
     + Relative Temporal PE added to memory keys
     * Q from decoder: [1, 10, 32]
     * K = memory + relative_PE(gap=1): [1, 10, 32]
     * V from memory (prev frame): [1, 10, 32]
     * MHA: Q,K,V proj -> 8 heads -> Softmax -> Output proj
  PASS      MemAttn Output: Match! Max diff: 4.77e-07, Mean diff: 7.61e-08

  Layer 70: Frame 2 Decoder Output Validation
  ---------------------------------------------------------------------------
  PASS Decoder Output F2: Match! Max diff: 5.96e-07, Mean diff: 9.03e-08
  PASS Cls Scores F2: Match! Max diff: 2.24e-08, Mean diff: 6.70e-09
  PASS Reg Points F2: Match! Max diff: 5.96e-08, Mean diff: 2.24e-09
    Decoder queries: torch.Size([10, 1, 32])
    Cls: (2, 1, 10, 3), Reg: (2, 1, 10, 5, 2)

  ───────────────────────────────────────────────────────────────────────────
  MULTI-FRAME SUMMARY: 3 frames processed
    Frame 0: decoder=torch.Size([10, 1, 32]), cls=(1, 10, 3), reg=torch.Size([1, 10, 5, 2])
    Frame 1: decoder=torch.Size([10, 1, 32]), cls=(1, 10, 3), reg=torch.Size([1, 10, 5, 2])
    Frame 2: decoder=torch.Size([10, 1, 32]), cls=(1, 10, 3), reg=torch.Size([1, 10, 5, 2])
  ───────────────────────────────────────────────────────────────────────────

  ===========================================================================
  COMPLETE MAPTRACKER PIPELINE:
  ===========================================================================
  Input: Multi-camera images [1, 2, 3, 28, 50]
    |
  Image Backbone (ResNet-50 Bottleneck): Extract multi-scale features
    |
  FPN Neck: 3-level feature pyramid -> all 32 channels
    |
  BEVFormer Encoder (2 layers): Camera features -> BEV space
    |
  BEV features [1, 32, 20, 10]
    |
  BEV Backbone: Embedding -> Reshape -> Permute
    + History warping via GridSample
    |
  Segmentation Head: Conv2d(3x3) -> ReLU -> Upsample -> Conv2d -> ReLU -> Conv2d(1x1)
    -> seg_preds [1, 3, 40, 20]
    |
  Detection Head: Input Proj -> BEV PosEmbed -> Flatten
    |
  Transformer Decoder (2 layers):
    Each: Self-Attn -> LN -> DeformAttn -> LN -> MemAttn -> LN -> FFN -> LN
    -> RegressionBranch: Linear->LN->ReLU->Linear->LN->ReLU->Linear -> Sigmoid
    -> ClassificationBranch: Linear(32->3)
    |
  Query Propagation (MotionMLP):
    Embedder(sin/cos) -> Concat -> Linear -> LN -> ReLU -> Linear + Residual
    |
  Multi-Frame Temporal (3 frames):
    Ego-motion warp grid -> GridSample -> BEV fusion
    -> Decoder with Memory Cross-Attn (ACTIVE on frames 1+)
    |
  Output:
    * Class scores: [2, 1, 10, 3]
    * Reference points: [2, 1, 10, 5, 2]
    * Semantic mask: [40, 20]
  ===========================================================================

  ===========================================================================
  VALIDATION SUMMARY
  ===========================================================================

  Component                                     Status     Details
  ---------------------------------------------------------------------------
  Layer 1: Camera Input                         PASS       torch.Size([1, 2, 3, 28, 50])
  Layer 2: Backbone Stem                        PASS       torch.Size([2, 8, 7, 13])
  Layer 3: ResNet-50 Stages                     PASS       C3:torch.Size([2, 16, 4, 7]), C4:torch.Size([2, 32, 2, 4]), C5:torch.Size([2, 64, 1, 2])
  Layer 4: FPN Multi-scale                      PASS       3 levels: torch.Size([2, 32, 4, 7]) to torch.Size([2, 32, 1, 2])
  Layer 5: BEV Query Init                       PASS       torch.Size([1, 200, 32])
  Layer 6: Temporal Self-Attn                   PASS       torch.Size([1, 200, 32])
  Layer 7: LayerNorm                            PASS       torch.Size([1, 200, 32])
  Layer 8: Spatial Cross-Attn                   PASS       torch.Size([1, 200, 32])
  Layer 9: LayerNorm                            PASS       torch.Size([1, 200, 32])
  Layer 10: FFN                                 PASS       torch.Size([1, 200, 32])
  Layer 11: LayerNorm                           PASS       torch.Size([1, 200, 32])
  Layer 12: Temporal Self-Attn                  PASS       torch.Size([1, 200, 32])
  Layer 13: LayerNorm                           PASS       torch.Size([1, 200, 32])
  Layer 14: Spatial Cross-Attn                  PASS       torch.Size([1, 200, 32])
  Layer 15: LayerNorm                           PASS       torch.Size([1, 200, 32])
  Layer 16: FFN                                 PASS       torch.Size([1, 200, 32])
  Layer 17: LayerNorm                           PASS       torch.Size([1, 200, 32])
  Layer 18: BEV Input                           PASS       torch.Size([1, 32, 20, 10])
  Layer 19: BEV Embedding                       PASS       torch.Size([200, 32])
  Layer 20: Reshape+Permute                     PASS       torch.Size([1, 32, 20, 10])
  Layer 21: GridSample                          PASS       torch.Size([1, 2, 32, 20, 10])
  Layer 22: SegHead Conv_in                     PASS       torch.Size([1, 32, 20, 10])
  Layer 23: SegHead Up+Conv                     PASS       torch.Size([1, 32, 40, 20])
  Layer 24: SegHead Preds                       PASS       torch.Size([1, 3, 40, 20])
  Layer 25: SegHead Feats                       PASS       torch.Size([1, 32, 20, 10])
  Layer 26: Input Proj                          PASS       torch.Size([1, 32, 20, 10])
  Layer 27: BEV PosEmbed                        PASS       torch.Size([1, 32, 20, 10])
  Layer 28: BEV Flatten                         PASS       torch.Size([200, 1, 32])
  Layer 29: Query Init                          PASS       Q:torch.Size([1, 10, 32]), Ref:torch.Size([1, 10, 5, 2])
  Layer 30: Self-Attn L1                        PASS       torch.Size([10, 1, 32])
  Layer 31: LN L1                               PASS       torch.Size([10, 1, 32])
  Layer 32: DeformAttn L1                       PASS       torch.Size([10, 1, 32])
  Layer 33: LN L1                               PASS       torch.Size([10, 1, 32])
  Layer 34: MemAttn L1                          PASS       torch.Size([10, 1, 32])
  Layer 35: LN L1                               PASS       torch.Size([10, 1, 32])
  Layer 36: FFN L1                              PASS       torch.Size([10, 1, 32])
  Layer 37: LN L1                               PASS       torch.Size([10, 1, 32])
  Layer 38: RegBranch L1                        PASS       torch.Size([1, 10, 5, 2])
  Layer 39: ClsBranch L1                        PASS       torch.Size([1, 10, 3])
  Layer 40: Self-Attn L2                        PASS       torch.Size([10, 1, 32])
  Layer 41: LN L2                               PASS       torch.Size([10, 1, 32])
  Layer 42: DeformAttn L2                       PASS       torch.Size([10, 1, 32])
  Layer 43: LN L2                               PASS       torch.Size([10, 1, 32])
  Layer 44: MemAttn L2                          PASS       torch.Size([10, 1, 32])
  Layer 45: LN L2                               PASS       torch.Size([10, 1, 32])
  Layer 46: FFN L2                              PASS       torch.Size([10, 1, 32])
  Layer 47: LN L2                               PASS       torch.Size([10, 1, 32])
  Layer 48: RegBranch L2                        PASS       torch.Size([1, 10, 5, 2])
  Layer 49: ClsBranch L2                        PASS       torch.Size([1, 10, 3])
  Final Cls Scores                              PASS       (2, 1, 10, 3)
  Final Reg Points                              PASS       (2, 1, 10, 5, 2)
  Layer 50: Embedder                            PASS       torch.Size([10, 147])
  Layer 51: MotionMLP                           PASS       torch.Size([10, 32])
  Layer 52: Post-Processing                     PASS       scores + labels + seg mask
  Final Output                                  PASS       Vec:(1, 10, 5, 2), Seg:(40, 20)
  Layer 54: PE1D Table                          PASS       (100,32)
  Layer 55: WarpGrid F1                         PASS       (20,10,2)
  Layer 56: WarpBEV F1                          PASS       torch.Size([1, 32, 20, 10])
  Layer 57: BEVFusion F1                        PASS       torch.Size([200, 1, 32])
  Layer 58: Seg F1                              PASS       torch.Size([1, 3, 40, 20])
  Layer 59: MotionMLP F1                        PASS       (10,32)
  Layer 60: MemAttn L1 F1                       PASS       torch.Size([10, 1, 32])
  Layer 61: MemAttn L2 F1                       PASS       torch.Size([10, 1, 32])
  Layer 62: Decoder F1                          PASS       dec+cls+reg
  Layer 63: WarpGrid F2                         PASS       (20,10,2)
  Layer 64: WarpBEV F2                          PASS       torch.Size([1, 32, 20, 10])
  Layer 65: BEVFusion F2                        PASS       torch.Size([200, 1, 32])
  Layer 66: Seg F2                              PASS       torch.Size([1, 3, 40, 20])
  Layer 67: MotionMLP F2                        PASS       (10,32)
  Layer 68: MemAttn L1 F2                       PASS       torch.Size([10, 1, 32])
  Layer 69: MemAttn L2 F2                       PASS       torch.Size([10, 1, 32])
  Layer 70: Decoder F2                          PASS       dec+cls+reg
  Layer 71: Multi-Frame (3f)                    PASS       warp+fusion+memAttn+MLP×2

  73/73 components validated
  Total layers processed: 71

  SUCCESS! Complete MapTracker architecture validated with TTSim compute functions!

================================================================================
TEST SUMMARY
================================================================================
Layer-by-Layer Validation with TTSim........................ PASS

Total: 1/1 tests passed

All tests passed! TTSim computations match PyTorch!
```

---

## test_positional_encoding1d.py  —  PASS

### stdout

```
======================================================================
Testing PositionalEncoding1D - PyTorch vs ttsim
======================================================================

Input shape: [2, 10, 256]
Input values:
[[[ 4.9671414e-01 -1.3826430e-01  6.4768857e-01 ...  1.0324652e+00
   -1.5193700e+00 -4.8423406e-01]
  [ 1.2669111e+00 -7.0766944e-01  4.4381943e-01 ... -8.3095014e-01
    2.7045682e-01 -5.0238110e-02]
  [-2.3894805e-01 -9.0756369e-01 -5.7677132e-01 ... -7.0317644e-01
   -3.4988489e-02  1.7708006e+00]
  ...
  [-1.6013280e-01  6.7134005e-01  2.1319664e-01 ...  1.1210307e+00
    2.0706492e-04 -9.3003213e-03]
  [-3.2789472e-01  1.5519068e-01  8.2509828e-01 ... -9.9806064e-01
   -3.8397127e-01  2.5020021e-01]
  [ 1.9956675e+00  3.1099186e+00  6.0672307e-01 ...  3.3755065e-03
    3.2782117e-01  9.2427015e-01]]

 [[-1.0138960e+00  8.5687160e-02 -9.2542464e-01 ... -3.2124278e-01
    1.8325571e+00  8.1415176e-01]
  [ 4.8206672e-01  3.6873314e-01  3.9379731e-01 ...  6.9272274e-01
   -1.2693305e+00  1.7025146e+00]
  [ 2.0232880e-01  1.6318569e+00 -7.3303300e-01 ...  4.7516775e-01
   -2.4038409e-03 -5.8927166e-01]
  ...
  [-2.4907185e-01 -1.6728939e+00  4.0421286e-01 ... -2.5367609e-01
   -3.4516472e-01  4.8925453e-01]
  [ 6.8635356e-01 -8.3889270e-01  6.7838512e-02 ... -8.2611197e-01
   -6.8701410e-01 -1.3505560e+00]
  [-7.5871453e-02 -1.6846676e+00  1.9091652e-01 ... -5.8945239e-01
   -1.5824280e+00  8.1683701e-01]]]
Input stats:
  Min: -3.241267
  Max: 3.926238
  Mean: 0.006421
  Std: 0.996481

TEST 1: PyTorch PositionalEncoding1D (reference)
----------------------------------------------------------------------
  Output shape: (2, 10, 256)
  Output values:
[[[ 0.0000000e+00  1.0000000e+00  0.0000000e+00 ...  1.0000000e+00
    0.0000000e+00  1.0000000e+00]
  [ 8.4147096e-01  5.4030234e-01  8.0196178e-01 ...  1.0000000e+00
    1.0746078e-04  1.0000000e+00]
  [ 9.0929741e-01 -4.1614684e-01  9.5814437e-01 ...  1.0000000e+00
    2.1492156e-04  1.0000000e+00]
  ...
  [ 6.5698659e-01  7.5390226e-01  2.2877480e-01 ...  9.9999970e-01
    7.5222540e-04  9.9999976e-01]
  [ 9.8935825e-01 -1.4550003e-01  9.1735768e-01 ...  9.9999958e-01
    8.5968612e-04  9.9999964e-01]
  [ 4.1211849e-01 -9.1113025e-01  8.6723864e-01 ...  9.9999946e-01
    9.6714683e-04  9.9999952e-01]]

 [[ 0.0000000e+00  1.0000000e+00  0.0000000e+00 ...  1.0000000e+00
    0.0000000e+00  1.0000000e+00]
  [ 8.4147096e-01  5.4030234e-01  8.0196178e-01 ...  1.0000000e+00
    1.0746078e-04  1.0000000e+00]
  [ 9.0929741e-01 -4.1614684e-01  9.5814437e-01 ...  1.0000000e+00
    2.1492156e-04  1.0000000e+00]
  ...
  [ 6.5698659e-01  7.5390226e-01  2.2877480e-01 ...  9.9999970e-01
    7.5222540e-04  9.9999976e-01]
  [ 9.8935825e-01 -1.4550003e-01  9.1735768e-01 ...  9.9999958e-01
    8.5968612e-04  9.9999964e-01]
  [ 4.1211849e-01 -9.1113025e-01  8.6723864e-01 ...  9.9999946e-01
    9.6714683e-04  9.9999952e-01]]]
  Output stats:
    Min: -0.999998
    Max: 1.000000
    Mean: 0.479811
    Std: 0.519405

TEST 2: ttsim PositionalEncoding1D
----------------------------------------------------------------------
  Input shape: Shape([2, 10, 256])
  Output shape: Shape([2, 10, 256])
  Output values:
[[[ 0.0000000e+00  1.0000000e+00  0.0000000e+00 ...  1.0000000e+00
    0.0000000e+00  1.0000000e+00]
  [ 8.4147096e-01  5.4030234e-01  8.0196178e-01 ...  1.0000000e+00
    1.0746078e-04  1.0000000e+00]
  [ 9.0929741e-01 -4.1614684e-01  9.5814437e-01 ...  1.0000000e+00
    2.1492156e-04  1.0000000e+00]
  ...
  [ 6.5698659e-01  7.5390226e-01  2.2877480e-01 ...  9.9999970e-01
    7.5222540e-04  9.9999976e-01]
  [ 9.8935825e-01 -1.4550003e-01  9.1735768e-01 ...  9.9999958e-01
    8.5968612e-04  9.9999964e-01]
  [ 4.1211849e-01 -9.1113025e-01  8.6723864e-01 ...  9.9999946e-01
    9.6714683e-04  9.9999952e-01]]

 [[ 0.0000000e+00  1.0000000e+00  0.0000000e+00 ...  1.0000000e+00
    0.0000000e+00  1.0000000e+00]
  [ 8.4147096e-01  5.4030234e-01  8.0196178e-01 ...  1.0000000e+00
    1.0746078e-04  1.0000000e+00]
  [ 9.0929741e-01 -4.1614684e-01  9.5814437e-01 ...  1.0000000e+00
    2.1492156e-04  1.0000000e+00]
  ...
  [ 6.5698659e-01  7.5390226e-01  2.2877480e-01 ...  9.9999970e-01
    7.5222540e-04  9.9999976e-01]
  [ 9.8935825e-01 -1.4550003e-01  9.1735768e-01 ...  9.9999958e-01
    8.5968612e-04  9.9999964e-01]
  [ 4.1211849e-01 -9.1113025e-01  8.6723864e-01 ...  9.9999946e-01
    9.6714683e-04  9.9999952e-01]]]
  Output .data is None? False
  [OK] SUCCESS: Output data was computed!
  ttsim output stats:
    Min: -0.999998
    Max: 1.000000
    Mean: 0.479811
    Std: 0.519405

TEST 3: Numerical Comparison
----------------------------------------------------------------------
  Tolerance: atol=1e-06, rtol=1e-05
  Max absolute difference: 0.0000000000
  Mean absolute difference: 0.0000000000
  Arrays match (allclose): True
  [PASS] ttsim matches PyTorch (within atol=1e-06, rtol=1e-05)

TEST 4: Graph Connectivity (Sin/Cos interleaving via TTSim ops)
----------------------------------------------------------------------
  pe_table exists: True
  pe_table shape: Shape([1000, 256])
  pe_table.data is None? False
  pe_table[:, :org_channels] matches emb_cache: True
  [PASS] pe_table data matches emb_cache
  TTSim interleave ops (Unsqueeze, ConcatX, Reshape): all present
  [PASS] Sin/Cos are graph-connected through interleaving ops
  [PASS] Sin/Cos interleaving pattern verified (sin at even, cos at odd indices)

======================================================================
Test Complete!
======================================================================
```

---

## test_upsample_block.py  —  PASS

### stdout

```
================================================================================
Testing UpsampleBlock: ttsim vs PyTorch
================================================================================

Input shape: (2, 64, 8, 10)
ins=64, outs=128, h=8, w=10

--------------------------------------------------------------------------------
PyTorch Implementation
--------------------------------------------------------------------------------
PyTorch output shape: torch.Size([2, 128, 16, 20])
PyTorch output range: [0.000000, 3.252455]
tensor([[[[0.0000e+00, 2.7798e-01, 5.5596e-01,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          [2.5709e-01, 2.8357e-01, 3.1004e-01,  ..., 2.7762e-01,
           1.7254e-01, 6.7461e-02],
          [5.1418e-01, 2.8915e-01, 6.4126e-02,  ..., 5.5524e-01,
           3.4508e-01, 1.3492e-01],
          ...,
          [1.0749e-01, 4.5885e-01, 8.1021e-01,  ..., 1.8273e+00,
           9.1367e-01, 0.0000e+00],
          [8.5995e-01, 6.5374e-01, 4.4754e-01,  ..., 1.0579e+00,
           5.2893e-01, 0.0000e+00],
          [1.6124e+00, 8.4864e-01, 8.4864e-02,  ..., 2.8840e-01,
           1.4420e-01, 0.0000e+00]],

         [[1.0470e+00, 7.3701e-01, 4.2699e-01,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          [5.5842e-01, 3.9307e-01, 2.2773e-01,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          [6.9803e-02, 4.9134e-02, 2.8466e-02,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          ...,
          [4.5622e-01, 7.3180e-01, 1.0074e+00,  ..., 6.0047e-02,
           5.8459e-01, 1.1091e+00],
          [4.2572e-01, 4.6990e-01, 5.1408e-01,  ..., 5.4811e-02,
           4.2102e-01, 7.8723e-01],
          [3.9521e-01, 2.0800e-01, 2.0800e-02,  ..., 4.9576e-02,
           2.5745e-01, 4.6532e-01]],

         [[2.5208e-01, 9.7745e-01, 1.7028e+00,  ..., 6.9289e-01,
           3.4644e-01, 0.0000e+00],
          [3.0399e-01, 7.2774e-01, 1.1515e+00,  ..., 3.8625e-01,
           3.5184e-01, 3.1744e-01],
          [3.5591e-01, 4.7802e-01, 6.0013e-01,  ..., 7.9607e-02,
           3.5724e-01, 6.3488e-01],
          ...,
          [4.5833e-02, 3.6614e-01, 6.8645e-01,  ..., 2.1118e+00,
           1.0559e+00, 0.0000e+00],
          [3.6667e-01, 3.6399e-01, 3.6132e-01,  ..., 1.4331e+00,
           7.1654e-01, 0.0000e+00],
          [6.8750e-01, 3.6184e-01, 3.6184e-02,  ..., 7.5441e-01,
           3.7720e-01, 0.0000e+00]],

         ...,

         [[0.0000e+00, 1.3155e-01, 2.6310e-01,  ..., 3.8772e-02,
           3.8772e-01, 7.3666e-01],
          [0.0000e+00, 1.4765e-01, 2.9529e-01,  ..., 2.0678e-02,
           2.0678e-01, 3.9289e-01],
          [0.0000e+00, 1.6374e-01, 3.2749e-01,  ..., 2.5848e-03,
           2.5848e-02, 4.9111e-02],
          ...,
          [4.5966e-02, 3.4074e-02, 2.2181e-02,  ..., 7.6294e-02,
           3.6421e-01, 6.5212e-01],
          [3.6773e-01, 1.9848e-01, 2.9235e-02,  ..., 3.5293e-01,
           3.3950e-01, 3.2606e-01],
          [6.8949e-01, 3.6289e-01, 3.6289e-02,  ..., 6.2957e-01,
           3.1479e-01, 0.0000e+00]],

         [[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 4.7502e-01,
           2.3751e-01, 0.0000e+00],
          [5.6028e-01, 3.0723e-01, 5.4190e-02,  ..., 6.1083e-01,
           3.0542e-01, 0.0000e+00],
          [1.1206e+00, 6.1447e-01, 1.0838e-01,  ..., 7.4665e-01,
           3.7332e-01, 0.0000e+00],
          ...,
          [4.9915e-01, 2.9743e-01, 9.5702e-02,  ..., 8.3004e-02,
           8.6922e-02, 9.0840e-02],
          [2.4958e-01, 4.0908e-01, 5.6859e-01,  ..., 6.6404e-01,
           6.9538e-01, 7.2672e-01],
          [0.0000e+00, 5.2073e-01, 1.0415e+00,  ..., 1.2451e+00,
           1.3038e+00, 1.3626e+00]],

         [[4.2636e-01, 6.6993e-01, 9.1351e-01,  ..., 4.6027e-01,
           4.2404e-01, 3.8781e-01],
          [2.2739e-01, 4.2374e-01, 6.2010e-01,  ..., 2.5073e-01,
           2.7871e-01, 3.0669e-01],
          [2.8424e-02, 1.7755e-01, 3.2668e-01,  ..., 4.1196e-02,
           1.3338e-01, 2.2557e-01],
          ...,
          [0.0000e+00, 5.2799e-01, 1.0560e+00,  ..., 1.9327e-02,
           1.9327e-01, 3.6721e-01],
          [0.0000e+00, 4.5389e-01, 9.0778e-01,  ..., 3.7214e-02,
           3.7214e-01, 7.0706e-01],
          [0.0000e+00, 3.7979e-01, 7.5958e-01,  ..., 5.5100e-02,
           5.5101e-01, 1.0469e+00]]],


        [[[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          [0.0000e+00, 6.7544e-01, 1.3509e+00,  ..., 7.0094e-01,
           3.5047e-01, 0.0000e+00],
          [0.0000e+00, 1.3509e+00, 2.7017e+00,  ..., 1.4019e+00,
           7.0094e-01, 0.0000e+00],
          ...,
          [0.0000e+00, 1.2842e-02, 2.5683e-02,  ..., 1.1771e-02,
           1.1771e-01, 2.2365e-01],
          [0.0000e+00, 1.0273e-01, 2.0546e-01,  ..., 5.8854e-03,
           5.8855e-02, 1.1182e-01],
          [0.0000e+00, 1.9262e-01, 3.8525e-01,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00]],

         [[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 4.8142e-01,
           6.1902e-01, 7.5662e-01],
          [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 7.8022e-01,
           5.9188e-01, 4.0353e-01],
          [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 1.0790e+00,
           5.6473e-01, 5.0442e-02],
          ...,
          [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 2.9187e-02,
           2.9187e-01, 5.5455e-01],
          [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 1.4593e-02,
           1.4593e-01, 2.7728e-01],
          [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00]],

         [[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 4.3303e-01,
           3.8820e-01, 3.4338e-01],
          [0.0000e+00, 1.5526e-01, 3.1051e-01,  ..., 6.2103e-01,
           5.0216e-01, 3.8329e-01],
          [0.0000e+00, 3.1051e-01, 6.2103e-01,  ..., 8.0903e-01,
           6.1612e-01, 4.2321e-01],
          ...,
          [0.0000e+00, 3.0566e-02, 6.1132e-02,  ..., 2.0739e-01,
           1.3262e+00, 2.4450e+00],
          [0.0000e+00, 1.5283e-02, 3.0566e-02,  ..., 7.0078e-01,
           1.0261e+00, 1.3514e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 1.1942e+00,
           7.2593e-01, 2.5770e-01]],

         ...,

         [[0.0000e+00, 2.3948e-01, 4.7896e-01,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          [5.1641e-01, 3.9952e-01, 2.8263e-01,  ..., 6.6560e-02,
           3.3280e-02, 0.0000e+00],
          [1.0328e+00, 5.5956e-01, 8.6290e-02,  ..., 1.3312e-01,
           6.6560e-02, 0.0000e+00],
          ...,
          [9.0749e-01, 6.6212e-01, 4.1675e-01,  ..., 3.3957e-01,
           1.6979e-01, 0.0000e+00],
          [5.1595e-01, 3.6380e-01, 2.1165e-01,  ..., 1.6979e-01,
           8.4893e-02, 0.0000e+00],
          [1.2440e-01, 6.5473e-02, 6.5473e-03,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00]],

         [[0.0000e+00, 8.0218e-01, 1.6044e+00,  ..., 6.6579e-01,
           3.8279e-01, 9.9795e-02],
          [0.0000e+00, 4.2783e-01, 8.5566e-01,  ..., 1.1919e+00,
           8.1254e-01, 4.3317e-01],
          [0.0000e+00, 5.3479e-02, 1.0696e-01,  ..., 1.7180e+00,
           1.2423e+00, 7.6655e-01],
          ...,
          [1.7655e-01, 1.6846e-01, 1.6037e-01,  ..., 3.0521e-01,
           1.7609e-01, 4.6977e-02],
          [4.3368e-01, 2.6602e-01, 9.8362e-02,  ..., 2.2890e-01,
           1.2619e-01, 2.3488e-02],
          [6.9080e-01, 3.6358e-01, 3.6358e-02,  ..., 1.5258e-01,
           7.6291e-02, 0.0000e+00]],

         [[4.1717e-01, 4.9384e-01, 5.7051e-01,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          [2.4689e-01, 2.7622e-01, 3.0556e-01,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          [7.6598e-02, 5.8600e-02, 4.0602e-02,  ..., 0.0000e+00,
           0.0000e+00, 0.0000e+00],
          ...,
          [4.5068e-01, 2.7570e-01, 1.0072e-01,  ..., 7.8795e-02,
           3.9398e-02, 0.0000e+00],
          [2.2534e-01, 4.2660e-01, 6.2786e-01,  ..., 6.3036e-01,
           3.1518e-01, 0.0000e+00],
          [0.0000e+00, 5.7750e-01, 1.1550e+00,  ..., 1.1819e+00,
           5.9097e-01, 0.0000e+00]]]])

--------------------------------------------------------------------------------
ttsim Implementation
--------------------------------------------------------------------------------

Injecting PyTorch weights into ttsim...
Weight injection complete
ttsim output shape: (2, 128, 16, 20)
ttsim output range: [0.000000, 3.252455]
[[[[0.00000000e+00 2.77979076e-01 5.55958152e-01 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   [2.57090867e-01 2.83566505e-01 3.10042113e-01 ... 2.77620673e-01
    1.72540754e-01 6.74608052e-02]
   [5.14181733e-01 2.89153934e-01 6.41260743e-02 ... 5.55241346e-01
    3.45081508e-01 1.34921610e-01]
   ...
   [1.07494108e-01 4.58851635e-01 8.10209155e-01 ... 1.82733297e+00
    9.13666487e-01 0.00000000e+00]
   [8.59952867e-01 6.53744698e-01 4.47536439e-01 ... 1.05786872e+00
    5.28934360e-01 0.00000000e+00]
   [1.61241150e+00 8.48637640e-01 8.48637670e-02 ... 2.88404465e-01
    1.44202232e-01 0.00000000e+00]]

  [[1.04704094e+00 7.37014294e-01 4.26987618e-01 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   [5.58421850e-01 3.93074304e-01 2.27726743e-01 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   [6.98027313e-02 4.91342880e-02 2.84658428e-02 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   ...
   [4.56225276e-01 7.31796205e-01 1.00736701e+00 ... 6.00474887e-02
    5.84587872e-01 1.10912824e+00]
   [4.25716013e-01 4.69899893e-01 5.14083683e-01 ... 5.48114814e-02
    4.21018600e-01 7.87225664e-01]
   [3.95206720e-01 2.08003551e-01 2.08003540e-02 ... 4.95754704e-02
    2.57449299e-01 4.65323091e-01]]

  [[2.52077132e-01 9.77450073e-01 1.70282292e+00 ... 6.92885041e-01
    3.46442521e-01 0.00000000e+00]
   [3.03992271e-01 7.27734864e-01 1.15147746e+00 ... 3.86246026e-01
    3.51842523e-01 3.17439049e-01]
   [3.55907440e-01 4.78019536e-01 6.00131631e-01 ... 7.96069726e-02
    3.57242554e-01 6.34878099e-01]
   ...
   [4.58335094e-02 3.66143614e-01 6.86453760e-01 ... 2.11175680e+00
    1.05587840e+00 0.00000000e+00]
   [3.66668075e-01 3.63993585e-01 3.61319035e-01 ... 1.43308318e+00
    7.16541588e-01 0.00000000e+00]
   [6.87502623e-01 3.61843497e-01 3.61843482e-02 ... 7.54409432e-01
    3.77204716e-01 0.00000000e+00]]

  ...

  [[0.00000000e+00 1.31547511e-01 2.63095021e-01 ... 3.87716629e-02
    3.87716651e-01 7.36661613e-01]
   [0.00000000e+00 1.47645742e-01 2.95291483e-01 ... 2.06782222e-02
    2.06782222e-01 3.92886221e-01]
   [0.00000000e+00 1.63743973e-01 3.27487946e-01 ... 2.58477777e-03
    2.58477777e-02 4.91107777e-02]
   ...
   [4.59660999e-02 3.40738446e-02 2.21815854e-02 ... 7.62937963e-02
    3.64209086e-01 6.52124345e-01]
   [3.67728800e-01 1.98482066e-01 2.92353071e-02 ... 3.52932870e-01
    3.39497536e-01 3.26062173e-01]
   [6.89491451e-01 3.62890244e-01 3.62890251e-02 ... 6.29571974e-01
    3.14785987e-01 0.00000000e+00]]

  [[0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 4.75019902e-01
    2.37509951e-01 0.00000000e+00]
   [5.60279012e-01 3.07234794e-01 5.41905388e-02 ... 6.10833704e-01
    3.05416852e-01 0.00000000e+00]
   [1.12055802e+00 6.14469588e-01 1.08381078e-01 ... 7.46647477e-01
    3.73323739e-01 0.00000000e+00]
   ...
   [4.99151289e-01 2.97426879e-01 9.57023948e-02 ... 8.30044895e-02
    8.69222879e-02 9.08400714e-02]
   [2.49575645e-01 4.09080714e-01 5.68585753e-01 ... 6.64035916e-01
    6.95378304e-01 7.26720572e-01]
   [0.00000000e+00 5.20734489e-01 1.04146898e+00 ... 1.24506736e+00
    1.30383420e+00 1.36260104e+00]]

  [[4.26361203e-01 6.69934869e-01 9.13508534e-01 ... 4.60269183e-01
    4.24041331e-01 3.87813449e-01]
   [2.27392659e-01 4.23744619e-01 6.20096564e-01 ... 2.50732511e-01
    2.78711319e-01 3.06690097e-01]
   [2.84240823e-02 1.77554294e-01 3.26684505e-01 ... 4.11957987e-02
    1.33381277e-01 2.25566745e-01]
   ...
   [0.00000000e+00 5.27992666e-01 1.05598533e+00 ... 1.93269327e-02
    1.93269342e-01 3.67211729e-01]
   [0.00000000e+00 4.53892052e-01 9.07784104e-01 ... 3.72137912e-02
    3.72137934e-01 7.07062006e-01]
   [0.00000000e+00 3.79791379e-01 7.59582758e-01 ... 5.51006496e-02
    5.51006496e-01 1.04691231e+00]]]


 [[[0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   [0.00000000e+00 6.75434530e-01 1.35086906e+00 ... 7.00942397e-01
    3.50471199e-01 0.00000000e+00]
   [0.00000000e+00 1.35086906e+00 2.70173812e+00 ... 1.40188479e+00
    7.00942397e-01 0.00000000e+00]
   ...
   [0.00000000e+00 1.28415301e-02 2.56830603e-02 ... 1.17709218e-02
    1.17709227e-01 2.23647520e-01]
   [0.00000000e+00 1.02732241e-01 2.05464482e-01 ... 5.88546088e-03
    5.88546135e-02 1.11823760e-01]
   [0.00000000e+00 1.92622945e-01 3.85245889e-01 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]]

  [[0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 4.81419265e-01
    6.19021237e-01 7.56623209e-01]
   [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 7.80222535e-01
    5.91877460e-01 4.03532386e-01]
   [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 1.07902575e+00
    5.64733684e-01 5.04415482e-02]
   ...
   [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 2.91869324e-02
    2.91869342e-01 5.54551721e-01]
   [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 1.45934662e-02
    1.45934671e-01 2.77275860e-01]
   [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]]

  [[0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 4.33026344e-01
    3.88203025e-01 3.43379676e-01]
   [0.00000000e+00 1.55256033e-01 3.10512066e-01 ... 6.21027648e-01
    5.02161264e-01 3.83294821e-01]
   [0.00000000e+00 3.10512066e-01 6.21024132e-01 ... 8.09028864e-01
    6.16119385e-01 4.23209965e-01]
   ...
   [0.00000000e+00 3.05659361e-02 6.11318722e-02 ... 2.07392037e-01
    1.32620645e+00 2.44502091e+00]
   [0.00000000e+00 1.52829681e-02 3.05659361e-02 ... 7.00778127e-01
    1.02607000e+00 1.35136187e+00]
   [0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 1.19416416e+00
    7.25933433e-01 2.57702738e-01]]

  ...

  [[0.00000000e+00 2.39481881e-01 4.78963763e-01 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   [5.16412914e-01 3.99519980e-01 2.82626987e-01 ... 6.65598437e-02
    3.32799219e-02 0.00000000e+00]
   [1.03282583e+00 5.59558034e-01 8.62901732e-02 ... 1.33119687e-01
    6.65598437e-02 0.00000000e+00]
   ...
   [9.07491684e-01 6.62122607e-01 4.16753531e-01 ... 3.39571118e-01
    1.69785559e-01 0.00000000e+00]
   [5.15945315e-01 3.63797873e-01 2.11650416e-01 ... 1.69785559e-01
    8.48927796e-02 0.00000000e+00]
   [1.24398910e-01 6.54731095e-02 6.54731086e-03 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]]

  [[0.00000000e+00 8.02183568e-01 1.60436714e+00 ... 6.65791214e-01
    3.82793218e-01 9.97951850e-02]
   [0.00000000e+00 4.27831262e-01 8.55662525e-01 ... 1.19191599e+00
    8.12544346e-01 4.33172464e-01]
   [0.00000000e+00 5.34789078e-02 1.06957816e-01 ... 1.71804082e+00
    1.24229527e+00 7.66549766e-01]
   ...
   [1.76549867e-01 1.68457776e-01 1.60365701e-01 ... 3.05209726e-01
    1.76093370e-01 4.69769947e-02]
   [4.33675259e-01 2.66018540e-01 9.83618125e-02 ... 2.28896201e-01
    1.26192361e-01 2.34884974e-02]
   [6.90800607e-01 3.63579273e-01 3.63579281e-02 ... 1.52582675e-01
    7.62913376e-02 0.00000000e+00]]

  [[4.17174667e-01 4.93843913e-01 5.70513189e-01 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   [2.46886551e-01 2.76222080e-01 3.05557579e-01 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   [7.65983909e-02 5.86001687e-02 4.06019390e-02 ... 0.00000000e+00
    0.00000000e+00 0.00000000e+00]
   ...
   [4.50682878e-01 2.75701731e-01 1.00720562e-01 ... 7.87953213e-02
    3.93976606e-02 0.00000000e+00]
   [2.25341439e-01 4.26602393e-01 6.27863348e-01 ... 6.30362570e-01
    3.15181285e-01 0.00000000e+00]
   [0.00000000e+00 5.77502966e-01 1.15500593e+00 ... 1.18192971e+00
    5.90964854e-01 0.00000000e+00]]]]

================================================================================
Comparison
================================================================================

Max absolute difference: 0.0000035763
Mean absolute difference: 0.0000001780
Std absolute difference: 0.0000002170

[PASS] PASS: Outputs match within tolerance (atol=1e-05, rtol=1e-05)

================================================================================
```

---

## test_vectorinstancememory.py  —  PASS

### stdout

```
======================================================================
VectorInstanceMemory Comparison: PyTorch vs ttsim
======================================================================

======================================================================
TEST 1: PositionalEncoding1D
----------------------------------------------------------------------

Configuration:
  channels=256
  batch_size=2, seq_len=10

PyTorch PositionalEncoding1D:
  Output shape: (2, 10, 256)
  Output stats:
    Min:  -0.999998
    Max:  1.000000
    Mean: 0.479811
    Std:  0.519405
  Sample [0, 0, :5]: [0. 1. 0. 1. 0.]

ttsim PositionalEncoding1D:
  Output shape: Shape([2, 10, 256])
  Output .data is None? False
  Output stats:
    Min:  -0.999998
    Max:  1.000000
    Mean: 0.479811
    Std:  0.519405
  Sample [0, 0, :5]: [0. 1. 0. 1. 0.]

Numerical Comparison:
  Max absolute difference:  0.0000000000e+00
  Mean absolute difference: 0.0000000000e+00
  [PASS] Outputs match within tolerance (atol=1e-06, rtol=1e-05)

======================================================================
TEST 2: Memory Initialization
----------------------------------------------------------------------

Configuration:
  dim_in=256, number_ins=50
  bank_size=4, mem_len=4
  batch_size=2

PyTorch VectorInstanceMemory:
  mem_bank shape: torch.Size([4, 2, 150, 256])
  mem_bank_seq_id shape: torch.Size([4, 2, 150])
  mem_bank_trans shape: torch.Size([4, 2, 3])
  mem_bank_rot shape: torch.Size([4, 2, 3, 3])
  num_ins[0]: 0

ttsim VectorInstanceMemory:
  mem_bank shape: (4, 2, 150, 256)
  mem_bank_seq_id shape: (4, 2, 150)
  mem_bank_trans shape: (4, 2, 3)
  mem_bank_rot shape: (4, 2, 3, 3)
  num_ins[0]: 0

Comparison:
  [PASS] Memory initialization matches

======================================================================
TEST 3: Transformation Matrices
----------------------------------------------------------------------

Configuration:
  N=3 historical frames
  history_trans shape: (3, 3)
  history_rot shape: (3, 3, 3)
  curr_trans shape: (3,)
  curr_rot shape: (3, 3)

Input data:
  history_trans:
[[ 0.49671414 -0.1382643   0.64768857]
 [ 1.5230298  -0.23415338 -0.23413695]
 [ 1.5792128   0.7674347  -0.46947438]]
  curr_trans: [-0.9080241 -1.4123037  1.4656488]

PyTorch prepare_transformation_batch:
  curr2prev shape: (3, 4, 4)
  prev2curr shape: (3, 4, 4)
  curr2prev stats:
    Min:  -2.495635
    Max:  1.829611
    Mean: 0.102674

ttsim prepare_transformation_batch:
  curr2prev shape: (3, 4, 4)
  prev2curr shape: (3, 4, 4)
  curr2prev stats:
    Min:  -2.495635
    Max:  1.829611
    Mean: 0.102674

Numerical Comparison:
  curr2prev:
    Max diff:  1.6840362704e-07
    Mean diff: 1.7874293267e-08
  prev2curr:
    Max diff:  1.1954899137e-07
    Mean diff: 1.2542298357e-08

Transformation validity:
  PyTorch inverse check (curr2prev @ prev2curr = I): True
  ttsim inverse check (curr2prev @ prev2curr = I):   True

  [PASS] Transformation matrices match (atol=1e-05, rtol=0.0001)

======================================================================
TEST 4: Positional Encoding Cache Consistency
----------------------------------------------------------------------

Configuration:
  channels=128

Cache comparison:
  PyTorch cache shape: (1000, 128)
  ttsim cache shape: (1000, 128)
  Max cache difference:  0.0000000000e+00
  Mean cache difference: 0.0000000000e+00

Sequence length tests:
  [PASS] seq_len=5: cache match
  [PASS] seq_len=20: cache match
  [PASS] seq_len=100: cache match

  [PASS] Positional encoding cache consistent (atol=1e-06, rtol=1e-05)

======================================================================
Test Complete!
======================================================================
```

---


