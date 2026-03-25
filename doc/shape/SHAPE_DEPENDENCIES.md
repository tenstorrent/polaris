# Shape Class Dependencies

This document lists all code in the repository that depends on the `Shape` class declared in `ttsim.ops.tensor`.

## Direct Imports

### Files that directly import Shape from ttsim.ops.tensor:

1. **`ttsim/front/ttnn/tensor.py`**
   - Line 6: `from ttsim.ops.tensor import SimTensor, Shape`
   - Uses Shape directly for creating Shape objects
   - Exports Shape as `ttnn.Shape` for TTNN workloads

## Indirect Dependencies (via SimTensor)

All files that use `SimTensor` depend on Shape indirectly, since `SimTensor.shape` is a `Shape` object. Files that import or use `SimTensor`:

### Core TTSim Modules:

1. **`ttsim/ops/tensor.py`** - Defines Shape and SimTensor classes
2. **`ttsim/ops/op.py`** - Uses SimTensor, has comment about Shape type (line 82-86)
3. **`ttsim/ops/desc/helpers.py`** - Imports SimTensor, uses tensor.shape
4. **`ttsim/ops/desc/tensor.py`** - Uses SimTensor.shape extensively for shape inference
5. **`ttsim/ops/desc/reduction.py`** - Uses SimTensor
6. **`ttsim/ops/desc/nn.py`** - Uses SimTensor
7. **`ttsim/ops/desc/math.py`** - Uses SimTensor
8. **`ttsim/ops/desc/custom.py`** - Uses SimTensor
9. **`ttsim/front/ttnn/tensor.py`** - Tensor class extends SimTensor, uses Shape directly
10. **`ttsim/front/ttnn/op.py`** - Uses Tensor (which uses Shape)
11. **`ttsim/front/ttnn/ttnn_shim.py`** - Uses Shape objects extensively (15+ direct accesses to `._shape`)
12. **`ttsim/front/functional/op.py`** - Uses SimTensor
13. **`ttsim/front/functional/tensor_op.py`** - Uses SimTensor.shape
14. **`ttsim/front/functional/sim_nn.py`** - Uses SimTensor
15. **`ttsim/front/onnx/onnx2nx.py`** - Uses SimTensor
16. **`ttsim/graph/wl_graph.py`** - Uses SimTensor.shape for ONNX export
17. **`ttsim/stats/hlmstats.py`** - Uses SimTensor
18. **`ttsim/back/device.py`** - Uses SimTensor

### Workload Files (63 files total):

All workload files that create or manipulate tensors depend on Shape:

#### TTNN Workloads:
- `workloads/ttnn/vit/run_ttnn_functional_vit.py` - Uses `ttnn.Shape([...])` directly
- `workloads/ttnn/vit/ttnn_functional_vit.py` - Uses Tensor
- `workloads/ttnn/vit/test_ttnn_functional_vit.py` - Uses Tensor
- `workloads/ttnn/vgg_unet/model_preprocessing.py` - Uses `ttnn.Shape([...])` extensively
- `workloads/ttnn/resnet50/utils.py` - Uses `ttnn.Shape`
- `workloads/ttnn/resnet50/ttnn_functional_resnet50.py` - Uses Tensor
- `workloads/ttnn/llama3/*.py` - Multiple files using Tensor
- `workloads/ttnn/bert/*.py` - Uses Tensor
- `workloads/ttnn/tt_mnist.py` - Uses Tensor

#### Functional Workloads:
- `workloads/basicresnet.py` - Uses SimTensor
- `workloads/basicmlp.py` - Uses SimTensor
- `workloads/basiclenet.py` - Uses SimTensor
- `workloads/BasicLLM.py` - Uses SimTensor
- `workloads/BasicDLRM.py` - Uses SimTensor
- `workloads/Yolo_v7.py` - Uses SimTensor
- `workloads/Yolo_v8.py` - Uses SimTensor
- `workloads/EfficientViT/EfficientViT_Cls.py` - Uses SimTensor
- `workloads/EfficientViT/EfficientViT_Seg.py` - Uses SimTensor
- `workloads/LeViT/LeViT.py` - Uses SimTensor
- `workloads/Swin-Transformer/SwinTransformer.py` - Uses SimTensor
- `workloads/UNet/unet_model.py` - Uses SimTensor
- `workloads/UNet/unet_parts.py` - Uses SimTensor
- `workloads/UNet/test_unet.py` - Uses SimTensor
- `workloads/llama2/model.py` - Uses SimTensor
- `workloads/llama2/test_model.py` - Uses SimTensor
- `workloads/llm/transformer_model.py` - Uses SimTensor
- `workloads/llm/rope.py` - Uses SimTensor
- `workloads/llm/attention.py` - Uses SimTensor
- `workloads/mamba2/ttsim_mamba2_simple.py` - Uses SimTensor
- `workloads/mamba2/test_ttsim_mamba2_simple.py` - Uses SimTensor
- `workloads/diffusers/*.py` - Multiple files (15+ files) using SimTensor
- `workloads/bevdepth/*.py` - Multiple files using SimTensor

### Test Files:

1. **`tests/test_datatype_bytes.py`** - Imports SimTensor
2. **`tests/test_ops/test_activations.py`** - Uses make_tensor
3. **`tests/test_ops/test_batchnorm.py`** - Imports SimTensor
4. **`tests/test_ops/test_batchnorm_new.py`** - Uses make_tensor
5. **`tests/test_ops/test_conv.py`** - Uses make_tensor
6. **`tests/test_ops/test_conv_and_bn.py`** - Uses SimTensor
7. **`tests/test_ops/test_dropout.py`** - Uses make_tensor
8. **`tests/test_ops/test_eltwisebinary.py`** - Uses make_tensor
9. **`tests/test_ops/test_eltwiseunary.py`** - Uses make_tensor
10. **`tests/test_ops/test_gather.py`** - Uses make_tensor
11. **`tests/test_ops/test_layernorm.py`** - Uses make_tensor
12. **`tests/test_ops/test_matmul.py`** - Uses make_tensor
13. **`tests/test_ops/test_maxpool2d.py`** - Imports SimTensor
14. **`tests/test_ops/test_pooling.py`** - Uses make_tensor
15. **`tests/test_ops/test_reductions.py`** - Uses make_tensor
16. **`tests/test_ops/test_reshape.py`** - Uses make_tensor
17. **`tests/test_ops/test_split.py`** - Uses make_tensor
18. **`tests/test_ops/test_transpose.py`** - Uses make_tensor
19. **`tests/test_workloads/test_yolov7.py`** - Uses SimTensor
20. **`tests/test_workloads/test_yolov8.py`** - Uses SimTensor
21. **`tests/test_back/test_memory_bandwidth.py`** - Imports SimTensor
22. **`tests/common.py`** - Uses SimTensor

### Tools:

1. **`tools/run_onnx_shape_inference.py`** - Uses SimTensor

## Direct Shape Usage Patterns

### 1. Direct Shape Creation:
- `ttsim/front/ttnn/tensor.py` - Creates Shape objects
- `workloads/ttnn/vit/run_ttnn_functional_vit.py` - Uses `ttnn.Shape([...])`
- `workloads/ttnn/vgg_unet/model_preprocessing.py` - Uses `ttnn.Shape([...])` extensively
- `workloads/ttnn/resnet50/utils.py` - Uses `ttnn.Shape([...])`

### 2. Shape Property Access:
All files that access `tensor.shape` depend on Shape:
- `ttsim/ops/desc/tensor.py` - Extensive use of `tensor.shape`
- `ttsim/front/ttnn/tensor.py` - Uses `self.shape`
- `ttsim/front/ttnn/ttnn_shim.py` - Uses `tensor.shape` and `tensor.logical_shape()`
- `ttsim/front/functional/tensor_op.py` - Uses `shape` parameter
- `ttsim/graph/wl_graph.py` - Uses `tval.shape` for ONNX export

### 3. Shape Manipulation:
Files that manipulate Shape objects directly:
- `ttsim/front/ttnn/tensor.py` - Slicing, concatenation with Shape
- `ttsim/front/ttnn/ttnn_shim.py` - Direct access to `._shape` (needs fixing)
- `ttsim/ops/desc/tensor.py` - List operations on Shape
- `ttsim/front/functional/tensor_op.py` - List concatenation with Shape

## Files with Critical Dependencies

These files have direct dependencies that may break with the Shape class change:

### High Priority (Direct Shape Operations):
1. **`ttsim/front/ttnn/tensor.py`** - Direct Shape import, Shape operations
2. **`ttsim/front/ttnn/ttnn_shim.py`** - 15+ direct `._shape` accesses
3. **`ttsim/ops/desc/tensor.py`** - Extensive shape manipulation
4. **`ttsim/front/functional/tensor_op.py`** - Shape concatenation

### Medium Priority (Shape Property Usage):
1. **`ttsim/graph/wl_graph.py`** - Shape iteration for ONNX
2. **`ttsim/ops/op.py`** - Shape type conversion
3. All shape inference functions in `ttsim/ops/desc/*.py`

### Low Priority (Indirect via SimTensor):
- All workload files (they use SimTensor but may not directly manipulate Shape)
- All test files (they use make_tensor/SimTensor but typically don't manipulate Shape directly)

## Summary Statistics

- **Direct imports of Shape**: 1 file (`ttsim/front/ttnn/tensor.py`)
- **Files using SimTensor**: ~63 files
- **Files with direct Shape manipulation**: ~10 files
- **Files with critical Shape dependencies**: 4 files (see High Priority above)
- **Total files depending on Shape (directly or indirectly)**: ~100+ files

## Notes

1. Most files depend on Shape indirectly through `SimTensor.shape` property
2. The Shape class implements `__iter__`, `__getitem__`, `__len__`, and `__eq__`, so many operations work transparently
3. Files that directly access `._shape` or use list operations on Shape need updates
4. `ttnn.Shape` in workloads is actually `ttsim.ops.tensor.Shape` imported through `ttsim.front.ttnn.tensor`
