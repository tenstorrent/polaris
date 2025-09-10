# Generic PyTorch → TTSim porting guide

This guide illustrates how to change standard PyTorch modules so that they become TTSIM API compliant, and can therefore be simulated with Polaris

## Project setup and imports

TTSIM API provides a proxy to torch compliant operators via `ttsim.front.functional.op` module.
Similarly, torch `nn.module` proxy is implemented via `ttsim.front.functional.sim_nn` module

Hence the first change is to replace the following lines (as applicable):

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

to

```Python
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
```

This should work whenever you create all the necessary operators and tensors in the constructor
(`__init__` function) inside a torch.nn module. However, there are many situations, where operators
and tensors are created dynamically, inside the forward function. An example is that the batch size
is usually passed on via the input tensor shape at the `forward` function. In those cases, we need
to add the following import:

```Python
import ttsim.front.functional.tensor_op as T
```

Lastly, we use numpy as a backend - so based on need (e.g., creating a tensor of certain shape), you
may have to include that in your imports as well

## Base class and call semantics

- These translate directly, and therefore are quite simple:
    - `nn.Module` → `SimNN.Module`
    - `forward(...)` → `__call__(...)`.

- For TTSIM API, we insist that every SimNN.Module, operator and tensor is given a unique, semantically meaningful
  name. This greatly helps during debug. Therefore, we pass an object-name parameter during
  construction time:
    - Constructor signature includes `objname: str`
    - Set `self.name = objname`

- After constructing ops in a module, we need to associate the operator-module linkage, which is achieved via the
  `SimNN.Module::link_op2module` method. Hence, usually the last statement for Modules is a call: `super().link_op2module()`

## Core Modules/Operators Examples

| PyTorch              |    TTSIM                  |
|---------------------:|---------------------------|
| `nn.Sequential(...)` | `F.Sequential(name, ...) `|
| `nn.Conv2d(...)`     | `F.Conv2d(name, ...)`     |
| `nn.BatchNorm2d(...)`| `F.BatchNorm2d(name, ...)`|
| `nn.ReLU(...)`       | `F.Relu(name, ...)`       |
| `nn.MaxPool2d(...)`  | `F.MaxPool2d(name, ...)`  |
| `nn.Upsample(...)`   | `F.Upsample(name, ...)`   |
| `nn.Linear(...)`     | `F.Linear(name, ...)`     |
| `nn.Embedding(...)`  | `F.Embedding(name, ...)`  |

## Shapes, views, and broadcasting
- Use `x.shape` (NCHW for CNN; [B, T, ...] for Transformer).
- CNN padding order in TT‑Sim is `[left, right, top, bottom]`.
- View/reshape:
    - PyTorch: `x.view(...)`, `x.reshape(...)`
    - TTSim: `x.view(...)` or `x.reshape(...)` on `SimTensor` (as supported).
- Broadcasting helpers: write a `reshape_for_broadcast(...)` that asserts expected shapes and returns a properly viewed tensor.

## Naming and uniqueness
- Every TTSim op requires a unique name. Derive sub-op names from `self.name`:
    - Examples: `f'{self.name}_conv1'`, `..._bn1'`, `..._relu1'`, `..._matmul'`, `..._softmax'`
- Pass contextual `objname` from parents to children, e.g., `DoubleConv(f'{self.name}_conv', ...)`.

## Example pattern
PyTorch Code:
```
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```

TTSIM Code:
```
class DoubleConv(SimNN.Module):
    def __init__(self, objname, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.name = objname
        self.double_conv = F.Sequential(f'{self.name}_seq',
            F.Conv2d(f'{self.name}_conv1', in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            F.BatchNorm2d(f'{self.name}_bn1', mid_channels),
            F.Relu(f'{self.name}_relu1'),
            F.Conv2d(f'{self.name}_conv2', mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            F.BatchNorm2d(f'{self.name}_bn2', out_channels),
            F.Relu(f'{self.name}_relu2'),
        )
        super().link_op2module()

    def __call__(self, x):
        return self.double_conv(x)
```

## Behavioral parity notes
- Match kernel sizes, paddings, strides, and bias flags.
- Replace unavailable activations (e.g., `SiLU`) with supported ones (e.g., `Relu`) if acceptable; otherwise implement via primitives.
- Validate concat dimensions (channel dim is `axis=1` for NCHW).

## Common pitfalls
- Missing unique names for ops → graph errors.
- Forgetting `forward` → `__call__`.
- Incorrect pad order `[l, r, t, b]`.
- Creating raw Python lists where tensors are required; use `F._from_data/F._from_shape`.
- Not linking ops to modules when needed (`super().link_op2module()`).
- Broadcasting shape mismatches; assert expected shapes in helpers.

## Validating tensor shapes
- Dump the tensor shapes from PyTorch and TTSim execution to validate if they match.
```
import sys
import inspect

def print_tensor_shapes(func):
    def tracer(frame, event, arg):
        if event == 'return' and frame.f_code == func.__code__:
            func_name = frame.f_code.co_name
            locals_at_return = frame.f_locals
            print(f"[DEBUG tensors inside '{func_name}']:")
            for name, val in locals_at_return.items():
                if hasattr(val, "shape"):
                    try:
                        shape = tuple(val.shape)
                    except Exception:
                        shape = val.shape
                    print(f"  {name}: shape={shape}, type={type(val).__name__}")
            print()
        return tracer

    def wrapper(*args, **kwargs):
        old_tracer = sys.gettrace()
        sys.settrace(tracer)
        try:
            return func(*args, **kwargs)
        finally:
            sys.settrace(old_tracer)
    return wrapper
...

class Attention(nn.Module):
...
    @print_tensor_shapes
    def forward(
        self,
        x: torch.Tensor,
        ...)
```

## Adding new operators to TTSIM

If your workload has an operator which is not implemented in TTSIM, you need to do the following steps:

**1) Check the API signature of the PyTorch operator**
  - Find the equivalent ONNX operator(s). Reference: https://onnx.ai/onnx/operators/. This may or
    may not be a simple 1:1 mapping. Pay attention to the attributes mismatch between Pytorch and
    ONNX

**2) Implement or reuse a `SimOp` class**
  - Check if the equivalent ONNX operator(s) are missing in Polaris operators at `ttsim/ops/op.py`
  - If the operator(s) are missing, you need to implement them as derived `SimOp` classes. Minimal
    requirement for a such a class is input/output tensor count checks, attribute checks, shape
    inference and precision checks. The `get_perf_counts` method needs to implement the shape inference
    and instruction/memory-operation estimates for operator execution
  - On the other hand, the equivalent operator may already be there, in which case we can simply
    reuse it

**3) Register the op_type in the SimOp factory**
  - Map the ONNX `op_type` string to your `SimOp` in the factory so ONNX import can construct it:

**4) Add a functional wrapper so that the PyTorch equivalent operator has a TTSIM proxy**
  - Provide a `ttsim.front.functional` wrapper so workloads can use the op in Python modules.

**5) Use the TTSIM functional wrapper in your workload code**
  - Create the op handle during module init and call it with the required inputs:

Here is an example. `Conv2d` operator in PyTorch has an ONNX equivalent `Conv`. However, their
attribute names mismatch. We first implement and register the ONNX compliant operator as a `SimOp`
in `ttsim/ops/op.py`:

```Python
class ConvOp(SimOp):
    def __init__(self, opinfo):
        super().__init__(opinfo)
        self.opclass_str: str = 'Conv'

        #perform IO checks
        check_io_counts(self, in_counts=[2,3], out_counts=[1,1])
        ...

    def get_perf_counts(self, inT, outT, **kwargs):
        #check input tensor shapes, and do shape inference for output tensors
        assert inT[0].check_shape(), f"Illegal Shape for {inT[0]}"
        assert inT[1].check_shape(), f"Illegal Shape for {inT[1]}"
        if len(inT) == 3: assert inT[2].check_shape(), f"Illegal Shape for {inT[2]}"
        ...

def SimOpFactory(optype: str) -> type[SimOp]:
    cls2optype: Dict[type[SimOp], list[str]] = {
            ...
            ConvOp : ['Conv'],
            ...

```

Next we provide a functional wrapper in `ttsim/front/functional/op.py`. Notice how the attribute
translation is done to ensure PyTorch to ONNX compatibility:

```Python

def Conv2d(name, in_channels, out_channels, kernel_size, **kwargs):
    kernel_dims = (kernel_size, kernel_size)
    arg_defaults = {
            'stride': 1, 'padding': 0, 'dilation': 1, 'groups': 1,
            'bias': True, 'padding_mode': 'zeros', 'device': None, 'dtype': None
            }
    eff_args   = common.get_kwargs_with_defaults('Conv', args=kwargs, default_args=arg_defaults)
    stride     = common.make_tuple(eff_args['stride'], 2)
    padding    = common.make_tuple(eff_args['padding'], 2*2)
    dilation   = common.make_tuple(eff_args['dilation'], 2)
    param_dims = [out_channels, in_channels // eff_args['groups'], *kernel_dims]
    conv_param = _from_shape(name+'.param', param_dims, is_param=True)
    op_hndl = SimOpHandle(name, 'Conv', params=[(1, conv_param)], ipos=[0],
                          group=eff_args['groups'], # Torch to ONNX mapping
                          strides=stride,           # Torch to ONNX mapping
                          pads=padding,             # Torch to ONNX mapping
                          dilations=dilation,       # Torch to ONNX mapping
                          )
    return op_hndl
```

So, now in your workload, you can write something like this:

```Python
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

class SimpleModule(SimNN.Module):
    def __init__(self, name, **kwargs):
        self.conv_op = F.Conv2d(self.name + f'.conv', ic, oc, kernel_size=k, padding=p, stride=s)
        ...

    def __call__(self, x: SimTensor):
        return self.conv_op(x)
```

Tips:
- **Naming**: SimOp `opclass_str` string must exactly match what you register in the factory.
- **Input Counts**: Choose the correct functional wrapper (`UnaryOperator`, `BinaryOperator`, `TernaryOperator`, or `VariadicInputOpHandle`) so input ordering matches your `SimOp`.
- **Complex Mappings**: If the PyTorch to ONNX is complex, and involves several operators, use a
  `SimNN.Module` instead of `SimOpHandle`
- **Custom Operators**: Some workloads implement custom operators for efficiency or propriety
  reasons. It is best to discuss this with Polaris team before embarking on coding/development to
  support such workloads

## ONNX model graph generation:

Exporting a TTSIM model to ONNX involves three steps: prepare inputs, run the forward to materialize the graph, then export.

**1) Prepare input tensors**
  - Define a `create_input_tensors` that constructs named inputs with `F._from_shape` (or `F._from_data`).
```
def create_input_tensors(self):
    self.input_tensors = {
            'x_in': F._from_shape('x_in', ...),
            }
    return
```

**2) Run the model to connect ops and tensors**
  - Build the model, create inputs, and execute `__call__` to connect handles.
```
rn_model = ResNet(k,v)
rn_model.create_input_tensors()
y = rn_model()
print('Input:    ', rn_model.input_tensors['x_in'].shape)
print('Output:   ', y.shape)
```

**3) Construct graph and export ONNX**
  - Obtain the forward graph and write ONNX with optional checker.
```
def get_forward_graph(self):
    GG = super()._get_forward_graph(self.input_tensors)
    return GG
```
```
gg = rn_model.get_forward_graph()
print('Dumping ONNX...')
gg.graph2onnx(f'resnet50.onnx', do_model_check=True)
```

Notes:
- Ensure `super().link_op2module()` is called after constructing ops in `__init__` so they register under the module.
- The keys of `self.input_tensors` become the graph inputs; names must be unique and match your module’s expected inputs.
- You can also build a tensor map on the fly and call `Module._get_forward_graph({...})` directly when inputs are created outside the module.
