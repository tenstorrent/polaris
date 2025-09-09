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
- PyTorch Code:
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

- TTSim Code:
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
