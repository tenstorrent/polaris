import torch
import numpy as np
import ttsim.front.functional.op as F
from ttsim.ops import SimOpFactory, SimTensor
from polaris import execute_simophandle

TILE_LAYOUT = 0
L1_MEMORY_CONFIG = 1
float32 = np.float32
bfloat16 = np.float16
DRAM_MEMORY_CONFIG = 2
ROW_MAJOR_LAYOUT = 0

def open_device(device_id=0, device_type='ttnn', device_name=None):
    # empty function to match the interface
    return

def close_device(device_id=0, device_type='ttnn', device_name=None):
    # empty function to match the interface
    return

def full(shape, fill_value, dtype, layout=None, device=None, memory_config=None, name=None):
    # Create a SimTensor with the specified shape
    return SimTensor({
        'name': name,
        'shape': list(shape),
        'dtype': dtype})

def ones(shape, dtype, layout=None, device=None, memory_config=None, name=None):
    # Create a SimTensor with the specified shape
    return SimTensor({
        'name': name,
        'shape': list(shape),
        'dtype': dtype})

zeros = ones

def Tensor(data, device=None, layout=None, name=None):
    data_type = data.dtype
    return SimTensor({
        'name': name,
        'shape': list(data.shape),
        'dtype': data_type
        })

def relu(x, name='Relu'):
    op_handl = F.Relu(name)
    d = execute_simophandle(op_handl, [x])
    return d

def linear(x, w, bias, name='Linear'):
    M, N = w.shape
    op_handl = F.Linear(name, M, N)
    d = execute_simophandle(op_handl, [x])
    return d

# add kwargs to match extra arguments
def add(x, y, name='AddOp'):
    op_handl = F.Add(name, params = [(0,x)], ipos=[1])
    d = execute_simophandle(op_handl, [y])
    return d

def mul(x, y, name='MulOp'):
    op_handl = F.Mul(name, params = [(0,x)], ipos=[1])
    d = execute_simophandle(op_handl, [y])
    return d

def matmul(x, y, memory_config=DRAM_MEMORY_CONFIG, name='MatMulOp'):
    op_handl = F.MatMul(name, params = [(0,x)], ipos=[1])
    d = execute_simophandle(op_handl, [y])
    return d

def to_layout(tt_tensor, layout):
    # empty function to match the interface
    return tt_tensor

def to_torch(tt_tensor, dtype=None, device=None):
    # empty function to match the interface
    return tt_tensor

def from_torch(torch_tensor, dtype=None, layout=None, device=None, name=None):
    np_data = torch_tensor.detach().cpu().numpy()
    return F._from_data(name, np_data)

def from_torch_var(name, torch_tensor, is_param=False, is_const=False):
    np_data = torch_tensor.detach().cpu().numpy()
    return F._from_data(name, np_data, is_param=is_param, is_const=is_const)