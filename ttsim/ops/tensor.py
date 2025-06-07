#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import functools, operator
import numpy as np

class SimTensor:
    def __init__(self, cfg):
        self.name        = cfg['name']                # String
        self.shape       = cfg.get('shape')           # List
        self.dtype       = cfg.get('dtype')           # Numpy datatype 
        self.data        = cfg.get('data', None)      # Actual data (numpy array)
        self.resolve     = cfg.get('resolve','_')     # Has the tensor shape been resolved (intermediate tensor shapes) (Boolean)
        self.op_in       = cfg.get('op_in', [])       # Which operators is this "input" for (consumer list)
        self.op_out      = cfg.get('op_out', [])      # Which operators is this "output" of (producer list)
        self.is_param    = cfg.get('is_param', False) # Is it parameter? Boolean
        self.is_const    = cfg.get('is_const', False) # Is it constant? Boolean
        self.has_grad    = cfg.get('has_grad', True)  # Has a gradient during bwd pass? Boolean
        self.link_module = None                       # Associated Module

    def set_module(self, m): self.link_module = m

    def __str__(self):
        s  = f"SimTensor({self.name}) shape={self.shape}, dtype={self.dtype}, "
        s += f"is_param={self.is_param}, "
        s += f"is_const={self.is_const}, "
        s += f"has_grad={self.has_grad}, "
        s += f"op_in={self.op_in}, "
        s += f"op_out={self.op_out}, "
        if self.data is None:
            s += f"data={self.data}"
        elif self.rank() > 0 and self.nelems() > 5:
            s += "data=(...)"
        else:
            s += f"data={self.data.tolist()}"
        return s

    def rank(self): return len(self.shape)

    # Note: data count may not be a simple product of shape dims - may need to provide a custom func
    def nelems(self):
        trank = self.rank()
        if trank > 0:
            res = functools.reduce(operator.mul, (k for k in self.shape), 1)
        elif trank == 0:
            res = 1
        else:
            assert False, f"What kinda tensor {self.name} is this? {self.shape}"
        if self.data is not None:
            assert isinstance(self.data, tuple([np.ndarray, np.float32, np.bool_])), f'data should be ndarray, is {type(self.data)}'
            res1 = self.data.size
            assert res1 == res, f"Mismatch SimTensor({self.name}).nelems = {res} and np.size={res1}"
        return res

    # Note:
    #   data size may not be just data-count * precision, because you may have compression/sparsity
    #   how is the tensor stored in memory? channel first, batch later or something else. may have
    #   to represent tiling formats here.
    def nbytes(self):
        return self.nelems() * self.dtype.itemsize #assumes np.dtype

    def check_shape(self):
        if self.shape is None:
            return False
        elif all([ isinstance(d, int) or isinstance(d, np.int64) for d in self.shape]):
            return True
        else:
            return False

def make_tensor(name: str) -> SimTensor:
    return SimTensor({'name': name, 'shape': [], 'dtype': None})

