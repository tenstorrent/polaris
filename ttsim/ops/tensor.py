#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import functools, operator
import numpy as np
from ttsim.utils.types import get_bpe, get_sim_dtype

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
        if self.link_module is not None:
            s += f", link_module={self.link_module.name}"
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
    # Note: Caching nbytes for instance methods can cause memory leaks due to references held by lru_cache.
    # If caching is needed, consider using a static cache or external memoization.
    def nbytes(self, itemprec=None):
        def typesize(dtype):
            if isinstance(dtype, np.dtype):
                return dtype.itemsize
            elif isinstance(dtype, str):
                return get_bpe(get_sim_dtype(dtype))
            else:
                raise TypeError(f"Unsupported dtype type: {type(dtype)}")
        if itemprec is None:
            assert self.dtype is not None, f"SimTensor({self.name}) has no dtype to calculate nbytes"
            itemsize = typesize(self.dtype)
        else:
            itemsize = typesize(itemprec)
        return self.nelems() * itemsize #assumes np.dtype

    def check_shape(self):
        if self.shape is None:
            return False
        elif all([ isinstance(d, int) or isinstance(d, np.int64) for d in self.shape]):
            return True
        else:
            return False

    def clone(self, clone_num:int):
        cloned_tensor = make_tensor(self.name + '.clone_{clone_num}')
        cloned_tensor.shape       = self.shape
        cloned_tensor.dtype       = self.dtype
        cloned_tensor.data        = self.data
        cloned_tensor.resolve     = self.resolve
        cloned_tensor.op_in       = self.op_in
        cloned_tensor.op_out      = self.op_out
        cloned_tensor.is_param    = self.is_param
        cloned_tensor.is_const    = self.is_const
        cloned_tensor.has_grad    = self.has_grad
        cloned_tensor.link_module = self.link_module
        return cloned_tensor

    def transpose(self, dim0=None, dim1=None, perm=None):
        """
        Transpose tensor dimensions
        
        Args:
            dim0, dim1: Two dimensions to swap (for 2D transpose)
            perm: Permutation of dimensions (for multi-dimensional)
        
        Returns:
            SimTensor: Transposed tensor
        """
        # Import here to avoid circular imports
        import ttsim.front.functional.op as F
        
        if perm is not None:
            # Multi-dimensional transpose
            transpose_op = F.Transpose(f"{self.name}.transpose", perm=perm)
        elif dim0 is not None and dim1 is not None:
            # 2D transpose - swap two dimensions
            perm = list(range(len(self.shape)))
            perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
            transpose_op = F.Transpose(f"{self.name}.transpose", perm=perm)
        else:
            # Default: transpose last two dimensions (like PyTorch .T)
            if len(self.shape) < 2:
                return self
            perm = list(range(len(self.shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            transpose_op = F.Transpose(f"{self.name}.transpose", perm=perm)
        
        # Set module context if available
        if self.link_module is not None:
            transpose_op.set_module(self.link_module)
        
        return transpose_op(self)
    
    def reshape(self, *shape):
        """Reshape tensor to new shape"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        # Import here to avoid circular imports
        import ttsim.front.functional.op as F
        
        reshape_op = F.ReshapeFixed(f"{self.name}.reshape", list(shape))
        
        # Set module context if available
        if self.link_module is not None:
            reshape_op.set_module(self.link_module)
            
        return reshape_op(self)
    
    def squeeze(self, dim=None):
        """Remove dimensions of size 1"""
        # Import here to avoid circular imports
        import ttsim.front.functional.op as F
        
        if dim is not None:
            # Squeeze specific dimension
            if self.shape[dim] != 1:
                raise ValueError(f"Cannot squeeze dim {dim} with size {self.shape[dim]} != 1")
            squeeze_axes = [dim]
        else:
            # Squeeze all dimensions of size 1
            squeeze_axes = [i for i, size in enumerate(self.shape) if size == 1]
            if not squeeze_axes:
                return self
        
        axes_tensor = F._from_data(f"{self.name}.squeeze_axes", 
                                  np.array(squeeze_axes, dtype=np.int64), is_const=True)
        
        # Set module context if available
        if self.link_module is not None:
            axes_tensor.set_module(self.link_module)
        
        squeeze_op = F.Squeeze(f"{self.name}.squeeze")
        if self.link_module is not None:
            squeeze_op.set_module(self.link_module)
        
        return squeeze_op(self, axes_tensor)
    
    def unsqueeze(self, dim):
        """Add dimension of size 1 at specified position"""
        # Import here to avoid circular imports
        import ttsim.front.functional.op as F
        
        # Calculate new shape
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        
        reshape_op = F.ReshapeFixed(f"{self.name}.unsqueeze", new_shape)
        
        # Set module context if available
        if self.link_module is not None:
            reshape_op.set_module(self.link_module)
        
        return reshape_op(self)

def make_tensor(name: str) -> SimTensor:
    return SimTensor({'name': name, 'shape': [], 'dtype': None})
