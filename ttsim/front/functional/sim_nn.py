#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

###############################################################
# Poor Man's Module/ModuleList inspired by PyTorch Signature
###############################################################
from typing import Iterator
import ttsim.front.functional.op as F
from ttsim.ops import SimTensor
from ttsim.graph import WorkloadGraph

class Module:
    # Type declarations for INSTANCE attributes
    name: str

    def __init__(self):
        self._tensors  = {}
        self._op_hndls = {}
        self._submodules  = {}

    def __setattr__(self, name, value):
        if isinstance(value, SimTensor):
            self._tensors[name] = value
        elif isinstance(value, (F.SimOpHandle, F.SplitOpHandle, F.VariadicInputOpHandle)):
            #IMPLICIT_INPUTS are not constructed till SimOpHandle::__call__
            # is executed; so, we need to do this after the __call__ is done :-(
            # For an example of this: check SplitOpHandle::__call__
            self._op_hndls[name] = value
            if hasattr(value, 'params') and len(value.params) > 0:
                for _,ptensor in value.params:
                    self._tensors[ptensor.name] = ptensor
        elif isinstance(value, F.SimOpHandleList):
            for o in value:
                self._op_hndls[o.name] = o
                if len(o.params) > 0:
                    for _,ptensor in o.params:
                        self._tensors[ptensor.name] = ptensor
        elif isinstance(value, Module):
            self._submodules[name] = value
        elif isinstance(value, ModuleList):
            for m in value:
                self._submodules[name + "." + m.name] = m
        else:
            pass
        super().__setattr__(name, value)

    def link_op2module(self):
        for _op_name, _op in self._op_hndls.items():
            _op.set_module(self)
        for k,v in self._submodules.items():
            v.link_op2module()
        return

    def create_intermediate_tensor(self, tname):
        return SimTensor({'name': self.name + "." + tname})

    def create_data_tensor(self, tname, /, data, is_param=False, is_const=False):
        return F._from_data(self.name + '.' + tname, data, is_param=is_param, is_const=is_const)

    def create_shape_tensor(self, tname, /, shape, is_param=False, is_const=False):
        return F._from_shape(self.name + '.' + tname, shape, is_param, is_const=is_const)

    def get_tensors(self, tbl):
        #v.imp note: attributes across instances have same names, so we should not
        #use the attr-name to accumulate ALL tensors across submodules...
        for k,v in self._tensors.items():
            tbl[v.name] = v
        for k,v in self._op_hndls.items():
            if len(v.implicit_inputs) > 0:
                itensor = v.implicit_inputs[0]
                tbl[itensor.name] = itensor
        for k,v in self._submodules.items():
            v.get_tensors(tbl)
        return tbl

    def get_ops(self, tbl: dict):
        #v.imp note: attributes across instances have same names, so we should not
        #use the attr-name to accumulate ALL ops across submodules...
        for k,v in self._op_hndls.items():
            tbl[v.sim_op.name] = v.sim_op
        for k,v in self._submodules.items():
            v.get_ops(tbl)
        return tbl

    def _get_forward_graph(self, input_tensors):
        # Intended to be called only from subclasses
        # Get Tensors...
        ttbl = {}
        for tname, t in input_tensors.items():
            if isinstance(t, list):
                for ii,tt in enumerate(t):
                    assert isinstance(tt, SimTensor), f"{tname}[{ii}] not a SimTensor!!\n{tt}"
                    ttbl[tt.name] = tt
            elif isinstance(t, SimTensor):
                ttbl[t.name] = t
            else:
                assert False, f"input_tensor {tname} should be an instance of (SimTensor|List[SimTensor])!!\n{t}"

        self.get_tensors(ttbl)

        #Get Ops...
        otbl: dict = {}
        self.get_ops(otbl)

        #Graph Construction...
        gg = WorkloadGraph(self.name)

        #Add Tensors to Graph...
        for _,tensor in ttbl.items():
            gg.add_tensor(tensor)

        #Add Ops to Graph...
        for _,op in otbl.items():
            gg.add_op(op)

        #Construct Graph
        gg.construct_graph()

        return gg

    def __str__(self, indent_width=0):
        indent0 = ' ' * indent_width * 4
        indent1 = ' ' * (indent_width+1) * 4
        indent2 = ' ' * (indent_width+2) * 4
        s = f"{indent0}MODULE: {self.name}\n"
        s += f"{indent0}TENSORS:\n"
        for k,v in self._tensors.items():
            s += f"{indent1}{k}:{v}\n"

        s += f"{indent0}OPS:\n"
        for k,v in self._op_hndls.items():
            s += f"{indent1}{k}:{v.sim_op}\n"
            if len(v.params) > 0:
                s += f"{indent2}PARAMS:\n"
                for _,ptensor in v.params:
                    s += f"{indent2}{ptensor}\n"
            if len(v.implicit_inputs) > 0:
                s += f"{indent2}IMPLICIT_INPUTS:\n"
                for itensor in v.implicit_inputs:
                    s += f"{indent2}{itensor}\n"

        s += f"{indent0}SUBMODULES:\n"
        for k,v in self._submodules.items():
            s += f"{indent1}{k}:\n"
            s += v.__str__(indent_width+1)
        return s

    def __call__(self, *args, **kwargs):
        # "Pure Virtual" function, should never get called.
        # Defined to ensure Module class is "callable", and ensure no static type check fails (mypy)
        raise AssertionError

class ModuleList:
    def __init__(self, modules):
        self._modules_in_list = {}

        assert len(modules) > 0, f"Empty ModuleList at construction!!"

        for i, module in enumerate(modules):
            assert module is not None, f"'None' module passed to ModuleList"
            assert isinstance(module, Module), f"{module} is not a Module subclass"
            self._modules_in_list[str(i)] = module

        #check all module names in the list are unique...
        assert len(self) == len(set(self._modules_in_list)), \
                f"Module Names in ModuleList are not unique : {[m.name for m in self._modules_in_list.values()]}!!"

    def __len__(self):
        return len(self._modules_in_list)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._modules_in_list[str(i)] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            idx = idx + len(self) if idx < 0 else idx
            if idx < 0 or idx >= len(self):
                raise IndexError(f'out-of-bound-index: {idx}')
            return self._modules_in_list[str(idx)]
        else:
            raise TypeError(f'Invalid index Type: {type(idx)}')

    def __iter__(self) -> Iterator[Module]:
        for i in range(len(self)):
            yield self[i]



    #we want to make this immutable after construction...
    # so restricting setitem / delitem / append / insert / extend
    def __setitem__(self, idx, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def __delitem__(self, idx):
        raise RuntimeError("ModuleList is immutable after construction")

    def append(self, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def extend(self, modules):
        raise RuntimeError("ModuleList is immutable after construction")

    def insert(self, index, module):
        raise RuntimeError("ModuleList is immutable after construction")

    def __call__(self, *x):
        raise RuntimeError("ModuleList is not Callable")

