#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.functional.sim_nn as sim_nn
from   ttsim.ops.tensor import SimTensor
from typing import Union

class SimOpTester(sim_nn.Module):
    """
        Base Class for building unit tests of SimOps.
        Usually used for testing a single SimOp.
        But, could be used to test a sequence as well, for simple networks
        Used by most test in test_ops/
    """
    def __init__(self, name: str, cfgentry: dict):
        super().__init__()

        self.name = name
        self.input_tensors: Union[dict[str, SimTensor], None] = None
        self.bs: Union[int, None] = None

        self.__setup__(cfgentry)

        super().link_op2module()

    def forward_graph(self):
        """
            It is *NOT* named get_forward_graph, because:
            * it is defined in the base class
            * with a different signature
        """
        fwd_graph = super()._get_forward_graph(self.input_tensors)
        return fwd_graph

    def set_batch_size(self, batch):
        self.bs = batch

    def __setup__(self, cfgentry: dict):
        raise NotImplementedError('__setup__ must be implemented in the instance')

    def create_input_tensors(self):
        raise NotImplementedError('create_input_tensors must be implemented in the instance')

    def __call__(self):
        raise NotImplementedError('call must be implemented in the instance')

    def analytical_param_count(self) -> int:
        return 10 + 1 if self.bs is None else 0  # Use self.bs to avoid "method may be static" warnings

