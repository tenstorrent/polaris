#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

class SimOpDescRegistry:
    def __init__(self):
        self._registry = {}
        return

    def register(self, **kwargs):
        opname = kwargs['opname']
        if opname not in self._registry:
            self._registry[opname] = {}
        self._registry[opname]= kwargs
        return

    def get_opdesc(self, opname):
        try:
            return self._registry[opname]
        except KeyError:
            raise KeyError(f"{opname} not supported in SimOpDescRegistry!!")

    def get_shape_inference_function(self, opname):
        assert opname in self._registry, f"operator= {opname} is not registered with SimOpDescRegistry"
        return self._registry[opname]['shape_inf_func']

    def has_shape_inference_function(self, opname) -> bool:
        return self.get_shape_inference_function(opname) is not None

# Global registry instance
_global_registry: Optional[SimOpDescRegistry] = None

def get_opdesc_registry() -> SimOpDescRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = SimOpDescRegistry()
    return _global_registry

def register_ops(group, optbl):
    op_fields = [
            'opname', 'arity_class', 'domain', 'support_level',
            'version', 'since_version', 'max_input', 'min_input', 'max_output',
            'min_output', 'shape_inf_func', 'is_deprecated', 'has_attr',
            'has_function', 'has_function_template', 'has_context_dependent_function',
            ]
    for rec in optbl:
        cfg = {f: v for f,v in zip(op_fields, rec)}
        cfg['group'] = group
        get_opdesc_registry().register(**cfg)
