#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

#this script is inspired by onnx/defs/gen_doc.py
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ttsim.utils.common import print_csv, print_yaml
from ttsim.ops import SimTensor
from ttsim.ops.desc.registry import get_opdesc_registry

# we get the following annoying warning with onnx:
# onnx/backend/test/case/node/bitwisexor.py:42: RuntimeWarning: invalid value encountered in cast
# using filterwarnings below to supress this!!
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Tuple, Iterable, Any
from collections import defaultdict

import onnx
from onnx import defs, helper
from onnx.defs import OpSchema
from onnx.backend.sample.ops import collect_sample_implementations
from onnx.backend.test.case import collect_snippets

#helper variable for printing
PFX = " "*4

#helper functions
def get_domain(_dom: str) -> str:
    return 'ai.onnx' if _dom == "" else _dom

def format_attr_value(value: Any) -> str:
    if isinstance(value, float):
        formatted = str(f"{value:e}")
    elif isinstance(value, (bytes, bytearray)):
        formatted = str(value.decode('utf-8'))
    else:
        formatted = str(value)
    return formatted

def get_attr_defval(_attr):
    if _attr.default_value.name:
        val = helper.get_attribute_value(_attr.default_value)
        val = [format_attr_value(v) for v in val] if isinstance(val, list) else format_attr_value(val)
    else:
        val = None
    return val

def get_support_level(_support: OpSchema.SupportType) -> str:
    if _support == OpSchema.SupportType.EXPERIMENTAL:
        res = "EXPERIMENTAL"
    elif _support == OpSchema.SupportType.COMMON:
        res = "COMMON"
    else:
        raise ValueError(f"Unknown Support Level: {_support}")
    return res

#classes
class OpType:
    def __init__(self, optype):
        self.type_param     = optype.type_param_str
        self.allowedTypes   = optype.allowed_type_strs
        self.desc           = optype.description
        return

    def __str__(self):
        ostr  = f"{PFX*2}type_param  : {self.type_param}\n"
        ostr += f"{PFX*2}allowedTypes:\n"
        for x in self.allowedTypes:
            ostr += f"{PFX*4}{x}\n"
        ostr += f"{PFX*2}desc: {self.desc}\n"
        return ostr

class OpParam:
    def __init__(self, param: OpSchema.FormalParameter):
        self.name               = param.name
        self.type               = param.type_str
        self.optional           = False
        self.variadic           = False
        self.homogeneous        = False
        self.differentiable     = False
        self.non_differentiable = False
        self.desc               = param.description

        if param.option == OpSchema.FormalParameterOption.Optional:
            self.optional = True

        if param.option == OpSchema.FormalParameterOption.Variadic:
            self.variadic = True

        if param.is_homogeneous:
            self.homogeneous = True

        if param.differentiation_category == OpSchema.DifferentiationCategory.Differentiable:
            self.differentiable = True

        if param.differentiation_category == OpSchema.DifferentiationCategory.NonDifferentiable:
            self.non_differentiable = True

        return

    def __str__(self):
        ostr  = f"{PFX*2}name:               {self.name              }\n"
        ostr += f"{PFX*2}type:               {self.type              }\n"
        ostr += f"{PFX*2}optional:           {self.optional          }\n"
        ostr += f"{PFX*2}variadic:           {self.variadic          }\n"
        ostr += f"{PFX*2}homogeneous:        {self.homogeneous       }\n"
        ostr += f"{PFX*2}differentiable:     {self.differentiable    }\n"
        ostr += f"{PFX*2}non_differentiable: {self.non_differentiable}\n"
        ostr += f"{PFX*2}desc:               {self.desc              }\n"
        return ostr

class OpAttrib:
    def __init__(self, attr):
        self.name          = attr.name
        self.type          = attr.type
        self.desc          = attr.description
        self.required      = True if attr.required else False
        self.default_value = get_attr_defval(attr)
        self.doc           = attr.default_value.doc_string
        return

    def __str__(self):
        ostr  = f"{PFX*2}name: {self.name}\n"
        ostr += f"{PFX*2}type: {self.type}\n"
        ostr += f"{PFX*2}desc: {self.desc}\n"
        ostr += f"{PFX*2}required: {self.required}\n"
        ostr += f"{PFX*2}default_value: {self.default_value}\n"
        ostr += f"{PFX*2}doc_str: {self.doc}\n"
        return ostr

class OpRecord:
    def __init__(self, schema: OpSchema, _SNIPPETS, _SAMPLES):
        self.name                           = schema.name
        self.domain                         = get_domain(schema.domain)
        self.deprecated                     = schema.deprecated
        #check if these attributes are valid....
        self.has_function                   = getattr(schema, "has_function", False)
        self.has_function_template          = getattr(schema, "has_function_template", False)
        self.function_body                  = getattr(schema, "function_body", None)
        self.has_context_dependent_function = getattr(schema, "has_context_dependent_function", None)
        self.all_function_opset_versions    = getattr(schema, "all_function_opset_versions", None)

        self.max_input                      = schema.max_input
        self.min_input                      = schema.min_input
        self.max_output                     = schema.max_output
        self.min_output                     = schema.min_output
        self.since_version                  = schema.since_version
        self.support_level                  = get_support_level(schema.support_level)
        self.attributes                     = [OpAttrib(a) for _,a in schema.attributes.items()] if schema.attributes else None
        self.inputs                         = [OpParam(in_) for in_ in schema.inputs] if schema.inputs else None
        self.outputs                        = [OpParam(out_) for out_ in schema.outputs] if schema.outputs else None
        self.allowed_types                  = [OpType(t_) for t_ in schema.type_constraints] if schema.type_constraints else None
        self.doc                            = schema.doc
        self.code_snippet                   = _SNIPPETS.get(schema.name, None)
        self.sample_implementation          = _SAMPLES.get(schema.name.lower(), None)
        return

    def summary(self) -> Dict[Any, Any]:
        res = dict(
                name                           = self.name,
                domain                         = self.domain,
                deprecated                     = self.deprecated,
                has_function                   = self.has_function,
                has_function_template          = self.has_function_template,
                function_body                  = self.function_body,
                has_context_dependent_function = self.has_context_dependent_function,
                all_function_opset_versions    = self.all_function_opset_versions,
                max_input                      = self.max_input,
                min_input                      = self.min_input,
                max_output                     = self.max_output,
                min_output                     = self.min_output,
                since_version                  = self.since_version,
                support_level                  = self.support_level,
                )
        in_arity  = self.min_input if self.min_input == self.max_input else float('inf')
        out_arity = self.min_output if self.min_output == self.max_output else float('inf')
        var_max_0 = self.max_input if self.max_input < 1000 else '*'
        var_max_1 = self.max_output if self.max_output < 1000 else '*'

        op_arity_class  = None
        if in_arity != float('inf') and out_arity != float('inf'):
            #fixed in/out arity
            op_arity_class = f'ARITY_{in_arity}->{out_arity}'
        elif out_arity != float('inf'):
            #fixed out arity, variadic in arity
            var_max = self.max_input if self.max_input < 1000 else '*'
            op_arity_class = f'ARITY_VARIADIC[{self.min_input}-{var_max_0}]->{out_arity}'
        elif in_arity != float('inf'):
            #fixed in arity, variadic out arity
            var_max = self.max_output if self.max_output < 1000 else '*'
            op_arity_class = f'ARITY_{in_arity}->VARIADIC[{self.min_output}-{var_max_1}]'
        else:
            #variadic in/out arity
            op_arity_class = f'ARITY_VARIADIC[{self.min_input}-{var_max_0}]->VARIADIC[{self.min_output}-{var_max_1}]'

        res['in_arity']  = in_arity
        res['out_arity'] = out_arity
        res['op_arity_class']  = op_arity_class
        res['has_attr']  = True if self.attributes else False

        # check implementation via descriptor registry
        try:
            _ = get_opdesc_registry().get_shape_inference_function(self.name)
            res['is_implemented'] = True
        except Exception:
            res['is_implemented'] = False
        
        return res

    def __str__(self):
        ostr  = f"{PFX}name                           = {self.name                           }\n"
        ostr += f"{PFX}domain                         = {self.domain                         }\n"
        ostr += f"{PFX}deprecated                     = {self.deprecated                     }\n"
        ostr += f"{PFX}has_function                   = {self.has_function                   }\n"
        ostr += f"{PFX}has_function_template          = {self.has_function_template          }\n"
        ostr += f"{PFX}function_body                  = {self.function_body                  }\n"
        ostr += f"{PFX}has_context_dependent_function = {self.has_context_dependent_function }\n"
        ostr += f"{PFX}all_function_opset_versions    = {self.all_function_opset_versions    }\n"
        ostr += f"{PFX}max_input                      = {self.max_input                      }\n"
        ostr += f"{PFX}min_input                      = {self.min_input                      }\n"
        ostr += f"{PFX}max_output                     = {self.max_output                     }\n"
        ostr += f"{PFX}min_output                     = {self.min_output                     }\n"
        ostr += f"{PFX}since_version                  = {self.since_version                  }\n"
        ostr += f"{PFX}support_level                  = {self.support_level                  }\n"

        if self.attributes:
            ostr += f"{PFX}attributes:\n"
            for attr_ in self.attributes:
                ostr += f"{attr_}\n"
            ostr += "\n"

        if self.inputs:
            ostr += f"{PFX}inputs:\n"
            for in_ in self.inputs:
                ostr += f"{in_}\n"
            ostr += "\n"

        if self.outputs:
            ostr += f"{PFX}outputs:\n"
            for out_ in self.outputs:
                ostr += f"{out_}\n"
            ostr += "\n"

        if self.allowed_types:
            ostr += f"{PFX}allowed_types:\n"
            for type_ in self.allowed_types:
                ostr += f"{type_}\n"
            ostr += "\n"

        if self.doc:
            ostr += f"{PFX}doc:\n"
            ostr += f"{'-'*40}\n"
            ostr += f"{self.doc}\n"
            ostr += f"{'-'*40}\n"

        if self.code_snippet:
            ostr += f"{PFX}code_snippet:\n"
            ostr += f"{'-'*40}\n"
            ostr += f"{self.code_snippet}\n"
            ostr += f"{'-'*40}\n"

        if self.sample_implementation:
            ostr += f"{PFX}sample_implementation:\n"
            ostr += f"{'-'*40}\n"
            ostr += f"{self.sample_implementation}\n"
            ostr += f"{'-'*40}\n"

        return ostr

#USEFUL TYPEDEFs for MYPY:
#    OpRecords             : [OpSchema]
#    DomainMap             : domain_name -> version -> schema_list
#    DomainSupportLevelMap : domain_name -> support_level -> schema_name -> schema
#    SchemaVersions        : [(schema_name, cur_schema, all_versions_schema_list)]
#    SupportLevels         : [(support_level, SchemaVersions)]
#    Domains               : [(domain_name, SupportLevels)]
OpRecords             = List[OpRecord]
DomainMap             = Dict[str, Dict[int, OpRecords]]
DomainSupportLevelMap = Dict[str, Dict[int, Dict[str, OpRecords]]]
SchemaVersions        = List[Tuple[str, OpRecord, OpRecords]]
SupportLevels         = List[Tuple[int, SchemaVersions]]
Domains               = List[Tuple[str, SupportLevels]]

class ONNXDomains:
    def __init__(self):
        self.SNIPPETS                                = collect_snippets()
        self.SAMPLE_IMPLEMENTATIONS                  = collect_sample_implementations()
        self.dv_index        : DomainMap             = defaultdict(lambda: defaultdict(list))
        self.dsn_index       : DomainSupportLevelMap = defaultdict(lambda: defaultdict( lambda: defaultdict(list)))
        self.operator_schemas: Domains               = []
        self.existing_ops    : set[str]              = set()

        for schema in defs.get_all_schemas_with_history():
            oprec = OpRecord(schema, self.SNIPPETS, self.SAMPLE_IMPLEMENTATIONS)
            self.dv_index[get_domain(schema.domain)][schema.since_version].append(oprec)
            self.dsn_index[get_domain(schema.domain)][int(schema.support_level)][schema.name].append(oprec)

        for domain_name, _supportmap in self.dsn_index.items():
            processed_support_map = []
            for _support_lvl, _schema_map in sorted(_supportmap.items()):
                processed_schema_map = []
                for _n, unsorted_versions in sorted(_schema_map.items()):
                    versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                    oprec    = versions[-1]
                    if oprec.name in self.existing_ops: continue
                    self.existing_ops.add(oprec.name)
                    processed_schema_map.append((_n,oprec,versions))
                processed_support_map.append((_support_lvl, processed_schema_map))
            self.operator_schemas.append((domain_name, processed_support_map))
        return

    def get_ops_summary(self, dom_name):
        res = []
        for _dom, _support_lvls in self.operator_schemas:
            if _dom != dom_name: continue
            for _lvl, _opversions in _support_lvls:
                for _op_num, (_opname, _oprec, _) in enumerate(_opversions):
                    res.append(_oprec.summary())
        return res

if __name__ == '__main__':
    #domains = ["ai.onnx", 'ai.onnx.preview.training', 'ai.onnx.ml']
    domain  = ONNXDomains()
    ai_onnx_ops = domain.get_ops_summary("ai.onnx")
    ai_onnx_ops = sorted(ai_onnx_ops, key=lambda x: (x['in_arity'], x['out_arity']))
    print_csv(ai_onnx_ops[0].keys(), ai_onnx_ops, "ai_onnx_ops.csv")

#    op_info_fields = ['deprecated', 'min_input', 'max_input', 'min_output', 'max_output']
#    op_map = {}
#    for o in ai_onnx_ops:
#        arity = "group_" + str(o['in_arity']) + "_" + str(o['out_arity'])
#        oinfo = {f: o[f] for f in op_info_fields}
#        oname = o['name']
#        otype = 'functions' if o['has_function'] else 'operators'
#        if otype not in op_map: op_map[otype] = {}
#        if arity not in op_map[otype]: op_map[otype][arity] = {}
#        if 'n/a' in arity:
#            op_map[otype][arity][oname] = oinfo
#        else:
#            op_map[otype][arity][oname] = {'deprecated': o['deprecated']}
#    print_yaml(op_map, "ai_onnx_ops.yaml")
