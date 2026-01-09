#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ttsim.utils.common import parse_yaml
from ttsim.utils.types import validate_datatype

type LayerName = str
type LayerSequence = List[LayerName]
type PipeName = str
type TypeName = str


class WL2ArchDatatypes(BaseModel):
    """
    Represents the data types used in workload to architecture mapping.
    """
    global_type: TypeName = Field(..., description="Global type for the data")
    override: Dict[LayerName, TypeName] = Field(..., description="Overrides for specific data types")

    def __str__(self):
        return f"Global Type: {self.global_type}, Overrides: {self.override}"

    def layer_2_datatype(self, layer_name: LayerName) -> TypeName:
        """
        Get the data type for a specific layer.
        If no override is found, return the global type.
        """
        return self.override.get(layer_name, self.global_type) if self.override else self.global_type

    def update_global_type(self, new_global_type: TypeName) -> None:
        """
        Update the global data type.
        """
        if not isinstance(new_global_type, str):
            raise TypeError("new_global_type must be a string")
        # Validate the datatype to ensure it's supported
        validate_datatype(new_global_type)
        self.global_type = new_global_type

    @staticmethod
    def from_dict(spec: dict[str, Any]) -> 'WL2ArchDatatypes':
        """
        Create a WL2ArchDatatypes instance from a dictionary.
        """
        override: Dict[LayerName, TypeName] = {}
        global_type: TypeName = spec.get('global_type', None)
        override_spec = spec.get('override', dict())
        if global_type is None:
            raise AssertionError(f'global_type must be set in {spec}')
        # Validate global_type before converting to lowercase
        validate_datatype(global_type)
        global_type = global_type.lower()
        for kk, vv in override_spec.items():
            key_upper = kk.upper()
            value_lower = vv.lower()
            if key_upper in override and override[key_upper] != value_lower:
                raise AssertionError(f'override {kk} already set to {override[key_upper]}, trying to set to {value_lower}')
            # Validate each override datatype before converting to lowercase
            validate_datatype(vv)
            override[key_upper] = value_lower

        return WL2ArchDatatypes(global_type=global_type, override=override)


class WL2ArchRemovalLayers(BaseModel):
    """
    Represents the null layers in workload to architecture mapping.
    """
    layer_names: set[LayerName] = Field(..., description="Name of the null layer")

    def __str__(self):
        return f"Null Layers: {self.layer_names}"

    def is_included(self, layer_name: LayerName) -> bool:
        return layer_name in self.layer_names

    @staticmethod
    def from_list(layer_names: List[LayerName]) -> 'WL2ArchRemovalLayers':
        """
        Create a WL2ArchRemovalLayers instance from a list of layer names.
        """
        if not layer_names:
            raise ValueError("layer_names must not be empty")
        return WL2ArchRemovalLayers(layer_names={x.upper() for x in layer_names})


class WL2ArchFusedLayers(BaseModel):
    """
    Represents the fused layers in workload to architecture mapping.
    """
    layer_sequences: List[List[LayerName]] = Field(..., description="Sequences of fused layers")

    def __str__(self):
        return f"Fused Layers: {self.layer_sequences}"

    def get_fused_layer_sequences(self):
        """
        Get the sequences of fused layers.
        """
        for seq in self.layer_sequences:
            yield seq

    @staticmethod
    def from_list(spec: List[List[LayerName]]) -> 'WL2ArchFusedLayers':
        """
        Create a WL2ArchFusedLayers instance from a list of layer sequences.
        """
        op_fusion_list = [[y.upper() for y in x] for x in spec]

        if not op_fusion_list:
            raise ValueError("layer_sequences must not be empty")
        return WL2ArchFusedLayers(layer_sequences=op_fusion_list)


class WL2ArchLayer2ComputePipe(BaseModel):
    """
    Represents the mapping from layers to compute pipelines in workload to architecture mapping.
    """
    wl_map: Dict[LayerName, PipeName] = Field(..., description="Mapping of workload to architecture details")

    def __str__(self):
        return f"Layer 2 compute pipe map {self.wl_map}"

    def layer_2_pipe(self, layer_name: LayerName) -> PipeName:
        pipe: Optional[str]  = self.wl_map.get(layer_name, None)
        if pipe is None:
            raise AssertionError(f'Layer {layer_name} not found in workload to architecture mapping')
        return pipe

    @staticmethod
    def from_dict(spec: dict[str, Any]) -> 'WL2ArchLayer2ComputePipe':
        """
        Create a WL2ArchLayer2ComputePipe instance from a dictionary.
        """
        op2rsrc = {}
        assert 'compute' in spec, "Attribute(compute) missing in op_rsrc_spec"
        for op_pipe, op_list in spec['compute'].items():
            op2rsrc.update({o.upper(): op_pipe.lower() for o in op_list})
        return WL2ArchLayer2ComputePipe(wl_map=op2rsrc)


class WL2ArchMap(BaseModel):
    """
    Represents the workload to architecture mapping.
    """
    data_type_spec: WL2ArchDatatypes = Field(..., description="Data type specifications for operations")
    removal_spec: WL2ArchRemovalLayers = Field(..., description="Null layers specifications")
    fusion_spec: WL2ArchFusedLayers = Field(..., description="Fused layers specifications")
    rsrc_spec: WL2ArchLayer2ComputePipe = Field(..., description="Resource specifications for operations")

    def layer_2_datatype(self, layer_name: LayerName) -> TypeName:
        """
        Get the data type for a specific layer.
        If no override is found, return the global type.
        """
        return self.data_type_spec.layer_2_datatype(layer_name)

    def __str__(self):
        return (f"Workload to Architecture Map:\n"
                f"Data Types: {self.data_type_spec}\n"
                f"Null Layers: {self.removal_spec}\n"
                f"Fused Layers: {self.fusion_spec}\n"
                f"Resource Specs: {self.rsrc_spec}")

    @staticmethod
    def from_yaml(cfg_yaml_file: str) -> 'WL2ArchMap':
        """
        Create a WL2ArchMap instance from a YAML configuration file.
        """
        cfg_dict = parse_yaml(cfg_yaml_file)
        required_fields = ['op_data_type_spec', 'op_removal_spec', 'op_fusion_spec', 'op_rsrc_spec']
        for ff in required_fields:
            assert ff in cfg_dict, f'required attribute: {ff} missing in workload map file: {cfg_yaml_file}'

        data_type_spec = WL2ArchDatatypes.from_dict(cfg_dict['op_data_type_spec'])
        removal_spec = WL2ArchRemovalLayers.from_list(cfg_dict['op_removal_spec'])
        fusion_spec = WL2ArchFusedLayers.from_list(cfg_dict['op_fusion_spec'])
        rsrc_spec = WL2ArchLayer2ComputePipe.from_dict(cfg_dict['op_rsrc_spec'])

        return WL2ArchMap(
            data_type_spec=data_type_spec,
            removal_spec=removal_spec,
            fusion_spec=fusion_spec,
            rsrc_spec=rsrc_spec
        )


class WL2ArchTypeSpec:

    instance: Optional[WL2ArchDatatypes] = None

    @classmethod
    def get_instance(cls) -> WL2ArchDatatypes:
        """
        Get the instance of WL2ArchDatatypes.
        """
        if cls.instance is None:
            raise AssertionError("WL2ArchTypeSpec instance not set. Call set_instance() before accessing the singleton.")
        return cls.instance

    @classmethod
    def has_instance(cls) -> bool:
        """
        Check if the instance of WL2ArchDatatypes is set.
        """
        return cls.instance is not None

    @classmethod
    def get_global_datatype(cls) -> TypeName:
        """
        Get the default data type.
        """
        return cls.get_instance().global_type

    @classmethod
    def layer_2_datatype(cls, layer_name: LayerName) -> TypeName:
        """
        Get the data type for a specific layer.
        If no override is found, return the global type.
        """
        return cls.get_instance().layer_2_datatype(layer_name)

    @classmethod
    def set_instance(cls, instance: WL2ArchDatatypes, force: bool = False) -> None:
        """
        Set the instance of WL2ArchDatatypes.
        If force is True, replaces any existing instance.
        Altogether preventing a new instance could be problematic in testing scenarios or when reconfiguring the system.
        """
        if not isinstance(instance, WL2ArchDatatypes):
            raise TypeError("instance must be of type WL2ArchDatatypes")
        if cls.instance is not None and not force:
            raise AssertionError("WL2ArchTypeSpec instance already set")
        cls.instance = instance


def get_wlmapspec_from_yaml(cfg_yaml_file) -> WL2ArchMap:
    mapspec = WL2ArchMap.from_yaml(cfg_yaml_file)
    WL2ArchTypeSpec.set_instance(mapspec.data_type_spec)
    return mapspec
