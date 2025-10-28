#!/usr/bin/env python3
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic models for workload descriptions.

This module provides Pydantic models for validating workload configuration files
(e.g., config/all_workloads.yaml) and utilities for loading them from local files or URLs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from ttsim.utils.common import parse_yaml


class WorkloadConfig(BaseModel):
    """
    Pydantic model for a single workload configuration.
    
    Represents an individual workload entry from a workloads YAML file.
    """
    
    api: str = Field(..., description='API type (e.g., TTSIM)')
    name: str = Field(..., description='Workload name')
    basedir: str = Field(..., description='Base directory for the workload')
    module: str = Field(..., description='Module specification (format: ClassName@file.ext or file.ext)')
    params: Optional[Dict[str, Any]] = Field(None, description='Optional workload parameters')
    instances: Dict[str, Dict[str, Any]] = Field(..., description='Workload instances with their configurations')
    
    class Config:
        extra = 'forbid'
        populate_by_name = True
    
    @field_validator('module')
    @classmethod
    def validate_module_format(cls, v: str) -> str:
        """
        Validate the module attribute format based on file extension.
        
        Rules:
        - .py files: can be 'ClassName@file.py' or 'file.py'
        - .html files: must be 'file.html' (no @ separator)
        - .onnx files: must be 'file.onnx' (no @ separator)
        - Other extensions: error and terminate
        
        Args:
            v (str): Module string to validate.
            
        Returns:
            str: Validated module string.
            
        Raises:
            ValueError: If the module format is invalid.
        """
        # Check if module contains @ separator
        has_separator = '@' in v
        
        if has_separator:
            # Format: ClassName@filename
            parts = v.split('@')
            if len(parts) != 2:
                raise ValueError(f'Module with @ separator must have exactly one @: {v}')
            class_name, filename = parts
            if not class_name.strip():
                raise ValueError(f'Class name cannot be empty in module: {v}')
            if not filename.strip():
                raise ValueError(f'Filename cannot be empty in module: {v}')
        else:
            # Format: filename only
            filename = v
        
        # Extract file extension
        if '.' not in filename:
            raise ValueError(f'Module filename must have an extension: {filename}')
        
        extension = filename.rsplit('.', 1)[-1].lower()
        
        # Validate extension and format
        valid_extensions = {'py', 'html', 'onnx'}
        if extension not in valid_extensions:
            raise ValueError(
                f'Unsupported model file extension: .{extension} in module "{v}". '
                f'Only .py, .html, and .onnx files are supported. '
                f'Terminating due to invalid extension.'
            )
        
        # Validate format rules by extension
        if extension == 'py':
            # .py files can have either format
            pass
        elif extension in ['html', 'onnx']:
            # .html and .onnx files must NOT have @ separator
            if has_separator:
                raise ValueError(
                    f'Module with .{extension} extension must use "file.{extension}" format, '
                    f'not "ClassName@file.{extension}" format: {v}'
                )
        
        return v
    
    @field_validator('instances')
    @classmethod
    def validate_instances_is_dict(cls, v: Any) -> Dict[str, Dict[str, Any]]:
        """
        Validate that instances field is a dictionary.
        
        Args:
            v: Value to validate.
            
        Returns:
            Dict[str, Dict[str, Any]]: Validated instances dictionary.
            
        Raises:
            ValueError: If instances is not a dictionary.
        """
        if not isinstance(v, dict):
            raise ValueError(f'Instances must be a dictionary, got {type(v).__name__}')
        
        # Validate that each instance value is also a dictionary
        for instance_name, instance_config in v.items():
            if not isinstance(instance_config, dict):
                raise ValueError(
                    f'Instance "{instance_name}" configuration must be a dictionary, '
                    f'got {type(instance_config).__name__}'
                )
        
        return v
    
    @field_validator('params')
    @classmethod
    def validate_params_is_dict(cls, v: Optional[Any]) -> Optional[Dict[str, Any]]:
        """
        Validate that params field is a dictionary if provided.
        
        Args:
            v: Value to validate.
            
        Returns:
            Optional[Dict[str, Any]]: Validated params dictionary or None.
            
        Raises:
            ValueError: If params is provided but not a dictionary.
        """
        if v is not None and not isinstance(v, dict):
            raise ValueError(f'Params must be a dictionary, got {type(v).__name__}')
        return v
    
    def get_model_extension(self) -> str:
        """
        Get the file extension of the model file (without the dot).
        
        Returns:
            str: File extension without dot (e.g., 'py', 'html', 'onnx').
        """
        # Extract filename from module (handle both formats)
        if '@' in self.module:
            filename = self.module.split('@')[1]
        else:
            filename = self.module
        
        # Extract and return extension without dot
        extension = filename.rsplit('.', 1)[-1].lower()
        return extension
    
    def get_module_class(self) -> Optional[str]:
        """
        Get the class name from the module attribute if present.
        
        Returns:
            Optional[str]: Class name if module uses 'ClassName@file' format, None otherwise.
        """
        if '@' in self.module:
            return self.module.split('@')[0]
        return None
    
    def get_module_filename(self) -> str:
        """
        Get the filename from the module attribute.
        
        Returns:
            str: Filename (e.g., 'BasicLLM.py', 'model.onnx').
        """
        if '@' in self.module:
            return self.module.split('@')[1]
        return self.module


class WorkloadsFile(BaseModel):
    """
    Pydantic model for a workloads file containing multiple workload configurations.
    
    Represents the top-level structure of a workloads YAML file.
    """
    
    workloads: List[WorkloadConfig] = Field(..., description='List of workload configurations')
    
    class Config:
        extra = 'forbid'
        populate_by_name = True
    
    @field_validator('workloads')
    @classmethod
    def validate_workloads(cls, v: List[WorkloadConfig]) -> List[WorkloadConfig]:
        """
        Validate workloads list for common issues.
        
        Args:
            v: List of workload configurations to validate.
            
        Returns:
            List[WorkloadConfig]: Validated workloads list.
            
        Raises:
            ValueError: If workloads list is empty or contains duplicate names.
        """
        if not v:
            raise ValueError('workloads must contain at least one workload')
        
        # Check for duplicate workload names
        names = [w.name for w in v]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            # Get first duplicate
            first_dup = duplicates[0]
            raise ValueError(f'Duplicate workload name: {first_dup}')
        
        return v
    
    def get_workload_by_name(self, name: str) -> Optional[WorkloadConfig]:
        """
        Get a workload configuration by name.
        
        Args:
            name (str): Workload name to search for.
            
        Returns:
            Optional[WorkloadConfig]: Workload configuration if found, None otherwise.
        """
        for workload in self.workloads:
            if workload.name == name:
                return workload
        return None
    
    def get_workloads_by_extension(self, extension: str) -> List[WorkloadConfig]:
        """
        Get all workloads with a specific model file extension.
        
        Args:
            extension (str): File extension to filter by (without dot, e.g., 'py', 'html').
            
        Returns:
            List[WorkloadConfig]: List of workloads matching the extension.
        """
        return [wl for wl in self.workloads if wl.get_model_extension() == extension.lower()]


def load_workloads_file(filepath: str | Path, use_cache: bool = True) -> WorkloadsFile:
    """
    Load and validate a workloads YAML file from local path or URL.
    
    This function uses the existing ttsim.utils.common.parse_yaml utility which
    supports both local files and URLs through locator_handle.
    
    Args:
        filepath (str | Path): Path or URL to the workloads YAML file.
        use_cache (bool): Whether to use cache for URL fetching (default: True).
                         Note: Currently parse_yaml doesn't expose cache control,
                         so this parameter is for future compatibility.
        
    Returns:
        WorkloadsFile: Validated workloads file model.
        
    Raises:
        ValueError: If the file format is invalid or validation fails.
        FileNotFoundError: If the local file doesn't exist.
        requests.HTTPError: If URL fetch fails.
    """
    try:
        logger.debug('Loading workloads file from: {}', filepath)
        data = parse_yaml(filepath)
        
        if data is None:
            raise ValueError(f'Failed to parse YAML from {filepath}: file is empty or invalid')
        
        if not isinstance(data, dict):
            raise ValueError(f'Expected dictionary at top level, got {type(data).__name__}')
        
        if 'workloads' not in data:
            raise ValueError('Missing required "workloads" key in YAML file')
        
        workloads_file = WorkloadsFile(**data)
        logger.info('Successfully loaded and validated {} workloads from {}', 
                   len(workloads_file.workloads), filepath)
        
        return workloads_file
        
    except Exception as e:
        logger.error('Failed to load workloads file from {}: {}', filepath, e)
        raise


def load_workload_config(filepath: str | Path, workload_name: str) -> WorkloadConfig:
    """
    Load a specific workload configuration by name from a workloads file.
    
    Args:
        filepath (str | Path): Path or URL to the workloads YAML file.
        workload_name (str): Name of the workload to load.
        
    Returns:
        WorkloadConfig: Validated workload configuration.
        
    Raises:
        ValueError: If the workload name is not found or validation fails.
    """
    workloads_file = load_workloads_file(filepath)
    
    workload = workloads_file.get_workload_by_name(workload_name)
    if workload is None:
        available_names = [wl.name for wl in workloads_file.workloads]
        raise ValueError(
            f'Workload "{workload_name}" not found in {filepath}. '
            f'Available workloads: {available_names}'
        )
    
    logger.info('Loaded workload configuration: {}', workload_name)
    return workload
