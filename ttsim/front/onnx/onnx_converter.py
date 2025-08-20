#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ONNX to TTSIM Operator Converter

This module converts ONNX operators to TTSIM-compatible operators by:
1. Converting ONNX attributes to TTSIM tensor inputs where needed
2. Transforming operator semantics to match TTSIM expectations
3. Adding necessary constant tensors for attribute-to-tensor conversions

Key Differences:
- ONNX Unsqueeze: 1 input (data) + axes attribute 
- TTSIM Unsqueeze: 2 inputs (data + axes_tensor)

- ONNX Reshape: 1 input (data) + shape attribute OR 2 inputs (data + shape)
- TTSIM Reshape: 2 inputs (data + shape_tensor)
"""
import numpy as np
from typing import Dict, List, Any, Tuple
from ttsim.ops import SimTensor

def numpy_dtype_to_ttsim_str(np_dtype) -> str:
    """Convert numpy dtype to TTSIM dtype string."""
    # Handle both numpy dtype objects and type objects
    if hasattr(np_dtype, 'name'):
        dtype_name = np_dtype.name
    else:
        dtype_name = str(np_dtype.__name__) if hasattr(np_dtype, '__name__') else str(np_dtype)
    
    # Map numpy dtype names to TTSIM dtype strings
    dtype_map = {
        'bool': 'BOOL',
        'int8': 'INT8', 
        'int16': 'INT16',
        'int32': 'INT32', 
        'int64': 'INT64',
        'uint8': 'UINT8',
        'uint16': 'UINT16', 
        'uint32': 'UINT32',
        'uint64': 'UINT64',
        'float16': 'FLOAT16',
        'float32': 'FLOAT32',
        'float64': 'FLOAT64'
    }
    
    return dtype_map.get(dtype_name, 'FLOAT32')  # Default fallback

class ONNXToTTSIMConverter:
    """Converts ONNX operators to TTSIM-compatible format."""
    
    def __init__(self):
        self.conversion_registry = {
            'Unsqueeze': self._convert_unsqueeze,
            'Squeeze': self._convert_squeeze, 
            'Reshape': self._convert_reshape,
            'Transpose': self._convert_transpose,
            'Gather': self._convert_gather,
            'Softmax': self._convert_softmax,
            'Gemm': self._convert_gemm,
            'MaxPool': self._convert_maxpool,
            'AveragePool': self._convert_averagepool,
            # Add more operators as needed
        }
        self.generated_tensors = []  # Track generated constant tensors
        
    def convert_graph(self, onnx_graph_info: Dict, onnx_tensor_info: Dict) -> Tuple[Dict, Dict]:
        """
        Convert entire ONNX graph to TTSIM format.
        
        Returns:
            - Modified graph info with converted operators
            - Modified tensor info with added constant tensors
        """
        converted_nodes = []
        updated_tensors = dict(onnx_tensor_info)
        
        # Convert dtypes in existing tensors 
        for tensor_name, tensor_info in updated_tensors.items():
            if 'dtype' in tensor_info:
                tensor_info['dtype'] = numpy_dtype_to_ttsim_str(tensor_info['dtype'])
            # Keep data as numpy arrays - TTSIM needs them for computation
        
        for node in onnx_graph_info['node']:
            try:
                # Ensure node has required fields
                if 'attrs' not in node:
                    node['attrs'] = {}
                if 'inList' not in node:
                    node['inList'] = []
                if 'outList' not in node:
                    node['outList'] = []
                    
                if node['optype'] in self.conversion_registry:
                    converted_node, new_tensors = self.conversion_registry[node['optype']](node)
                    converted_nodes.append(converted_node)
                    updated_tensors.update(new_tensors)
                else:
                    # Pass through operators that don't need conversion
                    converted_nodes.append(node)
            except Exception as e:
                print(f"ERROR converting node {node.get('name', 'unknown')} of type {node.get('optype', 'unknown')}: {e}")
                # Pass through unchanged on error
                converted_nodes.append(node)
        
        # Update graph with converted nodes
        converted_graph = dict(onnx_graph_info)
        converted_graph['node'] = converted_nodes
        
        return converted_graph, updated_tensors
    
    def _convert_unsqueeze(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX Unsqueeze to TTSIM format.
        
        ONNX: Unsqueeze(data) with axes=[1,3] attribute
        TTSIM: Unsqueeze(data, axes_tensor) with axes as input tensor
        """
        axes = node['attrs'].get('axes', [])
        if axes is None:
            axes = []
        if not axes:
            print(f"WARNING: Unsqueeze operator {node['name']} has empty or missing 'axes' attribute")
            axes = [0]  # Default to axis 0
        
        # Create axes tensor name
        axes_tensor_name = f"{node['name']}_axes_tensor"
        
        # Create constant tensor for axes
        axes_tensor_info = {
            'name': axes_tensor_name,
            'shape': [len(axes)],
            'dtype': 'INT64',
            'data': np.array(axes, dtype=np.int64),  # Keep as numpy array
            'resolve': 'C',  # Constant
            'op_in': [],
            'op_out': [node['name']]
        }
        
        # Modify node to include axes tensor as input
        converted_node = dict(node)
        converted_node['inList'] = node['inList'] + [axes_tensor_name]
        # Remove axes from attributes since it's now an input
        converted_node['attrs'] = {k: v for k, v in node['attrs'].items() if k != 'axes'}
        
        return converted_node, {axes_tensor_name: axes_tensor_info}
    
    def _convert_squeeze(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX Squeeze to TTSIM format.
        Similar to Unsqueeze - convert axes attribute to input tensor.
        """
        axes = node['attrs'].get('axes', [])
        if axes is None:
            axes = []
        if not axes:
            print(f"WARNING: Squeeze operator {node['name']} has empty or missing 'axes' attribute")
            # If no axes specified, squeeze all dimensions of size 1
            # This requires runtime shape info, so we'll create empty axes for now
            axes = []
        
        # Create axes tensor name
        axes_tensor_name = f"{node['name']}_axes_tensor"
        
        # Create constant tensor for axes
        axes_tensor_info = {
            'name': axes_tensor_name,
            'shape': [len(axes)] if axes else [0],
            'dtype': 'INT64',
            'data': np.array(axes, dtype=np.int64) if axes else np.array([], dtype=np.int64),
            'resolve': 'C',
            'op_in': [],
            'op_out': [node['name']]
        }
        
        # Modify node
        converted_node = dict(node)
        converted_node['inList'] = node['inList'] + [axes_tensor_name]
        converted_node['attrs'] = {k: v for k, v in node['attrs'].items() if k != 'axes'}
        
        return converted_node, {axes_tensor_name: axes_tensor_info}
    
    def _convert_reshape(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX Reshape to TTSIM format.
        
        ONNX Reshape can have:
        - 1 input (data) + shape attribute, OR  
        - 2 inputs (data + shape tensor)
        
        TTSIM always expects 2 inputs (data + shape tensor).
        """
        if len(node['inList']) == 2:
            # Already has shape as input tensor, no conversion needed
            return node, {}
        
        # Need to convert shape attribute to tensor
        shape = node['attrs'].get('shape', None)
        if shape is None:
            print(f"WARNING: Reshape operator {node['name']} missing shape information, cannot convert")
            return node, {}  # Return unchanged if no shape info
        
        # Create shape tensor name
        shape_tensor_name = f"{node['name']}_shape_tensor"
        
        # Create constant tensor for shape
        shape_tensor_info = {
            'name': shape_tensor_name,
            'shape': [len(shape)],
            'dtype': 'INT64',
            'data': np.array(shape, dtype=np.int64),  # Keep as numpy array
            'resolve': 'C',
            'op_in': [],
            'op_out': [node['name']]
        }
        
        # Modify node
        converted_node = dict(node)
        converted_node['inList'] = node['inList'] + [shape_tensor_name]
        converted_node['attrs'] = {k: v for k, v in node['attrs'].items() if k != 'shape'}
        
        return converted_node, {shape_tensor_name: shape_tensor_info}
    
    def _convert_transpose(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX Transpose to TTSIM format if needed.
        ONNX: Transpose(data) with perm=[0,2,1] attribute
        Check if TTSIM expects perm as input tensor.
        """
        # For now, assume TTSIM Transpose works with attributes
        # This may need adjustment based on TTSIM implementation
        return node, {}
    
    def _convert_gather(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX Gather to TTSIM format.
        ONNX: Gather(data, indices) with axis=0 attribute
        TTSIM: May need axis as input tensor - check implementation
        """
        # For now, assume TTSIM Gather works with attributes
        # This may need adjustment based on TTSIM implementation  
        return node, {}
    
    def _convert_softmax(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX Softmax to TTSIM format.
        ONNX: Softmax(data) with axis=-1 attribute
        """
        # For now, assume TTSIM Softmax works with attributes
        return node, {}
    
    def _convert_gemm(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX Gemm to TTSIM MatMul (simplified version).
        
        ONNX Gemm: Y = alpha * A * B + beta * C
        - A, B: input matrices 
        - C: bias (optional)
        - alpha, beta: scalars (default 1.0)
        - transA, transB: transpose flags (default 0)
        
        TTSIM: Use MatMul for A*B, warn about unsupported features
        """
        # Get Gemm attributes
        alpha = node['attrs'].get('alpha', 1.0)
        beta = node['attrs'].get('beta', 1.0)  
        transA = node['attrs'].get('transA', 0)
        transB = node['attrs'].get('transB', 0)
        
        # Warn about unsupported features
        if alpha != 1.0 or beta != 1.0:
            print(f"WARNING: Gemm operator {node['name']} has alpha={alpha}, beta={beta}. Only alpha=1, beta=1 supported. Results may be incorrect.")
        
        # Handle transpose operations - this is critical for correct matrix multiplication
        needs_transpose = transA != 0 or transB != 0
        if needs_transpose:
            print(f"INFO: Gemm operator {node['name']} has transA={transA}, transB={transB}. Need to handle matrix transposition for correct MatMul.")
            
        if len(node['inList']) > 2:
            print(f"WARNING: Gemm operator {node['name']} has bias input. Bias addition not implemented. Results may be incorrect.")
        
        # Convert Gemm to MatMul (simple approach)
        converted_node = dict(node)
        converted_node['optype'] = 'MatMul'
        
        # Keep only A and B inputs, ignore bias for now
        if node['inList'] is not None and len(node['inList']) >= 2:
            converted_node['inList'] = node['inList'][:2]
        else:
            print(f"ERROR: Gemm operator {node['name']} has insufficient inputs: {node.get('inList', [])}")
            converted_node['inList'] = node['inList'] if node['inList'] else []
            
        # Pass transpose information to MatMul as custom attributes
        # TTSIM MatMul might be able to handle these, or we need to implement the logic
        converted_node['attrs'] = {}
        if transA != 0:
            converted_node['attrs']['transA'] = transA
        if transB != 0:
            converted_node['attrs']['transB'] = transB
        
        return converted_node, {}
    
    def _convert_maxpool(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX MaxPool to TTSIM format.
        Ensure kernel_shape attribute is present.
        """
        converted_node = dict(node)
        
        # Check if kernel_shape is missing and try to infer from other attributes
        if 'kernel_shape' not in node['attrs'] or node['attrs']['kernel_shape'] is None:
            # Try to infer from strides or provide default
            strides = node['attrs'].get('strides', [2, 2])  # Common default
            print(f"WARNING: MaxPool operator {node['name']} missing kernel_shape, using strides as fallback: {strides}")
            converted_node['attrs']['kernel_shape'] = strides
        
        return converted_node, {}
    
    def _convert_averagepool(self, node: Dict) -> Tuple[Dict, Dict]:
        """
        Convert ONNX AveragePool to TTSIM format.
        Ensure kernel_shape attribute is present.
        """
        converted_node = dict(node)
        
        # Check if kernel_shape is missing and try to infer from other attributes  
        if 'kernel_shape' not in node['attrs'] or node['attrs']['kernel_shape'] is None:
            # Try to infer from strides or provide default
            strides = node['attrs'].get('strides', [2, 2])  # Common default
            print(f"WARNING: AveragePool operator {node['name']} missing kernel_shape, using strides as fallback: {strides}")
            converted_node['attrs']['kernel_shape'] = strides
            
        return converted_node, {}

def convert_onnx_to_ttsim(onnx_graph_info: Dict, onnx_tensor_info: Dict) -> Tuple[Dict, Dict]:
    """
    Main conversion function.
    
    Args:
        onnx_graph_info: ONNX graph information from onnx2nx.py
        onnx_tensor_info: ONNX tensor information from onnx2nx.py
        
    Returns:
        Tuple of (converted_graph_info, converted_tensor_info)
    """
    converter = ONNXToTTSIMConverter()
    return converter.convert_graph(onnx_graph_info, onnx_tensor_info)