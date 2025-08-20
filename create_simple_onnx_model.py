#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Create a simple ONNX model for testing the ONNX-to-TTSIM converter.
This model contains operators that need conversion (Unsqueeze, Squeeze, Reshape).
"""
import torch
import torch.onnx
import numpy as np
from pathlib import Path

class SimpleTestModel(torch.nn.Module):
    """Simple model with operators that need conversion."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
        
    def forward(self, x):
        # x shape: [batch, 10]
        
        # Add dimension: [batch, 10] -> [batch, 1, 10] 
        x = x.unsqueeze(1)  # This creates ONNX Unsqueeze operator
        
        # Linear transformation: [batch, 1, 10] -> [batch, 1, 5]
        x = self.linear(x)
        
        # Remove dimension: [batch, 1, 5] -> [batch, 5]
        x = x.squeeze(1)  # This creates ONNX Squeeze operator
        
        # Reshape: [batch, 5] -> [batch, 1, 5]  
        x = x.view(-1, 1, 5)  # This creates ONNX Reshape operator
        
        return x

def create_simple_test_model():
    """Create and export a simple ONNX model."""
    
    # Create output directory
    output_dir = Path("onnx_models")
    output_dir.mkdir(exist_ok=True)
    
    # Create model
    model = SimpleTestModel()
    model.eval()
    
    # Create dummy input
    batch_size = 2
    input_size = 10
    dummy_input = torch.randn(batch_size, input_size)
    
    # Export to ONNX
    output_path = output_dir / "simple_test_model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
    
    print(f"✓ Simple test model created: {output_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(str(output_path))
    
    print("\nModel operators:")
    for node in onnx_model.graph.node:
        print(f"  - {node.op_type}: {node.name}")
        if hasattr(node, 'attribute') and node.attribute:
            for attr in node.attribute:
                print(f"    - {attr.name}: {attr}")
    
    return output_path

if __name__ == "__main__":
    create_simple_test_model()