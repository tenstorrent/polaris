#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Script to create ONNX models for Polaris workloads.
Converts PyTorch and Hugging Face models to ONNX format.
"""
import os
import torch
import torch.onnx
from pathlib import Path
import torchvision.models as models
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

def create_output_dir():
    """Create onnx_models directory if it doesn't exist."""
    output_dir = Path("onnx_models")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def create_resnet50_onnx(output_dir):
    """Create ResNet50 ONNX model."""
    print("Creating ResNet50 ONNX model...")
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_dir / "resnet50.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("✓ ResNet50 ONNX model created")

def create_mobilenet_onnx(output_dir):
    """Create MobileNetV2 ONNX model."""
    print("Creating MobileNetV2 ONNX model...")
    
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_dir / "mobilenet_v2.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("✓ MobileNetV2 ONNX model created")

def create_bert_onnx(output_dir):
    """Create BERT Base ONNX model."""
    print("Creating BERT Base ONNX model...")
    
    # Load pre-trained BERT model
    model_name = "bert-base-uncased"
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.eval()
    
    # Create dummy inputs
    sequence_length = 512
    batch_size = 1
    
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, sequence_length))
    dummy_attention_mask = torch.ones(batch_size, sequence_length)
    dummy_token_type_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        output_dir / "bert_base.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'}
        }
    )
    print("✓ BERT Base ONNX model created")

def create_gpt2_onnx(output_dir):
    """Create GPT2 ONNX model."""
    print("Creating GPT2 ONNX model...")
    
    # Load pre-trained GPT2 model
    model_name = "gpt2"
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()
    
    # Create dummy inputs
    sequence_length = 1024
    batch_size = 1
    
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, sequence_length))
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input_ids,
        output_dir / "gpt2.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("✓ GPT2 ONNX model created")

def main():
    """Main function to create all ONNX models."""
    print("Creating ONNX models for Polaris workloads...")
    print("This may take a few minutes to download and convert models...\n")
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir.absolute()}\n")
    
    try:
        # Create all models
        create_resnet50_onnx(output_dir)
        create_mobilenet_onnx(output_dir)
        create_bert_onnx(output_dir)
        create_gpt2_onnx(output_dir)
        
        print(f"\n✅ All ONNX models created successfully in {output_dir}/")
        print("\nGenerated files:")
        for onnx_file in output_dir.glob("*.onnx"):
            file_size = onnx_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {onnx_file.name} ({file_size:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Error creating ONNX models: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install torch torchvision transformers onnx")

if __name__ == "__main__":
    main()