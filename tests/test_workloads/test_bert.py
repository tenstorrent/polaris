#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import workloads.BERT as BERT

def test_bert(session_temp_directory):
    """
    Simple test for BERT model.
    Creates a BERT model, performs inference analytics, and optionally exports to ONNX.
    """
    output_dir = str(session_temp_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    # Test both base and large model variants
    model_variants = ["bert-base-uncased"]
    
    for model_name in model_variants:
        # Create the model
        bert_model = BERT.convert_huggingface_to_simnn(model_name, model_name)
        
        # Run the model to get outputs
        start_logits, end_logits = bert_model()
        
        # Generate the forward graph
        gg = bert_model.get_forward_graph()
        
        # Export to ONNX in the temp directory (optional)
        out_onnx_file = os.path.join(output_dir, f"{model_name.replace('-', '_')}.onnx")
        gg.graph2onnx(out_onnx_file)
        
        # Basic assertion to verify test ran correctly
        assert start_logits is not None
        assert end_logits is not None
        assert os.path.exists(out_onnx_file)