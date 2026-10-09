#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Purpose: RetinaNet Workflow Validation Script.
This script acts as the final check in the development pipeline. It confirms 
that the modified RetinaNet model remains mathematically functional while 
incorporating the explicit hardware labels required for Tenstorrent 
simulation. It bridges the gap between PyTorch modeling and TT-Polaris execution.
"""

import os
import sys
from loguru import logger
# Add project root to path to ensure ttsim and workloads are findable
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from workloads.Retinanet.model import RetinaNet

def run_standalone(outdir: str = '.') -> None:
    """
    Renamed from main() per reviewer feedback. 
    Handles model instantiation and graph dumping for TT-Polaris integration.
    """
    # Define configurations similar to ip_workloads.yaml structure
    retinanet_cfgs = {
        'retinanet_rn50_608': {
            "num_classes": 80,
            "img_size": 608,
            "resnet_depth": 50,
        }
    }

    for name, cfg in retinanet_cfgs.items():
        logger.debug(f"Creating {name} model...")
        model = RetinaNet(name, cfg)

        model.create_input_tensors()  # Ensure input tensors are created with correct shapes and labels
        # Run forward pass
        cls_concat, reg_concat, final_bbox_coords = model()

        logger.debug(f"Output Shapes for {name}:")
        logger.debug(f"  Classification: {cls_concat.shape}")     
        logger.debug(f"  Regression:     {reg_concat.shape}") 
             
        logger.debug(f"  BBox Coords:    {final_bbox_coords.shape}")  

        # Internal Assertions
        assert len(final_bbox_coords.shape) == 3, "BBox should be 3D [Batch, Anchors, 4]"
        
        # Dump Graph (Standard practice for run_standalone in TT-Polaris)
        logger.debug(f"Dumping ONNX to {outdir}...")
        
        gg = model.get_forward_graph()
        gg.graph2onnx(f'{outdir}/{name}.onnx', do_model_check=False)

        logger.debug('-' * 40)
if __name__ == "__main__":
    run_standalone()