#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Purpose: This document tracks the functional parity between the standard PyTorch inference and the TT-Polaris hardware simulation.
### Validation Script

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from loguru import logger
import torch # type: ignore
from retinanet.model import resnet50 # type: ignore

def main():
    num_classes = 80
    img_size = 608

    model = resnet50(num_classes=num_classes, pretrained=False)
    model.eval()

    # Create dummy input tensor
    x = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        # Trace through the backbone
        x0 = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x1 = model.layer1(x0)
        x2 = model.layer2(x1)
        x3 = model.layer3(x2)
        x4 = model.layer4(x3)

        # Trace through the Feature Pyramid Network (FPN)
        features = model.fpn([x2, x3, x4])

        # Verify Detection Head outputs
        reg = torch.cat([model.regressionModel(f) for f in features], dim=1)
        cls = torch.cat([model.classificationModel(f) for f in features], dim=1)

        # Anchor and Box Transformation Logic
        anchors = model.anchors(x) 
        transformed_anchors = model.regressBoxes(anchors, reg)
        final_bbox_coords = model.clipBoxes(transformed_anchors, x)

    logger.debug("PyTorch cls shape: {}", cls.shape)              # Raw scores
    logger.debug("PyTorch reg shape: {}", reg.shape)              # Raw offsets
    logger.debug("Final BBox Coords shape: {}", final_bbox_coords.shape) # Actual Pixels
if __name__ == "__main__":
    main()
