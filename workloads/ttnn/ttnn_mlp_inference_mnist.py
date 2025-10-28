#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

#import torch
#import torchvision
#import torchvision.transforms as transforms
#import numpy as np

import ttsim.front.ttnn as ttnn
from loguru import logger
import numpy as np

def filter_ttnn_attrs(attrs_dict):
    return {k: v for k, v in attrs_dict.items() if not isinstance(v, ttnn.Tensor)}


def main():
    # Open Tenstorrent device
    device = ttnn.open_device(device_id=0)

    try:
        logger.info("\n--- MLP Inference Using TT-NN on MNIST ---")

        # Load MNIST data
        #transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        #testset    = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
        testloader = [
                (
                    ttnn._rand(shape=(28, 28), dtype=ttnn.float32, device=device),
                    ttnn.Tensor(shape=(1,), dtype=ttnn.int64, device=device,
                                data=np.array([0], dtype=np.int64))
                 )
                for _ in range(5)
                ]

        # Pretrained weights
        #weights = torch.load("mlp_mnist_weights.pt")
        #W1 = weights["W1"]
        #b1 = weights["b1"]
        #W2 = weights["W2"]
        #b2 = weights["b2"]
        #W3 = weights["W3"]
        #b3 = weights["b3"]

        #Random weights for MLP - will not predict correctly
        #torch.manual_seed(0)
        #W1 = torch.randn((128, 28 * 28), dtype=torch.float32)
        #b1 = torch.randn((128,), dtype=torch.float32)
        #W2 = torch.randn((64, 128), dtype=torch.float32)
        #b2 = torch.randn((64,), dtype=torch.float32)
        #W3 = torch.randn((10, 64), dtype=torch.float32)
        #b3 = torch.randn((10,), dtype=torch.float32)

        W1 = ttnn._rand(shape=(128, 28 * 28), dtype=ttnn.float32, device=device)
        b1 = ttnn._rand(shape=(128,),         dtype=ttnn.float32, device=device)
        W2 = ttnn._rand(shape=(64, 128),      dtype=ttnn.float32, device=device)
        b2 = ttnn._rand(shape=(64,),          dtype=ttnn.float32, device=device)
        W3 = ttnn._rand(shape=(10, 64),       dtype=ttnn.float32, device=device)
        b3 = ttnn._rand(shape=(10,),          dtype=ttnn.float32, device=device)

        correct = 0
        total   = 0

        for i, (image, label) in enumerate(testloader):
            if i >= 5:
                break

            # TODO: Review Suggestion
            # The image.view(1, -1).to(ttnn.float32) call is problematic because image is 
            # already a ttnn.Tensor from line 34, but the .to() method expects a ttnn DataType, 
            # not a ttnn tensor method. 
            # This should be ttnn.to_layout(image.view(1, -1), dtype=ttnn.float32) or similar.
            image = image.view(1, -1).to(ttnn.float32)

            # Input tensor
            image_tt = ttnn.from_torch(image, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            image_tt = ttnn.to_layout(image_tt, ttnn.TILE_LAYOUT)

            #torch tensor transpose W1.T
            #torch tensor view b1.view(1, -1)
            #ttnn.argmax -- tensor item()

            # Layer 1
            W1_tt = ttnn.from_torch(W1.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            W1_tt = ttnn.to_layout(W1_tt, ttnn.TILE_LAYOUT)
            b1_tt = ttnn.from_torch(b1.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            b1_tt = ttnn.to_layout(b1_tt, ttnn.TILE_LAYOUT)


            out1  = ttnn.linear(image_tt, W1_tt, bias=b1_tt)
            out1  = ttnn.relu(out1)

            # Layer 2
            W2_tt = ttnn.from_torch(W2.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            W2_tt = ttnn.to_layout(W2_tt, ttnn.TILE_LAYOUT)
            b2_tt = ttnn.from_torch(b2.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            b2_tt = ttnn.to_layout(b2_tt, ttnn.TILE_LAYOUT)
            out2 = ttnn.linear(out1, W2_tt, bias=b2_tt)
            out2 = ttnn.relu(out2)

            # Layer 3
            W3_tt = ttnn.from_torch(W3.T, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            W3_tt = ttnn.to_layout(W3_tt, ttnn.TILE_LAYOUT)
            b3_tt = ttnn.from_torch(b3.view(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            b3_tt = ttnn.to_layout(b3_tt, ttnn.TILE_LAYOUT)
            out3 = ttnn.linear(out2, W3_tt, bias=b3_tt)

            # Convert result back to torch
            prediction = ttnn.to_torch(out3)


            predicted_label = ttnn.argmax(prediction, dim=1).item()

            g = device.get_graph()
            g.graph2onnx('ttnn_mlp_inference_mnist.onnx', do_model_check=False,
                         filter_op_attrs=filter_ttnn_attrs)

            correct += predicted_label == label.item()
            total += 1

            logger.info(f"Sample {i+1}: Predicted={predicted_label}, Actual={label.item()}")

        logger.info(f"\nTT-NN MLP Inference Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
