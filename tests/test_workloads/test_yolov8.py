#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import workloads.Yolo_v8 as y8

def test_yolov8s(session_temp_directory):
    output_dir = str(session_temp_directory)
    os.makedirs(output_dir, exist_ok=True)
    cfg_dir  = 'config/yolov8_cfgs/'
    cfg_file = 'yolov8s.yaml'
    cfg_path = os.path.join(cfg_dir, cfg_file)
    out_onnx = os.path.join(output_dir, cfg_file.replace('.yaml', '.onnx'))

    # Create the YOLOv8s object with the specified configuration
    yolo_obj = y8.YOLO8('yolov8s', {
        'bs'           : 1,
        'in_channels'  : 3,
        'in_resolution': 640,
        'yaml_cfg_path': cfg_path,
        })

    param_count = yolo_obj.analytical_param_count()
    print(f"    #params= {param_count/1e6:.2f}M")
    yolo_obj.create_input_tensors()
    print("    input shape=", yolo_obj.input_tensors['yolo_input'].shape)
    yolo_out = yolo_obj()
    box_outputs, cls_outputs = yolo_out
    print("    box output shapes=", [y.shape for y in box_outputs])
    print("    class output shapes=", [y.shape for y in cls_outputs])
    gg = yolo_obj.get_forward_graph()
    print("    exporting to onnx:", out_onnx)
    gg.graph2onnx(out_onnx)
    print()
