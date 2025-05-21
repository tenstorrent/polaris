#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import workloads.Yolo_v7 as y7

def test_yolov7(session_temp_directory):
    output_dir = str(session_temp_directory)
    os.makedirs(output_dir, exist_ok=True)
    cfg_dir  = 'config/yolov7_cfgs/deploy'
    cfg_file = 'yolov7-tiny.yaml'
    cfg_path = os.path.join(cfg_dir, cfg_file)
    out_onnx = os.path.join(output_dir, cfg_file.replace('.yaml', '.onnx'))
    yolo_obj = y7.YOLO7('yolov7-tiny', {
        'bs'           : 1,
        'in_channels'  : 3,
        'in_resolution': 256,
        'yaml_cfg_path': cfg_path,
        })
    param_count = yolo_obj.analytical_param_count()
    print(f"    #params= {param_count/1e6:.2f}M")
    yolo_obj.create_input_tensors()
    print("    input shape=", yolo_obj.input_tensors['yolo_input'].shape)
    yolo_out = yolo_obj()
    print("    output shapes=", [y.shape for y in yolo_out])
    gg = yolo_obj.get_forward_graph()
    print("    exporting to onnx:", out_onnx)
    gg.graph2onnx(out_onnx)
    print()
