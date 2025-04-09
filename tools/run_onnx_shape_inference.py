#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import argparse
import onnx
from onnx import helper, shape_inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_onnx_shape_inference')

    parser.add_argument('--input',  '-i', required=True, help="Input ONNX File")
    args = parser.parse_args()

    model_file = args.input
    assert model_file.endswith('.onnx'), f"Input Onnx File:{model_file} should end as \".onnx\"!!"

    o_model_file = model_file.replace('.onnx','') + '.shape_inf.onnx'
    print(model_file)
    print(o_model_file)


    model = onnx.load(model_file)
    onnx.checker.check_model(model)

    o_model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(o_model)
    onnx.save(o_model, o_model_file)
