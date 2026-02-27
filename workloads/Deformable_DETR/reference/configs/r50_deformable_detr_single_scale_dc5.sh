#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale_dc5
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --dilation \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
