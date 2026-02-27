#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

set -x

EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    ${PY_ARGS}
