#!/bin/bash

python3 eval_depth.py\
        --sys-best-effort-determinism \
        --model-name "eval_kitti_depth" \
        --model-load sgdepth_eccv_test/zhou_full/checkpoints/epoch_20 \
        --depth-validation-loaders "kitti_zhou_test"