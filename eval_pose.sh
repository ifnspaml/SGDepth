#!/bin/bash

srun python3 eval_pose.py\
        --sys-best-effort-determinism \
        --model-name "eval_kitti_pose" \
        --model-load sgdepth_eccv_test/zhou_full/checkpoints/epoch_20 \
        --pose-validation-loaders "kitti_odom09_validation"
