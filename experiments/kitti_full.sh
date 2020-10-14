#!/bin/bash

python3 ../train.py \
        --experiment-class sgdepth_eccv_test \
        --model-name kitti_full \
        --masking-enable \
        --masking-from-epoch 15 \
        --masking-linear-increase
