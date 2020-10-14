#!/bin/bash

python3 ../train.py \
        --experiment-class sgdepth_eccv_test \
        --model-name zhou_full \
        --depth-training-loaders "kitti_zhou_train" \
        --train-batches-per-epoch 7293 \
        --masking-enable \
        --masking-from-epoch 15 \
        --masking-linear-increase

