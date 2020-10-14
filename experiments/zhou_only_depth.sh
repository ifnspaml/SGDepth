#!/bin/bash

python3 ../train.py \
        --experiment-class sgdepth_eccv_test \
        --model-name zhou_only_depth \
        --depth-training-loaders "kitti_zhou_train" \
        --train-batches-per-epoch 7293 \
        --depth-training-batch-size 12 \
        --segmentation-training-loaders "" \
        --train-depth-grad-scale 1.0 \
        --train-segmentation-grad-scale 0.0
