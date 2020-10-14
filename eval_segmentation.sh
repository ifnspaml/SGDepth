#!/bin/bash

python3 eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-name "eval_cs" \
        --model-load sgdepth_eccv_test/zhou_full/checkpoints/epoch_20 \
        --segmentation-validation-loaders "cityscapes_validation"

python3 eval_segmentation.py \
        --sys-best-effort-determinism \
        --model-name "eval_kitti" \
        --model-load sgdepth_eccv_test/zhou_full/checkpoints/epoch_20 \
        --segmentation-validation-loaders "kitti_2015_train" \
        --segmentation-validation-resize-width 640 \
        --segmentation-validation-resize-height 192