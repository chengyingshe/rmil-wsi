#!/bin/bash

# The coordinates of the patches are saved, you can change the path to any directory you want.
save_dir="/home/scy/changhai_project/wsi/rmil-wsi/datasets/mydatasets/camelyon16-temp/patches"
# The directory where the WSI files are stored, you can change it to any directory you want.
wsi_dir="/home/scy/changhai_project/wsi/rmil-wsi/datasets/mydatasets/camelyon16-temp/wsi"
# The WSI format, you can set it based on the format of your WSI
wsi_format="tif"
# The log file name, you can change it to any name you want.
log_name="SAL.log"

# Normally, you don't need to change following lines.
# to set the patch size, please set it at `configs/resolution.py`
python create_patches_fp.py \
        --source $wsi_dir \
        --save_dir $save_dir\
        --preset tcga.csv \
        --patch_level 0 \
        --wsi_format $wsi_format \
        --seg \
        --patch \
        --stitch