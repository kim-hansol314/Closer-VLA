#!/bin/bash

#SBATCH --job-name=modify-rlds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --time=23:59:59
#SBATCH --job-name=rlds
#SBATCH --output=logs/rlds-%J.log
#SBATCH --error=logs/rlds-%J.err

# increase limit on number of files opened in parallel to 20k --> conversion opens up to 1k temporary files in /tmp to store dataset during conversion
ulimit -n 20000
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TF_FORCE_GPU_ALLOW_GROWTH=true  # 启用动态显存分配
# dataset: bridge_dataset : $VLA_DATA_DIR=/DATA/disk0/lyx/dataset/Bridge, 
# or fractal20220817_data : $VLA_DATA_DIR=/DATA/disk0/lyx/dataset/Fractal,
# or libero: 
# $VLA_DATA_DIR=/DATA/disk0/lyx/dataset/Libero/libero_spatial_no_noops_with_depth
# $VLA_DATA_DIR=/DATA/disk0/lyx/dataset/Libero/libero_object_no_noops_with_depth
# $VLA_DATA_DIR=/DATA/disk0/lyx/dataset/Libero/libero_goal_no_noops_with_depth
# $VLA_DATA_DIR=/DATA/disk0/lyx/dataset/Libero/libero_10_no_noops_with_depth
# ps : use whole libero dataset for training
# $VLA_DATA_DIR=/DATA/disk0/lyx/dataset/Libero/libero_whole_datasets
# subdataset names:
# libero_dataset # the same
# $TRANSFORMERS_CACHE=/DATA/disk0/lyx/dataset/paligemma_model/huggingface_cache
python scripts/data/modify_rlds_dataset.py \
    --dataset=bridge_dataset \
    --data_dir=$VLA_DATA_DIR \
    --target_dir=$VLA_DATA_DIR/resize_224 \
    --mods=resize_and_jpeg_encode \
    --n_workers=20 \
    --max_episodes_in_memory=200
