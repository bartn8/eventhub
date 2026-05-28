#!/bin/bash

DATA_ROOT_VAL="event-stereo/datasets/dsec"
# compute codebase path from this script location (one level up to event-stereo)
SCRIPT_DIR="$( dirname "$( realpath "$0" )" )"
CODEBASE_PATH="$( realpath "$SCRIPT_DIR/../.." )"
CONDA_PATH="/leonardo_scratch/fast/EUHPC_D35_218/miniconda3/bin/conda"
CONDA_ENV=eventhub
ER="tencode"

cd $CODEBASE_PATH
eval "$( $CONDA_PATH shell.bash hook)"
conda activate $CONDA_ENV

RESUME=""
# RESUME="--resume"

SEED="--seed 42"

DATALOADER_WORKERS="--num_workers 8"

PRETRAIN=""
# PRETRAIN="--pretrain_path ..."

echo "CUDA_VISIBLE_DEVICES=0 python src/main.py --data_root_validation $DATA_ROOT_VAL --validate --save_root trainings/lidar_gt/${ER}/baseline \
  --config_path configs/train/baseline/lidar_gt/config_${ER}.yaml $SEED $RESUME $DATALOADER_WORKERS $PRETRAIN"

CUDA_VISIBLE_DEVICES=0 python src/main.py --data_root_validation $DATA_ROOT_VAL --validate --save_root trainings/lidar_gt/${ER}/baseline \
  --config_path configs/train/baseline/lidar_gt/config_${ER}.yaml $SEED $RESUME $DATALOADER_WORKERS $PRETRAIN
