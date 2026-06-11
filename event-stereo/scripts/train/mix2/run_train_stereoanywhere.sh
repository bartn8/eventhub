#!/bin/bash

DATA_ROOT_VAL="datasets/dsec"
# compute codebase path from this script location (one level up to event-stereo)
SCRIPT_DIR="$( dirname "$( realpath "$0" )" )"
CODEBASE_PATH="$( realpath "$SCRIPT_DIR/../.." )"
USERNAME="$(whoami)"
CONDA_PATH="/home/${USERNAME}/miniconda3/bin/conda"
CONDA_ENV=eventhub
ER="tencode"

cd $CODEBASE_PATH
eval "$( $CONDA_PATH shell.bash hook)"
conda activate $CONDA_ENV

RESUME=""
# RESUME="--resume"

SEED="--seed 42"

DATALOADER_WORKERS="--num_workers 8"

PRETRAIN="--pretrain_path $CODEBASE_PATH/weights/stereoanywhere_vda_vits.pth"

echo "CUDA_VISIBLE_DEVICES=0 python src/main.py --data_root_validation $DATA_ROOT_VAL --validate --save_root trainings/mix2/${ER}/stereoanywhere \
  --config_path configs/train/stereoanywhere/mix2/config_${ER}.yaml $SEED $RESUME $DATALOADER_WORKERS $PRETRAIN"

CUDA_VISIBLE_DEVICES=0 python src/main.py --data_root_validation $DATA_ROOT_VAL --validate --save_root trainings/mix2/${ER}/stereoanywhere \
  --config_path configs/train/stereoanywhere/mix2/config_${ER}.yaml $SEED $RESUME $DATALOADER_WORKERS $PRETRAIN
