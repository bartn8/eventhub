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

PRETRAIN="--pretrain_path $CODEBASE_PATH/weights/stereoanywhere_vda_vits.pth"

echo "CUDA_VISIBLE_DEVICES=0 python src/main.py --data_root_validation $DATA_ROOT_VAL --validate --save_root trainings/mix4/${ER}/stereoanywhere \
  --config_path configs/train/stereoanywhere/mix4/config_${ER}.yaml $SEED $RESUME $DATALOADER_WORKERS $PRETRAIN"

CUDA_VISIBLE_DEVICES=0 python src/main.py --data_root_validation $DATA_ROOT_VAL --validate --save_root trainings/mix4/${ER}/stereoanywhere \
  --config_path configs/train/stereoanywhere/mix4/config_${ER}.yaml $SEED $RESUME $DATALOADER_WORKERS $PRETRAIN
