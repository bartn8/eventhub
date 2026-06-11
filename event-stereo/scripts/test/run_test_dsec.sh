#!/bin/bash

DATA_PATH="datasets/dsec"
# compute codebase path from this script location (one level up to event-stereo)
SCRIPT_DIR="$( dirname "$( realpath "$0" )" )"
CODEBASE_PATH="$( realpath "$SCRIPT_DIR/../.." )"
USERNAME="$(whoami)"
CONDA_PATH="/home/${USERNAME}/miniconda3/bin/conda"
CONDA_ENV=eventhub
ER=tencode

printf "DATA_PATH: %s\n" "$DATA_PATH"
printf "CODEBASE_PATH: %s\n" "$CODEBASE_PATH"
printf "CONDA_PATH: %s\n" "$CONDA_PATH"
printf "CONDA_ENV: %s\n" "$CONDA_ENV"
printf "ER: %s\n" "$ER"

cd $CODEBASE_PATH
eval "$( $CONDA_PATH shell.bash hook)"
conda activate $CONDA_ENV

mkdir -p results/dsec/val/mix1/
mkdir -p results/dsec/val/mix2/
mkdir -p results/dsec/val/mix3/
mkdir -p results/dsec/val/mix4/
mkdir -p results/dsec/val/lidar_gt/


# MIX 1 training 
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/baseline/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix1/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/foundationstereo/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix1/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix1/${ER}_stereoanywhere.txt

# MIX 2 training
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/baseline/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix2/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/foundationstereo/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix2/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix2/${ER}_stereoanywhere.txt

# MIX 3 training
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/baseline/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix3/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/foundationstereo/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix3/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix3/${ER}_stereoanywhere.txt

# MIX 4 training
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/baseline/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix4/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/foundationstereo/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix4/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/mix4/${ER}_stereoanywhere.txt

# LiDAR Supervised training
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/baseline/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/lidar_gt/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/foundationstereo/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/lidar_gt/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/dsec/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split validation 1>> results/dsec/val/lidar_gt/${ER}_stereoanywhere.txt
