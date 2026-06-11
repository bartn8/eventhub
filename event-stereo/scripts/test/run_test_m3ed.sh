#!/bin/bash

DATA_PATH="datasets/M3ED/processed"
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

# MIX 1 training 
mkdir -p results/m3ed/indoor/mix1/
mkdir -p results/m3ed/outdoor_day/mix1/
mkdir -p results/m3ed/outdoor_night/mix1/

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/baseline/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix1/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix1/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix1/${ER}_baseline.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix1/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix1/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix1/${ER}_foundationstereo.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix1/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix1/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix1/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix1/${ER}_stereoanywhere.txt

# MIX 2 training
mkdir -p results/m3ed/indoor/mix2/
mkdir -p results/m3ed/outdoor_day/mix2/
mkdir -p results/m3ed/outdoor_night/mix2/

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/baseline/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix2/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix2/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix2/${ER}_baseline.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix2/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix2/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix2/${ER}_foundationstereo.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix2/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix2/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix2/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix2/${ER}_stereoanywhere.txt

# MIX 3 training
mkdir -p results/m3ed/indoor/mix3/
mkdir -p results/m3ed/outdoor_day/mix3/
mkdir -p results/m3ed/outdoor_night/mix3/

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/baseline/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix3/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix3/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix3/${ER}_baseline.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix3/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix3/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix3/${ER}_foundationstereo.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix3/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix3/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix3/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix3/${ER}_stereoanywhere.txt

# MIX 4 training
mkdir -p results/m3ed/indoor/mix4/
mkdir -p results/m3ed/outdoor_day/mix4/
mkdir -p results/m3ed/outdoor_night/mix4/

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/baseline/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix4/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix4/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix4/${ER}_baseline.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix4/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix4/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix4/${ER}_foundationstereo.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/mix4/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/mix4/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/mix4/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/mix4/${ER}_stereoanywhere.txt

# LiDAR Supervised training

mkdir -p results/m3ed/indoor/lidar_gt/
mkdir -p results/m3ed/outdoor_day/lidar_gt/
mkdir -p results/m3ed/outdoor_night/lidar_gt/

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/baseline/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/lidar_gt/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/lidar_gt/${ER}_baseline.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/baseline/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/lidar_gt/${ER}_baseline.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/lidar_gt/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/lidar_gt/${ER}_foundationstereo.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/foundationstereo/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/lidar_gt/${ER}_foundationstereo.txt

CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_indoor 1>> results/m3ed/indoor/lidar_gt/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_day 1>> results/m3ed/outdoor_day/lidar_gt/${ER}_stereoanywhere.txt
CUDA_VISIBLE_DEVICES=0 python src/inference.py --test_config configs/test/m3ed/config_${ER}.yaml --data_root $DATA_PATH --checkpoint_path trainings/lidar_gt/${ER}/stereoanywhere/weights/final.pth --save_root tmp --split val_outdoor_night 1>> results/m3ed/outdoor_night/lidar_gt/${ER}_stereoanywhere.txt
