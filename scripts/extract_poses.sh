#!/bin/sh
#SBATCH --job-name=ex-poses-scannet
#SBATCH --partition=boost_usr_prod
#SBATCH -o /leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/tmp/stdout_extract_poses_scannet.txt
#SBATCH -e /leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/tmp/stderr_extract_poses_scannet.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_D24_081

VENV_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/nerfstudio/bin/activate"
CODEBASE_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/svraster"
DATASET_PATH="/leonardo_work/EUHPC_D24_081/datasets/scannet++_train_nvs_svraster"

# Activate the venv environment
source $VENV_PATH
# Change to the codebase directory
cd $CODEBASE_PATH

module load cuda/12.3
module load ninja
module load gcc

CUDA_VISIBLE_DEVICES=0 python extract_poses.py $DATASET_PATH/*/svraster_inputs/