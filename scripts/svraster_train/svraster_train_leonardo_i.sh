#!/bin/sh
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_D24_081
#SBATCH --mail-user=luca.bartolomei5@unibo.it
#SBATCH --mail-type=END,FAIL

# source "/home/luca/miniconda3/etc/profile.d/conda.sh"
# conda activate nerfstudio

# WORKER_ID=0
NUM_WORKERS=26

VENV_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/nerfstudio/bin/activate"
CODEBASE_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/svraster"
# DATASET_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/datasets/nerfstereo-dataset"
DATASET_PATH="/leonardo_work/EUHPC_D24_081/datasets/scannet++_train_nvs"
#/0a5c013435/dslr/svraster_inputs/"
OUTPUT_PATH="/leonardo_work/EUHPC_D24_081/datasets/scannet++_train_nvs_svraster"
MAST3R_REPO_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/mast3r"

echo "Worker ID: $WORKER_ID"
# Check if WORKER_ID is set, if not, default to 0
if [ -z "$WORKER_ID" ]; then
    WORKER_ID=0
    NUM_WORKERS=1
    echo "WORKER_ID not set. Defaulting to 0."
fi

# Activate the venv environment
source $VENV_PATH
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_PATH
# Change to the codebase directory
cd $CODEBASE_PATH

module load cuda/12.3
module load ninja
module load gcc

#python train.py --cfg_files cfg/scannetpp.yaml
# --vox_geo_mode triinterp3 --save_quantized --lambda_normal_dmean 0.0005
# --lambda_normal_dmed 0.0005 --lambda_depthanythingv2 0.01 --lambda_T_inside 0.01
# --lambda_mast3r_metric_depth 0.0 --lambda_ascending 0.01 --lambda_sparse_depth 0.0
# --res_downscale 1
# --source_path /home/luca/Scrivania/datasets/scannet++_train_nvs/0a5c013435/dslr/svraster_inputs/
#  --model_path /home/luca/Scrivania/datasets/scannet++_train_nvs_svraster/0a5c013435

# Fixed parameters
FIXED_PARAMS="--res_downscale 1 --vox_geo_mode triinterp3 --save_quantized"
FIXED_PARAMS+=" --cfg_files cfg/scannetpp.yaml"
FIXED_PARAMS+=" --mast3r_repo_path $MAST3R_REPO_PATH"
# FIXED_PARAMS+=" --save_optimizer --checkpoint_iterations 10000"
# FIXED_PARAMS+=" --bound_mode forward"
FIXED_PARAMS+=" --lambda_normal_dmean 0.0005 --lambda_normal_dmed 0.0005"
FIXED_PARAMS+=" --lambda_depthanythingv2 0.01 --lambda_T_inside 0.01"
FIXED_PARAMS+=" --lambda_mast3r_metric_depth 0.0 --lambda_ascending 0.01 --lambda_sparse_depth 0.0"

SCENE_COUNTER=0
for i in $DATASET_PATH/*
do
    # Check if this combination is for my ID
    if [ $((SCENE_COUNTER % NUM_WORKERS)) -ne $WORKER_ID ]; then
        SCENE_COUNTER=$((SCENE_COUNTER + 1))
        continue
    fi

    #get last part of the path
    i=${i##*/}
    
    echo "SVRaster training on scene id $i (Scene SEQ: $SCENE_COUNTER; Worker: $WORKER_ID; Num Workers: $NUM_WORKERS)"
    echo "Dataset path: $DATASET_PATH/$i/dslr/svraster_inputs/"

    # Check if this scene has already been trained
    # CHECK_PATH="$OUTPUT_PATH/${i}/checkpoints/iter020000_model.pt"
    CHECK_PATH="$OUTPUT_PATH/${i}/svraster_inputs/checkpoints/iter020000_model.pt"
    if [ -f "$CHECK_PATH" ]; then
        echo "This scene has already been trained. Skipping..."
        SCENE_COUNTER=$((SCENE_COUNTER + 1))
        continue
    fi

    echo "Output path: $OUTPUT_PATH/${i}"
    
    # Run the training script 
    # $i inside output path is created inside train.py
    echo "train.py --source_path $DATASET_PATH/$i/dslr/svraster_inputs/ --model_path $OUTPUT_PATH/${i} $FIXED_PARAMS"
    python train.py --source_path $DATASET_PATH/$i/dslr/svraster_inputs/ --model_path $OUTPUT_PATH/${i} $FIXED_PARAMS

    SCENE_COUNTER=$((SCENE_COUNTER + 1))
done                        

echo "All combinations processed for worker $WORKER_ID."
