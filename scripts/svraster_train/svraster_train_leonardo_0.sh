#!/bin/sh
#SBATCH --job-name=nsd-train-0
#SBATCH --partition=boost_usr_prod
#SBATCH -o /leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/tmp/stdout_nsd_train_0.txt
#SBATCH -e /leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/tmp/stderr_nsd_train_0.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_D24_081

# source "/home/luca/miniconda3/etc/profile.d/conda.sh"
# conda activate nerfstudio

WORKER_ID=0
NUM_WORKERS=1

VENV_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/nerfstudio/bin/activate"
CODEBASE_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/svraster"
DATASET_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/datasets/nerfstereo-dataset"
OUTPUT_PATH="/leonardo_work/IscrC_VFM4Ev/datasets/nerfstereo-dataset-svraster-half-2"
MAST3R_REPO_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/mast3r"

echo "Worker ID: $WORKER_ID"

# Activate the venv environment
source $VENV_PATH
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_PATH
# Change to the codebase directory
cd $CODEBASE_PATH

module load cuda/12.3
module load ninja
module load gcc

# Fixed parameters
FIXED_PARAMS="--res_downscale 2 --vox_geo_mode triinterp3 --save_quantized"
FIXED_PARAMS+=" --mast3r_repo_path $MAST3R_REPO_PATH"
# FIXED_PARAMS+=" --save_optimizer --checkpoint_iterations 10000"
FIXED_PARAMS+=" --bound_mode forward"
FIXED_PARAMS+=" --lambda_normal_dmean 0.0005 --lambda_normal_dmed 0.0005"
FIXED_PARAMS+=" --lambda_depthanythingv2 0.01 --lambda_T_inside 0.01"
FIXED_PARAMS+=" --lambda_mast3r_metric_depth 0.0 --lambda_ascending 0.01 --lambda_sparse_depth 0.0"

SCENE_COUNTER=0
for i in $DATASET_PATH/0105
do
    # Check if this combination is for my ID
    if [ $((SCENE_COUNTER % NUM_WORKERS)) -ne $WORKER_ID ]; then
        SCENE_COUNTER=$((SCENE_COUNTER + 1))
        continue
    fi

    # MOD_RESULT=`expr "$SCENE_COUNTER" % "$NUM_WORKERS"`
    # if [ "$MOD_RESULT" -ne "$WORKER_ID" ]; then
    #     SCENE_COUNTER=`expr "$SCENE_COUNTER" + 1`
    #     continue
    # fi

    #get last part of the path
    i=${i##*/}
    
    echo "SVRaster training on scene id $i (Scene SEQ: $SCENE_COUNTER; Worker: $WORKER_ID; Num Workers: $NUM_WORKERS)"
    echo "Dataset path: $DATASET_PATH/$i"

    # Check if this scene has already been trained
    CHECK_PATH="$OUTPUT_PATH/${i}/checkpoints/iter020000_model.pt"
    if [ -f "$CHECK_PATH" ]; then
        echo "This scene has already been trained. Skipping..."
        continue
    fi

    echo "Output path: $OUTPUT_PATH/${i}"
    
    # Run the training script 
    # $i inside output path is created inside train.py
    echo "train.py --source_path $DATASET_PATH/$i/ --model_path $OUTPUT_PATH $FIXED_PARAMS"
    python train.py --source_path $DATASET_PATH/$i/ --model_path $OUTPUT_PATH $FIXED_PARAMS 

    SCENE_COUNTER=$((SCENE_COUNTER + 1))
done                        

echo "All combinations processed for worker $WORKER_ID."
