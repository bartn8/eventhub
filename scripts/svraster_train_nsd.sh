#!/bin/sh

# Based on nerfstereo (NSD) training config.
# Required env vars: VENV_PATH, CODEBASE_PATH, DATASET_PATH, OUTPUT_PATH, MAST3R_REPO_PATH

set -eu

: "${VENV_PATH:?Set VENV_PATH to your venv activate script}"
: "${CODEBASE_PATH:?Set CODEBASE_PATH to the svraster codebase}"
: "${DATASET_PATH:?Set DATASET_PATH to the nerfstereo dataset root}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the svraster outputs root}"
# : "${MAST3R_REPO_PATH:?Set MAST3R_REPO_PATH to the mast3r repo}"

# Activate the venv environment
# shellcheck disable=SC1090
. "$VENV_PATH"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Change to the codebase directory
cd "$CODEBASE_PATH"

# Fixed parameters
FIXED_PARAMS="--res_downscale 2 --vox_geo_mode triinterp3 --save_quantized"
FIXED_PARAMS+=" --bound_mode forward"
FIXED_PARAMS+=" --lambda_normal_dmean 0.0005 --lambda_normal_dmed 0.0005"
FIXED_PARAMS+=" --lambda_depthanythingv2 0.01 --lambda_T_inside 0.01"
FIXED_PARAMS+=" --lambda_mast3r_metric_depth 0.0 --lambda_ascending 0.0 --lambda_sparse_depth 0.0"
# FIXED_PARAMS="$FIXED_PARAMS --mast3r_repo_path $MAST3R_REPO_PATH"

SCENE_COUNTER=0
for i in "$DATASET_PATH"/*
do
    scene_id=${i##*/}

    mkdir -p "$OUTPUT_PATH/$scene_id"

    echo "SVRaster training on scene id $scene_id (Scene SEQ: $SCENE_COUNTER)"
    echo "Dataset path: $DATASET_PATH/$scene_id"
    echo "Output path: $OUTPUT_PATH/$scene_id"
    echo "----------------------------------------"

    echo "train.py --source_path $DATASET_PATH/$scene_id/ --model_path $OUTPUT_PATH/$scene_id $FIXED_PARAMS"
    python train.py --source_path "$DATASET_PATH/$scene_id/" --model_path "$OUTPUT_PATH/$scene_id" $FIXED_PARAMS

    SCENE_COUNTER=$((SCENE_COUNTER + 1))
done

echo "All combinations processed."
