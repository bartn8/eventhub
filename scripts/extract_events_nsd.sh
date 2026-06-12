#!/bin/bash

# Required env vars: CODEBASE_PATH, DATASET_PATH, OUTPUT_PATH

set -eu

: "${CODEBASE_PATH:?Set CODEBASE_PATH to the svraster codebase}"
: "${DATASET_PATH:?Set DATASET_PATH to the input dataset root}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the output dataset root}"

FOCAL_SCALE=1.25
WIDTH=640
HEIGHT=480

BASELINE_RND=""
BASELINE_STD=0.05
BASELINES="0.1 0.3 0.5"

WHAT_TO_RENDERS_LEFT="color"
WHAT_TO_RENDERS_CENTER="color depth depth_median ao sizeconf alpha eventsv2 forward_flow backward_flow"
WHAT_TO_RENDERS_RIGHT="color eventsv2"

START_ID=0
POSE_WINDOW_SIZE=-1
MAX_POSES=30
DURATIONS="0.01 0.02 0.05 0.1 0.2 0.5 1"
MAX_SCENES=1000
INITIAL_NUM_BISECTIONS=5
M_WEIGHT=0.0
REFRACTORY_US=0
MIN_DT_PERC=0.025
MAX_DT_PERC=0.1
MIN_EVENTS_PER_FRAME=50000
MAX_EVENTS_PER_FRAME=650000
# VELOCITY_FNS="cos2 1.0 0.0|cos2 2.0 0.0|cos2 4.0 0.0|linear|linear|linear|gamma 0.25|gamma 0.5|gamma 2.0"

FIXED_ARGS=""
FIXED_ARGS="$FIXED_ARGS --fix_aspect_ratio"
FIXED_ARGS="$FIXED_ARGS --baselines $BASELINES"
FIXED_ARGS="$FIXED_ARGS $BASELINE_RND"
FIXED_ARGS="$FIXED_ARGS --rnd_baseline_std $BASELINE_STD"
FIXED_ARGS="$FIXED_ARGS --focal_scale $FOCAL_SCALE"
FIXED_ARGS="$FIXED_ARGS --width $WIDTH"
FIXED_ARGS="$FIXED_ARGS --height $HEIGHT"
FIXED_ARGS="$FIXED_ARGS --render_choices_left $WHAT_TO_RENDERS_LEFT"
FIXED_ARGS="$FIXED_ARGS --render_choices_center $WHAT_TO_RENDERS_CENTER"
FIXED_ARGS="$FIXED_ARGS --render_choices_right $WHAT_TO_RENDERS_RIGHT"
FIXED_ARGS="$FIXED_ARGS --rnd_contrast_thresholds"
FIXED_ARGS="$FIXED_ARGS --refractory_us $REFRACTORY_US"
FIXED_ARGS="$FIXED_ARGS --min_dt_perc $MIN_DT_PERC"
FIXED_ARGS="$FIXED_ARGS --max_dt_perc $MAX_DT_PERC"
FIXED_ARGS="$FIXED_ARGS --min_events_per_frame $MIN_EVENTS_PER_FRAME"
FIXED_ARGS="$FIXED_ARGS --max_events_per_frame $MAX_EVENTS_PER_FRAME"
FIXED_ARGS="$FIXED_ARGS --initial_num_bisections $INITIAL_NUM_BISECTIONS"
FIXED_ARGS="$FIXED_ARGS --rnd_poses"
FIXED_ARGS="$FIXED_ARGS --max_poses $MAX_POSES"
FIXED_ARGS="$FIXED_ARGS --super_sampling 1.5"
FIXED_ARGS="$FIXED_ARGS --gauss_blur_kernel 3"
FIXED_ARGS="$FIXED_ARGS --gauss_noise_std 0.00125"
FIXED_ARGS="$FIXED_ARGS --rnd_gain_min 0.7"
FIXED_ARGS="$FIXED_ARGS --rnd_offset_max 0.0025"
FIXED_ARGS="$FIXED_ARGS --num_noise_events 100"
FIXED_ARGS="$FIXED_ARGS --rnd_invert_t"

# Activate the venv environment
USERNAME="$(whoami)"
CONDA_PATH="/home/${USERNAME}/miniconda3/bin/conda"
CONDA_ENV=eventhub
eval "$( $CONDA_PATH shell.bash hook)"
conda activate $CONDA_ENV

# Change to the codebase directory
cd "$CODEBASE_PATH"

SCENE_COUNTER=0
for i in "$DATASET_PATH"/*
do
    scene_id=${i##*/}

    if [ -d "$OUTPUT_PATH/$scene_id" ]; then
        echo "Output folder $OUTPUT_PATH/$scene_id already exists. Skipping scene $scene_id."
        SCENE_COUNTER=$((SCENE_COUNTER + 1))
        continue
    fi

    for traj in h v z
    do
        SEED=$(date +%s)
        SEED=$((SEED + SCENE_COUNTER))

        # # Randomly select a velocity function from the list
        # FN_INDEX=$((RANDOM % ${#VELOCITY_FNS[@]}))
        # VEL_FN_STR=${VELOCITY_FNS[$FN_INDEX]}
        VEL_FN_STR="linear"

        # Randomly select a duration from the list
        DUR_INDEX=$((RANDOM % ${#DURATIONS[@]}))
        DURATION=${DURATIONS[$DUR_INDEX]}

        echo "Processing scene: $i with trajectory: $traj"
        echo "python myrender.py $i ../configs/param_traj/$traj.json \
            --seed $SEED --output_postfix _$traj --output_dir $OUTPUT_PATH/$scene_id \
            $FIXED_ARGS --velocity_fn $VEL_FN_STR --duration $DURATION"

        python myrender.py "$i" "../configs/param_traj/$traj.json" \
            --seed "$SEED" --output_postfix "_$traj" --output_dir "$OUTPUT_PATH/$scene_id" \
            $FIXED_ARGS --velocity_fn "$VEL_FN_STR" --duration "$DURATION"
    done

    SCENE_COUNTER=$((SCENE_COUNTER + 1))
    if [ $SCENE_COUNTER -ge $MAX_SCENES ]; then
        break
    fi
done

echo "All combinations processed."
