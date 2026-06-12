#!/bin/sh

# Clean local run script (no slurm, no workers). Uses GEN3 config.
# Required env vars: CODEBASE_PATH, DATASET_PATH, DATASET_MESH_PATH, OUTPUT_PATH

set -eu

: "${CODEBASE_PATH:?Set CODEBASE_PATH to the svraster codebase}"
: "${DATASET_PATH:?Set DATASET_PATH to the scannet++_train_nvs_svraster root}"
: "${DATASET_MESH_PATH:?Set DATASET_MESH_PATH to the scannet++_train_nvs root}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the output dataset root}"

FOCAL_SCALE=1
WIDTH=640
HEIGHT=480

BASELINE_RND=""
BASELINE_STD=0.05
BASELINES="0.05 0.08 0.1"

WHAT_TO_RENDERS_LEFT="color"
WHAT_TO_RENDERS_CENTER="color depth depth_median depth_mesh ao sizeconf alpha eventsv2"
WHAT_TO_RENDERS_RIGHT="color eventsv2"

N_SAMPLES=30
START_ID=0
POSE_WINDOW_SIZE=-1
DURATIONS="60 120 180 240 300"
MAX_SCENES=1000
INITIAL_NUM_BISECTIONS=6
M_WEIGHT=1.0
REFRACTORY_US=0
MIN_DT_PERC=0.05
MAX_DT_PERC=0.25
MIN_EVENTS_PER_FRAME=50000
MAX_EVENTS_PER_FRAME=650000
VELOCITY_FNS="cos2 1.0 0.0|cos2 2.0 0.0|cos2 4.0 0.0|linear|linear|linear|gamma 0.25|gamma 0.5|gamma 2.0"

FIXED_ARGS=""
FIXED_ARGS="$FIXED_ARGS --fix_aspect_ratio"
FIXED_ARGS="$FIXED_ARGS --baselines $BASELINES"
FIXED_ARGS="$FIXED_ARGS $BASELINE_RND"
FIXED_ARGS="$FIXED_ARGS --rnd_baseline_std $BASELINE_STD"
FIXED_ARGS="$FIXED_ARGS --focal_scale $FOCAL_SCALE"
FIXED_ARGS="$FIXED_ARGS --width $WIDTH"
FIXED_ARGS="$FIXED_ARGS --height $HEIGHT"
FIXED_ARGS="$FIXED_ARGS --n_samples $N_SAMPLES"
FIXED_ARGS="$FIXED_ARGS --start_id $START_ID"
FIXED_ARGS="$FIXED_ARGS --pose_window_size $POSE_WINDOW_SIZE"
FIXED_ARGS="$FIXED_ARGS --render_choices_left $WHAT_TO_RENDERS_LEFT"
FIXED_ARGS="$FIXED_ARGS --render_choices_center $WHAT_TO_RENDERS_CENTER"
FIXED_ARGS="$FIXED_ARGS --render_choices_right $WHAT_TO_RENDERS_RIGHT"
FIXED_ARGS="$FIXED_ARGS --rnd_contrast_thresholds"
FIXED_ARGS="$FIXED_ARGS --lsq_spline"
FIXED_ARGS="$FIXED_ARGS --m_weight $M_WEIGHT"
FIXED_ARGS="$FIXED_ARGS --refractory_us $REFRACTORY_US"
FIXED_ARGS="$FIXED_ARGS --min_dt_perc $MIN_DT_PERC"
FIXED_ARGS="$FIXED_ARGS --max_dt_perc $MAX_DT_PERC"
FIXED_ARGS="$FIXED_ARGS --min_events_per_frame $MIN_EVENTS_PER_FRAME"
FIXED_ARGS="$FIXED_ARGS --max_events_per_frame $MAX_EVENTS_PER_FRAME"
FIXED_ARGS="$FIXED_ARGS --initial_num_bisections $INITIAL_NUM_BISECTIONS"
FIXED_ARGS="$FIXED_ARGS --clip_z_th 45 55"
FIXED_ARGS="$FIXED_ARGS --super_sampling 1.5"
FIXED_ARGS="$FIXED_ARGS --gauss_blur_kernel 3"
FIXED_ARGS="$FIXED_ARGS --gauss_noise_std 0.00125"
FIXED_ARGS="$FIXED_ARGS --rnd_gain_min 0.7"
FIXED_ARGS="$FIXED_ARGS --rnd_offset_max 0.0025"
FIXED_ARGS="$FIXED_ARGS --num_noise_events 100"
FIXED_ARGS="$FIXED_ARGS --rnd_invert_t"
FIXED_ARGS="$FIXED_ARGS --spline_offset -0.1"

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
    hash=${i##*/}

    if [ -d "$OUTPUT_PATH/$hash" ]; then
        echo "Output folder $OUTPUT_PATH/$hash already exists. Skipping scene $hash."
        SCENE_COUNTER=$((SCENE_COUNTER + 1))
        continue
    fi

    for traj in param0
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
        echo "PYOPENGL_PLATFORM=egl python myrender_fly_through_limit.py $i/svraster_inputs ../configs/param_traj/$traj.json \
            --mesh_path $DATASET_MESH_PATH/$hash/scans/mesh_aligned_0.05.ply \
            --boundary_poses_idxs_path $DATASET_PATH/$hash/svraster_inputs/boundary_pose_indices.txt \
            --seed $SEED --output_postfix _$traj --output_dir $OUTPUT_PATH/$hash \
            $FIXED_ARGS --velocity_fn $VEL_FN_STR --duration $DURATION"

        PYOPENGL_PLATFORM=egl python myrender_fly_through_limit.py "$i/svraster_inputs" "../configs/param_traj/$traj.json" \
            --mesh_path "$DATASET_MESH_PATH/$hash/scans/mesh_aligned_0.05.ply" \
            --boundary_poses_idxs_path "$DATASET_PATH/$hash/svraster_inputs/boundary_pose_indices.txt" \
            --seed "$SEED" --output_postfix "_$traj" --output_dir "$OUTPUT_PATH/$hash" \
            $FIXED_ARGS --velocity_fn "$VEL_FN_STR" --duration "$DURATION"
    done

    SCENE_COUNTER=$((SCENE_COUNTER + 1))
    if [ $SCENE_COUNTER -ge $MAX_SCENES ]; then
        break
    fi
done

echo "All combinations processed."
