#!/bin/sh

NUM_WORKERS=26
WORK_DIR="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo"
STDERR_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/tmp/stderr_scannet_train_"
STDOUT_PATH="/leonardo_scratch/fast/IscrC_VFM4Ev/projects/ev-nerf-stereo/tmp/stdout_scannet_train_"

# for i in $(seq 0 $((NUM_WORKERS - 1)))
for i in $(seq 0 $((NUM_WORKERS - 1)))
do
    echo "Submitting job for worker ID: $i"
    sbatch -D $WORK_DIR --job-name=$i-scannet-train --export=WORKER_ID=$i -o ${STDOUT_PATH}${i}.txt -e ${STDERR_PATH}${i}.txt scripts/svraster_train/svraster_train_leonardo_i.sh
done
