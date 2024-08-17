#!/usr/bin/env bash
set -x

GPUS='0'
PORT=25503
GPUS_PER_NODE=1
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="resnet101"
OUTPUT_DIR="./results/OCPG_${BACKBONE}_scratch_a2d_boxsup"
EXP_NAME="OCPG_${BACKBONE}_scratch_a2d"
CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --with_box_refine --binary --freeze_text_encoder --supervision=box \
  --exp_name=${EXP_NAME} \
  --output_dir=${OUTPUT_DIR} \
  --backbone=${BACKBONE} \
  --dataset_file a2d \
  --batch_size 2 \
  --epochs 12 --lr_drop 3 5


