#!/usr/bin/env bash
set -x

GPUS='1'
PORT=25505
GPUS_PER_NODE=1
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="resnet101"
OUTPUT_DIR="./results/SgMg_${BACKBONE}_scratch_a2d_pointsup"
EXP_NAME="SgMg_${BACKBONE}_scratch_a2d"
RESUME="results/SgMg_resnet101_scratch_a2d_pointsup_partialsup/checkpoint0001.pth"
CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --with_box_refine --binary --freeze_text_encoder --supervision point \
  --exp_name=${EXP_NAME} \
  --output_dir=${OUTPUT_DIR} \
  --backbone=${BACKBONE} \
  --dataset_file a2d \
  --batch_size 4 \
  --epochs 12 --lr_drop 3 5 \
  --eval \
  --resume=${RESUME}


