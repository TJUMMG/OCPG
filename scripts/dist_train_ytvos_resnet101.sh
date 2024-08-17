#!/usr/bin/env bash
set -x

GPUS='0'
PORT=25500
GPUS_PER_NODE=1
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="resnet101"
# BACKBONE_PRETRAINED="./checkpoints/backbones/swin_tiny_patch244_window877_kinetics400_1k.pth"
OUTPUT_DIR="./results/SgMg_${BACKBONE}_scratch_ytvos"
EXP_NAME="SgMg_${BACKBONE}_scratch"
CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --with_box_refine --binary --freeze_text_encoder \
  --output_dir=${OUTPUT_DIR} \
  --exp_name=${EXP_NAME} \
  --backbone=${BACKBONE} \
  --dataset_file ytvos \
  --amp