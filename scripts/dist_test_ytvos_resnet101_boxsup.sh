#!/usr/bin/env bash
set -x

GPUS='1'
GPUS_PER_NODE=1
CPUS_PER_TASK=6
PORT=29500
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="resnet101"
# BACKBONE_PRETRAINED="./checkpoints/backbones/swin_base_patch244_window877_kinetics600_22k.pth"
OUTPUT_DIR="./results/SgMg_${BACKBONE}_eval_ytvos_boxsup"
CHECKPOINT="./results/SgMg_resnet101_scratch_ytvos_boxsup/checkpoint0009.pth"
python inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
  --eval \
  --ngpu=${GPUS_PER_NODE} \
  --output_dir=${OUTPUT_DIR} \
  --resume=${CHECKPOINT} \
  --backbone=${BACKBONE}


