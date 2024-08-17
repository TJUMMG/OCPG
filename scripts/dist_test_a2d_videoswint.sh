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

BACKBONE="video_swin_t_p4w7"
BACKBONE_PRETRAINED="./checkpoints/backbones/swin_base_patch244_window877_kinetics600_22k.pth"
OUTPUT_DIR="./results/results/SgMg_${BACKBONE}_finetune_a2d"
EXP_NAME="SgMg_${BACKBONE}_finetune_a2d"
PRETRAINED_WEIGHTS="checkpoints/sgmg_videoswint_a2d.pth"
TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --with_box_refine --binary --freeze_text_encoder \
  --exp_name=${EXP_NAME} \
  --output_dir=${OUTPUT_DIR} \
  --backbone=${BACKBONE} \
  --dataset_file a2d \
  --batch_size 2 \
  --epochs 6 --lr_drop 3 5 \
  --pretrained_weights=${PRETRAINED_WEIGHTS} \
  --eval \


