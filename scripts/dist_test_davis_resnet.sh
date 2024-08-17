#!/usr/bin/env bash
set -x

GPUS='0'
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
OUTPUT_DIR="./results/SgMg_${BACKBONE}_eval"
CHECKPOINT="./results/SgMg_resnet101_scratch_ytvos_boxsup/checkpoint0009.pth"
python inference_davis.py --with_box_refine --binary --freeze_text_encoder \
  --eval \
  --ngpu=${GPUS_PER_NODE} \
  --output_dir=${OUTPUT_DIR} \
  --resume=${CHECKPOINT} \
  --backbone=${BACKBONE} \
  --backbone_pretrained=${BACKBONE_PRETRAINED}


# evaluation
ANNO0_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_0"
ANNO1_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_1"
ANNO2_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_2"
ANNO3_DIR=${OUTPUT_DIR}/"DVS_Annotations"/"anno_3"
echo "Annotations store at : ${ANNO0_DIR}"
rm ${ANNO0_DIR}"/global_results-val.csv"
rm ${ANNO0_DIR}"/per-sequence_results-val.csv"
rm ${ANNO1_DIR}"/global_results-val.csv"
rm ${ANNO1_DIR}"/per-sequence_results-val.csv"
rm ${ANNO2_DIR}"/global_results-val.csv"
rm ${ANNO2_DIR}"/per-sequence_results-val.csv"
rm ${ANNO3_DIR}"/global_results-val.csv"
rm ${ANNO3_DIR}"/per-sequence_results-val.csv"

python3 eval_davis.py --results_path=${ANNO0_DIR}
python3 eval_davis.py --results_path=${ANNO1_DIR}
python3 eval_davis.py --results_path=${ANNO2_DIR}
python3 eval_davis.py --results_path=${ANNO3_DIR}

echo "Working path is: ${OUTPUT_DIR}"



