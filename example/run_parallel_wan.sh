#!/bin/bash
export OMP_NUM_THREADS=1

export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


MODEL_ID="wan2.2-14b"
GPUS=8

#PARALLEL_ARGS="--use_cfg_parallel --ulysses_size $((GPUS / 2)) "
PARALLEL_ARGS="--ulysses_size ${GPUS} "

torchrun --nproc_per_node=$GPUS parallel_sageattn_wan_i2v.py \
    --model $MODEL_ID \
    $PARALLEL_ARGS \
    --warmup_round 1 \
    --eval_round 1 \
    --height 1280 \
    --width 720 \
    --num_inference_steps 4 \
    --num_frames 81 \
    --guidance_scale 1.0 \
    --profile \
    --seed 0
