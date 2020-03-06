#!/bin/bash

data_dir="./data/imagenet100"
output_dir="./output/imagenet100/baseLR0.4_alpha0.99_crop0.08_kall_t0.1_AMPO1"
python -m torch.distributed.launch --master_port 12347 --nproc_per_node=8 \
    train.py \
    --data-dir ${data_dir} \
    --dataset imagenet100 \
    --base-learning-rate 0.4 \
    --alpha 0.99 \
    --crop 0.08 \
    --nce-k 126689 \
    --nce-t 0.1 \
    --amp-opt-level O1 \
    --output-dir ${output_dir}

python -m torch.distributed.launch --master_port 12348 --nproc_per_node=4 \
    eval.py \
    --dataset imagenet100 \
    --data-dir ${data_dir} \
    --pretrained-model ${output_dir}/current.pth \
    --output-dir ${output_dir}/eval

