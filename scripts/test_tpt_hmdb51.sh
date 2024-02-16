#!/bin/bash

data_root='/media/ssd8T'
testsets='hmdb51'
# arch=RN50
arch=ViT-B/16
bs=16
ctx_init=None
seed_list=(0 1024 1240 16 6111 59)

# /home/liang142/anaconda3/envs/tsm/bin/torchrun  --nproc_per_node=1 test.py --config ${config} --weights ${weight} ${@:3}

for seed in "${seed_list[@]}"; do
    /home/liang142/anaconda3/envs/tsm/bin/torchrun  --nproc_per_node=1 ./tpt_classification.py ${data_root} --test_sets ${testsets} \
    -a ${arch} -b ${bs} --gpu 0 \
    --tpt --ctx_init ${ctx_init} \
    --test_crops 3 --test_clips 4 --config '/media/ssd8T/TPT-video/configs/hmdb51/hmdb51_vitb-16-f16.yaml' --seed ${seed} \
    --logging /media/ssd8T/TPT-video/logging/hmdb51_vit-b-16-f16-seed${seed}.log
done
# '/media/ssd8T/TPT-video/configs/k400/k400_train_rgb_vitb-16-f8.yaml'