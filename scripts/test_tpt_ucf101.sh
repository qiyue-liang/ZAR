#!/bin/bash

data_root='/media/ssd8T'
# arch=RN50
arch=ViT-B/16
bs=16
ctx_init=None
seed_list=(0)
selection_p_list=(1)
# seed_list=(0)
test_crop=3
# seed_list=(1024)
# /home/liang142/anaconda3/envs/tsm/bin/torchrun  --nproc_per_node=1 test.py --config ${config} --weights ${weight} ${@:3}

for seed in "${seed_list[@]}"; do
    for selection_p in "${selection_p_list[@]}"; do
        /home/liang142/anaconda3/envs/tsm/bin/torchrun  --nproc_per_node=1 ./tpt_classification.py ${data_root} \
        -a ${arch} -b ${bs} --gpu 0 \
        --tpt --ctx_init ${ctx_init} \
        --test_crops ${test_crop} --test_clips 4 --config '/media/ssd8T/TPT-video/configs/ucf101/ucf101_vitb-16-f16.yaml' --seed ${seed} \
        --logging /media/ssd8T/TPT-video/logging/ucf101_vit-b-16-f16-seed${seed}-selection_p${selection_p}-test_crop${test_crop}.log
    done
done
# '/media/ssd8T/TPT-video/configs/k400/k400_train_rgb_vitb-16-f8.yaml'