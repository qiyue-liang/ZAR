#!/bin/bash

data_root='/media/ssd8T'
# arch=RN50
arch=ViT-B/16
bs=16
ctx_init=None
# seed_list=(0 1024 1240 16 6111 59)
seed_list=(1024)
test_crop=3
# /home/liang142/anaconda3/envs/tsm/bin/torchrun  --nproc_per_node=1 test.py --config ${config} --weights ${weight} ${@:3}
selection_p=0.8
for seed in "${seed_list[@]}"; do
    /home/liang142/anaconda3/envs/tsm/bin/torchrun  --nproc_per_node=1 ./tpt_classification.py ${data_root} \
    -a ${arch} -b ${bs} --gpu 0 \
    --tpt --ctx_init ${ctx_init} --selection_p ${selection_p}\
    --test_crops ${test_crop} --test_clips 4 --config '/media/ssd8T/TPT-video/configs/kinetics600/kinetics600_vitb-16-f16.yaml' --seed ${seed} \
    --logging /media/ssd8T/TPT-video/logging/kinetics600/kinetics600_vit-b-16-f16-seed${seed}-selection_p${selection_p}-crop${test_crop}.log
done
# '/media/ssd8T/TPT-video/configs/k400/k400_train_rgb_vitb-16-f8.yaml'