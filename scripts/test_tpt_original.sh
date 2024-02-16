#!/bin/bash

data_root='/media/ssd8T'
testsets=$1
arch=RN50
# arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

# /home/liang142/anaconda3/envs/tsm/bin/torchrun  --nproc_per_node=1 test.py --config ${config} --weights ${weight} ${@:3}

python ./tpt_classification_original.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --ctx_init ${ctx_init}