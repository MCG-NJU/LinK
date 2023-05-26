#!/bin/bash

dataset='semantic_kitti' # ['semantic_kitti', 'nuscenes']
model='linkunet' # ['linkunet', 'linkencoder', 'minkunet', 'spvcnn']

target=''
# datetime=$(printf '%(%m%d-%H%M%S)T\n' -1)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) model="$2"; shift ;;
        -t|--target) target="$2"; shift ;;
        --gpus) gpus="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$target" ]; then
    rundir="runs/${dataset}/${model}/default"
else
    rundir="runs/${dataset}/${model}/${target}"
fi

torchpack dist-run -np ${gpus} python train.py configs/${dataset}/${model}/default.yaml --run-dir ${rundir}
