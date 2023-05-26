#!/bin/bash

unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING
unset OMP_NUM_THREADS
unset NCCL_LL_THRESHOLD
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE

# export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# export OMP_NUM_THREADS=1
# export NCCL_LL_THRESHOLD=0
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1


# ------ config these ------ #
angle=$1
config="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3_flip_rot.py"
work_dir=${ROT_TEST_WORK_DIR}
# -------------------------- #

echo "==> work_dir: ${work_dir}"
echo "==> angle: ${angle}"


export PYTHONPATH="${work_dir}/backup" # use `det3d` from backup
export PYTHONPATH="${PYTHONPATH}:${PWD}/nuscenes-devkit/python-sdk"

# update `backup/det3d/datasets/pipelines/preprocess.py` and `backup/det3d/models/bbox_heads/center_head.py` 
# with soft link from main menu
rm ${work_dir}/backup/det3d/datasets/pipelines/preprocess.py
rm ${work_dir}/backup/det3d/models/bbox_heads/center_head.py
ln -sr det3d/datasets/pipelines/preprocess.py ${work_dir}/backup/det3d/datasets/pipelines/
ln -sr det3d/models/bbox_heads/center_head.py ${work_dir}/backup/det3d/models/bbox_heads/
echo "==> preprocess.py and centerhead.py updated"


# # auto-select gpu
# nvidia-smi -q -d Memory | grep -A4 GPU | grep Used > choosegpu
# gpuid=$(python -c "import numpy as np; print(np.argmin([int(x.split()[2]) for x in open('choosegpu','r').readlines()]))")
# echo "==> running on GPU: ${gpuid}"
# export CUDA_VISIBLE_DEVICES=${gpuid}
# rm choosegpu


# flip test with given angle
echo "==> flip_rot testing"
export TT_ROT_ANGLE=${angle}

python \
  tools/dist_test.py ${config} \
  --work_dir ${work_dir}/flip_rot${angle} \
  --checkpoint ${work_dir}/epoch_20.pth
