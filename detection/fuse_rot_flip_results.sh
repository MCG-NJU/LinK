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
ngpus="4"
master_port="23456"
config="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3_flip_rot.py"
# work_dir="work_dirs/0075cbgs_cos3x7_group2_17+3"
work_dir=${ROT_TEST_WORK_DIR}
# -------------------------- #

angles=( "6.25" "-6.25" "12.5" "-12.5" "25" "-25" "0" )

echo "==> work_dir: ${work_dir}"
echo "==> angles: ${angles[@]}"

export PYTHONPATH="${work_dir}/backup" # use `det3d` from backup
export PYTHONPATH="${PYTHONPATH}:${PWD}/nuscenes-devkit/python-sdk"


# flip test with 7 angles
# for i in "${angles[@]}"
# do
#   echo "==> flip_rot testing with angle $i"
#   export TT_ROT_ANGLE=$i
#   python -m torch.distributed.launch --nproc_per_node=${ngpus} --master_port ${master_port} \
#     tools/dist_test.py ${config} \
#     --work_dir ${work_dir}/flip_rot${i} \
#     --checkpoint ${work_dir}/epoch_20.pth

# done


# fuse all flip_rot results
echo "==> fusing flip_rot results"
infos=()
for i in "${angles[@]}"
do
  infos+=( "${work_dir}/flip_rot$i/infos_test_10sweeps_withvelo.json" )
  # infos+=( "${work_dir}/flip_rot$i/infos_val_10sweeps_withvelo_filter_True.json" )
done

# convert array into a comma-separated string
printf -v paths '%s,' "${infos[@]}"
paths="${paths%,}"

# echo "paths: ${paths}"
python nms_better2.py --work_dir ${work_dir}/flip_rot_fuse --paths ${paths}


# # delete intermediate results
# echo "==> removing intermediate results"
# for i in "${angles[@]}"
# do
#   rm -r "${work_dir}/flip_rot$i"
# done
