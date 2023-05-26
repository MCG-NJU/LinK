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

export PYTHONPATH="${PWD}:${PWD}/nuscenes-devkit/python-sdk"

ngpus="4"
config="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3.py"
ep="20"

# python tools/dist_test.py ${config} \
python -m torch.distributed.launch --nproc_per_node=${ngpus} --master_port 23456 \
  tools/dist_test.py ${config} \
  --work_dir ../atEp${ep} \
  --checkpoint ../epoch_${ep}.pth

