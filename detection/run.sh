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


ngpus="8"
vox_sz="0075"
config="configs/nusc/voxelnet/nusc_centerpoint_voxelnet_${vox_sz}voxel_fix_bn_z_elkv3.py"

work_dir="work_dirs/exp_name"


python \
  -m torch.distributed.launch --nproc_per_node=${ngpus} --master_port 23460 \
  tools/train.py ${config} \
  --work_dir ${work_dir} \
#   --resume_from ${work_dir}/epoch_3.pth

