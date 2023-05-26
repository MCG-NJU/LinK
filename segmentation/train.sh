# for amd cpu, uncomment the below two lines
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./start_multigpu.sh -t release --gpus 8
