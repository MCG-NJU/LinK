# Getting Started on nuScenes
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint)'s original document.

## Prepare data

### Download data and organise as follows

```
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata
```

Create a symlink to the dataset root 
```bash
mkdir data && cd data
ln -s DATA_ROOT 
mv DATA_ROOT nuScenes # rename to nuScenes
```
Remember to change the DATA_ROOT to the actual path in your system. 


### Create data

Data creation should be under the gpu environment.

```
# nuScenes
python tools/create_data.py nuscenes_data_prep --root_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --nsweeps=10
```

In the end, the data and info files should be organized as follows

```
# For nuScenes Dataset 
└── LinK/detection/
       └── data    
              └── nuScenes 
                     ├── samples       <-- key frames
                     ├── sweeps        <-- frames without annotation
                     ├── maps          <-- unused
                     |── v1.0-trainval <-- metadata and annotations
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
```

## Training

Run the following command to train LinK using 4 GPUs.

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 23456 ./tools/train.py \
  configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3.py \
  --work_dir work_dirs/0075cbgs_cos3x7_group2
```

The model checkpoints and logs will be saved to the `work_dir` above.

### Fading strategy of GT sampling

We use the fading strategy of GT sampling, i.e., 15 epochs with GT sampling and then 5 epochs without this. To do so, after completing training 15 epochs, we break the training process and resume it by the following steps.

1. Turn off the GT sampling in config.
   See `configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3.py`, line 92.

2. Resume the training process by

   ```bash
   python -m torch.distributed.launch --nproc_per_node=4 --master_port 23456 ./tools/train.py \
     configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3.py \
     --work_dir work_dirs/0075cbgs_cos3x7_group2_15+5 \
     --resume_from work_dirs/0075cbgs_cos3x7_group2/epoch_15.pth
   ```

## Evaluation

### Evaluate on validation set

Run the following command to evaluate LinK on validation set, using 4 GPUs.

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 23456 tools/dist_test.py \
  configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3.py \
  --work_dir work_dirs/0075cbgs_cos3x7_group2_15+5/atEp20 \
  --checkpoint work_dirs/0075cbgs_cos3x7_group2_15+5/epoch_20.pth
```

The evaluation result and metrics will be saved to `work_dirs/0075cbgs_cos3x7_group2_15+5/atEp20/`. Change the arguments accordingly for your own model.

### Evaluate on test set

Run the following command to evaluate LinK on the test set.

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port 23456 tools/dist_test.py \
  configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_elkv3.py \
  --work_dir work_dirs/0075cbgs_cos3x7g2_trainval2_15+5/atEp20 \
  --checkpoint work_dirs/0075cbgs_cos3x7g2_trainval2_15+5/epoch_20.pth \
  --testset
```

This model is trained using the training set and validation set of nuScenes. Change the arguments accordingly for your own model.

The detection result will be saved to `work_dirs/0075cbgs_cos3x7g2_trainval2_15+5/atEp20/`. To get the metrics of your own model on test set, submit it to the [official nuScenes detection server](https://eval.ai/web/challenges/challenge-page/356/overview).

### Evaluate with flipping and rotation TTA

For the evaluation with flipping and rotation TTA, we provide a bash script `fuse_rot_flip_results.sh`. This script will evaluate the model with 7 angles (0, ±6.25, ±12.5, ±25), each with 4 kinds of flipping (no flip, x-flip, y-flip, xy-flip). The final detection result is fused by `nms_better2.py` (ref: https://github.com/dvlab-research/FocalsConv/blob/master/CenterPoint/test_aug_examples/nms_better2.py).

Config the bash shell accordingly for your use case. The final detection result will be saved to `{work_dir}/flip_rot_fuse/`.

The pretrained models and configurations are in [MODEL ZOO](../../README.md).