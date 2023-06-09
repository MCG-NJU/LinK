# Installation

We follow the most of [installation](https://github.com/tianweiy/CenterPoint/blob/master/docs/INSTALL.md) of [CenterPoint](https://github.com/tianweiy/CenterPoint), with slight modifications to build the environment.

## Platform

- OS: Ubuntu 16.04/18.04
- Python: 3.6.5/3.7.10 
- PyTorch: 1.1/1.9/1.10.1
- CUDA: 10.2 (Other version like 11.3 should also be ok but not guaranteed).




## Basic installation

```bash
cd LinK/detection
conda create --name LinK_det python=3.6
conda activate LinK_det
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt

# add *LinK/detection* to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_LinK/detection"
echo $PYTHONPATH # make sure the configuration works.
source ~/.bashrc
conda activate LinK_det
```

## Advanced Installation 

### nuScenes dev-kit

```bash
pip install nuscenes-devkit
```

### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export CUDA_PATH=/usr/local/cuda-10.2
export CUDA_HOME=/usr/local/cuda-10.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rotated NMS 
cd ROOT_DIR/det3d/ops/iou3d_nms
python setup.py build_ext --inplace
```

### [spconv](https://github.com/traveller59/spconv)

```bash
pip install spconv-cu102
```

### modified TorchSparse

```bash
pip install ./torchsparse-u/
```