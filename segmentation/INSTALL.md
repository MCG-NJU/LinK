# Installation 

We provide installation instructions for the semantic segmentation task here. The project is built upon the codebase of [spvnas](https://github.com/mit-han-lab/spvnas). You can follow most of the instructions provided by spvnas to complete the initial setup. However, due to specific dependency issues we encountered on our own machines (Ubuntu 16.04), we have included more detailed steps for our environment. These additional instructions aim to facilitate a successful execution of our code on your machines.

## Setup


* Create a new conda environment:

```bash
conda create -n LinK_seg python=3.7 -y
conda activate LinK_seg
```

* Install [pytorch](https://pytorch.org/get-started/previous-versions/) following official instructions. We use a combination of CUDA 11.3+PyTorch 1.11. Other combinations with similar versions should also be OK. Feel free to try. 

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

* Install numda and cython. 

```bash
conda install numba cython
```

* Install ```torchpack```.
```bash
pip install torchpack
```

Install the development tool for nuScenes:

```bash
pip install nuscenes-devkit
```


* Install the modified torchsparse, named ```torchsparse-u```. 
```bash
cd segmentation
sudo apt-get install libsparsehash-dev
pip install ./torchsparse-u/
```


* Install ```mpi4py```.

```bash
pip install mpi4py
```

> In most cases, the setup of environment is completed here. If encountering problems about mpi4py, reinstall it from source code as follows. We use gcc/g++-7.5.0 to build the dependency.

1. Install OpenMPI. 

```bash
mkdir -p ~/Downloads && cd ~/Downloads
wget -c https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
tar -xvf openmpi-4.0.0.tar.gz
cd openmpi-4.0.0
mkdir build && cd build
```

```bash
../configure --prefix=/usr/local/openmpi-4.0.0 --disable-mpi-fortran --enable-mpi-cxx
sudo make -j8
sudo make install -j8
```
>> for not-root user, change the ```--prefix``` to a user local path and remove the ```sudo```.

Configure the environment variables.
```bash
vim ~/.bashrc
```

Add the following content at the end of ```~/.bashrc```. 
```bash
export OPENMPI=/usr/local/openmpi-4.0.0
export PATH=$OPENMPI/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPI/lib:$LD_LIBRARY_PATH
export INCLUDE=$OPENMPI/include:$INCLUDE
export CPATH=$OPENMPI/include:$CPATH
export MANPATH=$OPENMPI/share/man:$MANPATH
```
Refresh the configuration with 

```bash
source ~/.bashrc
conda activate LinK_seg # make sure in the correct virtual environment
```
Check if ```openmpi``` is installed correctly:
```bash
which mpirun
```
If correctly, the output should be the installation path for ```openmpi```, like ```/usr/local/openmpi-4.0.0/bin/mpirun```.


2. Install [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html).

```bash
cd ~/Downloads
wget -c https://github.com/mpi4py/mpi4py/releases/download/3.1.3/mpi4py-3.1.3.tar.gz
tar -xvf mpi4py-3.1.3.tar.gz
cd mpi4py-3.1.3
```

Setting the ```openmpi``` path:

```bash
vim mpi.cfg
```

Locate [openmpi] and modify the ```mpi_dir``` to your installed path for ```openmpi```.

```bash
L53: # Open MPI example
L54: # ----------------
L55: [openmpi]
L56: mpi_dir              = /usr/local/openmpi-4.0.0
L57: mpicc                = %(mpi_dir)s/bin/mpicc
L58: mpicxx               = %(mpi_dir)s/bin/mpicxx
L59: #include_dirs         = %(mpi_dir)s/include
L60: #libraries            = mpi
L61: library_dirs         = %(mpi_dir)s/lib
L62: runtime_library_dirs = %(library_dirs)s
```

Compile and install ```mpi4py```.

```bash
python setup.py build --mpi=openmpi
python setup.py install
```


# Dataset Preparation

* SemanticKITTI

Create the SemanticKITTI folder within the ```segmentation``` folder. 
```bash
mkdir -p data/SemanticKITTI/dataset/
cd data/SemanticKITTI/dataset
```

Download SemanticKITTI from the [official website](http://www.semantic-kitti.org/). Extrac all the files in the SemanticKITTI folder, the file structure should be like

```bash
segmentation/
    data/SemanticKITTI/dataset/sequences/
        00/
            calib.txt
            poses.txt
            times.txt
            velodyne/
            labels/
        01/
        02/
        ...
        21/
```

* nuScenes

Create the nuScenes folder within the ```segmentation``` folder. 

```bash
mkdir -p data/nuscenes
cd data/nuscenes
```

Download the dataset(v1.0) from the [official website](https://www.nuscenes.org/) and organize as follow: 

```bash
segmentation/
    data/nuscenes/
        v1.0-trainval
        v1.0-test
        samples
        sweeps
        maps
        lidarseg
```