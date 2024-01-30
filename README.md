# TorchAxf
We proposed "TorchAxf: Enabling Rapid Simulation of Approximate DNN Models using GPU-based Floating-Point Computing Framework" in MASCOTS 2023. (https://ieeexplore.ieee.org/abstract/document/10387653)
![image](https://github.com/rhkr9609/TorchAxf/assets/45326283/71269312-d161-4ce2-9dd1-0a397820142d)

TorchAxf: a **floating-point** computing framework developed to accelerate the emulation of **approximate** Deep Neural Network (DNN) models, including Spiking Neural Networks (SNNs). Leveraging PyTorch and CUDA for GPU acceleration, TorchAxf provides a comprehensive environment for integrating various types of approximate adders and multipliers, **supporting both standard and custom floating-point formats.**

TorchAxf is capable of computing PyTorch's conv1d and conv2d functions using approximate floating-point arithmetic.

Provide CNN models: <br/>
Pytorch: ResNet, VGGNet, DenseNet, GoogleNet, Inception-v3, MobileNetv2 <br/>
 
Models are not provided, however, they can be easily applied and thoroughly tested within the TorchAxf framework.: <br/>
SpykeTorch(SDNN, DCSNN, R-STDP): https://github.com/miladmozafari/SpykeTorch <br/>
SENet: https://github.com/moskomule/senet.pytorch <br/>
AdderNet: https://github.com/huawei-noah/AdderNet

# pre-setup
This manual was written based on Ubuntu 20.04 with Docker, Pytorch and CUDA 11.8. And you need an NVIDIA GPU (over 8GB GPU memory). <br/>
Before configuring your environment, check if your GPU driver is the latest version suitable for PyTorch, and update it if necessary. 

Pytorch docker image download and run
```
$ docker pull pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
$ docker run -it --gpus all --name TorchAxf -w /home -dt pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel /bin/bash
$ docker exec -it TorchAxf /bin/bash
```
require package install in docker container
```
$ apt-get update
$ apt-get install wget vim git unzip -y
```

# Setup Framework
```
$ git clone https://github.com/rhkr9609/TorchAxf.git
$ cd TorchAxf
```


pre-trained weight
TorchAxf used pre-trained weight by Huy Phan (https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master)
```
wget https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
unzip gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
rm gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
```
state_ditcts move in models folder
# Run guideline


<!--
Ubuntu 20.04 pre-setup commands:
```
$ apt-get update
$ apt-get upgrade -y
$ apt-get install gcc g++ python3 python3-pip libxml2-dev wget vim git unzip -y
```

Install the appropriate CUDA 11.8 version for PyTorch at the following: <br/>
CUDA 11.8 download link: [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)

Set path in ~/.bashrc:
```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
Source and Test:
```
$ source ~/.bashrc
$ nvcc --version
```


If you successfully install CUDA, you can see nvcc version.

Install Pytorch: https://pytorch.org/get-started/locally/

Install pyncrtc and cupy
```
$ pip3 install pynvrtc
```
https://docs.cupy.dev/en/stable/install.html
-->