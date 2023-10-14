# TorchAxf
We proposed "TorchAxf: Enabling Rapid Simulation of Approximate DNN Models using GPU-based Floating-Point Computing Framework" in MASCOTS 2023.
![image](https://github.com/rhkr9609/TorchAxf/assets/45326283/71269312-d161-4ce2-9dd1-0a397820142d)


we reference adapt(https://github.com/dimdano/adapt)

# pre-setup
This manual was written based on Ubuntu 20.04 with docker. And you need an NVIDIA GPU (over 8GB GPU memory). \
Before configuring your environment, check if your GPU driver is the latest version suitable for PyTorch, and update it if necessary. 


Ubuntu 20.04 with the following commands run:
```
$ apt-get update
$ apt-get upgrade -y
$ apt-get install gcc g++ python3 python3-pip libxml2-dev wget vim git unzip -y
```

Install the appropriate CUDA version for PyTorch at the following https://developer.nvidia.com/cuda-toolkit-archive 

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

Install pyncrtc cupy
```
$ pip3 install pynvrtc
```
https://docs.cupy.dev/en/stable/install.html

# pre-trained weight
TorchAxf used pre-trained weight by Huy Phan (https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master)
```
wget https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
unzip gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
rm gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip
```

# Run guideline
