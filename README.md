# TorchAxf
Approximate DNN Framework

# pre-setup
ubuntu 20.04 at least \
CUDA \
PyTorch \
CuPy \
pynvrtc \
This manual was written based on Ubuntu 20.04 with docker. And you need an NVIDIA GPU (over 8GB GPU memory). \
Before configuring your environment, check if your GPU driver is the latest version suitable for PyTorch, and update it if necessary. \


Ubuntu 20.04 with the following commands run:
```
$ apt-get update
$ apt-get upgrade -y
$ apt-get install gcc g++ python3 python3-pip wget libxml2-dev vim git -y
```
