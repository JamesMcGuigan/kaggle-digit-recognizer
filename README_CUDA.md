# Dependencies Install
```
# Python3 + Pip + Venv
sudo apt-get install python3 python3-pip python3-venv
./requirements.sh

# Build Tools
sudo apt-get install build-essential libssl-dev libffi-dev python-dev

# Node
sudo apt install nodejs 
npm imstall -g yarn
```

# ANACONDA Install
https://askubuntu.com/a/841224
```
CONTREPO=https://repo.continuum.io/archive/
# Stepwise filtering of the html at $CONTREPO
# Get the topmost line that matches our requirements, extract the file name.
ANACONDAURL=$(wget -q -O - $CONTREPO index.html | grep "Anaconda3-" | grep "Linux" | grep "86_64" | head -n 1 | cut -d \" -f 2)
wget -O ~/Downloads/anaconda.sh $CONTREPO$ANACONDAURL
bash ~/Downloads/anaconda.sh
echo export PATH="$PATH:./.anaconda3/bin" > ~/.bashrc
```

---

# Win10 WSL Python

NOTE: CUDA does not work on Win10 WSL - so use Windows Binaries with Alias
- http://www.erogol.com/using-windows-wsl-for-deep-learning-development/

~/.bashrc
```
alias conda="conda.exe"
alias python="python.exe"
alias python3="python.exe"
alias ipython="ipython.exe"
alias nosetests="nosetests.exe"
alias pip="pip.exe"
alias nvidia-smi="/mnt/c/Program\ Files/NVIDIA\ Corporation/NVSMI/nvidia-smi.exe"
```
---
# Ubuntu CUDA 

Use explicitly supported LTS version: Ubuntu 18.04 or Fedora 29 
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### Install NVIDIA Drivers 
```
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
reboot
nvidia-smi  # validate GPU visible
```
Or GUI: Software & Updates -> Additional Drivers -> nvidia-driver-440

Be sure to enter UEFI password in Grub on reboot

### Install CUDA 
https://www.tensorflow.org/install/gpu#install_cuda_with_apt

NOTE: Tensorflow 2 does not support CUDA 9

CUDA 10.1 (installed as per tensorflow docs) throws `can't find libcublas.so.10.0` errors.
The libs exist in `/usr/local/cuda-10.1/targets/x86_64-linux/lib/` but are misnamed. 

Workaround is to modify instructions to downgrade to CUDA 10.0 or upgrade to CUDA 10.2
```
# Uninstall packages from tensorflow installation instructions 
sudo apt-get remove cuda-10-1 \
    libcudnn7 \
    libcudnn7-dev \
    libnvinfer6 \
    libnvinfer-dev \
    libnvinfer-plugin6

# WORKS: Downgrade to CUDA-10.0
sudo apt-get install -y --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.4.38-1+cuda10.0  \
    libcudnn7-dev=7.6.4.38-1+cuda10.0;
sudo apt-get install -y --no-install-recommends \
    libnvinfer6=6.0.1-1+cuda10.0 \
    libnvinfer-dev=6.0.1-1+cuda10.0 \
    libnvinfer-plugin6=6.0.1-1+cuda10.0;
```

Upgrading to CUDA 10.2 seems to suffer from the same problem
```
# BROKEN: Upgrade to CUDA-10.2 - has the same problem
# use `apt show -a libcudnn7 libnvinfer7` to find 10.2 compatable version numbers
sudo apt-get install -y --no-install-recommends \
    cuda-10-2 \
    libcudnn7=7.6.5.32-1+cuda10.2  \
    libcudnn7-dev=7.6.5.32-1+cuda10.2;
sudo apt-get install -y --no-install-recommends \
    libnvinfer7=7.0.0-1+cuda10.2 \
    libnvinfer-dev=7.0.0-1+cuda10.2 \
    libnvinfer-plugin7=7.0.0-1+cuda10.2;
```

### Test GPU Visibility in Python
```
python3
>>> import tensorflow as tf
>>> tf.test.is_gpu_available()
``` 

### FutureWarnings
- https://github.com/tensorflow/tensorflow/issues/30427

Two solutions:
- `pip3 install tf-nightly-gpu`
- `pip3 install "numpy<1.17"`
