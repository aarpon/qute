# qute

Framework to support deep-learning based computer-vision research in microscopy image analysis. Leverages and extends several [PyTorch](https://pytorch.org)-based framework and tools.

* [PyTorch](https://pytorch.org)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)
* [MONAI](https://monai.io)

## Installation

### Install prerequisites

* Install Miniconda from: https://docs.anaconda.com/miniconda/ (for Linux, macOS, and Windows)
* Install CUDA:
  * Linux: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64
  * Windows: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
  * macOS does not support CUDA; qute will use `mps` on M1 processors.

### Install qute

```bash
$ git clone https://github.com/aarpon/qute
$ cd qute
$ conda create -n qute-env python
$ conda activate qute-env
$ pip install -e .
```

On Windows, PyTorch with CUDA acceleration has to be explicitly installed:

```bash
$ python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## First steps

To get started, try:

```bash
$ python qute/examples/cell_segmentation_demo_unet.py 
```

For an example on how to use `ray[tune]` to optimize hyper-parameters, see `qute/examples/cell_segmentation_hp_optim_demo_unet.py`.

