# qute

Framework to support deep-learning based computer-vision research in microscopy image analysis. Leverages and extends several [PyTorch](https://pytorch.org)-based framework and tools.

* [PyTorch](https://pytorch.org)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)
* [MONAI](https://monai.io)

## Installation

```bash
$ git clone https://github.com/aarpon/qute
$ cd qute
$ conda create -n qute-env python
$ conda activate qute-env
$ pip install -e .
```

## First steps

To get started, try:

```bash
$ python qute/examples/cell_segmentation_demo_unet.py 
```

For an example on how to use `ray[tune]` to optimize hyper-parameters, see `qute/examples/cell_segmentation_hp_optim_demo_unet.py`.

