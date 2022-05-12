# qute

Leverages and extends several [PyTorch](https://pytorch.org)-based framework and tools.

* [PyTorch](https://pytorch.org)
* [PyTorch Lightning](https://www.pytorchlightning.ai/)
  * [Lightning Bolts](https://github.com/PyTorchLightning/lightning-bolts)
  * [Lightning Flash](https://github.com/PyTorchLightning/lightning-flash)
* [MONAI](https://monai.io)

## Installation

```bash
$ git clone https://github.com/aarpon/qute
$ cd qute
$ conda env create -f ./environment.yml
$ conda activate qute-env
$ pip install -e .
```

## First steps

To get started, try:

```bash
$ python qute/examples/cell_segmentation_demo_unet.py 
```
