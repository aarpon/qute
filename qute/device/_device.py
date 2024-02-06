import torch
from pytorch_lightning.accelerators import Accelerator


def get_device() -> torch.device:
    """Return available PyTorch device depending on platform.

    Returns
    -------

    device: torch.device
        Device that can be used for training.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_accelerator() -> str:
    """Return available PyTorch Lightning accelerator depending on platform.

    Returns
    -------

    accelerator: str
        Accelerator that can be used for training.
    """

    if torch.cuda.is_available():
        return "gpu"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"
