import torch


def get_device() -> torch.device:
    """Return available device depending on platform.

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
