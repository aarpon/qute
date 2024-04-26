# ******************************************************************************
# Copyright © 2022 - 2024, ETH Zurich, D-BSSE, Aaron Ponti
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Apache License Version 2.0
# which accompanies this distribution, and is available at
# https://www.apache.org/licenses/LICENSE-2.0.txt
#
# Contributors:
#   Aaron Ponti - initial API and implementation
# ******************************************************************************

import torch


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
