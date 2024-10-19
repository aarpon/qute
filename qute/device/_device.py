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


def cuda_does_gpu_support_16bit_mixed_precision() -> bool:
    """Check if the CUDA GPU supports 16-bit mixed precision.

    Returns
    -------
    result: bool
        True if the CUDA GPU supports 16-bit mixed precision, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    # Check for CUDA capability
    cuda_capability = torch.cuda.get_device_capability()

    # CUDA capability 7.0 (Volta) and above support 16-bit mixed precision
    return cuda_capability[0] >= 7


def cuda_get_gpu_memory_info() -> tuple[int, int]:
    """Return the maximum and currently available GPU memory for CUDA devices.

    Returns
    -------
    total_memory: int
        Total GPU memory in bytes (0 if CUDA is not available).

    free_memory: int
        Current free memory in bytes (0 if CUDA is not available).
    """
    if not torch.cuda.is_available():
        return 0, 0

    # Get free and total memory
    free_memory, total_memory = torch.cuda.mem_get_info()

    return total_memory, free_memory


def cuda_free_memory() -> None:
    """Free unused CUDA memory.

    If CUDA is available, attempt to release any unused CUDA memory back to the device.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
