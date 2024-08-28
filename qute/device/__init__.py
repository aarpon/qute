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

from ._device import (
    cuda_does_gpu_support_16bit_mixed_precision,
    cuda_free_memory,
    cuda_get_gpu_memory_info,
    get_accelerator,
    get_device,
)

__doc__ = "Simplify multi-platform device support."
__all__ = [
    "cuda_does_gpu_support_16bit_mixed_precision",
    "cuda_free_memory",
    "cuda_get_gpu_memory_info",
    "get_accelerator",
    "get_device",
]
