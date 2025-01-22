#  ********************************************************************************
#  Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************

from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from monai.utils import set_determinism


def set_global_rng_seed(seed: Union[int, None], workers: bool = False):
    """Set a global random number generator seed for NumPy, PyTorch, PyTorch Lightning and MONAI.

    seed: int|None
        Seed for random number generators. Set to None to disable determinism.

    workers: bool
        Whether to set the seed also for the worker processes (of PyTorch Lightning).
    """

    # Set the global random number generator seed for reproducibility
    set_determinism(seed=seed)

    # Explicitly set the seed for NumPy (though it's already handled by monai.utils.set_determinism)
    np.random.seed(seed)

    # Explicitly set the seed for PyTorch (though it's already handled by monai.utils.set_determinism)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Explicitly set the seed for PyTorch Lightning
    pl.seed_everything(seed, workers=workers)
