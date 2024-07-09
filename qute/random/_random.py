#  ********************************************************************************
#  Copyright © 2022 - 2024, ETH Zurich, D-BSSE, Aaron Ponti
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


def set_global_rng_seed(seed: Union[int, None]):
    """Set a global random number generator seed for NumPy, PyTorch, PyTorch Lightning and MONAI.

    seed: int|None
        Seed for random number generators. Set to None to disable determinism.
    """

    # Set the global random number generator seed for reproducibility
    set_determinism(seed=seed)

    # Explicitly set the seed for NumPy (though it's already handled by monai.utils.set_determinism)
    np.random.seed(seed)

    # Explicitly set the seed for PyTorch (though it's already handled by monai.utils.set_determinism)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Explicitly set the seed for PyTorch Lightning
    pl.seed_everything(seed)
