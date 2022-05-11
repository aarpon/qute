#  ********************************************************************************
#   Copyright © 2022-, ETH Zurich, D-BSSE, Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the Apache License Version 2.0
#   which accompanies this distribution, and is available at
#   https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Contributors:
#       Aaron Ponti - initial API and implementation
#  ******************************************************************************/

import torch
from monai.transforms import Transform


class MinMaxNormalize(Transform):
    """Normalize a tensor to [0, 1] using given min and max absolute intensities.
    Args:
        min_intensity: int
            Minimum intensity to normalize against (optional, default = 0).
        max_intensity: int
            Maximum intensity to normalize against (optional, default = 65535).
    """

    def __init__(self, min_intensity: int = 0, max_intensity: int = 65535) -> None:
        """Constructor"""
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, image: torch.tensor) -> torch.tensor:
        """
        Apply the transform to `image`.
        @return a stack of images with the same width and height as `label` and
            with `num_classes` planes.
        """
        return (image - self.min_intensity) / (self.max_intensity - self.min_intensity)
