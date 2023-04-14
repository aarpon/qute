#  ********************************************************************************
#   Copyright © 2022 - 2003, ETH Zurich, D-BSSE, Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the Apache License Version 2.0
#   which accompanies this distribution, and is available at
#   https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Contributors:
#       Aaron Ponti - initial API and implementation
#  ******************************************************************************/

import time
from typing import Optional

import numpy as np


def sample(
    image: np.ndarray,
    patch_size: tuple,
    y0: Optional[int] = None,
    x0: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, int, int]:
    """Returns a (random) subset of given shape from the passed 2D image.

    Parameters
    ----------

    image: numpy array
        Original intensity image.

    patch_size: tuple
        Size (y, x) of the subset of the image to be randomly extracted.

    y0: Optional[int]
        y component of the top left corner of the extracted region.
        If omitted (default), it will be randomly generated.

    x0: Optional[int]
        x component of the top left corner of the extracted region.
        If omitted (default), it will be randomly generated.

    seed: Optional[int]
        Random generator seed to reproduce the sampling. Omit to create a
        new random sample every time.

    Returns
    -------

    result: tuple[np.ndarray, int, int]
        Subset of the image of given size; y coordinate of the top-left corner of
        the extracted subset; x coordinate of the top-left corner of the extracted subset.
    """

    if image.ndim != 2:
        raise ValueError("The image must be 2D.")

    # Initialize random-number generator
    if seed is None:
        seed = time.time_ns()
    rng = np.random.default_rng(seed)

    # Get starting point
    max_y = image.shape[0] - patch_size[0]
    max_x = image.shape[1] - patch_size[1]
    if y0 is None:
        y0 = int(rng.uniform(0, max_y))
    if x0 is None:
        x0 = int(rng.uniform(0, max_x))

    # Return the subset and the starting coordinates
    return image[y0 : y0 + patch_size[0], x0 : x0 + patch_size[1]], y0, x0
