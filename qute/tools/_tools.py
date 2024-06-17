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


def calculate_receptive_field(
    num_levels: int, kernel_size: int = 3, pool_size: int = 2
):
    """Calculates the receptive field of the U-Net.

    Parameters
    ----------

    num_levels: int
        Number of levels in the contracting path (first half of the U) including the bottleneck layer.

    kernel_size: int = 3
        Kernel size.

    pool_size: int = 2
        Max pool size.

    Returns
    -------
    receptive_field: int
        Size of the receptive field of the U-Net.
    """
    # Initial receptive field
    receptive_field = 1

    # Increment per convolution layer
    increment = kernel_size - 1

    # Calculate the receptive field
    for level in range(num_levels):
        # Apply the effect of two convolutional layers
        receptive_field += 2 * increment
        # Apply the pooling layer, except for the last level (bottleneck)
        if level < num_levels - 1:
            receptive_field *= pool_size

    # Return the calculated receptive field
    return receptive_field


def num_levels_for_object_size(
    object_size: float, kernel_size: int = 3, pool_size: int = 2
):
    """Calculates the number of levels in the contracting path of the U-Net so that the
    receptive field is large enough to cover it.

    Parameters
    ----------

    object_size: float
        Size of the objects to be covered by the receptive field of the U-Net (e.g., median diameter of all labels).

    kernel_size: int = 3
        Kernel size.

    pool_size: int = 2
        Max pool size.

    Returns
    -------

    num_levels: int
        Number of levels in the contracting path.
    """
    # Determine the number of levels needed for a receptive field of at least 72
    num_levels = 1
    while True:
        receptive_field = calculate_receptive_field(num_levels)
        if receptive_field >= object_size:
            break
        num_levels += 1

    return num_levels
