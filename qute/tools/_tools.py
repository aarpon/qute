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

import math


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
        receptive_field = calculate_receptive_field(num_levels, kernel_size, pool_size)
        if receptive_field >= object_size:
            break
        num_levels += 1

    return num_levels


def compute_dynunet_params(
    input_dims: tuple,
    desired_levels: int,
    base_filters: int = 32,
    pooling_stride: int = 2,
    kernel_size_val: int = 3,
    deep_supr_num: int = None,
):
    """Compute default parameters for a DynUNet given input dimensions and integrates
    a deep supervision parameter ensuring it doesn't exceed the allowable number.

    Parameters
    ----------

    input_dims: tuple
        Spatial dimensions (e.g., (H, W) for 2D or (H, W, D) for 3D).

    desired_levels: int
        Desired number of levels (depth of the network).

    base_filters: int
        Number of filters for the first level.

    pooling_stride: int
        Pooling factor (stride) for down-sampling.

    kernel_size_val: int
        Kernel size value to use in each dimension.

    deep_supr_num: int
        Desired number of deep supervision outputs. If None or too high, it is set to (num_levels - 2).

    Returns
    -------

    kernel_sizes: list of tuples
        Kernel sizes per level.

    strides: list of tuples
        Stride values per level.

    upsample_kernel_sizes: list of tuples
        Upsampling kernel sizes.

    filters: list
        Number of convolution filters per level.

    deep_supr_num: int
        The adjusted number of deep supervision outputs.
    """
    num_spatial_dims = len(input_dims)

    # Determine the maximum allowed levels based on the smallest dimension.
    max_possible_levels = int(math.floor(math.log(min(input_dims), pooling_stride))) + 1
    num_levels = min(desired_levels, max_possible_levels)

    # Create kernel sizes: same kernel size in each spatial dimension at every level.
    kernel_sizes = [(kernel_size_val,) * num_spatial_dims for _ in range(num_levels)]

    # Set strides: first level uses no downsampling, all subsequent levels use the pooling_stride.
    strides = [(1,) * num_spatial_dims] + [
        (pooling_stride,) * num_spatial_dims for _ in range(1, num_levels)
    ]

    # Define filter configuration: double the base filters at each level.
    filters = [base_filters * (2**level) for level in range(num_levels)]

    # Define upsample kernel sizes for levels where upsampling is performed.
    # There are exactly num_levels - 1 upsample layers.
    upsample_kernel_sizes = [
        (pooling_stride,) * num_spatial_dims for _ in range(num_levels - 1)
    ]

    # In MONAI's DynUNet, deep_supr_num must be strictly less than the number of upsample layers.
    max_deep_supr_allowed = (num_levels - 1) - 1
    if deep_supr_num is None or deep_supr_num >= (num_levels - 1):
        old_value = deep_supr_num
        deep_supr_num = max_deep_supr_allowed
        if old_value is None:
            print(
                f"deep_supr_num was not provided; setting to maximum allowed value: {deep_supr_num}"
            )
        else:
            print(
                f"Requested deep_supr_num {old_value} exceeds maximum allowed (number of upsample layers - 1); adjusting to {deep_supr_num}"
            )

    return kernel_sizes, strides, upsample_kernel_sizes, filters, deep_supr_num


if __name__ == "__main__":
    # Example compute_dynunet_params() usage:

    # For a 2D input.
    input_dimensions = (512, 512)

    # Desired number of levels
    desired_levels = 5

    # User requested deep supervision outputs
    requested_deep_supr_num = 3

    params = compute_dynunet_params(
        input_dimensions, desired_levels, deep_supr_num=requested_deep_supr_num
    )
    k_sizes, s_values, up_k_sizes, filter_list, final_deep_supr_num = params

    print("Kernel sizes per level:", k_sizes)
    print("Strides per level:", s_values)
    print("Upsample kernel sizes:", up_k_sizes)
    print("Filter configuration:", filter_list)
    print("Deep supervision outputs:", final_deep_supr_num)
