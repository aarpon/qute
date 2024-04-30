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

from typing import Union

import monai.data
import numpy as np
import torch


def scale_dist_transform_by_region(
    dt: np.ndarray,
    regions: np.ndarray,
    reverse: bool = False,
    do_not_zero: bool = False,
    in_place: bool = False,
):
    """Scales the distance transform values by region.

    Parameters
    ----------
    dt: np.ndarray
        The distance transform array.

    regions: np.ndarray
        The region array.

    in_place: bool (Optional)
        Specifies whether the scaling should happen in place. Default is False.

    reverse: bool
        Whether to reverse the direction of the normalized distance transform: from 1.0 at the center of the
        objects and 0.0 at the periphery, to (close to) 0.0 at the center and 1.0 at the periphery.

    do_not_zero: bool
        This is only considered if `reverse` is True. Set to True not to allow that the center pixels in each
        region have an inverse distance transform of 0.0.

    Returns
    -------
    The updated distance transform array.
    """

    # Get all regions
    ids = np.unique(regions)

    # Should the process happen in place?
    if in_place:
        work_dt = dt
    else:
        work_dt = dt.copy()

    # Process all regions
    for i in ids:

        # Skip background
        if i == 0:
            continue

        # Extract the distance transform for current region and scale it
        indices = regions == i
        values = work_dt[indices]
        if reverse:
            # Reverse the direction of the distance transform: make sure to stretch
            # the maximum to 1.0; we can keep a minimum larger than 0.0 in the center.
            if do_not_zero:
                # Do not set the distance at the center to 0.0; the gradient is
                # slightly lower, depending on the original range.
                tmp = work_dt[indices]
                tmp = (tmp.max() + 1) - tmp
                work_dt[indices] = tmp / tmp.max()
            else:
                # Plain linear inverse
                min_value = work_dt[indices].min()
                max_value = work_dt[indices].max()
                work_dt[indices] = (work_dt[indices] - max_value) / (
                    min_value - max_value
                )
        else:
            work_dt[indices] = work_dt[indices] / values.max()

    # Return updated dt
    return work_dt


def get_tensor_num_spatial_dims(
    data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray],
    with_batch_dim: bool = False,
) -> int:
    """Returns the spatial batch size of the tensor (either 2D or 3D).

    Accepted input geometries are:

    3D with with_batch_dim == True    |    3D with with_batch_dim == False
          [B, C, D, H, W]                          [C, D, H, W]

    2D with with_batch_dim == True    |    2D with with_batch_dim == False
           [B, C, H, W]                             [C, H, W]

    Parameters
    ----------

    data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
        Input tensor.

    with_batch_dim: bool = False
        Whether the first dimension of the tensor should be considered as the batch dimension.

    Returns
    -------

    effective_size: int
        Either 2 for two-dimensional data or 3 for three-dimensional data.
    """

    if with_batch_dim and len(data.shape) not in [4, 5]:
        raise ValueError("Unexpected input geometry.")

    if not with_batch_dim and len(data.shape) not in [3, 4]:
        raise ValueError("Unexpected input geometry.")

    # Do we have a 2D or 3D tensor (excluding batch and channel dimensions)?
    num_spatial_dims = len(data.shape) - (1 if with_batch_dim else 0) - 1
    return num_spatial_dims


def extract_subvolume(image, bbox):
    """
    Extracts a subset from a given 2D or 3D image using a bounding box.

    Parameters:
    image (np.ndarray): The n-dimensional image.
    bbox (tuple): The bounding box with format (min_dim1, max_dim1, ..., min_dimN, max_dimN),
    where N is the number of dimensions in the image.

    Returns:
    np.ndarray: The extracted subvolume.
    """
    # Validate bounding box
    if len(bbox) != 2 * image.ndim:
        raise ValueError("Bounding box format does not match image dimensions.")

    # Construct the slice object for each dimension
    ndim = image.ndim
    slices = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))
    return image[slices]


def insert_subvolume(image, subvolume, bbox, masked: bool = False):
    """
    Inserts a subvolume into an image using a specified bounding box.

    Parameters
    ----------

    image: np.ndarray
        The original n-dimensional image or volume.

    subvolume: (np.ndarray)
        The subvolume or subimage to insert, which must fit within the dimensions specified by bbox.

    bbox: tuple
        The bounding box with format (min_dim1, max_dim1, ..., min_dimN, max_dimN),
        where N is the number of dimensions in the image. The bounding box specifies where to insert the subvolume.

    masked: bool (Optional)
        If True, overwrite only non-zero values in the image.

    Returns
    -------
    image: np.ndarray
        The image with the subvolume inserted.
    """
    ndim = image.ndim
    # Validate bounding box and sub-volume
    if len(bbox) != 2 * image.ndim:
        raise ValueError("Bounding box format does not match image dimensions.")
    if not (all(subvolume.shape[i] == bbox[i + ndim] - bbox[i] for i in range(ndim))):
        raise ValueError(
            "Sub-volume dimensions must match the bounding box dimensions."
        )

    # Construct the slice object for each dimension
    slices = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))

    # Insert the subvolume into the image
    if masked:
        non_neg_indices = subvolume > 0
        image[slices][non_neg_indices] = subvolume[non_neg_indices]
    else:
        image[slices] = subvolume
    return image
