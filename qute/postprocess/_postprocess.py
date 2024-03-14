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
from pathlib import Path
from typing import Union

import numpy as np
from natsort import natsorted
from scipy.ndimage import binary_dilation
from skimage.measure import label
from skimage.morphology import ball, disk
from tifffile import imread, imwrite


# @TODO Turn this into a Transform
def two_class_mask_to_label(
    in_folder: Union[Path, str],
    out_folder: Union[Path, str],
    class_id: int = 1,
    border_thickness: int = 1,
    min_size: int = 1,
):
    """Converts individual objects from a two-class image to label while optionally dilating
    the individual instances (that are assumed to be spatially separated).

    To prevent fusion of objects, the border is rebuilt from the object of the given class.

    in_folder: Union[Path, str]
        Folder withe label images to process.

    out_folder: Union[Path, str]
        Target folder where to store the converted mask images.

    class_id: int (Optional, default is 1)
        The class id of the object.

    border_thickness: int (Optional, default is 1)
        Extension of the border to add to the objects.
        Set to 0 to omit.

    min_size: int (Optional, default is 1)
        Minimum size for objects to be kept.
    """

    # Make sure both the output folders exist
    Path(out_folder).mkdir(exist_ok=True)

    # Footprint for dilation
    footprint = None

    # Process all files
    for in_name in natsorted(in_folder.glob("*.tif")):

        # Read the image
        mask = imread(in_name)

        # Make sure to drop the singleton channel dimension, if it is there
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # Allocate output
        out = np.zeros(mask.shape, dtype=mask.dtype)

        # Create a black-and-white mask for the requested class_id
        mask = mask == class_id

        # Get all connected components
        labels = label(mask, background=0, return_num=False, connectivity=1)

        # Keep track of (valid) label number
        valid_label_number = 0

        # Process all individual labels
        for lbl in np.unique(labels):

            # Ignore background
            if lbl == 0:
                continue

            # Prepare the label for processing
            current_mask = np.zeros(mask.shape, dtype=mask.dtype)
            indices = labels == lbl
            current_mask[indices] = 1

            # Initialize the footprint if needed
            if border_thickness > 0 and footprint is None:
                if labels.ndim == 2:
                    footprint = disk(border_thickness)
                else:
                    footprint = ball(border_thickness)

            # Dilate the bw mask for this label
            if footprint is not None:
                current_mask = binary_dilation(current_mask, footprint)

            # Store in the output as label
            indices = current_mask == 1
            if np.sum(indices) >= min_size:
                out[indices] = valid_label_number
                valid_label_number += 1

        # Write the 2-class mask image to the output folder
        out_name = out_folder / f"{in_name.name}"
        imwrite(out_name, out)
        print(f"Saved {out_name}.")

    print("Post-processing completed.")
