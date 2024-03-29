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

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from natsort import natsorted
from scipy.fft import fft2
from scipy.ndimage import binary_erosion
from skimage.morphology import ball, disk
from tifffile import imread, imwrite
from tqdm import tqdm


def extract_intensity_stats(
    image_list: list,
    label_list: list,
    fg_classes: Optional[list],
    low_perc: float = 0.5,
    high_perc: float = 99.5,
) -> tuple:
    """Returns min, max, low and high percentile of intensities over foreground classes.

    Parameters
    ----------
    image_list: list
        List of paths for the intensity TIFF images.

    label_list: list
        List of paths for the label TIFF images. Importantly, label image at index i must
        correspond to intensity image at index i.

    fg_classes: Optional[list[int]]
        List of foreground classes to be considered to extract the intensities to process. If omitted,
        all classes but 0 (background) will be used.

    low_perc: Optional[float]
        Low percentile. Default 0.5

    high_perc: Optional[float]
        High percentile. Default 99.5

    Please note: the image lists are supposed to be sorted so that the `ith` element of one list matches
    the `ith` element of the other.

    Returns
    -------

    mean: global mean foreground intensity
    std: global standard deviation of the foreground intensities
    p_low: low percentile of all foreground intensities
    p_high: high percentile of all foreground intensities
    """

    # Check the arguments
    if len(image_list) != len(label_list):
        raise ValueError("Image and label lists must have the same lengths.")

    # Check the percentiles
    if low_perc < 0 or low_perc > high_perc or low_perc >= 100.0:
        raise ValueError(
            "The low percentile must be between 0.0 and 100.0 and be lower than the high percentile."
        )
    if high_perc < 0 or low_perc > high_perc or high_perc >= 100.0:
        raise ValueError(
            "The high percentile must be between 0.0 and 100.0 and be higher than the low percentile."
        )

    # Initialize an array to hold all intensities
    all_foreground_intensities = None

    # Process all images
    for image_name, label_name in tqdm(
        zip(image_list, label_list), total=len(image_list)
    ):

        # Read images
        image = imread(image_name)
        label = imread(label_name)

        # Create a mask using the requested foreground classes
        mask = np.zeros(label.shape, dtype=bool)
        for c in fg_classes:
            mask = np.logical_or(mask, label == c)

        # Extract the intensities over the foreground masks
        intensities = image[mask]

        # Append to all_foreground_intensities
        if all_foreground_intensities is None:
            all_foreground_intensities = intensities
        else:
            all_foreground_intensities = np.concatenate(
                (all_foreground_intensities, intensities), axis=0
            )

    # Calculate the statistics
    p_low, p_high = np.percentile(all_foreground_intensities, (low_perc, high_perc))
    mean = all_foreground_intensities.mean()
    std = np.max([all_foreground_intensities.std(), 1e-8])

    # Return the extracted statistics
    return mean, std, p_low, p_high


def extract_fft_stats(
    image_list: list,
) -> tuple:
    """Returns global min and max values for the real and imaginary parts of the Fourier transforms of all images.

    Parameters
    ----------
    image_list: list
        List of paths for the intensity TIFF images.

    Returns
    -------

    mean_real: global mean of all real parts from all Fourier transforms.
    std_real: global std of all real parts from all Fourier transforms.
    mean_imag: global mean of all imaginary parts from all Fourier transforms.
    std_imag: global std of all imaginary parts from all Fourier transforms.
    """

    # Initialize arrays to hold all real and imaginary components
    all_real_components = None
    all_imag_components = None

    # Process all images
    for image_name in tqdm(image_list, total=len(image_list)):

        # Read image
        image = imread(image_name)

        # Calculate the Fourier transform
        f = fft2(image, workers=os.cpu_count())

        # Append components to arrays
        if all_real_components is None:
            all_real_components = f.real.ravel()
        else:
            all_real_components = np.concatenate(
                (all_real_components, f.real.ravel()), axis=0
            )
        if all_imag_components is None:
            all_imag_components = f.imag.ravel()
        else:
            all_imag_components = np.concatenate(
                (all_imag_components, f.imag.ravel()), axis=0
            )

    # Calculate the statistics
    mean_real = all_real_components.mean()
    std_real = np.max([all_real_components.std(), 1e-8])
    mean_imag = all_imag_components.mean()
    std_imag = np.max([all_imag_components.std(), 1e-8])

    # Return the extracted statistics
    return mean_real, std_real, mean_imag, std_imag


# @TODO Turn this into a Transform
def labels_to_two_class_masks(
    in_folder: Union[Path, str], out_folder: Union[Path, str], border_thickness: int = 1
):
    """Converts a labels image or stack to a masks with two non-background (0) classes: object (1) and border (2).

    Converts one class from a multi-class image to a labels image while optionally dilating
    the individual instances (that are assumed to be spatially separated).

    in_folder: Union[Path, str]
        Folder withe label images to process.

    out_folder: Union[Path, str]
        Target folder where to store the converted mask images.

    border_thickness: int (Optional, default is 1)
        Extension of the border to add to the objects.
        Set to 0 to omit.
    """

    def process_label(lbl, footprint):
        """Create and add object borders as new class."""

        # Add the border as class 2
        eroded = binary_erosion(lbl, footprint)
        border = lbl - eroded
        return lbl + border  # Count border pixels twice

    # Make sure both the output folders exist
    Path(out_folder).mkdir(exist_ok=True)

    # Footprint for erosion
    footprint = None

    # Process all files
    for in_name in natsorted(in_folder.glob("*.tif")):
        lbl = imread(in_name)

        # Allocate output
        out = np.zeros(lbl.shape, dtype=lbl.dtype)

        # Process all individual labels
        for l in np.unique(lbl):

            # Ignore background
            if l == 0:
                continue

            # Prepare the label for processing
            mask = np.zeros(lbl.shape, dtype=lbl.dtype)
            indices = lbl == l
            mask[indices] = 1

            # Initialize the footprint if needed
            if footprint is None:
                if lbl.ndim == 2:
                    footprint = disk(border_thickness)
                else:
                    footprint = ball(border_thickness)

            # Create a 2-class mask for this label
            mask_tc = process_label(mask, footprint)

            # Store in the output
            out[indices] = mask_tc[indices]

        # Write the 2-class mask image to the output folder
        out_name = out_folder / f"{in_name.name}"
        imwrite(out_name, out)
        print(out_name)
