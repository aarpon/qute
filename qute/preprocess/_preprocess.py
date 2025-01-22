# ******************************************************************************
# Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Apache License Version 2.0
# which accompanies this distribution, and is available at
# https://www.apache.org/licenses/LICENSE-2.0.txt
#
# Contributors:
#   Aaron Ponti - initial API and implementation
# ******************************************************************************

import os
from typing import Optional

import numpy as np
from scipy.fft import fft2
from skimage.measure import regionprops
from tifffile import imread
from tqdm import tqdm


def extract_intensity_stats(
    image_list: list,
    mask_list: list,
    fg_classes: Optional[list] = None,
    low_perc: float = 0.5,
    high_perc: float = 99.5,
) -> tuple:
    """Returns min, max, low and high percentile of intensities over foreground classes.

    Parameters
    ----------
    image_list: list
        List of paths for the intensity TIFF images.

    mask_list: list
        List of paths for the mask TIFF images. Importantly, label image at index i must
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
    if len(image_list) != len(mask_list):
        raise ValueError("Image and mask lists must have the same lengths.")

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
    for image_name, mask_name in tqdm(
        zip(image_list, mask_list), total=len(image_list)
    ):
        # Read images
        image = imread(image_name)
        mask = imread(mask_name)

        # Create a mask using the requested foreground classes
        bw = np.zeros(mask.shape, dtype=bool)
        if fg_classes is not None:
            for c in fg_classes:
                bw = np.logical_or(bw, mask == c)
        else:
            bw = mask > 0

        # Extract the intensities over the foreground masks
        intensities = image[bw]

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


def extract_median_object_size(label_list: list) -> tuple[float, float, float]:
    """Returns the median size of all labels.

    Parameters
    ----------

    label_list: list
        List of paths for the label TIFF images.

    Returns
    -------

    mn: float
        Min size of all objects.

    med: float
        Median size of all labels.

    mx: float
        Max size of all labels.
    """

    all_sizes = []

    # Process all images
    for label_name in tqdm(label_list):
        # Read images
        label_img = imread(label_name)

        # Process all labels
        props = regionprops(label_img)

        for prop in props:
            all_sizes.append(prop.feret_diameter_max)

    # Return the median of all sizes
    return np.min(all_sizes), np.median(all_sizes), np.max(all_sizes)


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
