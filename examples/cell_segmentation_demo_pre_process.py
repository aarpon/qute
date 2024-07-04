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

import json
from pathlib import Path

import userpaths
from natsort import natsorted

from qute.data.io import get_cell_segmentation_demo_dataset
from qute.preprocess import extract_fft_stats, extract_intensity_stats

DOWNLOAD_DIR = Path(userpaths.get_my_documents()) / "qute" / "data"
LOW_PERC = 0.5
HIGH_PERC = 99.5

# Download and extract the demo dataset if needed.
get_cell_segmentation_demo_dataset(DOWNLOAD_DIR, three_classes=True)

# Get all image and label file names
image_file_names = natsorted(
    list((Path(DOWNLOAD_DIR) / "demo_segmentation_3_classes" / "images").glob("*.tif"))
)
label_file_names = natsorted(
    list((Path(DOWNLOAD_DIR) / "demo_segmentation_3_classes" / "labels").glob("*.tif"))
)

# Extract intensity stats
mean, std, p_low, p_high = extract_intensity_stats(
    image_file_names,
    label_file_names,
    fg_classes=[1, 2],
    low_perc=LOW_PERC,
    high_perc=HIGH_PERC,
)

# Save the stats
stats = {"mean": mean, "std": std, "p_low": p_low, "p_high": p_high}
stats_filename = (
    Path(DOWNLOAD_DIR) / "demo_segmentation_3_classes" / "intensity_stats.json"
)
with open(stats_filename, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False)
print(f"Saved intensity stats to {stats_filename}")

# Print the stats
print(
    f"Dataset intensity stats: mean = {mean}, std = {std}, {LOW_PERC} percentile = {p_low}, "
    f"{HIGH_PERC} percentile = {p_high}."
)

# Extract FFT stats
mean_real, std_real, mean_imag, std_imag = extract_fft_stats(image_file_names)

# Save the stats
stats_fft = {
    "mean_real": mean_real,
    "std_real": std_real,
    "mean_imag": mean_imag,
    "std_imag": std_imag,
}
fft_filename = Path(DOWNLOAD_DIR) / "demo_segmentation_3_classes" / "fft_stats.json"
with open(fft_filename, "w", encoding="utf-8") as f:
    json.dump(stats_fft, f, ensure_ascii=False)
print(f"Saved FFT stats to {fft_filename}")

# Print the stats
print(
    f"Dataset FFT stats: mean_real = {mean_real}, std_real = {std_real}, "
    f"mean_imag = {mean_imag}, std_imag = {std_imag}."
)
