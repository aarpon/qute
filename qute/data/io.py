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

from glob import glob
from pathlib import Path
from shutil import rmtree
from typing import Optional, Union
from zipfile import ZipFile

import requests
import userpaths

__doc__ = "Input/output commodity functions."
__all__ = [
    "get_cell_segmentation_demo_dataset",
    "get_cell_segmentation_idt_demo_dataset",
    "get_cell_restoration_demo_dataset",
]


def get_cell_segmentation_demo_dataset(
    download_dir: Optional[Union[Path, str]] = None, three_classes: bool = True
):
    """If not yet present, download and expands segmentation demo dataset.

    Parameters
    ----------

    download_dir: Path | str = Path()
        Directory where the cell segmentation datasets will be downloaded and extracted.

    three_classes: bool
        If True, the segmentation demo with three classes (Background, Object, Border) will be downloaded;
        if False, the segmentation demo with two classes (Background, Object).

    Returns
    -------

    path: Path of the extracted segmentation demo dataset.
    """

    # Data folder
    if download_dir is None:
        data_folder = Path(userpaths.get_my_documents()) / "qute" / "data"
    else:
        data_folder = Path(download_dir).resolve()

    # Make sure the folder exists
    data_folder.mkdir(parents=True, exist_ok=True)

    if three_classes:
        dataset_name = "demo_segmentation_3_classes"
        archive_url = "https://polybox.ethz.ch/index.php/s/YfPPa54ahuSfylJ/download"
        archive_file_size = 69416524
    else:
        dataset_name = "demo_segmentation_2_classes"
        archive_url = "https://polybox.ethz.ch/index.php/s/Q0H9O5OggjNjJ3s/download"
        archive_file_size = 69831143

    # Check if the demo data folder already exists
    demo_folder = data_folder / dataset_name
    images_folder = demo_folder / "images"
    labels_folder = demo_folder / "labels"

    # Is the data already present?
    if demo_folder.is_dir():
        if images_folder.is_dir():
            if len(glob(str(images_folder / "*.tif*"))) == 90:
                if labels_folder.is_dir():
                    if len(glob(str(labels_folder / "*.tif*"))) == 90:
                        return demo_folder

    # Is the zip archive already present?
    archive_found = False
    if (data_folder / f"{dataset_name}.zip").is_file():
        if (data_folder / f"{dataset_name}.zip").stat().st_size == archive_file_size:
            archive_found = True
        else:
            (data_folder / f"{dataset_name}.zip").unlink()

    if not archive_found:
        # Get binary stream
        r = requests.get(archive_url)

        # Target file
        with open(data_folder / f"{dataset_name}.zip", "wb") as f:
            f.write(r.content)

        # Inform
        num_bytes = (data_folder / f"{dataset_name}.zip").stat().st_size
        print(f"Downloaded '{dataset_name}.zip' ({num_bytes} bytes).")

    # Make sure there are no remnants of previous extractions
    if demo_folder.is_dir():
        rmtree(demo_folder)

    # Extract zip file
    with ZipFile(data_folder / f"{dataset_name}.zip", "r") as z:
        # Extract all the contents of zip file
        z.extractall(data_folder)

    return demo_folder


def get_cell_segmentation_idt_demo_dataset(
    download_dir: Optional[Union[Path, str]] = None
):
    """If not yet present, download and expands segmentation demo dataset.

    Parameters
    ----------

    download_dir: Path | str = Path()
        Directory where the cell segmentation datasets will be downloaded and extracted.

    Returns
    -------

    path: Path of the extracted segmentation demo dataset.
    """

    # Data folder
    if download_dir is None:
        data_folder = Path(userpaths.get_my_documents()) / "qute" / "data"
    else:
        data_folder = Path(download_dir).resolve()

    # Make sure the folder exists
    data_folder.mkdir(parents=True, exist_ok=True)

    # Dataset metadata
    dataset_name = "demo_segmentation_idt"
    archive_url = "https://polybox.ethz.ch/index.php/s/5OkHEZumO4UPMpf/download"
    archive_file_size = 69598065

    # Check if the demo data folder already exists
    demo_folder = data_folder / dataset_name
    images_folder = demo_folder / "images"
    labels_folder = demo_folder / "labels"

    # Is the data already present?
    if demo_folder.is_dir():
        if images_folder.is_dir():
            if len(glob(str(images_folder / "*.tif*"))) == 90:
                if labels_folder.is_dir():
                    if len(glob(str(labels_folder / "*.tif*"))) == 90:
                        return demo_folder

    # Is the zip archive already present?
    archive_found = False
    if (data_folder / f"{dataset_name}.zip").is_file():
        if (data_folder / f"{dataset_name}.zip").stat().st_size == archive_file_size:
            archive_found = True
        else:
            (data_folder / f"{dataset_name}.zip").unlink()

    if not archive_found:
        # Get binary stream
        r = requests.get(archive_url)

        # Target file
        with open(data_folder / f"{dataset_name}.zip", "wb") as f:
            f.write(r.content)

        # Inform
        num_bytes = (data_folder / f"{dataset_name}.zip").stat().st_size
        print(f"Downloaded '{dataset_name}.zip' ({num_bytes} bytes).")

    # Make sure there are no remnants of previous extractions
    if demo_folder.is_dir():
        rmtree(demo_folder)

    # Extract zip file
    with ZipFile(data_folder / f"{dataset_name}.zip", "r") as z:
        # Extract all the contents of zip file
        z.extractall(data_folder)

    return demo_folder


def get_cell_restoration_demo_dataset(download_dir: Optional[Union[Path, str]] = None):
    """If not yet present, download and expands restoration demo dataset.

    Parameters
    ----------

    download_dir: Path | str = Path()
        Directory where the cell segmentation datasets will be downloaded and extracted.

    three_classes = bool
        If True, the segmentation demo with three classes (Background, Object, Border) will be downloaded;
        if False, the segmentation demo with two classes (Background, Object).

    Returns
    -------

    path: Path of the extracted restoration demo dataset.
    """

    # Archive URL
    archive_url = "https://polybox.ethz.ch/index.php/s/3UWRc7hKLfop4s8/download"

    # Data folder
    if download_dir is None:
        data_folder = Path(userpaths.get_my_documents()) / "qute" / "data"
    else:
        data_folder = Path(download_dir).resolve()

    # Make sure the folder exists
    data_folder.mkdir(parents=True, exist_ok=True)

    # Check if the demo data folder already exists
    demo_folder = data_folder / "demo_restoration"
    images_folder = demo_folder / "images"
    targets_folder = demo_folder / "targets"

    # Is the data already present?
    if demo_folder.is_dir():
        if images_folder.is_dir():
            if len(glob(str(images_folder / "*.tif*"))) == 90:
                if targets_folder.is_dir():
                    if len(glob(str(targets_folder / "*.tif*"))) == 90:
                        return demo_folder

    # Is the zip archive already present?
    archive_found = False
    if (data_folder / "demo_restoration.zip").is_file():
        if (data_folder / "demo_restoration.zip").stat().st_size == 113365383:
            archive_found = True
        else:
            (data_folder / "demo_restoration.zip").unlink()

    if not archive_found:
        # Get binary stream
        r = requests.get(archive_url)

        # Target file
        with open(data_folder / "demo_restoration.zip", "wb") as f:
            f.write(r.content)

        # Inform
        print(
            f"Downloaded 'demo_restoration.zip' ({(data_folder / 'demo_restoration.zip').stat().st_size} bytes)."
        )

    # Make sure there are no remnants of previous extractions
    if demo_folder.is_dir():
        rmtree(demo_folder)

    # Extract zip file
    with ZipFile(data_folder / "demo_restoration.zip", "r") as z:
        # Extract all the contents of zip file
        z.extractall(data_folder)

    return demo_folder
