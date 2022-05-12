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

from typing import Union
from glob import glob
from pathlib import Path
from shutil import rmtree
from zipfile import ZipFile
import requests


def get_cell_segmentation_dataset(download_dir: Union[Path, str] = None, three_classes: bool = True):
    """If not yet present, download and expands segmentation demo dataset.
    
    Parameters
    ----------

    download_dir: Path | str = Path()
        Directory where the cell segmentation datasets will be downloaded and extracted.

    three_classes = bool
        If True, the segmentation demo with three classes (Background, Object, Border) will be downloaded;
        if False, the segmentation demo with two classes (Background, Object).
    
    Returns
    -------

    path: Path of the extracted segmentation demo dataset.
    """

    # Data folder
    if download_dir is None:
        data_folder = Path.home() / ".qute" / "data"
    else:
        data_folder = Path(download_dir).resolve()

    # Make sure the folder exists
    data_folder.mkdir(parents=True, exist_ok=True)

    if three_classes:
        dataset_name = "demo_segmentation_3_classes"
        archive_file_size = 69416524
    else:
        dataset_name = "demo_segmentation_2_classes"
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
        r = requests.get(f"https://obit.ethz.ch/qute/{dataset_name}.zip")

        # Target file
        with open(data_folder / f"{dataset_name}.zip", 'wb') as f:
            f.write(r.content)

        # Inform
        num_bytes = (data_folder / f"{dataset_name}.zip").stat().st_size
        print(f"Downloaded '{dataset_name}.zip' ({num_bytes} bytes).")

    # Make sure there are no remnants of previous extractions
    if demo_folder.is_dir():
        rmtree(demo_folder)

    # Extract zip file
    with ZipFile(data_folder / f"{dataset_name}.zip", 'r') as z:
        # Extract all the contents of zip file
        z.extractall(data_folder)

    return demo_folder


def get_cell_restoration_dataset(download_dir: Union[Path, str] = None):
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

    # Data folder
    if download_dir is None:
        data_folder = Path.home() / ".qute" / "data"
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
        r = requests.get("https://obit.ethz.ch/qute/demo_restoration.zip")

        # Target file
        with open(data_folder / "demo_restoration.zip", 'wb') as f:
            f.write(r.content)

        # Inform
        print(f"Downloaded 'demo_restoration.zip' ({(data_folder / 'demo_restoration.zip').stat().st_size} bytes).")

    # Make sure there are no remnants of previous extractions
    if demo_folder.is_dir():
        rmtree(demo_folder)

    # Extract zip file
    with ZipFile(data_folder / "demo_restoration.zip", 'r') as z:
        # Extract all the contents of zip file
        z.extractall(data_folder)

    return demo_folder
