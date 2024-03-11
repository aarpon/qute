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

import zipfile
from pathlib import Path

import pytest
import torch

from qute.transforms import (
    AddBorderd,
    AddNormalizedDistanceTransform,
    AddNormalizedDistanceTransformd,
    CellposeLabelReader,
    CustomTIFFReader,
)


@pytest.fixture(autouse=False)
def extract_test_transforms_data(tmpdir):
    """Fixture to execute asserts before and after a test is run"""

    #
    # Setup
    #

    expected_files = [
        Path(__file__).parent / "data" / "cellpose.npy",
        Path(__file__).parent / "data" / "labels.tif",
    ]

    # Make sure to extract the test data if it is not already there
    need_to_extract = False
    for file_name in expected_files:
        if not file_name.is_file():
            need_to_extract = True
            break

    if need_to_extract:
        archive_filename = Path(__file__).parent / "data" / "test_transforms_data.zip"
        with zipfile.ZipFile(archive_filename, "r") as zip_ref:
            zip_ref.extractall(Path(__file__).parent / "data")

    yield  # This is where the testing happens

    #
    # Teardown
    #

    # Do whatever is needed to clean up:
    # - Nothing for the moment


def test_add_borderd(extract_test_transforms_data):

    # Load CellPose dataset
    reader = CellposeLabelReader()
    label = reader(Path(__file__).parent / "data" / "cellpose.npy")
    assert label.shape == (300, 300), "Unexpected image size."

    # Add to data dictionary
    data = {}
    data["label"] = label

    # Add the border
    addb = AddBorderd()
    data = addb(data)

    # Make sure that the batch and channel dimensions are present
    assert data["label"].shape == (300, 300), "Unexpected image size."

    # Check that there are indeed three classes
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."


def test_custom_tiff_reader(extract_test_transforms_data):

    # Load TIFF file with default arguments
    reader = CustomTIFFReader()
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert not hasattr(image, "meta"), "Metadata should not be present."

    # Load TIFF file with (ensure_channel_first=False)
    reader = CustomTIFFReader(ensure_channel_first=False)
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (26, 300, 300), "Unexpected image shape."
    assert not hasattr(image, "meta"), "Metadata should not be present."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=True)
    reader = CustomTIFFReader(ensure_channel_first=True, as_meta_tensor=True)
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert image.meta["pixdim"] == (1.0, 1.0, 1.0), "Wrong voxel size."

    # Load TIFF file with (ensure_channel_first=False, as_meta_tensor=True)
    reader = CustomTIFFReader(ensure_channel_first=False, as_meta_tensor=True)
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert image.meta["pixdim"] == (1.0, 1.0, 1.0), "Wrong voxel size."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=True, pixdim=(0.5, 0.1, 0.1))
    reader = CustomTIFFReader(
        ensure_channel_first=True, as_meta_tensor=True, pixdim=(0.5, 0.1, 0.1)
    )
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert image.meta["pixdim"] == (0.5, 0.1, 0.1), "Wrong voxel size."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=False, pixdim=(0.5, 0.1, 0.1))
    reader = CustomTIFFReader(
        ensure_channel_first=True, as_meta_tensor=False, pixdim=(0.5, 0.1, 0.1)
    )
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert not hasattr(image, "meta"), "Metadata should not be present."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=True, pixdim=(0.5, 0.1, 0.1), dtype=torch.int32)
    reader = CustomTIFFReader(
        ensure_channel_first=True,
        as_meta_tensor=True,
        pixdim=(0.5, 0.1, 0.1),
        dtype=torch.int32,
    )
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert image.meta["pixdim"] == (0.5, 0.1, 0.1), "Wrong voxel size."
    assert image.dtype == torch.int32, "Unexpected datatype."


def test_add_normalized_transform(extract_test_transforms_data):

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(dtype=torch.int32)
    label = reader(Path(__file__).parent / "data" / "labels.tif")
    assert label.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert not hasattr(label, "meta"), "Metadata should not be present."

    # Make sure this is a label image
    assert len(torch.unique(label)) == 69, "Wrong number of labels."

    #
    # AddNormalizedDistanceTransformd
    #

    # Create the dictionary
    image = torch.zeros(label.shape, dtype=torch.float32)
    data = {"image": image, "label": label}

    # Pass it to the Transform
    ndt = AddNormalizedDistanceTransformd(
        image_key="image", label_key="label", reverse=True, do_not_zero=True
    )
    data_out = ndt(data)

    assert data_out["image"].shape == (2, 26, 300, 300), "Unexpected image shape."
    assert (
        torch.min(data_out["image"][1, :, :, :][data_out["image"][1, :, :, :] > 0])
        == 0.1250
    ), "Unexpected minimum pixel value."
    assert (
        torch.max(data_out["image"][1, :, :, :]) == 1.0
    ), "Unexpected maximum pixel value."

    #
    # AddNormalizedDistanceTransform
    #

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(dtype=torch.int32)
    label = reader(Path(__file__).parent / "data" / "labels.tif")

    # Transform the image
    ndt = AddNormalizedDistanceTransform(reverse=True, do_not_zero=True)
    label_out = ndt(label)

    assert label_out.shape == (2, 26, 300, 300), "Unexpected image shape."
    assert (
        torch.min(label_out[1, :, :, :][label_out[1, :, :, :] > 0]) == 0.1250
    ), "Unexpected minimum pixel value."
    assert torch.max(label_out[1, :, :, :]) == 1.0, "Unexpected maximum pixel value."

    # Compare with the previous result
    assert torch.all(
        data_out["image"][1, :, :, :] == label_out[1, :, :, :]
    ), "Unexpected result."
