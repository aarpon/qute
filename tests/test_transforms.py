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

import numpy as np
import pytest
import torch
from monai.data import MetaTensor
from monai.transforms import Spacing
from tifffile import imread

from qute.transforms import (
    AddBorderd,
    NormalizedDistanceTransform,
    NormalizedDistanceTransformd,
    OneHotToMask,
    OneHotToMaskBatch,
)
from qute.transforms.io import (
    CellposeLabelReader,
    CustomND2Reader,
    CustomND2Readerd,
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

    # Check that there are indeed three classes
    assert data["label"].shape == (300, 300), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."

    #
    # Test support for various dimensions
    #

    # Simple 2D
    input_2d = torch.zeros((10, 10), dtype=torch.int32)
    input_2d[4:7, 4:7] = 1
    data["label"] = input_2d
    addb = AddBorderd(label_key="label", border_width=1)
    data = addb(data)

    # Check that there are indeed three classes with the expected number of pixels
    assert data["label"].shape == (10, 10), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."
    assert torch.sum(data["label"] == 1) == 1, "One pixel with class 1 expected."
    assert torch.sum(data["label"] == 2) == 8, "Eight pixels with class 2 expected."

    # 2D with channel first
    input_2d_c = torch.zeros((1, 10, 10), dtype=torch.int32)
    input_2d_c[0, 4:7, 4:7] = 1
    data["label"] = input_2d_c
    addb = AddBorderd(label_key="label", border_width=1)
    data = addb(data)

    # Check that there are indeed three classes with the expected number of pixels
    assert data["label"].shape == (1, 10, 10), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."
    assert torch.sum(data["label"] == 1) == 1, "One pixel with class 1 expected."
    assert torch.sum(data["label"] == 2) == 8, "Eight pixels with class 2 expected."

    # 3D
    input_3d = torch.zeros((10, 10, 10), dtype=torch.int32)
    input_3d[4:7, 4:7, 4:7] = 1
    data["label"] = input_3d
    addb = AddBorderd(label_key="label", border_width=1)
    data = addb(data)

    # Check that there are indeed three classes with the expected number of pixels
    assert data["label"].shape == (10, 10, 10), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."
    assert torch.sum(data["label"] == 1) == 1, "One pixel with class 1 expected."
    assert (
        torch.sum(data["label"] == 2) == 26
    ), "Twenty-six pixels with class 2 expected."

    # 3D with channel first
    input_3d_c = torch.zeros((1, 10, 10, 10), dtype=torch.int32)
    input_3d_c[0, 4:7, 4:7, 4:7] = 1
    data["label"] = input_3d_c
    addb = AddBorderd(label_key="label", border_width=1)
    data = addb(data)

    # Check that there are indeed three classes with the expected number of pixels
    assert data["label"].shape == (1, 10, 10, 10), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."
    assert torch.sum(data["label"] == 1) == 1, "One pixel with class 1 expected."
    assert (
        torch.sum(data["label"] == 2) == 26
    ), "Twenty-six pixels with class 2 expected."

    # Make sure that 3D with more than one channel is not supported
    input_3d_2c = torch.zeros((2, 10, 10, 10), dtype=torch.int32)  # 2 channels
    input_3d_2c[0, 4:7, 4:7, 4:7] = 1
    data["label"] = input_3d_2c
    addb = AddBorderd(label_key="label", border_width=1)
    with pytest.raises(Exception):
        _ = addb(data)


def test_custom_tiff_reader(extract_test_transforms_data):

    # Load TIFF file with default arguments
    reader = CustomTIFFReader()
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert not hasattr(image, "meta"), "Metadata should not be present."
    assert not hasattr(image, "affine"), "'affine' property should not be present."

    # Load TIFF file with (ensure_channel_first=False)
    reader = CustomTIFFReader(ensure_channel_first=False)
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (26, 300, 300), "Unexpected image shape."
    assert not hasattr(image, "meta"), "Metadata should not be present."
    assert not hasattr(image, "affine"), "'affine' property should not be present."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=True)
    reader = CustomTIFFReader(ensure_channel_first=True, as_meta_tensor=True)
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert hasattr(image, "affine"), "'affine' property missing."
    assert torch.all(
        image.affine == image.meta["affine"]
    ), "Inconsistent affine matrices."
    assert torch.all(
        image.meta["affine"].diag() == torch.tensor((1.0, 1.0, 1.0, 1.0))
    ), "Wrong voxel size."
    assert image.meta["affine"].shape == (4, 4), "Unexpected affine shape."

    # Load TIFF file with (ensure_channel_first=False, as_meta_tensor=True)
    reader = CustomTIFFReader(ensure_channel_first=False, as_meta_tensor=True)
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert hasattr(image, "affine"), "'affine' property missing."
    assert torch.all(
        image.affine == image.meta["affine"]
    ), "Inconsistent affine matrices."
    assert torch.all(
        image.meta["affine"].diag() == torch.tensor((1.0, 1.0, 1.0, 1.0))
    ), "Wrong voxel size."
    assert image.meta["affine"].shape == (4, 4), "Unexpected affine shape."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=True, voxel_size=(0.5, 0.1, 0.1))
    reader = CustomTIFFReader(
        ensure_channel_first=True, as_meta_tensor=True, voxel_size=(0.5, 0.1, 0.1)
    )
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert hasattr(image, "affine"), "'affine' property missing."
    assert torch.all(
        image.affine == image.meta["affine"]
    ), "Inconsistent affine matrices."
    assert torch.all(
        torch.isclose(
            image.meta["affine"].diag(),
            torch.tensor((0.5, 0.1, 0.1, 1.0), dtype=image.meta["affine"].dtype),
        )
    ), "Wrong voxel size."
    assert image.meta["affine"].shape == (4, 4), "Unexpected affine shape."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=False, voxel_size=(0.5, 0.1, 0.1))
    reader = CustomTIFFReader(
        ensure_channel_first=True, as_meta_tensor=False, voxel_size=(0.5, 0.1, 0.1)
    )
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert not hasattr(image, "meta"), "Metadata should not be present."
    assert not hasattr(image, "affine"), "'affine' property should not be present."

    # Load TIFF file with (ensure_channel_first=True, as_meta_tensor=True, voxel_size=(0.5, 0.1, 0.1), dtype=torch.int32)
    reader = CustomTIFFReader(
        ensure_channel_first=True,
        as_meta_tensor=True,
        voxel_size=(0.5, 0.1, 0.1),
        dtype=torch.int32,
    )
    image = reader(Path(__file__).parent / "data" / "labels.tif")
    assert image.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert hasattr(image, "meta"), "Missing metadata."
    assert hasattr(image, "affine"), "'affine' property missing."
    assert torch.all(
        image.affine == image.meta["affine"]
    ), "Inconsistent affine matrices."
    assert torch.all(
        torch.isclose(
            image.meta["affine"].diag(),
            torch.tensor((0.5, 0.1, 0.1, 1.0), dtype=image.meta["affine"].dtype),
        )
    ), "Wrong voxel size."
    assert image.meta["affine"].shape == (4, 4), "Unexpected affine shape."
    assert image.dtype == torch.int32, "Unexpected datatype."

    # Apply spacing
    sp = Spacing(
        pixdim=(1.0, 0.2, 0.2),
        mode="nearest",
    )
    resampled_image = sp(image)
    assert resampled_image.shape == (1, 14, 151, 151), "Unexpected image shape."
    assert hasattr(resampled_image, "meta"), "Missing metadata."
    assert hasattr(resampled_image, "affine"), "'affine' property missing."
    assert torch.all(
        resampled_image.affine == resampled_image.meta["affine"]
    ), "Inconsistent affine matrices."
    assert torch.all(
        torch.isclose(
            resampled_image.meta["affine"].diag(),
            torch.tensor(
                (1.0, 0.2, 0.2, 1.0), dtype=resampled_image.meta["affine"].dtype
            ),
        )
    ), "Wrong voxel size."
    assert resampled_image.meta["affine"].shape == (4, 4), "Unexpected affine shape."
    assert resampled_image.dtype == torch.float32, "Unexpected datatype."

    #
    # Test back-and-forth transformation on a larger dataset
    #

    # Set initial and target voxel sizes
    voxel_size = [1.0, 0.241, 0.241]
    target_voxel_size = [0.241, 0.241, 0.241]

    # Initialize source tensor
    in_affine = torch.zeros((4, 4))
    in_affine[0, 0] = voxel_size[0]
    in_affine[1, 1] = voxel_size[1]
    in_affine[2, 2] = voxel_size[2]
    in_affine[3, 3] = 1.0
    source = MetaTensor(
        torch.zeros((1, 20, 300, 300), dtype=torch.int32), affine=in_affine
    )

    # Forward transform
    sp = Spacing(pixdim=target_voxel_size, mode="nearest")
    target = sp(source)

    assert target.shape == (1, 80, 300, 300), "Unexpected target shape."

    # Inverse transform
    inv_sp = Spacing(pixdim=voxel_size, mode="nearest")
    inv_source = inv_sp(target)

    assert inv_source.shape == source.shape, "Unexpected inverted source shape."
    assert inv_source.shape == (1, 20, 300, 300), "Unexpected inverted source shape."


def test_add_normalized_transform(extract_test_transforms_data):

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(dtype=torch.int32)
    label = reader(Path(__file__).parent / "data" / "labels.tif")
    assert label.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert not hasattr(label, "meta"), "Metadata should not be present."

    # Make sure this is a label image
    assert len(torch.unique(label)) == 69, "Wrong number of labels."

    #
    # NormalizedDistanceTransformd
    #

    # Create the dictionary
    image = torch.zeros(label.shape, dtype=torch.float32)
    data = {"image": image, "label": label}

    # Pass it to the Transform
    ndt = NormalizedDistanceTransformd(keys=("label",), reverse=True, do_not_zero=True)
    data_out = ndt(data)

    assert data_out["image"].shape == (1, 26, 300, 300), "Unexpected image shape."
    assert data_out["label"].shape == (1, 26, 300, 300), "Unexpected labels shape."
    assert torch.all(data_out["image"] == 0.0)
    assert (
        torch.min(data_out["label"][data_out["label"] > 0]) == 0.1250
    ), "Unexpected minimum pixel value."
    assert torch.max(data_out["label"] == 1.0), "Unexpected maximum pixel value."

    #
    # NormalizedDistanceTransform
    #

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(dtype=torch.int32, as_meta_tensor=True)
    label = reader(Path(__file__).parent / "data" / "labels.tif")

    # Transform the image
    ndt = NormalizedDistanceTransform(reverse=True, do_not_zero=True)
    label_out = ndt(label)

    assert label_out.shape == (1, 26, 300, 300), "Unexpected image shape."
    assert (
        torch.min(label_out[label_out > 0]) == 0.1250
    ), "Unexpected minimum pixel value."
    assert torch.max(label_out) == 1.0, "Unexpected maximum pixel value."

    # Compare with the previous result
    assert torch.all(data_out["label"] == label_out), "Unexpected result."


def test_to_label(tmpdir):

    # Create 2D label ground truth (classes 0, 1, 2)
    gt_2d = torch.zeros((1, 60, 60), dtype=torch.int32)
    gt_2d[0, :, 20:40] = 1
    gt_2d[0, :, 40:60] = 2

    # Create 3D label ground truth (classes 0, 1, 2)
    gt_3d = torch.zeros((1, 10, 60, 60), dtype=torch.int32)
    gt_3d[0, :, :, 20:40] = 1
    gt_3d[0, :, :, 40:60] = 2

    # Create 2D one-hot data (CHW) (data 1, 2, 3)
    oh_2d = torch.zeros((3, 60, 60), dtype=torch.int32)
    oh_2d[0, :, 0:20] = 1
    oh_2d[1, :, 20:40] = 2
    oh_2d[2, :, 40:60] = 3

    # Create 3D one-hot data (CDHW) (data 1, 2, 3)
    oh_3d = torch.zeros((3, 10, 60, 60), dtype=torch.int32)
    oh_3d[0, :, :, 0:20] = 1
    oh_3d[1, :, :, 20:40] = 2
    oh_3d[2, :, :, 40:60] = 3

    # Check the synthetic data
    assert gt_2d.shape == (1, 60, 60), "Unexpected 2D ground truth shape."
    assert gt_3d.shape == (1, 10, 60, 60), "Unexpected 3D ground truth shape."
    assert oh_2d.shape == (3, 60, 60), "Unexpected 2D one-hot truth shape."
    assert oh_3d.shape == (3, 10, 60, 60), "Unexpected 3D one-hot truth shape."
    assert oh_2d.shape == (3, 60, 60), "Unexpected 2D one-hot truth shape."
    assert oh_3d.shape == (3, 10, 60, 60), "Unexpected 3D one-hot truth shape."

    #
    # Test the 2D data
    #

    # Call OneHotToMask() and check the output
    to_label = OneHotToMask()
    out_oh_2d = to_label(oh_2d)

    assert out_oh_2d.shape == (
        1,
        60,
        60,
    ), "Unexpected shape of result of OneHotToMask() for 2D data."
    assert torch.equal(
        gt_2d, out_oh_2d
    ), "Result of OneHotToMask() for 2D data does not match ground truth."

    #
    # Test the 3D data
    #

    # Call OneHotToMask() on oh_3d and check the output
    to_label = OneHotToMask()
    out_oh_3d = to_label(oh_3d)

    assert out_oh_3d.shape == (
        1,
        10,
        60,
        60,
    ), "Unexpected shape of result of OneHotToMask() for 3D data."
    assert torch.equal(
        gt_3d, out_oh_3d
    ), "Result of OneHotToMask() for 3D data does not match ground truth."

    # Check that other shapes are not supported
    to_label = OneHotToMask()

    with pytest.raises(ValueError) as e_info:
        # Single 2D image (H, W)
        _ = to_label(torch.zeros((60, 60), dtype=torch.int32))

    with pytest.raises(ValueError) as e_info:
        # Batch of 3D images (B, C, D, H, W)
        _ = to_label(torch.zeros((3, 3, 10, 60, 60), dtype=torch.int32))


def test_to_label_batch(tmpdir):

    # Create batched 2D label ground truth (classes 0, 1, 2)
    gt_2d = torch.zeros((2, 1, 60, 60), dtype=torch.int32)
    gt_2d[0, 0, :, 20:40] = 1
    gt_2d[0, 0, :, 40:60] = 2
    gt_2d[1, 0, :, 20:40] = 1
    gt_2d[1, 0, :, 40:60] = 2

    # Create batched 3D label ground truth (classes 0, 1, 2)
    gt_3d = torch.zeros((2, 1, 10, 60, 60), dtype=torch.int32)
    gt_3d[0, 0, :, :, 20:40] = 1
    gt_3d[0, 0, :, :, 40:60] = 2
    gt_3d[1, 0, :, :, 20:40] = 1
    gt_3d[1, 0, :, :, 40:60] = 2

    # Create batched 2D one-hot data (BCHW) (data 1, 2, 3)
    oh_2d = torch.zeros((2, 3, 60, 60), dtype=torch.int32)
    oh_2d[0, 0, :, 0:20] = 1  # B = 0
    oh_2d[0, 1, :, 20:40] = 2
    oh_2d[0, 2, :, 40:60] = 3
    oh_2d[
        1, 0, :, 0:20
    ] = 4  # B = 1 (different values, to test that result in the same classifications)
    oh_2d[1, 1, :, 20:40] = 5
    oh_2d[1, 2, :, 40:60] = 6

    # Create 3D one-hot data (CDHW) (data 1, 2, 3)
    oh_3d = torch.zeros((2, 3, 10, 60, 60), dtype=torch.int32)
    oh_3d[0, 0, :, :, 0:20] = 1  # B = 0
    oh_3d[0, 1, :, :, 20:40] = 2
    oh_3d[0, 2, :, :, 40:60] = 3
    oh_3d[
        1, 0, :, :, 0:20
    ] = 4  # B = 1 (different values, to test that result in the same classifications)
    oh_3d[1, 1, :, :, 20:40] = 5
    oh_3d[1, 2, :, :, 40:60] = 6

    # Check the synthetic data
    assert gt_2d.shape == (2, 1, 60, 60), "Unexpected 2D batched ground truth shape."
    assert gt_3d.shape == (
        2,
        1,
        10,
        60,
        60,
    ), "Unexpected 3D batched  ground truth shape."
    assert oh_2d.shape == (2, 3, 60, 60), "Unexpected 2D batched one-hot truth shape."
    assert oh_3d.shape == (
        2,
        3,
        10,
        60,
        60,
    ), "Unexpected batched 3D one-hot truth shape."
    assert oh_2d.shape == (2, 3, 60, 60), "Unexpected 2D batched one-hot truth shape."
    assert oh_3d.shape == (
        2,
        3,
        10,
        60,
        60,
    ), "Unexpected 3D batched one-hot truth shape."

    #
    # Test the batched 2D data
    #

    # Call OneHotToMask() and check the output
    to_label_batch = OneHotToMaskBatch()
    out_oh_2d = to_label_batch(oh_2d)

    assert out_oh_2d.shape == (
        2,
        1,
        60,
        60,
    ), "Unexpected shape of result of OneHotToMask() for 2D data."
    assert torch.equal(
        gt_2d, out_oh_2d
    ), "Result of OneHotToMask() for 2D data does not match ground truth."

    #
    # Test the 3D data
    #

    # Call OneHotToMask() on oh_3d and check the output
    to_label_batch = OneHotToMaskBatch()
    out_oh_3d = to_label_batch(oh_3d)

    assert out_oh_3d.shape == (
        2,
        1,
        10,
        60,
        60,
    ), "Unexpected shape of result of OneHotToMask() for 3D data."
    assert torch.equal(
        gt_3d, out_oh_3d
    ), "Result of OneHotToMask() for 3D data does not match ground truth."

    # Check that other shapes are not supported
    to_label_batch = OneHotToMaskBatch()

    with pytest.raises(ValueError) as e_info:
        # Single 2D image (H, W)
        _ = to_label_batch(torch.zeros((60, 60), dtype=torch.int32))

    with pytest.raises(ValueError) as e_info:
        # 5 batches of 3D images (5, B, C, D, H, W)
        _ = to_label_batch(torch.zeros((5, 3, 3, 10, 60, 60), dtype=torch.int32))


def test_normalized_inverse_distance_transform(tmpdir):

    #
    # 3D
    #

    # CDHW
    label_3d = imread(Path(__file__).parent / "data" / "labels.tif")

    # Add channel dimension
    label_3d_c = label_3d[np.newaxis, :]

    # Process
    data_3d = {"label": label_3d_c}
    ndt = NormalizedDistanceTransformd(keys=("label",), reverse=True, do_not_zero=True)
    data_3d_t = ndt(data_3d)

    # Test
    assert data_3d_t["label"].shape == label_3d_c.shape, "Unexpected output shape."
    assert (
        torch.max(data_3d_t["label"]) == 1.0
    ), "The tranform does not appear to have been normalized correctly."

    #
    # 2D
    #

    # CHW
    label_2d = label_3d[17]

    # Add channel dimension
    label_2d_c = label_2d[np.newaxis, :]

    # Process
    data_2d = {"label": label_2d_c}
    ndt = NormalizedDistanceTransformd(keys=("label",), reverse=True, do_not_zero=True)
    data_2d_t = ndt(data_2d)
    assert data_2d_t["label"].shape == label_2d_c.shape, "Unexpected output shape."

    # Test
    assert data_2d_t["label"].shape == label_2d_c.shape, "Unexpected output shape."
    assert (
        torch.max(data_2d_t["label"]) == 1.0
    ), "The tranform does not appear to have been normalized correctly."
