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
from skimage.measure import label

from qute.transforms.io import CellposeLabelReader, CustomTIFFReader
from qute.transforms.objects import (
    LabelToTwoClassMask,
    LabelToTwoClassMaskd,
    NormalizedDistanceTransform,
    NormalizedDistanceTransformd,
    OneHotToMask,
    OneHotToMaskBatch,
    TwoClassMaskToLabel,
    TwoClassMaskToLabeld,
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


def test_label_to_two_class_mask(extract_test_transforms_data):

    # Load CellPose dataset (two objects)
    reader = CellposeLabelReader()
    label = reader(Path(__file__).parent / "data" / "cellpose.npy")
    assert label.shape == (300, 300), "Unexpected image size."

    # Test passing a 2D array - it must fail.
    with pytest.raises(ValueError):
        _ = LabelToTwoClassMask()(label)

    # Add the channel dimension
    label = label[np.newaxis, :, :]

    # Pass the augmented array
    mask = LabelToTwoClassMask()(label)

    # Check that there are indeed three classes
    assert mask.shape == (1, 300, 300), "Unexpected output shape."
    assert len(torch.unique(mask)) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(mask) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."

    #
    # Test BW input
    #

    # Binarize the image
    label[label > 0] = 1

    # Add to data dictionary
    data = {"label": label}

    # Pass the augmented array
    data = LabelToTwoClassMaskd(keys=("label",))(data)

    # Check that there are indeed three classes with the expected number of pixels
    assert data["label"].shape == (1, 300, 300), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."

    # Check that the result of this run is the same as the result of the previous
    assert torch.all(mask == data["label"]), "Unexpected result of the bw input."

    # Load 3D labels dataset (many objects)
    reader = CustomTIFFReader(dtype=torch.int32)
    label = reader(Path(__file__).parent / "data" / "labels.tif")
    assert label.shape == (1, 26, 300, 300), "Unexpected image size."

    data["label"] = label
    data = LabelToTwoClassMaskd(keys=("label",), border_thickness=1)(data)

    # Check that there are indeed three classes with the expected number of pixels
    assert data["label"].shape == (1, 26, 300, 300), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."
    assert (
        torch.sum(data["label"] == 1) == 52496
    ), "Unexpected nuber of pixels with class 1."
    assert (
        torch.sum(data["label"] == 2) == 53251
    ), "Twenty-six pixels with class 2 expected."

    # Same as before but with BW mask (notice that the re-labeling will create
    # a slightly different set of starting objects)
    reader = CustomTIFFReader(dtype=torch.int32)
    label = reader(Path(__file__).parent / "data" / "labels.tif")
    assert label.shape == (1, 26, 300, 300), "Unexpected image size."

    data["label"] = label > 0
    data = LabelToTwoClassMaskd(keys=("label",), border_thickness=1)(data)

    # Check that there are indeed three classes with the expected number of pixels
    assert data["label"].shape == (1, 26, 300, 300), "Unexpected output shape."
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
    assert torch.all(
        torch.unique(data["label"]) == torch.tensor((0, 1, 2))
    ), "Expected classes [0, 1, 2]."
    assert (
        torch.sum(data["label"] == 1) == 55335
    ), "Unexpected nuber of pixels with class 1."
    assert (
        torch.sum(data["label"] == 2) == 50412
    ), "Twenty-six pixels with class 2 expected."

    # Make sure that 3D with more than one channel is not supported
    input_3d_2c = torch.zeros((2, 10, 10, 10), dtype=torch.int32)  # 2 channels
    input_3d_2c[0, 4:7, 4:7, 4:7] = 1
    data["label"] = input_3d_2c
    tr = LabelToTwoClassMaskd(keys=("label",), border_thickness=1)
    with pytest.raises(Exception):
        _ = tr(data)


def test_two_class_mask_to_labels_2d(extract_test_transforms_data):

    #
    # 2D
    #

    # Load 2D labels image
    reader = CustomTIFFReader(dtype=torch.int32)
    labels = reader(Path(__file__).parent / "data" / "labels_2d.tif")
    assert labels.shape == (1, 100, 100), "Unexpected image size."
    assert len(np.unique(labels)) - 1 == 14, "Unexpected number of labels."

    # Test LabelToTwoClassMask

    # Create a 2-class mask
    two_class = LabelToTwoClassMask(border_thickness=1, drop_eroded=False)(labels)
    assert two_class.shape == (1, 100, 100), "Unexpected result size."
    assert len(two_class.unique()) == 3, "Unexpected number of labels."

    # Create a 2-class mask (dropping eroded objects)
    two_class_eroded = LabelToTwoClassMask(border_thickness=1, drop_eroded=True)(labels)
    assert two_class_eroded.shape == (1, 100, 100), "Unexpected result size."
    assert len(two_class_eroded.unique()) == 3, "Unexpected number of labels."

    # Compare
    # The mask `two_class` and the mask `two_mask_eroded` are expected to have the same number of pixels
    # for both class 1 and 2, since no objects are eroded away.
    assert torch.all(
        (two_class == 1) == (two_class_eroded == 1)
    ), "Number of pixels with class 1 should match!"
    assert torch.sum(two_class == 2) == torch.sum(
        two_class_eroded == 2
    ), "Number of pixels with class 1 should match!"
    assert (
        torch.sum(two_class == 1) == 1199
    ), "Unexpected number of pixels with class 1!"
    assert torch.sum(two_class == 2) == 500, "Unexpected number of pixels with class 2!"

    # Count objects - we should have the same number as in the initial labels image
    two_class_mask = two_class == 1
    count_labels, num = label(two_class_mask.squeeze(), background=0, return_num=True)
    assert num == 14, "Unexpected number of labels."
    assert (
        len(np.unique(count_labels)) - 1 == len(np.unique(labels)) - 1
    ), "Unexpected number of labels."

    # Now go back
    reconstructed_label = TwoClassMaskToLabel(border_thickness=1)(two_class)
    assert (
        len(np.unique(reconstructed_label)) - 1 == 14
    ), "Unexpected number of reconstructed labels."

    # Test LabelToTwoClassMaskd

    data = {"label": labels}

    # Create a 2-class mask
    data_two_class = LabelToTwoClassMaskd(
        keys=("label",), border_thickness=1, drop_eroded=False
    )(data)
    assert data_two_class["label"].shape == (1, 100, 100), "Unexpected result size."
    assert len(data_two_class["label"].unique()) == 3, "Unexpected number of labels."

    # Create a 2-class mask (dropping eroded objects)
    data_two_class_eroded = LabelToTwoClassMaskd(
        keys=("label",), border_thickness=1, drop_eroded=True
    )(data)
    assert data_two_class_eroded["label"].shape == (
        1,
        100,
        100,
    ), "Unexpected result size."
    assert (
        len(data_two_class_eroded["label"].unique()) == 3
    ), "Unexpected number of labels."

    # Compare
    # The mask `data_two_class` and the mask `data_two_class_eroded` are expected to have the same number of pixels
    # for both class 1 and 2, since no objects are eroded away.
    assert torch.all(
        (data_two_class["label"] == 1) == (data_two_class_eroded["label"] == 1)
    ), "Number of pixels with class 1 should match!"
    assert torch.sum(data_two_class["label"] == 2) == torch.sum(
        data_two_class_eroded["label"] == 2
    ), "Number of pixels with class 2 should match!"
    assert (
        torch.sum(data_two_class["label"] == 1) == 1199
    ), "Unexpected number of pixels with class 2 in `two_class`!"
    assert (
        torch.sum(data_two_class_eroded["label"] == 2) == 500
    ), "Unexpected number of pixels with class 2 in `two_class_eroded`!"

    # Count objects - we should have the same number as in the initial labels image
    two_class_mask = data_two_class["label"] == 1
    count_labels, num = label(two_class_mask.squeeze(), background=0, return_num=True)
    assert num == 14, "Unexpected number of labels."
    assert len(np.unique(count_labels)) - 1 == 14, "Unexpected number of labels."

    # Now go back
    data_reconstructed_label = TwoClassMaskToLabeld(
        keys=("label",), border_thickness=1
    )(data_two_class)
    assert (
        len(np.unique(data_reconstructed_label["label"])) - 1 == 14
    ), "Unexpected number of reconstructed labels."


def test_two_class_mask_to_labels_3d(extract_test_transforms_data):

    #
    # 3D
    #

    # Load 3D labels image
    reader = CustomTIFFReader(dtype=torch.int32)
    labels = reader(Path(__file__).parent / "data" / "labels.tif")
    assert labels.shape == (1, 26, 300, 300), "Unexpected image size."
    assert len(np.unique(labels)) - 1 == 68, "Unexpected number of labels."

    # Test LabelToTwoClassMask

    # Create a 2-class mask
    two_class = LabelToTwoClassMask(border_thickness=1, drop_eroded=False)(labels)
    assert two_class.shape == (1, 26, 300, 300), "Unexpected result size."
    assert len(two_class.unique()) == 3, "Unexpected number of labels."

    # Create a 2-class mask (dropping eroded objects)
    two_class_eroded = LabelToTwoClassMask(border_thickness=1, drop_eroded=True)(labels)
    assert two_class_eroded.shape == (1, 26, 300, 300), "Unexpected result size."
    assert len(two_class_eroded.unique()) == 3, "Unexpected number of labels."

    # Compare
    # The mask `two_class` will have some border pixels coming from class 2 that will not be
    # present in `two_mask_eroded` since their object was dropped.
    assert torch.all(
        (two_class == 1) == (two_class_eroded == 1)
    ), "Unexpected reconstruction!"
    assert torch.sum(two_class == 2) > torch.sum(
        two_class_eroded == 2
    ), "Two many pixels with class 2 in `two_class_eroded`!"
    assert (
        torch.sum(two_class == 2) == 53251
    ), "Unexpected number of pixels with class 2 in `two_class`!"
    assert (
        torch.sum(two_class_eroded == 2) == 49441
    ), "Unexpected number of pixels with class 2 in `two_class_eroded`!"

    # Count objects - 18 objects are eroded away because they are too flat.
    two_class_mask = two_class == 1
    count_labels, num = label(two_class_mask.squeeze(), background=0, return_num=True)
    assert num == 50, "Unexpected number of labels."
    assert len(np.unique(count_labels)) - 1 == 50, "Unexpected number of labels."

    # Now go back
    reconstructed_label = TwoClassMaskToLabel(border_thickness=1)(two_class)
    assert (
        len(np.unique(reconstructed_label)) - 1 == 50
    ), "Unexpected number of reconstructed labels."

    # Test LabelToTwoClassMaskd

    data = {"label": labels}

    # Create a 2-class mask
    data_two_class = LabelToTwoClassMaskd(
        keys=("label",), border_thickness=1, drop_eroded=False
    )(data)
    assert data_two_class["label"].shape == (1, 26, 300, 300), "Unexpected result size."
    assert len(data_two_class["label"].unique()) == 3, "Unexpected number of labels."

    # Create a 2-class mask (dropping eroded objects)
    data_two_class_eroded = LabelToTwoClassMaskd(
        keys=("label",), border_thickness=1, drop_eroded=True
    )(data)
    assert data_two_class_eroded["label"].shape == (
        1,
        26,
        300,
        300,
    ), "Unexpected result size."
    assert (
        len(data_two_class_eroded["label"].unique()) == 3
    ), "Unexpected number of labels."

    # Compare
    # The mask `two_class` will have some border pixels coming from class 2 that will not be
    # present in `two_mask_eroded` since their object was dropped.
    assert torch.all(
        (data_two_class["label"] == 1) == (data_two_class_eroded["label"] == 1)
    ), "Unexpected reconstruction!"
    assert torch.sum(data_two_class["label"] == 2) > torch.sum(
        data_two_class_eroded["label"] == 2
    ), "Two many pixels with class 2 in `two_class_eroded`!"
    assert (
        torch.sum(data_two_class["label"] == 2) == 53251
    ), "Unexpected number of pixels with class 2 in `two_class`!"
    assert (
        torch.sum(data_two_class_eroded["label"] == 2) == 49441
    ), "Unexpected number of pixels with class 2 in `two_class_eroded`!"

    # Count objects - 18 objects are eroded away because they are too flat.
    two_class_mask = data_two_class["label"] == 1
    count_labels, num = label(two_class_mask.squeeze(), background=0, return_num=True)
    assert num == 50, "Unexpected number of labels."
    assert len(np.unique(count_labels)) - 1 == 50, "Unexpected number of labels."

    # Now go back
    data_reconstructed_label = TwoClassMaskToLabeld(
        keys=("label",), border_thickness=1
    )(data_two_class)
    assert (
        len(np.unique(data_reconstructed_label["label"])) - 1 == 50
    ), "Unexpected number of reconstructed labels."


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


def test_normalized_distance_transform(extract_test_transforms_data):

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
    assert 0.08944272249937057 == pytest.approx(
        torch.min(data_out["label"][data_out["label"] > 0]).item()
    ), "Unexpected minimum pixel value."
    assert torch.max(data_out["label"]) == 1.0, "Unexpected maximum pixel value."

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
    assert 0.08944272249937057 == pytest.approx(
        torch.min(data_out["label"][data_out["label"] > 0]).item()
    ), "Unexpected minimum pixel value."
    assert torch.max(label_out) == 1.0, "Unexpected maximum pixel value."

    # Compare with the previous result
    assert torch.all(data_out["label"] == label_out), "Unexpected result."


def test_normalized_distance_transform_with_seeds(extract_test_transforms_data):

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
    ndt = NormalizedDistanceTransformd(
        keys=("label",),
        reverse=True,
        do_not_zero=True,
        add_seed_channel=True,
        seed_radius=1,
    )
    data_out = ndt(data)

    assert data_out["image"].shape == (1, 26, 300, 300), "Unexpected image shape."
    assert data_out["label"].shape == (2, 26, 300, 300), "Unexpected labels shape."
    assert torch.all(data_out["image"] == 0.0)
    assert 0.08944272249937057 == pytest.approx(
        torch.min(data_out["label"][data_out["label"] > 0]).item()
    ), "Unexpected minimum pixel value."
    assert torch.max(data_out["label"][0]) == 1.0, "Unexpected maximum pixel value."
    assert torch.max(data_out["label"][1]) == 1.0, "Unexpected maximum pixel value."

    #
    # NormalizedDistanceTransform
    #

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(dtype=torch.int32, as_meta_tensor=True)
    label = reader(Path(__file__).parent / "data" / "labels.tif")

    # Transform the image
    ndt = NormalizedDistanceTransform(
        reverse=True, do_not_zero=True, add_seed_channel=True, seed_radius=1
    )
    label_out = ndt(label)

    assert label_out.shape == (2, 26, 300, 300), "Unexpected image shape."
    assert 0.08944272249937057 == pytest.approx(
        torch.min(data_out["label"][data_out["label"] > 0]).item()
    ), "Unexpected minimum pixel value."
    assert torch.max(data_out["label"][0]) == 1.0, "Unexpected maximum pixel value."
    assert torch.max(data_out["label"][1]) == 1.0, "Unexpected maximum pixel value."

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

    with pytest.raises(ValueError):
        # Single 2D image (H, W)
        _ = to_label(torch.zeros((60, 60), dtype=torch.int32))

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
        # Single 2D image (H, W)
        _ = to_label_batch(torch.zeros((60, 60), dtype=torch.int32))

    with pytest.raises(ValueError):
        # 5 batches of 3D images (5, B, C, D, H, W)
        _ = to_label_batch(torch.zeros((5, 3, 3, 10, 60, 60), dtype=torch.int32))
