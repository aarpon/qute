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

import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from qute.losses import CombinedMSEBinaryDiceCELoss
from qute.metrics import CombinedInvMeanAbsoluteErrorBinaryDiceMetric
from qute.transforms.io import CustomTIFFReader
from qute.transforms.objects import NormalizedDistanceTransformd


@pytest.fixture(autouse=False)
def extract_test_transforms_data(tmpdir):
    """Fixture to execute asserts before and after a test is run"""

    #
    # Setup
    #

    expected_files = [
        Path(__file__).parent / "data" / "cellpose.npy",
        Path(__file__).parent / "data" / "labels.tif",
        Path(__file__).parent / "data" / "labels_2d.tif",
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


def test_metrics_3d(extract_test_transforms_data):

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(
        dtype=torch.int32, as_meta_tensor=True, ensure_channel_first=True
    )
    label_image = reader(Path(__file__).parent / "data" / "labels.tif")
    num_labels_before = len(np.unique(label_image)) - 1
    assert num_labels_before == 68, "Unexpected number of labels in start image."

    # Create dictionary
    data = {
        "image": MetaTensor(torch.zeros(label_image.shape), dtype=torch.float32),
        "label": label_image,
    }

    #
    # WITHOUT BATCH DIMENSION
    #

    # Transform the image with the INVERSE distance transform
    i_ndt = NormalizedDistanceTransformd(
        keys=("label",),
        reverse=True,
        do_not_zero=True,
        add_seed_channel=True,
        seed_radius=1,
        with_batch_dim=False,
    )
    i_ndt_out = i_ndt(data)
    assert i_ndt_out["label"].shape == (2, 26, 300, 300), "Unexpected image shape."

    # Test CombinedInvMeanAbsoluteErrorBinaryDiceMetric
    metric = CombinedInvMeanAbsoluteErrorBinaryDiceMetric(
        alpha=0.5,
        max_mae_value=1.0,
        regression_channel=0,
        classification_channel=1,
        with_batch_dim=False,
    )

    metric_value = metric(i_ndt_out["label"], i_ndt_out["label"])
    assert metric_value == 1.0, "Unexpected loss value."

    #
    # WITH BATCH DIMENSION
    #

    # Add batch dimension
    batch_dim = 3
    label_image_batch = torch.zeros(
        size=[batch_dim] + list(label_image.shape), dtype=label_image.dtype
    )

    # Now create a fake batch
    for i in range(batch_dim):
        label_image_batch[i] = label_image

    # Create new dictionary
    data = {
        "image": MetaTensor(torch.zeros(label_image_batch.shape), dtype=torch.float32),
        "label": label_image_batch,
    }

    # Transform the image with the INVERSE distance transform
    i_ndt_batch = NormalizedDistanceTransformd(
        keys=("label",),
        reverse=True,
        do_not_zero=True,
        add_seed_channel=True,
        seed_radius=1,
        with_batch_dim=True,
    )
    i_ndt_batch_out = i_ndt_batch(data)
    assert i_ndt_batch_out["label"].shape == (
        batch_dim,
        2,
        26,
        300,
        300,
    ), "Unexpected image shape."

    # Test CombinedInvMeanAbsoluteErrorBinaryDiceMetric
    loss_batch = CombinedInvMeanAbsoluteErrorBinaryDiceMetric(
        alpha=0.5,
        max_mae_value=1.0,
        regression_channel=0,
        classification_channel=1,
        with_batch_dim=True,
    )
    loss_batch_value = loss_batch(i_ndt_batch_out["label"], i_ndt_batch_out["label"])
    assert loss_batch_value == 1.0, "Unexpected loss value."


def test_metrics_2d(extract_test_transforms_data):

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(
        dtype=torch.int32, as_meta_tensor=True, ensure_channel_first=True
    )
    label_image = reader(Path(__file__).parent / "data" / "labels_2d.tif")
    num_labels_before = len(np.unique(label_image)) - 1
    assert num_labels_before == 14, "Unexpected number of labels in start image."

    # Create dictionary
    data = {
        "image": MetaTensor(torch.zeros(label_image.shape), dtype=torch.float32),
        "label": label_image,
    }

    #
    # WITHOUT BATCH DIMENSION
    #

    # Transform the image with the INVERSE distance transform
    i_ndt = NormalizedDistanceTransformd(
        keys=("label",),
        reverse=True,
        do_not_zero=True,
        add_seed_channel=True,
        seed_radius=1,
        with_batch_dim=False,
    )
    i_ndt_out = i_ndt(data)
    assert i_ndt_out["label"].shape == (2, 100, 100), "Unexpected image shape."

    # Test CombinedInvMeanAbsoluteErrorBinaryDiceMetric
    metric = CombinedInvMeanAbsoluteErrorBinaryDiceMetric(
        alpha=0.5,
        max_mae_value=1.0,
        regression_channel=0,
        classification_channel=1,
        with_batch_dim=False,
    )

    metric_value = metric(i_ndt_out["label"], i_ndt_out["label"])
    assert metric_value == 1.0, "Unexpected loss value."

    #
    # WITH BATCH DIMENSION
    #

    # Add batch dimension
    batch_dim = 3
    label_image_batch = torch.zeros(
        size=[batch_dim] + list(label_image.shape), dtype=label_image.dtype
    )

    # Now create a fake batch
    for i in range(batch_dim):
        label_image_batch[i] = label_image

    # Create new dictionary
    data = {
        "image": MetaTensor(torch.zeros(label_image_batch.shape), dtype=torch.float32),
        "label": label_image_batch,
    }

    # Transform the image with the INVERSE distance transform
    i_ndt_batch = NormalizedDistanceTransformd(
        keys=("label",),
        reverse=True,
        do_not_zero=True,
        add_seed_channel=True,
        seed_radius=1,
        with_batch_dim=True,
    )
    i_ndt_batch_out = i_ndt_batch(data)
    assert i_ndt_batch_out["label"].shape == (
        batch_dim,
        2,
        100,
        100,
    ), "Unexpected image shape."

    # Test CombinedInvMeanAbsoluteErrorBinaryDiceMetric
    loss_batch = CombinedInvMeanAbsoluteErrorBinaryDiceMetric(
        alpha=0.5,
        max_mae_value=1.0,
        regression_channel=0,
        classification_channel=1,
        with_batch_dim=True,
    )
    loss_batch_value = loss_batch(i_ndt_batch_out["label"], i_ndt_batch_out["label"])
    assert loss_batch_value == 1.0, "Unexpected loss value."


def test_losses_3d(extract_test_transforms_data):

    # Load TIFF file with (dtype=torch.int32)
    reader = CustomTIFFReader(
        dtype=torch.int32, as_meta_tensor=True, ensure_channel_first=True
    )
    label_image = reader(Path(__file__).parent / "data" / "labels.tif")
    num_labels_before = len(np.unique(label_image)) - 1
    assert num_labels_before == 68, "Unexpected number of labels in start image."

    # Create dictionary
    data = {
        "image": MetaTensor(torch.zeros(label_image.shape), dtype=torch.float32),
        "label": label_image,
    }

    #
    # WITHOUT BATCH DIMENSION
    #

    # Transform the image with the INVERSE distance transform
    i_ndt = NormalizedDistanceTransformd(
        keys=("label",),
        reverse=True,
        do_not_zero=True,
        add_seed_channel=True,
        seed_radius=1,
        with_batch_dim=False,
    )
    i_ndt_out = i_ndt(data)
    assert i_ndt_out["label"].shape == (2, 26, 300, 300), "Unexpected image shape."

    # Test CombinedMSEBinaryDiceCELoss
    loss = CombinedMSEBinaryDiceCELoss(
        alpha=0.5,
        regression_channel=0,
        classification_channel=1,
        include_background=True,
        with_batch_dim=False,
    )

    loss_value = loss(i_ndt_out["label"], i_ndt_out["label"])
    assert loss_value == 0.0, "Unexpected loss value."

    #
    # WITH BATCH DIMENSION
    #

    # Add batch dimension
    batch_dim = 3
    label_image_batch = torch.zeros(
        size=[batch_dim] + list(label_image.shape), dtype=label_image.dtype
    )

    # Now create a fake batch
    for i in range(batch_dim):
        label_image_batch[i] = label_image

    # Create new dictionary
    data = {
        "image": MetaTensor(torch.zeros(label_image_batch.shape), dtype=torch.float32),
        "label": label_image_batch,
    }

    # Transform the image with the INVERSE distance transform
    i_ndt_batch = NormalizedDistanceTransformd(
        keys=("label",),
        reverse=True,
        do_not_zero=True,
        add_seed_channel=True,
        seed_radius=1,
        with_batch_dim=True,
    )
    i_ndt_batch_out = i_ndt_batch(data)
    assert i_ndt_batch_out["label"].shape == (
        batch_dim,
        2,
        26,
        300,
        300,
    ), "Unexpected image shape."

    # Test CombinedInvMeanAbsoluteErrorBinaryDiceMetric
    loss_batch = CombinedInvMeanAbsoluteErrorBinaryDiceMetric(
        alpha=0.5,
        max_mae_value=1.0,
        regression_channel=0,
        classification_channel=1,
        with_batch_dim=True,
    )
    loss_batch_value = loss_batch(i_ndt_batch_out["label"], i_ndt_batch_out["label"])
    assert loss_batch_value == 1.0, "Unexpected loss value."


# class CustomDiceCELoss(nn.Module):
#     """Dice and Xentropy loss"""
#
#     def __init__(self, to_onehot_y=True, softmax=True):
#         super().__init__()
#         self.dice = DiceLoss(to_onehot_y=to_onehot_y, softmax=softmax)
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#     def forward(self, y_pred, y_true):
#         dice = self.dice(y_pred, y_true)
#         # CrossEntropyLoss target needs to have shape (B, D, H, W)
#         # Target from pipeline has shape (B, 1, D, H, W)
#         #cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
#         cross_entropy = self.cross_entropy(y_pred, y_true)
#         return dice + cross_entropy
#
#
# def test_dice_ce_loss():
#     # Create an example tensor with shape [B, C, H, W]
#     output = torch.tensor([[[[0, 0.1, 0.9], [0.8, 0, 0.8], [0.9, 0.8, 0.2]]]], dtype=torch.float32)
#     target = torch.tensor([[[[0, 0.1, 0.9], [0.8, 0, 0.8], [0.9, 0.8, 0.2]]]], dtype=torch.float32)
#
#     output = output.unsqueeze(1)
#     target = target.unsqueeze(1)
#
#     # One-hot encode the target tensor
#     #output = torch.nn.functional.one_hot(output, num_classes=2).float()
#     # target = torch.nn.functional.one_hot(target, num_classes=2).float()
#
#     print(output.shape)
#     print(target.shape)
#
#     # Initialize DiceCELoss
#     dice_ce_loss_fn = CustomDiceCELoss(to_onehot_y=True, softmax=True)
#
#     # Compute the DiceCE loss
#     loss_value = dice_ce_loss_fn(output, target)
#
#     print(f"DiceCE Loss: {loss_value.item()}")
