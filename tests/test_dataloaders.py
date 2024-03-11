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

import pytest

from qute.campaigns import SegmentationCampaignTransforms
from qute.data.demos import CellSegmentationDemo


def test_k_folds():

    # Initialize default, example Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms()

    # Initialize data module
    data_module = CellSegmentationDemo(campaign_transforms=campaign_transforms)

    # Datasets are not yet defined
    assert data_module.train_dataset is None, "Training data is already defined!"
    assert data_module.val_dataset is None, "Validation data is already defined!"
    assert data_module.test_dataset is None, "Test data is already defined!"

    # Initialize data module with k-fold cross-validation
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms, num_folds=5
    )

    # Datasets are not yet defined
    assert data_module.train_dataset is None, "Training data is already defined!"
    assert data_module.val_dataset is None, "Validation data is already defined!"
    assert data_module.test_dataset is None, "Test data is already defined!"


def test_setup():

    # Initialize default, example Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms()

    # Initialize data module (no k-fold cross-validation)
    data_module = CellSegmentationDemo(campaign_transforms=campaign_transforms)

    # Run the prepare/setup steps
    data_module.prepare_data()
    data_module.setup(stage="train")

    # Datasets must be defined
    assert data_module.train_dataset is not None, "Training data was not defined!"
    assert data_module.val_dataset is not None, "Validation data was not defined!"
    assert data_module.test_dataset is not None, "Test data was not defined!"

    # Test the expected number of training, validation and test images in the datasets
    assert (
        len(data_module.train_dataset) == 63
    ), "Unexpected number of training images/labels pairs."
    assert (
        len(data_module.val_dataset) == 18
    ), "Unexpected number of validation images/labels pairs."
    assert (
        len(data_module.test_dataset) == 9
    ), "Unexpected number of test images/labels pairs."

    # Check the training dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.train_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 63, "Unexpected number of images returned."
    assert n_labels == 63, "Unexpected number of labels returned."

    # Check the validation dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.val_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 18, "Unexpected number of images returned."
    assert n_labels == 18, "Unexpected number of labels returned."

    # Check the test dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.test_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 9, "Unexpected number of images returned."
    assert n_labels == 9, "Unexpected number of labels returned."

    # Initialize data module with k-fold cross-validation (10 folds)
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms, num_folds=10
    )

    # Run the prepare/setup steps
    data_module.prepare_data()
    data_module.setup(stage="train")

    # Datasets must be defined
    assert (
        len(data_module.train_dataset) == 72
    ), "Unexpected number of training images/labels pairs."
    assert (
        len(data_module.val_dataset) == 9
    ), "Unexpected number of validation images/labels pairs."
    assert (
        len(data_module.test_dataset) == 9
    ), "Unexpected number of test images/labels pairs."

    # Initialize data module with k-fold cross-validation (5 folds)
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms, num_folds=5
    )

    # Run the prepare/setup steps
    data_module.prepare_data()
    data_module.setup(stage="train")

    # Datasets must be defined
    assert (
        len(data_module.train_dataset) == 64
    ), "Unexpected number of training images/labels pairs."
    assert (
        len(data_module.val_dataset) == 17
    ), "Unexpected number of validation images/labels pairs."
    assert (
        len(data_module.test_dataset) == 9
    ), "Unexpected number of test images/labels pairs."

    # Check the training dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.train_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 64, "Unexpected number of images returned."
    assert n_labels == 64, "Unexpected number of labels returned."

    # Check the validation dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.val_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 17, "Unexpected number of images returned."
    assert n_labels == 17, "Unexpected number of labels returned."

    # Check the test dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.test_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 9, "Unexpected number of images returned."
    assert n_labels == 9, "Unexpected number of labels returned."

    # Change to fold number 1: please notice that since the training + validation set contains
    # 81 values (that are not divisible by 5 without a remainder), the first fold will split
    # into 64 + 17, while the others will split into 65 + 16.
    data_module.set_fold(1)
    assert data_module.current_fold == 1, "Unexpected value for current fold."

    # Check dataset sizes
    assert (
        len(data_module.train_dataset) == 65
    ), "Unexpected number of training images/labels pairs."
    assert (
        len(data_module.val_dataset) == 16
    ), "Unexpected number of validation images/labels pairs."
    assert (
        len(data_module.test_dataset) == 9
    ), "Unexpected number of test images/labels pairs."

    # Check the training dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.train_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 65, "Unexpected number of images returned."
    assert n_labels == 65, "Unexpected number of labels returned."

    # Check the validation dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.val_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 16, "Unexpected number of images returned."
    assert n_labels == 16, "Unexpected number of labels returned."

    # Check the test dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.test_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 9, "Unexpected number of images returned."
    assert n_labels == 9, "Unexpected number of labels returned."

    # Change to fold number 2
    data_module.set_fold(2)
    assert data_module.current_fold == 2, "Unexpected value for current fold."

    # Check dataset sizes
    assert (
        len(data_module.train_dataset) == 65
    ), "Unexpected number of training images/labels pairs."
    assert (
        len(data_module.val_dataset) == 16
    ), "Unexpected number of validation images/labels pairs."
    assert (
        len(data_module.test_dataset) == 9
    ), "Unexpected number of test images/labels pairs."

    # Check the training dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.train_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 65, "Unexpected number of images returned."
    assert n_labels == 65, "Unexpected number of labels returned."

    # Check the validation dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.val_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 16, "Unexpected number of images returned."
    assert n_labels == 16, "Unexpected number of labels returned."

    # Check the test dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.test_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 9, "Unexpected number of images returned."
    assert n_labels == 9, "Unexpected number of labels returned."

    # Change to fold number 3
    data_module.set_fold(3)
    assert data_module.current_fold == 3, "Unexpected value for current fold."

    # Check dataset sizes
    assert (
        len(data_module.train_dataset) == 65
    ), "Unexpected number of training images/labels pairs."
    assert (
        len(data_module.val_dataset) == 16
    ), "Unexpected number of validation images/labels pairs."
    assert (
        len(data_module.test_dataset) == 9
    ), "Unexpected number of test images/labels pairs."

    # Check the training dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.train_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 65, "Unexpected number of images returned."
    assert n_labels == 65, "Unexpected number of labels returned."

    # Check the validation dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.val_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 16, "Unexpected number of images returned."
    assert n_labels == 16, "Unexpected number of labels returned."

    # Check the test dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.test_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 9, "Unexpected number of images returned."
    assert n_labels == 9, "Unexpected number of labels returned."

    # Change to fold number 4
    data_module.set_fold(4)
    assert data_module.current_fold == 4, "Unexpected value for current fold."

    # Check dataset sizes
    assert (
        len(data_module.train_dataset) == 65
    ), "Unexpected number of training images/labels pairs."
    assert (
        len(data_module.val_dataset) == 16
    ), "Unexpected number of validation images/labels pairs."
    assert (
        len(data_module.test_dataset) == 9
    ), "Unexpected number of test images/labels pairs."

    # Check the training dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.train_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 65, "Unexpected number of images returned."
    assert n_labels == 65, "Unexpected number of labels returned."

    # Check the validation dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.val_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 16, "Unexpected number of images returned."
    assert n_labels == 16, "Unexpected number of labels returned."

    # Check the test dataloader
    n_images = 0
    n_labels = 0
    for image, label in data_module.test_dataloader():
        assert len(image.shape) == 4, "Wrong dimensionality of `image`."
        assert len(label.shape) == 4, "Wrong dimensionality of `label`."
        assert image.shape[1] == 1, "Image must have one channel."
        assert label.shape[1] == 3, "Image must have three channels."
        n_images += image.shape[0]
        n_labels += label.shape[0]

    assert n_images == 9, "Unexpected number of images returned."
    assert n_labels == 9, "Unexpected number of labels returned."

    # Check that 5 is not an acceptable fold number
    with pytest.raises(ValueError):
        data_module.set_fold(5)
