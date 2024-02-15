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
from abc import ABC, abstractmethod

import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    AsDiscreted,
    Compose,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandSpatialCropd,
)

from qute.transforms import (
    CustomTIFFReader,
    CustomTIFFReaderd,
    MinMaxNormalize,
    MinMaxNormalized,
    Scale,
    Scaled,
    ToLabel,
    ToPyTorchLightningOutputd,
    ZNormalize,
    ZNormalized,
)


class CampaignTransforms(ABC):
    """Abstract base class that defines all transforms needed for a full training campaign."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_train_transforms(self):
        """Return a composition of (dictionary) MapTransforms needed to train on a patch."""
        pass

    @abstractmethod
    def get_valid_transforms(self):
        """Return a composition of Transforms needed to validate on a patch."""
        pass

    @abstractmethod
    def get_test_transforms(self):
        """Return a composition of Transforms needed to test on a patch."""
        pass

    @abstractmethod
    def get_val_metrics_transforms(self):
        """Define default transforms for validation metric calculation on a patch."""
        pass

    @abstractmethod
    def get_test_metrics_transforms(self):
        """Define default transforms for testing metric calculation on a patch."""
        pass

    @abstractmethod
    def get_inference_transforms(self):
        """Define inference transforms to predict on patch."""
        pass

    @abstractmethod
    def get_post_inference_transforms(self):
        """Define post inference transforms to apply after prediction on patch."""
        pass


class SegmentationCampaignTransforms(CampaignTransforms):
    """Example segmentation campaign transforms."""

    def __init__(
        self, num_classes: int = 3, patch_size: tuple = (640, 640), num_patches: int = 1
    ):
        """Constructor.

        By default, these transforms apply to a single-channel input image to
        predict three output classes.
        """
        super().__init__()

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches

    def get_train_transforms(self):
        """Return a composition of Transforms needed to train (patch)."""
        train_transforms = Compose(
            [
                CustomTIFFReaderd(
                    keys=["image", "label"],
                    ensure_channel_first=True,
                    dtype=torch.float32,
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=1.0,
                    neg=1.0,
                    num_samples=self.num_patches,
                    image_key="image",
                    image_threshold=0.0,
                    allow_smaller=False,
                    lazy=False,
                ),
                ZNormalized(keys=["image"]),
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
                RandGaussianNoised(keys="image", prob=0.2),
                RandGaussianSmoothd(keys="image", prob=0.2),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
                ToPyTorchLightningOutputd(),
            ]
        )
        return train_transforms

    def get_valid_transforms(self):
        """Return a composition of Transforms needed to validate (patch)."""
        val_transforms = Compose(
            [
                CustomTIFFReaderd(
                    keys=["image", "label"],
                    ensure_channel_first=True,
                    dtype=torch.float32,
                ),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=self.patch_size,
                    pos=1.0,
                    neg=1.0,
                    num_samples=self.num_patches,
                    image_key="image",
                    image_threshold=0.0,
                    allow_smaller=False,
                    lazy=False,
                ),
                ZNormalized(keys=["image"]),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
                ToPyTorchLightningOutputd(),
            ]
        )
        return val_transforms

    def get_test_transforms(self):
        """Return a composition of Transforms needed to test (patch)."""
        return self.get_valid_transforms()

    def get_inference_transforms(self):
        """Define inference transforms to predict (patch)."""
        inference_transforms = Compose(
            [
                CustomTIFFReader(
                    ensure_channel_first=True,
                    dtype=torch.float32,
                ),
                ZNormalize(),
            ]
        )
        return inference_transforms

    def get_post_inference_transforms(self):
        """Define post inference transforms to apply after prediction on patch."""
        post_inference_transforms = Compose([ToLabel()])
        return post_inference_transforms

    def get_val_metrics_transforms(self):
        """Define default transforms for validation metric calculation (patch)."""
        val_metrics_transforms = Compose(
            [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
        )
        return val_metrics_transforms

    def get_test_metrics_transforms(self):
        """Define default transforms for testing metric calculation (patch)."""
        return self.get_val_metrics_transforms()


class RestorationCampaignTransforms(CampaignTransforms):
    """Example restoration campaign transforms."""

    def __init__(
        self,
        min_intensity: int = 0,
        max_intensity: int = 65535,
        patch_size: tuple = (640, 640),
        num_patches: int = 1,
    ):
        """Constructor.

        By default, these transforms apply to a single-channel input image to
        predict a single-channel output.
        """
        super().__init__()

        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.patch_size = patch_size
        self.num_patches = num_patches

    def get_train_transforms(self):
        """Return a composition of Transforms needed to train (patch)."""
        train_transforms = Compose(
            [
                CustomTIFFReaderd(
                    keys=["image", "label"],
                    ensure_channel_first=True,
                    dtype=torch.float32,
                ),
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=self.patch_size,
                    random_size=False,
                ),
                MinMaxNormalized(
                    keys=["image", "label"], min_intensity=0.0, max_intensity=15472.0
                ),
                # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
                # RandGaussianNoised(keys="image", prob=0.2),
                # RandGaussianSmoothd(keys="image", prob=0.2),
                ToPyTorchLightningOutputd(
                    image_dtype=torch.float32,
                    label_dtype=torch.float32,
                ),
            ]
        )
        return train_transforms

    def get_valid_transforms(self):
        """Return a composition of Transforms needed to validate (patch)."""
        val_transforms = Compose(
            [
                CustomTIFFReaderd(
                    keys=["image", "label"],
                    ensure_channel_first=True,
                    dtype=torch.float32,
                ),
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=self.patch_size,
                    random_size=False,
                ),
                MinMaxNormalized(
                    keys=["image", "label"], min_intensity=0.0, max_intensity=15472.0
                ),
                ToPyTorchLightningOutputd(
                    image_dtype=torch.float32,
                    label_dtype=torch.float32,
                ),
            ]
        )
        return val_transforms

    def get_test_transforms(self):
        """Return a composition of Transforms needed to test (patch)."""
        return self.get_valid_transforms()

    def get_inference_transforms(self):
        """Define inference transforms to predict (patch)."""
        inference_transforms = Compose(
            [
                CustomTIFFReader(
                    ensure_channel_first=True,
                    dtype=torch.float32,
                ),
                MinMaxNormalize(
                    min_intensity=self.min_intensity, max_intensity=self.max_intensity
                ),
            ]
        )
        return inference_transforms

    def get_post_inference_transforms(self):
        """Define post inference transforms to apply after prediction on patch."""
        post_inference_transforms = Compose(
            [
                Scale(
                    factor=self.max_intensity,
                    dtype=torch.int32,
                )
            ]
        )
        return post_inference_transforms

    def get_val_metrics_transforms(self):
        """Define default transforms for validation metric calculation (patch)."""
        val_metrics_transforms = Compose([])
        return val_metrics_transforms

    def get_test_metrics_transforms(self):
        """Define default transforms for testing metric calculation (patch)."""
        return self.get_val_metrics_transforms()
