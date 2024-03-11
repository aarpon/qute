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
from typing import Optional

import numpy as np
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
    Spacing,
    Spacingd,
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
        """Return a composition of (dictionary) MapTransforms needed to train on a patch.

        These transforms are applied to the training dataset, to prepare the inputs to
        be fed into the model for the forward pass of training.
        """
        pass

    @abstractmethod
    def get_valid_transforms(self):
        """Return a composition of Transforms needed to validate on a patch.

        These transforms are applied to the validation dataset, to prepare the inputs to
        be fed into the model for validation.
        """
        pass

    @abstractmethod
    def get_test_transforms(self):
        """Return a composition of Transforms needed to test on a patch.

        These transforms are applied to the test dataset, to prepare the inputs to
        be fed into the model for testing.
        """
        pass

    @abstractmethod
    def get_val_metrics_transforms(self):
        """Define default transforms for validation metric calculation on a patch.

        These transforms are applied to the validation dataset after the images have
        gone through the forward pass, to prepare the output -- if needed -- for the
        validation metrics to be applied.
        """
        pass

    @abstractmethod
    def get_test_metrics_transforms(self):
        """Define default transforms for testing metric calculation on a patch.

        These transforms are applied to the test dataset after the images have
        gone through the forward pass, to prepare the output -- if needed -- for the
        test metrics to be applied.
        """
        pass

    @abstractmethod
    def get_inference_transforms(self):
        """Define inference transforms to predict on patch.

        These transforms are applied to the output of the full inference (that is the
        full images, not the patches) to prepare them to be saved to disk as final
        inference images.
        """
        pass

    @abstractmethod
    def get_post_inference_transforms(self):
        """Define post inference transforms to apply after prediction on patch.

        These transforms are applied to the images that will go through full inference.
        Please notice that the patch size will be the same as for training, a sliding
        windows approach will be used to predict the whole image.
        """
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
                RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(-2, -1)),
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


class SegmentationCampaignTransforms3D(CampaignTransforms):
    """Example 3D segmentation campaign transforms."""

    def __init__(
        self,
        num_classes: int = 3,
        patch_size: tuple = (20, 300, 300),
        num_patches: int = 1,
        pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
        to_isotropic: bool = False,
        upscale_z: bool = True,
    ):
        """Constructor.

        By default, these transforms apply to a single-channel input image to
        predict three output classes.

        PARAMETERS
        ----------

        num_classes: int = 3
            Number ouf output classes to predict.

        patch_size: tuple = (20, 300, 300)
            Patch size to pass through the neural network.

        num_patches: int = 1
            Number of patches per image to extract.

        pixdim: Optional[tuple] = (1.0, 1.0, 1.0)
            Voxel size (`pixdim`, in MONAI parlance) to use for setting the metadata of the image.
            Omit to set to (1.0, 1.0, 1.0).

        to_isotropic: bool (Optional, False)
            Se to True to resample the image to near-isotropic XYZ resolution.
            Ignored if all voxel sizes are the same.

        upscale_z: bool = True
            Only considered it `to_isotropic` is True.
            If True, interpolate z to reach the resolution of x and y. Please notice that it is assumed
            that the z resolution is worse than the x and y resolution.

            If False, sub-sample x and y to reach the resolution of z. Please notice that it is assumed
            that the z resolution is worse than the x and y resolution.
        """
        super().__init__()

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pixdim = np.array(pixdim)
        self.scaled_pixdim = self.pixdim
        self.to_isotropic = to_isotropic
        self.upscale_z = upscale_z

        if self.to_isotropic:
            # @TODO Implement completely as soon as issue [7529](https://github.com/Project-MONAI/MONAI/issues/7529)
            # @TODO is fixed or more information is collected.
            print("WARNING: Resampling currently disabled.")
            # Should we upscale the image to keep the higher resolution, or downscale it
            # to preserve the lowest resolution?
            if self.upscale_z:
                # x and y are left untouched; z is scaled up to
                # match (rounded) anisotropic resolution
                self.scaled_pixdim[0] = self.scaled_pixdim[1:2].mean()

            else:
                # z is left untouched; x and y are scaled down
                # to match (rounded) anisotropic resolution
                self.scaled_pixdim[1:] = self.scaled_pixdim[0]

    def get_train_transforms(self):
        """Return a composition of Transforms needed to train (patch)."""
        train_transforms = Compose(
            [
                CustomTIFFReaderd(
                    keys=["image", "label"],
                    ensure_channel_first=True,
                    dtype=torch.float32,
                    as_meta_tensor=True,
                    pixdim=self.pixdim,
                ),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=self.scaled_pixdim,
                #     mode=("bilinear", "nearest"),
                # ),
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
                    as_meta_tensor=True,
                    pixdim=self.pixdim,
                ),
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=self.scaled_pixdim,
                #     mode=("bilinear", "nearest"),
                # ),
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
                    as_meta_tensor=True,
                    pixdim=self.pixdim,
                ),
                # Spacing(
                #     pixdim=self.scaled_pixdim,
                #     mode="nearest",
                # ),
                ZNormalize(),
            ]
        )
        return inference_transforms

    def get_post_inference_transforms(self):
        """Define post inference transforms to apply after prediction on patch."""
        if self.to_isotropic:
            post_inference_transforms = Compose(
                [
                    ToLabel(),
                    # Spacing(pixdim=self.pixdim, mode='nearest')
                ]
            )
        else:
            post_inference_transforms = Compose(
                [
                    ToLabel(),
                ]
            )
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
