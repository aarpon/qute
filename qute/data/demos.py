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

import os
from pathlib import Path
from typing import Optional, Union

import userpaths

from qute.campaigns import CampaignTransforms
from qute.data.dataloaders import DataModuleLocalFolder
from qute.data.io import (
    get_cell_restoration_demo_dataset,
    get_cell_segmentation_demo_dataset,
    get_cell_segmentation_idt_demo_dataset,
)

__doc__ = "Demo dataloaders."
__all__ = [
    "CellSegmentationDemo",
    "CellSegmentationDemoIDT",
    "CellRestorationDemo",
]


class CellSegmentationDemo(DataModuleLocalFolder):
    """DataLoader for the Cell Segmentation Demo."""

    def __init__(
        self,
        campaign_transforms: CampaignTransforms,
        download_dir: Union[Path, str, None] = None,
        three_classes: bool = True,
        num_folds: int = 1,
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        batch_size: int = 8,
        inference_batch_size: int = 2,
        patch_size: tuple = (512, 512),
        num_patches: int = 1,
        source_images_sub_folder: str = "images",
        target_images_sub_folder: str = "labels",
        seed: int = 42,
        num_workers: Optional[int] = os.cpu_count() - 1,
        num_inference_workers: Optional[int] = os.cpu_count() - 1,
        pin_memory: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------

        campaign_transforms: CampaignTransforms
            Define all transforms necessary for training, validation, testing and (full) prediction.
            @see `qute.transforms.CampaignTransforms` for documentation.

        download_dir: Path | str = Path()
            Directory where the cell segmentation datasets will be downloaded and extracted.

        three_classes: bool = True
            Whether to download and extract the demo dataset with 3 labels (classes), or the one with two.

        num_folds: int = 1
            Set to a number larger than one to set up k-fold cross-validation. All images
            that do not belong to the test set fraction as defined by `test_fraction` will
            be rotated for k-fold cross-validations. In this regime, the training set will
            contain n * (k-1)/k images, while the validation set will contain n / k images
            (with k = num_folds, and n = number of images in the training + validation set).

        train_fraction: float = 0.7
            Fraction of images and corresponding labels that go into the training set.

        val_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.

        batch_size: int = 8
            Size of one batch of image pairs.

        inference_batch_size: int = 2
            Size of one batch of image pairs for full inference.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        num_patches: int = 1
            Number of patches per image to be extracted (and collated in the batch).

        source_images_sub_folder: str = "images"
            Name of the images sub-folder. It can be used to override the default "images".

        target_images_sub_folder: str = "labels"
            Name of the labels sub-folder. It can be used to override the default "labels".

        seed: int = 42
            Seed for all random number generators.

        num_workers: int
            Number of workers to be used in the training, validation and test data loaders.

        num_inference_workers: int
            Number of workers to be used in the inference data loader.

        pin_memory: bool = True
            Whether to pin the GPU memory.
        """

        # Check that download_dir is set
        if download_dir is None:
            download_dir = Path(userpaths.get_my_documents()) / "qute" / "data"

        # Download directory is parent of the actual data_dir that we pass to the parent class
        self.download_dir = Path(download_dir).resolve()

        # Make the number of classes explicit
        self.three_classes = three_classes
        if self.three_classes:
            self.num_classes = 3
        else:
            self.num_classes = 2
        data_dir = self.download_dir / f"demo_segmentation_{self.num_classes}_classes"

        # Call base constructor
        super().__init__(
            campaign_transforms=campaign_transforms,
            data_dir=data_dir,
            num_folds=num_folds,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            patch_size=patch_size,
            num_patches=num_patches,
            source_images_sub_folder=source_images_sub_folder,
            target_images_sub_folder=target_images_sub_folder,
            seed=seed,
            num_workers=num_workers,
            num_inference_workers=num_inference_workers,
            pin_memory=pin_memory,
        )

    def prepare_data(self):
        """Prepare the data on main thread."""

        # Download and extract the demo dataset if needed.
        get_cell_segmentation_demo_dataset(
            self.download_dir, three_classes=self.three_classes
        )

    def setup(self, stage):
        """Set up data on each GPU."""
        # Call parent setup()
        return super().setup(stage)


class CellSegmentationDemoIDT(DataModuleLocalFolder):
    """DataLoader for the Cell Segmentation Demo (Inverse Distance Transform)."""

    def __init__(
        self,
        campaign_transforms: CampaignTransforms,
        download_dir: Union[Path, str, None] = None,
        num_folds: int = 1,
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        batch_size: int = 8,
        inference_batch_size: int = 2,
        patch_size: tuple = (512, 512),
        num_patches: int = 1,
        source_images_sub_folder: str = "images",
        target_images_sub_folder: str = "labels",
        seed: int = 42,
        num_workers: Optional[int] = os.cpu_count() - 1,
        num_inference_workers: Optional[int] = os.cpu_count() - 1,
        pin_memory: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------

        campaign_transforms: CampaignTransforms
            Define all transforms necessary for training, validation, testing and (full) prediction.
            @see `qute.transforms.CampaignTransforms` for documentation.

        download_dir: Path | str = Path()
            Directory where the cell segmentation datasets will be downloaded and extracted.

        num_folds: int = 1
            Set to a number larger than one to set up k-fold cross-validation. All images
            that do not belong to the test set fraction as defined by `test_fraction` will
            be rotated for k-fold cross-validations. In this regime, the training set will
            contain n * (k-1)/k images, while the validation set will contain n / k images
            (with k = num_folds, and n = number of images in the training + validation set).

        train_fraction: float = 0.7
            Fraction of images and corresponding labels that go into the training set.

        val_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.

        batch_size: int = 8
            Size of one batch of image pairs.

        inference_batch_size: int = 2
            Size of one batch of image pairs for full inference.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        num_patches: int = 1
            Number of patches per image to be extracted (and collated in the batch).

        source_images_sub_folder: str = "images"
            Name of the images sub-folder. It can be used to override the default "images".

        target_images_sub_folder: str = "labels"
            Name of the labels sub-folder. It can be used to override the default "labels".

        seed: int = 42
            Seed for all random number generators.

        num_workers: int
            Number of workers to be used in the training, validation and test data loaders.

        num_inference_workers: int
            Number of workers to be used in the inference data loader.

        pin_memory: bool = True
            Whether to pin the GPU memory.
        """

        # Check that download_dir is set
        if download_dir is None:
            download_dir = Path(userpaths.get_my_documents()) / "qute" / "data"

        # Download directory is parent of the actual data_dir that we pass to the parent class
        self.download_dir = Path(download_dir).resolve()
        data_dir = self.download_dir / f"demo_segmentation_idt"

        # Call base constructor
        super().__init__(
            campaign_transforms=campaign_transforms,
            data_dir=data_dir,
            num_folds=num_folds,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            patch_size=patch_size,
            num_patches=num_patches,
            source_images_sub_folder=source_images_sub_folder,
            target_images_sub_folder=target_images_sub_folder,
            seed=seed,
            num_workers=num_workers,
            num_inference_workers=num_inference_workers,
            pin_memory=pin_memory,
        )

    def prepare_data(self):
        """Prepare the data on main thread."""

        # Download and extract the demo dataset if needed.
        get_cell_segmentation_idt_demo_dataset(self.download_dir)

    def setup(self, stage):
        """Set up data on each GPU."""
        # Call parent setup()
        return super().setup(stage)


class CellRestorationDemo(DataModuleLocalFolder):
    """DataLoader for the Cell Restoration Demo."""

    def __init__(
        self,
        campaign_transforms: CampaignTransforms,
        download_dir: Union[Path, str, None] = None,
        num_folds: int = 1,
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        batch_size: int = 8,
        inference_batch_size: int = 2,
        patch_size: tuple = (512, 512),
        num_patches: int = 1,
        source_images_sub_folder: str = "images",
        target_images_sub_folder: str = "targets",
        seed: int = 42,
        num_workers: Optional[int] = os.cpu_count() - 1,
        num_inference_workers: Optional[int] = os.cpu_count() - 1,
        pin_memory: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------

        campaign_transforms: CampaignTransforms
            Define all transforms necessary for training, validation, testing and (full) prediction.
            @see `qute.transforms.CampaignTransforms` for documentation.

        download_dir: Path | str = Path()
            Directory where the cell segmentation datasets will be downloaded and extracted.

        num_folds: int = 1
            Set to a number larger than one to set up k-fold cross-validation. All images
            that do not belong to the test set fraction as defined by `test_fraction` will
            be rotated for k-fold cross-validations. In this regime, the training set will
            contain n * (k-1)/k images, while the validation set will contain n / k images
            (with k = num_folds, and n = number of images in the training + validation set).

        train_fraction: float = 0.7
            Fraction of images and corresponding labels that go into the training set.

        val_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.

        batch_size: int = 8
            Size of one batch of image pairs.

        inference_batch_size: int = 2
            Size of one batch of image pairs for full inference.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        num_patches: int = 1
            Number of patches per image to be extracted (and collated in the batch).

        source_images_sub_folder: str = "images"
            Name of the images sub-folder. It can be used to override the default "images".

        target_images_sub_folder: str = "labels"
            Name of the labels sub-folder. It can be used to override the default "labels".

        seed: int = 42
            Seed for all random number generators.

        num_workers: int
            Number of workers to be used in the training, validation and test data loaders.

        num_inference_workers: int
            Number of workers to be used in the inference data loader.

        pin_memory: bool = True
            Whether to pin the GPU memory.
        """

        # Check that download_dir is set
        if download_dir is None:
            download_dir = Path(userpaths.get_my_documents()) / "qute" / "data"

        # Download directory is parent of the actual data_dir that we pass to the parent class
        self.download_dir = Path(download_dir).resolve()

        # Data directory
        data_dir = self.download_dir / f"demo_restoration"

        # Call base constructor
        super().__init__(
            campaign_transforms=campaign_transforms,
            data_dir=data_dir,
            num_folds=num_folds,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            patch_size=patch_size,
            num_patches=num_patches,
            source_images_sub_folder=source_images_sub_folder,
            target_images_sub_folder=target_images_sub_folder,
            seed=seed,
            num_workers=num_workers,
            num_inference_workers=num_inference_workers,
            pin_memory=pin_memory,
        )

    def prepare_data(self):
        """Prepare the data on main thread."""

        # Download and extract the demo dataset if needed.
        get_cell_restoration_demo_dataset(self.download_dir)

    def setup(self, stage):
        """Set up data on each GPU."""
        # Call parent setup()
        return super().setup(stage)
