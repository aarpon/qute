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

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
from monai.data import ArrayDataset, DataLoader, Dataset, list_data_collate
from natsort import natsorted
from numpy.random import default_rng
from sklearn.model_selection import KFold

from qute.campaigns import CampaignTransforms

__doc__ = "Dataloaders."
__all__ = [
    "DataModuleLocalFolder",
]


@contextmanager
def open_with_fallback(filename=None):
    if filename:
        file = open(filename, "w")
    else:
        file = sys.stdout
    try:
        yield file
    finally:
        if file is not sys.stdout:
            file.close()


class DataModuleLocalFolder(pl.LightningDataModule):
    """DataLoader for local folder containing source and target image sub-folders."""

    def __init__(
        self,
        campaign_transforms: CampaignTransforms,
        data_dir: Union[Path, str] = Path(),
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
        source_images_label: str = "image",
        target_images_label: str = "label",
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

        data_dir: Path | str = Path()
            Data directory, containing two source and target image sub-folders. The subfolder names
            and labels can be overwritten.

        num_folds: int = 1
            Set to a number larger than one to set up k-fold cross-validation. All images
            that do not belong to the test set fraction as defined by `test_fraction` will
            be rotated for k-fold cross-validations. In this regime, the training set will
            contain n * (k-1)/k images, while the validation set will contain n / k images
            (with k = num_folds, and n = number of images in the training + validation set).

        train_fraction: float = 0.7
            Fraction of images and corresponding labels that go into the training set.
            Please notice that this will be ignored if num_folds is larger than 1.

        val_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.
            Please notice that this will be ignored if num_folds is larger than 1.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.
            This is always used, independent of whether num_folds is 1 or larger.

        batch_size: int = 8
            Size of one batch of image pairs for training, validation and testing.

        inference_batch_size: int = 2
            Size of one batch of image pairs for full inference.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        num_patches: int = 1
            Number of patches per image to be extracted (and collated in the batch).

        source_images_sub_folder: str = "images"
            Name of the source images sub-folder. It can be used to override the default "images".

        target_images_sub_folder: str = "labels"
            Name of the target images sub-folder. It can be used to override the default "labels".

        source_images_label: str = "image"
            Label for the source images to be used by the Transforms. It can be used to override the default "image".

        target_images_label: str = "label"
            Label for the target images to be used by the Transforms. It can be used to override the default "label".

        seed: int = 42
            Seed for all random number generators.

        num_workers: int = os.cpu_count()
            Number of workers to be used in the training, validation and test data loaders.

        num_inference_workers: int
            Number of workers to be used in the inference data loader.

        pin_memory: bool = True
            Whether to pin the GPU memory.
        """

        super().__init__()
        self.campaign_transforms = campaign_transforms
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.num_workers = num_workers
        self.num_inference_workers = num_inference_workers
        self.pin_memory = pin_memory

        # Set the sub-folder names and labels
        self.source_images_sub_folder = source_images_sub_folder
        self.target_images_sub_folder = target_images_sub_folder
        self.source_images_label = source_images_label
        self.target_images_label = target_images_label

        # Set the seed
        if seed is None:
            seed = time.time_ns()
        self.seed = seed

        # Make sure the fractions add to 1.0
        total: float = train_fraction + val_fraction + test_fraction
        self.train_fraction: float = train_fraction / total
        self.val_fraction: float = val_fraction / total
        self.test_fraction: float = test_fraction / total

        # Keep track of all images and labels
        self._all_images: np.ndarray = np.empty(shape=0)
        self._all_labels: np.ndarray = np.empty(shape=0)

        # Set the information about k-fold cross-validation
        self.num_folds = num_folds
        self.current_fold = 0
        self.k_folds = None
        self._test_indices: np.ndarray = np.empty(shape=0)
        self._train_val_indices: np.ndarray = np.empty(shape=0)
        self._train_indices: np.ndarray = np.empty(shape=0)
        self._val_indices: np.ndarray = np.empty(shape=0)

        # Declare datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def print_sets(self, filename: Union[None, Path, str] = None):
        """Print the list of images/labels in the training, validation and test sets
        either to file or to standard output."""

        if len(self._train_indices) == 0:
            print("No images in the training, validation and test sets.")
            return

        # Write either to file or standard output
        with open_with_fallback(filename) as f:
            end = "\n"

            f.write(f"Training set ({len(self._train_indices)} image pairs):{end}")

            for i in range(len(self._train_indices)):
                f.write(
                    f"{i:5}: "
                    f"({self.source_images_label}: {self._all_images[self._train_indices[i]].name}, \t"
                    f"{self.target_images_label}: {self._all_labels[self._train_indices[i]].name}){end}"
                )

            f.write(f"\nValidation set ({len(self._val_indices)} image pairs):\n")
            for i in range(len(self._val_indices)):
                f.write(
                    f"{i:5}: "
                    f"({self.source_images_label}: {self._all_images[self._val_indices[i]].name}, \t"
                    f"{self.target_images_label}: {self._all_labels[self._val_indices[i]].name}){end}"
                )

            f.write(f"\nTest set ({len(self._test_indices)} image pairs):{end}")
            for i in range(len(self._test_indices)):
                f.write(
                    f"{i:5}: "
                    f"({self.source_images_label}: {self._all_images[self._test_indices[i]].name}, \t"
                    f"{self.target_images_label}: {self._all_labels[self._test_indices[i]].name}){end}"
                )

    def setup(self, stage):
        """Prepare the data once."""

        if (
            self.train_dataset is not None
            and self.val_dataset is not None
            and self.test_dataset is not None
        ):
            # Data is already prepared
            return

        # Scan the "images" and "labels" folders and clean them
        image_candidates = list(
            (self.data_dir / self.source_images_sub_folder).glob("*.tif")
        )
        image_candidates = [
            image for image in image_candidates if not str(image).startswith("._")
        ]
        label_candidates = list(
            (self.data_dir / self.target_images_sub_folder).glob("*.tif")
        )
        label_candidates = [
            label for label in label_candidates if not str(label).startswith("._")
        ]

        # Store them
        self._all_images = np.array(natsorted(image_candidates))
        self._all_labels = np.array(natsorted(label_candidates))

        # Check that we found some images
        if len(self._all_images) == 0:
            raise Exception("Could not find any images to process.")

        # Check that there are as many images as labels
        if len(self._all_images) != len(self._all_labels):
            raise Exception("The number of images does not match the number of labels.")

        # Partition the data into a test set, and a combined training + validation set,
        # that can be used to easily create folds if requested.from
        rng = default_rng(seed=self.seed)
        shuffled_indices = rng.permutation(len(self._all_images))

        # Extract the indices for the test set and assign everything else to the combined
        # training + validation set
        num_test_images = int(round(self.test_fraction * len(self._all_images)))
        if num_test_images == 0:
            num_test_images = 1
        self._test_indices = shuffled_indices[-num_test_images:]
        self._train_val_indices = shuffled_indices[:-num_test_images]

        if self.num_folds == 1:
            # Update the training and validation fractions since the test set
            # has been removed already
            updated_val_fraction = self.val_fraction / (1.0 - self.test_fraction)
            num_val_images = int(
                round(updated_val_fraction * len(self._train_val_indices))
            )
            if num_val_images == 0:
                num_val_images = 1
            self._val_indices = self._train_val_indices[-num_val_images:]
            self._train_indices = self._train_val_indices[:-num_val_images]

            # Create the datasets
            self._create_datasets()

        else:
            # Use k-fold split to set the indices for current fold - the datasets
            # will be automatically re-created
            self.set_fold(self.current_fold)

        assert len(self._train_indices) > 0, "Current split yields 0 training images!"
        assert len(self._val_indices) > 0, "Current split yields 0 validation images!"
        assert len(self._test_indices) > 0, "Current split yields 0 testing images!"
        assert len(self._train_indices) + len(self._val_indices) + len(
            self._test_indices
        ) == len(self._all_images), "Something went wrong with the partitioning!"

    def set_fold(self, fold: int):
        """Set current fold for k-fold cross-validation."""
        if self.num_folds == 1:
            # We don't need to do anything, but if the user sets
            # the fold to 0 we have a valid non-action
            if fold == 0:
                self.current_fold = 0
                return
            else:
                raise ValueError("k-fold cross-validation is not active.")
        else:
            if fold < 0 or fold >= self.num_folds:
                raise ValueError(
                    f"`fold` must be in the range (0..{self.num_folds - 1})."
                )
            else:
                if self.k_folds is None:
                    self.k_folds = KFold(n_splits=self.num_folds, shuffle=False)
                self.current_fold = fold
                for i, (fold_train_indices, fold_val_indices) in enumerate(
                    self.k_folds.split(self._train_val_indices)
                ):
                    if i == self.current_fold:
                        self._train_indices = fold_train_indices
                        self._val_indices = fold_val_indices
                        break

                # Recreate the datasets
                self._create_datasets()

    def _create_datasets(self):
        """Create datasets based on current splits."""
        # Create the training dataset
        train_files = [
            {self.source_images_label: image, self.target_images_label: label}
            for image, label in zip(
                self._all_images[self._train_indices],
                self._all_labels[self._train_indices],
            )
        ]
        self.train_dataset = Dataset(
            data=train_files, transform=self.campaign_transforms.get_train_transforms()
        )

        # Create the validation dataset
        val_files = [
            {self.source_images_label: image, self.target_images_label: label}
            for image, label in zip(
                self._all_images[self._val_indices], self._all_labels[self._val_indices]
            )
        ]
        self.val_dataset = Dataset(
            data=val_files, transform=self.campaign_transforms.get_valid_transforms()
        )

        # Create the testing dataset
        test_files = [
            {self.source_images_label: image, self.target_images_label: label}
            for image, label in zip(
                self._all_images[self._test_indices],
                self._all_labels[self._test_indices],
            )
        ]
        self.test_dataset = Dataset(
            data=test_files, transform=self.campaign_transforms.get_test_transforms()
        )

        # Inform
        print(
            f"Working with {len(self._train_indices)} training, "
            f"{len(self._val_indices)} validation, and "
            f"{len(self._test_indices)} test image/label pairs.",
            end=" ",
        )
        if self.num_folds > 1:
            print(
                f"K-Folds cross-validator: {self.num_folds} folds (0 .. {self.num_folds - 1}). "
                f"Current fold: {self.current_fold}."
            )
        else:
            print("One-way split.")

    def train_dataloader(self):
        """Return DataLoader for Training data."""
        return DataLoader(
            self.train_dataset,
            shuffle=False,  # Already shuffled
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Return DataLoader for Validation data."""
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Return DataLoader for Testing data."""
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=list_data_collate,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def inference_dataloader(
        self, input_folder: Union[Path, str], fmt_filter: str = "*.tif"
    ):
        """Return DataLoader for full inference."""

        # Scan for images
        image_names = natsorted(list(Path(input_folder).glob(fmt_filter)))

        # Create a DataSet
        inference_dataset = ArrayDataset(
            image_names,
            img_transform=self.campaign_transforms.get_inference_transforms(),
            seg=None,
            seg_transform=None,
            labels=None,
            label_transform=None,
        )

        # If there are fewer images than workers do not pin memory and do not
        # fire up more workers than necessary.
        pin_memory = self.pin_memory
        inference_batch_size = self.inference_batch_size
        num_inference_workers = self.num_inference_workers
        persistent_workers = True if self.num_inference_workers > 0 else False
        n_images = len(image_names)
        if n_images <= self.inference_batch_size:
            pin_memory = False
            inference_batch_size = n_images
            num_inference_workers = n_images
            persistent_workers = False

        # Return the DataLoader
        return DataLoader(
            inference_dataset,
            batch_size=inference_batch_size,
            num_workers=num_inference_workers,
            collate_fn=list_data_collate,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
