import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import yaml
from monai.data import Dataset
from monai.transforms import (
    AsDiscreted,
    Compose,
    LoadImaged,
    RandRotate90d,
    RandSpatialCropd,
)
from natsort import natsorted
from numpy.random import default_rng
from torch.utils.data import DataLoader

from qute.data.io import (
    get_cell_restoration_demo_dataset,
    get_cell_segmentation_demo_dataset,
)
from qute.transforms import MinMaxNormalized, ToPyTorchOutputd


class DataModuleLocalFolder(pl.LightningDataModule):
    """DataLoader for local folder containing 'images' and 'labels' sub-folders."""

    def __init__(
        self,
        data_dir: Union[Path, str] = Path(),
        num_classes: int = 3,
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        batch_size: int = 8,
        patch_size: tuple = (512, 512),
        train_transforms_dict: Optional[list] = None,
        val_transforms_dict: Optional[list] = None,
        test_transforms_dict: Optional[list] = None,
        images_sub_folder: str = "images",
        labels_sub_folder: str = "labels",
        image_range_intensities: Optional[tuple[int, int]] = None,
        seed: int = 42,
        num_workers: Optional[int] = os.cpu_count(),
        pin_memory: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------

        data_dir: Path | str = Path()
            Data directory, containing two sub-folders "images" and "labels". The subfolder names can be overwritten.

        num_classes: int = 3
            Number of output classes (labels).

        train_fraction: float = 0.7
            Fraction of images and corresponding labels that go into the training set.

        val_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.

        batch_size: int = 8
            Size of one batch of image pairs.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        train_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the training images and labels. If omitted some default transforms will be applied.

        val_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the validation images and labels. If omitted some default transforms will be applied.

        test_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the test images and labels. If omitted some default transforms will be applied.

        images_sub_folder: str = "images"
            Name of the images sub-folder. It can be used to override the default "images".

        labels_sub_folder: str = "labels"
            Name of the labels sub-folder. It can be used to override the default "labels".

        seed: int = 42
            Seed for all random number generators.

        num_workers: int = os.cpu_count()
            Number of workers to be used in the data loaders.

        pin_memory: bool = True
            Whether to pin the GPU memory.
        """

        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.batch_size = batch_size
        self.patch_size = patch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_classes = num_classes

        # Normalization range
        if image_range_intensities is None:
            self.image_range_intensities = (0, 65535)
        else:
            self.image_range_intensities = image_range_intensities

        # Set the subfolder names
        self.images_sub_folder = images_sub_folder
        self.labels_sub_folder = labels_sub_folder

        # Set the transforms is any are passed
        self.train_transforms_dict = None
        if train_transforms_dict is not None:
            self.train_transforms_dict = train_transforms_dict
        else:
            self.train_transforms_dict = self.get_train_transforms_dict()

        self.val_transforms_dict = None
        if val_transforms_dict is not None:
            self.val_transforms_dict = val_transforms_dict
        else:
            self.val_transforms_dict = self.get_val_transforms_dict()

        self.test_transforms_dict = None
        if test_transforms_dict is not None:
            self.test_transforms_dict = test_transforms_dict
        else:
            self.test_transforms_dict = self.get_test_transforms_dict()

        # Set the seed
        if seed is None:
            seed = time.time_ns()
        self.seed = seed

        # Make sure the fractions add to 1.0
        total: float = train_fraction + val_fraction + test_fraction
        self.train_fraction: float = train_fraction / total
        self.val_fraction: float = val_fraction / total
        self.test_fraction: float = test_fraction / total

        # Declare lists of training, validation, and testing images and labels
        self.images: list[Path] = []
        self.labels: list[Path] = []

        # Keep track of the file names of the training, validation, and testing sets
        self.train_images: list[Path] = []
        self.train_labels: list[Path] = []
        self.val_images: list[Path] = []
        self.val_labels: list[Path] = []
        self.test_images: list[Path] = []
        self.test_labels: list[Path] = []

        # Declare datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage):
        """Prepare the data once."""

        if (
            self.train_dataset is not None
            and self.val_dataset is not None
            and self.test_dataset is not None
        ):
            # Data is already prepared
            return

        # Scan the "images" and "labels" folders
        self.images = natsorted(
            list((self.data_dir / self.images_sub_folder).glob("*.tif"))
        )
        self.labels = natsorted(
            list((self.data_dir / self.labels_sub_folder).glob("*.tif"))
        )

        # Check that we found some images
        if len(self.images) == 0:
            raise Exception("Could not find any images to process.")

        # Check that there are as many images as labels
        if len(self.images) != len(self.labels):
            raise Exception("The number of images does not match the number of labels.")

        # Shuffle a copy of the file names
        rng = default_rng(seed=self.seed)
        shuffled_indices = rng.permutation(len(self.images))
        shuffled_images = np.array(self.images.copy())[shuffled_indices].tolist()
        shuffled_labels = np.array(self.labels.copy())[shuffled_indices].tolist()

        # Partition images and labels into training, validation and testing datasets
        train_len = round(self.train_fraction * len(shuffled_images))
        len_rest = len(shuffled_images) - train_len
        updated_val_fraction = self.val_fraction / (
            self.val_fraction + self.test_fraction
        )
        val_len = round(updated_val_fraction * len_rest)
        test_len = len_rest - val_len

        # Split the sets
        self.train_images = shuffled_images[:train_len]
        self.train_labels = shuffled_labels[:train_len]
        self.val_images = shuffled_images[train_len:-test_len]
        self.val_labels = shuffled_labels[train_len:-test_len]
        self.test_images = shuffled_images[-test_len:]
        self.test_labels = shuffled_labels[-test_len:]

        assert len(self.train_images) + len(self.val_images) + len(
            self.test_images
        ) == len(shuffled_images), "Something went wrong with the partitioning!"

        # Create the training dataset
        train_files = [
            {"image": image, "label": label}
            for image, label in zip(self.train_images, self.train_labels)
        ]
        self.train_dataset = Dataset(
            data=train_files, transform=self.train_transforms_dict
        )

        # Create the validation dataset
        val_files = [
            {"image": image, "label": label}
            for image, label in zip(self.val_images, self.val_labels)
        ]
        self.val_dataset = Dataset(data=val_files, transform=self.val_transforms_dict)

        # Create the testing dataset
        test_files = [
            {"image": image, "label": label}
            for image, label in zip(self.test_images, self.test_labels)
        ]
        self.test_dataset = Dataset(
            data=test_files, transform=self.test_transforms_dict
        )

    def train_dataloader(self):
        """Return DataLoader for Training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        """Return DataLoader for Validation data."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        """Return DataLoader for Testing data."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_train_transforms_dict(self):
        """Define default training set transforms."""
        train_transforms = Compose(
            [
                LoadImaged(
                    keys=["image", "label"],
                    reader="PILReader",
                    image_only=True,
                    ensure_channel_first=True,
                    dtype=np.float32,
                ),
                MinMaxNormalized(
                    min_intensity=self.image_range_intensities[0],
                    max_intensity=self.image_range_intensities[1],
                ),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=self.patch_size, random_size=False
                ),
                RandRotate90d(keys=["image", "label"], prob=1.0, spatial_axes=(0, 1)),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
                ToPyTorchOutputd(),
            ]
        )
        return train_transforms

    def get_val_transforms_dict(self):
        """Define default validation set transforms."""
        val_transforms = Compose(
            [
                LoadImaged(
                    keys=["image", "label"],
                    reader="PILReader",
                    image_only=True,
                    ensure_channel_first=True,
                    dtype=np.float32,
                ),
                MinMaxNormalized(
                    min_intensity=self.image_range_intensities[0],
                    max_intensity=self.image_range_intensities[1],
                ),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=self.patch_size, random_size=False
                ),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
                ToPyTorchOutputd(),
            ]
        )
        return val_transforms

    def get_test_transforms_dict(self):
        """Define default test set transforms."""
        test_transforms = Compose(
            [
                LoadImaged(
                    keys=["image", "label"],
                    reader="PILReader",
                    image_only=True,
                    ensure_channel_first=True,
                    dtype=np.float32,
                ),
                MinMaxNormalized(
                    min_intensity=self.image_range_intensities[0],
                    max_intensity=self.image_range_intensities[1],
                ),
                RandSpatialCropd(
                    keys=["image", "label"], roi_size=self.patch_size, random_size=False
                ),
                AsDiscreted(keys=["label"], to_onehot=self.num_classes),
                ToPyTorchOutputd(),
            ]
        )
        return test_transforms


class CellSegmentationDemo(DataModuleLocalFolder):
    """DataLoader for the Cell Segmentation Demo."""

    def __init__(
        self,
        download_dir: Union[Path, str] = Path.home() / ".qute" / "data",
        three_classes: bool = True,
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        batch_size: int = 8,
        patch_size: tuple = (512, 512),
        train_transforms_dict: Optional[list] = None,
        val_transforms_dict: Optional[list] = None,
        test_transforms_dict: Optional[list] = None,
        images_sub_folder: str = "images",
        labels_sub_folder: str = "labels",
        seed: int = 42,
        num_workers: Optional[int] = os.cpu_count(),
        pin_memory: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------

        download_dir: Path | str = Path()
            Directory where the cell segmentation datasets will be downloaded and extracted.

        three_classes: bool = True
            Whether to download and extract the demo dataset with 3 labels (classes), or the one with two.

        train_fraction: float = 0.7
            Fraction of images and corresponding labels that go into the training set.

        val_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.

        batch_size: int = 8
            Size of one batch of image pairs.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        train_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the training images and labels. If omitted some default transforms will be applied.

        val_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the validation images and labels. If omitted some default transforms will be applied.

        test_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the test images and labels. If omitted some default transforms will be applied.

        images_sub_folder: str = "images"
            Name of the images sub-folder. It can be used to override the default "images".

        labels_sub_folder: str = "labels"
            Name of the labels sub-folder. It can be used to override the default "labels".

        seed: int = 42
            Seed for all random number generators.

        num_workers: int = os.cpu_count()
            Number of workers to be used in the data loaders.

        pin_memory: bool = True
            Whether to pin the GPU memory.
        """

        # Download directory is parent of the actual data_dir that we pass to the parent class
        self.download_dir = Path(download_dir).resolve()

        # Make the number of classes explicit
        self.three_classes = three_classes
        if self.three_classes:
            self.num_classes = 3
        else:
            self.num_classes = 2
        data_dir = self.download_dir / f"demo_segmentation_{self.num_classes}_classes"

        # Parse the metadata file
        with open(data_dir / "metadata.yaml") as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)

        # Call base constructor
        super().__init__(
            data_dir=data_dir,
            num_classes=self.num_classes,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            batch_size=batch_size,
            patch_size=patch_size,
            train_transforms_dict=train_transforms_dict,
            val_transforms_dict=val_transforms_dict,
            test_transforms_dict=test_transforms_dict,
            images_sub_folder=images_sub_folder,
            labels_sub_folder=labels_sub_folder,
            image_range_intensities=(
                metadata["min_intensity"],
                metadata["max_intensity"],
            ),
            seed=seed,
            num_workers=num_workers,
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


class CellRestorationDemo(DataModuleLocalFolder):
    """DataLoader for the Cell Restoration Demo."""

    def __init__(
        self,
        download_dir: Union[Path, str] = Path.home() / ".qute" / "data",
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        batch_size: int = 8,
        patch_size: tuple = (512, 512),
        train_transforms_dict: Optional[list] = None,
        val_transforms_dict: Optional[list] = None,
        test_transforms_dict: Optional[list] = None,
        images_sub_folder: str = "images",
        labels_sub_folder: str = "targets",
        seed: int = 42,
        num_workers: Optional[int] = os.cpu_count(),
        pin_memory: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------

        download_dir: Path | str = Path()
            Directory where the cell segmentation datasets will be downloaded and extracted.

        train_fraction: float = 0.7
            Fraction of images and corresponding labels that go into the training set.

        val_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.

        batch_size: int = 8
            Size of one batch of image pairs.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        train_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the training images and labels. If omitted some default transforms will be applied.

        val_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the validation images and labels. If omitted some default transforms will be applied.

        test_transforms_dict: Optional[list] = None
            Dictionary transforms to be applied to the test images and labels. If omitted some default transforms will be applied.

        images_sub_folder: str = "images"
            Name of the images sub-folder. It can be used to override the default "images".

        labels_sub_folder: str = "targets"
            Name of the labels sub-folder. It can be used to override the default "labels".

        seed: int = 42
            Seed for all random number generators.

        num_workers: int = os.cpu_count()
            Number of workers to be used in the data loaders.

        pin_memory: bool = True
            Whether to pin the GPU memory.
        """

        # Download directory is parent of the actual data_dir that we pass to the parent class
        self.download_dir = Path(download_dir).resolve()

        # Data directory
        data_dir = self.download_dir / f"demo_restoration"

        # Parse the metadata file
        with open(data_dir / "metadata.yaml") as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)

        # Call base constructor
        super().__init__(
            data_dir=data_dir,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            batch_size=batch_size,
            patch_size=patch_size,
            train_transforms_dict=train_transforms_dict,
            val_transforms_dict=val_transforms_dict,
            test_transforms_dict=test_transforms_dict,
            images_sub_folder=images_sub_folder,
            labels_sub_folder=labels_sub_folder,
            image_range_intensities=(
                metadata["min_intensity"],
                metadata["max_intensity"],
            ),
            seed=seed,
            num_workers=num_workers,
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
