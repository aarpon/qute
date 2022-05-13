from typing import Optional
from pathlib import Path
from natsort import natsorted
import numpy as np
from numpy.random import default_rng
import time
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from qute.data.datasets import ImageLabelDataset
from qute.data.io import get_cell_segmentation_demo_dataset
import os

from qute.transforms import MinMaxNormalize


class DataModuleLocalFolder(pl.LightningDataModule):
    """DataLoader for local folder containing 'images' and 'labels' sub-folders."""

    def __init__(
            self,
            data_dir: Path | str = Path(),
            num_classes: int = 3,
            train_fraction: float = 0.7,
            valid_fraction: float = 0.2,
            test_fraction: float = 0.1,
            batch_size: int = 8,
            patch_size: tuple = (512, 512),
            num_patches: int = 1,
            images_transform: Optional[list] = None,
            labels_transform: Optional[list] = None,
            images_sub_folder: str = "images",
            labels_sub_folder: str = "labels",
            seed: int = 42,
            num_workers: int = os.cpu_count(),
            pin_memory: bool = True
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

        valid_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.
            
        batch_size: int = 8
            Size of one batch of image pairs.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        num_patches: int = 1
            Number of patches to be extracted from images and labels.
            PLEASE NOTE: this is currently hard-coded to 1.

        images_transform: Optional[list] = None
            Transforms to to be applied to the images. If omitted some default transforms will be applied.

        labels_transform: Optional[list] = None
            Transforms to to be applied to the labels. If omitted some default transforms will be applied.

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
        self.num_patches = num_patches

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_classes = num_classes

        # Set the subfolder names
        self.images_sub_folder = images_sub_folder
        self.labels_sub_folder = labels_sub_folder

        # Set the transforms is any are passed
        self.images_transform = None
        if images_transform is not None:
            self.images_transform = images_transform
        self.labels_transform = None
        if labels_transform is not None:
            self.labels_transform = labels_transform

        # Set the seed
        if seed is None:
            seed = time.time_ns()
        self.seed = seed

        # Metadata
        self.metadata = {}

        # Make sure the fractions add to 1.0
        total = train_fraction + valid_fraction + test_fraction
        self.train_fraction = train_fraction / total
        self.valid_fraction = valid_fraction / total
        self.test_fraction = test_fraction / total

        # Declare lists of training, validation, and testing images and labels
        self.images = []
        self.labels = []

        # Keep track of the file names of the training, validation, and testing sets
        self.train_images = []
        self.train_labels = []
        self.valid_images = []
        self.valid_labels = []
        self.test_images = []
        self.test_labels = []

        # Declare datasets
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage):
        """Prepare the data once."""

        if self.train_dataset is not None and self.valid_dataset is not None and self.test_dataset is not None:
            # Data is already prepared
            return

        # Scan the "images" and "labels" folders
        self.images = natsorted(list((self.data_dir / self.images_sub_folder).glob("*.tif")))
        self.labels = natsorted(list((self.data_dir / self.labels_sub_folder).glob("*.tif")))

        # Parse the metadata file
        with open(self.data_dir / "metadata.yaml") as f:
            self.metadata = yaml.load(f, Loader=yaml.FullLoader)

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
        updated_valid_fraction = self.valid_fraction / (self.valid_fraction + self.test_fraction)
        valid_len = round(updated_valid_fraction * len_rest)
        test_len = len_rest - valid_len

        # Split the sets
        self.train_images = shuffled_images[:train_len]
        self.train_labels = shuffled_labels[:train_len]
        self.valid_images = shuffled_images[train_len:-test_len]
        self.valid_labels = shuffled_labels[train_len:-test_len]
        self.test_images = shuffled_images[-test_len:]
        self.test_labels = shuffled_labels[-test_len:]

        assert len(self.train_images) + len(self.valid_images) + len(self.test_images) == len(shuffled_images), \
            "Something went wrong with the partitioning!"

        # If no transforms are set, put the defaults here
        if self.images_transform is None:
            self.images_transform = transforms.Compose([
                transforms.ToTensor(),
                MinMaxNormalize(
                    min_intensity=self.metadata["min_intensity"],
                    max_intensity=self.metadata["max_intensity"]
                )
            ])

        if self.labels_transform is None:
            self.labels_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # Create the training dataset
        self.train_dataset = ImageLabelDataset(
            self.train_images,
            self.train_labels,
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            transform=self.images_transform,
            target_transform=self.labels_transform
        )

        # Create the validation dataset
        self.valid_dataset = ImageLabelDataset(
            self.valid_images,
            self.valid_labels,
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            transform=self.images_transform,
            target_transform=self.labels_transform
        )

        # Create the testing dataset
        self.test_dataset = ImageLabelDataset(
            self.test_images,
            self.test_labels,
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            transform=self.images_transform,
            target_transform=self.labels_transform
        )

    def train_dataloader(self):
        """Return DataLoader for Training data."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        """Return DataLoader for Validation data."""
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        """Return DataLoader for Testing data."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


class CellSegmentationDemo(DataModuleLocalFolder):
    """DataLoader for the Cell Segmentation Demo."""

    def __init__(
            self,
            download_dir: Path | str = Path.home() / ".qute" / "data",
            three_classes: bool = True,
            train_fraction: float = 0.7,
            valid_fraction: float = 0.2,
            test_fraction: float = 0.1,
            batch_size: int = 8,
            patch_size: tuple = (512, 512),
            num_patches: int = 1,
            images_transform: Optional[list] = None,
            labels_transform: Optional[list] = None,
            images_sub_folder: str = "images",
            labels_sub_folder: str = "labels",
            seed: int = 42,
            num_workers: int = os.cpu_count(),
            pin_memory: bool = True
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

        valid_fraction: float = 0.2
            Fraction of images and corresponding labels that go into the validation set.

        test_fraction: float = 0.1
            Fraction of images and corresponding labels that go into the test set.
            
        batch_size: int = 8
            Size of one batch of image pairs.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        num_patches: int = 1
            Number of patches to be extracted from images and labels.
            PLEASE NOTE: this is currently hard-coded to 1.

        images_transform: Optional[list] = None
            Transforms to to be applied to the images. If omitted some default transforms will be applied.

        labels_transform: Optional[list] = None
            Transforms to to be applied to the labels. If omitted some default transforms will be applied.

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

        # Call base constructor
        super().__init__(
            data_dir=data_dir, num_classes=self.num_classes, train_fraction=train_fraction,
            valid_fraction=valid_fraction, test_fraction=test_fraction, batch_size=batch_size,
            patch_size=patch_size, num_patches=num_patches, images_transform=images_transform,
            labels_transform=labels_transform, images_sub_folder=images_sub_folder,
            labels_sub_folder=labels_sub_folder, seed=seed, num_workers=num_workers, 
            pin_memory=pin_memory
        )

    def prepare_data(self):
        """Prepare the data on main thread."""

        # Download and extract the demo dataset if needed.
        get_cell_segmentation_demo_dataset(self.download_dir, three_classes=self.three_classes)

    def setup(self, stage):
        """Set up data on each GPU."""
        # Call parent setup()
        return super().setup(stage)
