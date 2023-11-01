import os
from pathlib import Path
from typing import Optional, Union

import userpaths

from qute.data.dataloaders import SegmentationDataModuleLocalFolder
from qute.data.io import get_cell_segmentation_demo_dataset


class CellSegmentationDemo(SegmentationDataModuleLocalFolder):
    """DataLoader for the Cell Segmentation Demo."""

    def __init__(
        self,
        download_dir: Union[Path, str, None] = None,
        three_classes: bool = True,
        train_fraction: float = 0.7,
        val_fraction: float = 0.2,
        test_fraction: float = 0.1,
        batch_size: int = 8,
        inference_batch_size: int = 2,
        patch_size: tuple = (512, 512),
        num_patches: int = 1,
        train_transforms_dict: Optional[list] = None,
        val_transforms_dict: Optional[list] = None,
        test_transforms_dict: Optional[list] = None,
        images_sub_folder: str = "images",
        labels_sub_folder: str = "labels",
        seed: int = 42,
        num_workers: Optional[int] = os.cpu_count(),
        num_inference_workers: Optional[int] = 2,
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

        inference_batch_size: int = 2
            Size of one batch of image pairs for full inference.

        patch_size: tuple = (512, 512)
            Size of the patch to be extracted (at random positions) from images and labels.

        num_patches: int = 1
            Number of patches per image to be extracted (and collated in the batch).

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
            data_dir=data_dir,
            num_classes=self.num_classes,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            patch_size=patch_size,
            num_patches=num_patches,
            train_transforms_dict=train_transforms_dict,
            val_transforms_dict=val_transforms_dict,
            test_transforms_dict=test_transforms_dict,
            images_sub_folder=images_sub_folder,
            labels_sub_folder=labels_sub_folder,
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
