#  ********************************************************************************
#   Copyright © 2022-, ETH Zurich, D-BSSE, Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the Apache License Version 2.0
#   which accompanies this distribution, and is available at
#   https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Contributors:
#       Aaron Ponti - initial API and implementation
#  ******************************************************************************/

import numpy as np
from tifffile import imread
from torch.utils.data import Dataset

from qute.data.utils import sample


class ImageLabelDataset(Dataset):
    """Dataset that maps an gray-value image to a label image."""

    def __init__(
        self,
        images_path_list: list,
        labels_path_list: list,
        patch_size: tuple = (512, 512),
        img_dtype: np.dtype = np.float32,
        label_dtype: np.dtype = np.int32,
        transform=None,
        target_transform=None,
    ):
        """
        Constructor.

        Parameters
        ----------

        images_path_list: list
            List of image paths.

        labels_path_list: list
            List of label image paths.

        patch_size: tuple = (512, 512)
            Size of each patch to be extracted.

        img_dtype: Optional[np.dtype]
            Data type for the loaded image. If not specified, it defaults to np.float32.

        label_dtype: Optional[np.dtype]
            Data type for the loaded label image. If not specified, it defaults to np.int32.

        transform=None
            Transforms to be applied to the images.

        target_transform=None
            Transforms to be applied to the labels.

        """
        self.images = images_path_list
        self.labels = labels_path_list
        self.patch_size = patch_size
        self.img_dtype = img_dtype
        self.label_dtype = label_dtype
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return lenght of image list."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get item at requested index."""

        # Get the image and label file names
        image_filename = self.images[idx]
        label_filename = self.labels[idx]

        # Load the images
        image = imread(image_filename).astype(self.img_dtype)
        label = imread(label_filename).astype(self.label_dtype)

        # Extract the same random area from both images
        sub_image, y0, x0 = sample(image, patch_size=self.patch_size)
        sub_label, _, _ = sample(label, patch_size=self.patch_size, y0=y0, x0=x0)

        if self.transform:
            sub_image = self.transform(sub_image)
        if self.target_transform:
            sub_label = self.target_transform(sub_label)
        return sub_image, sub_label
