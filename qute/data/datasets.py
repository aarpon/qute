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
from torch.utils.data import Dataset
from tifffile import imread
from qute.data.utils import sample


class ImageLabelDataset(Dataset):
    def __init__(
            self,
            images_path_list: list,
            labels_path_list: list,
            patch_size: tuple = (128, 128),
            num_patches: int = 1,
            transform=None,
            target_transform=None
    ):
        self.images = images_path_list
        self.labels = labels_path_list
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Get the image and label file names
        image_filename = self.images[idx]
        label_filename = self.labels[idx]

        # Load the images
        image = imread(image_filename).astype(np.float32)
        label = imread(label_filename).astype(np.int32)

        # @TODO Extract `num_patches` patches
        sub_image, y0, x0 = sample(image, patch_size=self.patch_size)
        sub_label, _, _ = sample(label, patch_size=self.patch_size, y0=y0, x0=x0)

        if self.transform:
            sub_image = self.transform(sub_image)
        if self.target_transform:
            sub_label = self.target_transform(sub_label)
        return sub_image, sub_label
