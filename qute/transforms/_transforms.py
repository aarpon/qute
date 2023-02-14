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

from typing import Dict

import numpy as np
import torch
from monai.transforms import Transform


class MinMaxNormalize(Transform):
    """Normalize a tensor to [0, 1] using given min and max absolute intensities."""

    def __init__(self, min_intensity: int = 0, max_intensity: int = 65535) -> None:
        """Constructor

        Parameters
        ----------

        min_intensity: int
            Minimum intensity to normalize against (optional, default = 0).
        max_intensity: int
            Maximum intensity to normalize against (optional, default = 65535).

        Returns
        -------

        norm: tensor
            Normalized tensor.
        """
        super().__init__()
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.range_intensity = self.max_intensity - self.min_intensity

    def __call__(self, image: torch.tensor) -> torch.tensor:
        """
        Apply the transform to `image`.

        Returns
        -------

        result: tensor
            A stack of images with the same width and height as `label` and with `num_classes` planes.
        """
        return (image - self.min_intensity) / self.range_intensity


class MinMaxNormalized(Transform):
    """Normalize the "image" tensor to [0, 1] using given min and max absolute intensities from the data dictionary."""

    def __init__(self, min_intensity: int = 0, max_intensity: int = 65535) -> None:
        """Constructor

        Parameters
        ----------

        min_intensity: int
            Minimum intensity to normalize against (optional, default = 0).
        max_intensity: int
            Maximum intensity to normalize against (optional, default = 65535).
        """
        super().__init__()
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.range_intensity = self.max_intensity - self.min_intensity

    def __call__(self, data: dict) -> dict:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with normalized "image" tensor.
        """
        data["image"] = (data["image"] - self.min_intensity) / self.range_intensity
        return data


class ToLabel(Transform):
    """
    Converts tensor from one-hot representation to 2D label image.
    """

    def __init__(self, dtype=torch.int32):
        super().__init__()
        self.dtype = dtype

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Unwrap the dictionary."""
        if data.ndim == 4:
            data = data.argmax(axis=1).type(self.dtype)
        elif data.ndim == 3:
            data = data.argmax(axis=0).type(self.dtype)
        else:
            raise ValueError("The input tensor must be of size (NCHW) or (HW).")

        return data


class ToPyTorchOutputd(Transform):
    """
    Simple converter to pass from the dictionary output of Monai transfors to the expected (image, label) tuple used by PyTorch Lightning.
    """

    def __init__(
        self,
        image_key: str = "image",
        image_dtype: torch.dtype = torch.float32,
        label_key: str = "label",
        label_dtype: torch.dtype = torch.int32,
    ):
        super().__init__()
        self.image_key = image_key
        self.label_key = label_key
        self.image_dtype = image_dtype
        self.label_dtype = label_dtype

    def __call__(self, data: dict) -> tuple:
        """Unwrap the dictionary."""
        return data[self.image_key].type(self.image_dtype), data[self.label_key].type(
            self.label_dtype
        )


class ZNormalize(Transform):
    """Standardize the passed tensor by subracting the mean and dividing by the standard deviation."""

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: torch.Tensor
            Normalized tensor.
        """
        return (data - data.mean()) / data.std()


class ZNormalized(Transform):
    """Standardize the "image" tensor by subracting the mean and dividing by the standard deviation."""

    def __init__(self, image_key: str = "image") -> None:
        """Constructor"""
        super().__init__()
        self.image_key = image_key

    def __call__(self, data: dict) -> dict:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with normalized "image" tensor.
        """
        mn = data[self.image_key].mean()
        sd = data[self.image_key].std()
        data[self.image_key] = (data[self.image_key] - mn) / sd
        return data


class DebugInformer(Transform):
    """
    Simple reporter to be added to a Composed list of Transforms
    to return some information. The data is returned untouched.
    """

    def __init__(self, *args, **kwargs):
        """Constructor.

        Parameters
        ----------

        name: str
            Name of the Informer. It is printed as a prefix to the output.
        """
        super().__init__()
        self.name = ""
        if "name" in kwargs:
            self.name = kwargs["name"]

    def __call__(self, data):
        """Call the Transform."""
        prefix = f"{self.name} :: " if self.name != "" else ""
        if type(data) == tuple and len(data) > 1:
            data = data[0]
        if type(data) == torch.Tensor:
            print(
                f"{prefix}"
                f"Type = Torch Tensor: "
                f"size = {data.size()}, "
                f"type = {data.dtype}, "
                f"min = {data.min()}, "
                f"mean = {data.double().mean()}, "
                f"median = {torch.median(data).item()}, "
                f"max = {data.max()}"
            )
        elif type(data) == np.ndarray:
            print(
                f"{prefix}"
                f"Type = Numpy Array: "
                f"size = {data.shape}, "
                f"type = {data.dtype}, "
                f"min = {data.min()}, "
                f"mean = {data.mean()}, "
                f"median = {np.median(data)}, "
                f"max = {data.max()}"
            )
        elif type(data) == str:
            print(f"{prefix}String: value = '{str}'")
        elif str(type(data)).startswith("<class 'itk."):
            # This is a bit of a hack..."
            data_numpy = np.array(data)
            print(
                f"{prefix}"
                f"Type = ITK Image: "
                f"size = {data_numpy.shape}, "
                f"type = {data_numpy.dtype}, "
                f"min = {data_numpy.min()}, "
                f"mean = {data_numpy.mean()}, "
                f"median = {np.median(data_numpy)}, "
                f"max = {data_numpy.max()}"
            )
        else:
            try:
                print(f"{prefix}{type(data)}: {str(data)}")
            except:
                print(f"{prefix}Unknown type!")
        return data


class DebugMinNumVoxelCheckerd(Transform):
    """
    Simple reporter to be added to a Composed list of Transforms
    to return some information. The data is returned untouched.
    """

    def __init__(self, keys: Dict, class_num: int, min_fraction: float = 0.0):
        """Constructor.

        Parameters
        ----------

        class_num: int
            Number of the class to check for presence.
        """
        super().__init__()
        self.keys = keys
        self.class_num = class_num
        self.min_fraction = min_fraction

    def __call__(self, data):
        """Call the Transform."""

        # Get the keys
        keys = data.keys()

        # Only work on the expected keys
        for key in keys:
            if key not in self.keys:
                continue

            # The Tensor should be in OneHot format, and the class number should
            # map to the index of the first dimension.
            if self.class_num > (data[key].shape[0] - 1):
                raise Exception("`num_class` is out of boundaries.")

            num_voxels = data[key][self.class_num].sum()
            if self.min_fraction == 0.0:
                if num_voxels > 0:
                    return data

            if self.min_fraction > 0.0:
                area = data[key].shape[1] * data[key].shape[2]
                if num_voxels / area >= self.min_fraction:
                    return data

            raise Exception(
                f"{self.name}: The number of voxels for {self.class_num} is lower than expected ({num_voxels})."
            )

        # Return data
        return data
