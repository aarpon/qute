#  ********************************************************************************
#   Copyright © 2022 - 2003, ETH Zurich, D-BSSE, Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the Apache License Version 2.0
#   which accompanies this distribution, and is available at
#   https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Contributors:
#       Aaron Ponti - initial API and implementation
#  ******************************************************************************/

import random
from copy import deepcopy

import numpy as np
import torch
from kornia.morphology import erosion
from monai.data import MetaTensor
from monai.transforms import Transform
from skimage.measure import label, regionprops
from skimage.morphology import disk


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

    def __call__(self, image: np.ndarray) -> np.ndarray:
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

    def __init__(
        self,
        image_key: str = "image",
        min_intensity: int = 0,
        max_intensity: int = 65535,
    ) -> None:
        """Constructor

        Parameters
        ----------

        image_key: str
            Key for the image in the data dictionary.
        min_intensity: int
            Minimum intensity to normalize against (optional, default = 0).
        max_intensity: int
            Maximum intensity to normalize against (optional, default = 65535).
        """
        super().__init__()
        self.image_key = image_key
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
        data[self.image_key] = (
            data[self.image_key] - self.min_intensity
        ) / self.range_intensity
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


class ToPyTorchLightningOutputd(Transform):
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
    """Standardize the passed tensor by subtracting the mean and dividing by the standard deviation."""

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
    """Standardize the "image" tensor by subtracting the mean and dividing by the standard deviation."""

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


class SelectPatchesByLabeld(Transform):
    """Pick labels at random and extract the requested window from both label and image."""

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        patch_size: tuple[int, int] = (128, 128),
        label_indx: int = 1,
        num_patches: int = 1,
        no_batch_dim: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------

        image_key: str
            Key for the image in the data dictionary.
        label_key: str
            Key for the label in the data dictionary.
        patch_size: tuple[int, int]
            Size of the window to be extracted centered on the selected label.
        num_patches: int
            Number of stacked labels to be returned. The Transform will sample with repetitions if the number of objects
            in the image with the requested label index is lower than the number of requested windows.
        no_batch_dim: bool
            If only one window is returned, the batch dimension can be omitted, to return a (channel, height, width)
            tensor. This ensures that the dataloader does not require a custom collate function to handle the double
            batching. If num_windows > 1, this parameter is ignored.
        """
        super().__init__()
        self.label_key = label_key
        self.image_key = image_key
        self.patch_size = patch_size
        self.label_indx = label_indx
        self.num_patches = num_patches
        self.no_batch_dim = no_batch_dim

    def __call__(self, data: dict) -> dict:
        """
        Select the requested number of labels from the label image, and extract region of defined window size for both
        image and label in stacked Tensors in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with extracted windows for the "image" and "label" tensors.
        """

        # Get the 2D label image to work with
        label_img = label(data[self.label_key][..., :, :].squeeze() == self.label_indx)
        sy, sx = label_img.shape

        # Get the window half-side lengths
        wy = self.patch_size[0] // 2
        wx = self.patch_size[1] // 2

        # Get the number of distinct objects in the image with the requested label index
        regions = regionprops(label_img)

        # Build a list of valid labels, by dropping all labels that are too close to the
        # borders to allow for extracting a valid window of requested size.
        valid_labels = []
        cached_regions = {}
        for region in regions:

            # Get the centroid
            cy, cx = region.centroid

            # Get the boundaries of the area to extract
            cy = round(cy)
            cx = round(cx)

            # Get the boundaries
            y0 = cy - wy
            y = cy + wy
            x0 = cx - wx
            x = cx + wx

            if y0 >= 0 and y < sy and x0 >= 0 and x < sx:
                # This window is completely contained in the image, keep (and cache) it
                valid_labels.append(region.label)
                cached_regions[region.label] = {"y0": y0, "y": y, "x0": x0, "x": x}

        # Number of valid labels in the image
        num_labels = len(valid_labels)

        # Get the range of labels to select from
        if num_labels >= self.num_patches:
            selected_labels = random.sample(valid_labels, k=self.num_patches)
        else:
            # Allow for repetitions
            selected_labels = random.choices(valid_labels, k=self.num_patches)

        # Initialize returned list with shallow copy to preserve key ordering
        ret: list = [dict(data) for _ in range(self.num_patches)]

        # Meta keys
        meta_keys = set(data.keys()).difference({self.label_key, self.image_key})

        # Deep copy all the unmodified data
        if len(meta_keys) > 0:
            for i in range(self.num_patches):
                for key in meta_keys:
                    ret[i][key] = deepcopy(data[key])

        # Extract the areas around the selected labels
        for i, selected_label in enumerate(selected_labels):
            # Extract the cached region boundaries
            region = cached_regions[selected_label]

            # Get the region
            y0 = region["y0"]
            y = region["y"]
            x0 = region["x0"]
            x = region["x"]

            # Store it
            ret[i][self.image_key] = data[self.image_key][..., y0:y, x0:x]
            ret[i][self.label_key] = data[self.label_key][..., y0:y, x0:x]

        # Return the new data
        return ret


class AddFFT2(Transform):
    """Calculates the power spectrum of the selected single-channel image and adds it as a second plane."""

    def __init__(self) -> None:
        """Constructor"""
        super().__init__()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: torch.Tensor
            Image with added normalized power spectrum as second plane.
        """
        # Make sure that the dimensions of the data are correct
        if not (data.dim() == 3 and data.shape[0] == 1):
            raise ValueError(
                "The image tensor must be of dimensions (1 x height x width)!"
            )

        # Calculate the power spectrum of the image
        f = torch.fft.fft2(data).abs()

        # Set the DC to 0.0
        f[0, 0, 0] = 0.0

        # Normalize
        f = (f - f.mean()) / f.std()

        # Add it as a new plane
        data = torch.cat((data, f), dim=0)

        # Return the updated tensor
        return data


class AddFFT2d(Transform):
    """Adds the FFT2 of a single-channel image as a second plane."""

    def __init__(
        self,
        image_key: str = "image",
    ) -> None:
        """Constructor

        Parameters
        ----------

        image_key: str
            Key for the image in the data dictionary.
        """
        super().__init__()
        self.image_key = image_key

    def __call__(self, data: dict) -> dict:
        """
        Calculates the power spectrum of the selected single-channel image and adds it as a second plane.

        Returns
        -------

        data: dict
            Updated dictionary with modified "image" tensor.
        """

        # Make sure that the dimensions of the data are correct
        if not (data[self.image_key].dim() == 3 and data[self.image_key].shape[0] == 1):
            raise ValueError(
                "The image tensor must be of dimensions (1 x height x width)!"
            )

        # Calculate the power spectrum of the image
        f = torch.fft.fft2(data[self.image_key]).abs()

        # Set the DC to 0.0
        f[0, 0, 0] = 0.0

        # Normalize
        f = (f - f.mean()) / f.std()

        # Add it as a new plane
        data[self.image_key] = torch.cat((data[self.image_key], f), dim=0)

        # Return the updated data dictionary
        return data


class AddBorderd(Transform):
    """Add a border class to the (binary) label image.

    Please notice that the border is obtained from the erosion of the objects (that is, it does not extend outside of the original connected components.
    """

    def __init__(self, label_key: str = "label", border_width: int = 3) -> None:
        """Constructor

        Parameters
        ----------

        label_key: str
            Key for the label image in the data dictionary.
        border_width: int
            Width of the border to be eroded from the binary image and added as new class.
        """
        super().__init__()
        self.label_key = label_key
        self.border_width = border_width

    def process_label(self, lbl, footprint):
        """Create and add object borders as new class."""
        eroded = erosion(lbl, footprint)
        border = torch.bitwise_or(lbl, eroded)
        return lbl + border  # Count border pixels twice

    def __call__(self, data: dict) -> dict:
        """
        Add  the requested number of labels from the label image, and extract region of defined window size for both image and label in stacked Tensors in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with new and "label" tensor.
        """

        # Make sure the label datatype is torch.int32
        if data["label"].dtype != torch.int32:
            data["label"] = data["label"].to(torch.int32)

        # Get the 2D label image to work with
        label_img = data["label"][..., :, :].squeeze()

        # Make sure there are only two classes: background and foreground
        if len(torch.unique(label_img)) > 2:
            raise ValueError("The label image(s) must be binary!")

        # Prepare the structuring element for the erosion
        footprint = torch.tensor(disk(self.border_width), dtype=torch.int32)

        # Number of dimensions in the data tensor
        num_dims = len(data["label"].shape)

        # Prepare the input
        if num_dims == 2:

            # Add batch and channel dimension
            label_batch = data["label"].unsqueeze(0).unsqueeze(0)

            # Process and update
            data["label"] = (
                self.process_label(label_batch, footprint).squeeze(0).squeeze(0)
            )

        elif num_dims == 3:
            # We have a channel dimension: make sure there only one channel!
            if data["label"].shape[0] > 1:
                raise ValueError("The label image must be binary!")

            # Add batch dimension
            label_batch = data["label"].unsqueeze(0)

            # Process and update
            data["label"][0, :, :] = self.process_label(label_batch, footprint).squeeze(
                0
            )

        elif num_dims == 4:

            # We have a batch. Make sure that there is only one channel!
            if data["label"][0].shape[0] > 1:
                raise ValueError("The label image must be binary!")

            # Process and update
            data["label"] = self.process_label(data["label"], footprint)

        else:
            raise ValueError('Unexpected number of dimensions for data["label"].')

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
        elif type(data) == dict:
            # A dictionary, most likely of "image" and "label"
            print(f"{prefix}Dictionary with keys: ", end=" ")
            for key in data.keys():
                value = data[key]
                t = type(value)
                if t is MetaTensor:
                    print(
                        f"'{key}': shape={data[key].shape}, dtype={data[key].dtype};",
                        end=" ",
                    )
                elif t is dict:
                    print(f"'{key}': dict;", end=" ")
                else:
                    print(f"'{key}': {t};", end=" ")
            print()
        elif type(data) == MetaTensor:
            print(f"{prefix}MONAI MetaTensor: shape={data.shape}, dtype={data.dtype}")
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

    def __init__(self, keys: tuple, class_num: int, min_fraction: float = 0.0):
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
                f"The number of voxels for {self.class_num} is lower than expected ({num_voxels})."
            )

        # Return data
        return data
