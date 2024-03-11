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

import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import monai.data
import numpy as np
import torch
from kornia.morphology import erosion
from monai.data import MetaTensor
from monai.transforms import MapTransform, Transform
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import disk
from tifffile import imread

from qute.transforms.util import scale_dist_transform_by_region


class CellposeLabelReader(Transform):
    """Loads a Cellpose label file and returns it as a NumPy array."""

    def __init__(self, to_int32: bool = True) -> None:
        """Constructor

        Parameters
        ----------

        to_int32: bool
         Set to True to convert to np.int32.

        Returns
        -------

        labels: np.ndarray
            Labels array.
        """
        super().__init__()
        self.to_int32 = to_int32

    def __call__(self, file_name: Union[Path, str]) -> np.ndarray:
        """
        Load the file and return the labels tensor.

        Parameters
        ----------

        file_name: str
            File name

        Returns
        -------

        labels: ndarray
            The labels array from the CellPose labels file.
        """
        data = np.load(Path(file_name).resolve(), allow_pickle=True)
        d = data[()]
        if self.to_int32:
            return d["masks"].astype(np.int32)
        else:
            return d["masks"]


class CustomTIFFReader(Transform):
    """Loads a TIFF file using the tifffile library."""

    def __init__(
        self,
        ensure_channel_first: bool = True,
        dtype: torch.dtype = torch.float32,
        as_meta_tensor: bool = False,
        pixdim: Optional[tuple] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------

        ensure_channel_first: bool
            Ensure that the image is in the channel first format.

        dtype: torch.dtype = torch.float32
            Type of the image.

        as_meta_tensor: bool (default = False)
            Set to True to return a MONAI MetaTensor, set to False for a PyTorch Tensor.

        pixdim: Optional[tuple]
            Set the voxel size (`pixdim`, in MONAI parlance) as metadata to the MetaTensor (only if `as_meta_tensor` is
            True; otherwise it is ignored).
        """
        super().__init__()
        self.ensure_channel_first = ensure_channel_first
        self.dtype = dtype
        self.as_meta_tensor = as_meta_tensor
        self.pixdim = tuple(pixdim) if pixdim is not None else None

    def __call__(self, file_name: Union[Path, str]) -> torch.Tensor:
        """
        Load the file and return the image/labels Tensor.

        Parameters
        ----------

        file_name: str
            File name

        Returns
        -------

        tensor: torch.Tensor | monai.MetaTensor
            Tensor with requested type and shape.
        """

        # File path
        image_path = str(Path(file_name).resolve())

        # Load and process image
        data = torch.Tensor(imread(image_path).astype(np.float32))
        if self.as_meta_tensor:
            if self.pixdim is not None:
                if data.ndim != len(self.pixdim):
                    raise ValueError(
                        "The size of `pixdim` does not natch the dimensionality of the image."
                    )
            else:
                self.pixdim = tuple((1.0 for _ in range(data.ndim)))
            data = MetaTensor(data, meta={"pixdim": self.pixdim})
        if self.ensure_channel_first:
            data = data.unsqueeze(0)
        if self.dtype is not None:
            data = data.to(self.dtype)
        return data


class CustomTIFFReaderd(MapTransform):
    """Loads a TIFF file using the tifffile library."""

    def __init__(
        self,
        keys: tuple[str] = ("image", "label"),
        ensure_channel_first: bool = True,
        dtype: torch.dtype = torch.float32,
        as_meta_tensor: bool = False,
        pixdim: Optional[tuple] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str]
            Keys for the data dictionary.

        ensure_channel_first: bool
            Ensure that the image is in the channel first format.

        dtype: torch.dtype = torch.float32
            Type of the image.

        as_meta_tensor: bool (default = False)
            Set to True to return a MONAI MetaTensor, set to False for a PyTorch Tensor.

        pixdim: Optional[tuple]
            Set the voxel size (`pixdim`, in MONAI parlance) as metadata to the MetaTensor (only if `as_meta_tensor` is
            True; otherwise it is ignored).
        """
        super().__init__(keys=keys)
        self.keys = keys
        self.ensure_channel_first = ensure_channel_first
        self.dtype = dtype
        self.as_meta_tensor = as_meta_tensor
        self.pixdim = pixdim

    def __call__(self, data: dict) -> dict:
        """
        Load the files and return the image and labels Tensors in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with normalized "image" tensor.
        """

        # Work on a copy of the dictionary
        d = dict(data)

        for key in self.keys:

            # Get arguments
            image_path = str(Path(d[key]).resolve())

            # Load and process images
            image = torch.Tensor(imread(image_path).astype(np.float32))
            if self.as_meta_tensor:
                if self.pixdim is not None:
                    if image.ndim != len(self.pixdim):
                        raise ValueError(
                            "The size of `pixdim` does not natch the dimensionality of the image."
                        )
                else:
                    self.pixdim = tuple((1.0 for _ in range(image.ndim)))
                image = MetaTensor(image, meta={"pixdim": self.pixdim})
            if self.ensure_channel_first:
                image = image.unsqueeze(0)
            if self.dtype is not None:
                image = image.to(self.dtype)
            d[key] = image  # (Meta)Tensor(image)

        return d


class MinMaxNormalize(Transform):
    """Normalize a tensor to [0, 1] using given min and max absolute intensities."""

    def __init__(
        self,
        min_intensity: float = 0.0,
        max_intensity: float = 65535.0,
        in_place: bool = True,
    ) -> None:
        """Constructor

        Parameters
        ----------

        min_intensity: float
            Minimum intensity to normalize against (optional, default = 0.0).
        max_intensity: float
            Maximum intensity to normalize against (optional, default = 65535.0).

        Returns
        -------

        norm: tensor
            Normalized tensor.
        """
        super().__init__()
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.range_intensity = self.max_intensity - self.min_intensity
        self.in_place = in_place

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `image`.

        Returns
        -------

        result: tensor
            A stack of images with the same width and height as `label` and with `num_classes` planes.
        """
        if not self.in_place:
            if isinstance(data, torch.Tensor):
                data = data.clone()
            elif isinstance(data, np.ndarray):
                data = data.copy()
            else:
                raise TypeError(
                    "Unsupported data type. Data should be a PyTorch Tensor or a NumPy array."
                )
        return (data - self.min_intensity) / self.range_intensity


class MinMaxNormalized(MapTransform):
    """Normalize the "image" tensor to [0, 1] using given min and max absolute intensities from the data dictionary."""

    def __init__(
        self,
        keys: tuple[str] = ("image", "label"),
        min_intensity: float = 0.0,
        max_intensity: float = 65535.0,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str]
            Keys for the data dictionary.
        min_intensity: float
            Minimum intensity to normalize against (optional, default = 0.0).
        max_intensity: float
            Maximum intensity to normalize against (optional, default = 65535.0).
        """
        super().__init__(keys=keys)
        self.keys = keys
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

        # Work on a copy of the input dictionary data
        d = dict(data)

        # Process the images
        for key in self.keys:
            d[key] = (d[key] - self.min_intensity) / self.range_intensity
        return d


class Scale(Transform):
    """Scale the image by a constant factor and optionally type-casts it."""

    def __init__(
        self,
        factor: float = 65535.0,
        dtype: torch.dtype = torch.int32,
        in_place: bool = True,
    ) -> None:
        """Constructor

        Parameters
        ----------

        factor: float
            Factor by which to scale the images (optional, default = 65535.0).
        dtype: torch.dtype
            Data type of the final image (optional, default = torch.int32).

        Returns
        -------

        norm: tensor
            Normalized tensor.
        """
        super().__init__()
        self.factor = factor
        self.dtype = dtype
        self.in_place = in_place

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `image`.

        Returns
        -------

        result: tensor
            Tensor or NumPy array with scaled intensities and type-cast.
        """
        if not self.in_place:
            if isinstance(data, torch.Tensor):
                data = data.clone()
            elif isinstance(data, np.ndarray):
                data = data.copy()
            else:
                raise TypeError(
                    "Unsupported data type. Data should be a PyTorch Tensor or a NumPy array."
                )

        # Process the images
        return (data * self.factor).to(self.dtype)


class Scaled(MapTransform):
    """Scale the images in the data dictionary by a constant factor and optionally type-casts them."""

    def __init__(
        self,
        keys: tuple[str] = ("image", "label"),
        factor: float = 65535.0,
        dtype: torch.dtype = torch.int32,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str]
            Keys for the data dictionary.
        factor: float
            Factor by which to scale the images (optional, default = 65535.0).
        dtype: torch.dtype
            Data type of the final image (optional, default = torch.int32).
        """
        super().__init__(keys=keys)
        self.keys = keys
        self.factor = factor
        self.dtype = dtype

    def __call__(self, data: dict) -> dict:
        """
        Apply the transform to the tensors in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with scaled images.
        """

        # Work on a copy of the input dictionary data
        d = dict(data)

        # Process the images
        for key in self.keys:
            d[key] = (d[key] * self.factor).to(self.dtype)
        return d


class ToLabel(Transform):
    """
    Converts Tensor from one-hot representation to 2D label image.
    """

    def __init__(self, dtype=torch.int32, in_place: bool = True):
        super().__init__()
        self.dtype = dtype
        self.in_place = in_place

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Transform one-hot Tensor to 2D label image."""
        if not self.in_place:
            if isinstance(data, torch.Tensor) or isinstance(
                data, monai.data.MetaTensor
            ):
                data = data.clone()
            elif isinstance(data, np.ndarray):
                data = data.copy()
            else:
                raise TypeError(
                    "Unsupported data type. Data should be a PyTorch Tensor or a NumPy array."
                )

        if data.ndim == 5:
            # 3D data: (B, C, Z, H, W)
            data = data.argmax(axis=1).type(self.dtype)
        elif data.ndim == 4:
            # 2D data: (B, C, H, W)
            data = data.argmax(axis=1).type(self.dtype)
        elif data.ndim == 3:
            # 2D data: (C, H, W)
            data = data.argmax(axis=0).type(self.dtype)
        else:
            raise ValueError(
                "The input tensor must be of size (B, C, Z, H, W), (B, C, H, W) or (C, H, W)."
            )

        return data


class ToPyTorchLightningOutputd(MapTransform):
    """
    Simple converter to pass from the dictionary output of Monai transforms to the expected (image, label) tuple used by PyTorch Lightning.
    """

    def __init__(
        self,
        image_key: str = "image",
        image_dtype: torch.dtype = torch.float32,
        label_key: str = "label",
        label_dtype: torch.dtype = torch.int32,
    ):
        super().__init__(keys=[image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.image_dtype = image_dtype
        self.label_dtype = label_dtype

    def __call__(self, data: dict) -> tuple:
        """Unwrap the dictionary."""

        # Work on a copy of the input dictionary data
        d = dict(data)

        return d[self.image_key].type(self.image_dtype), d[self.label_key].type(
            self.label_dtype
        )


class ZNormalize(Transform):
    """Standardize the passed tensor by subtracting the mean and dividing by the standard deviation."""

    def __init__(self, in_place: bool = True) -> None:
        """Constructor"""
        super().__init__()
        self.in_place = in_place

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: torch.Tensor
            Normalized tensor.
        """
        if not self.in_place:
            if isinstance(data, torch.Tensor):
                data = data.clone()
            elif isinstance(data, np.ndarray):
                data = data.copy()
            else:
                raise TypeError(
                    "Unsupported data type. Data should be a PyTorch Tensor or a NumPy array."
                )

        return (data - data.mean()) / data.std()


class ZNormalized(MapTransform):
    """Standardize the "image" tensor by subtracting the mean and dividing by the standard deviation."""

    def __init__(self, keys: tuple[str]) -> None:
        """Constructor"""
        super().__init__(keys=keys)
        self.keys = keys

    def __call__(self, data: dict) -> dict:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with normalized "image" tensor.
        """

        # Work on a copy of the input dictionary data
        d = dict(data)

        for key in self.keys:
            mn = d[key].mean()
            sd = d[key].std()
            d[key] = (d[key] - mn) / sd
        return d


class ClippedZNormalize(Transform):
    """Standardize the passed tensor by subtracting the mean and dividing by the standard deviation."""

    def __init__(
        self,
        mean: float,
        std: float,
        min_clip: float,
        max_clip: float,
        in_place: bool = True,
    ) -> None:
        """Constructor"""
        super().__init__()
        self.mean = mean
        self.std = np.max([std, 1e-8])
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.in_place = in_place

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: torch.Tensor
            Normalized tensor.
        """
        if not self.in_place:
            if isinstance(data, torch.Tensor):
                data = data.clone()
            elif isinstance(data, np.ndarray):
                data = data.copy()
            else:
                raise TypeError(
                    "Unsupported data type. Data should be a PyTorch Tensor or a NumPy array."
                )

        tmp = torch.clip(data, self.min_clip, self.max_clip)
        tmp = (tmp - self.mean) / self.std
        return tmp


class ClippedZNormalized(MapTransform):
    """Standardize the "image" tensor by subtracting the mean and dividing by the standard deviation."""

    def __init__(
        self,
        mean: float,
        std: float,
        min_clip: float,
        max_clip: float,
        image_key: str = "image",
    ) -> None:
        """Constructor"""
        super().__init__(keys=[image_key])
        self.image_key = image_key
        self.mean = mean
        self.std = np.max([std, 1e-8])
        self.min_clip = min_clip
        self.max_clip = max_clip

    def __call__(self, data: dict) -> dict:
        """
        Apply the transform to the "image" tensor in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with normalized "image" tensor.
        """

        # Work on a copy of the input dictionary data
        d = dict(data)

        d[self.image_key] = torch.clip(d[self.image_key], self.min_clip, self.max_clip)
        d[self.image_key] = (d[self.image_key] - self.mean) / self.std
        return d


class SelectPatchesByLabeld(MapTransform):
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
        super().__init__(keys=[image_key, label_key])
        self.label_key = label_key
        self.image_key = image_key
        self.patch_size = patch_size
        self.label_indx = label_indx
        self.num_patches = num_patches
        self.no_batch_dim = no_batch_dim

    def __call__(self, data: dict) -> list[dict]:
        """
        Select the requested number of labels from the label image, and extract region of defined window size for both
        image and label in stacked Tensors in the data dictionary.

        Returns
        -------

        data: list[dict]
            List of dictionaries with extracted windows for the "image" and "label" tensors.
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
    """Calculates the Fourier transform of the selected single-channel image and adds its z-normalized real
    and imaginary parts as two additional planes."""

    def __init__(
        self,
        mean_real: Optional[float] = None,
        std_real: Optional[float] = None,
        mean_imag: Optional[float] = None,
        std_imag: Optional[float] = None,
        in_place: bool = True,
    ) -> None:
        """Constructor"""
        super().__init__()
        self.mean_real = mean_real
        self.std_real = std_real
        self.mean_imag = mean_imag
        self.std_imag = std_imag
        self.in_place = in_place

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

        # Calculate the Fourier transform of the image
        f = torch.fft.fftshift(torch.fft.fft2(data))

        # Normalize
        if self.mean_real is not None and self.std_real is not None:
            f.real = (f.real - self.mean_real) / self.std_real
        else:
            f.real = (f.real - f.real.mean()) / f.real.std()
        if self.mean_imag is not None and self.std_imag is not None:
            f.imag = (f.imag - self.mean_imag) / self.std_imag
        else:
            f.imag = (f.imag - f.imag.mean()) / f.imag.std()

        # Do we modify the input Tensor in place?
        if not self.in_place:
            data = data.clone()

        # Add it as a new plane
        data = torch.cat((data, f.real, f.imag), dim=0)

        # Return the updated tensor
        return data


class AddFFT2d(MapTransform):
    """Calculates the Fourier transform of the selected single-channel image and adds its z-normalized real
    and imaginary parts as two additional planes."""

    def __init__(
        self,
        image_key: str = "image",
        mean_real: Optional[float] = None,
        std_real: Optional[float] = None,
        mean_imag: Optional[float] = None,
        std_imag: Optional[float] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------

        image_key: str
            Key for the image in the data dictionary.
        """
        super().__init__(keys=[image_key])
        self.image_key = image_key
        self.mean_real = mean_real
        self.std_real = std_real
        self.mean_imag = mean_imag
        self.std_imag = std_imag

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

        # Calculate the Fourier transform of the image
        f = torch.fft.fftshift(torch.fft.fft2(data[self.image_key]))

        # Normalize
        if self.mean_real is not None and self.std_real is not None:
            f.real = (f.real - self.mean_real) / self.std_real
        else:
            f.real = (f.real - f.real.mean()) / f.real.std()
        if self.mean_imag is not None and self.std_imag is not None:
            f.imag = (f.imag - self.mean_imag) / self.std_imag
        else:
            f.imag = (f.imag - f.imag.mean()) / f.imag.std()

        # Make a copy of the original dictionary
        d = dict(data)

        # Add it as a new plane
        d[self.image_key] = torch.cat((d[self.image_key], f.real, f.imag), dim=0)

        # Return the updated data dictionary
        return d


class AddBorderd(MapTransform):
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
        super().__init__(keys=[label_key])
        self.label_key = label_key
        self.border_width = border_width

    def process_label(self, lbl, footprint):
        """Create and add object borders as new class."""

        # Add the border as class 2
        eroded = erosion(lbl, footprint)
        border = lbl - eroded
        return lbl + border  # Count border pixels twice

    def __call__(self, data: dict) -> dict:
        """
        Add  the requested number of labels from the label image, and extract region of defined window size for both image and label in stacked Tensors in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with new and "label" tensor.
        """

        # Make a copy of the input dictionary
        d = dict(data)

        # Make sure the label datatype is torch.int32
        if type(d["label"]) is np.ndarray:
            d["label"] = torch.Tensor(d["label"].astype(np.int32)).to(torch.int32)

        if d["label"].dtype != torch.int32:
            d["label"] = d["label"].to(torch.int32)

        # Get the 2D label image to work with
        label_img = d["label"][..., :, :].squeeze()

        # Make sure there are only two classes: background and foreground
        if len(torch.unique(label_img)) > 2:
            label_img[label_img > 0] = 1

        # Prepare the structuring element for the erosion
        footprint = torch.tensor(disk(self.border_width), dtype=torch.int32)

        # Number of dimensions in the data tensor
        num_dims = len(d["label"].shape)

        # Prepare the input
        if num_dims == 2:

            # Add batch and channel dimension
            label_batch = d["label"].unsqueeze(0).unsqueeze(0)

            # Process and update
            d["label"] = (
                self.process_label(label_batch, footprint).squeeze(0).squeeze(0)
            )

        elif num_dims == 3:
            # We have a channel dimension: make sure there only one channel!
            if d["label"].shape[0] > 1:
                raise ValueError("The label image must be binary!")

            # Add batch dimension
            label_batch = d["label"].unsqueeze(0)

            # Process and update
            d["label"][0, :, :] = self.process_label(label_batch, footprint).squeeze(0)

        elif num_dims == 4:

            # We have a batch. Make sure that there is only one channel!
            if d["label"][0].shape[0] > 1:
                raise ValueError("The label image must be binary!")

            # Process and update
            d["label"] = self.process_label(d["label"], footprint)

        else:
            raise ValueError('Unexpected number of dimensions for data["label"].')

        return d


class AddNormalizedDistanceTransform(Transform):
    """Calculates and normalizes the distance transform per region of the selected pixel class from a labels image
    and adds it as an additional plane to the image."""

    def __init__(
        self,
        pixel_class: int = 1,
        reverse: bool = False,
        do_not_zero: bool = False,
        in_place: bool = True,
    ) -> None:
        """Constructor

        Parameters
        ----------

        pixel_class: int
            Class of the pixels to be used for calculating the distance transform.

        reverse: bool
            Whether to reverse the direction of the normalized distance transform: from 1.0 at the center of the
            objects and 0.0 at the periphery, to 0.0 at the center and 1.0 at the periphery.

        do_not_zero: bool (optional, default is False)
            This is only considered if `reverse` is True. Set to True not to allow that the center pixels in each
            region have an inverse distance transform of 0.0.

        in_place: bool (optional, default is True)
            Set to True to modify the Tensor in place.
        """
        super().__init__()
        self.pixel_class = pixel_class
        self.reverse = reverse
        self.do_not_zero = do_not_zero
        self.in_place = in_place

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        """
        Calculates and normalizes the distance transform per region of the selected pixel class from a labels image
        and adds it as an additional plane to the image.

        Returns
        -------

        data: np.ndarray
            Updated array with the normalized distance transform added as a new plane.
        """

        # This Transform works on NumPy arrays
        if data.dtype in [torch.float32, torch.int32]:
            data_label = np.array(data.cpu())
        else:
            data_label = data

        # Make sure that the dimensions of the data are correct
        if not (data_label.ndim in [3, 4] and data_label.shape[0] == 1):
            raise ValueError(
                "The image array must be of dimensions (1 x height x width)!"
            )

        # Binarize the labels
        bw = data_label > 0

        # Calculate distance transform
        dt = distance_transform_edt(bw, return_distances=True)

        # Normalize the distance transform by object in place
        dt = scale_dist_transform_by_region(
            dt,
            data_label,
            reverse=self.reverse,
            do_not_zero=self.do_not_zero,
            in_place=True,
        )

        # Add the scaled distance transform as a new channel to the input image
        data = torch.cat(
            (data, torch.tensor(dt).to(torch.float32)),
            dim=0,
        )

        # Return the updated data dictionary
        return data


class AddNormalizedDistanceTransformd(MapTransform):
    """Calculates and normalizes the distance transform per region from the labels image (from an instance segmentation)
    and adds it as an additional plane to the image."""

    def __init__(
        self,
        image_key: str = "image",
        label_key: str = "label",
        reverse: bool = False,
        do_not_zero: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------

        image_key: str
            Key for the image in the data dictionary.

        label_key: str
            Key for the label in the data dictionary.

        reverse: bool
            Whether to reverse the direction of the normalized distance transform: from 1.0 at the center of the
            objects and 0.0 at the periphery, to 0.0 at the center and 1.0 at the periphery.

        do_not_zero: bool (optional, default is False)
            This is only considered if `reverse` is True. Set to True not to allow that the center pixels in each
            region have an inverse distance transform of 0.0.
        """
        super().__init__(keys=[image_key, label_key])
        self.image_key = image_key
        self.label_key = label_key
        self.reverse = reverse
        self.do_not_zero = do_not_zero

    def __call__(self, data: dict) -> dict:
        """
        Calculates and normalizes the distance transform per region from the labels image (from an instance segmentation)
        and adds it as an additional plane to the image

        Returns
        -------

        data: dict
            Updated dictionary with modified "image" tensor.
        """

        # Make a copy of the input dictionary
        d = dict(data)

        # This Transform works on NumPy arrays
        if d["label"].dtype in [torch.float32, torch.int32]:
            data_label = np.array(d["label"].cpu())
        else:
            data_label = d[self.label_key]

        # Make sure that the dimensions of the data are correct
        if not (data_label.ndim in [3, 4] and data_label.shape[0] == 1):
            raise ValueError(
                "The image array must be of dimensions (1 { x depth} x height x width)!"
            )

        # Binarize the labels
        bw = data_label > 0

        # Calculate distance transform
        dt = distance_transform_edt(bw, return_distances=True)

        # Normalize the distance transform by object in place
        dt = scale_dist_transform_by_region(
            dt,
            data_label,
            reverse=self.reverse,
            do_not_zero=self.do_not_zero,
            in_place=True,
        )

        # Add the scaled distance transform as a new channel to the input image
        d[self.image_key] = torch.cat(
            (d[self.image_key], torch.tensor(dt).to(torch.float32)),
            dim=0,
        )

        # Return the updated data dictionary
        return d


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


class DebugMinNumVoxelCheckerd(MapTransform):
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
        super().__init__(keys=keys)
        self.keys = keys
        self.class_num = class_num
        self.min_fraction = min_fraction

    def __call__(self, data):
        """Call the Transform."""

        # Make a copy of the input dictionary
        d = dict(data)

        # Get the keys
        keys = d.keys()

        # Only work on the expected keys
        for key in keys:
            if key not in self.keys:
                continue

            # The Tensor should be in OneHot format, and the class number should
            # map to the index of the first dimension.
            if self.class_num > (d[key].shape[0] - 1):
                raise Exception("`num_class` is out of boundaries.")

            num_voxels = d[key][self.class_num].sum()
            if self.min_fraction == 0.0:
                if num_voxels > 0:
                    return d

            if self.min_fraction > 0.0:
                area = d[key].shape[1] * d[key].shape[2]
                if num_voxels / area >= self.min_fraction:
                    return d

            raise Exception(
                f"The number of voxels for {self.class_num} is lower than expected ({num_voxels})."
            )

        # Return data
        return d
