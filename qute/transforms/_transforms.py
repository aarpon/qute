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
from typing import Optional, Union

import monai.data
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform, Transform
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import ball, disk

from qute.transforms.util import (
    extract_subvolume,
    get_tensor_num_spatial_dims,
    insert_subvolume,
)


class LabelToTwoClassMask(Transform):
    """Maps a labels image to a two-class mask: object and border.

    Supported and expected are either 2D data of shape (C, H, W) or 3D
    data of shape (C, D, H, W).

    The alternative possibility of 2D batch data (B, C, H, W) can't robustly
    be distinguished from 3D (C, D, H, W) data and will result in unexpected
    labels.
    """

    def __init__(self, border_thickness: int = 1, drop_eroded: bool = False):
        """Constructor.

        Parameters
        ----------

        border_thickness: int = 1
            Thickness of the border to be created. Please notice that the border is obtained from
            the erosion of the objects (that is, it does not extend outside the original connected
            components.

        drop_eroded: bool = False
            Objects that are eroded away because smaller than the structuring element (e.g., flat
             in z direction) will have only border. Set `drop_eroded = True` to remove also the
             border.

        Please notice that if the input tensor is black-and-white, connected component analysis will be applied by
        the transform before creating the two classes. This may cause objects to fuse.
        """
        super().__init__()
        if border_thickness <= 0:
            raise ValueError("The border thickness cannot be zero!")
        self.border_thickness = border_thickness
        self.drop_eroded = drop_eroded

    def __call__(
        self, data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Maps a labels image to a two-class mask.

        Parameters
        ----------

        data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
            Input tensor

        Returns
        -------

        tensor: torch.Tensor | monai.MetaTensor
            Tensor as with two-class mask.
        """

        if not type(data) in [torch.Tensor, monai.data.MetaTensor, np.ndarray]:
            raise TypeError(f"Unsupported input type {type(data)}.")

        # Keep track of whether we are working with MONAI MetaTensor
        is_meta_tensor = type(data) is monai.data.MetaTensor

        # Get the number of spatial dimensions
        num_spatial_dims = get_tensor_num_spatial_dims(data, with_batch_dim=False)

        # Make sure the channel dimension has only one element
        if data.shape[0] != 1:
            raise ValueError("The input tensor must have only one channel!")

        # Footprint (structuring element) for erosion
        if num_spatial_dims == 2:
            footprint = torch.tensor(disk(self.border_thickness), dtype=torch.int32)
        else:
            footprint = torch.tensor(ball(self.border_thickness), dtype=torch.int32)

        # Original size
        original_size = data.shape

        # Make sure to work with int32 tensors
        if type(data) is np.ndarray:
            data = torch.tensor(data)
        data.to(torch.int32)

        # Remove singleton dimensions (in a copy)
        data = data.squeeze()

        # Allocate output (with the original shape)
        out = torch.zeros(original_size, dtype=torch.int32)

        # Make sure we have labels
        unique_labels = torch.unique(data)
        if len(unique_labels) == 2 and torch.all(unique_labels == torch.tensor([0, 1])):
            data_np = data.cpu().detach().numpy()
            labels_np = label(data_np, background=0).astype(np.int32)
            data = torch.from_numpy(labels_np)

        # Calculate bounding boxes
        regions = regionprops(data.numpy())

        # Process all labels serially
        for region in regions:

            if region.label == 0:
                continue

            # Extract the subvolume
            cropped_masks = extract_subvolume(data, region.bbox)

            # Select object
            mask = torch.tensor(cropped_masks == region.label, dtype=torch.int32)

            # Perform erosion
            eroded_mask = binary_erosion(mask, footprint)

            # Crete object and border
            border = mask - eroded_mask
            bw_tc = mask + border  # Count border pixels twice

            # Drop objects that have been eroded away?
            if (bw_tc == 1).sum() == 0 and self.drop_eroded:
                continue

            # Insert it into dt_out
            bbox = region.bbox
            while bw_tc.ndim < out.ndim:
                bw_tc = bw_tc[np.newaxis, :]
                m = len(bbox) // 2
                bbox = tuple([0] + list(bbox[:m]) + [1] + list(bbox[m:]))
            out = insert_subvolume(out, bw_tc, bbox, masked=True)

        # If needed, pack the result into a MetaTensor and
        # transfer the metadata dictionary.
        if is_meta_tensor:
            out = MetaTensor(out, meta=data.meta.copy())

        return out


class LabelToTwoClassMaskd(MapTransform):
    """Maps labels images to two-class masks.

    Supported and expected are either 2D data of shape (C, H, W) or 3D
    data of shape (C, D, H, W).

    The alternative possibility of 2D batch data (B, C, H, W) can't robustly
    be distinguished from 3D (C, D, H, W) data and will result in unexpected
    labels.
    """

    def __init__(
        self,
        keys: tuple[str] = ("image", "label"),
        border_thickness: int = 1,
        drop_eroded: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str]
            Keys for the data dictionary.

        border_thickness: int = 1
            Thickness of the border to be created. Please notice that the border is obtained from
            the erosion of the objects (that is, it does not extend outside the original connected
            components.

        drop_eroded: bool = False
            Objects that are eroded away because smaller than the structuring element (e.g., flat
             in z direction) will have only border. Set `drop_eroded = True` to remove also the
             border.
        """
        super().__init__(keys=keys)
        self.keys = keys
        self.border_thickness = border_thickness
        self.drop_eroded = drop_eroded

    def __call__(self, data: dict) -> dict:
        """
        Maps a labels image to a two-class mask.

        Returns
        -------

        data: dict
            Updated dictionary.
        """

        # Work on a copy of the input dictionary data
        d = dict(data)

        # Process the images
        for key in self.keys:
            transform = LabelToTwoClassMask(
                border_thickness=self.border_thickness, drop_eroded=self.drop_eroded
            )
            d[key] = transform(d[key])
        return d


class TwoClassMaskToLabel(Transform):
    """Maps a two-class (object and border) mask to labels image.

    Supported and expected are either 2D data of shape (C, H, W) or 3D
    data of shape (C, D, H, W).

    The alternative possibility of 2D batch data (B, C, H, W) can't robustly
    be distinguished from 3D (C, D, H, W) data and will result in unexpected
    labels.
    """

    def __init__(self, object_class: int = 1, border_thickness: int = 1):
        """Constructor.

        Parameters
        ----------

        object_class: int = 1
            Class of the object pixels. Usually, object pixels have class 1 and border pixels have class 0.

        border_thickness: Optional[int] = None
            Thickness of the border to be added. If set, it should be the same as the one used in
            LabelToTwoClassMask(d), but it is optional (and disabled by default).
        """
        super().__init__()
        self.object_class = object_class
        self.border_thickness = border_thickness

    def __call__(
        self, data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Maps a two-class mask to a labels image.

        Parameters
        ----------

        data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
            Input tensor

        Returns
        -------

        tensor: torch.Tensor | monai.MetaTensor
            Tensor as with label image.
        """

        if not type(data) in [torch.Tensor, monai.data.MetaTensor, np.ndarray]:
            raise TypeError(f"Unsupported input type {type(data)}.")

        # Keep track of whether we are working with MONAI MetaTensor
        is_meta_tensor = type(data) is monai.data.MetaTensor

        # Get the number of spatial dimensions
        num_spatial_dims = get_tensor_num_spatial_dims(data, with_batch_dim=False)

        # Make sure the channel dimension has only one element
        if data.shape[0] != 1:
            raise ValueError("The input tensor must have only one channel!")

        # Original shape
        original_shape = data.shape

        # Make sure to work with int32 tensors
        if type(data) is np.ndarray:
            data = torch.tensor(data)
        data.to(torch.int32)

        # Remove singleton dimensions (in a copy)
        data = data.squeeze()

        # Extract the pixels with selected class
        data_mask = data == self.object_class

        # Run a connected-component analysis on the mask
        labels = torch.tensor(label(data_mask, background=0), dtype=torch.int32)

        # Do we need to dilate?
        if self.border_thickness is not None and self.border_thickness > 0:

            # Allocate result
            labels_dilated = torch.zeros(data.shape, dtype=torch.int32)

            # Footprint (structuring element) for dilation
            if num_spatial_dims == 2:
                footprint = torch.tensor(disk(self.border_thickness), dtype=torch.int32)
            else:
                footprint = torch.tensor(ball(self.border_thickness), dtype=torch.int32)

            # Process all labels serially
            for lbl in torch.unique(labels):

                if lbl == 0:
                    continue

                # Select object
                mask = torch.tensor(labels == lbl, dtype=torch.int32)

                # Find the indices of the non-zero elements
                positions = np.where(mask > 0)
                if not positions[0].size:
                    raise ValueError("No connected components found.")

                # Determine bounds and apply borders for each dimension
                bounds = []
                for dim in range(num_spatial_dims):
                    min_bound = positions[dim].min() - self.border_thickness
                    max_bound = positions[dim].max() + self.border_thickness + 1
                    bounds.append((max(min_bound, 0), min(max_bound, mask.shape[dim])))

                # Unpack bounds to slicing format
                slices = tuple(
                    slice(min_bound, max_bound) for min_bound, max_bound in bounds
                )

                # Crop the (extended) mask
                cropped_mask = mask[slices]

                # Perform dilation
                dilated_mask = binary_dilation(cropped_mask, footprint)

                # Store in the output
                labels_dilated[slices][dilated_mask > 0] = lbl

            # Set labels to the dilated version
            labels = labels_dilated

        # Restore original shape
        while len(labels.shape) < len(original_shape):
            labels = labels.unsqueeze(0)

        # If needed, pack the result into a MetaTensor and
        # transfer the metadata dictionary.
        if is_meta_tensor:
            labels = MetaTensor(labels, meta=data.meta.copy())

        return labels


class TwoClassMaskToLabeld(MapTransform):
    """Maps a two-class (object and border) mask to labels image.

    Supported and expected are either 2D data of shape (C, H, W) or 3D
    data of shape (C, D, H, W).

    The alternative possibility of 2D batch data (B, C, H, W) can't robustly
    be distinguished from 3D (C, D, H, W) data and will result in unexpected
    labels.
    """

    def __init__(
        self,
        keys: tuple[str] = ("image", "label"),
        object_class: int = 1,
        border_thickness: int = 1,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str]
            Keys for the data dictionary.

        object_class: int = 1
            Class of the object pixels. Usually, object pixels have class 1 and border pixels have class 0.

        border_thickness: Optional[int] = None
            Thickness of the border to be added. If set, it should be the same as the one used in
            LabelToTwoClassMask(d), but it is optional (and disabled by default).
        """
        super().__init__(keys=keys)
        self.keys = keys
        self.object_class = object_class
        self.border_thickness = border_thickness

    def __call__(self, data: dict) -> dict:
        """
        Maps a labels image to a two-class mask.

        Returns
        -------

        data: dict
            Updated dictionary.
        """

        # Work on a copy of the input dictionary data
        d = dict(data)

        # Process the images
        for key in self.keys:
            transform = TwoClassMaskToLabel(
                object_class=self.object_class, border_thickness=self.border_thickness
            )
            d[key] = transform(d[key])
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
        keys: tuple[str, ...] = ("image", "label"),
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

    def __call__(self, data: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the transform to `image`.

        Returns
        -------

        result: Union[np.ndarray, torch.Tensor]
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


class OneHotToMask(Transform):
    """
    Converts Tensor from one-hot representation to 2D/3D mask image.
    Supported and expected are either 2D data of shape (C, H, W) or 3D
    data of shape (C, D, H, W).

    The alternative possibility of 2D batch data (B, C, H, W) can't robustly
    be distinguished from 3D (C, D, H, W) data and will result in unexpected
    labels.
    """

    def __init__(self, dtype=torch.int32):
        super().__init__()
        self.dtype = dtype

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Transform one-hot Tensor to 2D/3D label image."""
        if data.ndim in [3, 4]:
            # Either 2D data: (C, H, W) or 3D data: (C, D, H, W).
            # The alternative possibility of 2D batch data (B, C, H, W) is not supported
            # (but can't be distinguished) and will result in unexpected labels.
            return data.argmax(axis=0, keepdim=True).type(self.dtype)
        elif data.ndim == 5:
            # Unsupported >=3D batch data: (B, C, Z, H, W)
            raise ValueError(
                "(B, C, Z, H, W) dimensions not supported! Please pass independent tensors from a batch."
            )
        else:
            # Unclear dimensionality
            raise ValueError(
                "The input tensor must be of dimensions (C, D, H, W) or (C, H, W)."
            )


class OneHotToMaskBatch(Transform):
    """
    Converts batches of Tensors from one-hot representation to 2D/3D mask image.
    Supported and expected are either batches of 2D data of shape (B, C, H, W) or
    batches of 3D data of shape (B, C, D, H, W).

    The alternative possibility of single 3D data (C, D, H, W) can't robustly
    be distinguished from batched 2D (B, C, H, W) data and will result in unexpected
    labels.
    """

    def __init__(self, dtype=torch.int32):
        super().__init__()
        self.dtype = dtype

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Transform batches of one-hot Tensors to batches of 2D/3D label images."""
        if data.ndim in [4, 5]:
            # Either 2D batch data: (B, C, H, W) or 3D batch data: (B, C, D, H, W).
            # The alternative possibility of single 3D data (C, D, H, W) can't robustly
            # be distinguished from batched 2D (B, C, H, W) data and will result in unexpected
            # labels.
            return data.argmax(axis=1, keepdim=True).type(self.dtype)
        else:
            # Unclear dimensionality
            raise ValueError(
                "The batched input tensor must be of dimensions (B, C, D, H, W) or (B, C, H, W)."
            )


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


# class AddBorderd(MapTransform):
#     """Add a border class to the (binary) label image.
#
#     Please notice that the border is obtained from the erosion of the objects (that is, it does not extend outside
#     the original connected components.
#     """
#
#     def __init__(self, label_key: str = "label", border_width: int = 3) -> None:
#         """Constructor
#
#         Parameters
#         ----------
#
#         label_key: str
#             Key for the label image in the data dictionary.
#         border_width: int
#             Width of the border to be eroded from the binary image and added as new class.
#         """
#         super().__init__(keys=[label_key])
#         self.label_key = label_key
#         self.border_width = border_width
#
#     def __call__(self, data: dict) -> dict:
#         """
#         Add  the requested number of labels from the label image, and extract region of defined window size for both image and label in stacked Tensors in the data dictionary.
#
#         Returns
#         -------
#
#         data: dict
#             Updated dictionary with new and "label" tensor.
#         """
#
#         def process_label(lbl, footprint):
#             """Create and add object borders as new class."""
#
#             # Add the border as class 2
#             eroded = binary_erosion(lbl, footprint)
#             border = lbl - eroded
#             return lbl + border  # Count border pixels twice
#
#         # Make a copy of the input dictionary
#         d = dict(data)
#
#         # Make sure the label datatype is torch.int32
#         if type(d["label"]) is np.ndarray:
#             d["label"] = torch.Tensor(d["label"].astype(np.int32)).to(torch.int32)
#
#         if d["label"].dtype != torch.int32:
#             d["label"] = d["label"].to(torch.int32)
#
#         # Make sure there are only two classes: background and foreground
#         if len(torch.unique(d["label"])) > 2:
#             d["label"][d["label"] > 0] = 1
#
#         # Do we have a 2D or a 3D image? Extract the image or stack to work with and
#         # prepare the structuring element for erosion accordingly
#         if d["label"].ndim == 2:
#
#             # A plain 2D image
#             label_img = d["label"][..., :, :].squeeze()
#
#             # Create 2D disk footprint
#             footprint = torch.tensor(disk(self.border_width), dtype=torch.int32)
#
#             # Process and update
#             d["label"] = process_label(label_img, footprint)
#
#         elif d["label"].ndim == 3:
#
#             if d["label"].shape[0] == 1:
#
#                 # A 2D image with a channel or z dimension of 1
#                 label_img = d["label"][..., :, :].squeeze()
#
#                 # Create 2D disk footprint
#                 footprint = torch.tensor(disk(self.border_width), dtype=torch.int32)
#
#                 # Process and update
#                 d["label"] = process_label(label_img, footprint).unsqueeze(0)
#
#             else:
#
#                 # A 3D image with a z extent larger of 1
#                 label_img = d["label"]
#
#                 # Create 3D ball footprint
#                 footprint = torch.tensor(ball(self.border_width), dtype=torch.int32)
#
#                 # Process and update
#                 d["label"] = process_label(label_img, footprint)
#
#         elif d["label"].ndim == 4:
#
#             if d["label"].shape[0] > 1:
#                 raise ValueError("CDHW tensor must have only one channel!")
#
#             # A 3D image with a z extent larger of 1
#             label_img = d["label"][..., :, :].squeeze()
#
#             # Create 3D ball footprint
#             footprint = torch.tensor(ball(self.border_width), dtype=torch.int32)
#
#             # Process and update
#             d["label"] = process_label(label_img, footprint).unsqueeze(0)
#
#         else:
#             raise ValueError("Unsupported image dimensions!")
#
#         return d


class NormalizedDistanceTransform(Transform):
    """Calculates and normalizes the distance transform per region of the selected pixel class from a labels image."""

    def __init__(
        self,
        reverse: bool = False,
        do_not_zero: bool = False,
        in_place: bool = True,
        with_batch_dim: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------

        reverse: bool
            Whether to reverse the direction of the normalized distance transform: from 1.0 at the center of the
            objects and 0.0 at the periphery, to 0.0 at the center and 1.0 at the periphery.

        do_not_zero: bool (optional, default is False)
            This is only considered if `reverse` is True. Set to True not to allow that the center pixels in each
            region have an inverse distance transform of 0.0.

        in_place: bool (optional, default is True)
            Set to True to modify the Tensor in place.

        with_batch_dim: bool (Optional, default is False)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.
        """
        super().__init__()
        self.reverse = reverse
        self.do_not_zero = do_not_zero
        self.in_place = in_place
        self.with_batch_dim = with_batch_dim

    def _process_single(self, data_label):
        """Process a single image (of a potential batch)."""
        # Prepare data out
        dt_out = np.zeros(data_label.shape, dtype=np.float32)

        # Remove singleton dimensions (in a copy)
        data_label = data_label.copy().squeeze()

        # Make sure that the input is of integer type
        if not data_label.dtype.kind == "i":
            data_label = data_label.astype(np.int32)

        # Calculate bounding boxes
        regions = regionprops(data_label)

        # Process all labels serially
        for region in regions:

            if region.label == 0:
                continue

            # Extract the subvolume
            cropped_mask = extract_subvolume(data_label, region.bbox) > 0

            # Calculate distance transform
            dt_tmp = distance_transform_edt(cropped_mask, return_distances=True)

            # Normalize the distance transform in place
            in_mask_indices = dt_tmp > 0.0
            if self.reverse:
                # Reverse the direction of the distance transform: make sure to stretch
                # the maximum to 1.0; we can keep a minimum larger than 0.0 in the center.
                if self.do_not_zero:
                    # Do not set the distance at the center to 0.0; the gradient is
                    # slightly lower, depending on the original range.
                    tmp = dt_tmp[in_mask_indices]
                    tmp = (tmp.max() + 1) - tmp
                    dt_tmp[in_mask_indices] = tmp / tmp.max()
                else:
                    # Plain linear inverse
                    min_value = dt_tmp[in_mask_indices].min()
                    max_value = dt_tmp[in_mask_indices].max()
                    dt_tmp[in_mask_indices] = (dt_tmp[in_mask_indices] - max_value) / (
                        min_value - max_value
                    )
            else:
                dt_tmp[in_mask_indices] = dt_tmp[in_mask_indices] / dt_tmp.max()

            # Insert it into dt_out
            bbox = region.bbox
            while dt_tmp.ndim < dt_out.ndim:
                dt_tmp = dt_tmp[np.newaxis, :]
                m = len(bbox) // 2
                bbox = tuple([0] + list(bbox[:m]) + [1] + list(bbox[m:]))
            dt_out = insert_subvolume(dt_out, dt_tmp, bbox)

        return dt_out

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        """
        Calculates and normalizes the distance transform per region of the selected pixel class from a labels image
        and adds it as an additional plane to the image.

        Returns
        -------

        data: np.ndarray
            Updated array with the normalized distance transform added as a new plane.
        """

        if not type(data) in [torch.Tensor, monai.data.MetaTensor, np.ndarray]:
            raise TypeError(f"Unsupported input type {type(data)}.")

        # Do we have a 2D or 3D tensor (excluding batch and channel dimensions)?
        effective_dims = get_tensor_num_spatial_dims(data, self.with_batch_dim)

        if effective_dims not in [2, 3]:
            raise ValueError("Unsupported geometry.")

        # Do we have a NumPy array?
        is_np_array = type(data) is np.ndarray

        # Keep track of whether we are working with MONAI MetaTensor
        is_meta_tensor = False
        meta = None
        if not is_np_array:
            is_meta_tensor = type(data) is monai.data.MetaTensor
            if is_meta_tensor:
                meta = data.meta.copy()

        # This Transform works on NumPy arrays
        if not is_np_array:
            data_label = np.array(data.cpu())
        else:
            data_label = data

        if self.with_batch_dim:
            dt_final = np.zeros(data_label.shape, dtype=np.float32)
            for b in range(data_label.shape[0]):
                dt_final[b] = self._process_single(data_label[b])
        else:
            dt_final = self._process_single(data_label)

        # Cast to a tensor
        dt = torch.from_numpy(dt_final)
        if is_meta_tensor:
            dt = MetaTensor(dt, meta=meta)

        # Return the updated data dictionary
        return dt


class NormalizedDistanceTransformd(MapTransform):
    """Calculates and normalizes the distance transform per region from the labels image (from an instance segmentation)."""

    def __init__(
        self,
        keys: tuple[str, ...] = ("label",),
        reverse: bool = False,
        do_not_zero: bool = False,
        with_batch_dim: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str, ...]
            Keys fot the tensor to be transformed. This transform makes sense for label or mask images only.

        reverse: bool
            Whether to reverse the direction of the normalized distance transform: from 1.0 at the center of the
            objects and 0.0 at the periphery, to 0.0 at the center and 1.0 at the periphery.

        do_not_zero: bool (optional, default is False)
            This is only considered if `reverse` is True. Set to True not to allow that the center pixels in each
            region have an inverse distance transform of 0.0.

        with_batch_dim: bool (Optional, default is False)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.
        """
        super().__init__(keys=keys)
        self.keys = keys
        self._transform = NormalizedDistanceTransform(
            reverse=reverse, do_not_zero=do_not_zero, with_batch_dim=with_batch_dim
        )

    def __call__(self, data: dict) -> dict:
        """
        Calculates and normalizes the distance transform per region from the labels image (from an instance segmentation).

        Returns
        -------

        data: dict
            Updated dictionary with modified tensors.
        """

        # Make a copy of the input dictionary
        d = dict(data)

        for key in self.keys:
            dt = self._transform(d[key])
            d[key] = dt

        # Return the updated data dictionary
        return d
