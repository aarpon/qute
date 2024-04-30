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

import numpy as np
import torch
from monai.transforms import MapTransform, Transform


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
