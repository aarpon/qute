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

import pathlib
from pathlib import Path
from typing import NamedTuple, Optional, Union

import monai.data
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.transforms import MapTransform, Transform
from nd2reader import ND2Reader
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import ball, disk
from tifffile import imread

from qute.transforms.util import (
    get_tensor_num_spatial_dims,
    scale_dist_transform_by_region,
)


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


class CustomND2Reader(Transform):
    """Loads a Nikon ND2 file using the nd2reader library."""

    def __init__(
        self,
        ensure_channel_first: bool = True,
        dtype: torch.dtype = torch.float32,
        as_meta_tensor: bool = False,
        voxel_size: Optional[tuple] = None,
        voxel_size_from_file: bool = False,
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
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.

        voxel_size: Optional[tuple]
            Set the voxel size [y, x, z] as metadata to the MetaTensor (only if `as_meta_tensor` is
            True; otherwise it is ignored).
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.

        voxel_size_from_file: Optional[tuple]
            If voxel size is not specified, as_meta_tensor is set to True, and set voxel_size_from_file is True
            the voxel size is read from the file and set it [y, x, z] as metadata to the MetaTensor.
            If both `as_meta_tensor` is True and either `voxel_size` is specified or voxel_size_from_file
            is True, then `ensure_channel_first` must also be True.
        """
        super().__init__()
        self.ensure_channel_first = ensure_channel_first
        self.dtype = dtype
        self.as_meta_tensor = as_meta_tensor
        self.voxel_size = tuple(voxel_size) if voxel_size is not None else None
        self.voxel_size_from_file = voxel_size_from_file

    def _parse_voxel_sizes(self, reader, meta) -> tuple[float, ...]:
        """
        Parse metadata of ND2 file and extracts pixel size and z step.
        """

        # Build the array step by step
        voxel_sizes = [0.0, 0.0, 0.0]  # y, x, z

        if "pixel_microns" in reader.metadata:
            p = reader.metadata["pixel_microns"]
            voxel_sizes[0] = p
            voxel_sizes[1] = p

        # If there is only one plane, we leave the z step to 0.0
        z_coords = None
        if meta.num_planes > 1:
            if (
                "z_coordinates" in reader.metadata
                and reader.metadata["z_coordinates"] is not None
            ):
                z_coords = np.array(reader.metadata["z_coordinates"])
            elif (
                hasattr(reader.parser._raw_metadata, "z_data")
                and reader.parser._raw_metadata.z_data is not None
            ):
                z_coords = np.array(reader.parser._raw_metadata.z_data)
            else:
                print("Could not read z coordinates!")

        if z_coords is not None:
            z_steps = np.zeros(meta.num_series * meta.num_timepoints)
            for i, z in enumerate(range(0, len(z_coords), meta.num_planes)):
                z_range = z_coords[z : z + meta.num_planes]
                z_steps[i] = np.mean(np.diff(z_range))

            voxel_sizes[2] = z_steps.mean()

        # Return the voxel sizes
        return tuple(voxel_sizes)

    def _parse_geometry(self, reader):
        """
        Parse geometry of ND2 file and sets `bundle_axis` and `_iter_axis` properties of ND2Reader.
        """

        # Initialize _geometry
        num_series: int = 1
        num_timepoints: int = 1
        num_channels: int = 1
        num_planes: int = 1
        geometry: str = "xy"
        iter_axis: str = ""

        class Meta(NamedTuple):
            geometry: str
            num_planes: int
            num_channels: int
            num_timepoints: int
            num_series: int
            iter_axis: str

        if "z" in reader.sizes and reader.sizes["z"] > 1:
            num_planes = reader.sizes["z"]
            geometry = "z" + geometry
        if "c" in reader.sizes and reader.sizes["c"] > 1:
            num_channels = reader.sizes["c"]
            geometry = "c" + geometry
        if "t" in reader.sizes and reader.sizes["t"] > 1:
            num_timepoints = reader.sizes["t"]
            geometry = "t" + geometry

        reader.bundle_axes = geometry

        # Axis to iterate upon
        if "z" in reader.sizes:
            reader.iter_axes = ["z"]
        if "c" in reader.sizes:
            reader.iter_axes = ["c"]
        if "t" in reader.sizes:
            reader.iter_axes = ["t"]
        if "v" in reader.sizes:
            self._num_series = reader.sizes["v"]
            reader.iter_axes = ["v"]

        # Set the main axis for iteration
        iter_axis = reader.iter_axes[0]

        # Fill the metadata tuple
        meta = Meta(
            geometry=geometry,
            num_planes=num_planes,
            num_channels=num_channels,
            num_timepoints=num_timepoints,
            num_series=num_series,
            iter_axis=iter_axis,
        )

        return meta

    def __call__(
        self,
        file_name: Union[Path, str],
        series_num: int = 0,
        timepoint: int = 0,
        channel: int = 0,
        plane: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load the file and return the image/labels Tensor.

        Parameters
        ----------

        file_name: str
            File name

        series_num: int = 0
            Number of series to read.

        timepoint: int = 0
            Timepoint to read.

        channel: int = 0
            Channel to read.

        plane: Optional[int] = None
            Plane to read. If not specified, and the dataset is 3D, the whole stack will be returned.

        Returns
        -------

        tensor: torch.Tensor | monai.MetaTensor
            Tensor with requested type and shape.
        """

        # Check the consistency of the input arguments
        if self.as_meta_tensor and self.voxel_size is not None:
            if not self.ensure_channel_first:
                raise ValueError(
                    f"If both `as_meta_tensor` is True and `voxel_size` is specified,"
                    f"`ensure_channel_first` must be True."
                )

        # File path
        file_path = str(Path(file_name).resolve())

        # Load and process image
        reader = ND2Reader(file_path)
        meta = self._parse_geometry(reader)

        # Check the compatibility of the requested dimensions
        if series_num + 1 > meta.num_series:
            raise ValueError("The requested series number does not exist.")

        if timepoint + 1 > meta.num_timepoints:
            raise ValueError("The requested time point does not exist.")

        if channel + 1 > meta.num_channels:
            raise ValueError("The requested channel does not exist.")

        if plane is not None and plane + 1 > meta.num_planes:
            raise ValueError("The requested plane does not exist.")

        match meta.geometry:
            case "xy":
                data = torch.Tensor(reader[0].astype(np.float32))
            case _:
                raise NotImplementedError(
                    "Support for this geometry is not implemented yet."
                )

        if self.as_meta_tensor:
            if self.voxel_size is None and self.voxel_size_from_file:
                # Get the voxel size
                self.voxel_size = self._parse_voxel_sizes(reader, meta)
            else:
                self.voxel_size = tuple([1.0, 1.0, 1.0])

            # To pass a voxel size to the MetaTensor, we need to define the
            # corresponding affine transform:
            affine = torch.tensor(
                [
                    [self.voxel_size[0], 0.0, 0.0, 0.0],
                    [0.0, self.voxel_size[1], 0.0, 0.0],
                    [0.0, 0.0, self.voxel_size[2], 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            data = MetaTensor(data, affine=affine)

        if self.ensure_channel_first:
            data = data.unsqueeze(0)
        if self.dtype is not None:
            data = data.to(self.dtype)
        return data


class CustomND2Readerd(MapTransform):
    """Loads TIFF files using the tifffile library."""

    def __init__(
        self,
        keys: tuple[str, ...] = ("image", "label"),
        ensure_channel_first: bool = True,
        dtype: torch.dtype = torch.float32,
        as_meta_tensor: bool = False,
        voxel_size: Optional[tuple] = None,
        voxel_size_from_file: bool = False,
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
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.

        voxel_size: Optional[tuple]
            Set the voxel size [y, x, z] as metadata to the MetaTensor (only if `as_meta_tensor` is
            True; otherwise it is ignored).
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.

        voxel_size_from_file: Optional[tuple]
            If voxel size is not specified, as_meta_tensor is set to True, and set voxel_size_from_file is True
            the voxel size is read from the file and set it [y, x, z] as metadata to the MetaTensor.
            If both `as_meta_tensor` is True and either `voxel_size` is specified or voxel_size_from_file
            is True, then `ensure_channel_first` must also be True.
        """
        super().__init__(keys=keys)
        self.keys = keys
        self.tensor_reader = CustomND2Reader(
            ensure_channel_first=ensure_channel_first,
            dtype=dtype,
            as_meta_tensor=as_meta_tensor,
            voxel_size=voxel_size,
            voxel_size_from_file=voxel_size_from_file,
        )

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

            # Use the single tensor reader
            out = self.tensor_reader(image_path)

            # Assign the tensor to the corresponding key
            d[key] = out  # (Meta)Tensor(image)

        return d


class CustomTIFFReader(Transform):
    """Loads a TIFF file using the tifffile library."""

    def __init__(
        self,
        ensure_channel_first: bool = True,
        dtype: torch.dtype = torch.float32,
        as_meta_tensor: bool = False,
        voxel_size: Optional[tuple] = None,
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
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.

        voxel_size: Optional[tuple]
            Set the voxel size [y, x, z] as metadata to the MetaTensor (only if `as_meta_tensor` is
            True; otherwise it is ignored).
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.

        """
        super().__init__()
        self.ensure_channel_first = ensure_channel_first
        self.dtype = dtype
        self.as_meta_tensor = as_meta_tensor
        self.voxel_size = tuple(voxel_size) if voxel_size is not None else None

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

        # Check the consistency of the input arguments
        if self.as_meta_tensor and self.voxel_size is not None:
            if not self.ensure_channel_first:
                raise ValueError(
                    f"If both `as_meta_tensor` is True and `voxel_size` is specified,"
                    f"`ensure_channel_first` must be True."
                )

        # File path
        image_path = str(Path(file_name).resolve())

        # Load and process image
        data = torch.Tensor(imread(image_path).astype(np.float32))
        if self.as_meta_tensor:
            if self.voxel_size is not None:
                if data.ndim != len(self.voxel_size):
                    raise ValueError(
                        "The size of `voxel_size` does not natch the dimensionality of the image."
                    )
            else:
                self.voxel_size = tuple([1.0, 1.0, 1.0])

            # To pass a voxel size to the MetaTensor, we need to define the
            # corresponding affine transform:
            affine = torch.tensor(
                [
                    [self.voxel_size[0], 0.0, 0.0, 0.0],
                    [0.0, self.voxel_size[1], 0.0, 0.0],
                    [0.0, 0.0, self.voxel_size[2], 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            data = MetaTensor(data, affine=affine)

        if self.ensure_channel_first:
            data = data.unsqueeze(0)
        if self.dtype is not None:
            data = data.to(self.dtype)
        return data


class CustomTIFFReaderd(MapTransform):
    """Loads TIFF files using the tifffile library."""

    def __init__(
        self,
        keys: tuple[str, ...] = ("image", "label"),
        ensure_channel_first: bool = True,
        dtype: torch.dtype = torch.float32,
        as_meta_tensor: bool = False,
        voxel_size: Optional[tuple] = None,
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
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.

        voxel_size: Optional[tuple]
            Set the voxel size [y, x, z] as metadata to the MetaTensor (only if `as_meta_tensor` is
            True; otherwise it is ignored).
            If both `as_meta_tensor` is True and `voxel_size` is specified, `ensure_channel_first`
            must be True.
        """
        super().__init__(keys=keys)
        self.keys = keys
        self.tensor_reader = CustomTIFFReader(
            ensure_channel_first=ensure_channel_first,
            dtype=dtype,
            as_meta_tensor=as_meta_tensor,
            voxel_size=voxel_size,
        )

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

            # Use the single tensor reader
            out = self.tensor_reader(image_path)

            # Assign the tensor to the corresponding key
            d[key] = out  # (Meta)Tensor(image)

        return d


class CustomResampler(Transform):
    """Resamples a tensor to a new voxel size.

    Accepted geometries are (C, D, H, W) for 3D and (C, H, W) for 2D.
    """

    def __init__(
        self,
        target_voxel_size: tuple[float, ...],
        input_voxel_size: Optional[tuple[float, ...]] = None,
        mode: str = "nearest",
        with_batch_dim: bool = False,
    ):
        """Constructor

        Parameters
        ----------

        target_voxel_size: tuple[float, ...]
            Target voxel size (z, y, x). For 2D data, set z = 1.

        input_voxel_size: Optional[tuple[float, ...]]
            Input voxel size. If omitted, and if the input tensor is a MONAI MetaTensor,
            the voxel size will be extracted from the tensor metadata (see CustomTiffReader).
            Please notice that only the scaling components of the affine matrix will be
            considered; all others will be silently ignored.

        mode: str (Optional, default is "neareast")
            Interpolation mode: one of "nearest", or "bilinear" (for 2D data) and "trilinear" for 3D data.

        with_batch_dim: bool (Optional, default is False)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.
        """
        super().__init__()
        self.target_voxel_size = target_voxel_size
        self.input_voxel_size = input_voxel_size
        self.mode = mode
        self.with_batch_dim = with_batch_dim

    def __call__(
        self, data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Resample the tensor and return it.

         Parameters
         ----------

         data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
             Input tensor.

         Returns
         -------

         tensor: torch.Tensor | monai.MetaTensor
             Tensor with requested type and shape.
        """

        # Keep track of whether we are working with MONAI MetaTensor
        is_meta_tensor = type(data) is monai.data.MetaTensor

        # If input_voxel_size is not set, and we have a MetaTensor, let's
        # try to extract the calibration from the metadata
        if self.input_voxel_size is None and is_meta_tensor:
            if hasattr(data, "affine"):
                if data.affine.shape == (4, 4):
                    self.input_voxel_size = (
                        float(data.affine[0, 0]),
                        float(data.affine[1, 1]),
                        float(data.affine[2, 2]),
                    )

        # If input_voxel_size is still None, raise an exception
        if self.input_voxel_size is None:
            raise ValueError("Please specify `input_voxel_size`.")

        # Do we have a 2D or 3D tensor (excluding batch and channel dimensions)?
        effective_dims = get_tensor_num_spatial_dims(data, self.with_batch_dim)

        if effective_dims not in [2, 3]:
            raise ValueError("Unsupported geometry.")

        # Is the mode correct?
        self.mode = self.mode.lower()
        if self.mode not in ["nearest", "bilinear", "trilinear"]:
            raise ValueError(f"Unexpected interpolation mode {self.mode}")

        # Make sure that the mode matches the geometry
        if self.mode != "nearest":
            if effective_dims == 2 and self.mode == "trilinear":
                self.mode = "bilinear"
                print("Changed `trilinear` to `bilinear` for 2D data.")
            elif effective_dims == 3 and self.mode == "bilinear":
                self.mode = "trilinear"
                print("Changed `bilinear` to `trilinear` for 3D data.")
            elif effective_dims not in [2, 3]:
                raise ValueError("Unsupported geometry.")

        # Do we have a NumPy array?
        if type(data) is np.ndarray:
            data = torch.from_numpy(data)

        # Calculate the output shape based on the ratio of voxel sizes
        ratios = np.array(self.input_voxel_size) / np.array(self.target_voxel_size)
        output_size = np.round(
            data.shape[-effective_dims:] * ratios[-effective_dims:]
        ).astype(int)
        output_size[output_size == 0.0] = 1.0
        output_size = tuple(output_size)

        # Prepare the arguments for the transformation: torch.nn.functional.interpolate expects the batch dimension
        if is_meta_tensor:
            data = data.as_tensor()

        if not self.with_batch_dim:
            data = data.unsqueeze(0)

        # Make sure the tensor is not of integer type, or the interpolation will fail
        current_dtype = data.dtype
        convert_back = False
        if data.dtype in {
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            data = data.to(torch.float32)
            convert_back = True

        # Run the interpolation
        if self.mode == "nearest":
            data = F.interpolate(
                data,
                size=output_size,
                mode=self.mode,
            )
        else:
            data = F.interpolate(
                data, size=output_size, mode=self.mode, align_corners=False
            )

        # If necessary, convert back
        if convert_back:
            data = data.to(current_dtype)

        # Remove the batch dimension if needed
        if not self.with_batch_dim:
            data = data.squeeze(0)

        # In case of a MONAI MetaTensor, we update the metadata
        if is_meta_tensor:
            affine = torch.tensor(
                [
                    [self.target_voxel_size[0], 0.0, 0.0, 0.0],
                    [0.0, self.target_voxel_size[1], 0.0, 0.0],
                    [0.0, 0.0, self.target_voxel_size[2], 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            data = MetaTensor(data, affine=affine)

        return data


class CustomResamplerd(MapTransform):
    """Resamples a tensor to a new voxel size.

    Accepted geometries are (C, D, H, W) for 3D and (C, H, W) for 2D.
    """

    def __init__(
        self,
        keys: tuple[str, ...],
        target_voxel_size: tuple[float, ...],
        input_voxel_size: Optional[tuple[float, ...]] = None,
        mode: tuple[str, ...] = ("trilinear", "nearest"),
    ):
        """Constructor

        Parameters
        ----------

        keys: tuple[str]
            Keys for the data dictionary.

        target_voxel_size: tuple[float, ...]
            Target voxel size (z, y, x). For 2D data, set z = 1.

        input_voxel_size: Optional[tuple[float, ...]]
            Input voxel size. If omitted, and if the input tensor is a MONAI MetaTensor,
            the voxel size will be extracted from the tensor metadata (see CustomTiffReader).
            Please notice that only the scaling components of the affine matrix will be
            considered; all others will be silently ignored.

        mode: str (Optional, default is "neareast")
            Interpolation mode: one of "nearest", or "bilinear" (for 2D data) and "trilinear" for 3D data.

        """
        super().__init__(keys=keys)
        self.keys = keys
        self.target_voxel_size = target_voxel_size
        self.input_voxel_size = input_voxel_size
        self.mode = mode

    def __call__(
        self, data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Resample the tensor and return it.

         Parameters
         ----------

         data: Union[torch.tensor, monai.data.MetaTensor, np.ndarray]
             Input tensor

         Returns
         -------

         tensor: torch.Tensor | monai.MetaTensor
             Resampled tensor.
        """

        # Make sure that we have the same number of keys and modes of interpolation
        if len(self.keys) != len(self.mode):
            raise ValueError("Number of keys and interpolation modes do not match!")

        for key, mode in zip(self.keys, self.mode):
            interpolator = CustomResampler(
                target_voxel_size=self.target_voxel_size,
                input_voxel_size=self.input_voxel_size,
                mode=mode,
            )
            data[key] = interpolator(data[key])

        return data


class LabelToTwoClassMask(Transform):
    """Maps a labels image to a two-class mask."""

    def __init__(self, border_thickness: int = 1):
        """Constructor"""
        super().__init__()
        self.border_thickness = border_thickness

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

        # Footprint (structuring element) for erosion
        if num_spatial_dims == 2:
            footprint = torch.tensor(disk(self.border_thickness), dtype=torch.int32)
        else:
            footprint = torch.tensor(ball(self.border_thickness), dtype=torch.int32)

        # Make sure to bring the structuring element to the same
        # number of dimensions as the data
        for d in range(len(data.shape) - num_spatial_dims):
            footprint = footprint.unsqueeze(0)

        # Make sure to work with int32 tensors
        if type(data) is np.ndarray:
            data = torch.tensor(data)
        data.to(torch.int32)

        # Allocate output
        out = torch.zeros(data.shape, dtype=data.dtype)

        # Process all individual labels
        for lbl in data.unique():

            # Ignore background
            if lbl == 0:
                continue

            # Prepare the label for processing
            bw = torch.zeros(data.shape, dtype=data.dtype)
            bw[data == lbl] = 1

            # Perform erosion
            eroded_bw = binary_erosion(bw, footprint)

            # Store object and border it in the output
            out[bw == 1] = 2
            out[eroded_bw == 1] = 1

        # If needed, pack the result into a MetaTensor and
        # transfer the metadata dictionary.
        if is_meta_tensor:
            out = MetaTensor(out, meta=data.meta.copy())

        return out


class LabelToTwoClassMaskd(MapTransform):
    """Maps a labels image to a two-class mask."""

    def __init__(
        self,
        keys: tuple[str] = ("image", "label"),
        border_thickness: int = 1,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str]
            Keys for the data dictionary.
        border_thickness: int = 1
            Thickness of the border to add.
        """
        super().__init__(keys=keys)
        self.keys = keys
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
            transform = LabelToTwoClassMask(border_thickness=self.border_thickness)
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


class AddBorderd(MapTransform):
    """Add a border class to the (binary) label image.

    Please notice that the border is obtained from the erosion of the objects (that is, it does not extend outside
    the original connected components.
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

    def __call__(self, data: dict) -> dict:
        """
        Add  the requested number of labels from the label image, and extract region of defined window size for both image and label in stacked Tensors in the data dictionary.

        Returns
        -------

        data: dict
            Updated dictionary with new and "label" tensor.
        """

        def process_label(lbl, footprint):
            """Create and add object borders as new class."""

            # Add the border as class 2
            eroded = binary_erosion(lbl, footprint)
            border = lbl - eroded
            return lbl + border  # Count border pixels twice

        # Make a copy of the input dictionary
        d = dict(data)

        # Make sure the label datatype is torch.int32
        if type(d["label"]) is np.ndarray:
            d["label"] = torch.Tensor(d["label"].astype(np.int32)).to(torch.int32)

        if d["label"].dtype != torch.int32:
            d["label"] = d["label"].to(torch.int32)

        # Make sure there are only two classes: background and foreground
        if len(torch.unique(d["label"])) > 2:
            d["label"][d["label"] > 0] = 1

        # Do we have a 2D or a 3D image? Extract the image or stack to work with and
        # prepare the structuring element for erosion accordingly
        if d["label"].ndim == 2:

            # A plain 2D image
            label_img = d["label"][..., :, :].squeeze()

            # Create 2D disk footprint
            footprint = torch.tensor(disk(self.border_width), dtype=torch.int32)

            # Process and update
            d["label"] = process_label(label_img, footprint)

        elif d["label"].ndim == 3:

            if d["label"].shape[0] == 1:

                # A 2D image with a channel or z dimension of 1
                label_img = d["label"][..., :, :].squeeze()

                # Create 2D disk footprint
                footprint = torch.tensor(disk(self.border_width), dtype=torch.int32)

                # Process and update
                d["label"] = process_label(label_img, footprint).unsqueeze(0)

            else:

                # A 3D image with a z extent larger of 1
                label_img = d["label"]

                # Create 3D ball footprint
                footprint = torch.tensor(ball(self.border_width), dtype=torch.int32)

                # Process and update
                d["label"] = process_label(label_img, footprint)

        elif d["label"].ndim == 4:

            if d["label"].shape[0] > 1:
                raise ValueError("CDHW tensor must have only one channel!")

            # A 3D image with a z extent larger of 1
            label_img = d["label"][..., :, :].squeeze()

            # Create 3D ball footprint
            footprint = torch.tensor(ball(self.border_width), dtype=torch.int32)

            # Process and update
            d["label"] = process_label(label_img, footprint).unsqueeze(0)

        else:
            raise ValueError("Unsupported image dimensions!")

        return d


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

    def _extract_subvolume(self, image, bbox):
        """
        Extracts a subset from a given 2D or 3D image using a bounding box.

        Parameters:
        image (np.ndarray): The n-dimensional image.
        bbox (tuple): The bounding box with format (min_dim1, max_dim1, ..., min_dimN, max_dimN),
        where N is the number of dimensions in the image.

        Returns:
        np.ndarray: The extracted subvolume.
        """
        # Validate bounding box
        if len(bbox) != 2 * image.ndim:
            raise ValueError("Bounding box format does not match image dimensions.")

        # Construct the slice object for each dimension
        ndim = image.ndim
        slices = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))
        return image[slices]

    def _insert_subvolume(self, image, subvolume, bbox):
        """
        Inserts a subvolume into an image using a specified bounding box.

        Parameters:
        image (np.ndarray): The original n-dimensional image or volume.
        subvolume (np.ndarray): The subvolume or subimage to insert, which must fit within the dimensions specified by bbox.
        bbox (tuple): The bounding box with format (min_dim1, max_dim1, ..., min_dimN, max_dimN),
                      where N is the number of dimensions in the image. The bounding box specifies where to insert the subvolume.

        Returns:
        np.ndarray: The image with the subvolume inserted.
        """
        ndim = image.ndim
        # Validate bounding box and sub-volume
        if len(bbox) != 2 * image.ndim:
            raise ValueError("Bounding box format does not match image dimensions.")
        if not (
            all(subvolume.shape[i] == bbox[i + ndim] - bbox[i] for i in range(ndim))
        ):
            raise ValueError(
                "Sub-volume dimensions must match the bounding box dimensions."
            )

        # Construct the slice object for each dimension
        slices = tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))

        # Insert the subvolume into the image
        image[slices] = subvolume
        return image

    def _process_single(self, data_label):
        """Process a single image (of a potential batch)."""
        # Prepare data out
        dt_out = np.zeros(data_label.shape, dtype=np.float32)

        # Remove singleton dimensions
        data_label = data_label.squeeze()

        # Calculate bounding boxes
        regions = regionprops(data_label)

        # Process all labels serially
        for region in regions:

            if region.label == 0:
                continue

            # Extract the subvolume
            cropped_mask = self._extract_subvolume(data_label, region.bbox) > 0

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
            dt_out = self._insert_subvolume(dt_out, dt_tmp, bbox)

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
            dt_final = np.zeros(data_label.shape, dtype=float)
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


class DebugCheckAndFixAffineDimensions(Transform):
    """Check that the affine transform is (4 x 4)."""

    def __init__(self, expected_voxel_size, *args, **kwargs):
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
        self.expected_voxel_size = expected_voxel_size

    def __call__(self, data):
        """Call the Transform."""
        prefix = f"{self.name} :: " if self.name != "" else ""
        if type(data) is list:
            for item in data:
                if type(item) == dict:
                    for key in item.keys():
                        sub_item = item[key]
                        if type(sub_item) is MetaTensor and hasattr(sub_item, "affine"):

                            if sub_item.affine.shape != (4, 4):
                                print(
                                    f"{prefix}Affine matrix needs correcting! Current shape is {sub_item.affine.shape}"
                                )
                                sub_item.affine = torch.tensor(
                                    [
                                        [self.expected_voxel_size[0], 0.0, 0.0, 0.0],
                                        [0.0, self.expected_voxel_size[1], 0.0, 0.0],
                                        [0.0, 0.0, self.expected_voxel_size[2], 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )
                            elif (
                                sub_item.affine[0, 0] != self.expected_voxel_size[0]
                                or sub_item.affine[1, 1] != self.expected_voxel_size[1]
                                or sub_item.affine[2, 2] != self.expected_voxel_size[2]
                                or sub_item.affine[0, 3] != 0.0
                            ):
                                print(
                                    f"{prefix}Affine matrix needs correcting! Current matrix is {item.affine}"
                                )
                                sub_item.affine = torch.tensor(
                                    [
                                        [self.expected_voxel_size[0], 0.0, 0.0, 0.0],
                                        [0.0, self.expected_voxel_size[1], 0.0, 0.0],
                                        [0.0, 0.0, self.expected_voxel_size[2], 0.0],
                                        [0.0, 0.0, 0.0, 1.0],
                                    ]
                                )
                            else:
                                # The affine matrix is fine, nothing to correct
                                return data

        if type(data) is dict:
            for key in data.keys():
                item = data[key]
                if type(item) is MetaTensor and hasattr(item, "affine"):

                    if item.affine.shape != (4, 4):
                        print(
                            f"{prefix}Affine matrix needs correcting! Current shape is {item.affine.shape}"
                        )
                        item.affine = torch.tensor(
                            [
                                [self.expected_voxel_size[0], 0.0, 0.0, 0.0],
                                [0.0, self.expected_voxel_size[1], 0.0, 0.0],
                                [0.0, 0.0, self.expected_voxel_size[2], 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    elif (
                        item.affine[0, 0] != self.expected_voxel_size[0]
                        or item.affine[1, 1] != self.expected_voxel_size[1]
                        or item.affine[2, 2] != self.expected_voxel_size[2]
                        or item.affine[0, 3] != 0.0
                    ):
                        print(
                            f"{prefix}Affine matrix needs correcting! Current matrix is {item.affine}"
                        )
                        item.affine = torch.tensor(
                            [
                                [self.expected_voxel_size[0], 0.0, 0.0, 0.0],
                                [0.0, self.expected_voxel_size[1], 0.0, 0.0],
                                [0.0, 0.0, self.expected_voxel_size[2], 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    else:
                        # The affine matrix is fine, nothing to correct
                        pass

        elif type(data) is MetaTensor and hasattr(data, "affine"):
            if data.affine.shape != (4, 4):
                print(
                    f"{prefix}Affine matrix needs correcting! Current shape is {data.affine.shape}"
                )
                data.affine = torch.tensor(
                    [
                        [self.expected_voxel_size[0], 0.0, 0.0, 0.0],
                        [0.0, self.expected_voxel_size[1], 0.0, 0.0],
                        [0.0, 0.0, self.expected_voxel_size[2], 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            elif (
                data.affine[0, 0] != self.expected_voxel_size[0]
                or data.affine[1, 1] != self.expected_voxel_size[1]
                or data.affine[2, 2] != self.expected_voxel_size[2]
                or data.affine[0, 3] != 0.0
            ):
                print(
                    f"{prefix}Affine matrix needs correcting! Current matrix is {data.affine}"
                )
                data.affine = torch.tensor(
                    [
                        [self.expected_voxel_size[0], 0.0, 0.0, 0.0],
                        [0.0, self.expected_voxel_size[1], 0.0, 0.0],
                        [0.0, 0.0, self.expected_voxel_size[2], 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            else:
                # The affine matrix is fine, nothing to correct
                pass

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
            # This is a bit of a hack...
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
                    if hasattr(data[key], "meta") and "affine" in data[key].meta:
                        a = data[key].meta["affine"]
                        print(
                            f"'{key}': voxel size=({a[0, 0]}, {a[1, 1]}, {a[2, 2]});",
                            end=" ",
                        )
                elif t is dict:
                    print(f"'{key}': dict;", end=" ")
                elif t is pathlib.PosixPath:
                    print(f"'{key}': path={str(data[key])};", end=" ")
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
