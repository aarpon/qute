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
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform, Transform
from nd2reader import ND2Reader
from tifffile import imread


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
            image_path = str(Path(str(d[key])).resolve())

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
            image_path = str(Path(str(d[key])).resolve())

            # Use the single tensor reader
            out = self.tensor_reader(image_path)

            # Assign the tensor to the corresponding key
            d[key] = out  # (Meta)Tensor(image)

        return d
