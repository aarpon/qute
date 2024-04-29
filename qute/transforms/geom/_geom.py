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
from torch.nn import functional as F

from qute.transforms.util import get_tensor_num_spatial_dims


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
