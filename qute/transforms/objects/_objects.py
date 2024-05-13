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
from typing import Union

import monai.data
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform, Transform
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    label as ndi_label,
)
from skimage.measure import label, regionprops
from skimage.morphology import ball, disk
from skimage.segmentation import watershed

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

    @TODO: Add with_batch_dim support.
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


class NormalizedDistanceTransformd(MapTransform):
    """Calculates and normalizes the distance transform per region from the labels image (from an instance segmentation)."""

    def __init__(
        self,
        keys: tuple[str, ...] = ("label",),
        reverse: bool = False,
        do_not_zero: bool = False,
        add_seed_channel: bool = False,
        seed_radius: int = 1,
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

        add_seed_channel: bool = False
            Whether to also add a white disk of radius `seed_radius` at the center of mass of each label in a second
            channel.

        seed_radius: int = 1
            Radius of the disk to be added at the center of mass of each label in a second channel. Ignored if
            add_seed_channel is False.

        with_batch_dim: bool (Optional, default is False)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.
        """
        super().__init__(keys=keys)
        self.keys = keys
        self._transform = NormalizedDistanceTransform(
            reverse=reverse,
            do_not_zero=do_not_zero,
            add_seed_channel=add_seed_channel,
            seed_radius=seed_radius,
            with_batch_dim=with_batch_dim,
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


class NormalizedDistanceTransform(Transform):
    """Calculates and normalizes the distance transform per region of the selected pixel class from a labels image."""

    def __init__(
        self,
        reverse: bool = False,
        do_not_zero: bool = False,
        in_place: bool = True,
        add_seed_channel: bool = False,
        seed_radius: int = 1,
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

        add_seed_channel: bool = False
            Whether to also add a white disk of radius `seed_radius` at the center of mass of each label in a second
            channel.

        seed_radius: int = 1
            Radius of the disk to be added at the center of mass of each label in a second channel. Ignored if
            add_seed_channel is False.

        with_batch_dim: bool (Optional, default is False)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.
        """
        super().__init__()
        self.reverse = reverse
        self.do_not_zero = do_not_zero
        self.in_place = in_place
        self.add_seed_channel = add_seed_channel
        self.seed_radius = seed_radius
        self.with_batch_dim = with_batch_dim

    def _process_single(self, data_label):
        """Process a single image (of a potential batch)."""
        # Prepare data out
        dt_out = np.zeros(data_label.shape, dtype=np.float32)

        # If needed, allocate the seeds stack
        dt_seeds = None
        disk_seed = None
        if self.add_seed_channel:
            dt_seeds = np.zeros(dt_out.shape, dtype=np.float32)

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

            # Calculate the center of the connected component
            seed_tmp = None
            if self.add_seed_channel:
                if disk_seed is None:
                    disk_seed = ball(radius=self.seed_radius)
                center_of_mass = np.round(
                    np.mean(np.argwhere(dt_tmp > 0), axis=0)
                ).astype(int)
                seed_tmp = np.zeros(cropped_mask.shape, dtype=dt_tmp.dtype)
                seed_tmp[tuple(center_of_mass)] = 1.0
                seed_tmp = binary_dilation(seed_tmp, structure=disk_seed).astype(
                    np.float32
                )

            # Insert it into dt_out
            bbox = region.bbox
            while dt_tmp.ndim < dt_out.ndim:
                dt_tmp = dt_tmp[np.newaxis, :]
                if self.add_seed_channel:
                    seed_tmp = seed_tmp[np.newaxis, :]
                m = len(bbox) // 2
                bbox = tuple([0] + list(bbox[:m]) + [1] + list(bbox[m:]))
            dt_out = insert_subvolume(dt_out, dt_tmp, bbox)
            if self.add_seed_channel:
                dt_seeds = insert_subvolume(dt_seeds, seed_tmp, bbox)

        if self.add_seed_channel:
            return np.concatenate((dt_out, dt_seeds), axis=0)
        else:
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


class WatershedAndLabelTransformd(MapTransform):
    """Calculates the watershed transform and returns a labeled image.

    The first channel is expected to be a distance transform (either direct or inverse, and optionally normalized).
    Optionally, the second channel can contain seeds for the watershed transform.
    """

    def __init__(
        self,
        keys: tuple[str, ...] = ("label",),
        use_seed_channel: bool = True,
        with_batch_dim: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------

        keys: tuple[str, ...]
            Keys fot the tensor to be transformed. This transform makes sense for label or mask images only.

        use_seed_channel: bool
            Whether to use a seed channel for the watershed transform. It is expected that the image to

        with_batch_dim: bool (Optional, default is False)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.
        """
        super().__init__(keys=keys)
        self.keys = keys
        self._transform = WatershedAndLabelTransform(
            use_seed_channel=use_seed_channel, with_batch_dim=with_batch_dim
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


class WatershedAndLabelTransform(Transform):
    """Calculates the watershed transform and returns a labeled image.

    The first channel is expected to be a distance transform (either direct or inverse, and optionally normalized).
    Optionally, the second channel can contain seeds for the watershed transform.
    """

    def __init__(
        self,
        use_seed_channel: bool = True,
        with_batch_dim: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------

        use_seed_channel: bool
            Whether to use a seed channel for the watershed transform. It is expected that the image to

        with_batch_dim: bool (Optional, default is False)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.
        """
        super().__init__()
        self.use_seed_channel = use_seed_channel
        self.with_batch_dim = with_batch_dim

    def _process_single(self, data_label):
        """Process a single image (of a potential batch)."""

        # Prepare for the watershed algorithm
        mask = binary_fill_holes(data_label[0] > 0)
        dist = distance_transform_edt(mask)

        # Label seed points for the watershed?
        if self.use_seed_channel:
            seed_labels, _ = ndi_label(data_label[1])
        else:
            seed_labels = None

        # Run watershed
        labels = watershed(-dist, markers=seed_labels, mask=mask, connectivity=1)

        # Replace channel 0
        data_label[0] = labels.astype(np.int32)

        # Return result
        return data_label

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
