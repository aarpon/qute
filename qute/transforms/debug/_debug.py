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

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform, Transform


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
                            f"voxel size=({a[0, 0]}, {a[1, 1]}, {a[2, 2]});",
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
