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
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity

    def __call__(self, image: torch.tensor) -> torch.tensor:
        """
        Apply the transform to `image`.

        Returns
        -------

        result: tensor
            A stack of images with the same width and height as `label` and with `num_classes` planes.
        """
        return (image - self.min_intensity) / (self.max_intensity - self.min_intensity)


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
