from pathlib import Path

import numpy as np
import pytest
import torch

from qute.transforms import AddBorderd, CellposeLabelReader


def test_add_borderd():

    # Load CellPose dataset
    reader = CellposeLabelReader(Path(__file__).parent / "data" / "cellpose.npy")
    label = reader()
    assert label.shape == (300, 300), "Unexpected image size."

    # Add to data dictionary
    data = {}
    data["label"] = label

    # Add the border
    addb = AddBorderd()
    data = addb(data)

    # Make sure that the batch and channel dimensions are present
    assert data["label"].shape == (300, 300), "Unexpected image size."

    # Check that there are indeed three classes
    assert len(torch.unique(data["label"])) == 3, "Expected three classes."
