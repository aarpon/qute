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

import numpy as np
import pytest
import torch

from qute.transforms import AddBorderd, CellposeLabelReader


def test_add_borderd():

    # Load CellPose dataset
    reader = CellposeLabelReader()
    label = reader(Path(__file__).parent / "data" / "cellpose.npy")
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
