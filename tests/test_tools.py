#  ********************************************************************************
#  Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************

import numpy as np

from qute.tools import calculate_receptive_field, num_levels_for_object_size


def test_calculate_receptive_field():
    rfs = []
    for i in range(1, 6):
        rfs.append(calculate_receptive_field(i))

    assert np.all(np.array([5, 14, 32, 68, 140]) == rfs)


def test_num_levels_for_object_size():
    object_size = 31
    assert num_levels_for_object_size(object_size) == 3

    object_size = 60
    assert num_levels_for_object_size(object_size) == 4

    object_size = 80
    assert num_levels_for_object_size(object_size) == 5
