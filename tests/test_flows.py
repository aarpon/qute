#  ********************************************************************************
#  Copyright © 2022 - 2024, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************

import numpy as np

from qute.transforms.util import compute_2d_flows


def test_compute_2d_flow():

    label_img = np.zeros((10, 10), dtype=np.uint16)

    label_img[2:4, 2:4] = 1
    label_img[7:9, 7:9] = 2
    label_img[1:5, 6:8] = 3
    label_img[6:9, 2:5] = 3

    # Calculate flows
    flow_x, flow_y = compute_2d_flows(label_img=label_img)

    # Expected flows
    expected_flow_x = np.zeros((10, 10), dtype=np.float32)
    expected_flow_x[1, 6:8] = 0.70709676
    expected_flow_x[2, 2:4] = 0.70709676
    expected_flow_x[3, 2:4] = -0.70709676
    expected_flow_x[4, 6:8] = -0.70709676
    expected_flow_x[6, 2] = 0.70709676
    expected_flow_x[6, 3] = 0.99998
    expected_flow_x[6, 4] = 0.70709676
    expected_flow_x[7, 7:9] = 0.70709676
    expected_flow_x[8, 2] = -0.70709676
    expected_flow_x[8, 3] = -0.99998
    expected_flow_x[8, 4] = -0.70709676
    expected_flow_x[8, 7:9] = -0.70709676

    expected_flow_y = np.zeros((10, 10), dtype=np.float32)
    expected_flow_y[1, 6] = 0.70709676
    expected_flow_y[1, 7] = -0.70709676
    expected_flow_y[2:4, 2] = 0.70709676
    expected_flow_y[2:4, 3] = -0.70709676
    expected_flow_y[2:4, 6] = 0.99998
    expected_flow_y[2:4, 7] = -0.99998
    expected_flow_y[4, 6] = 0.70709676
    expected_flow_y[4, 7] = -0.70709676
    expected_flow_y[6, 2] = 0.70709676
    expected_flow_y[6, 4] = -0.70709676
    expected_flow_y[7, 2] = 0.99998
    expected_flow_y[7, 4] = -0.99998
    expected_flow_y[7:9, 7] = 0.70709676
    expected_flow_y[7:9, 8] = -0.70709676
    expected_flow_y[8, 2] = 0.70709676
    expected_flow_y[8, 4] = -0.70709676

    # Test
    assert flow_x.shape == (10, 10)
    assert np.all(np.isclose(flow_x, expected_flow_x))
    assert flow_y.shape == (10, 10)
    assert np.all(np.isclose(flow_y, expected_flow_y))
