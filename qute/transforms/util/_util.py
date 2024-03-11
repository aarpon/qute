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

import numpy as np


def scale_dist_transform_by_region(
    dt: np.ndarray,
    regions: np.ndarray,
    reverse: bool = False,
    do_not_zero: bool = False,
    in_place: bool = False,
):
    """Scales the distance transform values by region.

    Parameters
    ----------
    dt: np.ndarray
        The distance transform array.

    regions: np.ndarray
        The region array.

    in_place: bool (Optional)
        Specifies whether the scaling should happen in place. Default is False.

    reverse: bool
        Whether to reverse the direction of the normalized distance transform: from 1.0 at the center of the
        objects and 0.0 at the periphery, to (close to) 0.0 at the center and 1.0 at the periphery.

    do_not_zero: bool
        This is only considered if `reverse` is True. Set to True not to allow that the center pixels in each
        region have an inverse distance transform of 0.0.

    Returns
    -------
    The updated distance transform array.
    """

    # Get all regions
    ids = np.unique(regions)

    # Should the process happen in place?
    if in_place:
        work_dt = dt
    else:
        work_dt = dt.copy()

    # Process all regions
    for i in ids:

        # Skip background
        if i == 0:
            continue

        # Extract the distance transform for current region and scale it
        indices = regions == i
        values = work_dt[indices]
        if reverse:
            # Reverse the direction of the distance transform: make sure to stretch
            # the maximum to 1.0; we can keep a minimum larger than 0.0 in the center.
            if do_not_zero:
                # Do not set the distance at the center to 0.0; the gradient is
                # slightly lower, depending on the original range.
                tmp = work_dt[indices]
                tmp = (tmp.max() + 1) - tmp
                work_dt[indices] = tmp / tmp.max()
            else:
                # Plain linear inverse
                min_value = work_dt[indices].min()
                max_value = work_dt[indices].max()
                work_dt[indices] = (work_dt[indices] - max_value) / (
                    min_value - max_value
                )
        else:
            work_dt[indices] = work_dt[indices] / values.max()

    # Return updated dt
    return work_dt
