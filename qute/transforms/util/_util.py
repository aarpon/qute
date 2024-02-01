import numpy as np


def scale_dist_transform_by_region(
    dt: np.ndarray, regions: np.ndarray, reverse: bool = False, in_place: bool = False
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
        objects and 0.0 at the periphery, to 0.0 at the center and 1.0 at the periphery.

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
            # Reverse the direction of the distance transform?
            work_dt[indices] = 1.0 - work_dt[indices] / values.max()
        else:
            work_dt[indices] = work_dt[indices] / values.max()

    # Return updated dt
    return work_dt
