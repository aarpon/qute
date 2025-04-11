# ******************************************************************************
# Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Apache License Version 2.0
# which accompanies this distribution, and is available at
# https://www.apache.org/licenses/LICENSE-2.0.txt
#
# Contributors:
#   Aaron Ponti - initial API and implementation
# ******************************************************************************

from ._campaigns import (
    CampaignTransforms,
    RestorationCampaignTransforms,
    SegmentationCampaignTransforms2D,
    SegmentationCampaignTransforms3D,
    SegmentationCampaignTransformsIDT2D,
    SegmentationCampaignTransformsIDT3D,
    SelfSupervisedRestorationCampaignTransforms,
)

__doc__ = "Full training campaign definitions."
__all__ = [
    "CampaignTransforms",
    "RestorationCampaignTransforms",
    "SegmentationCampaignTransforms2D",
    "SegmentationCampaignTransforms3D",
    "SegmentationCampaignTransformsIDT2D",
    "SegmentationCampaignTransformsIDT3D",
    "SelfSupervisedRestorationCampaignTransforms",
]
