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

import sys
from pathlib import Path

from qute.director import SegmentationDirector3D

if __name__ == "__main__":

    # Configuration file
    config_file = Path(__file__).parent / "cell_segmentation_demo_unet_3d_config.ini"

    # Instantiate Director: make sure to instantiate it from a script entry point
    director = SegmentationDirector3D(config_file=config_file)

    # Run training
    director.run()

    # Properly exit
    sys.exit(0)
