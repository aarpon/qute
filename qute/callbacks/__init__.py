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

from ._unfreeze import ProgressiveUnfreezeCallback

__doc__ = "Custom PyTorch-Lightning callbacks."
__all__ = ["ProgressiveUnfreezeCallback"]
