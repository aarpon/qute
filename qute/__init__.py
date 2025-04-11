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

import os
import platform

if platform.system() == "Darwin" and platform.machine() == "arm64":
    # Allow falling back to CPU on Apple M1/M2 devices if
    # operations are not supported by MPS.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if platform.system() == "Linux":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


__version__ = "0.5.0"
__doc__ = f"""
This is the documentation of the `qute` library (version {__version__}).

`qute` leverages and extends several [PyTorch](https://pytorch.org/)-based framework and tools.

For getting-started instructions, see [qute on GitHub](https://github.com/aarpon/qute).
"""
