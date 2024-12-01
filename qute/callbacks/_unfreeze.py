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

import math

import pytorch_lightning as pl
import torch.nn as nn


def progressive_unfreeze(
    model: nn.Module,
    start_epoch: int,
    max_epochs: int,
    unfreeze_strategy: str = "linear",
):
    """
    Progressively unfreeze encoder layers during fine-tuning (backwards from n-1 towards 0)

    Parameters
    ----------

    model: nn.Module
        PyTorch model with encoder layers

    start_epoch: int
        Epoch to begin unfreezing

    max_epochs: int
        Total number of training epochs

    unfreeze_strategy: str
        One of "linear or "exponential"
    """

    def unfreeze_layers(model, num_layers_to_unfreeze: int):
        # Get encoder parameters in reverse order (top to bottom)
        encoder_params = list(reversed(list(model.encoder.parameters())))

        # Unfreeze
        for i, param in enumerate(encoder_params):
            if i < num_layers_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def on_epoch_start(epoch):
        if epoch >= start_epoch:
            total_encoder_layers = len(list(model.encoder.parameters()))

            if unfreeze_strategy == "linear":
                # Linearly increase unfrozen layers (from top)
                num_layers_to_unfreeze = int(
                    (epoch - start_epoch + 1)
                    * total_encoder_layers
                    / (max_epochs - start_epoch)
                )
            elif unfreeze_strategy == "exponential":
                # Exponentially increase unfrozen layers (from top)
                num_layers_to_unfreeze = int(
                    total_encoder_layers * (1 - math.exp(-(epoch - start_epoch + 1)))
                )
            else:
                raise ValueError(
                    '`unfreeze_strategy` must be one of "linear" or "exponential."'
                )

            unfreeze_layers(model, num_layers_to_unfreeze)

            # Optional: Log unfrozen layers
            print(
                f"Epoch {epoch}: Unfreezing {num_layers_to_unfreeze} top encoder layers."
            )

    return on_epoch_start


# Custom callback
class ProgressiveUnfreezeCallback(pl.Callback):
    """Progressively unfreeze encoder layers during fine-tuning (backwards from n-1 towards 0)."""

    def __init__(
        self,
        model: nn.Module,
        start_epoch: int,
        max_epochs: int,
        unfreeze_strategy: str = "linear",
    ):
        """
        Constructor.

        Parameters
        ----------

        model: nn.Module
            PyTorch model with encoder layers

        start_epoch: int
            Epoch to begin unfreezing

        max_epochs: int
            Total number of training epochs

        unfreeze_strategy: str
            One of "linear or "exponential"
        """
        self.unfreeze_fn = progressive_unfreeze(
            model, start_epoch, max_epochs, unfreeze_strategy
        )

    def on_epoch_start(self, trainer, pl_module):
        """Custom on_epoch_start hook."""
        self.unfreeze_fn(trainer.current_epoch)
