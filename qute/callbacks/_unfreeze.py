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
    encoder: nn.Module,
    start_epoch: int,
    end_epoch: int,
    max_epochs: int,
    unfreeze_strategy: str = "linear",
):
    """
    Progressively unfreeze encoder layers during fine-tuning within a specified epoch range

    Parameters
    ----------
    encoder: nn.Module
        The encoder module to progressively unfreeze
    start_epoch: int
        Epoch to begin unfreezing
    end_epoch: int
        Epoch by which all layers should be unfrozen
    max_epochs: int
        Total number of training epochs
    unfreeze_strategy: str
        One of "linear" or "exponential"
    """

    def unfreeze_layers(encoder, num_layers_to_unfreeze: int):
        encoder_params = list(reversed(list(encoder.parameters())))
        total_layers = len(encoder_params)

        # If we've reached end_epoch, unfreeze everything
        if num_layers_to_unfreeze >= total_layers:
            for param in encoder_params:
                param.requires_grad = True
        else:
            for i, param in enumerate(encoder_params):
                param.requires_grad = i < num_layers_to_unfreeze

    def on_train_epoch_start(epoch):
        total_encoder_layers = len(list(encoder.parameters()))

        if epoch < start_epoch:
            # Keep everything frozen before start_epoch
            unfreeze_layers(encoder, 0)
        elif epoch >= end_epoch:
            # Unfreeze everything after end_epoch
            unfreeze_layers(encoder, total_encoder_layers)
        else:
            # Progressive unfreezing between start_epoch and end_epoch
            if unfreeze_strategy == "linear":
                num_layers_to_unfreeze = int(
                    (epoch - start_epoch + 1)
                    * total_encoder_layers
                    / (end_epoch - start_epoch)
                )
            elif unfreeze_strategy == "exponential":
                num_layers_to_unfreeze = int(
                    total_encoder_layers * (1 - math.exp(-(epoch - start_epoch + 1)))
                )
            else:
                raise ValueError(
                    '`unfreeze_strategy` must be one of "linear" or "exponential"'
                )

            unfreeze_layers(encoder, num_layers_to_unfreeze)
            print(
                f"Epoch {epoch}: Unfreezing {num_layers_to_unfreeze} top encoder layers."
            )

    return on_train_epoch_start


# Custom callback
class ProgressiveUnfreezeCallback(pl.Callback):
    def __init__(
        self,
        encoder: nn.Module,
        start_epoch: int,
        end_epoch: int,
        max_epochs: int,
        unfreeze_strategy: str = "linear",
    ):
        self.unfreeze_fn = progressive_unfreeze(
            encoder, start_epoch, end_epoch, max_epochs, unfreeze_strategy
        )

    def on_train_epoch_start(self, trainer, pl_module):
        self.unfreeze_fn(trainer.current_epoch)
