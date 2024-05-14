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
import torch
from monai.losses import DiceCELoss
from torch.nn import MSELoss


class CombinedMSEDiceCELoss(torch.nn.Module):
    """
    Combined MSE and Dice Cross-Entropy Loss to handle the output of
    qute.transforms.objects.WatershedAndLabelTransform(). The input prediction
    and ground truth are expected to have one regression and one classification
    channel (e.g., inverse distance transform and seed points).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        regression_channel: int = 0,
        classification_channel: int = 1,
        *args,
        **kwargs,
    ):
        """Constructor.

        alpha: float
            Fraction of the MSELoss() to be combined with the corresponding (1 - alpha) fraction of the DiceCELoss.

        regression_channel: int = 0
            Regression channel (e.g., inverse distance transform), on which to apply the Mean Absolute Error metric.

        classification_channel: int = 1
            Classification channel (e.g., watershed seeds), on which to apply the Dice metric.
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.regression_channel = regression_channel
        self.classification_channel = classification_channel
        self.mse_loss = MSELoss()
        # Since it's a single-channel prediction, "to_onehot_y=True" and "softmax=True" are not necessary (and ignored)
        self.dice_ce_loss = DiceCELoss(include_background=True)

    def forward(self, output, target):
        """Update the state of the loss with new predictions and targets."""

        # Check the number of dimensions in output tensor to determine if it's 2D or 3D
        if len(output.shape) not in [4, 5]:
            raise ValueError(
                "Unexpected number of dimensions, expected 4 (2D) or 5 (3D)."
            )

        # Determine if the data is 2D or 3D and adjust indexing accordingly
        dim_idx = (slice(None),) * (len(output.shape) - 2)

        # Calculate MSE loss
        mse_loss = self.mse_loss(
            output[:, self.regression_channel, None, *dim_idx],
            target[:, self.regression_channel, None, *dim_idx],
        )

        # Calculate Dice CE loss (the one-hot conversion is done automatically)
        dice_ce_loss = self.dice_ce_loss(
            output[:, self.classification_channel, None, *dim_idx],
            target[:, self.classification_channel, None, *dim_idx],
        )

        # Combined the losses
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * dice_ce_loss

        # Return
        return combined_loss
