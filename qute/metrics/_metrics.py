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
from abc import ABC

import torch
import torchmetrics
from monai.metrics import DiceMetric
from torchmetrics import MeanAbsoluteError

from qute.transforms.util import get_tensor_num_spatial_dims


class CombinedInvMeanAbsoluteErrorBinaryDiceMetric(torchmetrics.Metric, ABC):
    """
    Combined Inverse Exponential Mean Absolute Error and Dice Metric to handle the output of
    qute.transforms.objects.WatershedAndLabelTransform(). The input prediction
    and ground truth are expected to have one regression and one binary classification
    channel (e.g., inverse distance transform and seed points). Supported dimensionality
    * [B, C, H, W] with `with_bach_dim` = True or [C, H, W] with `with_bach_dim` = False for 2D
    * [B, C, D, H, W] with `with_bach_dim` = True or [C, D, H, W] with `with_bach_dim` = False for 3D

    The Inverse Exponential Mean Absolute Error is computed as:
        ie_mae = torch.exp(-self.beta * mae(output, target))

    The Dice Metric is the one implemented in `monai.metrics.DiceMetric`:
    """

    def __init__(
        self,
        alpha: float = 0.5,
        max_mae_value: float = 1.0,
        regression_channel: int = 0,
        classification_channel: int = 1,
        foreground_class: int = 1,
        with_batch_dim: bool = True,
        dist_sync_on_step=False,
    ):
        """Constructor.

        num_classes: int = 2
            Number of classes for the Dice Metric calculation.

        alpha: float
            Fraction of the MeanAbsoluteError() to be combined with the corresponding (1 - alpha) fraction
            of the DiceMetric.

        max_mae_value: float
            Maximum possible value for normalizing the MAE. This should be chosen based on the range of the
            regression target.

        regression_channel: int = 0
            Regression channel (e.g., inverse distance transform), on which to apply the Mean Absolute Error metric.

        classification_channel: int = 1
            Classification channel (e.g., watershed seeds), on which to apply the Dice metric.

        foreground_class: int = 1
            Class corresponding to the foreground in the classification (usually, background is 0 and foreground is 1).

        with_batch_dim: bool (Optional, default is True)
            Whether the input tensor has a batch dimension or not. This is to distinguish between the
            2D case (B, C, H, W) and the 3D case (C, D, H, W). All other supported cases are clear.

        dist_sync_on_step: bool
            Whether the synchronization of metric states across all processes (nodes/GPUs) should occur after each
            training step (if True) or at the end of the epoch (if False). It can be left on False in most cases.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        if max_mae_value <= 0.0:
            raise ValueError("max_mae_value must a positive number (larger than zero).")

        self.alpha = alpha
        self.max_mae_value = max_mae_value
        self.mae_metric = MeanAbsoluteError()
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        self.regression_channel = regression_channel
        self.classification_channel = classification_channel
        self.foreground_class = foreground_class
        self.with_batch_dim = with_batch_dim
        self.add_state("total_metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_updates", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output, target):
        """Update the state of the metric with new predictions and targets."""

        if len(output.shape) not in [3, 4, 5]:
            raise ValueError("Unsupported geometry.")

        # Do we have a 2D or 3D tensor (excluding batch and channel dimensions)?
        effective_dims = get_tensor_num_spatial_dims(output, self.with_batch_dim)

        if effective_dims not in [2, 3]:
            raise ValueError("Unsupported geometry.")

        # For simplicity, let's make sure the input tensors have consistent dimensions
        if effective_dims == 2:
            if self.with_batch_dim:
                if len(output.shape) == 4:
                    # [B, C, W, H] -> [B, C, D, W, H]
                    output = output.unsqueeze(2)
                    target = target.unsqueeze(2)
                else:
                    raise ValueError("Unsupported geometry.")
            else:
                if len(output.shape) == 3:
                    # [C, W, H] -> [B, C, D, W, H]
                    output = output.unsqueeze(1).unsqueeze(0)
                    target = target.unsqueeze(1).unsqueeze(0)
                else:
                    raise ValueError("Unsupported geometry.")
        elif effective_dims == 3:
            if self.with_batch_dim:
                if len(output.shape) == 5:
                    # Already [B, C, D, W, H]
                    pass
                else:
                    raise ValueError("Unsupported geometry.")
            else:
                if len(output.shape) == 4:
                    # [C, D, W, H] -> [B, C, D, W, H]
                    output = output.unsqueeze(0)
                    target = target.unsqueeze(0)
                else:
                    # Already [B, C, D, W, H]
                    pass
        else:
            raise ValueError("Unsupported geometry.")

        # Calculate the MAE metric
        mae_value = self.mae_metric(
            output[:, self.regression_channel, ...].unsqueeze(1),
            target[:, self.regression_channel, ...].unsqueeze(1),
        )

        # Normalize and invert MAE
        inv_norm_mae = 1 - (mae_value / self.max_mae_value)

        # Calculate the DICE metric (ignore the background)
        dice_metric = self.dice_metric(
            self._as_discrete(output[:, self.classification_channel, ...].unsqueeze(1)),
            self._as_discrete(target[:, self.classification_channel, ...].unsqueeze(1)),
        )
        dice_metric = dice_metric[:, self.foreground_class].mean()

        # Combine them linearly
        num_updates = 1
        combined_metric = self.alpha * inv_norm_mae + (1 - self.alpha) * dice_metric

        # Accumulate the metric
        self.total_metric += combined_metric
        self.num_updates += num_updates

        # Return the combined metric
        return combined_metric

    def forward(self, output, target):
        """Update the state of the metric with new predictions and targets."""
        # Update the metrics
        self.update(output, target)

        # Return the computed value directly
        return self.compute()

    def compute(self):
        """Compute the final metric based on the state."""
        if self.num_updates == 0:
            return torch.tensor(0.0)
        return self.total_metric / self.num_updates

    def aggregate(self):
        """Aggregate the metrics."""
        return self.compute()

    @staticmethod
    def _as_discrete(logits):
        """Convert logits to classes and then convert to one-hot format."""

        if logits.dim() != 5:
            raise ValueError("Unsupported geometry.")

        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Apply a threshold to convert probabilities to binary class indices
        threshold = 0.5
        class_indices = (probabilities > threshold).long()

        # Apply one-hot encoding for binary classification
        one_hot = torch.nn.functional.one_hot(class_indices, num_classes=2)

        # Reshape the one-hot tensor to bring the channel dimension in the right position
        one_hot = one_hot.permute(0, 5, 2, 3, 4, 1).squeeze(-1).float()

        return one_hot
