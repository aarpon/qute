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
import torchmetrics
from monai.metrics import DiceMetric
from torchmetrics import MeanAbsoluteError


class CombinedMeanAbsoluteErrorBinaryDiceMetric(torchmetrics.Metric):
    """
    Combined Mean Absolute Error and Dice Metric to handle the output of
    qute.transforms.objects.WatershedAndLabelTransform(). The input prediction
    and ground truth are expected to have one regression and one binary classification
    channel (e.g., inverse distance transform and seed points).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        regression_channel: int = 0,
        classification_channel: int = 1,
        foreground_class: int = 1,
        dist_sync_on_step=False,
    ):
        """Constructor.

        num_classes: int = 2
            Number of classes for the Dice Metric calculation.

        alpha: float
            Fraction of the MeanAbsoluteError() to be combined with the corresponding (1 - alpha) fraction of the DiceMetric.

        regression_channel: int = 0
            Regression channel (e.g., inverse distance transform), on which to apply the Mean Absolute Error metric.

        classification_channel: int = 1
            Classification channel (e.g., watershed seeds), on which to apply the Dice metric.

        foreground_class: int = 1
            Class corresponding to the foreground in the classification (usually, background is 0 and foreground is 1).

        dist_sync_on_step: bool
            Whether the synchronization of metric states across all processes (nodes/GPUs) should occur after each
            training step (if True) or at the end of the epoch (if False). It can be left on False in most cases.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.alpha = alpha
        self.mae_metric = MeanAbsoluteError()
        self.dice_metric = DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        )
        self.regression_channel = regression_channel
        self.classification_channel = classification_channel
        self.foreground_class = foreground_class

        self.add_state("total_metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_updates", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output, target):
        """Update the state of the metric with new predictions and targets."""

        # Check the number of dimensions in output tensor to determine if it's 2D or 3D
        if len(output.shape) not in [4, 5]:
            raise ValueError(
                "Unexpected number of dimensions, expected 4 (2D) or 5 (3D)."
            )

        # Determine if the data is 2D or 3D and adjust indexing accordingly
        dim_idx = (slice(None),) * (len(output.shape) - 2)

        # Calculate the MAE metric
        mae_metric = self.mae_metric(
            output[:, self.regression_channel, None, *dim_idx],
            target[:, self.regression_channel, None, *dim_idx],
        )

        # Calculate the DICE metric (ignore the background)
        dice_metric = self.dice_metric(
            self._as_discrete(output[:, self.classification_channel, None, *dim_idx]),
            self._as_discrete(target[:, self.classification_channel, None, *dim_idx]),
        )[:, self.foreground_class].mean()

        # Combine them linearly
        combined_metric = self.alpha * mae_metric + (1 - self.alpha) * dice_metric

        # Accumulate the metric
        self.total_metric += combined_metric
        self.num_updates += 1

    def forward(self, output, target):
        """Update the state of the metric with new predictions and targets."""
        self.update(output, target)

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

        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Apply a threshold to convert probabilities to binary class indices
        threshold = 0.5
        class_indices = (probabilities > threshold).int()

        # Apply one-hot encoding for binary classification
        one_hot_encoded = torch.cat([1 - class_indices, class_indices], dim=1)

        return one_hot_encoded
