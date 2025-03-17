#  ********************************************************************************
#  Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************

from typing import Optional, Tuple

import monai
import torch
from monai.networks.nets import DynUNet as MonaiDynUNet

from qute.campaigns import CampaignTransforms
from qute.models.base_model import BaseModel

__doc__ = "DynUNet."
__all__ = [
    "DynUNet",
]


class DynUNet(BaseModel):
    """Wrap MONAI's DynUNet architecture into a PyTorch Lightning module.

    The default settings are compatible with a classification task, where
    a single-channel input image is transformed into a multi-class label image.
    """

    def __init__(
        self,
        *,
        campaign_transforms: CampaignTransforms,
        criterion: monai.losses,
        metrics: monai.metrics,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 3,
        class_names: Optional[Tuple[str, ...]] = None,
        kernel_size: Optional[Tuple[int, ...]] = None,
        strides: Optional[Tuple[int, ...]] = None,
        filters: Optional[Tuple[int]] = None,
        upsample_kernel_size: Optional[Tuple[int]] = None,
        learning_rate: float = 1e-2,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        lr_scheduler_class: torch.optim.lr_scheduler = torch.optim.lr_scheduler.LambdaLR,
        lr_scheduler_parameters: Optional[dict] = None,
        dropout: float = 0.0,
        deep_supervision: bool = False,
    ):
        super().__init__(
            campaign_transforms=campaign_transforms,
            criterion=criterion,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_parameters=lr_scheduler_parameters,
            class_names=class_names,
        )

        # Defaults
        if kernel_size is None:
            kernel_size = tuple([[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]])
        if strides is None:
            strides = tuple([[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]])
        if filters is None:
            filters = tuple([32, 64, 128, 256, 512])
        if upsample_kernel_size is None:
            upsample_kernel_size = tuple([[2, 2], [2, 2], [2, 2], [2, 2]])

        # Keep track of whether deep_supervision is on
        self.deep_supervision = deep_supervision

        # Instantiate MONAI's DynUNet model
        self.net = MonaiDynUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            filters=filters,
            upsample_kernel_size=upsample_kernel_size,
            deep_supervision=deep_supervision,
            res_block=True,
            dropout=dropout,
        )

    def training_step(self, batch, batch_idx):
        """Perform a training step."""

        # If there is no deep supervision, we calculate loss and metrics
        # as usual
        if not self.deep_supervision:
            return super().training_step(batch, batch_idx)

        # In case of supervision, we have a series of predictions
        x, y = batch
        y_hat = self.forward(x)

        # Check that the number of dimensions fits
        assert y_hat.ndim == 5, "Expected predictions dimension not found."

        # Compute the average loss for all feature maps (with uniform weights)
        # @see https://docs.monai.io/en/stable/networks.html#dynunet
        preds_unbind = torch.unbind(y_hat, dim=1)
        loss = 0.0
        weights = [1.0 / len(preds_unbind)] * len(preds_unbind)
        for pred, w in zip(preds_unbind, weights):
            loss += w * self.criterion(pred, y)

        # Log the loss
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}
