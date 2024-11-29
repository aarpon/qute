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

from typing import Optional, Tuple

import monai
import torch
from monai.networks.nets import AttentionUnet as MonaiAttentionUNet

from qute.campaigns import CampaignTransforms
from qute.models.base_model import BaseModel

__doc__ = "AttentionUNet."
__all__ = [
    "AttentionUNet",
]


class AttentionUNet(BaseModel):
    """Wrap MONAI's AttentionUNet architecture into a PyTorch Lightning module.

    The default settings are compatible with a classification task, where
    a single-channel input image is transformed into a multi-class label image.
    """

    def __init__(
        self,
        *,
        campaign_transforms: CampaignTransforms,
        criterion: monai.losses,
        metrics: monai.metrics,
        learning_rate: float = 1e-2,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        lr_scheduler_class: torch.optim.lr_scheduler = torch.optim.lr_scheduler.LambdaLR,
        lr_scheduler_parameters: Optional[dict] = None,
        class_names: Optional[Tuple[str, ...]] = None,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 3,
        channels: Tuple[int, ...] = (16, 32, 64),
        strides: Optional[Tuple[int, ...]] = None,
        dropout: float = 0.0,
    ):
        """
        Constructor.

        Parameters
        ----------

        campaign_transforms: CampaignTransforms
            Define all transforms necessary for training, validation, testing, and (full) prediction.
            @see `qute.transforms.CampaignTransforms` for documentation.

        criterion:  monai.losses.*
            Loss function to use during training.

        metrics: monai.metrics.*
            Metrics used for validation and test. Set to None to omit.

        learning_rate: float = 1e-2
            Learning rate for optimization.

        optimizer_class: torch.optim.Optimizer
            The optimizer class to use.

        lr_scheduler_class: torch.optim.lr_scheduler
            The learning rate scheduler class to use.

        lr_scheduler_parameters: Optional[dict] = None
            Dictionary of scheduler parameters.

        class_names: Optional[Tuple[str, ...]] = None
            Names of the output classes (for logging purposes). If omitted, they will default
            to ("class_0", "class_1", ...)

        spatial_dims: int = 2
            Whether 2D or 3D data.

        in_channels: int = 1
            Number of input channels.

        out_channels: int = 3
            Number of output channels (or labels, or classes)

        channels: Tuple[int, ...] = (16, 32, 64)
            Number of neurons per layer.

        strides: Optional[Tuple[int, ...]] = None
            Strides for down-sampling.

        dropout: float = 0.0
            Dropout ratio.
        """
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

        if class_names is None:
            class_names = tuple(f"class_{i}" for i in range(out_channels))

        self.class_names = class_names

        if strides is None:
            strides = (2,) * (len(channels) - 1)

        self.net = MonaiAttentionUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )

        # Log the hyperparameters
        self.save_hyperparameters(ignore=["criterion", "metrics"])

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.

        Returns
        -------
        y_hat: torch.Tensor
            Output tensor from the network.
        """
        y_hat = self.net(x)
        return y_hat
