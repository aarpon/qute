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
from monai.networks.nets import EfficientNetBN

from qute.campaigns import CampaignTransforms
from qute.models.base_model import BaseModel

__doc__ = "EfficientNet Model."
__all__ = [
    "EfficientNet",
]


class EfficientNet(BaseModel):
    """Wrap MONAI's EfficientNetBN architecture into a PyTorch Lightning module.

    The default settings are compatible with a classification task, where
    an input image is transformed into a multi-class output.
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
        model_name: str = "efficientnet-b0",
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 3,
        pretrained: bool = False,
        dropout: float = 0.2,
    ):
        """
        Constructor.

        Parameters
        ----------

        Parameters
        ----------

        campaign_transforms: CampaignTransforms
            Define all transforms necessary for training, validation, testing, and (full) prediction.

        criterion:  monai.losses
            Loss function to use during training.

        metrics: monai.metrics
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
            Names of the output classes (for logging purposes).

        spatial_dims: int = 2
            Whether 2D or 3D data.

        in_channels: int = 1
            Number of input channels.

        out_channels: int = 3
            Number of output channels (or labels, or classes)

        pretrained: bool
            Whether to load pretrained weights.

        dropout: float = 0.0
            Dropout ratio (currently unused).

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

        # Set class names if not provided
        if class_names is None:
            class_names = tuple(f"class_{i}" for i in range(out_channels))

        self.class_names = class_names

        # Store network-specific parameters
        self.model_name = model_name
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrained = pretrained
        self.dropout = dropout

        # Initialize the EfficientNet model
        self.net = EfficientNetBN(
            model_name=model_name,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=out_channels,
            pretrained=pretrained,
        )

        # Log the hyperparameters
        self.save_hyperparameters(ignore=["criterion", "metrics"])

    def forward(self, x):
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        y_hat : torch.Tensor
            Output tensor from the network.
        """
        y_hat = self.net(x)
        return y_hat
