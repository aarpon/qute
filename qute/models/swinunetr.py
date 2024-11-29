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
from pathlib import Path
from typing import Optional, Tuple, Union

import monai
import torch
from monai.networks.nets import SwinUNETR as MONAISwinUNETR

from qute.campaigns import CampaignTransforms
from qute.models.base_model import BaseModel

__doc__ = "SwinUNETR and related classes."
__all__ = [
    "SwinUNETR",
]


class SwinUNETR(BaseModel):
    """Wrap MONAI's SwinUNETR architecture into a PyTorch Lightning module.

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
        img_size: Tuple[int, int] = (640, 640),
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        feature_size: int = 24,
        dropout: float = 0.0,
    ):
        """
        Constructor.

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

        img_size: Tuple[int, int] = (640, 640)
            Input image size. Must be divisible by the patch size and window size.

        depths: Tuple[int, ...] = (2, 2, 2, 2)
            Depths of each stage in the Swin Transformer.

        num_heads: Tuple[int, ...] = (3, 6, 12, 24)
            Number of attention heads in different layers.

        feature_size: int = 24
            Feature size dimension.

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

        # Set class names if not provided
        if class_names is None:
            class_names = tuple(f"class_{i}" for i in range(out_channels))

        self.class_names = class_names

        # Initialize the network (include img_size)
        self.net = MONAISwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            drop_rate=dropout,
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

    def save_encoder_weights(self, filename: Union[str, Path]) -> None:
        """Save encoder weights."""
        if self.net is None or self.net.swinViT is None:
            return
        torch.save(self.net.swinViT.state_dict(), filename)

    def load_encoder_weights(self, filename: Union[str, Path]) -> None:
        """Load encoder weights."""
        if self.net is None or self.net.swinViT is None:
            return
        self.net.swinViT.load_state_dict(torch.load(filename, weights_only=True))

    def freeze_encoder(self):
        """Freeze the encoder weights."""
        if self.net is None or self.net.swinViT is None:
            return
        for param in self.net.swinViT.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the encoder weights."""
        if self.net is None or self.net.encoder is None:
            return
        for param in self.net.swinViT.parameters():
            param.requires_grad = True
