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


from typing import Optional

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import AttentionUnet as MonaiAttentionUNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR

from qute.campaigns import CampaignTransforms
from qute.models.unet import UNet


class AttentionUNet(UNet):
    """Wrap MONAI's AttentionUNet architecture into a PyTorch Lightning module.

    The default settings are compatible with a classification task, where
    a single-channel input image is transformed into a three-class label image.
    """

    def __init__(
        self,
        campaign_transforms: CampaignTransforms,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 3,
        class_names: Optional[tuple] = None,
        channels=(16, 32, 64),
        strides: Optional[tuple] = None,
        criterion=DiceCELoss(include_background=True, to_onehot_y=False, softmax=True),
        metrics=DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        ),
        learning_rate: float = 1e-2,
        optimizer_class=AdamW,
        lr_scheduler_class=PolynomialLR,
        lr_scheduler_parameters: dict = {"total_iters": 100, "power": 0.95},
        dropout: float = 0.0,
    ):
        """
        Constructor.

        Parameters
        ----------

        campaign_transforms: CampaignTransforms
            Define all transforms necessary for training, validation, testing and (full) prediction.
            @see `qute.transforms.CampaignTransforms` for documentation.

        spatial_dims: int = 2
            Whether 2D or 3D data.

        in_channels: int = 1
            Number of input channels.

        out_channels: int = 3
            Number of output channels (or labels, or classes)

        class_names: Optional[tuple] = None
            Names of the output classes (for logging purposes). If omitted, they will default
            to ("class_1", "class_2", ...)

        channels: tuple = (16, 32, 64)
            Number of neuron per layer.

        strides: Optional[tuple] = (2, 2)
            Strides for down-sampling.

        criterion: DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)
            Loss function. Please NOTE: for classification, the loss function must convert `y` to OneHot.
            The default loss function applies to a multi-label target where the background class is omitted.

        metrics: DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
            Metrics used for validation and test. Set to None to omit.

            The default metrics applies to a three-label target where the background (index = 0) class
            is omitted from calculation.

        learning_rate: float = 1e-2
            Learning rate for optimization.

        optimizer_class=AdamW
            Optimizer.

        lr_scheduler_class=PolynomialLR
            Learning rate scheduler.

        lr_scheduler_parameters={"total_iters": 100, "power": 0.99}
            Dictionary of scheduler parameters.

        dropout: float = 0.0
            Dropout ratio.
        """

        super().__init__(
            campaign_transforms=campaign_transforms,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            class_names=class_names,
            channels=channels,
            strides=strides,
            criterion=criterion,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_parameters=lr_scheduler_parameters,
            dropout=dropout,
        )

        self.campaign_transforms = campaign_transforms
        self.criterion = criterion
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.scheduler_class = lr_scheduler_class
        self.scheduler_parameters = lr_scheduler_parameters
        if class_names is None:
            class_names = (f"class_{i}" for i in range(out_channels))
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
