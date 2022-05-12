#  ********************************************************************************
#   Copyright © 2022-, ETH Zurich, D-BSSE, Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the Apache License Version 2.0
#   which accompanies this distribution, and is available at
#   https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Contributors:
#       Aaron Ponti - initial API and implementation
#
#   This file adapted from: https://github.com/hiepph/unet-lightning
#  ******************************************************************************/

import pytorch_lightning as pl
from monai.networks.nets import UNet as MonaiUNet
from monai.losses import GeneralizedDiceLoss
from torch.optim import AdamW


class UNet(pl.LightningModule):
    """Wrap MONAI's UNet architecture into a PyTorch Lightning module."""

    def __init__(
            self,
            dimensions: int = 2,
            in_channels: int = 1,
            out_channels: int = 3,
            channels: tuple = (16, 32, 64, 128),
            strides: tuple = (2, 2, 2),
            criterion=GeneralizedDiceLoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                batch=True
            ),
            learning_rate: float = 1e-2,
            optimizer_class=AdamW,
            num_res_units: int = 0
    ):
        """
        Constructor.

        Parameters
        ----------

        dimensions: int = 2
            Whether 2D or 3D data.

        in_channels: int = 1
            Number of input channels.

        out_channels: int = 3
            Number of output channels (or labels, or classes)

        channels: tuple = (16, 32, 64, 128)
            Number of neuron per layer.

        strides: tuple = (2, 2, 2)
            Strides for downsampling.

        criterion=GeneralizedDiceLoss(include_background=True, to_onehot_y=True, softmax=True, batch=True)
            Loss function. Please NOTE: the loss function must convert `y` to OneHot and apply softmax.

        learning_rate: float = 1e-2
            Learning rate for optimization.

        optimizer_class=AdamW
            Optimizer.

        num_res_units: int = 0
            Number of residual units for the UNet.
        """

        super().__init__()

        self.criterion = criterion
        self.lr = learning_rate
        self.optimizer_class = optimizer_class
        self.net = MonaiUNet(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )
        self.save_hyperparameters(ignore=["criterion"])

    def configure_optimizers(self):
        """Configure and return the optimizer."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        x, y = batch
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        x, y = batch
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        x, y = batch
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """The predict step creates a label image from the ouput one-hot tensor."""
        x, y = batch
        y_hat = self.net(x)
        label = y_hat.argmax(axis=1)
        return label
