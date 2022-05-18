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
import torchmetrics


class UNet(pl.LightningModule):
    """Wrap MONAI's UNet architecture into a PyTorch Lightning module.
    
    The default settings are compatible with a classification task, where
    a single-channel input image is transformed into a three-class label image.
    """

    def __init__(
            self,
            spatial_dims: int = 2,
            in_channels: int = 1,
            out_channels: int = 3,
            channels: tuple = (16, 32, 64, 128),
            strides: tuple = (2, 2, 2),
            criterion=GeneralizedDiceLoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                batch=True
            ),
            metrics=torchmetrics.JaccardIndex(
                num_classes=3,
                ignore_index=0),
            learning_rate: float = 1e-2,
            optimizer_class=AdamW,
            num_res_units: int = 0
    ):
        """
        Constructor.

        Parameters
        ----------

        spatial_dims: int = 2
            Whether 2D or 3D data.

        in_channels: int = 1
            Number of input channels.

        out_channels: int = 3
            Number of output channels (or labels, or classes)

        channels: tuple = (16, 32, 64, 128)
            Number of neuron per layer.

        strides: tuple = (2, 2, 2)
            Strides for down-sampling.

        criterion: GeneralizedDiceLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
            Loss function. Please NOTE: for classification, the loss function must convert `y` to OneHot and 
            apply softmax. The default loss function applies to a multi-label target where the background class 
            is omitted.

        metrics: JaccardIndex(num_classes=3, ignore_index=0)
            Metrics used for training, validation, and testing. Set to None to omit.

            The default metrics applies to a three-label target where the background (index = 0) class
            is omitted from calculation.

            Set to None to omit calculating and reporting it.

        learning_rate: float = 1e-2
            Learning rate for optimization.

        optimizer_class=AdamW
            Optimizer.

        num_res_units: int = 0
            Number of residual units for the UNet.
        """

        super().__init__()

        self.criterion = criterion
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.net = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        )
        self.save_hyperparameters(ignore=["criterion", "metrics"])

    def configure_optimizers(self):
        """Configure and return the optimizer."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        x, y = batch
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        if self.metrics is not None:
            self.log('train_metric', self.metrics(y_hat, y), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        x, y = batch
        y_hat = self.net(x)
        val_loss = self.criterion(y_hat, y)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        if self.metrics is None:
            return {"val_loss": val_loss}
        else:
            val_metric = self.metrics(y_hat, y)
            self.log('val_metric', val_metric, on_step=True, on_epoch=True)
            return {"val_loss": val_loss, "val_metric": val_metric}

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        x, y = batch
        y_hat = self.net(x)
        test_loss = self.criterion(y_hat, y)
        self.log('test_loss', test_loss)
        if self.metrics is None:
            return {"test_loss": test_loss}
        else:
            test_metric = self.metrics(y_hat, y)
            self.log('test_metric', test_metric)
            return {"test_loss": test_loss, "test_metric": test_metric}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """The predict step creates a label image from the output one-hot tensor."""
        x, _ = batch
        y_hat = self.net(x)
        label = y_hat.argmax(axis=1)
        return label
