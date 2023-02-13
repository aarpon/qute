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

from pathlib import Path
from typing import Tuple, Union

import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet as MonaiUNet
from natsort import natsorted
from tifffile import TiffWriter
from torch.optim import AdamW

import qute


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
        channels: tuple = (16, 32, 64, 128, 256),
        strides: tuple = (2, 2, 2, 2),
        criterion=DiceCELoss(include_background=False, to_onehot_y=False, softmax=True),
        metrics=DiceMetric(include_background=False, reduction="mean"),
        post_activation=None,
        learning_rate: float = 1e-2,
        optimizer_class=AdamW,
        num_res_units: int = 0,
        dropout: float = 0.0,
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

        criterion: DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)
            Loss function. Please NOTE: for classification, the loss function must convert `y` to OneHot.
            The default loss function applies to a multi-label target where the background class is omitted.

        metrics: GeneralizedDiceScore(include_background=False)
            Metrics used for training, validation, and testing. Set to None to omit.

            The default metrics applies to a three-label target where the background (index = 0) class
            is omitted from calculation.

            Set to None to omit calculating and reporting it.

        post_activation: None
            Post activation for the output of the forward pass in the prediction step (only).

        learning_rate: float = 1e-2
            Learning rate for optimization.

        optimizer_class=AdamW
            Optimizer.

        num_res_units: int = 0
            Number of residual units for the UNet.

        dropout: float = 0.0
            Dropout ratio.
        """

        super().__init__()

        self.criterion = criterion
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.post_activation = post_activation
        self.net = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        x, y = batch
        y_hat = self.net(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # if self.metrics is not None:
        #     val_metrics = self.metrics(y_hat, y).mean()
        #     self.log("val_metrics", val_metrics, on_step=False, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        x, y = batch
        y_hat = self.net(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """The predict step creates a label image from the output one-hot tensor."""
        x, _ = batch
        y_hat = self.net(x)
        if self.post_activation is not None:
            y_hat = self.post_activation(y_hat)
        label = y_hat.argmax(axis=1)
        return label

    def full_predict(
        self,
        data_loader: DataLoader,
        target_folder: Union[Path, str],
        post_transforms: list,
        roi_size: Tuple[int, int],
        batch_size: int,
        transpose: bool = True,
    ):
        """Predict on passed images using given model.

        Parameters
        ----------

        input_folder: Union[Path|str]
            Path to the folder that contains the images to predicted from.

        target_folder: Union[Path|str]
            Path to the folder where to store the predicted images.

        model_path: Union[Path|str]
            Full path to the model to use.

        Returns
        -------

        result: bool
            True if the prediction was successful, False otherwise.
        """

        # Make sure the target folder exists
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Switch to evaluation mode
        self.net.eval()

        # Process them
        with torch.no_grad():
            for images, indices in data_loader:
                # Apply sliding inference over ROI size
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=batch_size,
                    predictor=self.net,
                    device=device,
                )

                # Apply post-transforms?
                outputs = post_transforms(outputs)

                # Retrieve the image from the GPU (if needed)
                preds = outputs.cpu().numpy().squeeze()

                for pred, index in zip(preds, indices):
                    if transpose:
                        # Transpose to undo the effect of monai.transform.LoadImage(d)
                        pred = pred.T

                    # Save prediction image as tiff file
                    output_name = Path(target_folder) / f"pred_{index.item():04}.tif"
                    with TiffWriter(output_name) as tif:
                        tif.save(pred)

                    # Inform
                    print(f"Saved {output_name}.")

        print(f"Prediction completed.")

        # Return success
        return True
