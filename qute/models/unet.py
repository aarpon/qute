#  ********************************************************************************
#   Copyright © 2022 - 2003, ETH Zurich, D-BSSE, Aaron Ponti
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
from monai.transforms import Transform
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
        channels=(16, 32, 64),
        strides=(2, 2),
        criterion=DiceCELoss(include_background=False, to_onehot_y=False, softmax=True),
        metrics=DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        ),
        val_metrics_transforms=None,
        test_metrics_transforms=None,
        predict_post_transforms=None,
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

        channels: tuple = (16, 32, 64)
            Number of neuron per layer.

        strides: tuple = (2, 2)
            Strides for down-sampling.

        criterion: DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)
            Loss function. Please NOTE: for classification, the loss function must convert `y` to OneHot.
            The default loss function applies to a multi-label target where the background class is omitted.

        metrics: DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
            Metrics used for validation and test. Set to None to omit.

            The default metrics applies to a three-label target where the background (index = 0) class
            is omitted from calculation.

        val_metrics_transforms: None
            Post transform for the output of the forward pass in the validation step for metric calculation.

        test_metrics_transforms: None
            Post transform for the output of the forward pass in the test step for metric calculation.

        predict_post_transforms: None
            Post transform for the output of the forward pass in the prediction step (only).

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
        self.val_metrics_transforms = val_metrics_transforms
        self.test_metrics_transforms = test_metrics_transforms
        self.predict_post_transforms = predict_post_transforms
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
        self.log(
            "val_loss",
            torch.tensor([val_loss]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        if self.metrics is not None:
            if self.val_metrics_transforms is not None:
                val_metrics = self.metrics(self.val_metrics_transforms(y_hat), y).mean()
            else:
                val_metrics = self.metrics(y_hat, y).mean()
            self.log(
                "val_metrics", torch.tensor([val_metrics]), on_step=False, on_epoch=True
            )
        return val_loss

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        x, y = batch
        y_hat = self.net(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)
        if self.metrics is not None:
            if self.test_metrics_transforms is not None:
                test_metrics = self.metrics(
                    self.test_metrics_transforms(y_hat), y
                ).mean()
            else:
                test_metrics = self.metrics(y_hat, y).mean()
            self.log(
                "test_metrics",
                torch.tensor([test_metrics]),
                on_step=False,
                on_epoch=True,
            )
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """The predict step creates a label image from the output one-hot tensor."""
        x, _ = batch
        y_hat = self.net(x)
        if self.predict_post_transforms is not None:
            label = self.predict_post_transforms(y_hat).argmax(axis=1)
        else:
            label = y_hat.argmax(axis=1)
        return label

    def full_inference(
        self,
        data_loader: DataLoader,
        target_folder: Union[Path, str],
        inference_post_transforms: Transform,
        roi_size: Tuple[int, int],
        batch_size: int,
        overlap: float = 0.25,
        transpose: bool = True,
    ):
        """Run inference on full images using given model.

        Parameters
        ----------

        data_loader: DataLoader
            DataLoader for the image files names to be predicted on.

        target_folder: Union[Path|str]
            Path to the folder where to store the predicted images.

        inference_post_transforms: Transform
            Composition of transforms to be applied to the result of the forward pass of the network.

        roi_size: Tuple[int, int]
            Size of the patch for the sliding window prediction. It must match the patch size during training.

        batch_size: int
            Number of parallel batches to run.

        overlap: float
            Fraction of overlap between rois.

        transpose: bool
            Whether the transpose the image before saving, to compensate for the default behavior of monai.transforms.LoadImage().

        Returns
        -------

        result: bool
            True if the inference was successful, False otherwise.
        """

        # Make sure the target folder exists
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        # Device
        device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Switch to evaluation mode
        self.net.eval()

        # Process them
        c = 0
        with torch.no_grad():
            for images in data_loader:
                # Apply sliding inference over ROI size
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=batch_size,
                    overlap=overlap,
                    predictor=self.net,
                    device=device,
                )

                # Apply post-transforms?
                outputs = inference_post_transforms(outputs)

                # Retrieve the image from the GPU (if needed)
                preds = outputs.cpu().numpy().squeeze()

                for pred in preds:
                    if transpose:
                        # Transpose to undo the effect of monai.transform.LoadImage(d)
                        pred = pred.T

                    # Save prediction image as tiff file
                    output_name = Path(target_folder) / f"pred_{c:04}.tif"
                    c += 1
                    with TiffWriter(output_name) as tif:
                        tif.write(pred)

                    # Inform
                    print(f"Saved {output_name}.")

        print(f"Prediction completed.")

        # Return success
        return True
