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
from typing import Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, FocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet as MonaiUNet
from monai.transforms import Transform
from monai.utils import BlendMode
from tifffile import TiffWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR

from qute.device import get_device


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
        strides: Optional[tuple] = None,
        criterion=DiceCELoss(include_background=True, to_onehot_y=False, softmax=True),
        metrics=DiceMetric(
            include_background=True, reduction="mean", get_not_nans=False
        ),
        val_metrics_transforms=None,
        test_metrics_transforms=None,
        predict_post_transforms=None,
        learning_rate: float = 1e-2,
        optimizer_class=AdamW,
        lr_scheduler_class=PolynomialLR,
        lr_scheduler_parameters: dict = {"total_iters": 100, "power": 0.95},
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

        strides: Optional[tuple] = (2, 2)
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

        lr_scheduler_class=PolynomialLR
            Learning rate scheduler.

        lr_scheduler_parameters={"total_iters": 100, "power": 0.99}
            Dictionary of scheduler parameters.

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
        self.scheduler_class = lr_scheduler_class
        self.scheduler_parameters = lr_scheduler_parameters
        self.val_metrics_transforms = val_metrics_transforms
        self.test_metrics_transforms = test_metrics_transforms
        self.predict_post_transforms = predict_post_transforms
        if strides is None:
            strides = (2,) * (len(channels) - 1)
        self.net = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
        )

        # Log the hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        """Configure and return the optimizer and scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": self.scheduler_class(optimizer, **self.scheduler_parameters),
            "monitor": "val_loss",
            "interval": "step",  # Call "scheduler.step()" after every batch (1 step)
            "frequency": 1,  # Update scheduler after every step
            "strict": True,  # Ensures the scheduler is strictly followed (PyTorch Lightning parameter)
        }
        return [optimizer], [scheduler]

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
        device = get_device()

        # Make sure the model is on the device
        self.net.to(device)

        # Switch to evaluation mode
        self.net.eval()

        # Process them
        c = 0
        with torch.no_grad():
            for images in data_loader:
                # Apply sliding inference over ROI size
                outputs = sliding_window_inference(
                    inputs=images.to(device),
                    roi_size=roi_size,
                    sw_batch_size=batch_size,
                    overlap=overlap,
                    predictor=self.net,
                    mode=BlendMode.GAUSSIAN,
                    sigma_scale=0.125,
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

    @staticmethod
    def full_inference_ensemble(
        models: list,
        data_loader: DataLoader,
        target_folder: Union[Path, str],
        inference_post_transforms: Transform,
        roi_size: Tuple[int, int],
        batch_size: int,
        voting_mechanism: str = "mode",
        weights: Optional[list] = None,
        overlap: float = 0.25,
        transpose: bool = True,
        save_individual_preds: bool = False,
    ):
        """Run inference on full images using given model.

        Parameters
        ----------

        models: list
            List of trained UNet models to use for ensemble prediction.

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

        voting_mechanism: str = "mode"
            Voting mechanism to assign the final class among the predictions from the ensemble of models.
            One of "mode" (default" and "mean").
            "mode": pick the most common class among the predictions for each pixel.
            "mean": (rounded) weighted mean of the predicted classed per pixel. The `weights` argument defines
                    the relative contribution of the models.

        weights: Optional[list]
            List of weights for each of the contributions. Only used if `voting_mechanism` is "mean".

        overlap: float
            Fraction of overlap between rois.

        transpose: bool
            Whether the transpose the image before saving, to compensate for the default behavior of monai.transforms.LoadImage().

        save_individual_preds: bool
            Whether to save the individual predictions of each model.

        Returns
        -------

        result: bool
            True if the inference was successful, False otherwise.
        """

        if voting_mechanism not in ["mode", "mean"]:
            raise ValueError("`voting mechanism` must be one of 'mode' or 'mean'.")

        if voting_mechanism == "mean":
            if len(models) != len(weights):
                raise ValueError(
                    "The number of weights must match the number of models."
                )

            # Turn the weights into a NumPy array of float 32 bit
            weights = np.array(weights, dtype=np.float32)
            weights = weights / weights.sum()

        if not isinstance(models[0], UNet) or not hasattr(models[0], "net"):
            raise ValueError("The models must be of type `qute.models.unet.UNet`.")

        # Make sure the target folder exists
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        # If needed, create the sub-folders for the individual predictions
        if save_individual_preds:
            for f in range(len(models)):
                fold_subfolder = Path(target_folder) / f"fold_{f}"
                Path(fold_subfolder).mkdir(parents=True, exist_ok=True)
        # Device
        device = get_device()

        # Switch to evaluation mode on all models
        for model in models:
            model.net.eval()

        c = 0
        with (torch.no_grad()):
            for images in data_loader:

                predictions = [[] for _ in range(len(models))]

                # Not process all models
                for n, model in enumerate(models):

                    # Make sure the model is on the device
                    model.to(device)

                    # Apply sliding inference over ROI size
                    outputs = sliding_window_inference(
                        inputs=images.to(device),
                        roi_size=roi_size,
                        sw_batch_size=batch_size,
                        overlap=overlap,
                        predictor=model.net,
                        device=device,
                    )

                    # Apply post-transforms?
                    outputs = inference_post_transforms(outputs)

                    # Retrieve the image from the GPU (if needed)
                    preds = outputs.cpu().numpy().squeeze()

                    stored_preds = []
                    for pred in preds:
                        if transpose:
                            # Transpose to undo the effect of monai.transform.LoadImage(d)
                            pred = pred.T
                        stored_preds.append(pred)

                    # Store
                    predictions[n] = stored_preds

                # Iterate over all images in the batch
                pred_dim = len(models)
                batch_dim = len(predictions[0])

                for b in range(batch_dim):

                    # Apply selected voting mechanism
                    if voting_mechanism == "mean":
                        # Apply weighted mean (and rounding) of the predictions per pixel

                        # Iterate over all predictions from the models
                        for p in range(pred_dim):
                            if p == 0:
                                ensemble_pred = weights[p] * predictions[p][b]
                            else:
                                ensemble_pred += weights[p] * predictions[p][b]
                        ensemble_pred = np.round(ensemble_pred).astype(np.int32)

                    elif voting_mechanism == "mode":
                        # Select the mode of the predictions per pixel

                        # Store predictions in a stack
                        target = np.zeros(
                            (
                                pred_dim,
                                predictions[0][0].shape[0],
                                predictions[0][0].shape[1],
                            )
                        )

                        # Iterate over all predictions from the models
                        for p in range(pred_dim):
                            target[p, :, :] = predictions[p][b]
                        values, _ = torch.mode(torch.tensor(target), dim=0)
                        ensemble_pred = values.numpy().astype(np.int32)
                    else:
                        raise ValueError(
                            "`voting mechanism` must be one of 'mode' or 'mean'."
                        )

                    # Save ensemble prediction image as tiff file
                    output_name = Path(target_folder) / f"ensemble_pred_{c:04}.tif"
                    with TiffWriter(output_name) as tif:
                        tif.write(ensemble_pred)

                    # Inform
                    print(f"Saved {output_name}.")

                    # Save individual predictions?
                    if save_individual_preds:
                        # Iterate over all predictions from the models
                        for p in range(len(predictions)):
                            # Save prediction image as tiff file
                            output_name = (
                                Path(target_folder) / f"fold_{p}" / f"pred_{c:04}.tif"
                            )
                            with TiffWriter(output_name) as tif:
                                tif.write(predictions[p][b])

                    # Update global file counter c
                    c += 1

        print(f"Ensemble prediction completed.")

        # Return success
        return True
