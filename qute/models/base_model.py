#  ********************************************************************************
#  Copyright © 2022 - 2024, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************

from pathlib import Path
from typing import Optional, Tuple, Union

import monai
import numpy as np
import pytorch_lightning as pl
import torch
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from monai.transforms import Transform
from monai.utils import BlendMode
from tifffile import TiffWriter
from torch import nn

from qute.campaigns import CampaignTransforms
from qute.device import get_device

__doc__ = "BaseModel class for common functionality."
__all__ = [
    "BaseModel",
]


class BaseModel(pl.LightningModule):
    """
    Base model class for UNet architectures, extending PyTorch Lightning's LightningModule.
    This class encapsulates shared functionalities and configurations for different UNet variants.

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
    ):
        """
        Constructor.

        Parameters
        ----------

        campaign_transforms: CampaignTransforms
            Define all transforms necessary for training, validation, testing and (full) prediction.
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

        """
        super().__init__()

        self.campaign_transforms = campaign_transforms
        self.criterion = criterion
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.optimizer_class = optimizer_class
        self.lr_scheduler_class = lr_scheduler_class
        self.scheduler_parameters = lr_scheduler_parameters
        self.class_names = class_names

        # Placeholder for the network, to be defined in subclasses
        self.net = None

        # Log the hyperparameters
        self.save_hyperparameters(ignore=["criterion", "metrics"])

    def configure_optimizers(self):
        """Configure and return the optimizer and scheduler."""
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": self.lr_scheduler_class(
                optimizer, **self.scheduler_parameters
            ),
            "monitor": "val_loss",
            "interval": "step",  # Call "scheduler.step()" after every batch (1 step)
            "frequency": 1,  # Update scheduler after every step
            "strict": True,  # Ensures the scheduler is strictly followed (PyTorch Lightning parameter)
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        """Forward method to be implemented in subclasses."""
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Perform a validation step."""
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.criterion(y_hat, y)

        # Log the loss
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Update the metrics if needed
        if self.metrics is not None:
            if self.campaign_transforms.get_val_metrics_transforms() is not None:
                y_hat_transformed = (
                    self.campaign_transforms.get_val_metrics_transforms()(y_hat)
                )
            else:
                y_hat_transformed = y_hat

            val_metrics = self.metrics(y_hat_transformed, y)

            # Compute and log the mean metrics score per class
            mean_val_per_class = val_metrics.nanmean(dim=0)

            # Do we have more than one output class?
            if self.class_names and len(self.class_names) > 1:

                # Make sure to log the correct class name in case the background is not
                # considered in the calculation
                start = len(self.class_names) - mean_val_per_class.shape[0]

                for i, val_score in enumerate(mean_val_per_class):
                    self.log(
                        f"val_metrics_{self.class_names[start + i]}",
                        val_score.detach(),
                        on_step=False,
                        on_epoch=True,
                    )
            else:
                self.log(
                    "val_metrics",
                    mean_val_per_class.mean().detach(),
                    on_step=False,
                    on_epoch=True,
                )

        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        x, y = batch
        y_hat = self.forward(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)
        if self.metrics is not None:
            if self.campaign_transforms.get_test_metrics_transforms() is not None:
                y_hat_transformed = (
                    self.campaign_transforms.get_test_metrics_transforms()(y_hat)
                )
            else:
                y_hat_transformed = y_hat

            test_metrics = self.metrics(y_hat_transformed, y)

            # Compute and log the mean metrics score per class
            mean_test_per_class = test_metrics.nanmean(dim=0)

            # Do we have more than one output class?
            if self.class_names and len(self.class_names) > 1:

                # Make sure to log the correct class name in case the background is not
                # considered in the calculation
                start = len(self.class_names) - mean_test_per_class.shape[0]
                for i, test_score in enumerate(mean_test_per_class):
                    self.log(
                        f"test_metrics_{self.class_names[start + i]}",
                        test_score.detach(),
                        on_step=False,
                        on_epoch=True,
                    )
            else:
                self.log(
                    "test_metrics",
                    mean_test_per_class.mean().detach(),
                    on_step=False,
                    on_epoch=True,
                )

        return {"test_loss": test_loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """The predict step creates a label image from the output tensor."""
        x, _ = batch
        y_hat = self.forward(x)
        if self.campaign_transforms.get_post_inference_transforms() is not None:
            label = self.campaign_transforms.get_post_inference_transforms()(y_hat)
        else:
            label = y_hat
        return label

    def full_inference(
        self,
        data_loader: DataLoader,
        target_folder: Union[Path, str],
        roi_size: Tuple[int, ...],
        batch_size: int,
        overlap: float = 0.25,
        transpose: bool = True,
        output_dtype: Optional[Union[str, np.dtype]] = None,
        prefix: str = "pred_",
    ):
        """Run inference on full images using given model.

        Parameters
        ----------

        data_loader: DataLoader
            DataLoader for the image files names to be predicted on.

        target_folder: Union[Path|str]
            Path to the folder where to store the predicted images.

        roi_size: Tuple[int, int]
            Size of the patch for the sliding window prediction. It must match the patch size during training.

        batch_size: int
            Number of parallel batches to run.

        overlap: float
            Fraction of overlap between rois.

        transpose: bool
            Whether to transpose the image before saving, to compensate for the default behavior of monai.transforms.LoadImage().

        output_dtype: Optional[np.dtype]
            Optional NumPy dtype for the output image. Omit to save the output of inference without casting.

        prefix: str = "pred_"
            Prefix to append to the file name. Set to "" to keep the original file name.

        Returns
        -------

        result: bool
            True if the inference was successful, False otherwise.
        """

        # Make sure the target folder exists
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        # Retrieve file names from the dataloader
        if not hasattr(data_loader.dataset, "dataset") or not hasattr(
            data_loader.dataset.dataset, "data"
        ):
            raise ValueError("Incompatible dataset!")
        input_file_names = data_loader.dataset.dataset.data
        if len(input_file_names) == 0:
            print("No input files provided to process. Quitting.")
            return

        # Device
        device = get_device()

        # Make sure the model is on the device
        self.net.to(device)

        # Switch to evaluation mode
        self.net.eval()

        # Instantiate the inferer
        sliding_window_inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=batch_size,
            overlap=overlap,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            device=device,
        )

        # Process all images
        c = 0
        with torch.no_grad():
            for images in data_loader:
                # Apply sliding inference over ROI size
                outputs = sliding_window_inferer(
                    inputs=images.to(device),
                    network=self.net,
                )

                # Apply post-transforms?
                outputs = self.campaign_transforms.get_post_full_inference_transforms()(
                    outputs
                )

                # Retrieve the image from the GPU (if needed)
                preds = outputs.cpu().numpy()

                # Process one batch at a time
                for pred in preds:

                    # Drop the channel singleton dimension
                    if pred.shape[0] == 1:
                        pred = pred.squeeze(0)

                    if transpose:
                        # Transpose to undo the effect of monai.transform.LoadImage(d)
                        pred = pred.T

                    # Type-cast if needed
                    if output_dtype is not None:
                        # Make sure not to wrap around
                        if np.issubdtype(output_dtype, np.integer):
                            info = np.iinfo(output_dtype)
                            pred[pred < info.min] = info.min
                            pred[pred > info.max] = info.max
                        pred = pred.astype(output_dtype)

                    # Save prediction image as tiff file
                    output_name = (
                        Path(target_folder) / f"{prefix}{input_file_names[c].stem}.tif"
                    )
                    c += 1
                    with TiffWriter(output_name) as tif:
                        tif.write(
                            pred, compression="zlib", compressionargs={"level": 9}
                        )

                    # Inform
                    print(f"Saved {output_name}.")

        print("Prediction completed.")

        # Return success
        return True

    @staticmethod
    def full_inference_ensemble(
        models: list,
        data_loader: DataLoader,
        target_folder: Union[Path, str],
        post_full_inference_transforms: Transform,
        roi_size: Tuple[int, ...],
        batch_size: int,
        voting_mechanism: str = "mode",
        weights: Optional[list] = None,
        overlap: float = 0.25,
        transpose: bool = True,
        save_individual_preds: bool = False,
        output_dtype: Optional[Union[str, np.dtype]] = None,
        prefix: str = "pred_",
        ensemble_prefix: str = "ensemble_",
    ):
        """Run inference on full images using an ensemble of models.

        Parameters
        ----------

        models: list
            List of trained models inheriting from BaseModel to use for ensemble prediction.

        data_loader: DataLoader
            DataLoader for the image files names to be predicted on.

        target_folder: Union[Path|str]
            Path to the folder where to store the predicted images.

        post_full_inference_transforms: Transform
            Composition of transforms to be applied to the result of the sliding window inference (whole image).

        roi_size: Tuple[int, int]
            Size of the patch for the sliding window prediction. It must match the patch size during training.

        batch_size: int
            Number of parallel batches to run.

        voting_mechanism: str = "mode"
            Voting mechanism to assign the final class among the predictions from the ensemble of models.
            One of "mode" (default) and "mean".
            "mode": pick the most common class among the predictions for each pixel.
            "mean": (rounded) weighted mean of the predicted classes per pixel. The `weights` argument defines
            the relative contribution of the models.

        weights: Optional[list]
            List of weights for each of the contributions. Only used if `voting_mechanism` is "mean".

        overlap: float
            Fraction of overlap between rois.

        transpose: bool
            Whether to transpose the image before saving, to compensate for the default behavior of
            monai.transforms.LoadImage().

        save_individual_preds: bool
            Whether to save the individual predictions of each model.

        output_dtype: Optional[np.dtype]
            Optional NumPy dtype for the output image. Omit to save the output of inference without casting.

        prefix: str = "pred_"
            Prefix to append to the file name. Set to "" to keep the original file name.

        ensemble_prefix: str = "ensemble_pred_"
            Prefix to append to the ensemble prediction file name. Set to "" to keep the original file name.

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

        # Check if models are instances of BaseModel
        if not isinstance(models[0], BaseModel) or not hasattr(models[0], "net"):
            raise ValueError(
                "The models must inherit from `BaseModel` and have a `net` attribute."
            )

        # Make sure the target folder exists
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        # Retrieve file names from the dataloader
        if not hasattr(data_loader.dataset, "dataset") or not hasattr(
            data_loader.dataset.dataset, "data"
        ):
            raise ValueError("Incompatible dataset!")
        input_file_names = data_loader.dataset.dataset.data
        if len(input_file_names) == 0:
            print("No input files provided to process. Quitting.")
            return False

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

        # Instantiate the inferer
        sliding_window_inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=batch_size,
            overlap=overlap,
            mode=BlendMode.GAUSSIAN,
            sigma_scale=0.125,
            device=device,
        )

        c = 0
        with torch.no_grad():
            for images in data_loader:

                predictions = [[] for _ in range(len(models))]

                # Not process all models
                for n, model in enumerate(models):

                    # Make sure the model is on the device
                    model.to(device)

                    # Apply sliding inference over ROI size
                    outputs = sliding_window_inferer(
                        inputs=images.to(device),
                        network=model.net,
                    )

                    # Apply post-transforms?
                    outputs = post_full_inference_transforms(outputs)

                    # Retrieve the image from the GPU (if needed)
                    preds = outputs.cpu().numpy()

                    stored_preds = []
                    for pred in preds:

                        # Drop the channel singleton dimension
                        if pred.shape[0] == 1:
                            pred = pred.squeeze(0)

                        if transpose:
                            # Transpose to undo the effect of monai.transform.LoadImage(d)
                            pred = pred.T
                        stored_preds.append(pred)

                    # Store predictions for each model
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

                    # Type-cast if needed
                    if output_dtype is not None:
                        # Make sure not to wrap around
                        if np.issubdtype(output_dtype, np.integer):
                            info = np.iinfo(output_dtype)
                            ensemble_pred[ensemble_pred < info.min] = info.min
                            ensemble_pred[ensemble_pred > info.max] = info.max
                        ensemble_pred = ensemble_pred.astype(output_dtype)

                    # Save ensemble prediction image as tiff file
                    output_name = (
                        Path(target_folder)
                        / f"{ensemble_prefix}{input_file_names[c].stem}.tif"
                    )
                    with TiffWriter(output_name) as tif:
                        tif.write(
                            ensemble_pred,
                            compression="zlib",
                            compressionargs={"level": 9},
                        )

                    # Inform
                    print(f"Saved {output_name}.")

                    # Save individual predictions?
                    if save_individual_preds:
                        # Iterate over all predictions from the models
                        for p in range(len(predictions)):
                            # Save prediction image as tiff file
                            output_name = (
                                Path(target_folder)
                                / f"fold_{p}"
                                / f"{prefix}{input_file_names[c].stem}.tif"
                            )

                            # Get current prediction
                            current_pred = predictions[p][b]

                            # Type-cast if needed
                            if output_dtype is not None:
                                # Make sure not to wrap around
                                if np.issubdtype(output_dtype, np.integer):
                                    info = np.iinfo(output_dtype)
                                    current_pred[current_pred < info.min] = info.min
                                    current_pred[current_pred > info.max] = info.max
                                current_pred = current_pred.astype(output_dtype)

                            # Save
                            with TiffWriter(output_name) as tif:
                                tif.write(
                                    current_pred,
                                    compression="zlib",
                                    compressionargs={"level": 9},
                                )

                    # Update global file counter c
                    c += 1

        print("Ensemble prediction completed.")

        # Return success
        return True

    @staticmethod
    def load_from_checkpoint_and_swap_output_layer(
        checkpoint_path: Union[Path, str],
        new_out_channels: int,
        new_campaign_transforms: CampaignTransforms,
        new_criterion,
        new_metrics,
        class_names: tuple[str, ...],
        previous_out_channels: int = 1,
        strict: bool = True,
        verbose: bool = False,
        map_location: Optional[torch.device] = None,
    ):
        """Load a model from a checkpoint and modify it by replacing the last Conv2d layer
        with a new Conv2d layer that has a specified number of output channels.

        Parameters
        ----------
        checkpoint_path: Union[Path, str]
            Full path to the checkpoint file to load the model from.

        new_out_channels: int
            The number of output channels for the new Conv2d layer.

        new_campaign_transforms: CampaignTransforms
            New CampaignTransforms for the loaded model.

        new_criterion: loss function
            New criterion for the loaded model.

        new_metrics: metrics
            New metrics for the loaded model.

        class_names: tuple[str, ...]
            Class names for the new outputs.

        previous_out_channels: int = 1
            Number of output channels in the last convolutional layer of the loaded model.

        strict: bool = True
            Set to True for strict loading of the model (all modules and parameters must match).

        verbose: bool = False
            Set to True for verbose info when scanning the model.

        map_location: Optional[Union[str, torch.device]]
            The device to map the model's weights to when loading the checkpoint. Default is None.

        Returns
        -------
        model: The model with the last Conv2d layer replaced by a new Conv2d layer with the specified number of output channels.
        """
        # Check inputs
        if new_out_channels != len(class_names):
            raise ValueError(
                f"Please provide a valid number of class names ({new_out_channels})."
            )

        # Load the model from checkpoint
        model = BaseModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            strict=strict,
            campaign_transforms=new_campaign_transforms,
            criterion=new_criterion,
            metrics=new_metrics,
        )

        # Debug: assert that the campaign was replaced
        assert model.campaign_transforms == new_campaign_transforms

        # Debug: assert that the criterion was replaced
        assert model.criterion == new_criterion

        # Debug: assert that the metrics was replaced
        assert model.metrics == new_metrics

        # List to store all matching Conv2d layers
        matching_layers = []

        # Helper function to collect all matching Conv2d layers
        def collect_matching_conv2d_layers(
            module: nn.Module,
            previous_out_channels: int,
            depth: int = 0,
            parent_name="",
        ):
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                if isinstance(child, nn.Conv2d):
                    if verbose:
                        print(
                            f"Found Conv2d layer ('{full_name}') with {child.out_channels} output channel(s)"
                        )
                    if child.out_channels == previous_out_channels:
                        matching_layers.append((module, name, full_name, child))
                else:
                    collect_matching_conv2d_layers(
                        child, previous_out_channels, depth + 1, full_name
                    )

        # Collect all matching Conv2d layers
        collect_matching_conv2d_layers(model, previous_out_channels)

        # Ensure we found at least one matching layer
        if not matching_layers:
            raise ValueError(
                f"No Conv2d layer with {previous_out_channels} channel(s) found."
            )

        # Get the last matching layer
        parent_module, name, full_name, last_conv_layer = matching_layers[-1]
        in_channels = last_conv_layer.in_channels
        out_channels = last_conv_layer.out_channels

        # Print the identified last Conv2d layer if needed
        if verbose:
            print(
                f"Found {len(matching_layers)} module(s) with {in_channels} input channel(s) and {out_channels} output channel(s)."
            )
            print(
                f"Replacing last Conv2d layer ('{full_name}') with {in_channels} input channel(s) and {new_out_channels} output channel(s)."
            )

        # Assert that the last Conv2d's output channels match the expected previous_out_channels
        assert (
            out_channels == previous_out_channels
        ), f"Expected last Conv2d output channels to be {previous_out_channels}, but got {out_channels}"

        # Create the new Conv2d layer and initialize its weights
        new_conv = nn.Conv2d(in_channels, new_out_channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(new_conv.weight)
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0)

        # Replace the last Conv2d layer
        setattr(parent_module, name, new_conv)

        # Set the new class names
        model.class_names = class_names

        # Update hyperparameters to reflect new output channels and class names
        model.hparams.out_channels = new_out_channels
        model.hparams.class_names = class_names

        # Log the updated hyperparameters (including criterion and metrics)
        model.save_hyperparameters()

        # Return the loaded and modified model
        return model
