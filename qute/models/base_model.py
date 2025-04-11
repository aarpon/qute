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
import pytorch_lightning as pl
import torch

from qute.campaigns import CampaignTransforms

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
        lr_scheduler_class: Optional[
            torch.optim.lr_scheduler
        ] = torch.optim.lr_scheduler.LambdaLR,
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
            The learning rate scheduler class to use. Set to None to use a fixed learning rate.

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
        if self.lr_scheduler_class is None:
            return optimizer

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
            on_step=True,
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
                        on_step=True,
                        on_epoch=True,
                    )
            else:
                self.log(
                    "val_metrics",
                    mean_val_per_class.mean().detach(),
                    on_step=True,
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
