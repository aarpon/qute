# ******************************************************************************
# Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Apache License Version 2.0
# which accompanies this distribution, and is available at
# https://www.apache.org/licenses/LICENSE-2.0.txt
#
# Contributors:
#   Aaron Ponti - initial API and implementation
# ******************************************************************************

import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import userpaths
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.nn import MSELoss
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import MeanAbsoluteError

from qute import device
from qute.callbacks import ProgressiveUnfreezeCallback
from qute.campaigns import (
    SegmentationCampaignTransforms2D,
    SelfSupervisedRestorationCampaignTransforms,
)
from qute.data.demos import CellRestorationDemo, CellSegmentationDemo
from qute.models.swinunetr import SwinUNETR
from qute.random import set_global_rng_seed

torch.set_float32_matmul_precision("medium")

# Configuration
exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
CONFIG = {
    "seed": 2022,
    "batch_size": 8,
    "inference_batch_size": 4,
    "num_patches": 1,
    "patch_size": (640, 640),
    "learning_rate": 0.001,
    "in_channels": 1,
    "self_supervised_out_channels": 1,
    "classification_out_channels": 3,
    "include_background": True,
    "class_names": ["background", "cell", "membrane"],
    "self_supervised_max_epochs": 1000,
    "classification_max_epochs": 1000,
    "precision": "16-mixed",
    "models_dir": Path(userpaths.get_my_documents()) / "qute" / exp_name / "models",
    "results_dir": Path(userpaths.get_my_documents()) / "qute" / exp_name / "results",
}

if __name__ == "__main__":
    # Seeding
    set_global_rng_seed(CONFIG["seed"])

    # -----------------------------------------------------------------------------------------#
    #                                                                                         #
    # Self-supervised learning (restoration)                                                  #
    #                                                                                         #
    # -----------------------------------------------------------------------------------------#

    # Initialize default Self-Supervised Restoration Campaign Transforms
    self_supervised_campaign_transforms = SelfSupervisedRestorationCampaignTransforms(
        patch_size=CONFIG["patch_size"],
        num_patches=CONFIG["num_patches"],
    )

    # Data module
    self_supervised_data_module = CellRestorationDemo(
        campaign_transforms=self_supervised_campaign_transforms,
        seed=CONFIG["seed"],
        batch_size=CONFIG["batch_size"],
        patch_size=CONFIG["patch_size"],
        num_patches=CONFIG["num_patches"],
        inference_batch_size=CONFIG["inference_batch_size"],
    )

    # Calculate the number of steps per epoch
    self_supervised_data_module.prepare_data()
    self_supervised_data_module.setup("train")
    self_supervised_steps_per_epoch = len(
        self_supervised_data_module.train_dataloader()
    )

    # Loss
    self_supervised_criterion = MSELoss()

    # Metrics
    self_supervised_metrics = MeanAbsoluteError()

    # Learning rate scheduler
    self_supervised_lr_scheduler_class = OneCycleLR
    self_supervised_lr_scheduler_parameters = {
        "total_steps": self_supervised_steps_per_epoch
        * CONFIG["self_supervised_max_epochs"],
        "div_factor": 5.0,
        "max_lr": CONFIG["learning_rate"],
        "pct_start": 0.5,  # Fraction of total_steps at which the learning rate starts decaying after reaching max_lr
        "anneal_strategy": "cos",
    }

    # Initialize self-supervised model
    self_supervised_model = SwinUNETR(
        campaign_transforms=self_supervised_campaign_transforms,
        criterion=self_supervised_criterion,
        metrics=self_supervised_metrics,
        lr_scheduler_class=self_supervised_lr_scheduler_class,
        lr_scheduler_parameters=self_supervised_lr_scheduler_parameters,
        learning_rate=CONFIG["learning_rate"],
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["self_supervised_out_channels"],
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )

    # Training checkpoints
    self_supervised_model_checkpoint = ModelCheckpoint(
        dirpath=Path(userpaths.get_my_documents()) / "qute" / exp_name / "models",
        monitor="loss",
        mode="min",
        verbose=True,
    )
    self_supervised_lr_monitor = LearningRateMonitor(logging_interval="step")

    # Instantiate the Trainer
    self_supervised_trainer = pl.Trainer(
        default_root_dir=CONFIG["results_dir"],
        accelerator=device.get_accelerator(),
        devices=1,
        precision=CONFIG["precision"],
        # callbacks=[model_checkpoint, early_stopping, lr_monitor],
        callbacks=[self_supervised_model_checkpoint, self_supervised_lr_monitor],
        max_epochs=CONFIG["self_supervised_max_epochs"],
        log_every_n_steps=1,
    )

    # Train the model
    self_supervised_trainer.fit(self_supervised_model, self_supervised_data_module)

    # Reload the best model
    best_model = SwinUNETR.load_from_checkpoint(
        self_supervised_model_checkpoint.best_model_path,
        strict=False,
        criterion=self_supervised_criterion,
        metrics=self_supervised_metrics,
    )

    # Save the encoder weights for the best model
    self_supervised_encoder_weights_path = (
        Path(self_supervised_model_checkpoint.best_model_path).parent
        / "encoder_weights.pth"
    )
    best_model.save_encoder_weights(self_supervised_encoder_weights_path)

    # Test
    self_supervised_trainer.test(
        best_model, dataloaders=self_supervised_data_module.test_dataloader()
    )

    # -----------------------------------------------------------------------------------------#
    #                                                                                         #
    # Classification fine-tuning                                                              #
    #                                                                                         #
    # -----------------------------------------------------------------------------------------#

    # Initialize default Segmentation Campaign Transforms
    classification_campaign_transforms = SegmentationCampaignTransforms2D(
        patch_size=CONFIG["patch_size"],
        num_patches=CONFIG["num_patches"],
    )

    # Data module
    classification_data_module = CellSegmentationDemo(
        campaign_transforms=classification_campaign_transforms,
        seed=CONFIG["seed"],
        batch_size=CONFIG["batch_size"],
        patch_size=CONFIG["patch_size"],
        num_patches=CONFIG["num_patches"],
        inference_batch_size=CONFIG["inference_batch_size"],
    )

    # Metrics
    classification_metrics = DiceMetric(
        include_background=CONFIG["include_background"],
        reduction="mean_batch",
        get_not_nans=False,
    )

    # Set up loss function
    classification_criterion = DiceCELoss(
        include_background=CONFIG["include_background"],
        to_onehot_y=False,
        softmax=True,
    )

    # Calculate the number of steps per epoch
    classification_data_module.prepare_data()
    classification_data_module.setup("train")
    classification_steps_per_epoch = len(classification_data_module.train_dataloader())

    # Learning rate scheduler
    classification_lr_scheduler_class = OneCycleLR
    classification_lr_scheduler_parameters = {
        "total_steps": classification_steps_per_epoch
        * CONFIG["classification_max_epochs"],
        "div_factor": 5.0,
        "max_lr": CONFIG["learning_rate"],
        "pct_start": 0.5,  # Fraction of total_steps at which the learning rate starts decaying after reaching max_lr
        "anneal_strategy": "cos",
    }

    # Initialize classification model
    classification_model = SwinUNETR(
        campaign_transforms=classification_campaign_transforms,
        criterion=classification_criterion,
        metrics=classification_metrics,
        lr_scheduler_class=classification_lr_scheduler_class,
        lr_scheduler_parameters=classification_lr_scheduler_parameters,
        learning_rate=CONFIG["learning_rate"],
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["classification_out_channels"],
        feature_size=24,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
    )

    # Load pre-trained encoder weights
    classification_model.load_encoder_weights(self_supervised_encoder_weights_path)

    # Freeze encoder layers
    classification_model.freeze_encoder()

    # Training checkpoints
    classification_model_checkpoint = ModelCheckpoint(
        dirpath=Path(userpaths.get_my_documents()) / "qute" / exp_name / "models",
        monitor="loss",
        mode="min",
        verbose=True,
    )
    classification_lr_monitor = LearningRateMonitor(logging_interval="step")
    progressive_unfreeze = ProgressiveUnfreezeCallback(
        encoder=classification_model.net.swinViT,  # Encoder
        start_epoch=int(CONFIG["classification_max_epochs"] * 0.1),
        end_epoch=int(CONFIG["classification_max_epochs"] * 0.7),
        max_epochs=int(CONFIG["classification_max_epochs"]),
        unfreeze_strategy="linear",
    )

    # Instantiate the new Trainer
    classification_trainer = pl.Trainer(
        default_root_dir=CONFIG["results_dir"],
        accelerator=device.get_accelerator(),
        devices=1,
        precision=CONFIG["precision"],
        # callbacks=[model_checkpoint, early_stopping, lr_monitor],
        callbacks=[
            classification_model_checkpoint,
            classification_lr_monitor,
            progressive_unfreeze,
        ],
        max_epochs=CONFIG["classification_max_epochs"],
        log_every_n_steps=1,
    )

    # Fine-tune the model
    classification_trainer.fit(classification_model, classification_data_module)

    # Keep track of the best model checkpoint
    best_model_checkpoint_path = classification_model_checkpoint.best_model_path

    # Reload model
    best_model = SwinUNETR.load_from_checkpoint(
        best_model_checkpoint_path,
        strict=False,
        criterion=classification_criterion,
        metrics=classification_metrics,
        class_names=CONFIG["class_names"],
    )

    # Test
    classification_trainer.test(
        best_model, dataloaders=classification_data_module.test_dataloader()
    )

    # -----------------------------------------------------------------------------------------#
    #                                                                                          #
    # Prediction with fine-tuned classifier                                                    #
    #                                                                                          #
    # -----------------------------------------------------------------------------------------#

    # Save the full predictions (on the test set)
    best_model.full_inference(
        data_loader=classification_data_module.inference_dataloader(
            classification_data_module.data_dir / "images/"
        ),
        target_folder=CONFIG["results_dir"] / "full_predictions",
        roi_size=CONFIG["patch_size"],
        batch_size=CONFIG["inference_batch_size"],
        transpose=False,
    )

    sys.exit(0)
