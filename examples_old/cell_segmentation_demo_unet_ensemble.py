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

import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import userpaths
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.optim.lr_scheduler import OneCycleLR

from qute import device
from qute.campaigns import SegmentationCampaignTransforms2D
from qute.data.demos import CellSegmentationDemo
from qute.models.unet import UNet

torch.set_float32_matmul_precision("medium")

# Configuration
exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
CONFIG = {
    "seed": 2022,
    "num_folds": 5,
    "batch_size": 8,
    "inference_batch_size": 4,
    "num_classes": 3,
    "num_patches": 1,
    "patch_size": (640, 640),
    "learning_rate": 0.001,
    "include_background": True,
    "class_names": ["background", "cell", "membrane"],
    "max_epochs": 2000,
    "precision": "16-mixed",
    "models_dir": Path(userpaths.get_my_documents()) / "qute" / exp_name / "models",
    "results_dir": Path(userpaths.get_my_documents()) / "qute" / exp_name / "results",
}

if __name__ == "__main__":
    # Seeding
    seed_everything(CONFIG["seed"], workers=True)

    # Initialize default, example Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms2D(
        num_classes=CONFIG["num_classes"],
        patch_size=CONFIG["patch_size"],
        num_patches=CONFIG["num_patches"],
    )

    # Data module: set up for k-fold cross validation (k = CONFIG["num_folds"])
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms,
        seed=CONFIG["seed"],
        num_folds=CONFIG["num_folds"],
        batch_size=CONFIG["batch_size"],
        patch_size=CONFIG["patch_size"],
        num_patches=CONFIG["num_patches"],
        inference_batch_size=CONFIG["inference_batch_size"],
    )

    # Calculate the number of steps per epoch
    data_module.prepare_data()
    data_module.setup("train")
    steps_per_epoch = len(data_module.train_dataloader())

    # Loss
    criterion = DiceCELoss(
        include_background=CONFIG["include_background"], to_onehot_y=False, softmax=True
    )

    # Metrics
    metrics = DiceMetric(
        include_background=CONFIG["include_background"],
        reduction="mean_batch",
        get_not_nans=False,
    )

    # Run training with 5-fold cross-validation
    for fold in range(CONFIG["num_folds"]):

        # Set the fold for current training
        data_module.set_fold(fold)

        # Update steps per epoch
        steps_per_epoch = len(data_module.train_dataloader())

        # Learning rate scheduler
        lr_scheduler_class = OneCycleLR
        lr_scheduler_parameters = {
            "total_steps": steps_per_epoch * CONFIG["max_epochs"],
            "div_factor": 5.0,
            "max_lr": CONFIG["learning_rate"],
            "pct_start": 0.5,  # Fraction of total_steps at which the learning rate starts decaying after reaching max_lr
            "anneal_strategy": "cos",
        }

        # Initialize new UNet model
        model = UNet(
            campaign_transforms=campaign_transforms,
            in_channels=1,
            out_channels=CONFIG["num_classes"],
            class_names=CONFIG["class_names"],
            num_res_units=4,
            criterion=criterion,
            channels=(16, 32, 64),
            strides=(2, 2),
            metrics=metrics,
            learning_rate=CONFIG["learning_rate"],
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_parameters=lr_scheduler_parameters,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, mode="min"
        )  # Issues with Lightning's ES
        model_checkpoint = ModelCheckpoint(
            dirpath=CONFIG["models_dir"] / f"fold_{fold}", monitor="val_loss"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Instantiate the Trainer
        trainer = pl.Trainer(
            default_root_dir=CONFIG["results_dir"] / f"fold_{fold}",
            accelerator=device.get_accelerator(),
            devices=1,
            precision=CONFIG["precision"],
            # callbacks=[model_checkpoint, early_stopping, lr_monitor],
            callbacks=[model_checkpoint, lr_monitor],
            max_epochs=CONFIG["max_epochs"],
            log_every_n_steps=1,
        )

        # Store parameters
        # trainer.hparams = { }
        trainer.logger._default_hp_metric = False

        # Train with the optimal learning rate found above
        trainer.fit(model, datamodule=data_module)

        # Print path to best model
        print(f"Fold {fold}: best model = {model_checkpoint.best_model_path}")

    # @TODO - Check the final validation metrics for each model

    # Re-load all (best) models
    models = []
    for fold in range(CONFIG["num_folds"]):

        # Look for the model for current fold
        found = list(CONFIG["models_dir"].glob(f"fold_{fold}/*.ckpt"))
        if len(found) == 0:
            print(f"Could not find trained model for fold {fold}!")
            continue

        # Try loading the model
        try:
            model = UNet.load_from_checkpoint(found[0])
        except:
            print(f"Could not load the trained model {found[0]} for fold {fold}!")
            continue

        # Add it to the list
        models.append(model)

        # Inform
        print(f"Fold {fold}: re-loaded model = {found[0]}")

    print(f"Loaded {len(models)} of {CONFIG['num_folds']} trained models.")

    # Run ensemble prediction
    # @TODO Weigh the predictions by the final validation metrics
    UNet.full_inference_ensemble(
        models,
        data_loader=data_module.inference_dataloader(data_module.data_dir / "images/"),
        target_folder=CONFIG["results_dir"] / "ensemble_predictions",
        post_full_inference_transforms=campaign_transforms.get_post_full_inference_transforms(),
        roi_size=CONFIG["patch_size"],
        batch_size=CONFIG["inference_batch_size"],
        transpose=False,
        save_individual_preds=True,
        voting_mechanism="mode",
        weights=None,
    )

    sys.exit(0)
