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

SEED = 2022
BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 4
NUM_PATCHES = 1
PATCH_SIZE = (640, 640)
LEARNING_RATE = 0.001
INCLUDE_BACKGROUND = True
CLASS_NAMES = ["background", "cell", "membrane"]
try:
    PRECISION = 16 if torch.cuda.is_bf16_supported() else 32
except AssertionError:
    PRECISION = 32
MAX_EPOCHS = 2000
EXP_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = Path(userpaths.get_my_documents()) / "qute" / "models" / EXP_NAME
RESULTS_DIR = Path(userpaths.get_my_documents()) / "qute" / "results" / EXP_NAME
NUM_FOLDS = 5

if __name__ == "__main__":
    # Seeding
    seed_everything(SEED, workers=True)

    # Initialize default, example Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms2D(
        num_classes=3, patch_size=PATCH_SIZE, num_patches=NUM_PATCHES
    )

    # Data module: set up for k-fold cross validation (k = NUM_FOLDS)
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms,
        seed=SEED,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    # Calculate the number of steps per epoch
    data_module.prepare_data()
    data_module.setup("train")
    steps_per_epoch = len(data_module.train_dataloader())

    # Loss
    criterion = DiceCELoss(include_background=True, to_onehot_y=False, softmax=True)

    # Metrics
    metrics = DiceMetric(
        include_background=True, reduction="mean_batch", get_not_nans=False
    )

    # Run training with 5-fold cross-validation
    for fold in range(NUM_FOLDS):

        # Set the fold for current training
        data_module.set_fold(fold)

        # Update steps per epoch
        steps_per_epoch = len(data_module.train_dataloader())

        # Learning rate scheduler
        lr_scheduler_class = OneCycleLR
        lr_scheduler_parameters = {
            "total_steps": steps_per_epoch * MAX_EPOCHS,
            "div_factor": 5.0,
            "max_lr": LEARNING_RATE,
            "pct_start": 0.5,  # Fraction of total_steps at which the learning rate starts decaying after reaching max_lr
            "anneal_strategy": "cos",
        }

        # Initialize new UNet model
        model = UNet(
            campaign_transforms=campaign_transforms,
            in_channels=1,
            out_channels=3,
            class_names=CLASS_NAMES,
            num_res_units=4,
            criterion=criterion,
            channels=(16, 32, 64),
            strides=(2, 2),
            metrics=metrics,
            learning_rate=LEARNING_RATE,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_parameters=lr_scheduler_parameters,
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, mode="min"
        )  # Issues with Lightning's ES
        model_checkpoint = ModelCheckpoint(
            dirpath=MODEL_DIR / f"fold_{fold}", monitor="val_loss"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Instantiate the Trainer
        trainer = pl.Trainer(
            default_root_dir=RESULTS_DIR / f"fold_{fold}",
            accelerator=device.get_accelerator(),
            devices=1,
            precision=PRECISION,
            # callbacks=[model_checkpoint, early_stopping, lr_monitor],
            callbacks=[model_checkpoint, lr_monitor],
            max_epochs=MAX_EPOCHS,
            log_every_n_steps=1,
            val_check_interval=1.0,  # Run validation every epoch
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
    for fold in range(NUM_FOLDS):

        # Look for the model for current fold
        found = list(MODEL_DIR.glob(f"fold_{fold}/*.ckpt"))
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

    print(f"Loaded {len(models)} of {NUM_FOLDS} trained models.")

    # Run ensemble prediction
    # @TODO Weigh the predictions by the final validation metrics
    UNet.full_inference_ensemble(
        models,
        data_loader=data_module.inference_dataloader(data_module.data_dir / "images/"),
        target_folder=RESULTS_DIR / "ensemble_predictions",
        post_full_inference_transforms=campaign_transforms.get_post_full_inference_transforms(),
        roi_size=PATCH_SIZE,
        batch_size=INFERENCE_BATCH_SIZE,
        transpose=False,
        save_individual_preds=True,
        voting_mechanism="mode",
        weights=None,
    )

    sys.exit(0)
