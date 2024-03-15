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
from qute.campaigns import SegmentationCampaignTransforms3D
from qute.data.dataloaders import SegmentationDataModuleLocalFolder
from qute.models.unet import UNet

SEED = 2022
BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 1  # Set to 1 if the various images have different dimensions
NUM_PATCHES = 2
VOXEL_SIZE = (1.0, 0.241, 0.241)
TO_ISOTROPIC = True  # Resample data to isotropic resolution
UP_SCALE_Z = True  # Resample to higher number of planes in z
PATCH_SIZE = (16, 192, 192)  # Patch size used for training
LEARNING_RATE = 0.001
INCLUDE_BACKGROUND = False
CLASS_NAMES = ["background", "cell", "membrane"]
try:
    PRECISION = 16 if torch.cuda.is_bf16_supported() else 32
except AssertionError:
    PRECISION = 32
MAX_EPOCHS = 3
EXP_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = (
    Path("models").resolve() / EXP_NAME
)  # Set to the path you want to use to save models
RESULTS_DIR = (
    Path("results").resolve() / EXP_NAME
)  # Set to the path you want to use to save results


if __name__ == "__main__":
    # Seeding
    seed_everything(SEED, workers=True)

    # Initialize default, example Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms3D(
        num_classes=3,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        voxel_size=VOXEL_SIZE,
        to_isotropic=TO_ISOTROPIC,
        upscale_z=UP_SCALE_Z,
    )

    # Data module
    data_module = SegmentationDataModuleLocalFolder(
        campaign_transforms=campaign_transforms,
        data_dir=Path("data").resolve(),  # Point to the root of the data directory
        seed=SEED,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        images_sub_folder="images",
        labels_sub_folder="masks",
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    # Calculate the number of steps per epoch
    data_module.prepare_data()
    data_module.setup("train")
    steps_per_epoch = len(data_module.train_dataloader())

    # Loss
    criterion = DiceCELoss(
        include_background=INCLUDE_BACKGROUND, to_onehot_y=False, softmax=True
    )

    # Metrics
    metrics = DiceMetric(
        include_background=INCLUDE_BACKGROUND, reduction="mean", get_not_nans=False
    )

    # Learning rate scheduler
    lr_scheduler_class = OneCycleLR
    lr_scheduler_parameters = {
        "total_steps": steps_per_epoch * MAX_EPOCHS,
        "div_factor": 5.0,
        "max_lr": LEARNING_RATE,
        "pct_start": 0.5,  # Fraction of total_steps at which the learning rate starts decaying after reaching max_lr
        "anneal_strategy": "cos",
    }

    # Model
    model = UNet(
        campaign_transforms=campaign_transforms,
        in_channels=1,
        out_channels=3,
        spatial_dims=3,
        class_names=CLASS_NAMES,
        num_res_units=4,
        criterion=criterion,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        metrics=metrics,
        learning_rate=LEARNING_RATE,
        lr_scheduler_class=lr_scheduler_class,
        lr_scheduler_parameters=lr_scheduler_parameters,
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )  # Issues with Lightning's ES
    model_checkpoint = ModelCheckpoint(dirpath=MODEL_DIR, monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Instantiate the Trainer
    trainer = pl.Trainer(
        default_root_dir=RESULTS_DIR / EXP_NAME,
        accelerator=device.get_accelerator(),
        devices=1,
        precision=PRECISION,
        # callbacks=[model_checkpoint, early_stopping, lr_monitor],
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=1,
    )

    # Store parameters
    # trainer.hparams = { }
    trainer.logger._default_hp_metric = False

    # Train with the optimal learning rate found above
    trainer.fit(model, datamodule=data_module)

    # Print path to best model
    print(f"Best model: {model_checkpoint.best_model_path}")

    # Load weights from best model
    # For more flexibility, see:
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#initialize-with-other-parameters
    model = UNet.load_from_checkpoint(model_checkpoint.best_model_path)

    # Test
    trainer.test(model, dataloaders=data_module.test_dataloader())

    # Predict on the test dataset
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())

    # Save the full predictions (on the test set)
    model.full_inference(
        data_loader=data_module.inference_dataloader(data_module.data_dir / "images"),
        target_folder=RESULTS_DIR / "full_predictions",
        roi_size=PATCH_SIZE,
        batch_size=INFERENCE_BATCH_SIZE,
        transpose=False,
    )

    sys.exit(0)
