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

import numpy as np
import pytorch_lightning as pl
import torch
import userpaths
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.nn import MSELoss
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import MeanAbsoluteError

from qute import device
from qute.campaigns import RestorationCampaignTransforms
from qute.data.demos import CellRestorationDemo
from qute.models.unet import UNet

SEED = 2022
BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 4
NUM_PATCHES = 1
PATCH_SIZE = (640, 640)
LEARNING_RATE = 0.001
try:
    PRECISION = 16 if torch.cuda.is_bf16_supported() else 32
except AssertionError:
    PRECISION = 32
MAX_EPOCHS = 2000
EXP_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = Path(userpaths.get_my_documents()) / "qute" / "models" / EXP_NAME
RESULTS_DIR = Path(userpaths.get_my_documents()) / "qute" / "results" / EXP_NAME

if __name__ == "__main__":
    # Seeding
    seed_everything(SEED, workers=True)

    # Initialize default, example Restoration Campaign Transform
    campaign_transforms = RestorationCampaignTransforms(
        min_intensity=0,
        max_intensity=15472,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
    )

    # Data module
    data_module = CellRestorationDemo(
        campaign_transforms=campaign_transforms,
        seed=SEED,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        inference_batch_size=INFERENCE_BATCH_SIZE,
        labels_sub_folder="targets",
    )

    # Loss
    criterion = MSELoss()

    # Metrics
    metrics = MeanAbsoluteError()

    # Learning rate scheduler
    lr_scheduler_class = OneCycleLR
    lr_scheduler_parameters = {
        "total_steps": 8 * MAX_EPOCHS,  # Steps per epoch in this case is 8.
        "div_factor": 5.0,
        "max_lr": LEARNING_RATE,
        "pct_start": 0.5,  # Fraction of total_steps at which the learning rate starts decaying after reaching max_lr
        "anneal_strategy": "cos",
    }

    # Model
    model = UNet(
        campaign_transforms=campaign_transforms,
        in_channels=1,
        out_channels=1,
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
        data_loader=data_module.inference_dataloader(data_module.data_dir / "images/"),
        target_folder=RESULTS_DIR / "full_predictions",
        roi_size=PATCH_SIZE,
        batch_size=INFERENCE_BATCH_SIZE,
        transpose=False,
        output_dtype=np.uint16,
    )

    sys.exit(0)
