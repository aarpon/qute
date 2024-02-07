#  ********************************************************************************
#   Copyright © 2022 - 2003, ETH Zurich, D-BSSE, Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the Apache License Version 2.0
#   which accompanies this distribution, and is available at
#   https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Contributors:
#       Aaron Ponti - initial API and implementation
#  ******************************************************************************/
import os
import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import userpaths
from monai.losses import DiceCELoss, FocalLoss
from monai.metrics import DiceMetric
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from qute.data.demos import CellSegmentationDemo
from qute.models.unet import UNet

SEED = 2022
BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 4
NUM_PATCHES = 1
PATCH_SIZE = (640, 640)
PRECISION = 16 if torch.cuda.is_bf16_supported() else 32
MAX_EPOCHS = 250
EXP_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = Path(userpaths.get_my_documents()) / "qute" / "models" / EXP_NAME
RESULTS_DIR = Path(userpaths.get_my_documents()) / "qute" / "results" / EXP_NAME
NUM_FOLDS = 5

if __name__ == "__main__":
    # Seeding
    seed_everything(SEED, workers=True)

    # Data module: set up for k-fold cross validation (k = NUM_FOLDS)
    data_module = CellSegmentationDemo(
        seed=SEED,
        num_folds=NUM_FOLDS,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        inference_batch_size=INFERENCE_BATCH_SIZE,
        num_workers=os.cpu_count() - 1,
    )

    # Run the prepare/setup steps
    data_module.prepare_data()
    data_module.setup(stage="train")

    # Loss
    criterion = DiceCELoss(include_background=True, to_onehot_y=False, softmax=True)

    # Metrics
    metrics = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Run training with 5-fold cross-validation
    for fold in range(NUM_FOLDS):

        # Set the fold for current training
        data_module.set_fold(fold)

        # Initialize new UNet model
        model = UNet(
            in_channels=1,
            out_channels=3,
            num_res_units=4,
            criterion=criterion,
            channels=(16, 32, 64),
            strides=(2, 2),
            metrics=metrics,
            val_metrics_transforms=data_module.get_val_metrics_transforms(),
            test_metrics_transforms=data_module.get_test_metrics_transforms(),
            learning_rate=1e-2,
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
            default_root_dir=RESULTS_DIR / EXP_NAME / f"fold_{fold}",
            accelerator="gpu",
            devices=1,
            precision=PRECISION,
            callbacks=[model_checkpoint, early_stopping, lr_monitor],
            max_epochs=MAX_EPOCHS,
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
        [1.0] * len(models),
        data_loader=data_module.inference_dataloader(data_module.data_dir / "images/"),
        target_folder=RESULTS_DIR / "ensemble_predictions",
        inference_post_transforms=data_module.get_post_inference_transforms(),
        roi_size=PATCH_SIZE,
        batch_size=INFERENCE_BATCH_SIZE,
        transpose=False,
        save_individual_preds=True,
    )

    sys.exit(0)
