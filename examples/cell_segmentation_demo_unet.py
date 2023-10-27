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

import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from qute.data.dataloaders import CellSegmentationDemo
from qute.models.unet import UNet

SEED = 2022
BATCH_SIZE = 4
INFERENCE_BATCH_SIZE = 4
NUM_PATCHES = 4
PATCH_SIZE = (512, 512)
PRECISION = 16 if torch.cuda.is_bf16_supported() else 32
MAX_EPOCHS = 250

if __name__ == "__main__":
    # Seeding
    seed_everything(SEED, workers=True)

    # Data module
    data_module = CellSegmentationDemo(
        seed=SEED,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        num_patches=NUM_PATCHES,
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    # Loss
    criterion = DiceCELoss(include_background=True, to_onehot_y=False, softmax=True)

    # Metrics
    metrics = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Model
    model = UNet(
        in_channels=2,
        out_channels=3,
        num_res_units=4,
        criterion=criterion,
        channels=(16, 32, 64),
        strides=(2, 2),
        metrics=metrics,
        val_metrics_transforms=data_module.get_val_metrics_transforms(),
        test_metrics_transforms=data_module.get_test_metrics_transforms(),
        learning_rate=1e-3,
    )

    # # Compile the model
    # model = torch.compile(model)

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min"
    )  # Issues with Lightning's ES
    model_checkpoint = ModelCheckpoint(monitor="val_loss")

    # Instantiate the Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=PRECISION,
        callbacks=[model_checkpoint, early_stopping],
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=1,
    )
    trainer.logger._default_hp_metric = False

    # Train with the optimal learning rate found above
    trainer.fit(model, datamodule=data_module)

    # Print path to best model
    print(f"Best model: {model_checkpoint.best_model_path}")

    # Load weights from best model
    model = UNet.load_from_checkpoint(model_checkpoint.best_model_path)

    # Test
    trainer.test(model, dataloaders=data_module.test_dataloader())

    # Predict on the test dataset
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())

    # Save the full predictions (on the test set)
    model.full_inference(
        data_loader=data_module.inference_dataloader(data_module.data_dir / "images/"),
        target_folder=data_module.data_dir
        / f"full_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}/",
        inference_post_transforms=data_module.get_post_inference_transforms(),
        roi_size=PATCH_SIZE,
        batch_size=INFERENCE_BATCH_SIZE,
        transpose=True,
    )

    sys.exit(0)
