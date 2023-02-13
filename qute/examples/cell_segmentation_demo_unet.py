#  ********************************************************************************
#   Copyright © 2022-, ETH Zurich, D-BSSE, Aaron Ponti
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

import numpy as np
import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric, GeneralizedDiceScore
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tifffile import imwrite

from qute.data.dataloaders import CellSegmentationDemo
from qute.models.unet import UNet

SEED = 2022
BATCH_SIZE = (256, 256)

if __name__ == "__main__":
    # Seeding
    seed_everything(SEED, workers=True)

    # Data module
    data_module = CellSegmentationDemo(seed=SEED, batch_size=24, patch_size=BATCH_SIZE)

    # Loss
    criterion = DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)

    # Metrics
    metrics = DiceMetric(include_background=False, reduction="mean")

    # Model
    model = UNet(num_res_units=4, criterion=criterion, metrics=metrics)

    # Callbacks
    # early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    model_checkpoint = ModelCheckpoint(monitor="val_loss")

    # Instantiate the Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16 if torch.cuda.is_bf16_supported() else 32,
        callbacks=[model_checkpoint],
        max_epochs=5,
        log_every_n_steps=1,
    )
    trainer.logger._default_hp_metric = False

    # Find the best learning rate
    # trainer.tune(model, datamodule=data_module)

    # Train with the optimal learning rate found above
    trainer.fit(model, data_module)

    # Print path to best model
    print(f"Best model: {model_checkpoint.best_model_path}")

    # Load weights from best model
    model = UNet.load_from_checkpoint(model_checkpoint.best_model_path)

    # Test
    trainer.test(model, dataloaders=data_module.test_dataloader())

    # Predict on the test dataset
    predictions = trainer.predict(model, dataloaders=data_module.test_dataloader())

    # Save the full predictions (on the test set)
    model.full_predict(
        input_folder=data_module.data_dir / "images/",
        target_folder=data_module.data_dir
        / f"full_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}/",
        image_transforms=data_module.get_predict_transforms(),
        post_transforms=data_module.get_post_transforms(),
        roi_size=BATCH_SIZE,
        batch_size=4,
    )

    sys.exit(0)
