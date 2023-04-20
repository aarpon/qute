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

import numpy as np
import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

from qute.data.dataloaders import CellSegmentationDemo
from qute.models.unet import UNet

SEED = 2022
BATCH_SIZE = 24
INFERENCE_BATCH_SIZE = 4
PATCH_SIZE = (512, 512)
PRECISION = 16 if torch.cuda.is_bf16_supported() else 32
MAX_EPOCHS = 250


def train_fn(
    config, data_module, criterion, metrics, num_epochs=MAX_EPOCHS, num_gpus=1
):
    # Get current configuration parameters
    num_res_units = config["num_res_units"]
    learning_rate = config["learning_rate"]

    # Instantiate the model
    model = UNet(
        num_res_units=num_res_units,
        criterion=criterion,
        channels=(16, 32, 64),
        strides=(2, 2),
        metrics=metrics,
        val_metrics_transforms=data_module.get_val_metrics_transforms(),
        learning_rate=learning_rate,
    )

    # Tune report callback
    report_callback = TuneReportCallback(
        {"loss": "val_loss", "mean_accuracy": "val_metrics"}, on="validation_end"
    )

    # Instantiate the Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        precision=PRECISION,
        callbacks=[report_callback],
        logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        max_epochs=num_epochs,
        log_every_n_steps=1,
    )
    trainer.logger._default_hp_metric = False

    trainer.fit(model, datamodule=data_module)


def tune_fn(data_module, criterion, metrics, num_samples=10, num_epochs=MAX_EPOCHS):
    config = {
        "num_res_units": tune.choice([2, 3, 4, 5]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["num_res_units", "learning_rate"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_fn,
        data_module=data_module,
        criterion=criterion,
        metrics=metrics,
        num_epochs=num_epochs,
        num_gpus=1,
    )
    resources_per_trial = {"cpu": 1, "gpu": 1}

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_parameters, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name="tune_fn",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    # Seeding
    seed_everything(SEED, workers=True)

    # Data module
    data_module = CellSegmentationDemo(
        seed=SEED,
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    # Loss
    criterion = DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)

    # Metrics
    metrics = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # Run the optimization
    results = tune_fn(
        data_module, criterion, metrics, num_samples=10, num_epochs=MAX_EPOCHS
    )

    # Report
    print(f"Best hyper-parameters found: {results.get_best_result().config}")

    sys.exit(0)
