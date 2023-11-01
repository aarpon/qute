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

from qute.data.demos import CellSegmentationDemo
from qute.models.unet import UNet

#
# See:
#
# Using PyTorch Lightning with Tune: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
# Tune search algorithms: https://docs.ray.io/en/latest/tune/api/suggestion.html
#

SEED = 2022
BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 4
PATCH_SIZE = (512, 512)
PRECISION = 16 if torch.cuda.is_bf16_supported() else 32
MAX_EPOCHS = 250


def train_fn(config, criterion, metrics, num_epochs=MAX_EPOCHS, num_gpus=1):
    # Get current configuration parameters
    num_res_units = config["num_res_units"]
    learning_rate = config["learning_rate"]
    channels = config["channels"]
    dropout = config["dropout"]
    patch_size = config["patch_size"]
    num_patches = config["num_patches"]
    batch_size = config["batch_size"]

    # Instantiate data module
    data_module = CellSegmentationDemo(
        seed=SEED,
        batch_size=batch_size,
        patch_size=patch_size,
        num_patches=num_patches,
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    # Instantiate the model
    model = UNet(
        in_channels=2,
        out_channels=3,
        num_res_units=num_res_units,
        criterion=criterion,
        channels=channels,
        strides=None,
        metrics=metrics,
        val_metrics_transforms=data_module.get_val_metrics_transforms(),
        test_metrics_transforms=data_module.get_test_metrics_transforms(),
        learning_rate=learning_rate,
        dropout=dropout,
    )

    # Tune report callback
    report_callback = TuneReportCallback(
        {"loss": "val_loss", "dice": "val_metrics"}, on="validation_end"
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


def tune_fn(criterion, metrics, num_samples=10, num_epochs=MAX_EPOCHS):
    config = {
        "num_res_units": tune.choice([0, 1, 2, 3, 4]),
        "learning_rate": tune.loguniform(0.0005, 0.5),
        "channels": tune.choice([(16, 32), (16, 32, 64), (32, 64), (32, 64, 128)]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "patch_size": tune.choice([(256, 256), (384, 384), (512, 512)]),
        "num_patches": tune.choice([1, 2, 4, 8]),
        "batch_size": tune.choice([1, 2, 4, 8]),
    }

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[
            "num_res_units",
            "learning_rate",
            "channels",
            "dropout",
            "patch_size",
            "num_patches",
            "batch_size",
        ],
        metric_columns=["loss", "dice", "training_iteration"],
    )

    train_fn_with_parameters = tune.with_parameters(
        train_fn,
        criterion=criterion,
        metrics=metrics,
        num_epochs=num_epochs,
        num_gpus=1,
    )
    resources_per_trial = {"cpu": 0, "gpu": 1}

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

    # Loss
    criterion = DiceCELoss(include_background=False, to_onehot_y=False, softmax=True)

    # Metrics
    metrics = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # Run the optimization
    results = tune_fn(criterion, metrics, num_samples=25, num_epochs=MAX_EPOCHS)

    # Report
    print(f"Best hyper-parameters found: {results.get_best_result().config}")

    sys.exit(0)
