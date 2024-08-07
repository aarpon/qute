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

import os
import sys

import pytorch_lightning as pl
import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from qute.campaigns import SegmentationCampaignTransforms2D
from qute.data.demos import CellSegmentationDemo
from qute.models.unet import UNet

#
# See:
#
# Using PyTorch Lightning with Tune: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
# Tune search algorithms: https://docs.ray.io/en/latest/tune/api/suggestion.html
#

torch.set_float32_matmul_precision("medium")

# Configuration
CONFIG = {
    "seed": 2022,
    "inference_batch_size": 4,
    "num_classes": 3,
    "max_epochs": 2000,
    "precision": "16-mixed",
}


def train_fn(config, criterion, metrics, num_epochs=CONFIG["max_epochs"], num_gpus=1):
    # Get current configuration parameters
    num_res_units = config["num_res_units"]
    learning_rate = config["learning_rate"]
    channels = config["channels"]
    dropout = config["dropout"]
    patch_size = config["patch_size"]
    num_patches = config["num_patches"]
    batch_size = config["batch_size"]

    # Initialize default, example Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms2D(
        num_classes=CONFIG["num_classes"],
        patch_size=patch_size,
        num_patches=num_patches,
    )

    # Instantiate data module
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms,
        seed=CONFIG["seed"],
        batch_size=batch_size,
        patch_size=patch_size,
        num_patches=num_patches,
        inference_batch_size=CONFIG["inference_batch_size"],
    )

    # Instantiate the model
    model = UNet(
        campaign_transforms=campaign_transforms,
        in_channels=1,
        out_channels=CONFIG["num_classes"],
        num_res_units=num_res_units,
        criterion=criterion,
        channels=channels,
        strides=None,
        metrics=metrics,
        learning_rate=learning_rate,
        dropout=dropout,
    )

    # Tune report callback
    report_callback = TuneReportCheckpointCallback(
        {"loss": "val_loss", "dice": "val_metrics"}, on="validation_end"
    )

    # Instantiate the Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        precision=CONFIG["precision"],
        callbacks=[report_callback],
        logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        max_epochs=num_epochs,
        log_every_n_steps=1,
        val_check_interval=1.0,  # Run validation every epoch
    )
    trainer.logger._default_hp_metric = False

    trainer.fit(model, datamodule=data_module)


def tune_fn(criterion, metrics, num_samples=10, num_epochs=CONFIG["max_epochs"]):
    config = {
        "num_res_units": tune.choice([0, 1, 2, 3, 4]),
        "learning_rate": tune.loguniform(0.0005, 0.5),
        "channels": tune.choice([(16, 32), (16, 32, 64), (32, 64), (32, 64, 128)]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "patch_size": tune.choice([(256, 256), (384, 384), (512, 512), (640, 640)]),
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
    seed_everything(CONFIG["seed"], workers=True)

    # Loss
    criterion = DiceCELoss(include_background=True, to_onehot_y=False, softmax=True)

    # Metrics
    metrics = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Run the optimization
    results = tune_fn(
        criterion, metrics, num_samples=25, num_epochs=CONFIG["max_epochs"]
    )

    # Report
    print(f"Best hyper-parameters found: {results.get_best_result().config}")

    sys.exit(0)
