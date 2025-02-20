# ******************************************************************************
# Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the Apache License Version 2.0
# which accompanies this distribution, and is available at
# https://www.apache.org/licenses/LICENSE-2.0.txt
#
# Contributors:
#   Aaron Ponti - initial API and implementation
# ******************************************************************************

import math
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import Callback, CLIReporter, RunConfig
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from qute.campaigns import SegmentationCampaignTransforms2D
from qute.config import ConfigFactory
from qute.data.demos import CellSegmentationDemo
from qute.models.swinunetr import SwinUNETR
from qute.random import set_global_rng_seed

#
# See:
#
# Using PyTorch Lightning with Tune: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
# Tune search algorithms: https://docs.ray.io/en/latest/tune/api/suggestion.html
#

# Load global configuration
config_file = (
    Path(__file__).parent / "cell_segmentation_demo_swinunet_hyperparameters_config.ini"
)
GLOBAL_CONFIG = ConfigFactory.get_config(config_file)
GLOBAL_CONFIG.parse()


class NanLossStopCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        loss = result.get("loss")
        if loss is not None and (math.isnan(loss) or loss != loss):
            print(f"Trial {trial.trial_id} encountered NaN loss. Stopping trial.")
            # @TODO Stop the trial gracefully


def train_fn(
    optimization_config,
    criterion,
    metrics,
    num_epochs=GLOBAL_CONFIG.max_epochs,
    num_gpus=1,
):
    """Training function."""

    # Only accept valid combination of parameters: in particular,
    # `num_heads` and `depths` must have the same number of elements.
    if len(optimization_config["num_heads"]) != len(optimization_config["depths"]):
        # Return a poor score to mark this as an invalid configuration
        tune.report(loss=1e6)
        return

    # Initialize Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms2D(
        num_classes=GLOBAL_CONFIG.out_channels,
        patch_size=optimization_config["patch_size"],
        num_patches=optimization_config["num_patches"],
    )

    # Instantiate data module
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms,
        seed=GLOBAL_CONFIG.seed,
        batch_size=optimization_config["batch_size"],
        patch_size=optimization_config["patch_size"],
        num_patches=optimization_config["num_patches"],
        inference_batch_size=GLOBAL_CONFIG.inference_batch_size,
    )

    # Instantiate the model
    model = SwinUNETR(
        campaign_transforms=campaign_transforms,
        in_channels=1,
        out_channels=GLOBAL_CONFIG.out_channels,
        criterion=criterion,
        metrics=metrics,
        class_names=GLOBAL_CONFIG.class_names,
        img_size=optimization_config["patch_size"],  # Map patch_size to img_size
        depths=optimization_config["depths"],
        num_heads=optimization_config["num_heads"],
        feature_size=optimization_config["feature_size"],
        learning_rate=optimization_config["learning_rate"],
        lr_scheduler_class=None,  # Disable the LR scheduler
        dropout=optimization_config["dropout"],
    )

    # Tune report callback
    report_callback = TuneReportCheckpointCallback(
        {
            "loss": "val_loss",
            "dice_cell": "val_metrics_cell",
            "dice_membrane": "val_metrics_membrane",
        },
        on="validation_end",
    )

    # Instantiate the Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        precision=GLOBAL_CONFIG.precision,
        callbacks=[report_callback],
        logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        max_epochs=num_epochs,
        log_every_n_steps=1,
        val_check_interval=1.0,  # Run validation every epoch
    )
    trainer.logger._default_hp_metric = False

    trainer.fit(model, datamodule=data_module)


def tune_fn(criterion, metrics, num_samples=10, num_epochs=GLOBAL_CONFIG.max_epochs):
    """Tune function."""

    # Create an optimization function with the various parmeters
    # and their (range of) options.
    optimization_config = {
        "learning_rate": tune.loguniform(0.0005, 0.5),
        "dropout": tune.choice([0, 0.1, 0.25, 0.5]),
        "patch_size": tune.choice([(512, 512), (640, 640)]),
        "num_patches": tune.choice([2]),
        "batch_size": tune.choice([2]),
        "num_heads": tune.choice([(3, 6, 12, 24, 48)]),
        "feature_size": tune.choice([24, 48]),
        "depths": tune.choice([(2, 2, 2, 2, 2), (3, 3, 3, 3, 3), (4, 4, 4, 4, 4)]),
    }

    # Instantiate HyperOptSearch search algorithm
    search_alg = HyperOptSearch(metric="loss", mode="min")

    # Instantiate a scheduler
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=[
            "learning_rate",
            "dropout",
            "patch_size",
            "num_patches",
            "batch_size",
            "num_heads",
            "feature_size",
            "depths",
        ],
        metric_columns=["loss", "dice_cell", "dice_membrane", "training_iteration"],
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
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=RunConfig(
            name="tune_fn",
            progress_reporter=reporter,
            callbacks=[NanLossStopCallback()],
        ),
        param_space=optimization_config,
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    # Seeding
    set_global_rng_seed(GLOBAL_CONFIG.seed, workers=True)

    # Loss
    criterion = DiceCELoss(
        include_background=GLOBAL_CONFIG.include_background,
        to_onehot_y=False,
        softmax=True,
    )

    # Metrics
    metrics = DiceMetric(
        include_background=GLOBAL_CONFIG.include_background,
        reduction="mean",
        get_not_nans=False,
    )

    # Run the optimization
    results = tune_fn(
        criterion, metrics, num_samples=25, num_epochs=GLOBAL_CONFIG.max_epochs
    )

    # Report
    print(f"Best hyper-parameters found: {results.get_best_result().config}")

    sys.exit(0)
