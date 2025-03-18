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

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import userpaths
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.optimizers import LearningRateFinder
from torch.optim import AdamW

from qute.campaigns import SegmentationCampaignTransforms2D
from qute.config import ConfigFactory
from qute.data.demos import CellSegmentationDemo
from qute.device import get_device
from qute.models.factory import ModelFactory

if __name__ == "__main__":
    # Configuration file
    config_file = Path(__file__).parent / "cell_segmentation_demo_unet_config.ini"

    # Get the proper configuration parser
    config = ConfigFactory.get_config(config_file)

    # Parse it
    if config is None or not config.parse():
        raise Exception("Invalid config file")

    # Initialize default Segmentation Campaign Transform
    campaign_transforms = SegmentationCampaignTransforms2D(
        num_classes=config.out_channels,
        patch_size=config.patch_size,
        num_patches=config.num_patches,
    )

    # Data module
    data_module = CellSegmentationDemo(
        campaign_transforms=campaign_transforms,
        download_dir=config.project_dir,
        seed=config.seed,
        batch_size=config.batch_size,
        patch_size=config.patch_size,
        num_patches=config.num_patches,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        inference_batch_size=config.inference_batch_size,
        num_workers=0,  # LearningRateFinder only works when single-threaded
    )

    # Metric
    metric = DiceMetric(
        include_background=config.include_background,
        reduction="mean_batch",
        get_not_nans=False,
    )

    # Criterion
    criterion = DiceCELoss(
        include_background=config.include_background,
        to_onehot_y=False,
        softmax=True,
    )

    # Set up model
    model = ModelFactory.get_model(
        config=config,
        campaign_transforms=campaign_transforms,
        criterion=criterion,
        metrics=metric,
        lr_scheduler_class=None,
        lr_scheduler_params={},
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-7, weight_decay=1e-2)

    # Prepare the datamodule
    data_module.prepare_data()
    data_module.setup("train")

    # Run the learning rate finder with fastai approach: increases the learning rate in an
    # exponential manner and computes the training loss for each learning rate.
    lr_finder = LearningRateFinder(model, optimizer, criterion, device=get_device())
    print("* fastai approach")
    lr_finder.range_test(data_module.train_dataloader(), end_lr=100, num_iter=100)
    lr_finder.get_steepest_gradient()
    fig, ax = plt.subplots(1, 1, dpi=300)
    lr_finder.plot(ax=ax)
    out_file = Path(userpaths.get_my_documents()) / "lrfinder_fastai.png"
    fig.savefig(out_file, dpi=300)
    lr_finder.reset()
    print(f"Plot saved to {out_file}.")

    # Run the learning rate finder with Leslie Smith's approach: increases the learning rate
    # linearly and computes the evaluation loss for each learning rate.
    optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=1e-2)
    lr_finder = LearningRateFinder(model, optimizer, criterion, device=get_device())
    print("* Leslie Smith's approach")
    lr_finder.range_test(
        data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        end_lr=1,
        num_iter=100,
        step_mode="linear",
    )
    fig, ax = plt.subplots(1, 1, dpi=300)
    lr_finder.plot(log_lr=False, ax=ax)
    out_file = Path(userpaths.get_my_documents()) / "lrfinder_leslie_smith.png"
    fig.savefig(out_file, dpi=300)
    lr_finder.reset()
    print(f"Plot saved to {out_file}.")

    # Properly exit
    sys.exit(0)
