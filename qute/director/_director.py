#  ********************************************************************************
#  Copyright © 2022 - 2024, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.optim.lr_scheduler import OneCycleLR
from typing_extensions import override

from qute import device
from qute.campaigns import SegmentationCampaignTransforms2D
from qute.config import Config
from qute.data.dataloaders import DataModuleLocalFolder
from qute.data.demos import CellSegmentationDemo
from qute.models.unet import UNet
from qute.project import Project
from qute.random import set_global_rng_seed


class Director(ABC):
    """Abstract base class defining the interface for all directors."""

    def __init__(self, config_file: Union[Path, str]) -> None:

        # Check if the director is instantiated from a program entry point
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_module = caller_frame.f_globals["__name__"]
        if caller_module != "__main__":
            print(
                "Warning: the Director is not instantiated from the program entry point (`__main__`).",
                file=sys.stderr,
            )
            print(
                "This could cause issues in particular on macOS and Windows.",
                file=sys.stderr,
            )

        # Store the configuration file
        self.config_file = config_file

        # Parse it
        self.config = Config(self.config_file)
        if not self.config.parse():
            raise Exception("Invalid config file")

        # Keep a reference to the project
        self.project = None

        # Keep references to other important objects
        self.campaign_transforms = None
        self.data_module = None
        self.criterion = None
        self.metrics = None
        self.lr_scheduler_class = None
        self.lr_scheduler_parameters = None
        self.early_stopping = None
        self.model_checkpoint = None
        self.lr_monitor = None
        self.training_callbacks = []
        self.trainer = None
        self.model = None
        self.steps_per_epoch = 0

    @abstractmethod
    def _setup_campaign_transforms(self):
        """Set up campaign transforms."""
        raise NotImplementedError("Reimplement in child class.")

    @abstractmethod
    def _setup_data_module(self):
        """Set up data module."""
        raise NotImplementedError("Reimplement in child class.")

    @abstractmethod
    def _setup_loss(self):
        """Set up loss function."""
        raise NotImplementedError("Reimplement in child class.")

    @abstractmethod
    def _setup_metrics(self):
        """Set up metrics."""
        raise NotImplementedError("Reimplement in child class.")

    def run(self):
        """Run the process as configured in the configuration file."""

        # Seeding
        set_global_rng_seed(self.config.seed, workers=True)

        # Get the mode
        if self.config.trainer_mode == "train":
            self._train()

        elif self.config.trainer_mode == "resume":
            self._resume()

        elif self.config.trainer_mode == "predict":
            self._predict()

        else:
            raise ValueError(
                "Trainer mode must be one of 'train' or 'resume' or 'predict'."
            )

    def _train(self):
        """Run a training from scratch."""

        # Set up components common to train and resume
        self._setup_basis_for_training_and_resume()

        # Set up model
        self._setup_model()

        # Copy the configuration file to the run folder
        self.project.copy_configuration_file()

        # Train
        self.trainer.fit(self.model, datamodule=self.data_module)

        # Print path to best model
        print(f"Best model: {self.model_checkpoint.best_model_path}")
        print(f"Best model score: {self.model_checkpoint.best_model_score}")

        # Store the best score
        self.project.store_best_score(
            self.config.checkpoint_monitor, self.model_checkpoint.best_model_score
        )

        # Set it into the project
        self.project.selected_model_path = self.model_checkpoint.best_model_path

        # Re-load weights from best model: this model is a classification one,
        # so there is no need to change the output layer.
        model = UNet.load_from_checkpoint(
            self.project.selected_model_path, strict=False
        )

        # Test
        self.trainer.test(model, dataloaders=self.data_module.test_dataloader())

    def _resume(self):
        """Resume training from a saved state."""

        # Set up components common to train and resume
        self._setup_basis_for_training_and_resume()

        # Load specified model
        self.model = UNet.load_from_checkpoint(
            self.project.selected_model_path,
            class_names=self.config.class_names,
        )

        # Copy the configuration file to the run folder
        self.project.copy_configuration_file()

        # Train
        self.trainer.fit(self.model, datamodule=self.data_module)

        # Print path to best model
        print(f"Best model: {self.model_checkpoint.best_model_path}")
        print(f"Best model score: {self.model_checkpoint.best_model_score}")

        # Store the best score
        self.project.store_best_score(
            self.config.checkpoint_monitor, self.model_checkpoint.best_model_score
        )

        # Set it into the project
        self.project.selected_model_path = self.model_checkpoint.best_model_path

        # Re-load weights from best model: this model is a classification one,
        # so there is no need to change the output layer.
        model = UNet.load_from_checkpoint(
            self.project.selected_model_path, strict=False
        )

        # Test
        self.trainer.test(model, dataloaders=self.data_module.test_dataloader())

    def _predict(self):
        """Predict using a trained model."""

        # Check that the config contains the path to the model to load
        if self.config.source_model_path is None:
            raise ValueError("No path for model to load found in the configuration!")

        # Check that the source path for prediction is specified in the configuration
        if self.config.source_for_prediction is None:
            raise ValueError(
                "No source path for prediction specified  in the configuration!"
            )

        # Set up project
        self.project = Project(self.config)

        # Check that the model exists
        if not self.config.source_model_path.is_file():
            raise ValueError(
                f"The model {self.config.source_model_path} does not exist!"
            )

        # Initialize the campaign transforms
        self._setup_campaign_transforms()

        # Initialize data module
        self._setup_data_module()

        # Load existing model
        self.model = UNet.load_from_checkpoint(self.config.source_model_path)

        # Inform
        print(f"Predicting with model {self.config.source_model_path}")

        # Display target folder
        if self.config.target_for_prediction is None:
            print("Target for prediction not specified in configuration.")
        print(f"Predictions saved to {self.project.target_for_prediction}.")

        self.model.full_inference(
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.project.source_for_prediction
            ),
            target_folder=self.project.target_for_prediction,
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
        )

    def _setup_basis_for_training_and_resume(self):
        """Initialize the basic components for training or resume."""

        # Set up the project
        self._setup_project()

        # Set up the transform campaign
        self._setup_campaign_transforms()

        # Set up the data module
        self._setup_data_module()

        # Inform
        print(f"Working directory: {self.project.run_dir}")

        # Calculate the number of steps per epoch
        self.data_module.prepare_data()
        self.data_module.setup("train")
        self.steps_per_epoch = len(self.data_module.train_dataloader())

        # Print the train, validation and test sets to file
        self.data_module.print_sets(filename=self.project.run_dir / "image_sets.txt")

        # Set up loss function
        self._setup_loss()

        # Set up metrics
        self._setup_metrics()

        # Set up trainer callbacks
        self._setup_trainer_callbacks()

        # Set up the scheduler
        self._setup_scheduler()

        # Set up trainer
        self._setup_trainer()

    def _setup_project(self):
        """Set up the project."""
        if self.config is None:
            raise Exception("No configuration found.")

        # Initialize the project
        self.project = Project(self.config, clean=True)

    def _setup_scheduler(self):
        """Set up scheduler."""
        if self.data_module is None:
            raise Exception("Data module is not set.")

        # Set up learning rate scheduler
        self.lr_scheduler_class = OneCycleLR
        self.lr_scheduler_parameters = {
            "total_steps": self.steps_per_epoch * self.config.max_epochs,
            "div_factor": 5.0,
            "max_lr": self.config.learning_rate,
            "pct_start": 0.5,
            "anneal_strategy": "cos",
        }

    def _setup_trainer_callbacks(self):
        """Set up trainer callbacks."""

        # Callbacks
        if self.config.use_early_stopping:
            self.early_stopping = EarlyStopping(
                monitor=self.config.checkpoint_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.checkpoint_mode,
                verbose=True,
            )  # Issues with Lightning's ES
        self.model_checkpoint = ModelCheckpoint(
            dirpath=self.project.models_dir,
            monitor=self.config.checkpoint_monitor,
            mode=self.config.checkpoint_mode,
            verbose=True,
        )
        self.lr_monitor = LearningRateMonitor(logging_interval="step")

        # Store them
        if self.config.use_early_stopping:
            self.training_callbacks = [
                self.early_stopping,
                self.model_checkpoint,
                self.lr_monitor,
            ]
        else:
            self.training_callbacks = [
                self.model_checkpoint,
                self.lr_monitor,
            ]

    def _setup_trainer(self):
        """Set up the trainer."""
        # Instantiate the Trainer
        self.trainer = pl.Trainer(
            default_root_dir=self.project.results_dir,
            accelerator=device.get_accelerator(),
            devices=1,
            precision=self.config.precision,
            callbacks=self.training_callbacks,
            max_epochs=self.config.max_epochs,
            log_every_n_steps=1,
        )

        # Store parameters
        self.trainer.logger._default_hp_metric = False

    def _setup_model(self):
        """Set up the model."""

        # Set up the model
        self.model = UNet(
            campaign_transforms=self.campaign_transforms,
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            class_names=self.config.class_names,
            num_res_units=self.config.num_res_units,
            criterion=self.criterion,
            channels=self.config.channels,
            strides=self.config.strides,
            metrics=self.metrics,
            learning_rate=self.config.learning_rate,
            lr_scheduler_class=self.lr_scheduler_class,
            lr_scheduler_parameters=self.lr_scheduler_parameters,
        )

    def _load_model(self):
        """Load existing model."""

        # Load the model
        self.model = UNet.load_from_checkpoint(
            self.project.selected_model_path,
            class_names=self.config.class_names,
        )


class SegmentationDirector(Director):
    """Segmentation Training Director."""

    @override
    def _setup_metrics(self):
        """Set up metrics."""

        # Metrics
        self.metrics = DiceMetric(
            include_background=self.config.include_background,
            reduction="mean_batch",
            get_not_nans=False,
        )

    @override
    def _setup_loss(self):
        """Set up loss function."""

        # Set up loss function
        self.criterion = DiceCELoss(
            include_background=self.config.include_background,
            to_onehot_y=False,
            softmax=True,
        )

    @override
    def _setup_data_module(self):
        """Set up data module."""

        # Data module
        self.data_module = DataModuleLocalFolder(
            campaign_transforms=self.campaign_transforms,
            data_dir=self.config.data_dir,  # Point to the root of the data directory
            seed=self.config.seed,
            batch_size=self.config.batch_size,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
            train_fraction=self.config.train_fraction,
            val_fraction=self.config.val_fraction,
            test_fraction=self.config.test_fraction,
            source_images_sub_folder=self.config.source_images_sub_folder,
            target_images_sub_folder=self.config.target_images_sub_folder,
            source_images_label=self.config.source_images_label,
            target_images_label=self.config.target_images_label,
            inference_batch_size=self.config.inference_batch_size,
        )

    @override
    def _setup_campaign_transforms(self):
        """Set up campaign transforms."""

        # Initialize default, example Segmentation Campaign Transform
        self.campaign_transforms = SegmentationCampaignTransforms2D(
            num_classes=self.config.out_channels,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
        )


class CellSegmentationDemoDirector(SegmentationDirector):
    """Segmentation Training Director."""

    @override
    def _setup_data_module(self):
        """Set up data module."""

        # Data module
        self.data_module = CellSegmentationDemo(
            campaign_transforms=self.campaign_transforms,
            download_dir=self.config.project_dir,
            seed=self.config.seed,
            batch_size=self.config.batch_size,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
            train_fraction=self.config.train_fraction,
            val_fraction=self.config.val_fraction,
            test_fraction=self.config.test_fraction,
            inference_batch_size=self.config.inference_batch_size,
        )
