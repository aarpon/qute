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
from torch.nn import MSELoss
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import MeanAbsoluteError
from typing_extensions import override

from qute import device
from qute.campaigns import (
    RestorationCampaignTransforms,
    SegmentationCampaignTransforms2D,
)
from qute.config import Config
from qute.data.dataloaders import DataModuleLocalFolder
from qute.data.demos import CellRestorationDemo, CellSegmentationDemo
from qute.models.attention_unet import AttentionUNet
from qute.models.swinunetr import SwinUNETR
from qute.models.unet import UNet
from qute.project import Project
from qute.random import set_global_rng_seed


class _Director(ABC):
    """Abstract base class defining the interface for all directors."""

    def __init__(self, config_file: Union[Path, str]) -> None:
        """Constructor.

        Parameters
        ----------

        config_file: Union[Path, str]
            Full path to the configuration file.
        """

        # Check if the director is instantiated from a program entry point
        frame = inspect.currentframe()
        cont = True
        while cont:
            caller_module = frame.f_globals["__name__"]
            if caller_module == "qute.director._director":
                frame = frame.f_back
            else:
                cont = False
        caller_module = frame.f_globals["__name__"]
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
        self.model = self._setup_model(model=self.config.model_class)

        # Run common training and testing operations for 'train' and 'resume' trained modes.
        self._run_common_train_and_test()

    def _resume(self):
        """Resume training from a saved state."""

        # Set up components common to train and resume
        self._setup_basis_for_training_and_resume()

        # Load specified model
        model_class = self._get_model_class()
        self.model = model_class.load_from_checkpoint(
            self.project.selected_model_path,
            criterion=self.criterion,
            metrics=self.metrics,
            class_names=self.config.class_names,
        )

        # Run common training and testing operations for 'train' and 'resume' trained modes.
        self._run_common_train_and_test()

    def _predict(self):
        """Predict using a trained model."""

        # Check that the config contains the path to the model to load
        if self.config.source_model_path is None:
            raise ValueError("No path for model to load found in the configuration!")

        # Check that the source path for prediction is specified in the configuration
        if self.config.source_for_prediction is None:
            raise ValueError(
                "No source path for prediction specified in the configuration!"
            )

        # Set up project
        self.project = Project(self.config)

        # Check that the model exists
        if not self.config.source_model_path.is_file():
            raise ValueError(
                f"The model {self.config.source_model_path} does not exist!"
            )

        # Initialize the campaign transforms
        self.campaign_transforms = self._setup_campaign_transforms()

        # Initialize data module
        self.data_module = self._setup_data_module()

        # Load existing model
        model_class = self._get_model_class()
        self.model = model_class.load_from_checkpoint(self.config.source_model_path)

        # Inform
        print(f"Predicting with model {self.config.source_model_path}")

        # Display target folder
        if self.config.target_for_prediction is None:
            print("Target for prediction not specified in configuration.")
        print(f"Predictions saved to {self.project.target_for_prediction}.")

        # Run full inference
        self.model.full_inference(
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.project.source_for_prediction
            ),
            target_folder=self.project.target_for_prediction,
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            output_dtype=self.config.output_dtype,
        )

    def _setup_basis_for_training_and_resume(self):
        """Initialize the basic components for training or resume."""

        # Set up the project
        self.project = self._setup_project()

        # Set up the transform campaign
        self.campaign_transforms = self._setup_campaign_transforms()

        # Set up the data module
        self.data_module = self._setup_data_module()

        # Inform
        print(f"Working directory: {self.project.run_dir}")

        # Calculate the number of steps per epoch
        self.data_module.prepare_data()
        self.data_module.setup("train")
        self.steps_per_epoch = len(self.data_module.train_dataloader())

        # Print the train, validation and test sets to file
        self.data_module.print_sets(filename=self.project.run_dir / "image_sets.txt")

        # Set up loss function
        self.criterion = self._setup_loss()

        # Set up metrics
        self.metrics = self._setup_metrics()

        # Set up trainer callbacks
        (
            self.training_callbacks,
            self.early_stopping,
            self.model_checkpoint,
            self.lr_monitor,
        ) = self._setup_trainer_callbacks()

        # Set up the scheduler
        self.lr_scheduler_class, self.lr_scheduler_parameters = self._setup_scheduler()

        # Set up trainer
        self.trainer = self._setup_trainer()

    def _run_common_train_and_test(self):
        """Run common training and testing operations for 'train' and
        'resume' trained modes."""

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

        # Re-load weights from best model
        model_class = self._get_model_class()
        model = model_class.load_from_checkpoint(
            self.project.selected_model_path,
            strict=False,
            criterion=self.criterion,
            metrics=self.metrics,
        )

        # Test
        self.trainer.test(model, dataloaders=self.data_module.test_dataloader())

        # If there is no source_for_prediction in the configuration file,
        # we inform and skip the full inference
        if self.config.source_for_prediction is None:
            print(
                "Source for prediction not specified in configuration. Skipping full inference."
            )
            return

        # Display target folder
        if self.config.target_for_prediction is None:
            print("Target for prediction not specified in configuration.")
        print(f"Predictions saved to {self.project.target_for_prediction}.")

        # Run full inference
        self.model.full_inference(
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.project.source_for_prediction
            ),
            target_folder=self.project.target_for_prediction,
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            output_dtype=self.config.output_dtype,
        )

    def _setup_project(self):
        """Set up the project."""
        if self.config is None:
            raise Exception("No configuration found.")

        # Initialize the project
        project = Project(self.config, clean=True)

        # Return the project
        return project

    def _setup_scheduler(self):
        """Set up scheduler."""
        if self.data_module is None:
            raise Exception("Data module is not set.")

        # Set up learning rate scheduler
        lr_scheduler_class = OneCycleLR
        lr_scheduler_parameters = {
            "total_steps": self.steps_per_epoch * self.config.max_epochs,
            "div_factor": 5.0,
            "max_lr": self.config.learning_rate,
            "pct_start": 0.5,
            "anneal_strategy": "cos",
        }

        return lr_scheduler_class, lr_scheduler_parameters

    def _setup_trainer_callbacks(self):
        """Set up trainer callbacks."""

        # Callbacks
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=self.config.checkpoint_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.checkpoint_mode,
                verbose=True,
            )
        else:
            early_stopping = None
        model_checkpoint = ModelCheckpoint(
            dirpath=self.project.models_dir,
            monitor=self.config.checkpoint_monitor,
            mode=self.config.checkpoint_mode,
            verbose=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Add them to the training callbacks list
        if self.config.use_early_stopping:
            training_callbacks = [
                early_stopping,
                model_checkpoint,
                lr_monitor,
            ]
        else:
            training_callbacks = [
                model_checkpoint,
                lr_monitor,
            ]

        return training_callbacks, early_stopping, model_checkpoint, lr_monitor

    def _setup_trainer(self):
        """Set up the trainer."""
        # Instantiate the Trainer
        trainer = pl.Trainer(
            default_root_dir=self.project.results_dir,
            accelerator=device.get_accelerator(),
            devices=1,
            precision=self.config.precision,
            callbacks=self.training_callbacks,
            max_epochs=self.config.max_epochs,
            log_every_n_steps=1,
        )

        # Store parameters
        trainer.logger._default_hp_metric = False

        # Return the trainer
        return trainer

    def _setup_model(self, model: str = "unet"):
        """Set up the model.

        Parameters
        ----------

        model: str
            Model to use. One of "unet", "attention_unet", "swin_unetr"
        """

        if model not in ["unet", "attention_unet", "swin_unetr"]:
            raise ValueError(
                "The 'model' must be one of 'unet', 'attention_unet', or 'swin_unetr'."
            )

        if model == "unet":

            # Set up the UNet model
            model = UNet(
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

        elif model == "attention_unet":

            # Set up the Attention UNet model
            model = AttentionUNet(
                campaign_transforms=self.campaign_transforms,
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                class_names=self.config.class_names,
                criterion=self.criterion,
                channels=self.config.channels,
                strides=self.config.strides,
                metrics=self.metrics,
                learning_rate=self.config.learning_rate,
                lr_scheduler_class=self.lr_scheduler_class,
                lr_scheduler_parameters=self.lr_scheduler_parameters,
            )

        elif model == "swin_unetr":

            # Set up the SwinUNETR model
            model = SwinUNETR(
                campaign_transforms=self.campaign_transforms,
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                class_names=self.config.class_names,
                spatial_dims=len(self.config.patch_size),
                depths=self.config.depths,
                num_heads=self.config.num_heads,
                feature_size=self.config.feature_size,
                criterion=self.criterion,
                metrics=self.metrics,
                learning_rate=self.config.learning_rate,
                lr_scheduler_class=self.lr_scheduler_class,
                lr_scheduler_parameters=self.lr_scheduler_parameters,
            )

        else:
            raise ValueError(
                "The 'model' must be one of 'unet', 'attention_unet', or 'swin_unetr'."
            )

        # Inform
        print(f"Using model: {self._get_model_class()} ")

        # Return the model
        return model

    def _get_model_class(self):
        """Return the class of the model being used."""
        if self.config.model_class == "unet":
            model_class = UNet
        elif self.config.model_class == "attention_unet":
            model_class = AttentionUNet
        elif self.config.model_class == "swin_unetr":
            model_class = SwinUNETR
        else:
            raise ValueError(f"Bad value for model type {self.config.model_class};")
        return model_class


class _EnsembleDirector(_Director, ABC):
    """Ensemble Training Director."""

    def __init__(self, config_file: Union[Path, str], num_folds: int) -> None:
        """Constructor.

        Parameters
        ----------

        config_file: Union[Path, str]
            Full path to the configuration file.

        num_folds: int
            Number of folds for cross-correlation validation.
        """

        super().__init__(config_file=config_file)
        self.num_folds = num_folds
        self.current_fold = -1

        # Keep track of the trained models
        self._best_models = []

    def _setup_trainer_callbacks(self):
        """Set up trainer callbacks."""

        # Callbacks
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=self.config.checkpoint_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.checkpoint_mode,
                verbose=True,
            )
        else:
            early_stopping = None
        model_checkpoint = ModelCheckpoint(
            dirpath=self.project.models_dir / f"fold_{self.current_fold}",
            monitor=self.config.checkpoint_monitor,
            mode=self.config.checkpoint_mode,
            verbose=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        # Add them to the training callbacks list
        if self.config.use_early_stopping:
            training_callbacks = [
                early_stopping,
                model_checkpoint,
                lr_monitor,
            ]
        else:
            training_callbacks = [
                model_checkpoint,
                lr_monitor,
            ]

        return training_callbacks, early_stopping, model_checkpoint, lr_monitor

    def _setup_trainer(self):
        """Set up the trainer."""
        # Instantiate the Trainer
        trainer = pl.Trainer(
            default_root_dir=self.project.results_dir / f"fold_{self.current_fold}",
            accelerator=device.get_accelerator(),
            devices=1,
            precision=self.config.precision,
            callbacks=self.training_callbacks,
            max_epochs=self.config.max_epochs,
            log_every_n_steps=1,
        )

        # Store parameters
        trainer.logger._default_hp_metric = False

        # Return the trainer
        return trainer

    def _setup_basis_for_training_and_resume(self):
        """Initialize the basic components for training or resume."""

        # Set up the project
        self.project = self._setup_project()

        # Set up the transform campaign
        self.campaign_transforms = self._setup_campaign_transforms()

        # Set up the data module
        self.data_module = self._setup_data_module()

        # Inform
        print(f"Working directory: {self.project.run_dir}")

        # Calculate the number of steps per epoch
        self.data_module.prepare_data()
        self.data_module.setup("train")
        self.steps_per_epoch = len(self.data_module.train_dataloader())

        # Print the train, validation and test sets to file
        self.data_module.print_sets(filename=self.project.run_dir / "image_sets.txt")

        # Set up loss function
        self.criterion = self._setup_loss()

        # Set up metrics
        self.metrics = self._setup_metrics()

        # Set up trainer callbacks
        (
            self.training_callbacks,
            self.early_stopping,
            self.model_checkpoint,
            self.lr_monitor,
        ) = self._setup_trainer_callbacks()

        # Set up the scheduler
        self.lr_scheduler_class, self.lr_scheduler_parameters = self._setup_scheduler()

        # Set up trainer
        self.trainer = self._setup_trainer()

    def _train(self):
        """Run a training from scratch."""

        # Set up components common to train and resume
        self._setup_basis_for_training_and_resume()

        # Copy the configuration file to the run folder
        self.project.copy_configuration_file()

        # Run training with n-fold cross-validation
        for fold in range(self.num_folds):

            # Store current fold
            self.current_fold = fold

            # Print path to best model
            print(f"Fold {fold}: starting training.")

            # Set the fold for current training
            self.data_module.set_fold(self.current_fold)

            # Update the number of steps per epoch
            self.steps_per_epoch = len(self.data_module.train_dataloader())

            # Reset the learning rate scheduler
            self.lr_scheduler_class, self.lr_scheduler_parameters = (
                self._setup_scheduler()
            )

            # Initialize new model
            self.model = self._setup_model(model=self.config.model_class)

            # Set up trainer callbacks
            (
                self.training_callbacks,
                self.early_stopping,
                self.model_checkpoint,
                self.lr_monitor,
            ) = self._setup_trainer_callbacks()

            # Instantiate the Trainer
            self.trainer = pl.Trainer(
                default_root_dir=self.project.results_dir / f"fold_{self.current_fold}",
                accelerator=device.get_accelerator(),
                devices=1,
                precision=self.config.precision,
                callbacks=self.training_callbacks,
                max_epochs=self.config.max_epochs,
                log_every_n_steps=1,
            )

            # Store parameters
            # trainer.hparams = { }
            self.trainer.logger._default_hp_metric = False

            # Train with the optimal learning rate found above
            self.trainer.fit(self.model, datamodule=self.data_module)

            # Print path to best model
            print(f"Fold {fold}: best model = {self.model_checkpoint.best_model_path}")
            print(
                f"Fold {fold}: best model score: {self.model_checkpoint.best_model_score}"
            )

            # Store the best score
            self.project.store_best_score(
                monitor=self.config.checkpoint_monitor,
                score=float(self.model_checkpoint.best_model_score),
                fold=self.current_fold,
            )

            # Set it into the project
            self.project.selected_model_path = self.model_checkpoint.best_model_path

            # Re-load weights from best model
            model_class = self._get_model_class()
            model = model_class.load_from_checkpoint(
                self.model_checkpoint.best_model_path,
                strict=False,
                criterion=self.criterion,
                metrics=self.metrics,
            )

            # Append to list of best models for inference
            self._best_models.append(model)

            # Test
            self.trainer.test(model, dataloaders=self.data_module.test_dataloader())

        # If there is no source_for_prediction in the configuration file,
        # we inform and skip the full inference
        if self.config.source_for_prediction is None:
            print(
                "Source for prediction not specified in configuration. Skipping full inference."
            )
            return

        # Display target folder
        if self.config.target_for_prediction is None:
            print("Target for prediction not specified in configuration.")
        print(f"Predictions saved to {self.project.target_for_prediction}.")

        # Run ensemble prediction
        UNet.full_inference_ensemble(
            self._best_models,
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.project.source_for_prediction
            ),
            target_folder=self.project.target_for_prediction,
            post_full_inference_transforms=self.campaign_transforms.get_post_full_inference_transforms(),
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            save_individual_preds=True,
            voting_mechanism="mode",
            weights=None,
            output_dtype=self.config.output_dtype,
        )

    def _resume(self):
        """Resume ensemble training using trained models."""
        raise NotImplementedError("Currently not supported!")

    def _predict(self):
        """Predict using a trained model."""
        # Check that the config contains the path to the model to load
        if self.config.source_model_path is None:
            raise ValueError("No path for models to load found in the configuration!")

        # Check that the source path for prediction is specified in the configuration
        if self.config.source_for_prediction is None:
            raise ValueError(
                "No source path for prediction specified in the configuration!"
            )

        # Check the model path
        if not self.config.source_model_path.is_dir():
            raise ValueError("Invalid path for models!")

        # Set up project
        self.project = Project(self.config)

        # Initialize the campaign transforms
        self.campaign_transforms = self._setup_campaign_transforms()

        # Initialize data module
        self.data_module = self._setup_data_module()

        # Load all models
        models = self._load_models(models_dir=self.config.source_model_path)

        # Run ensemble prediction
        UNet.full_inference_ensemble(
            models,
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.project.source_for_prediction
            ),
            target_folder=self.project.target_for_prediction,
            post_full_inference_transforms=self.campaign_transforms.get_post_full_inference_transforms(),
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            save_individual_preds=True,
            voting_mechanism="mode",
            weights=None,
            output_dtype=self.config.output_dtype,
        )

    def _load_models(self, models_dir: Path):
        """Reload all model found in the model folds."""

        # Re-load all (best) models
        models = []
        fold = 0
        found = True
        while found:

            # Look for the model for current fold
            found = list(models_dir.glob(f"fold_{fold}/*.ckpt"))
            if len(found) == 0:
                found = False
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

            # Increase fold number
            fold += 1

        print(f"Loaded {len(models)} trained models.")
        return models


class RestorationDirector(_Director):
    """Restoration Training Director."""

    @override
    def _setup_metrics(self):
        """Set up metrics."""

        # Metrics
        metrics = MeanAbsoluteError()

        # Return metrics
        return metrics

    @override
    def _setup_loss(self):
        """Set up loss function."""

        # Set up loss function
        criterion = MSELoss()

        # Return loss
        return criterion

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

        # Initialize default, example Restoration Campaign Transform
        campaign_transforms = RestorationCampaignTransforms(
            min_intensity=0,
            max_intensity=15472,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
        )

        # Return campaign transforms
        return campaign_transforms


class SegmentationDirector(_Director):
    """Segmentation Training Director."""

    @override
    def _setup_metrics(self):
        """Set up metrics."""

        # Metrics
        metrics = DiceMetric(
            include_background=self.config.include_background,
            reduction="mean_batch",
            get_not_nans=False,
        )

        # Return metrics
        return metrics

    @override
    def _setup_loss(self):
        """Set up loss function."""

        # Set up loss function
        criterion = DiceCELoss(
            include_background=self.config.include_background,
            to_onehot_y=False,
            softmax=True,
        )

        # Return criterion
        return criterion

    @override
    def _setup_data_module(self):
        """Set up data module."""

        # Data module
        data_module = DataModuleLocalFolder(
            campaign_transforms=self.campaign_transforms,
            data_dir=self.config.data_dir,  # Point to the root of the data directory
            seed=self.config.seed,
            num_folds=1,  # No ensemble
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

        # Return data module
        return data_module

    @override
    def _setup_campaign_transforms(self):
        """Set up campaign transforms."""

        # Initialize default, example Segmentation Campaign Transform
        campaign_transforms = SegmentationCampaignTransforms2D(
            num_classes=self.config.out_channels,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
        )

        # Return the campaign transforms
        return campaign_transforms


class EnsembleSegmentationDirector(_EnsembleDirector, SegmentationDirector):
    """Ensemble Segmentation Training Director."""

    def __init__(self, config_file: Union[Path, str], num_folds: int) -> None:
        super().__init__(config_file, num_folds)

    def _setup_data_module(self):
        """Set up data module with folds."""

        # Data module
        data_module = DataModuleLocalFolder(
            campaign_transforms=self.campaign_transforms,
            data_dir=self.config.data_dir,  # Point to the root of the data directory
            seed=self.config.seed,
            num_folds=self.num_folds,  # Ensemble
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

        # Return data module
        return data_module

    def _setup_basis_for_training_and_resume(self):
        """Initialize the basic components for training or resume."""

        # Set up the project
        self.project = self._setup_project()

        # Set up the transform campaign
        self.campaign_transforms = self._setup_campaign_transforms()

        # Set up the data module (with folds)
        self.data_module = self._setup_data_module()

        # Calculate the number of steps per epoch
        self.data_module.prepare_data()
        self.data_module.setup("train")
        self.steps_per_epoch = len(self.data_module.train_dataloader())

        # Inform
        print(f"Working directory: {self.project.run_dir}")

        # Set up loss function
        self.criterion = self._setup_loss()

        # Set up metrics
        self.metrics = self._setup_metrics()


class CellRestorationDemoDirector(RestorationDirector):
    """Restoration Demo Training Director."""

    @override
    def _setup_data_module(self):
        """Set up data module."""

        # Data module
        data_module = CellRestorationDemo(
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

        # Return data module
        return data_module


class CellSegmentationDemoDirector(SegmentationDirector):
    """Segmentation Demo Training Director."""

    @override
    def _setup_data_module(self):
        """Set up data module."""

        # Data module
        data_module = CellSegmentationDemo(
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

        # Return data module
        return data_module


class EnsembleCellSegmentationDemoDirector(EnsembleSegmentationDirector):
    """Ensemble Segmentation Demo Training Director."""

    @override
    def _setup_data_module(self):
        """Set up data module."""

        # Data module
        data_module = CellSegmentationDemo(
            campaign_transforms=self.campaign_transforms,
            download_dir=self.config.project_dir,
            seed=self.config.seed,
            num_folds=self.num_folds,
            batch_size=self.config.batch_size,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
            train_fraction=self.config.train_fraction,
            val_fraction=self.config.val_fraction,
            test_fraction=self.config.test_fraction,
            inference_batch_size=self.config.inference_batch_size,
        )

        # Return the data module
        return data_module
