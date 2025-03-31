#  ********************************************************************************
#  Copyright © 2022 - 2025, ETH Zurich, D-BSSE, Aaron Ponti
#  All rights reserved. This program and the accompanying materials
#  are made available under the terms of the Apache License Version 2.0
#  which accompanies this distribution, and is available at
#  https://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Contributors:
#    Aaron Ponti - initial API and implementation
#  ******************************************************************************
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, Metric
from pytorch_lightning.callbacks import (
    Callback,
    Checkpoint,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.nn import Module, MSELoss
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torchmetrics import MeanAbsoluteError

from qute import device
from qute.campaigns import (
    CampaignTransforms,
    RestorationCampaignTransforms,
    SegmentationCampaignTransforms2D,
    SegmentationCampaignTransforms3D,
    SelfSupervisedRestorationCampaignTransforms,
)
from qute.config import ConfigFactory
from qute.data.dataloaders import DataModuleLocalFolder
from qute.data.demos import CellRestorationDemo, CellSegmentationDemo
from qute.data.inference import full_inference, full_inference_ensemble
from qute.models.attention_unet import AttentionUNet
from qute.models.base_model import BaseModel
from qute.models.dynunet import DynUNet
from qute.models.factory import ModelFactory
from qute.models.swinunetr import SwinUNETR
from qute.models.unet import UNet
from qute.project import Project
from qute.random import set_global_rng_seed


class Director(ABC):
    """Abstract base class defining the interface for all directors."""

    def __init__(
        self,
        config_file: Union[Path, str],
        num_workers: int = -1,
        *,
        campaign_transforms: Optional[CampaignTransforms] = None,
        data_module: Optional[pl.LightningDataModule] = None,
        criterion: Optional[Module] = None,
        metrics: Optional[Metric] = None,
        lr_scheduler_class: Optional[LRScheduler] = None,
        lr_scheduler_parameters: Optional[dict] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------

        config_file: Union[Path, str]
            Full path to the configuration file.

        num_workers: int
            Number of workers to use.

        campaign_transforms: Optional[CampaignTransforms] = None
            CampaignTransform to use for data transformations.

        data_module: Optional[pl.LightningDataModule] = None
            Data module to use to load data for training, validation, testing and prediction.

        criterion: Optional[Loss] = None
            Loss to use for optimization.

        metrics: Optional[Metric] = None
            Metrics to use for validation.

        lr_scheduler_class: Optional[LRScheduler] = None
            Learning rate scheduler to use.

        lr_scheduler_parameters: Optional[dict] = None
            Parameters for learning rate scheduler.
        """

        # Store the configuration file
        self.config_file = config_file

        # Number of workers
        if num_workers == -1:
            self.num_workers = os.cpu_count()
        else:
            self.num_workers = num_workers

        # Get the proper configuration parser
        self.config = ConfigFactory.get_config(config_file)

        # Parse it
        if self.config is None or not self.config.parse():
            raise Exception("Invalid config file")

        # Set up training precision
        self.trainer_precision = None
        self._set_precision()

        # Keep a reference to the project
        self.project = None

        # Set keyword arguments
        self.campaign_transforms = campaign_transforms
        self.data_module = data_module
        self.criterion = criterion
        self.metrics = metrics
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_parameters = lr_scheduler_parameters

        # Keep references to other important objects
        self.early_stopping = None
        self.model_checkpoint = None
        self.lr_monitor = None
        self.training_callbacks = []
        self.trainer = None
        self.model = None
        self.steps_per_epoch = 0

    @abstractmethod
    def _setup_default_campaign_transforms(self):
        """Set up default campaign transforms."""
        raise NotImplementedError("Reimplement in child class.")

    @abstractmethod
    def _setup_default_data_module(self):
        """Set up default data module."""
        raise NotImplementedError("Reimplement in child class.")

    @abstractmethod
    def _setup_default_loss(self):
        """Set up default loss function."""
        raise NotImplementedError("Reimplement in child class.")

    @abstractmethod
    def _setup_default_metrics(self):
        """Set up default metrics."""
        raise NotImplementedError("Reimplement in child class.")

    def _set_precision(self):
        """Set the precision for training based on the accelerator.

        One of "32", "16-mixed", "bf16-mixed", "16", "bf16".

        These are fully-supported by PyTorch-Lightning.

        "32":
            * Full 32-bit floating point precision (default).
            * Used when no precision argument is specified.
            * Suitable for tasks requiring high precision or for unsupported hardware configurations.

        "16-mixed":
            * Mixed precision training using FP16 (16-bit floating point).
            * Enabled via PyTorch's AMP (Automatic Mixed Precision).
            * Requires CUDA and supported GPUs (NVIDIA GPUs with Tensor Cores).

        "bf16-mixed":
            * Mixed precision training using bfloat16 (BF16).
            * Enabled via PyTorch's AMP.
            * Supported on NVIDIA Ampere GPUs or later, and some TPUs.
            * Provides higher range compared to FP16 but slightly lower precision.

        "16" (FP16 Full Precision):
            * Pure FP16 precision (not mixed).
            * Rarely used as it is less stable for training compared to "16-mixed".
            * Suitable for inference on FP16-capable hardware but not recommended for training.

        "bf16" (BF16 Full Precision):
            * Pure BF16 precision.
            * Similar to "16", this is typically used for inference rather than training.
        """

        ALLOWED_PRECISION = ["32", "16-mixed", "bf16-mixed", "16", "bf16"]
        if self.config.precision not in ALLOWED_PRECISION:
            raise ValueError(
                f"Unsupported precision: {self.config.precision}. Allowed values are {ALLOWED_PRECISION}."
            )

        # Device and precision settings
        accelerator = device.get_accelerator()
        if accelerator == "gpu":
            # PyTorch-Lighting will use the passed precision with its Trainer object.
            # If the hardware does not support it, PyTorch-Lightning will transparently
            # adapt it.
            trainer_precision = self.config.precision
            print(f'PyTorch Lightning Trainer precision = "{self.config.precision}"')

        elif accelerator == "mps":
            # MPS backend currently supports only float32 precision. Override the setting.
            trainer_precision = "32"
            print("MPS backend detected. Using float32 precision.")

        elif accelerator == "cpu":
            # CPU backend supports only float32 precision efficiently. Override the setting.
            trainer_precision = "32"
            print("CPU backend detected. Using float32 precision.")
        else:
            # Other accelerators - use the passed precision and hope for the best.
            trainer_precision = self.config.precision
            print(f"Using default precision: {trainer_precision}")

        # Store the trainer precision
        self.trainer_precision = trainer_precision

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
                "Trainer mode must be one of 'train', 'resume', or 'predict'."
            )

    def _train(self):
        """Run a training from scratch."""

        # Set up components common to train and resume
        self._setup_basis_for_training_and_resume()

        # Set up model
        self.model = ModelFactory.get_model(
            config=self.config,
            campaign_transforms=self.campaign_transforms,
            criterion=self.criterion,
            metrics=self.metrics,
            lr_scheduler_class=self.lr_scheduler_class,
            lr_scheduler_params=self.lr_scheduler_parameters,
        )

        # Run common training and testing operations for 'train' and 'resume' modes.
        self._run_common_train_and_resume()

    def _resume(self):
        """Resume training from a saved state."""

        # Set up components common to train and resume
        self._setup_basis_for_training_and_resume()

        # Load specified model
        model_class = ModelFactory.get_model_class(config=self.config)
        self.model = model_class.load_from_checkpoint(
            self.project.selected_model_path,
            criterion=self.criterion,
            metrics=self.metrics,
            class_names=self.config.class_names,
        )

        # Run common training and testing operations for 'train' and 'resume' modes.
        self._run_common_train_and_resume()

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

        # Copy the configuration file to the run folder
        self.project.copy_configuration_file()

        # Check that the model exists
        if not self.config.source_model_path.is_file():
            raise ValueError(
                f"The model {self.config.source_model_path} does not exist!"
            )

        # If no campaign transforms were passed, set up default
        if self.campaign_transforms is None:
            self.campaign_transforms = self._setup_default_campaign_transforms()

        # If not data module was passed, set up default
        if self.data_module is None:
            self.data_module = self._setup_default_data_module()

        # If not criterion (loss) was passed, set up default
        if self.criterion is None:
            self.metrics = self._setup_default_loss()

        # If not metrics were passed, set up default
        if self.metrics is None:
            self.metrics = self._setup_default_metrics()

        # Load existing model
        model_class = ModelFactory.get_model_class(config=self.config)
        self.model = model_class.load_from_checkpoint(
            self.config.source_model_path,
            criterion=self.criterion,
            metrics=self.metrics,
            class_names=self.config.class_names,
        )

        # Inform
        print(f"Predicting with model {self.config.source_model_path}")

        # Determine target folder for predictions
        if self.config.target_for_prediction is None:
            print("Target for prediction not specified in configuration.")
            target_for_prediction = self.project.run_dir / "predictions"
            print(f"Defaulting to {target_for_prediction}")
        else:
            target_for_prediction = self.config.target_for_prediction

        # Run full inference
        full_inference(
            self.model,
            campaign_transforms=self.campaign_transforms,
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.config.source_for_prediction
            ),
            target_folder=target_for_prediction,
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            prefix="",
            output_dtype=self.config.output_dtype,
        )

    def _setup_basis_for_training_and_resume(self):
        """Initialize the basic components for training or resume."""

        # Set up the project
        self.project = self._setup_project()

        # If no campaign transforms were passed, set up default
        if self.campaign_transforms is None:
            self.campaign_transforms = self._setup_default_campaign_transforms()

        # If not data module was passed, set up default
        if self.data_module is None:
            self.data_module = self._setup_default_data_module()

        # Inform
        print(f"Working directory: {self.project.run_dir}")

        # Calculate the number of steps per epoch
        self.data_module.prepare_data()
        self.data_module.setup("train")
        self.steps_per_epoch = len(self.data_module.train_dataloader())
        if self.steps_per_epoch == 0:
            raise ValueError("Number of steps per epoch is zero. Check your data.")
        print(f"Number of steps per epoch: {self.steps_per_epoch}")

        # Print the train, validation, and test sets to file
        self.data_module.print_sets(filename=self.project.run_dir / "image_sets.txt")

        # If not criterion (loss) was passed, set up default
        if self.criterion is None:
            self.criterion = self._setup_default_loss()

        # If not metrics were passed, set up default
        if self.metrics is None:
            self.metrics = self._setup_default_metrics()

        # Set up trainer callbacks
        (
            self.training_callbacks,
            self.early_stopping,
            self.model_checkpoint,
            self.lr_monitor,
        ) = self._setup_trainer_callbacks()

        # Set up the learning rate scheduler
        if self.lr_scheduler_class is None:
            self.lr_scheduler_class, self.lr_scheduler_parameters = (
                self._setup_default_scheduler()
            )

        # Set up trainer
        self.trainer = self._setup_trainer()

    def _run_common_train_and_resume(self):
        """Run common training and testing operations for 'train' and 'resume' modes."""

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
        model_class = ModelFactory.get_model_class(config=self.config)
        model = model_class.load_from_checkpoint(
            self.project.selected_model_path,
            strict=False,
            criterion=self.criterion,
            metrics=self.metrics,
            class_names=self.config.class_names,
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

        # Determine target folder for predictions
        if self.config.target_for_prediction is None:
            print("Target for prediction not specified in configuration.")
            target_for_prediction = self.project.run_dir / "predictions"
            print(f"Defaulting to {target_for_prediction}")
        else:
            target_for_prediction = self.config.target_for_prediction

        # Run full inference
        full_inference(
            self.model,
            campaign_transforms=self.campaign_transforms,
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.config.source_for_prediction
            ),
            target_folder=target_for_prediction,
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            prefix="",
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

    def _setup_default_scheduler(self):
        """Set up default learning rate scheduler and parameters."""
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
            precision=self.trainer_precision,
            callbacks=self.training_callbacks,
            max_epochs=self.config.max_epochs,
            log_every_n_steps=1,
        )

        # Store parameters
        trainer.logger._default_hp_metric = False

        # Return the trainer
        return trainer


class EnsembleDirector(Director):
    """Ensemble Training Director."""

    def __init__(
        self,
        config_file: Union[Path, str],
        num_folds: int,
        num_workers: int = -1,
        *,
        campaign_transforms: Optional[CampaignTransforms] = None,
        data_module: Optional[pl.LightningDataModule] = None,
        criterion: Optional[Module] = None,
        metrics: Optional[Metric] = None,
        lr_scheduler_class: Optional[LRScheduler] = None,
        lr_scheduler_parameters: Optional[dict] = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        config_file: Union[Path, str]
            Full path to the configuration file.
        num_folds: int
            Number of folds for cross-validation.
        num_workers: int
            Number of workers to use for data loading. Default is -1, which uses all available CPUs.
        """
        super().__init__(
            config_file=config_file,
            num_workers=num_workers,
            campaign_transforms=campaign_transforms,
            data_module=data_module,
            criterion=criterion,
            metrics=metrics,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_parameters=lr_scheduler_parameters,
        )
        self.num_folds = num_folds
        self.current_fold = -1

        # Keep track of the trained models
        self._best_models = []

    def _setup_trainer_callbacks(self):
        """Set up trainer callbacks."""
        # Callbacks
        training_callbacks = []
        early_stopping = None
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=self.config.checkpoint_monitor,
                patience=self.config.early_stopping_patience,
                mode=self.config.checkpoint_mode,
                verbose=True,
            )
            training_callbacks.append(early_stopping)

        model_checkpoint = ModelCheckpoint(
            dirpath=self.project.models_dir / f"fold_{self.current_fold}",
            monitor=self.config.checkpoint_monitor,
            mode=self.config.checkpoint_mode,
            verbose=True,
        )
        training_callbacks.append(model_checkpoint)

        lr_monitor = LearningRateMonitor(logging_interval="step")
        training_callbacks.append(lr_monitor)

        return training_callbacks, early_stopping, model_checkpoint, lr_monitor

    def _setup_trainer(self):
        """Set up the PyTorch Lightning trainer."""
        # Instantiate the Trainer
        trainer = pl.Trainer(
            default_root_dir=self.project.results_dir / f"fold_{self.current_fold}",
            accelerator=device.get_accelerator(),
            devices=1,
            precision=self.trainer_precision,
            callbacks=self.training_callbacks,
            max_epochs=self.config.max_epochs,
            log_every_n_steps=1,
        )

        # Store parameters
        trainer.logger._default_hp_metric = False

        # Return the trainer
        return trainer

    def _train(self):
        """Run a training from scratch with cross-validation."""
        # Set up components common to train and resume
        self._setup_basis_for_training_and_resume()

        # Copy the configuration file to the run folder
        self.project.copy_configuration_file()

        # Run training with n-fold cross-validation
        for fold in range(self.num_folds):
            # Store current fold
            self.current_fold = fold

            # Inform
            print(f"Fold {fold}: starting training.")

            # Set the fold for current training
            self.data_module.set_fold(self.current_fold)

            # Print the train, validation, and test sets to file
            self.data_module.print_sets(
                filename=self.project.run_dir / f"fold_{fold}_image_sets.txt"
            )

            # Update the number of steps per epoch
            self.steps_per_epoch = len(self.data_module.train_dataloader())
            if self.steps_per_epoch == 0:
                raise ValueError("Number of steps per epoch is zero. Check your data.")

            # If no learning rate scheduler pass passed, initialize default
            if self.lr_scheduler_class is None:
                self.lr_scheduler_class, self.lr_scheduler_parameters = (
                    self._setup_default_scheduler()
                )

            # Initialize new model
            self.model = ModelFactory.get_model(
                config=self.config,
                campaign_transforms=self.campaign_transforms,
                criterion=self.criterion,
                metrics=self.metric,
                lr_scheduler_class=self.lr_scheduler_class,
                lr_scheduler_params=self.lr_scheduler_parameters,
            )

            # Set up trainer callbacks
            (
                self.training_callbacks,
                self.early_stopping,
                self.model_checkpoint,
                self.lr_monitor,
            ) = self._setup_trainer_callbacks()

            # Instantiate the Trainer
            self.trainer = self._setup_trainer()

            # Train
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

            # Re-load weights from best model
            model_class = ModelFactory.get_model_class(config=self.config)
            model = model_class.load_from_checkpoint(
                self.model_checkpoint.best_model_path,
                strict=False,
                criterion=self.criterion,
                metrics=self.metrics,
                class_names=self.config.class_names,
            )

            # Append to list of best models for inference
            self._best_models.append(model)

            # Test
            self.trainer.test(model, dataloaders=self.data_module.test_dataloader())

        # If there is a source for prediction, run ensemble inference
        if self.config.source_for_prediction is not None:
            # Determine the target folder for predictions
            target_for_prediction = self.config.target_for_prediction
            if target_for_prediction is None:
                print("Target for prediction not specified in configuration.")
                target_for_prediction = self.project.run_dir / "predictions"
                print(f"Defaulting to {target_for_prediction}")

            # Display target folder
            print(f"Saving predictions to {target_for_prediction}.")

            # Run ensemble prediction
            self._run_ensemble_inference(target_for_prediction)
        else:
            print(
                "Source for prediction not specified in configuration. Skipping full inference."
            )

    def _run_ensemble_inference(self, target_for_prediction):
        """Run ensemble inference using the trained models."""
        # Run ensemble prediction
        full_inference_ensemble(
            models=self._best_models,
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.config.source_for_prediction
            ),
            target_folder=target_for_prediction,
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            save_individual_preds=True,
            voting_mechanism="mode",
            weights=None,
            prefix="",
            output_dtype=self.config.output_dtype,
        )

    def _predict(self):
        """Predict using a trained ensemble model."""

        # Check that the config contains the path to the models to load
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

        # Copy the configuration file to the run folder
        self.project.copy_configuration_file()

        # If no campaign transforms were passed, set up default
        if self.campaign_transforms is None:
            self.campaign_transforms = self._setup_default_campaign_transforms()

        # If not data module was passed, set up default
        if self.data_module is None:
            self.data_module = self._setup_default_data_module()

        # If not criterion (loss) was passed, set up default
        if self.criterion is None:
            self.metrics = self._setup_default_loss()

        # If not metrics were passed, set up default
        if self.metrics is None:
            self.metrics = self._setup_default_metrics()

        # Load all models
        models = self._load_models(models_dir=self.config.source_model_path)

        # Determine target folder for predictions
        if self.config.target_for_prediction is None:
            print("Target for prediction not specified in configuration.")
            target_for_prediction = self.project.run_dir / "predictions"
            print(f"Defaulting to {target_for_prediction}")
        else:
            target_for_prediction = self.config.target_for_prediction

        # Run ensemble prediction
        full_inference_ensemble(
            models=models,
            data_loader=self.data_module.inference_dataloader(
                input_folder=self.config.source_for_prediction
            ),
            target_folder=target_for_prediction,
            roi_size=self.config.patch_size,
            batch_size=self.config.inference_batch_size,
            transpose=False,
            save_individual_preds=True,
            voting_mechanism="mode",
            weights=None,
            prefix="",
            output_dtype=self.config.output_dtype,
        )

    def _load_models(self, models_dir: Path):
        """Reload all models found in the model folds."""
        # Re-load all (best) models
        models = []
        fold = 0
        while True:
            # Look for the model for current fold
            model_paths = list((models_dir / f"fold_{fold}").glob("*.ckpt"))
            if not model_paths:
                break  # No more models

            model_path = model_paths[0]  # Assuming one model per fold

            # Load the model
            model_class = ModelFactory.get_model_class(config=self.config)
            model = model_class.load_from_checkpoint(
                model_path,
                criterion=self.criterion,
                metrics=self.metrics,
                class_names=self.config.class_names,
            )

            # Add it to the list
            models.append(model)

            # Inform
            print(f"Fold {fold}: re-loaded model = {model_path}")

            # Increase fold number
            fold += 1

        print(f"Loaded {len(models)} trained models.")
        return models


class RestorationDirector(Director):
    """Restoration Training Director."""

    def _setup_default_metrics(self):
        """Set up metrics for restoration."""
        return MeanAbsoluteError()

    def _setup_default_loss(self):
        """Set up loss function for restoration."""
        return MSELoss()

    def _setup_default_data_module(self):
        """Set up data module for restoration."""
        # Data module
        data_module = DataModuleLocalFolder(
            campaign_transforms=self.campaign_transforms,
            data_dir=self.config.data_dir,
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
            num_workers=self.num_workers,
        )

        # Return the data module
        return data_module

    def _setup_default_campaign_transforms(self):
        """Set up campaign transforms for restoration."""
        # Initialize default Restoration Campaign Transform
        campaign_transforms = RestorationCampaignTransforms(
            min_intensity=0,
            max_intensity=15472,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
        )

        # Return campaign transforms
        return campaign_transforms


class SegmentationDirector(Director):
    """Segmentation Training Director."""

    def _setup_default_metrics(self):
        """Set up metrics for segmentation."""
        return DiceMetric(
            include_background=self.config.include_background,
            reduction="mean_batch",
            get_not_nans=False,
        )

    def _setup_default_loss(self):
        """Set up loss function for segmentation."""
        return DiceCELoss(
            include_background=self.config.include_background,
            to_onehot_y=False,
            softmax=True,
        )

    def _setup_default_data_module(self):
        """Set up data module for segmentation."""
        # Data module
        data_module = DataModuleLocalFolder(
            campaign_transforms=self.campaign_transforms,
            data_dir=self.config.data_dir,
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
            num_workers=self.num_workers,
        )

        # Return data module
        return data_module

    def _setup_default_campaign_transforms(self):
        """Set up campaign transforms for segmentation."""
        # Consistency check
        if self.config.is_3d:
            raise ValueError("Check the value of `is_3d` in the configuration file.")

        # Initialize default Segmentation Campaign Transform
        campaign_transforms = SegmentationCampaignTransforms2D(
            num_classes=self.config.out_channels,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
        )

        # Return the campaign transforms
        return campaign_transforms


class SegmentationDirector3D(SegmentationDirector):
    """Segmentation 3D Training Director."""

    def _setup_default_campaign_transforms(self):
        """Set up campaign transforms for 3D segmentation."""
        # Consistency check
        if not self.config.is_3d:
            raise ValueError("Check the value of `is_3d` in the configuration file.")

        # Initialize default Segmentation Campaign Transform for 3D
        campaign_transforms = SegmentationCampaignTransforms3D(
            num_classes=self.config.out_channels,
            patch_size=self.config.patch_size,
            num_patches=self.config.num_patches,
            voxel_size=self.config.voxel_size,
            to_isotropic=self.config.to_isotropic,
            upscale_z=self.config.up_scale_z,
        )

        # Return the campaign transforms
        return campaign_transforms


class SelfSupervisedDirector(RestorationDirector):
    """Self-Supervised Training Director."""

    def _setup_default_campaign_transforms(self):
        """Set up campaign transforms for self-supervised restoration."""
        return SelfSupervisedRestorationCampaignTransforms()


class EnsembleSegmentationDirector(EnsembleDirector, SegmentationDirector):
    """Ensemble Segmentation Training Director."""

    def __init__(
        self,
        config_file: Union[Path, str],
        num_folds: int,
        num_workers: int = -1,
        *,
        campaign_transforms: Optional[CampaignTransforms] = None,
        data_module: Optional[pl.LightningDataModule] = None,
        criterion: Optional[Module] = None,
        metrics: Optional[Metric] = None,
        lr_scheduler_class: Optional[LRScheduler] = None,
        lr_scheduler_parameters: Optional[dict] = None,
    ) -> None:
        super().__init__(
            config_file,
            num_folds,
            num_workers,
            campaign_transforms=campaign_transforms,
            data_module=data_module,
            criterion=criterion,
            metrics=metrics,
            lr_scheduler_class=lr_scheduler_class,
            lr_scheduler_parameters=lr_scheduler_parameters,
        )

    def _setup_default_data_module(self):
        """Set up data module with folds."""
        # Data module
        data_module = DataModuleLocalFolder(
            campaign_transforms=self.campaign_transforms,
            data_dir=self.config.data_dir,
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
            num_workers=self.num_workers,
        )

        # Return data module
        return data_module

    def _setup_basis_for_training_and_resume(self):
        """Initialize the basic components for training or resume."""

        # Call parent method
        super()._setup_basis_for_training_and_resume()

        # Ensure that data module is prepared with folds
        self.data_module.prepare_data()
        self.data_module.setup("train")


class CellRestorationDemoDirector(RestorationDirector):
    """Restoration Demo Training Director."""

    def _setup_default_data_module(self):
        """Set up data module for cell restoration demo."""
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
            num_workers=self.num_workers,
        )

        # Return data module
        return data_module


class CellSegmentationDemoDirector(SegmentationDirector):
    """Segmentation Demo Training Director."""

    def _setup_default_data_module(self):
        """Set up data module for cell segmentation demo."""
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
            num_workers=self.num_workers,
        )

        # Return data module
        return data_module


class EnsembleCellSegmentationDemoDirector(EnsembleSegmentationDirector):
    """Ensemble Segmentation Demo Training Director."""

    def _setup_default_data_module(self):
        """Set up data module for ensemble cell segmentation demo."""
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
            num_workers=self.num_workers,
        )
        return data_module
