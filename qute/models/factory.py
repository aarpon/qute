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

from abc import ABC
from typing import Optional

from monai.metrics import Metric
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler

from qute.campaigns import CampaignTransforms
from qute.config import Config
from qute.models.attention_unet import AttentionUNet
from qute.models.base_model import BaseModel
from qute.models.dynunet import DynUNet
from qute.models.swinunetr import SwinUNETR
from qute.models.unet import UNet


class ModelFactory(ABC):
    def __init__(self):
        raise Exception("Model Factory is not implemented")

    @staticmethod
    def get_model(
        config: Config,
        campaign_transforms: CampaignTransforms,
        criterion: Module,
        metrics: Metric,
        lr_scheduler_class: Optional[LRScheduler] = None,
        lr_scheduler_params: Optional[dict] = None,
    ) -> BaseModel:
        """Instantiate the model based on configuration parameters, criterion, metric and scheduler.

        Parameters
        ----------

        config: Config
            Loaded and processed configuration.

        campaign_transforms: Optional[CampaignTransforms] = None
            CampaignTransform to use for data transformations.

        criterion: Optional[Loss] = None
            Loss to use for optimization.

        metrics: Optional[Metric] = None
            Metrics to use for validation.

        lr_scheduler_class: Optional[LRScheduler] = None
            Learning rate scheduler to use.

        lr_scheduler_parameters: Optional[dict] = None
            Parameters for learning rate scheduler.

        Returns
        -------

        model: BaseModel
            The instantiated model object.
        """

        # Get the model class
        model_class = ModelFactory.get_model_class(config)

        # In case of a restoration model, we do not have class names
        if hasattr(config, "class_names"):
            class_names = config.class_names
        else:
            class_names = []

        # Prepare common model parameters
        model_params = {
            "campaign_transforms": campaign_transforms,
            "spatial_dims": 3 if config.is_3d else 2,
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
            "class_names": class_names,
            "criterion": criterion,
            "metrics": metrics,
            "learning_rate": config.learning_rate,
            "lr_scheduler_class": lr_scheduler_class,
            "lr_scheduler_parameters": (
                lr_scheduler_params if lr_scheduler_params is not None else {}
            ),
        }

        # Add additional model-specific parameters
        if config.model_class == "unet":
            model_params.update(
                {
                    "num_res_units": config.num_res_units,
                    "channels": config.channels,
                    "strides": config.strides,
                }
            )
        elif config.model_class == "attention_unet":
            model_params.update(
                {
                    "channels": config.channels,
                    "strides": config.strides,
                }
            )
        elif config.model_class == "swin_unetr":
            model_params.update(
                {
                    "depths": config.depths,
                    "num_heads": config.num_heads,
                    "feature_size": config.feature_size,
                }
            )
        elif config.model_class == "dynunet":
            # @TODO: add parameters to (and from) config
            model_params.update(
                {
                    "deep_supervision": True,
                }
            )

        # Instantiate the model
        model = model_class(**model_params)

        # Inform
        print(f"Using model: {model_class.__name__}")

        # Return the model
        return model

    @staticmethod
    def get_model_class(config: Config):
        """Return the class of the model being used."""

        # Get the model class
        models = {
            "unet": UNet,
            "attention_unet": AttentionUNet,
            "swin_unet": SwinUNETR,
            "dynunet": DynUNet,
        }
        model_class = models.get(config.model_class, None)
        if model_class is None:
            raise ValueError(
                "The 'model_class' must be one of 'unet', 'attention_unet', 'swin_unetr', or 'dynunet'."
            )
        return model_class
