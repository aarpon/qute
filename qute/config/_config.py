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
import configparser
import os
import re
from abc import ABC
from pathlib import Path
from typing import Union

import numpy as np
import userpaths

from qute.mode import TrainerMode


class ConfigFactory:
    def __init__(self):
        raise Exception("Config Factory is not implemented")

    @staticmethod
    def get_config(config_path: Union[str, Path]):
        """Return the appropriate Configuration object for the passed configuration file."""

        # Make sure to work with a Path
        config_path = Path(config_path)

        # Read the configuration file
        if not config_path.is_file():
            print(f"{config_path} not found!")
            return None
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            assert config["metadata"].name == "metadata"
        except:
            print(f"{config_path} is not a valid metadata file!")
            return None

        if config["metadata"]["project_type"] == "classification":
            return ClassificationConfig(config_path)
        elif config["metadata"]["project_type"] == "regression":
            return RegressionConfig(config_path)
        elif config["metadata"]["project_type"] == "self-supervised-classification":
            return SelfSupervisedClassificationConfig(config_path)
        else:
            print("Unsupported configuration file!")
            return None


class Config(ABC):
    """Abstract base class for configuration objects."""

    def __init__(self, config_file):
        self._config_file = Path(config_file).resolve()
        self._config = None

    def parse(self) -> bool:
        """Parse and validate the configuration file."""

        # Read the configuration file
        if not self._config_file.is_file():
            print("config.ini not found!")
            return False
        try:
            self._config = configparser.ConfigParser()
            self._config.read(self._config_file)
            assert self._config["settings"].name == "settings"
        except:
            return False

        # Validate the configuration
        return self._validate()

    @staticmethod
    def process_path(path: Union[Path, str, None]) -> Union[Path, None]:
        """Process a path string with optional environmental variables.

        Parameters
        ----------

        path: Union[Path, str, None]
            Full path that can optionally contain environment variables in the
            form ${ENV_VAR}. For instance: `${HOME}/Documents/qute`. If path is
            None, None is returned.

            Please notice that ${HOME} will be considered to point to the user path
            also in Windows, where it is not defined as an environment variable. All
            other variables, must be defined in os.environ.

        Returns
        -------

        path: Union[Path, None]
            Path with expanded environment variables (if present), or None.
        """

        # Find environment variables of the form ${ENV_VAR} in the string
        pattern = r"\$\{.+?\}"

        # Make sure to work with a string version of the absolute path
        # with forward slashes only
        path_str = str(Path(path))
        path_str = path_str.replace("\\", "/")

        # Find all substrings
        matches = re.findall(pattern, path_str)

        # Process matches
        for match in matches:
            # We treat $HOME specially
            if match == "${HOME}":
                match_rep = userpaths.get_profile()
            else:
                match_rep = os.getenv(match[2:-1], None)
                if match_rep is None:
                    raise ValueError(f"Undefined environment variable {match}.")
            index = path_str.find(match)
            if index == -1:
                raise ValueError(f"Could not find {match} in {path_str}.")
            path_str = f"{path_str[:index]}{match_rep}{path_str[index + len(match) :]}"

        # Make sure to remove double forward slashes potentially introduced
        # by the substitution
        path_str = re.sub(r"/+", "/", path_str)

        # Now cast to a pathlib.Path() and return
        return Path(path_str)

    @property
    def checkpoint_monitor(self):
        target = self._config["settings"]["checkpoint_monitor"]
        if "checkpoint_metrics_class" in self._config["settings"]:
            metrics_class = self._config["settings"]["checkpoint_metrics_class"]
        else:
            metrics_class = ""
        if target == "loss":
            return "val_loss"
        elif target == "metrics":
            if metrics_class == "":
                return "val_metrics"
            else:
                return f"val_metrics_{metrics_class}"
        else:
            raise ValueError("`checkpoint_monitor` must be one of 'loss' or 'metrics'.")

    @property
    def checkpoint_mode(self):
        if self.checkpoint_monitor == "val_loss":
            return "min"
        elif "val_metric" in self.checkpoint_monitor:
            return "max"
        else:
            raise ValueError("`checkpoint_monitor` must be one of 'loss' or 'metric'.")

    @property
    def use_early_stopping(self):
        use_early_stopping = self._config["settings"]["use_early_stopping"]
        return use_early_stopping.lower() == "true"

    @property
    def early_stopping_patience(self):
        early_stopping_patience = int(
            self._config["settings"]["early_stopping_patience"]
        )
        if self.use_early_stopping:
            if early_stopping_patience < 1:
                raise ValueError(
                    "`early_stopping_patience` must be greater than or equal to 1."
                )
        return early_stopping_patience

    @property
    def config_file(self) -> Path:
        return self._config_file

    @property
    def model_class(self):
        return self._config["settings"]["model_class"]

    @property
    def trainer_mode(self):
        return TrainerMode(self._config["settings"]["trainer_mode"])

    @property
    def project_dir(self):
        return Config.process_path(Path(self._config["settings"]["project_dir"]))

    @property
    def project_name(self):
        return Path(self.project_dir).name

    @property
    def data_dir(self):
        data_dir = self._config["settings"]["data_dir"]
        if Path(data_dir).is_absolute():
            data_dir = Path(data_dir)
        else:
            data_dir = self.project_dir / data_dir
        return Config.process_path(data_dir)

    @property
    def in_channels(self):
        return int(self._config["settings"]["in_channels"])

    @property
    def out_channels(self):
        return int(self._config["settings"]["out_channels"])

    @property
    def source_for_prediction(self):
        source_for_prediction = self._config["settings"]["source_for_prediction"]
        if source_for_prediction == "":
            return None
        return Config.process_path(Path(source_for_prediction))

    @property
    def target_for_prediction(self):
        target_for_prediction = self._config["settings"]["target_for_prediction"]
        if target_for_prediction == "":
            return None
        return Config.process_path(Path(target_for_prediction))

    @property
    def source_model_path(self):
        source_model_path = self._config["settings"]["source_model_path"]
        if source_model_path == "":
            return None
        return Config.process_path(Path(source_model_path))

    @property
    def source_images_sub_folder(self):
        return self._config["settings"]["source_images_sub_folder"]

    @property
    def target_images_sub_folder(self):
        return self._config["settings"]["target_images_sub_folder"]

    @property
    def source_images_label(self):
        return self._config["settings"]["source_images_label"]

    @property
    def target_images_label(self):
        return self._config["settings"]["target_images_label"]

    @property
    def train_fraction(self):
        train_fraction = self._config["settings"]["train_fraction"]
        if train_fraction == "":
            return None
        return float(train_fraction)

    @property
    def val_fraction(self):
        val_fraction = self._config["settings"]["val_fraction"]
        if val_fraction == "":
            return None
        return float(val_fraction)

    @property
    def test_fraction(self):
        test_fraction = self._config["settings"]["test_fraction"]
        if test_fraction == "":
            return None
        return float(test_fraction)

    @property
    def seed(self):
        return int(self._config["settings"]["seed"])

    @property
    def batch_size(self):
        return int(self._config["settings"]["batch_size"])

    @property
    def inference_batch_size(self):
        return int(self._config["settings"]["inference_batch_size"])

    @property
    def num_patches(self):
        return int(self._config["settings"]["num_patches"])

    @property
    def patch_size(self):
        patch_size_str = self._config["settings"]["patch_size"]
        patch_size = list(re.sub(r"\s+", "", patch_size_str).split(","))
        for i, element in enumerate(patch_size):
            patch_size[i] = int(element)
        return tuple(patch_size)

    @property
    def channels(self):
        channels_str = self._config["settings"]["channels"]
        channels = list(re.sub(r"\s+", "", channels_str).split(","))
        for i, element in enumerate(channels):
            channels[i] = int(element)
        return tuple(channels)

    @property
    def strides(self):
        strides_str = self._config["settings"]["strides"]
        if strides_str == "":
            return None
        strides = list(re.sub(r"\s+", "", strides_str).split(","))
        for i, element in enumerate(strides):
            strides[i] = int(element)
        return tuple(strides)

    @property
    def num_res_units(self):
        return int(self._config["settings"]["num_res_units"])

    @property
    def learning_rate(self):
        return float(self._config["settings"]["learning_rate"])

    @property
    def deep_supervision(self):
        deep_supervision_str = self._config["settings"]["deep_supervision"]
        return deep_supervision_str.lower() == "true"

    @property
    def deep_supr_num(self):
        return int(self._config["settings"]["deep_supr_num"])

    @property
    def res_block(self):
        res_block_str = self._config["settings"]["res_block"]
        return res_block_str.lower() == "true"

    @property
    def max_epochs(self):
        return int(self._config["settings"]["max_epochs"])

    @property
    def precision(self):
        return self._config["settings"]["precision"]

    @property
    def depths(self):
        depths_str = self._config["settings"]["depths"]
        if depths_str == "":
            return None
        depths = list(re.sub(r"\s+", "", depths_str).split(","))
        for i, element in enumerate(depths):
            depths[i] = int(element)
        return tuple(depths)

    @property
    def num_heads(self):
        num_heads_str = self._config["settings"]["num_heads"]
        if num_heads_str == "":
            return None
        num_heads = list(re.sub(r"\s+", "", num_heads_str).split(","))
        for i, element in enumerate(num_heads):
            num_heads[i] = int(element)
        return tuple(num_heads)

    @property
    def feature_size(self):
        return int(self._config["settings"]["feature_size"])

    @property
    def output_dtype(self):
        out_dtype = self._config["settings"]["output_dtype"]
        try:
            out_dtype = np.dtype(out_dtype)
        except (TypeError, ValueError):
            print(f"{out_dtype} is not a valid output dtype.")
        return out_dtype

    @property
    def is_3d(self):
        is_3d_str = self._config["settings"]["is_3d"]
        return is_3d_str.lower() == "true"

    @property
    def to_isotropic(self):
        to_isotropic_str = self._config["settings"]["to_isotropic"]
        return to_isotropic_str.lower() == "true"

    @property
    def up_scale_z(self):
        up_scale_z_str = self._config["settings"]["up_scale_z"]
        return up_scale_z_str.lower() == "true"

    @property
    def use_v2(self):
        use_v2_str = self._config["settings"]["use_v2"]
        return use_v2_str.lower() == "true"

    @property
    def voxel_size(self):
        voxel_size_str = self._config["settings"]["voxel_size"]
        if voxel_size_str == "":
            return None
        voxel_size = list(re.sub(r"\s+", "", voxel_size_str).split(","))
        for i, element in enumerate(voxel_size):
            voxel_size[i] = float(element)
        return tuple(voxel_size)

    def _validate(self):
        """Validate configuration."""

        # Check the model class
        if self.model_class not in ["unet", "attention_unet", "swin_unetr", "dynunet"]:
            print(
                "`model_class` must be one of 'unet', 'attention_unet', 'swin_unetr', or 'dynunet'."
            )
            return False

        # Check the trainer mode
        if self.trainer_mode not in ["train", "resume", "predict"]:
            print("`trainer_mode` must be one of 'train', 'resume', or 'predict'.")
            return False

        # Validate checkpoint_monitor
        checkpoint_monitor = self._config["settings"]["checkpoint_monitor"]
        if checkpoint_monitor not in ["loss", "metrics"]:
            print("`checkpoint_monitor` must be one of 'loss' or 'metrics'.")
            return False

        # Validate checkpoint_metric_class (for segmentation jobs)
        if "checkpoint_metrics_class" in self._config["settings"]:
            checkpoint_metrics_class = self._config["settings"][
                "checkpoint_metrics_class"
            ]
            if checkpoint_metrics_class != "":
                if checkpoint_metrics_class not in self.class_names:
                    print("`checkpoint_metrics_class` must be a valid class name.")
                    return False

        # Validate early stopping and patience
        if self._config["settings"]["use_early_stopping"].lower() not in [
            "true",
            "false",
        ]:
            print("Bad value for `use_early_stopping`.")
            return False

        patience = self._config["settings"]["early_stopping_patience"]
        if patience != "":
            try:
                patience = int(patience)
                assert patience > 0
            except:
                print("`early_stopping_patience` must be greater than 0.")
                return False

        # @TODO Complete the checks.

        # Return success
        return True


class ClassificationConfig(Config):
    """Classification configuration object."""

    def __init__(self, config_file):
        super().__init__(config_file)

    @property
    def class_names(self):
        if "class_names" in self._config["settings"]:
            class_names = self._config["settings"]["class_names"]
            class_names = re.sub(r"\s+", "", class_names).split(",")
            return tuple(class_names)
        else:
            return []

    @property
    def include_background(self):
        include_background = self._config["settings"]["include_background"]
        return include_background.lower() == "true"


class RegressionConfig(Config):
    """Classification configuration object."""

    def __init__(self, config_file):
        super().__init__(config_file)


class SelfSupervisedClassificationConfig(Config):
    """Classification configuration object."""

    def __init__(self, config_file):
        raise NotImplementedError("Not implemented yet!")
