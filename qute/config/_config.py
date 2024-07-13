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
import configparser
import re
from pathlib import Path

from qute.mode import TrainerMode


class Config:
    def __init__(self, config_file):

        self._config_file = Path(config_file).resolve()
        self._config = None

    def parse(self) -> bool:

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
        return True

    @property
    def trainer_mode(self):
        return TrainerMode(self._config["settings"]["trainer_mode"])

    @property
    def project_dir(self):
        return Path(self._config["settings"]["project_dir"])

    @property
    def project_name(self):
        return Path(self.project_dir).name

    @property
    def data_dir(self):
        data_dir = self._config["settings"]["data_dir"]
        if Path(data_dir).is_absolute():
            return Path(data_dir)
        else:
            return self.project_dir / data_dir

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
        return Path(source_for_prediction)

    @property
    def target_for_prediction(self):
        target_for_prediction = self._config["settings"]["target_for_prediction"]
        if target_for_prediction == "":
            return None
        return Path(target_for_prediction)

    @property
    def source_model_path(self):
        source_model_path = self._config["settings"]["source_model_path"]
        if source_model_path == "":
            return None
        return Path(source_model_path)

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
    def include_background(self):
        include_background = self._config["settings"]["include_background"]
        return include_background.lower() == "true"

    @property
    def class_names(self):
        if "class_names" in self._config["settings"]:
            class_names = self._config["settings"]["class_names"]
            class_names = re.sub(r"\s+", "", class_names).split(",")
            return tuple(class_names)
        else:
            return []

    @property
    def max_epochs(self):
        return int(self._config["settings"]["max_epochs"])

    @property
    def precision(self):
        return self._config["settings"]["precision"]
