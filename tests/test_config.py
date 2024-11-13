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
import os
import platform
import random
import string
from contextlib import contextmanager
from pathlib import Path

import pytest

from qute.config import Config
from qute.mode import TrainerMode


def test_reading_classification_conf():

    config = Config(
        Path(__file__).parent.parent
        / "config_samples"
        / "classification_project.ini_sample"
    )
    assert config.parse() is True, "Could not parse configuration file."
    assert config.trainer_mode == TrainerMode.TRAIN, "Wrong trainer mode."
    assert config.project_dir == Path("/path/to/project"), "Wrong project dir."
    assert config.project_name == "project", "Wrong project name."
    assert config.data_dir == Path("/path/to/project/ground_truth"), "Wrong data dir."
    assert config.in_channels == 1, "Wrong number of input channels."
    assert config.out_channels == 3, "Wrong number of outout channels."
    assert config.source_for_prediction is None, "Wrong source for prediction."
    assert config.target_for_prediction is None, "Wrong source for prediction."
    assert config.source_model_path is None, "Wrong source for prediction."
    assert config.source_images_sub_folder == "images", "Wrong source images subfolder."
    assert config.target_images_sub_folder == "labels", "Wrong target images subfolder."
    assert config.source_images_label == "image", "Wrong source images label."
    assert config.target_images_label == "label", "Wrong target images label."
    assert config.train_fraction == 0.7, "Wrong training fraction."
    assert config.val_fraction == 0.2, "Wrong validation fraction."
    assert config.test_fraction == 0.1, "Wrong test fraction."
    assert config.seed == 2022, "Wrong seed."
    assert config.batch_size == 8, "Wrong batch size."
    assert config.inference_batch_size == 4, "Wrong inference batch size."
    assert config.num_patches == 1, "Wrong number of patches."
    assert config.patch_size == (640, 640), "Wrong patch size."
    assert config.channels == (16, 32, 64), "Wrong channels."
    assert config.strides == (2, 2), "Wrong strides."
    assert config.num_res_units == 4, "Wrong number of residual units."
    assert config.learning_rate == 0.001, "Wrong learning rate."
    assert config.include_background, "Wrong value for include background."
    assert config.class_names == (
        "background",
        "cell",
        "membrane",
    ), "Wrong value for class names."
    assert config.max_epochs == 2000, "Wrong maximum number of eposchs."
    assert config.precision == "16-mixed", "Wrong precision."


def test_reading_regression_conf():

    config = Config(
        Path(__file__).parent.parent
        / "config_samples"
        / "regression_project.ini_sample"
    )
    assert config.parse() is True, "Could not parse configuration file."
    assert config.trainer_mode == TrainerMode.TRAIN, "Wrong trainer mode."
    assert config.project_dir == Path("/path/to/project"), "Wrong project dir."
    assert config.project_name == "project", "Wrong project name."
    assert config.data_dir == Path("/path/to/project/ground_truth"), "Wrong data dir."
    assert config.in_channels == 1, "Wrong number of input channels."
    assert config.out_channels == 1, "Wrong number of outout channels."
    assert config.source_for_prediction is None, "Wrong source for prediction."
    assert config.target_for_prediction is None, "Wrong source for prediction."
    assert config.source_model_path is None, "Wrong source for prediction."
    assert config.source_images_sub_folder == "images", "Wrong source images subfolder."
    assert (
        config.target_images_sub_folder == "targets"
    ), "Wrong target images subfolder."
    assert config.source_images_label == "image", "Wrong source images label."
    assert config.target_images_label == "target", "Wrong target images label."
    assert config.train_fraction == 0.7, "Wrong training fraction."
    assert config.val_fraction == 0.2, "Wrong validation fraction."
    assert config.test_fraction == 0.1, "Wrong test fraction."
    assert config.seed == 2022, "Wrong seed."
    assert config.batch_size == 8, "Wrong batch size."
    assert config.inference_batch_size == 4, "Wrong inference batch size."
    assert config.num_patches == 1, "Wrong number of patches."
    assert config.patch_size == (640, 640), "Wrong patch size."
    assert config.channels == (16, 32, 64), "Wrong channels."
    assert config.strides == (2, 2), "Wrong strides."
    assert config.num_res_units == 4, "Wrong number of residual units."
    assert config.learning_rate == 0.001, "Wrong learning rate."
    assert config.max_epochs == 2000, "Wrong maximum number of eposchs."
    assert config.precision == "16-mixed", "Wrong precision."


def test_process_path():
    @contextmanager
    def set_temp_env_var(key, value):
        # Save the original value of the environment variable, if it exists
        original_value = os.environ.get(key)

        # Set the new value for the environment variable
        os.environ[key] = value
        try:
            # Yield control back to the caller
            yield
        finally:
            # Restore the original value of the environment variable, if it existed
            if original_value is None:
                del os.environ[key]
            else:
                os.environ[key] = original_value

    path = "C:\\Users\\Username\\Documents\\"
    assert Config.process_path(path) == Path(
        "C:/Users/Username/Documents/"
    ), "Failed processing path."

    path = r"C:\Users\Username\Documents"
    assert Config.process_path(path) == Path(
        "C:/Users/Username/Documents/"
    ), "Failed processing path."

    path = "/home/Users/Username/Documents/"
    assert Config.process_path(path) == Path(
        "/home/Users/Username/Documents/"
    ), "Failed processing path."

    with set_temp_env_var("TEST_ENV", "test_subdir"):
        # Perform operations that require the temporary environment variable

        os_name = platform.system()
        if os_name == "Windows":

            path = r"${HOME}\Documents\${TEST_ENV}"
            username = os.getenv("USERNAME")
            assert Config.process_path(path) == Path(
                f"C:/Users/{username}/Documents/test_subdir"
            ), "Failed processing path."

        elif os_name == "Linux":

            path = "${HOME}/Documents/${TEST_ENV}"
            username = os.getenv("USER")
            assert Config.process_path(path) == Path(
                f"/home/{username}/Documents/test_subdir"
            ), "Failed processing path."

        elif os_name == "Darwin":

            path = "${HOME}/Documents/${TEST_ENV}"
            username = os.getenv("USER")
            assert Config.process_path(path) == Path(
                f"/Users/{username}/Documents/test_subdir"
            ), "Failed processing path."

        else:
            pass

    def generate_random_alphanumeric_string(length=24):
        # Define the characters to choose from (alphanumeric)
        characters = string.ascii_letters + string.digits

        # Generate a random string of the specified length
        random_string = "".join(random.choices(characters, k=length))

        return random_string

    # Check that a random 24-character string does not exist as environment variable
    with pytest.raises(ValueError):
        path = "${HOME}/Documents/${" + generate_random_alphanumeric_string() + "}"
        _ = Config.process_path(path)
