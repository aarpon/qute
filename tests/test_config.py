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
from pathlib import Path

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
    assert config.out_channels == 2, "Wrong number of outout channels."
    assert config.source_for_prediction is None, "Wrong source for prediction."
    assert config.target_for_prediction is None, "Wrong source for prediction."
    assert (
        not config.fine_tune_from_self_supervised
    ), "Wrong value for fine_tune_from_self_supervised."
    assert config.source_model_path is None, "Wrong source for prediction."
    assert config.source_images_sub_folder == "images", "Wrong source images subfolder."
    assert config.target_images_sub_folder == "labels", "Wrong target images subfolder."
    assert config.source_images_label == "image", "Wrong source images label."
    assert config.target_images_label == "label", "Wrong target images label."
    assert config.train_fraction is None, "Wrong training fraction."
    assert config.val_fraction is None, "Wrong validation fraction."
    assert config.test_fraction is None, "Wrong test fraction."
    assert config.seed == 2022, "Wrong seed."
    assert config.batch_size == 8, "Wrong batch size."
    assert config.inference_batch_size == 8, "Wrong inference batch size."
    assert config.num_patches == 4, "Wrong number of patches."
    assert config.patch_size == (1024, 1024), "Wrong patch size."
    assert config.channels == (16, 32, 64), "Wrong channels."
    assert config.strides == (2, 2), "Wrong strides."
    assert config.num_res_units == 4, "Wrong number of residual units."
    assert config.learning_rate == 0.001, "Wrong learning rate."
    assert config.include_background, "Wrong value for include background."
    assert config.class_names == ("background", "cell"), "Wrong value for class names."
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
    assert config.train_fraction is None, "Wrong training fraction."
    assert config.val_fraction is None, "Wrong validation fraction."
    assert config.test_fraction is None, "Wrong test fraction."
    assert config.seed == 2022, "Wrong seed."
    assert config.batch_size == 8, "Wrong batch size."
    assert config.inference_batch_size == 8, "Wrong inference batch size."
    assert config.num_patches == 4, "Wrong number of patches."
    assert config.patch_size == (1024, 1024), "Wrong patch size."
    assert config.channels == (16, 32, 64), "Wrong channels."
    assert config.strides == (2, 2), "Wrong strides."
    assert config.num_res_units == 4, "Wrong number of residual units."
    assert config.learning_rate == 0.001, "Wrong learning rate."
    assert config.max_epochs == 2000, "Wrong maximum number of eposchs."
    assert config.precision == "16-mixed", "Wrong precision."
