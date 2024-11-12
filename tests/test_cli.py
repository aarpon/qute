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

import pytest
import typer
from typer.testing import CliRunner

from qute import __version__
from qute.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_create_classification_config(tmp_path):

    # Pass directory
    out_config_dir = tmp_path
    result = runner.invoke(
        app, ["config", "create", "segmentation", str(out_config_dir)]
    )
    assert result.exit_code == 0
    assert Path(out_config_dir / "classification_project.ini").exists()

    # Pass file
    out_config_file = tmp_path / "classification_config.ini"
    result = runner.invoke(
        app, ["config", "create", "segmentation", str(out_config_file)]
    )
    assert result.exit_code == 0
    assert Path(out_config_file).exists()

    # Pass directory
    out_config_dir = tmp_path
    result = runner.invoke(app, ["config", "create", "regression", str(out_config_dir)])
    assert result.exit_code == 0
    assert Path(out_config_dir / "regression_project.ini").exists()

    # Pass file
    out_config_file = tmp_path / "regression_config.ini"
    result = runner.invoke(
        app, ["config", "create", "regression", str(out_config_file)]
    )
    assert result.exit_code == 0
    assert Path(out_config_file).exists()
