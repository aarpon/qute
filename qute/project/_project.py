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
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from qute.config import Config


class Project:
    """Sets up Project directory structure."""

    def __init__(self, config: Config, clean: bool = False):
        """

        Parameters
        ----------

        config: qute.config.Config
            Configuration for current project.

        clean: bool = False
            Set to True to clean the project directory from failed or incomplete runs.
        """

        # Store reference to the configuration
        self._config = config

        # Internal properties
        self._project_dir: Path = self._config.project_dir
        self._runs_dir: Path = self._project_dir / "runs"
        self._data_dir: Path = self._config.data_dir
        self._selected_model_path: Union[None, Path, str] = None
        self._target_for_prediction_path = self._config.target_for_prediction
        self._source_for_prediction_path = self._config.source_for_prediction

        # Set the model path
        self._set_selected_model(self._config.source_model_path)

        # Set up project
        self._project_dir.mkdir(exist_ok=True, parents=True)
        self._runs_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir.mkdir(exist_ok=True, parents=True)

        # Create new run
        self._run_dir = None
        self._models_dir = None
        self._results_dir = None
        self.new_run()

        # Clean if needed
        if clean is True:
            self.clean()

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir: Union[Path, str]):
        raise RuntimeError("Cannot override data_dir!")

    @property
    def selected_model_path(self) -> Path:
        return self._selected_model_path

    @selected_model_path.setter
    def selected_model_path(self, selected_model_path: Union[Path, str]):
        """Override the model from the configuration file."""
        self._set_selected_model(selected_model_path)

    @property
    def source_for_prediction(self) -> Path:
        return self._source_for_prediction_path

    @property
    def target_for_prediction(self) -> Path:
        return self._target_for_prediction_path

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    @models_dir.setter
    def models_dir(self, models_dir: Union[Path, str]):
        raise RuntimeError("Cannot override models_dir!")

    @property
    def results_dir(self) -> Path:
        return self._results_dir

    @results_dir.setter
    def results_dir(self, results_dir: Union[Path, str]):
        raise RuntimeError("Cannot override results_dir!")

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @run_dir.setter
    def run_dir(self, run_dir: Union[Path, str]):
        raise RuntimeError("Cannot override run_dir!")

    def new_run(self):
        """Create a new run with model and results subfolders."""
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_dir = self._runs_dir / name
        self._models_dir = self._run_dir / "models"
        self._models_dir.mkdir(parents=True)
        self._results_dir = self._run_dir / "results"
        self._results_dir.mkdir(parents=True)
        if self._target_for_prediction_path is None:
            self._target_for_prediction_path = self._project_dir / "predictions" / name
            self._target_for_prediction_path.mkdir(exist_ok=True, parents=True)

    def _set_selected_model(self, model_path: Union[None, Path, str] = None):
        # Make sure the passed project exists
        if model_path is None or model_path == "":
            self._selected_model_path = None
            return

        model_path = Path(model_path)
        if not model_path.is_file():
            raise IOError(f"The selected model {model_path} does not exist.")
        self._selected_model_path = model_path

    def models(self) -> List[Path]:
        """Return a list of all models available in the project."""
        if self._run_dir is None:
            return []
        return list(self._run_dir.rglob(f"*.ckpt"))

    def _is_valid_run_name(self, run) -> bool:
        """Check whether the run has a valid name."""
        # Check run directory name format
        name = run.name
        len_correct = len(name) == 15
        try:
            _ = int(name[:8])
            date_correct = True
        except ValueError:
            date_correct = False
        try:
            _ = int(name[-6:])
            time_correct = True
        except ValueError:
            time_correct = False
        if not len_correct or not date_correct or not time_correct:
            return False

        return True

    def clean(self):
        """Clean incomplete runs and predictions."""

        # Check runs
        for run in self._runs_dir.iterdir():
            to_clean = False
            if not run.is_dir():
                continue

            # Check run directory name format
            if not self._is_valid_run_name(run):
                continue

            # Make sure not to delete current run
            if self._run_dir == run:
                # This is current run and won't have any models or results yet
                continue

            models_dir = Path(run) / "models"
            if not models_dir.is_dir():
                to_clean = True
            else:
                models_found = list(models_dir.rglob(f"*.ckpt"))
                if len(models_found) == 0:
                    to_clean = True

            results_dir = Path(run) / "results"
            if not results_dir.is_dir():
                to_clean = True
            else:
                logs_found = list(results_dir.rglob(f"*version*"))
                if len(logs_found) == 0:
                    to_clean = True

            if to_clean:
                # Remove folder recursively
                try:
                    shutil.rmtree(self._runs_dir / run.name)
                except Exception as e:
                    print(e)

        # Check predictions
        root_predictions = self._project_dir / "predictions"
        for pred in root_predictions.iterdir():
            to_clean = False
            if not pred.is_dir():
                continue

            # Check run directory name format
            if not self._is_valid_run_name(pred):
                continue

            # Make sure not to delete current prediction folder
            if self._target_for_prediction_path == pred:
                # This is current run and won't have any models or results yet
                continue

            # Are there predictions?
            predictions_found = list(pred.rglob(f"*.tif"))
            if len(predictions_found) == 0:
                to_clean = True

            if to_clean:
                # Remove folder recursively
                try:
                    shutil.rmtree(root_predictions / pred.name)
                except Exception as e:
                    print(e)
