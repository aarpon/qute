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

import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import typer

from qute import __version__
from qute.config import (
    ClassificationConfig,
    Config,
    ConfigFactory,
    RegressionConfig,
    SelfSupervisedClassificationConfig,
)
from qute.director import (
    EnsembleCellSegmentationDemoDirector,
    RestorationDirector,
    SegmentationDirector,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Command-line interface to run various qute jobs.",
)


@app.command()
def run(
    config: str = typer.Option(
        ...,
        "--config",
        "-c",
        help="Full path to the configuration file.",
        show_default=False,
    ),
    ensemble: bool = typer.Option(
        False,
        "--ensemble",
        "-e",
        is_flag=True,
        help="Set to run an ensemble pipeline with specified number of folds.",
        show_default=True,
    ),
    num_folds: int = typer.Option(
        1,
        "-f",
        "--num_folds",
        help="Number of folds for cross-correlation validation (only used for ensemble pipelines).",
        show_default=True,
    ),
    num_workers: int = typer.Option(
        -1,
        "-n",
        "--num_workers",
        help="Number of workers to be used for the dataloaders. Defaults to the number of CPU cores.",
        show_default=False,
    ),
):
    """Run experiment specified by a configuration file."""

    # Check input argument config
    config_file = Path(config).resolve()
    if not config_file.is_file():
        raise ValueError(f"The specified config file {config_file} does not exist.")

    # Get correct configuration class
    config = ConfigFactory.get_config(config_file)

    if ensemble:
        if num_folds == 1:
            raise ValueError(
                "Please specify the number of folds for cross-correlation validation."
            )

    if not ensemble:
        # If ensemble is not required, just ignore andy passed number of folds.
        num_folds = 1

    # Instantiate Director
    if isinstance(config, ClassificationConfig):
        if num_folds == 1:
            director = SegmentationDirector(
                config_file=config_file, num_workers=num_workers
            )
        elif num_folds > 1:
            director = EnsembleCellSegmentationDemoDirector(
                config_file=config_file,
                num_folds=num_folds,
                num_workers=num_workers,
            )
        else:
            raise ValueError(f"Unexpected value for `num_folds` ({num_folds}).")
    elif isinstance(config, RegressionConfig):
        director = RestorationDirector(config_file=config_file, num_workers=num_workers)
    elif isinstance(config, SelfSupervisedClassificationConfig):
        raise ValueError(
            "Self-supervised classifications not explicitly supported by `qute run` yet."
        )
    else:
        raise ValueError(
            f"{config.__name__} not explicitly supported by `qute run` yet."
        )

    # Run training
    director.run()


@app.command()
def version(
    detailed: Optional[bool] = typer.Option(
        False,
        "-d",
        "--detailed",
        help="Show additional version information.",
        show_default=False,
    )
):
    """Print (detailed) version information."""
    if detailed:
        """Display detailed version information."""
        import subprocess

        import pytorch_lightning as pl
        import torch
        import torchvision

        typer.echo(f"qute {__version__}")
        typer.echo(
            f"PyTorch {torch.__version__} with torchvision {torchvision.__version__}"
        )
        typer.echo(f"PyTorch-Lightning {pl.__version__}")
        try:
            cmd = ["nvidia-smi", "--version"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output = result.stdout.decode("utf-8")
            typer.echo(output)
        except FileNotFoundError:
            pass
    else:
        typer.echo(__version__)


# Instantiate Typer
config_app = typer.Typer(name="config", help="Manage configuration options.")


@config_app.command()
def create(
    category: str = typer.Argument(
        ..., help="Category of the configuration file to write.", show_default=False
    ),
    target: str = typer.Argument(
        ..., help="Full path to the configuration file to write.", show_default=False
    ),
):
    """Create default project configuration files."""

    # Define a mapping for category to template file
    category_to_template = {
        "classification": "classification_project.ini",
        "c": "classification_project.ini",
        "regression": "regression_project.ini",
        "r": "regression_project.ini",
        "self-supervised-classification": "self_supervised_classification_project.ini",
        "ssc": "self_supervised_classification_project.ini",
    }

    # Check if the category is valid
    if category not in category_to_template:
        raise ValueError(
            "The category must be 'classification' or 'regression' for now."
        )

    # Determine target path
    target = Path(target)
    template_file = category_to_template[category]
    template_source = (
        Path(__file__).parent.parent.parent
        / "config_samples"
        / f"{template_file}_sample"
    )

    # Resolve final target file path
    if target.is_dir():
        target = target / template_file
    elif target.is_file():
        typer.echo(
            typer.style(
                f"Error: {target} already exists!",
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(1)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)

    # Attempt to copy the template file
    try:
        shutil.copy(template_source, target)
        print(f"Successfully generated {category} configuration file at {target}.")
    except Exception as e:
        print(f"Could not generate {category} configuration file at {target}: {e}")


@config_app.command(name="list")
def show_categories():
    """List supported configuration categories."""
    typer.echo(
        """
Available configuration categories:
  - classification ("c")
  - regression ("r")
  - self-supervised-classification ("ssc")
"""
    )


app.add_typer(config_app, name="config")
