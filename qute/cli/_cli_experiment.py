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

import typer

from qute.director import SegmentationDirector

# Instantiate Typer
experiment_app = typer.Typer(
    name="experiment", help="Run training, fine-tuning, prediction pipelines."
)


@experiment_app.command()
def run(
    category: str = typer.Argument(
        "segmentation", help="Category of the run.", show_default=True
    ),
    config: str = typer.Argument(
        ..., help="Full path to the configuration file.", show_default=False
    ),
):
    """Run a campaign specified by a configuration file."""

    # Check input argument category
    category = category.lower()
    if category != "segmentation":
        raise ValueError("The category must be 'segmentation' for now.")

    # Check input argument config
    config_file = Path(config).resolve()
    if not config_file.is_file():
        raise ValueError(f"The specified config file {config_file} does not exist.")

    # Run the task
    if category == "segmentation":
        # Instantiate Director: make sure to instantiate it from a script entry point
        director = SegmentationDirector(config_file=config_file)
    else:
        raise ValueError("The category must be 'segmentation' for now.")

    # Run training
    director.run()
