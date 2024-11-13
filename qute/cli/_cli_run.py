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
run_app = typer.Typer(
    name="run", help="Run training, fine-tuning, prediction pipelines."
)


@run_app.command()
def segmentation(
    config: str = typer.Argument(
        ..., help="Full path to the configuration file.", show_default=False
    ),
    num_workers: int = typer.Option(
        -1,
        "-n",
        "--num_workers",
        help="Number of workers to be used for the dataloaders. Defaults to the number of CPU cores.",
        show_default=False,
    ),
):
    """Run a segmentation experiment specified by a configuration file."""

    # Check input argument config
    config_file = Path(config).resolve()
    if not config_file.is_file():
        raise ValueError(f"The specified config file {config_file} does not exist.")

    # Instantiate Director
    director = SegmentationDirector(config_file=config_file, num_workers=num_workers)

    # Run training
    director.run()
