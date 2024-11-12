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
from pathlib import Path

import typer

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
        "segmentation": "classification_project.ini",
        "regression": "regression_project.ini",
    }

    # Check if the category is valid
    if category not in category_to_template:
        raise ValueError("The category must be 'segmentation' or 'regression' for now.")

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
    except Exception as e:
        print(f"Could not generate {category} configuration file at {target}: {e}")
