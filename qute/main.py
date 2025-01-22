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
import multiprocessing as mp
import sys
from typing import Optional

import typer

from qute import __version__
from qute.cli import config_app, run_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Command-line interface to run various qute jobs.",
)


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
    """Print version information."""
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


# Add sub-commands
app.add_typer(config_app)
app.add_typer(run_app)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    # Run app
    app()

    # Properly exit
    sys.exit(0)
