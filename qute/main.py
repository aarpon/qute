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

import typer

from qute import __version__
from qute.cli import config_app, run_app

app = typer.Typer(no_args_is_help=True)


@app.command()
def version():
    """Print version information."""
    typer.echo(__version__)


# Add sub-commands
app.add_typer(config_app)
app.add_typer(run_app)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass  # Start method is already set, ignore the error

    # Run app
    app()

    # Properly exit
    sys.exit(0)
