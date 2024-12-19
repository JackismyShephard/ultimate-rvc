"""
Module which defines the command-line interface for the Ultimate RVC
project.
"""

from __future__ import annotations

import sys

from cyclopts import App

from ultimate_rvc.cli.generate.main import new_app as generate_app

app = App(name="urvc-cli", help="CLI for the Ultimate RVC project.")

app.command(generate_app)


if __name__ == "__main__":
    sys.exit(app())
