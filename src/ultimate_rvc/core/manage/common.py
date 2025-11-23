"""
Common definitions for modules in the Ultimate RVC project that
facilitate management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import operator
import shutil
from pathlib import Path

if TYPE_CHECKING:
    from ultimate_rvc.typing_extra import StrPath


def get_named_items(
    directory: StrPath,
    exclude: str | None = None,
    include_suffix: bool = True,
) -> list[tuple[str, str]]:
    """
    Get the names and paths of all items in the provided directory
    sorted by name.

    Parameters
    ----------
    directory : str
        The path to the directory containing the items.
    exclude : str, optional
        The file extension to exclude from the list of items.
    include_suffix : bool, optional
        Whether to include the file extension in the names of the items.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the names and paths of all items in
        the directory sorted by name.

    """
    dir_path = Path(directory)
    if dir_path.is_dir():
        named_items = sorted(
            [
                (item_path.name if include_suffix else item_path.stem, str(item_path))
                for item_path in dir_path.iterdir()
                if item_path.suffix != exclude
            ],
        )
        return sorted(named_items, key=operator.itemgetter(0))
    return []


def get_items(
    directory: StrPath,
    only_stem: bool = True,
    exclude: str | None = None,
) -> list[str]:
    """
    Get the names or paths of all items in the provided directory.

    Parameters
    ----------
    directory : str
        The path to the directory containing the items.
    only_stem : bool, optional
        Whether to return only the stem of the items or their paths.
        For directories, returns the full name. For files, returns the
        name without the file extension.
    exclude : str, optional
        The file extension to exclude from the list of items.

    Returns
    -------
    list[str]
        A list of the names or paths of all items in the directory.

    """
    dir_path = Path(directory)
    if dir_path.is_dir():
        return sorted(
            [
                (
                    (item_path.name if item_path.is_dir() else item_path.stem)
                    if only_stem
                    else str(item_path)
                )
                for item_path in dir_path.iterdir()
                if item_path.suffix != exclude
            ],
        )
    return []


def delete_directory(path: StrPath) -> None:
    """
    Delete a directory and its contents if it exists.

    Parameters
    ----------
    path : str
        The path to the directory to delete.

    """
    dir_path = Path(path)
    if dir_path.is_dir():
        shutil.rmtree(dir_path)
