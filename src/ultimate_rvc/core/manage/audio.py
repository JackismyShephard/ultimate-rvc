"""Module which defines functions to manage audio files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
from pathlib import Path

from ultimate_rvc.common import AUDIO_DIR
from ultimate_rvc.core.common import (
    INTERMEDIATE_AUDIO_BASE_DIR,
    OUTPUT_AUDIO_DIR,
    SPEECH_DIR,
    TRAINING_AUDIO_DIR,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    InvalidLocationError,
    Location,
    NotFoundError,
    NotProvidedError,
    UIMessage,
)
from ultimate_rvc.core.manage.common import (
    delete_directory,
    get_items,
    get_named_items,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ultimate_rvc.typing_extra import StrPath


def get_saved_output_audio() -> list[tuple[str, str]]:
    """
    Get the name and path of all output audio files.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the name and path of each output
        audio file.

    """
    return get_named_items(OUTPUT_AUDIO_DIR)


def get_saved_speech_audio() -> list[tuple[str, str]]:
    """
    Get the name and path of all speech audio files.

    Returns
    -------
    list[tuple[str, Path]]
        A list of tuples containing the name and path of each
        speech audio file.

    """
    return get_named_items(SPEECH_DIR, exclude=".json")


def get_named_audio_datasets() -> list[tuple[str, str]]:
    """
    Get the name and path of all saved audio datasets.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the name and path of each saved
        audio dataset.

    """
    return get_named_items(TRAINING_AUDIO_DIR)


def get_audio_datasets() -> list[str]:
    """
    Get the paths of all saved audio datasets.

    Returns
    -------
    list[str]
        A list of the paths of all saved audio datasets.

    """
    return get_items(TRAINING_AUDIO_DIR, only_name=False)


def delete_audio(
    directory: StrPath,
    items: Sequence[StrPath],
    plural_entity: Entity = Entity.FILES,
    singular_entity: Entity = Entity.FILE,
    location: Location = Location.AUDIO_ROOT,
    ui_msg: UIMessage = UIMessage.NO_UPLOADED_FILES,
) -> None:
    """
    Delete the provided audio items.

    the provided audio items must be located in the root of the provided
    directory.

    Parameters
    ----------
    directory : StrPath
        The path to the directory containing the audio items to delete
    items : Sequence[StrPath]
        Paths to the audio items to delete.
    plural_entity : Entity, optional
        The plural form of the entity being deleted.
    singular_entity : Entity, optional
        The singular form of the entity being deleted.
    location : Location, optional
        The location where the files should be located.
    ui_msg : UIMessage, optional
        The message to display if no items are provided.

    Raises
    ------
    NotProvidedError
        If no paths are provided.
    NotFoundError
        If a provided path does not point to an existing audio item.
    InvalidLocationError
        If a provided path does not point to a location in the root of
        the provided directory.

    """
    if not items:
        raise NotProvidedError(entity=plural_entity, ui_msg=ui_msg)
    item_paths: list[Path] = []
    for item in items:
        item_path = Path(item)
        if not item_path.exists():
            raise NotFoundError(entity=singular_entity, location=item_path)
        if item_path.parent != Path(directory):
            raise InvalidLocationError(singular_entity, location, item_path)
        item_paths.append(item_path)
    for item_path in item_paths:
        if item_path.is_dir():
            shutil.rmtree(item_path)
        else:
            json_path = item_path.with_suffix(".json")
            item_path.unlink()
            json_path.unlink(missing_ok=True)


def delete_intermediate_audio(directories: Sequence[StrPath]) -> None:
    """
    Delete provided directories containing intermediate audio files.

    The provided directories must be located in the root of the
    intermediate audio base directory.

    Parameters
    ----------
    directories : Sequence[StrPath]
        Paths to directories containing intermediate audio files to
        delete.

    """
    delete_audio(
        directory=INTERMEDIATE_AUDIO_BASE_DIR,
        items=directories,
        plural_entity=Entity.DIRECTORIES,
        singular_entity=Entity.DIRECTORY,
        location=Location.INTERMEDIATE_AUDIO_ROOT,
        ui_msg=UIMessage.NO_SONG_DIRS,
    )


def delete_speech_audio(files: Sequence[StrPath]) -> None:
    """
    Delete provided speech audio files.

    The provided files must be located in the root of the speech audio
    directory.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the speech audio files to delete.

    """
    delete_audio(
        directory=SPEECH_DIR,
        items=files,
        plural_entity=Entity.FILES,
        singular_entity=Entity.FILE,
        location=Location.SPEECH_AUDIO_ROOT,
        ui_msg=UIMessage.NO_SPEECH_AUDIO_FILES,
    )


def delete_output_audio(files: Sequence[StrPath]) -> None:
    """
    Delete provided output audio files.

    The provided files must be located in the root of the output audio
    directory.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the output audio files to delete.

    """
    delete_audio(
        directory=OUTPUT_AUDIO_DIR,
        items=files,
        plural_entity=Entity.FILES,
        singular_entity=Entity.FILE,
        location=Location.OUTPUT_AUDIO_ROOT,
        ui_msg=UIMessage.NO_OUTPUT_AUDIO_FILES,
    )


def delete_dataset_audio(datasets: Sequence[StrPath]) -> None:
    """
    Delete provided datasets containing audio files.

    The provided datasets must be located in the root of the training
    audio directory.

    Parameters
    ----------
    datasets : Sequence[StrPath]
        Paths to the datasets to delete.

    """
    delete_audio(
        directory=TRAINING_AUDIO_DIR,
        items=datasets,
        plural_entity=Entity.DIRECTORIES,
        singular_entity=Entity.DIRECTORY,
        location=Location.TRAINING_AUDIO_ROOT,
        ui_msg=UIMessage.NO_DATASETS,
    )


def delete_all_intermediate_audio() -> None:
    """Delete all intermediate audio files."""
    delete_directory(INTERMEDIATE_AUDIO_BASE_DIR)


def delete_all_speech_audio() -> None:
    """Delete all speech audio files."""
    delete_directory(SPEECH_DIR)


def delete_all_output_audio() -> None:
    """Delete all output audio files."""
    delete_directory(OUTPUT_AUDIO_DIR)


def delete_all_dataset_audio() -> None:
    """Delete all dataset audio files."""
    delete_directory(TRAINING_AUDIO_DIR)


def delete_all_audio() -> None:
    """Delete all audio files."""
    delete_directory(AUDIO_DIR)
