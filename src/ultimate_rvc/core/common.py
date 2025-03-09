"""Common utility functions for the core of the Ultimate RVC project."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib
import json
import shutil
from pathlib import Path

from pydantic import AnyHttpUrl, TypeAdapter, ValidationError

from rich import print as rprint

from ultimate_rvc.common import (
    AUDIO_DIR,
    CUSTOM_EMBEDDER_MODELS_DIR,
    CUSTOM_PRETRAINED_MODELS_DIR,
    MODELS_DIR,
    TRAINING_MODELS_DIR,
    VOICE_MODELS_DIR,
    lazy_import,
)
from ultimate_rvc.core.exceptions import (
    AudioDirectoryEntity,
    AudioFileEntity,
    Entity,
    HttpUrlError,
    ModelEntity,
    ModelNotFoundError,
    NotFoundError,
    NotProvidedError,
    UIMessage,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import requests

    import gradio as gr

    from ultimate_rvc.typing_extra import Json, StrPath
else:
    requests = lazy_import("requests")


RVC_DOWNLOAD_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
INTERMEDIATE_AUDIO_BASE_DIR = AUDIO_DIR / "intermediate"
SPEECH_DIR = AUDIO_DIR / "speech"
OUTPUT_AUDIO_DIR = AUDIO_DIR / "output"
FLAG_FILE = MODELS_DIR / ".initialized"
TRAINING_AUDIO_DIR = AUDIO_DIR / "training"


def display_progress(
    message: str,
    percentage: float | None = None,
    progress_bar: gr.Progress | None = None,
) -> None:
    """
    Display progress message and percentage in console and potentially
    also Gradio progress bar.

    Parameters
    ----------
    message : str
        Message to display.
    percentage : float, optional
        Percentage to display.
    progress_bar : gr.Progress, optional
        The Gradio progress bar to update.

    """
    rprint(message)
    if progress_bar is not None:
        progress_bar(percentage, desc=message)


def remove_suffix_after(text: str, occurrence: str) -> str:
    """
    Remove suffix after the first occurrence of a substring in a string.

    Parameters
    ----------
    text : str
        The string to remove the suffix from.
    occurrence : str
        The substring to remove the suffix after.

    Returns
    -------
    str
        The string with the suffix removed.

    """
    location = text.rfind(occurrence)
    if location == -1:
        return text
    return text[: location + len(occurrence)]


def copy_files_to_new_dir(files: Sequence[StrPath], directory: StrPath) -> None:
    """
    Copy files to a new directory.

    Parameters
    ----------
    files : Sequence[StrPath]
        Paths to the files to copy.
    directory : StrPath
        Path to the directory to copy the files to.

    Raises
    ------
    NotFoundError
        If a file does not exist.

    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True)
    for file in files:
        file_path = Path(file)
        if not file_path.exists():
            raise NotFoundError(entity=Entity.FILE, location=file_path)
        shutil.copyfile(file_path, dir_path / file_path.name)


def copy_file_safe(src: StrPath, dest: StrPath) -> Path:
    """
    Copy a file to a new location, appending a number if a file with the
    same name already exists.

    Parameters
    ----------
    src : strPath
        The source file path.
    dest : strPath
        The candidate destination file path.

    Returns
    -------
    Path
        The final destination file path.

    """
    dest_path = Path(dest)
    src_path = Path(src)
    dest_dir = dest_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_path
    counter = 1

    while dest_file.exists():
        dest_file = dest_dir / f"{dest_path.stem} ({counter}){src_path.suffix}"
        counter += 1

    shutil.copyfile(src, dest_file)
    return dest_file


def json_dumps(thing: Json) -> str:
    """
    Dump a JSON-serializable object to a JSON string.

    Parameters
    ----------
    thing : Json
        The JSON-serializable object to dump.

    Returns
    -------
    str
        The JSON string representation of the object.

    """
    return json.dumps(thing, ensure_ascii=False, indent=4)


def json_dump(thing: Json, file: StrPath) -> None:
    """
    Dump a JSON-serializable object to a JSON file.

    Parameters
    ----------
    thing : Json
        The JSON-serializable object to dump.
    file : StrPath
        The path to the JSON file.

    """
    with Path(file).open("w", encoding="utf-8") as fp:
        json.dump(thing, fp, ensure_ascii=False, indent=4)


def json_load(file: StrPath, encoding: str = "utf-8") -> Json:
    """
    Load a JSON-serializable object from a JSON file.

    Parameters
    ----------
    file : StrPath
        The path to the JSON file.
    encoding : str, default='utf-8'
        The encoding of the JSON file.

    Returns
    -------
    Json
        The JSON-serializable object loaded from the JSON file.

    """
    with Path(file).open(encoding=encoding) as fp:
        return json.load(fp)


def get_hash(thing: Json, size: int = 5) -> str:
    """
    Get the hash of a JSON-serializable object.

    Parameters
    ----------
    thing : Json
        The JSON-serializable object to hash.
    size : int, default=5
        The size of the hash in bytes.

    Returns
    -------
    str
        The hash of the JSON-serializable object.

    """
    return hashlib.blake2b(
        json_dumps(thing).encode("utf-8"),
        digest_size=size,
    ).hexdigest()


# NOTE consider increasing size to 16 otherwise we might have problems
# with hash collisions
def get_file_hash(file: StrPath, size: int = 5) -> str:
    """
    Get the hash of a file.

    Parameters
    ----------
    file : StrPath
        The path to the file.
    size : int, default=5
        The size of the hash in bytes.

    Returns
    -------
    str
        The hash of the file.

    """
    with Path(file).open("rb") as fp:
        file_hash = hashlib.file_digest(
            fp,
            lambda: hashlib.blake2b(digest_size=size),  # type: ignore[reportArgumentType]
        )
    return file_hash.hexdigest()


def get_combined_file_hash(files: Sequence[StrPath], size: int = 5) -> str:
    """
    Get the combined hash of a sequence of files.

    Parameters
    ----------
    files : Sequence[StrPath]
        A sequence of paths to the files to hash.
    size : int, default=5
        The size of each hash in bytes.

    Returns
    -------
    str
        The combined hash of the files.

    """
    hasher = hashlib.blake2b(digest_size=size)

    for file in files:
        with Path(file).open("rb") as fp:
            for chunk in iter(lambda: fp.read(4096), b""):
                hasher.update(chunk)

    return hasher.hexdigest()


def get_file_size(url: str) -> int:
    """
    Get the size of a file at a given URL.

    Parameters
    ----------
    url : str
        The URL of the file to get the size of.

    Returns
    -------
    int
        The size of the file at the given URL.

    """
    response = requests.head(url)
    response.raise_for_status()
    return int(response.headers.get("content-length", 0))


def validate_audio_file_exists(
    audio_file: StrPath | None,
    entity: AudioFileEntity,
) -> Path:
    """
    Validate that the provided audio file path is defined and that it
    points to an existing instance of the provided audio file entity.

    Parameters
    ----------
    audio_file : StrPath, optional
        The audio file path to validate.
    entity : AudioFileEntity
        The entity that the audio file path should point to.

    Returns
    -------
    Path
        The validated audio file path.

    Raises
    ------
    NotProvidedError
        If the audio file path is not defined.
    NotFoundError
        If the audio file path does not point to an existing instance
        of the provided audio file entity.

    """
    if not audio_file:
        raise NotProvidedError(entity=entity)
    audio_file_path = Path(audio_file)
    if not audio_file_path.is_file():
        raise NotFoundError(entity=entity, location=audio_file_path)
    return audio_file_path


def validate_audio_dir_exists(
    audio_directory: StrPath | None,
    entity: AudioDirectoryEntity,
) -> Path:
    """
    Validate that the provided audio directory path is defined and that
    it points to an existing instance of the provided audio directory
    entity.

    Parameters
    ----------
    audio_directory : StrPath, optional
        The audio directory path to validate.
    entity : AudioDirectoryEntity
        The entity that the audio directory path should point to.

    Returns
    -------
    Path
        The validated audio directory path.

    Raises
    ------
    NotProvidedError
        If the audio directory path is not defined.
    NotFoundError
        If the audio directory path does not point to an existing
        instance of the provided audio directory entity.

    """
    match entity:
        case Entity.SONG_DIR:
            ui_msg = UIMessage.NO_SONG_DIR
        case _:
            ui_msg = None
    if not audio_directory:
        raise NotProvidedError(entity=entity, ui_msg=ui_msg)
    audio_dir_path = Path(audio_directory)
    if not audio_dir_path.is_dir():
        raise NotFoundError(entity=entity, location=audio_dir_path)
    return audio_dir_path


def validate_model_exists(name: str | None, entity: ModelEntity) -> Path:
    """
    Validate that the provided name is defined and that it identifies
    an existing instance of the provided model entity.

    Parameters
    ----------
    name : str | None
        The name of the model to validate.
    entity : Entity
        The entity that the name should identify.

    Returns
    -------
    Path
        The path to the model entity identified by the provided name.

    Raises
    ------
    NotProvidedError
        If the model name is not defined.
    ModelNotFoundError
        If the model name does not point to an existing instance of the
        provided model entity.

    """
    match entity:
        case Entity.VOICE_MODEL:
            ui_msg = UIMessage.NO_VOICE_MODEL
            directory = VOICE_MODELS_DIR
        case Entity.CUSTOM_EMBEDDER_MODEL:
            ui_msg = UIMessage.NO_CUSTOM_EMBEDDER_MODEL
            directory = CUSTOM_EMBEDDER_MODELS_DIR
        case Entity.TRAINING_MODEL:
            ui_msg = None
            directory = TRAINING_MODELS_DIR
        case Entity.CUSTOM_PRETRAINED_MODEL:
            ui_msg = UIMessage.NO_CUSTOM_PRETRAINED_MODEL
            directory = CUSTOM_PRETRAINED_MODELS_DIR
        case Entity.MODEL:
            ui_msg = UIMessage.NO_MODEL
            directory = MODELS_DIR

    directory_path = Path(directory)
    if not name:
        raise NotProvidedError(entity=Entity.MODEL_NAME, ui_msg=ui_msg)
    model_path = directory_path / name
    if not model_path.is_dir():
        raise ModelNotFoundError(entity, name=name)

    return model_path


def validate_url(url: str) -> None:
    """
    Validate a HTTP-based URL.

    Parameters
    ----------
    url : str
        The URL to validate.

    Raises
    ------
    HttpUrlError
        If the URL is invalid.

    """
    try:
        TypeAdapter(AnyHttpUrl).validate_python(url)
    except ValidationError:
        raise HttpUrlError(url) from None
