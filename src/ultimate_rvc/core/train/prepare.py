"""
Module which exposes functionality for creating and preprocessing
datasets for training voice conversion models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import shutil
from multiprocessing import cpu_count

from ultimate_rvc.common import TRAINING_MODELS_DIR, lazy_import
from ultimate_rvc.core.common import (
    TRAINING_AUDIO_DIR,
    validate_audio_dir_exists,
    validate_audio_file_exists,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    InvalidAudioFormatError,
    NotProvidedError,
    UIMessage,
)
from ultimate_rvc.typing_extra import AudioExt, AudioSplitMethod, TrainingSampleRate

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import static_ffmpeg

    from ultimate_rvc.typing_extra import StrPath
else:
    static_ffmpeg = lazy_import("static_ffmpeg")


def populate_dataset(name: str, audio_files: Sequence[StrPath]) -> Path:
    """
    Populate the dataset with the provided name with the provided audio
    files.

    If no dataset with the provided name exists, a new dataset with the
    provided name will be created. If any of audio files already exist
    in the dataset, they will be overwritten.

    Parameters
    ----------
    name : str
        The name of the dataset to populate.
    audio_files : list[StrPath]
        The audio files to populate the dataset with.

    Returns
    -------
    The path to the dataset with the provided name.

    Raises
    ------
    NotProvidedError
        If no dataset name or no audio files are provided.

    InvalidAudioFormatError
        If any of the provided audio files are not in a valid format.

    """
    if not name:
        raise NotProvidedError(Entity.DATASET_NAME)

    if not audio_files:
        raise NotProvidedError(Entity.FILES, ui_msg=UIMessage.NO_UPLOADED_FILES)

    static_ffmpeg.add_paths()

    import pydub.utils as pydub_utils  # noqa: PLC0415

    audio_paths: list[Path] = []
    for audio_file in audio_files:
        audio_path = validate_audio_file_exists(audio_file, Entity.FILE)
        audio_info = pydub_utils.mediainfo(str(audio_file))
        if not (
            audio_info["format_name"]
            in {
                AudioExt.WAV,
                AudioExt.FLAC,
                AudioExt.MP3,
                AudioExt.OGG,
                AudioExt.AAC,
            }
            or AudioExt.M4A in audio_info["format_name"]
        ):
            raise InvalidAudioFormatError(audio_path, [e.value for e in AudioExt])
        audio_paths.append(audio_path)

    dataset_path = TRAINING_AUDIO_DIR / name.strip()

    dataset_path.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_paths:
        shutil.copyfile(audio_path, dataset_path / audio_path.name)

    return dataset_path


def preprocess_dataset(
    model_name: str,
    dataset: StrPath,
    sample_rate: TrainingSampleRate = TrainingSampleRate.HZ_40K,
    split_method: AudioSplitMethod = AudioSplitMethod.AUTOMATIC,
    chunk_len: float = 3.0,
    overlap_len: float = 0.3,
    filter_audio: bool = True,
    clean_audio: bool = False,
    clean_strength: float = 0.7,
    cpu_cores: int = cpu_count(),
) -> None:
    """
    Preprocess a dataset of audio files for training a voice model.

    Parameters
    ----------
    model_name : str
        The name of the voice model to train. If no voice model
        with the provided name exists for training, a new voice model
        for training will be created with the provided name. If a voice
        model with the provided name already exists for training, then
        its currently associated dataset will be replaced with the
        provided dataset.
    dataset : StrPath
        The path to the dataset to preprocess.
    sample_rate : TrainingSampleRate, default=TrainingSampleRate.HZ_40K
        The target sample rate for the audio files in the provided
        dataset.
    split_method : AudioSplitMethod, default=AudioSplitMethod.AUTOMATIC
        The method to use for splitting the audio files in the provided
        dataset. Use the `Skip` method to skip splitting if the audio
        files are already split. Use the `Simple` method if excessive
        silence has already been removed from the audio files.
        Use the `Automatic` method for automatic silence detection and
        splitting around it.
    chunk_len: float, default=3.0
        length of split audio chunks when using the `Simple` split
        method.
    overlap_len: float, default=0.3
        length of overlap between split audio chunks when using the
        `Simple` split method.
    filter_audio : bool, default=True
        Whether to remove low-frequency sounds from the audio files in
        the provided dataset by applying a high-pass butterworth filter.
    clean_audio : bool, default=False
        Whether to clean the audio files in the provided dataset using
        noise reduction algorithms.
    clean_strength : float, default=0.7
        The intensity of the cleaning to apply to the audio files in the
        provided dataset.
    cpu_cores : int, default=cpu_count()
        The number of CPU cores to use for preprocessing.


    Raises
    ------
    NotProvidedError
        If no model name or dataset is provided.

    """
    if not model_name:
        raise NotProvidedError(Entity.MODEL_NAME)

    dataset_path = validate_audio_dir_exists(dataset, Entity.DATASET)

    model_path = TRAINING_MODELS_DIR / model_name.strip()
    model_path.mkdir(parents=True, exist_ok=True)

    # NOTE The lazy_import function does not work with the package below
    # so we import it here manually
    from ultimate_rvc.rvc.train.preprocess import (  # noqa: PLC0415
        preprocess as train_preprocess,
    )

    train_preprocess.preprocess_training_set(
        str(dataset_path),
        int(sample_rate),
        cpu_cores,
        str(model_path),
        split_method,
        filter_audio,
        clean_audio,
        clean_strength,
        chunk_len,
        overlap_len,
    )
