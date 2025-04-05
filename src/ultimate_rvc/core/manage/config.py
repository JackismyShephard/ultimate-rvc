"""Module which defines functions to manage configuration files."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import shutil
from pathlib import Path

from ultimate_rvc.common import CONFIG_DIR
from ultimate_rvc.core.common import json_dump, json_load
from ultimate_rvc.core.exceptions import (
    ConfigEntity,
    ConfigExistsError,
    ConfigNotFoundError,
    Entity,
    NotProvidedError,
    UIMessage,
)
from ultimate_rvc.core.manage.common import delete_directory, get_named_items
from ultimate_rvc.typing_extra import (
    AudioExt,
    AudioSplitMethod,
    Config,
    DeviceType,
    EmbedderModel,
    F0Method,
    IndexAlgorithm,
    PretrainedType,
    SampleRate,
    SongGenerationConfig,
    SpeechGenerationConfig,
    StrPath,
    TrainingConfig,
    TrainingF0Method,
    TrainingSampleRate,
    Vocoder,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T", bound=Config)

SONG_GENERATION_CONFIG_DIR = CONFIG_DIR / "song"
SPEECH_GENERATION_CONFIG_DIR = CONFIG_DIR / "speech"
TRAINING_CONFIG_DIR = CONFIG_DIR / "training"


def get_named_song_generation_configs() -> list[tuple[str, str]]:
    """
    Get the names and paths of all song generation configuration files.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the names and paths of all song
        generation configuration files.


    """
    return get_named_items(SONG_GENERATION_CONFIG_DIR, include_suffix=False)


def get_named_speech_generation_configs() -> list[tuple[str, str]]:
    """
    Get the names and paths of all speech generation configuration
    files.


    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the names and paths of all speech
        generation configuration files.

    """
    return get_named_items(SPEECH_GENERATION_CONFIG_DIR, include_suffix=False)


def get_named_training_configs() -> list[tuple[str, str]]:
    """
    Get the names and paths of all training configuration files.

    Returns
    -------
    list[tuple[str, str]]
        A list of tuples containing the names and paths of all training
        configuration files.

    """
    return get_named_items(TRAINING_CONFIG_DIR, include_suffix=False)


def load_config(file: StrPath, model_class: type[T]) -> T:  # noqa: UP047
    """
    Load a configuration from a JSON file into a Pydantic model.
    If the file does not exist, create a default instance of the model.

    Parameters
    ----------
    file : StrPath
        The path to the JSON file containing the configuration.
    model_class : type[T]
        The Pydantic model class to load the configuration into.


    Returns
    -------
    T
        An instance of the Pydantic model with the loaded configuration.
        If the file does not exist, a default instance of the model is
        returned.

    """
    file_path = Path(file)
    if file_path.is_file():
        data = json_load(file_path)
        model = model_class.model_validate(data)
    else:
        model = model_class()

    return model


def save_config(name: str, entity: ConfigEntity, model: Config) -> None:
    """
    Save a configuration to a JSON file.

    Parameters
    ----------
    name : str
        The name of the configuration to save.
    entity : ConfigEntity
        The entity type of the configuration to save.
    model : Config
        The Pydantic model containing the configuration to save.

    Raises
    ------
    NotProvidedError
        If no name is provided for the configuration to save.
    ConfigExistsError
        If a configuration with the provided name already exists.

    """
    match entity:
        case Entity.SONG_GENERATION_CONFIG:
            config_dir = SONG_GENERATION_CONFIG_DIR
        case Entity.SPEECH_GENERATION_CONFIG:
            config_dir = SPEECH_GENERATION_CONFIG_DIR
        case Entity.TRAINING_CONFIG:
            config_dir = TRAINING_CONFIG_DIR
        case _:
            config_dir = CONFIG_DIR

    if not name:
        raise NotProvidedError(entity=Entity.CONFIG_NAME)

    config_dir.mkdir(parents=True, exist_ok=True)
    name = name.strip()
    config_path = config_dir / f"{name}.json"
    if config_path.is_file():
        raise ConfigExistsError(entity=entity, name=name)
    json_dump(model.model_dump(), config_path)


def save_song_generation_config(
    name: str,
    n_octaves: int,
    n_semitones: int,
    f0_methods: list[F0Method],
    index_rate: float,
    rms_mix_rate: float,
    protect_rate: float,
    hop_length: int,
    autotune_audio: bool,
    autotune_strength: float,
    embedder_model: EmbedderModel,
    sid: int,
    output_sr: SampleRate,
    output_format: AudioExt,
    split_audio: bool,
    clean_audio: bool,
    clean_strength: float,
    room_size: float,
    wet_level: float,
    dry_level: float,
    damping: float,
    main_gain: int,
    inst_gain: int,
    backup_gain: int,
) -> None:
    """Save a song generation configuration to a JSON file."""
    model = SongGenerationConfig(
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_methods=f0_methods,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        autotune_audio=autotune_audio,
        autotune_strength=autotune_strength,
        embedder_model=embedder_model,
        sid=sid,
        output_sr=output_sr,
        output_format=output_format,
        split_audio=split_audio,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        room_size=room_size,
        wet_level=wet_level,
        dry_level=dry_level,
        damping=damping,
        main_gain=main_gain,
        inst_gain=inst_gain,
        backup_gain=backup_gain,
    )
    save_config(name, Entity.SONG_GENERATION_CONFIG, model)


def save_speech_generation_config(
    name: str,
    n_octaves: int,
    n_semitones: int,
    f0_methods: list[F0Method],
    index_rate: float,
    rms_mix_rate: float,
    protect_rate: float,
    hop_length: int,
    autotune_audio: bool,
    autotune_strength: float,
    embedder_model: EmbedderModel,
    sid: int,
    output_sr: SampleRate,
    output_format: AudioExt,
    tts_pitch_shift: int,
    tts_speed_change: int,
    tts_volume_change: int,
    split_audio: bool,
    clean_audio: bool,
    clean_strength: float,
    output_gain: int,
) -> None:
    """Save a speech generation configuration to a JSON file."""
    model = SpeechGenerationConfig(
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_methods=f0_methods,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        autotune_audio=autotune_audio,
        autotune_strength=autotune_strength,
        embedder_model=embedder_model,
        sid=sid,
        output_sr=output_sr,
        output_format=output_format,
        tts_pitch_shift=tts_pitch_shift,
        tts_speed_change=tts_speed_change,
        tts_volume_change=tts_volume_change,
        split_audio=split_audio,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        output_gain=output_gain,
    )
    save_config(name, Entity.SPEECH_GENERATION_CONFIG, model)


def save_training_config(
    name: str,
    sample_rate: TrainingSampleRate,
    filter_audio: bool,
    clean_audio: bool,
    clean_strength: float,
    split_method: AudioSplitMethod,
    chunk_len: float,
    overlap_len: float,
    f0_method: TrainingF0Method,
    hop_length: int,
    embedder_model: EmbedderModel,
    include_mutes: int,
    extraction_acceleration: DeviceType,
    num_epochs: int,
    batch_size: int,
    detect_overtraining: bool,
    overtraining_threshold: int,
    vocoder: Vocoder,
    index_algorithm: IndexAlgorithm,
    pretrained_type: PretrainedType,
    save_interval: int,
    save_all_checkpoints: bool,
    save_all_weights: bool,
    clear_saved_data: bool,
    upload_model: bool,
    training_acceleration: DeviceType,
    preload_dataset: bool,
    reduce_memory_usage: bool,
) -> None:
    """Save a training configuration to a JSON file."""
    model = TrainingConfig(
        sample_rate=sample_rate,
        filter_audio=filter_audio,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        split_method=split_method,
        chunk_len=chunk_len,
        overlap_len=overlap_len,
        f0_method=f0_method,
        hop_length=hop_length,
        embedder_model=embedder_model,
        include_mutes=include_mutes,
        extraction_acceleration=extraction_acceleration,
        num_epochs=num_epochs,
        batch_size=batch_size,
        detect_overtraining=detect_overtraining,
        overtraining_threshold=overtraining_threshold,
        vocoder=vocoder,
        index_algorithm=index_algorithm,
        pretrained_type=pretrained_type,
        save_interval=save_interval,
        save_all_checkpoints=save_all_checkpoints,
        save_all_weights=save_all_weights,
        clear_saved_data=clear_saved_data,
        upload_model=upload_model,
        training_acceleration=training_acceleration,
        preload_dataset=preload_dataset,
        reduce_memory_usage=reduce_memory_usage,
    )
    save_config(name, Entity.TRAINING_CONFIG, model)


def delete_configs(
    directory: StrPath,
    names: Sequence[str],
    entity: ConfigEntity = Entity.CONFIG,
    ui_msg: UIMessage = UIMessage.NO_MODELS,
) -> None:
    """
    Delete the configurations with the provided names.

    Parameters
    ----------
    directory : StrPath
        The path to the directory containing the configurations to
        delete.
    names : Sequence[str]
        Names of the configurations to delete.
    entity : ModelEntity, optional
        The configuration entity beng deleted.
    ui_msg : UIMessage, optional
        The message to display if no configuration names are provided.

    Raises
    ------
    NotProvidedError
        If no names of items are provided.
    ConfigNotFoundError
        If a configuration with a provided name does not exist.

    """
    if not names:
        raise NotProvidedError(entity=Entity.CONFIG_NAMES, ui_msg=ui_msg)
    config_file_paths: list[Path] = []
    for name in names:
        config_file_path = Path(directory) / name
        if not config_file_path.is_file():
            raise ConfigNotFoundError(entity=entity, name=name)
        config_file_paths.append(config_file_path)
    for config_file_path in config_file_paths:
        shutil.rmtree(config_file_path)


def delete_all_song_generation_configs() -> None:
    """Delete all song generation configuration files."""
    delete_directory(SONG_GENERATION_CONFIG_DIR)


def delete_all_speech_generation_configs() -> None:
    """Delete all speech generation configuration files."""
    delete_directory(SPEECH_GENERATION_CONFIG_DIR)


def delete_all_training_configs() -> None:
    """Delete all training configuration files."""
    delete_directory(TRAINING_CONFIG_DIR)


def delete_all_configs() -> None:
    """Delete all configuration files."""
    delete_directory(CONFIG_DIR)
