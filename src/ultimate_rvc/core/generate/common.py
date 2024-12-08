"""
Common definitions for modules in the Ultimate RVC project that
generate audio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import logging
from functools import cache
from itertools import starmap
from pathlib import Path

from rich import print as rprint

from ultimate_rvc.common import VOICE_MODELS_DIR, lazy_import
from ultimate_rvc.core.common import (
    display_progress,
    get_file_hash,
    get_hash,
    json_dump,
    json_dumps,
    json_load,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    NotFoundError,
    NotProvidedError,
    UIMessage,
    VoiceModelNotFoundError,
)
from ultimate_rvc.core.generate.typing_extra import (
    DirectoryMetaData,
    FileMetaData,
    RVCAudioMetaData,
    WaveifiedAudioMetaData,
)
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gradio as gr

    import ffmpeg
    import static_ffmpeg
    import static_sox

    # NOTE the only reason this module is imported here is so we can
    # annotate the return value of the _get_voice_converter function.
    from ultimate_rvc.rvc.infer.infer import VoiceConverter
    from ultimate_rvc.typing_extra import Json, StrPath
else:
    ffmpeg = lazy_import("ffmpeg")
    static_ffmpeg = lazy_import("static_ffmpeg")
    static_sox = lazy_import("static_sox")

logger = logging.getLogger(__name__)


def validate_exists(
    identifier: StrPath | None,
    entity: Entity,
) -> Path:
    """
    Validate that the provided identifier is not none and that it
    identifies an existing entity, which can be either a voice model,
    a song directory or an audio track.

    Parameters
    ----------
    identifier : StrPath | None
        The identifier to validate.
    entity : Entity
        The entity that the identifier should identify.

    Returns
    -------
    Path
        The path to the identified entity.

    Raises
    ------
    NotProvidedError
        If the identifier is None.
    NotFoundError
        If the identifier does not identify an existing entity.
    VoiceModelNotFoundError
        If the identifier does not identify an existing voice model.
    NotImplementedError
        If the provided entity is not supported.

    """
    match entity:
        case Entity.MODEL_NAME:
            if not identifier:
                raise NotProvidedError(entity=entity, ui_msg=UIMessage.NO_VOICE_MODEL)
            path = VOICE_MODELS_DIR / identifier
            if not path.is_dir():
                raise VoiceModelNotFoundError(str(identifier))
        case Entity.SONG_DIR | Entity.EMBEDDER_MODEL_CUSTOM:
            ui_msg = UIMessage.NO_SONG_DIR if entity == Entity.SONG_DIR else None
            if not identifier:
                raise NotProvidedError(entity=entity, ui_msg=ui_msg)
            path = Path(identifier)
            if not path.is_dir():
                raise NotFoundError(entity=entity, location=path)
        case (
            Entity.SONG
            | Entity.AUDIO_TRACK
            | Entity.VOCALS_TRACK
            | Entity.INSTRUMENTALS_TRACK
            | Entity.MAIN_VOCALS_TRACK
            | Entity.BACKUP_VOCALS_TRACK
        ):
            if not identifier:
                raise NotProvidedError(entity=entity)
            path = Path(identifier)
            if not path.is_file():
                raise NotFoundError(entity=entity, location=path)
        case _:
            error_msg = f"Entity {entity} not supported."
            raise NotImplementedError(error_msg)
    return path


def validate_all_exist(
    identifier_entity_pairs: Sequence[tuple[StrPath | None, Entity]],
) -> list[Path]:
    """
    Validate that all provided identifiers are not none and that they
    identify existing entities, which can be either voice models, song
    directories or audio tracks.

    Parameters
    ----------
    identifier_entity_pairs : Sequence[tuple[StrPath, Entity]]
        The pairs of identifiers and entities to validate.

    Returns
    -------
    list[Path]
        The paths to the identified entities.

    """
    return list(starmap(validate_exists, identifier_entity_pairs))


# NOTE consider increasing hash_size to 16. Otherwise
# we might have problems with hash collisions when using app as CLI
def get_unique_base_path(
    directory: StrPath,
    prefix: str,
    args_dict: Json,
    hash_size: int = 5,
) -> Path:
    """
    Get a unique base path (a path without any extension) for a file in
    a directory by hashing the arguments used to generate
    the audio that is stored or will be stored in that file.

    Parameters
    ----------
    directory :StrPath
        The path to a directory.
    prefix : str
        The prefix to use for the base path.
    args_dict : Json
        A JSON-serializable dictionary of named arguments used to
        generate the audio that is stored or will be stored in a file
        in the given directory.
    hash_size : int, default=5
        The size (in bytes) of the hash to use for the base path.

    Returns
    -------
    Path
        The unique base path for a file in a song directory.

    """
    directory_path = Path(directory)
    dict_hash = get_hash(args_dict, size=hash_size)
    while True:
        base_path = directory_path / f"{prefix}_{dict_hash}"
        json_path = base_path.with_suffix(".json")
        if json_path.exists():
            file_dict = json_load(json_path)
            if file_dict == args_dict:
                return base_path
            dict_hash = get_hash(dict_hash, size=hash_size)
            rprint("[~] Rehashing...")
        else:
            return base_path


def to_wav(
    audio_track: StrPath,
    directory: StrPath,
    prefix: str,
    accepted_formats: set[AudioExt] | None = None,
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Convert a given audio track to wav format if its current format is
    one of the given accepted formats.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to convert.
    directory : StrPath
        The path to the directory where the converted audio track
        will be saved.
    prefix : str
        The prefix to use for the name of the converted audio track.
    accepted_formats : set[AudioExt], optional
        The audio formats to accept for conversion. If None, the
        accepted formats are mp3, ogg, flac, m4a and aac.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the audio track in wav format or the original audio
        track if it is not in one of the accepted formats.

    """
    static_ffmpeg.add_paths()
    # NOTE have to import pydub.utils here because
    # 1. when using gradio pydub.utils is not available if only pydub
    #    itself is imported.
    # 2. when using cli static_ffmpeg must be imported before
    #    pydub.utils is imported.
    import pydub.utils as pydub_utils  # noqa: PLC0415

    if accepted_formats is None:
        accepted_formats = set(AudioExt) - {AudioExt.WAV}

    audio_path, song_dir_path = validate_all_exist(
        [(audio_track, Entity.AUDIO_TRACK), (directory, Entity.DIRECTORY)],
    )

    wav_path = audio_path

    audio_info = pydub_utils.mediainfo(str(audio_path))
    logger.info("Audio info:\n%s", json_dumps(audio_info))
    if any(
        (
            audio_info["format_name"]
            if accepted_format == AudioExt.M4A
            else accepted_format == audio_info["format_name"]
        )
        for accepted_format in accepted_formats
    ):
        args_dict = WaveifiedAudioMetaData(
            audio_track=FileMetaData(
                name=audio_path.name,
                hash_id=get_file_hash(audio_path),
            ),
        ).model_dump()

        paths = [
            get_unique_base_path(
                song_dir_path,
                prefix,
                args_dict,
            ).with_suffix(suffix)
            for suffix in [".wav", ".json"]
        ]
        wav_path, wav_json_path = paths
        if not all(path.exists() for path in paths):
            display_progress(
                "[~] Converting audio track to wav format...",
                percentage,
                progress_bar,
            )

            _, stderr = (
                ffmpeg.input(audio_path)
                .output(filename=wav_path, f="wav")
                .run(
                    overwrite_output=True,
                    quiet=True,
                )
            )
            logger.info("FFmpeg stderr:\n%s", stderr.decode("utf-8"))
            json_dump(args_dict, wav_json_path)

    return wav_path


def _get_rvc_files(model_name: str) -> tuple[Path, Path | None]:
    """
    Get the RVC model file and potential index file of a voice model.

    Parameters
    ----------
    model_name : str
        The name of the voice model to get the RVC files of.

    Returns
    -------
    model_file : Path
        The path to the RVC model file.
    index_file : Path | None
        The path to the RVC index file, if it exists.

    Raises
    ------
    NotFoundError
        If no model file exists in the voice model directory.


    """
    model_dir_path = validate_exists(model_name, Entity.MODEL_NAME)
    file_path_map = {
        ext: path
        for path in model_dir_path.iterdir()
        for ext in [".pth", ".index"]
        if ext == path.suffix
    }

    if ".pth" not in file_path_map:
        raise NotFoundError(
            entity=Entity.MODEL_FILE,
            location=model_dir_path,
            is_path=False,
        )

    model_file = file_path_map[".pth"]
    index_file = file_path_map.get(".index")
    return model_file, index_file


@cache
def _get_voice_converter() -> VoiceConverter:
    """
    Get a voice converter.

    Returns
    -------
    VoiceConverter
        A voice converter.

    """
    static_sox.add_paths()
    from ultimate_rvc.rvc.infer.infer import VoiceConverter  # noqa: PLC0415

    return VoiceConverter()


def convert(
    audio_track: StrPath,
    directory: StrPath,
    model_name: str,
    n_octaves: int = 0,
    n_semitones: int = 0,
    f0_methods: Sequence[F0Method] | None = None,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect_rate: float = 0.33,
    hop_length: int = 128,
    split_audio: bool = False,
    autotune_audio: bool = False,
    autotune_strength: float = 1.0,
    clean_audio: bool = False,
    clean_strength: float = 0.7,
    embedder_model: EmbedderModel = EmbedderModel.CONTENTVEC,
    embedder_model_custom: StrPath | None = None,
    sid: int = 0,
    entity: Literal["vocals", "audio"] = "audio",
    progress_bar: gr.Progress | None = None,
    percentage: float = 0.5,
) -> Path:
    """
    Convert an audio track using an RVC model.

    Parameters
    ----------
    audio_track : StrPath
        The path to the audio track to convert.
    directory : StrPath
        The path to the directory where the converted audio track
        will be saved.
    model_name : str
        The name of the model to use for voice conversion.
    n_octaves : int, default=0
        The number of octaves to pitch-shift the converted audio by.
    n_semitones : int, default=0
        The number of semitones to pitch-shift the converted audio by.
    f0_methods : Sequence[F0Method], optional
        The methods to use for pitch extraction. If None, the method
        used is rmvpe.
    index_rate : float, default=0.5
        The influence of the index file on the voice conversion.
    filter_radius : int, default=3
        The filter radius to use for the voice conversion.
    rms_mix_rate : float, default=0.25
        The blending rate of the volume envelope of the converted
        audio.
    protect_rate : float, default=0.33
        The protection rate for consonants and breathing sounds.
    hop_length : int, default=128
        The hop length to use for CREPE-based pitch extraction.
    split_audio : bool, default=False
        Whether to split the audio track into smaller segments before
        converting it.
    autotune_audio : bool, default=False
        Whether to apply autotune to the converted audio.
    autotune_strength : float, default=1.0
        The strength of the autotune to apply to the converted audio.
    clean_voice : bool, default=False
        Whether to clean the converted audio.
    clean_strength : float, default=0.7
        The intensity of the cleaning to apply to the converted audio.
    embedder_model : EmbedderModel, default=EmbedderModel.CONTENTVEC
        The model to use for generating speaker embeddings.
    embedder_model_custom : StrPath, optional
        The path to a directory with a custom model to use for
        generating speaker embeddings.
    sid : int, default=0
        The speaker id to use for multi-speaker models.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the converted audio track.

    """
    audio_path, directory_path, _ = validate_all_exist(
        [
            (
                audio_track,
                Entity.VOCALS_TRACK if entity == "vocals" else Entity.AUDIO_TRACK,
            ),
            (directory, Entity.SONG_DIR if entity == "vocals" else Entity.DIRECTORY),
            (model_name, Entity.MODEL_NAME),
        ],
    )
    if embedder_model == EmbedderModel.CUSTOM:
        embedder_model_custom = validate_exists(
            embedder_model_custom,
            Entity.EMBEDDER_MODEL_CUSTOM,
        )

    audio_path = to_wav(
        audio_path,
        directory_path,
        "20_Input",
        accepted_formats={AudioExt.M4A, AudioExt.AAC},
        progress_bar=progress_bar,
        percentage=percentage,
    )

    n_semitones = n_octaves * 12 + n_semitones
    f0_methods_set = set(f0_methods) if f0_methods else {F0Method.RMVPE}

    args_dict = RVCAudioMetaData(
        audio_track=FileMetaData(
            name=audio_path.name,
            hash_id=get_file_hash(audio_path),
        ),
        model_name=model_name,
        n_semitones=n_semitones,
        f0_methods=sorted(f0_methods_set),
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        split_audio=split_audio,
        autotune_audio=autotune_audio,
        autotune_strength=autotune_strength,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        embedder_model=embedder_model,
        embedder_model_custom=(
            DirectoryMetaData(
                name=Path(embedder_model_custom).name,
                path=str(embedder_model_custom),
            )
            if embedder_model_custom is not None
            else None
        ),
        sid=sid,
    ).model_dump()

    paths = [
        get_unique_base_path(
            directory_path,
            f"21_{entity.capitalize}_Converted",
            args_dict,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    converted_audio_path, converted_audio_json_path = paths

    if not all(path.exists() for path in paths):
        display_progress(
            f"[~] Converting {entity} using RVC...",
            percentage,
            progress_bar,
        )

        rvc_model_path, rvc_index_path = _get_rvc_files(model_name)

        voice_converter = _get_voice_converter()

        voice_converter.convert_audio(
            audio_input_path=str(audio_path),
            audio_output_path=str(converted_audio_path),
            model_path=str(rvc_model_path),
            index_path=str(rvc_index_path) if rvc_index_path else "",
            pitch=n_semitones,
            f0_methods=f0_methods_set,
            index_rate=index_rate,
            volume_envelope=rms_mix_rate,
            protect=protect_rate,
            hop_length=hop_length,
            split_audio=split_audio,
            f0_autotune=autotune_audio,
            f0_autotune_strength=autotune_strength,
            filter_radius=filter_radius,
            embedder_model=embedder_model,
            embedder_model_custom=(
                str(embedder_model_custom)
                if embedder_model_custom is not None
                else None
            ),
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            post_process=False,
            resample_sr=0,
            sid=sid,
        )
        json_dump(args_dict, converted_audio_json_path)
    return converted_audio_path
