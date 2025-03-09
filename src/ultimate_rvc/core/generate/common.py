"""
Common definitions for modules in the Ultimate RVC project that
generate audio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import logging
from functools import cache, reduce
from pathlib import Path

from rich import print as rprint

from ultimate_rvc.common import lazy_import
from ultimate_rvc.core.common import (
    get_file_hash,
    get_hash,
    json_dump,
    json_dumps,
    json_load,
    validate_audio_dir_exists,
    validate_audio_file_exists,
    validate_model_exists,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    NotFoundError,
    NotProvidedError,
    UIMessage,
)
from ultimate_rvc.core.generate.typing_extra import (
    AudioExtInternal,
    FileMetaData,
    MixedAudioMetaData,
    MixedAudioType,
    RVCAudioMetaData,
    StagedAudioMetaData,
    WaveifiedAudioMetaData,
)
from ultimate_rvc.typing_extra import (
    AudioExt,
    EmbedderModel,
    F0Method,
    RVCContentType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import ffmpeg
    import static_ffmpeg

    # NOTE the only reason this module is imported here is so we can
    # annotate the return value of the _get_voice_converter function.
    from ultimate_rvc.rvc.infer.infer import VoiceConverter
    from ultimate_rvc.typing_extra import Json, StrPath
else:
    # NOTE yt_dlp is not used here but importing it fixes erroneous user
    # warning
    yt_dlp = lazy_import("yt_dlp")
    ffmpeg = lazy_import("ffmpeg")
    static_ffmpeg = lazy_import("static_ffmpeg")

logger = logging.getLogger(__name__)


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


def wavify(
    audio_track: StrPath,
    directory: StrPath,
    prefix: str,
    accepted_formats: set[AudioExt] | None = None,
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

    audio_path = validate_audio_file_exists(audio_track, Entity.AUDIO_TRACK)
    dir_path = validate_audio_dir_exists(directory, Entity.DIRECTORY)

    wav_path = audio_path

    if accepted_formats is None:
        accepted_formats = set(AudioExt) - {AudioExt.WAV}

    # NOTE The lazy_import function does not work with pydub
    # so we import it here manually
    import pydub.utils as pydub_utils  # noqa: PLC0415

    audio_info = pydub_utils.mediainfo(str(audio_path))
    logger.info("Audio info:\n%s", json_dumps(audio_info))
    if any(
        (
            accepted_format in audio_info["format_name"]
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
                dir_path,
                prefix,
                args_dict,
            ).with_suffix(suffix)
            for suffix in [".wav", ".json"]
        ]
        wav_path, wav_json_path = paths
        if not all(path.exists() for path in paths):

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
    model_dir_path = validate_model_exists(model_name, Entity.VOICE_MODEL)
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
    custom_embedder_model: str | None = None,
    sid: int = 0,
    content_type: RVCContentType = RVCContentType.AUDIO,
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
    clean_audio : bool, default=False
        Whether to clean the converted audio.
    clean_strength : float, default=0.7
        The intensity of the cleaning to apply to the converted audio.
    embedder_model : EmbedderModel, default=EmbedderModel.CONTENTVEC
        The model to use for generating speaker embeddings.
    custom_embedder_model : str, optional
        The name of a custom embedder model to use for generating
        speaker embeddings.
    sid : int, default=0
        The speaker id to use for multi-speaker models.
    content_type : RVCContentType, default=RVCContentType.AUDIO
        The type of content to convert. Determines what is shown in
        display mesages and saved file names.

    Returns
    -------
    Path
        The path to the converted audio track.

    """
    match content_type:
        case RVCContentType.VOCALS:
            track_entity = Entity.VOCALS_TRACK
            directory_entity = Entity.SONG_DIR
        case RVCContentType.VOICE:
            track_entity = Entity.VOICE_TRACK
            directory_entity = Entity.DIRECTORY
        case RVCContentType.SPEECH:
            track_entity = Entity.SPEECH_TRACK
            directory_entity = Entity.DIRECTORY
        case RVCContentType.AUDIO:
            track_entity = Entity.AUDIO_TRACK
            directory_entity = Entity.DIRECTORY
    audio_path = validate_audio_file_exists(audio_track, track_entity)
    directory_path = validate_audio_dir_exists(directory, directory_entity)
    validate_model_exists(model_name, Entity.VOICE_MODEL)
    custom_embedder_model_path = None
    if embedder_model == EmbedderModel.CUSTOM:
        custom_embedder_model_path = validate_model_exists(
            custom_embedder_model,
            Entity.CUSTOM_EMBEDDER_MODEL,
        )

    audio_path = wavify(
        audio_path,
        directory_path,
        "20_Input",
        accepted_formats={AudioExt.M4A, AudioExt.AAC},
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
        custom_embedder_model=custom_embedder_model,
        sid=sid,
    ).model_dump()

    paths = [
        get_unique_base_path(
            directory_path,
            f"21_{content_type.capitalize()}_Converted",
            args_dict,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    converted_audio_path, converted_audio_json_path = paths

    if not all(path.exists() for path in paths):

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
                str(custom_embedder_model_path)
                if custom_embedder_model_path is not None
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


def _to_internal(audio_ext: AudioExt) -> AudioExtInternal:
    """
    Map an audio extension to an internally recognized format.

    Parameters
    ----------
    audio_ext : AudioExt
        The audio extension to map.

    Returns
    -------
    AudioExtInternal
        The internal audio extension.

    """
    match audio_ext:
        case AudioExt.M4A:
            return AudioExtInternal.IPOD
        case AudioExt.AAC:
            return AudioExtInternal.ADTS
        case _:
            return AudioExtInternal(audio_ext)


def _mix_audio(
    audio_track_gain_pairs: Sequence[tuple[StrPath, int]],
    output_file: StrPath,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
) -> None:
    """
    Mix multiple audio tracks.

    Parameters
    ----------
    audio_track_gain_pairs : Sequence[tuple[StrPath, int]]
        A sequence of pairs each containing the path to an audio track
        and the gain to apply to it.
    output_file : StrPath
        The path to the file to save the mixed audio to.
    output_sr : int, default=44100
        The sample rate of the mixed audio.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the mixed audio.

    """
    static_ffmpeg.add_paths()
    # NOTE The lazy_import function does not work with pydub
    # so we import it here manually
    import pydub  # noqa: PLC0415

    mixed_audio = reduce(
        lambda a1, a2: a1.overlay(a2),
        [
            pydub.AudioSegment.from_wav(audio_track) + gain
            for audio_track, gain in audio_track_gain_pairs
        ],
    )
    mixed_audio_resampled = mixed_audio.set_frame_rate(output_sr)
    mixed_audio_resampled.export(
        output_file,
        format=_to_internal(output_format),
    )


def mix_audio(
    audio_track_gain_pairs: Sequence[tuple[StrPath, int]],
    directory: StrPath,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    content_type: MixedAudioType = MixedAudioType.AUDIO,
) -> Path:
    """
    Mix one or more audio tracks.

    Parameters
    ----------
    audio_track_gain_pairs : Sequence[tuple[StrPath, int]]
        A sequence of pairs each containing the path to an audio track
        and the gain to apply to it.
    directory : StrPath
        The path to the directory where the mixed audio will be saved.
    output_sr : int, default=44100
        The sample rate of the mixed audio.
    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the mixed audio.
    content_type: MixedAudioType, default=MixedAudioType.AUDIO
        The type of mixed audio. Determines what is shown in display
        messages and saved file names.
    progress_bar : gr.Progress, optional
        Gradio progress bar to update.
    percentage : float, default=0.5
        Percentage to display in the progress bar.

    Returns
    -------
    Path
        The path to the mixed audio.

    Raises
    ------
    NotProvidedError
        If no audio tracks are provided.

    """
    match content_type:
        case MixedAudioType.AUDIO:
            directory_entity = Entity.DIRECTORY
        case MixedAudioType.SPEECH:
            directory_entity = Entity.DIRECTORY
        case MixedAudioType.SONG:
            directory_entity = Entity.SONG_DIR
    if not audio_track_gain_pairs:
        raise NotProvidedError(
            entity=Entity.AUDIO_TRACK_GAIN_PAIRS,
            ui_msg=UIMessage.NO_AUDIO_TRACK,
        )

    audio_path_gain_pairs = [
        (
            wavify(
                validate_audio_file_exists(audio_track, Entity.AUDIO_TRACK),
                directory,
                "50_Input",
            ),
            gain,
        )
        for audio_track, gain in audio_track_gain_pairs
    ]
    dir_path = validate_audio_dir_exists(directory, directory_entity)
    args_dict = MixedAudioMetaData(
        staged_audio_tracks=[
            StagedAudioMetaData(
                audio_track=FileMetaData(
                    name=audio_path.name,
                    hash_id=get_file_hash(audio_path),
                ),
                gain=gain,
            )
            for audio_path, gain in audio_path_gain_pairs
        ],
        output_sr=output_sr,
        output_format=output_format,
    ).model_dump()

    paths = [
        get_unique_base_path(
            dir_path,
            f"51_{content_type.capitalize()}_Mixed",
            args_dict,
        ).with_suffix(suffix)
        for suffix in ["." + output_format, ".json"]
    ]

    mix_path, mix_json_path = paths

    if not all(path.exists() for path in paths):

        _mix_audio(audio_path_gain_pairs, mix_path, output_sr, output_format)
        json_dump(args_dict, mix_json_path)
    return mix_path
