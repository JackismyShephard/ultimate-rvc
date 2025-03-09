"""
Module which defines functions and other definitions that facilitate
RVC-based TTS generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import anyio

from pydantic import ValidationError

from ultimate_rvc.common import lazy_import
from ultimate_rvc.core.common import (
    OUTPUT_AUDIO_DIR,
    SPEECH_DIR,
    copy_file_safe,
    display_progress,
    get_file_hash,
    json_dump,
    json_load,
)
from ultimate_rvc.core.exceptions import Entity, NotProvidedError, UIMessage
from ultimate_rvc.core.generate.common import (
    convert,
    get_unique_base_path,
    mix_audio,
    validate_model_exists,
)
from ultimate_rvc.core.generate.typing_extra import (
    EdgeTTSAudioMetaData,
    EdgeTTSKeys,
    EdgeTTSVoiceKey,
    EdgeTTSVoiceTable,
    EdgeTTSVoiceTagKey,
    FileMetaData,
    MixedAudioType,
    RVCAudioMetaData,
)
from ultimate_rvc.typing_extra import (
    AudioExt,
    EmbedderModel,
    F0Method,
    RVCContentType,
    StrPath,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gradio as gr

    import edge_tts

else:
    edge_tts = lazy_import("edge_tts")


def list_edge_tts_voices(
    locale: str | None = None,
    content_categories: list[str] | None = None,
    voice_personalities: list[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
    include_status_info: bool = False,
    include_codec_info: bool = False,
) -> tuple[EdgeTTSVoiceTable, EdgeTTSKeys]:
    """
    List Edge TTS voices based on provided filters.

    Parameters
    ----------
    locale : str, optional
        The locale to filter Edge TTS voices by.

    content_categories : list[str], optional
        The content categories to filter Edge TTS voices by.

    voice_personalities : list[str], optional
        The voice personalities to filter Edge TTS voices by.

    offset : int, default=0
        The offset to start listing Edge TTS voices from.

    limit : int, optional
        The limit on how many Edge TTS voices to list.

    include_status_info : bool, default=False
        Include status information for each Edge TTS voice.

    include_codec_info : bool, default=False
        Include codec information for each Edge TTS voice.

    Returns
    -------
        table : list[list[str]]
            A table containing information on the listed Edge TTS
            voices.
        keys : list[str]
            The keys used to generate the table.


    """
    keys: list[EdgeTTSVoiceKey] = [
        "Name",
        "FriendlyName",
        "ShortName",
        "Locale",
    ]

    if include_status_info:
        keys.append("Status")
    if include_codec_info:
        keys.append("SuggestedCodec")
    voice_tag_keys: list[EdgeTTSVoiceTagKey] = [
        "ContentCategories",
        "VoicePersonalities",
    ]
    all_keys: EdgeTTSKeys = keys + voice_tag_keys

    voices = anyio.run(edge_tts.list_voices)
    filtered_voices = [
        v
        for v in voices
        if (
            (locale is None or locale in v["Locale"])
            and (
                content_categories is None
                or any(
                    c in ", ".join(v["VoiceTag"]["ContentCategories"])
                    for c in content_categories
                )
            )
            and (
                voice_personalities is None
                or any(
                    p in ", ".join(v["VoiceTag"]["VoicePersonalities"])
                    for p in voice_personalities
                )
            )
        )
    ]
    if limit is not None:
        limited_voices = filtered_voices[offset : offset + limit]
    else:
        limited_voices = filtered_voices[offset:]

    table: list[list[str]] = []
    for voice in limited_voices:
        features = [voice[key] for key in keys]
        features.extend(
            [", ".join(voice["VoiceTag"][tag_key]) for tag_key in voice_tag_keys],
        )
        table.append(features)
    return table, all_keys


def get_edge_tts_voice_names() -> list[str]:
    """
    Get the the short names of all Edge TTS voices.

    Returns
    -------
    list[tuple[str, str]]
        The short names of all Edge TTS voices.

    """
    voices, keys = list_edge_tts_voices()
    return [voice[keys.index("ShortName")] for voice in voices]


def run_edge_tts(
    source: str,
    voice: str = "en-US-ChristopherNeural",
    pitch_shift: int = 0,
    speed_change: int = 0,
    volume_change: int = 0,
) -> Path:
    """
    Convert text to speech using edge TTS.

    Parameters
    ----------
    source : str
        A string or path to a file containing the text to be converted.

    voice : str, default="en-US-ChristopherNeural"
        The short name of the Edge TTS voice which should speak the
        provided text.

    pitch_shift : int, default=0
        The number of hertz to shift the pitch of the Edge TTS voice
        speaking the provided text.

    speed_change : int, default=0
        The percentual change to the speed of the Edge TTS voice
        speaking the provided text.

    volume_change : int, default=0
        The percentual change to the volume of the Edge TTS voice
        speaking the provided text.

    Returns
    -------
    Path
        The path to an audio track containing the spoken text.

    Raises
    ------
    NotProvidedError
        If no source is provided.

    """
    if not source:
        raise NotProvidedError(entity=Entity.SOURCE, ui_msg=UIMessage.NO_TEXT_SOURCE)

    source_path = Path(source)
    source_is_file = source_path.is_file()
    if source_is_file:
        with source_path.open("r", encoding="utf-8") as file:
            text = file.read()
    else:
        text = source

    args_dict = EdgeTTSAudioMetaData(
        text=text,
        file=(
            FileMetaData(name=source_path.name, hash_id=get_file_hash(source_path))
            if source_is_file
            else None
        ),
        voice=voice,
        pitch_shift=pitch_shift,
        speed_change=speed_change,
        volume_change=volume_change,
    ).model_dump()
    SPEECH_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        get_unique_base_path(
            SPEECH_DIR,
            "1_EdgeTTS_Audio",
            args_dict,
        ).with_suffix(suffix)
        for suffix in [".wav", ".json"]
    ]

    converted_audio_path, converted_audio_json_path = paths

    if not all(path.exists() for path in paths):

        pitch_shift_str = f"{pitch_shift:+}Hz"
        speed_change_str = f"{speed_change:+}%"
        volume_change_str = f"{volume_change:+}%"

        communicate = edge_tts.Communicate(
            text,
            voice,
            pitch=pitch_shift_str,
            rate=speed_change_str,
            volume=volume_change_str,
        )

        communicate.save_sync(str(converted_audio_path))

        json_dump(args_dict, converted_audio_json_path)

    return converted_audio_path


def _get_converted_speech_metadata(
    converted_speech_track: StrPath | None,
) -> RVCAudioMetaData | None:
    """
    Get the metadata associated with a converted speech track, if it
    exists.

    Parameters
    ----------
    converted_speech_track : str, optional
        A path to a converted speech track.

    Returns
    -------
    RVCAudioMetaData
        The metadata associated with the converted voice track.

    """
    if not converted_speech_track:
        return None
    converted_speech_path = Path(converted_speech_track)
    converted_speech_json_path = SPEECH_DIR / f"{converted_speech_path.stem}.json"
    if not converted_speech_json_path.is_file():
        return None
    converted_speech_dict = json_load(converted_speech_json_path)
    try:
        converted_speech_metadata = RVCAudioMetaData.model_validate(
            converted_speech_dict,
        )
    except ValidationError:
        return None
    return converted_speech_metadata


def _get_edge_tts_metadata(
    converted_speech_track: StrPath | None,
    converted_speech_metadata: RVCAudioMetaData | None = None,
) -> EdgeTTSAudioMetaData | None:
    """
    Get the metadata associated with the speech track that was converted
    to another voice using RVC.

    Parameters
    ----------
    converted_speech_track : StrPath
        A path to a converted speech track.

    converted_speech_metadata : RVCAudioMetaData, optional
        The metadata associated with the converted speech track.

    Returns
    -------
    EdgeTTSAudioMetaData
        The metadata associated with the speech track that was converted
        to another voice using RVC.

    """
    converted_speech_metadata = (
        converted_speech_metadata
        or _get_converted_speech_metadata(converted_speech_track)
    )
    if not converted_speech_metadata:
        return None
    speech_path = SPEECH_DIR / converted_speech_metadata.audio_track.name
    speech_json_path = speech_path.with_suffix(".json")
    if not speech_json_path.is_file():
        return None
    speech_dict = json_load(speech_json_path)
    try:
        speech_metadata = EdgeTTSAudioMetaData.model_validate(speech_dict)
    except ValidationError:
        return None
    return speech_metadata


def get_mixed_speech_track_name(
    source: StrPath | None = None,
    model_name: str | None = None,
    converted_speech_track: StrPath | None = None,
) -> str:
    """
    Generate a suitable name for a mixed speech track based
    on the source of input text and the RVC model used for speech
    conversion.

    If either source or model name is not provided, but the path to an
    existing converted speech track is provided, then the source and
    model name is inferred from the metadata associated with that track,
    if possible.

    Parameters
    ----------
    source : StrPath, optional
        A string or path to a file containing the text converted
        to speech.

    model_name : str, optional
        The name of the model used for speech conversion.

    converted_speech_track : str, optional
        A path to a converted speech track.

    Returns
    -------
    str
        The name of the speech track.

    """
    converted_speech_metadata = None
    if model_name is None:
        converted_speech_metadata = _get_converted_speech_metadata(
            converted_speech_track,
        )
        model_name = (
            converted_speech_metadata.model_name
            if converted_speech_metadata
            else "Unknown Speaker"
        )
    if source is None:
        speech_metadata = _get_edge_tts_metadata(
            converted_speech_track,
            converted_speech_metadata,
        )
        source_name = (
            Path(speech_metadata.file.name).stem
            if speech_metadata and speech_metadata.file
            else "Text"
        )
    else:
        source_path = Path(source)
        source_name = source_path.stem if source_path.is_file() else "Text"
    return f"{source_name} (Spoken by {model_name})"


def mix_speech(
    speech_track: StrPath,
    output_gain: int = 0,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    output_name: str | None = None,
) -> Path:
    """
    Mix a speech track.

    Parameters
    ----------
    speech_track : str
        A path to the speech track to mix.

    output_gain : int, default=0
        The gain to apply to the speech track.

    output_sr : int, default=44100
        The sample rate of the mixed speech track.

    output_format : AudioExt, default=AudioExt.MP3
        The audio format of the mixed speech track.

    output_name : str, optional
        The name of the mixed speech track.

    Returns
    -------
    Path
        The path to the mixed audio track.

    """
    mixed_audio_track = mix_audio(
        audio_track_gain_pairs=[(speech_track, output_gain)],
        directory=SPEECH_DIR,
        output_sr=output_sr,
        output_format=output_format,
        content_type=MixedAudioType.SPEECH,
    )

    output_name = output_name or get_mixed_speech_track_name(
        converted_speech_track=speech_track,
    )

    mixed_speech_path = OUTPUT_AUDIO_DIR / f"{output_name}.{output_format}"
    return copy_file_safe(mixed_audio_track, mixed_speech_path)


def run_pipeline(
    source: str,
    model_name: str,
    tts_voice: str = "en-US-ChristopherNeural",
    tts_pitch_shift: int = 0,
    tts_speed_change: int = 0,
    tts_volume_change: int = 0,
    n_octaves: int = 0,
    n_semitones: int = 0,
    f0_methods: Sequence[F0Method] | None = None,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.25,
    protect_rate: float = 0.33,
    hop_length: int = 128,
    split_speech: bool = False,
    autotune_speech: bool = False,
    autotune_strength: float = 1,
    clean_speech: bool = False,
    clean_strength: float = 0.7,
    embedder_model: EmbedderModel = EmbedderModel.CONTENTVEC,
    custom_embedder_model: str | None = None,
    sid: int = 0,
    output_gain: int = 0,
    output_sr: int = 44100,
    output_format: AudioExt = AudioExt.MP3,
    output_name: str | None = None,
    progress_bar: gr.Progress | None = None,
) -> tuple[Path, ...]:
    """
    Convert text to speech using a cascaded pipeline combining Edge TTS
    and RVC.

    The text is first converted to speech using Edge TTS, and then that
    speech is converted to a different voice using RVC.

    Parameters
    ----------
    source : str
        A string or path to a file containing the text to be converted
        to speech.

    model_name : str
        The name of the RVC model to use for speech conversion.

    tts_voice : str, default="en-US-ChristopherNeural"
        The short name of the Edge TTS voice to use for text-to-speech
        conversion.

    tts_pitch_shift : int, default=0
        The number of hertz to shift the pitch of the speech generated
        by Edge TTS.

    tts_speed_change : int, default=0
        The perecentual change to the speed of the speech generated by
        Edge TTS.

    tts_volume_change : int, default=0
        The percentual change to the volume of the speech generated by
        Edge TTS.

    n_octaves : int, default=0
        The number of octaves to shift the pitch of the speech converted
        using RVC.

    n_semitones : int, default=0
        The number of semitones to shift the pitch of the speech
        converted using RVC.

    f0_methods : list[F0Method], optional
        The methods to use for pitch extraction during RVC.

    index_rate : float, default=0.5
        The influence of the index file used during RVC.

    filter_radius : int, default=3
        The filter radius used during RVC.

    rms_mix_rate : float, default=0.25
        The blending rate of the volume envelope of the speech converted
        using RVC.

    protect_rate : float, default=0.33
        The protection rate for consonants and breathing sounds used
        during RVC.

    hop_length : int, default=128
        The hop length for CREPE-based pitch extraction used during RVC.

    split_speech : bool, default=False
        Whether to split the Edge TTS speech into smaller segments
        before converting it using RVC.

    autotune_speech : bool, default=False
        Whether to autotune the speech converted using RVC.

    autotune_strength : float, default=1
        The strength of the autotune applied to the converted speech.

    clean_speech : bool, default=False
        Whether to clean the speech converted using RVC.

    clean_strength : float, default=0.7
        The intensity of the cleaning applied to the converted speech.

    embedder_model : EmbedderModel, default=EmbedderModel.CONTENTVEC
        The model to use for generating speaker embeddings during RVC.

    custom_embedder_model : str, optional
        The name of a custom embedder model to use for generating
        speaker embeddings during RVC.

    sid : int, default=0
        The id of the speaker to use for multi-speaker RVC models.

    output_gain: int, default=0
        The gain to apply to the converted speech during mixing.

    output_sr : int, default=44100
        The sample rate of the mixed speech track.

    output_format : AudioExt, default=AudioExt.MP3
        The format of the mixed speech track.

    output_name : str, optional
        The name of the mixed speech track.

    progress_bar : gr.Progress, optional
        Gradio progress bar to update.

    Returns
    -------
    tuple[Path, ...]
        The path to a mixed audio track containing the converted speech,
        and the paths to any intermediate audio tracks that were
        generated along the way.

    """
    validate_model_exists(model_name, Entity.VOICE_MODEL)
    if embedder_model == EmbedderModel.CUSTOM:
        validate_model_exists(custom_embedder_model, Entity.CUSTOM_EMBEDDER_MODEL)
    display_progress("[~] Converting text using Edge TTS...", 0.0, progress_bar)
    speech_track = run_edge_tts(
        source,
        tts_voice,
        tts_pitch_shift,
        tts_speed_change,
        tts_volume_change,
    )
    display_progress("[~] Converting speech using RVC...", 0.33, progress_bar)
    converted_speech_track = convert(
        audio_track=speech_track,
        directory=SPEECH_DIR,
        model_name=model_name,
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_methods=f0_methods,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        split_audio=split_speech,
        autotune_audio=autotune_speech,
        autotune_strength=autotune_strength,
        clean_audio=clean_speech,
        clean_strength=clean_strength,
        embedder_model=embedder_model,
        custom_embedder_model=custom_embedder_model,
        sid=sid,
        content_type=RVCContentType.SPEECH,
    )
    display_progress("[~] Mixing speech track...", 0.66, progress_bar)
    mixed_speech_track = mix_speech(
        speech_track=converted_speech_track,
        output_gain=output_gain,
        output_sr=output_sr,
        output_format=output_format,
        output_name=output_name,
    )

    return mixed_speech_track, speech_track, converted_speech_track
