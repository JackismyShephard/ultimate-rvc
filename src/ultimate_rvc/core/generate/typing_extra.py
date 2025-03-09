"""
Module which defines extra types used by modules in the
ultimate_rvc.core.generate package.
"""

from __future__ import annotations

from typing import Literal

from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict

# NOTE these types are used at runtime by pydantic so cannot be
# relegated to a IF TYPE_CHECKING block
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method  # noqa: TC002


class SongSourceType(StrEnum):
    """The type of source providing the song to generate a cover of."""

    URL = auto()
    FILE = auto()
    SONG_DIR = auto()


class AudioExtInternal(StrEnum):
    """Audio file formats for internal use."""

    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    IPOD = "ipod"
    ADTS = "adts"


class DirectoryMetaData(BaseModel):
    """
    Metadata for a directory.

    Attributes
    ----------
    name : str
        The name of the directory.
    path : str
        The path of the directory.

    """

    name: str
    path: str


class FileMetaData(BaseModel):
    """
    Metadata for a file.

    Attributes
    ----------
    name : str
        The name of the file.
    hash_id : str
        The hash ID of the file.

    """

    name: str
    hash_id: str


class WaveifiedAudioMetaData(BaseModel):
    """
    Metadata for a waveified audio track.

    Attributes
    ----------
    audio_track : FileMetaData
        Metadata for the audio track that was waveified.

    """

    audio_track: FileMetaData


class SeparatedAudioMetaData(BaseModel):
    """
    Metadata for a separated audio track.

    Attributes
    ----------
    audio_track : FileMetaData
        Metadata for the audio track that was separated.
    model_name : str
        The name of the model used for separation.
    segment_size : int
        The segment size used for separation.

    """

    audio_track: FileMetaData
    model_name: str
    segment_size: int

    model_config = ConfigDict(protected_namespaces=())


class RVCAudioMetaData(BaseModel):
    """
    Metadata for a voice converted audio track.

    Attributes
    ----------
    voice_track : FileMetaData
        Metadata for the audio track that was voice converted.
    model_name : str
        The name of the model used for voice conversion.
    n_semitones : int
        The number of semitones the converted audio was pitch-shifted
        by.
    f0_methods : list[F0Method]
        The methods used for pitch extraction.
    index_rate : float
        The influence of the index file on the voice conversion.
    filter_radius : int
        The filter radius used for the voice conversion.
    rms_mix_rate : float
        The blending rate of the volume envelope of the converted
        audio.
    protect_rate : float
        The protection rate for consonants and breathing sounds used
        for the audio conversion.
    hop_length : int
        The hop length used for CREPE-based pitch extraction.
    split_audio : bool
        Whether the audio track was split before it was converted.
    autotune_audio : bool
        Whether autotune was applied to the converted audio.
    autotune_strength : float
        The strength of the autotune effect applied to the converted
        audio.
    clean_audio : bool
        Whether the converted audio was cleaned.
    clean_strength : float
        The intensity of the cleaning that was applied to the converted
        audio.
    embedder_model : EmbedderModel
        The model used for generating speaker embeddings.
    custom_embedder_model : str | None
        The name of a custom embedder model used for generating speaker
        embeddings.
    sid : int
        The speaker id used for multi-speaker conversion.

    """

    audio_track: FileMetaData
    model_name: str
    n_semitones: int
    f0_methods: list[F0Method]
    index_rate: float
    filter_radius: int
    rms_mix_rate: float
    protect_rate: float
    hop_length: int
    split_audio: bool
    autotune_audio: bool
    autotune_strength: float
    clean_audio: bool
    clean_strength: float
    embedder_model: EmbedderModel
    custom_embedder_model: str | None
    sid: int

    model_config = ConfigDict(protected_namespaces=())


class EffectedVocalsMetaData(BaseModel):
    """
    Metadata for an effected vocals track.

    Attributes
    ----------
    vocals_track : FileMetaData
        Metadata for the vocals track that effects were applied to.
    room_size : float
        The room size of the reverb effect applied to the vocals track.
    wet_level : float
        The wetness level of the reverb effect applied to the vocals
        track.
    dry_level : float
        The dryness level of the reverb effect. applied to the vocals
        track.
    damping : float
        The damping of the reverb effect applied to the vocals track.

    """

    vocals_track: FileMetaData
    room_size: float
    wet_level: float
    dry_level: float
    damping: float


class PitchShiftMetaData(BaseModel):
    """
    Metadata for a pitch-shifted audio track.

    Attributes
    ----------
    audio_track : FileMetaData
        Metadata for the audio track that was pitch-shifted.
    n_semitones : int
        The number of semitones the audio track was pitch-shifted by.

    """

    audio_track: FileMetaData
    n_semitones: int


class StagedAudioMetaData(BaseModel):
    """
    Metadata for a staged audio track.

    Attributes
    ----------
    audio_track : FileMetaData
        Metadata for the audio track that was staged.
    gain : float
        The gain applied to the audio track.

    """

    audio_track: FileMetaData
    gain: float


class MixedAudioMetaData(BaseModel):
    """
    Metadata for mixed audio.

    Attributes
    ----------
    staged_audio_tracks : list[StagedAudioMetaData]
        Metadata for the staged audio tracks that were mixed.

    output_sr : int
        The sample rate of the mixed audio.
    output_format : AudioExt
        The audio file format of the mixed audio.

    """

    staged_audio_tracks: list[StagedAudioMetaData]
    output_sr: int
    output_format: AudioExt


class EdgeTTSAudioMetaData(BaseModel):
    """
    Metadata for an audio track generated by Edge TTS.

    Attributes
    ----------
    text: str
        The text that was spoken to generate the audio track.
    file : FileMetaData, optional
        Metadata for file containing the text that was spoken to
        generate the audio track.
    voice : str
        The short name of the voice used for generating the audio track.
    pitch_shift : int
        The number of hertz the pitch of the voice speaking the
        provided text was shifted.
    speed_change : int
        The percentual change to the speed of the voice speaking the
        provided text.
    volume_change : int
        The percentual change to the volume of the voice speaking the
        provided text.

    """

    text: str
    file: FileMetaData | None
    voice: str
    pitch_shift: int
    speed_change: int
    volume_change: int


class MixedAudioType(StrEnum):
    """The valid types of mixed audio."""

    AUDIO = "audio"
    SONG = "song"
    SPEECH = "speech"


EdgeTTSVoiceTable = list[list[str]]
EdgeTTSVoiceKey = Literal[
    "Name",
    "FriendlyName",
    "ShortName",
    "Locale",
    "Status",
    "SuggestedCodec",
]
EdgeTTSVoiceTagKey = Literal[
    "ContentCategories",
    "VoicePersonalities",
]
EdgeTTSKeys = list[EdgeTTSVoiceKey | EdgeTTSVoiceTagKey]
