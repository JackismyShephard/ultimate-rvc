"""Extra typing for the Ultimate RVC project."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum, StrEnum
from os import PathLike

type StrPath = str | PathLike[str]

type Json = Mapping[str, Json] | Sequence[Json] | str | int | float | bool | None


class SeparationModel(StrEnum):
    """Enumeration of audio separation models."""

    UVR_MDX_NET_VOC_FT = "UVR-MDX-NET-Voc_FT.onnx"
    UVR_MDX_NET_KARA_2 = "UVR_MDXNET_KARA_2.onnx"
    REVERB_HQ_BY_FOXJOY = "Reverb_HQ_By_FoxJoy.onnx"


class SegmentSize(IntEnum):
    """Enumeration of segment sizes for audio separation."""

    SEG_64 = 64
    SEG_128 = 128
    SEG_256 = 256
    SEG_512 = 512
    SEG_1024 = 1024
    SEG_2048 = 2048


class F0Method(StrEnum):
    """Enumeration of pitch extraction methods."""

    RMVPE = "rmvpe"
    CREPE = "crepe"
    CREPE_TINY = "crepe-tiny"
    FCPE = "fcpe"


class EmbedderModel(StrEnum):
    """Enumeration of audio embedding models."""

    CONTENTVEC = "contentvec"
    CHINESE_HUBERT_BASE = "chinese-hubert-base"
    JAPANESE_HUBERT_BASE = "japanese-hubert-base"
    KOREAN_HUBERT_BASE = "korean-hubert-base"
    CUSTOM = "custom"


class RVCContentType(StrEnum):
    """Enumeration of valid content to convert with RVC."""

    VOCALS = "vocals"
    VOICE = "voice"
    SPEECH = "speech"
    AUDIO = "audio"


class SampleRate(IntEnum):
    """Enumeration of supported audio sample rates."""

    HZ_16000 = 16000
    HZ_44100 = 44100
    HZ_48000 = 48000
    HZ_96000 = 96000
    HZ_192000 = 192000


class AudioExt(StrEnum):
    """Enumeration of supported audio file formats."""

    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"
    AAC = "aac"


class TrainingSampleRate(StrEnum):
    """Enumeration of sample rates for training voice models."""

    HZ_32K = "32000"
    HZ_40K = "40000"
    HZ_41K = "41000"
    HZ_48K = "48000"


class RVCVersion(StrEnum):
    """Enumeration of RVC model versions."""

    V1 = "v1"
    V2 = "v2"


class TrainingF0Method(StrEnum):
    """Enumeration of pitch extraction methods for training."""

    RMVPE = "rmvpe"
    CREPE = "crepe"
    CREPE_TINY = "crepe-tiny"


class AudioSplitMethod(StrEnum):
    """
    Enumeration of methods to use for splitting audio files during
    dataset preprocessing.
    """

    SKIP = "Skip"
    SIMPLE = "Simple"
    AUTOMATIC = "Automatic"


class Vocoder(StrEnum):
    """Enumeration of vocoders for training voice models."""

    HIFI_GAN = "HiFi-GAN"
    MRF_HIFI_GAN = "MRF HiFi-GAN"
    REFINE_GAN = "RefineGAN"


class IndexAlgorithm(StrEnum):
    """Enumeration of indexing algorithms for training voice models."""

    AUTO = "auto"
    FAISS = "Faiss"
    KMEANS = "KMeans"
