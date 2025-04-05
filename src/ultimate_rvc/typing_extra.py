"""Extra typing for the Ultimate RVC project."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum, StrEnum
from os import PathLike

from pydantic import BaseModel

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


class DeviceType(StrEnum):
    """Enumeration of device types for training voice models."""

    AUTOMATIC = "Automatic"
    CPU = "CPU"
    GPU = "GPU"


class TrainingSampleRate(StrEnum):
    """Enumeration of sample rates for training voice models."""

    HZ_32K = "32000"
    HZ_40K = "40000"
    HZ_48K = "48000"


class PretrainedSampleRate(StrEnum):
    """Enumeration of valid sample rates for pretrained models."""

    HZ_32K = "32k"
    HZ_40K = "40k"
    HZ_44K = "44k"
    HZ_48K = "48k"


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

    AUTO = "Auto"
    FAISS = "Faiss"
    KMEANS = "KMeans"


class PretrainedType(StrEnum):
    """
    Enumeration of the possible types of pretrained models to finetune
    voice models on.
    """

    NONE = "None"
    DEFAULT = "Default"
    CUSTOM = "Custom"


class Config(BaseModel):
    """Base class for configuration options."""

    clean_strength: float = 0.7
    hop_length: int = 128
    embedder_model: EmbedderModel = EmbedderModel.CONTENTVEC


class GenerationConfig(Config):
    """Base class for generation configuration options."""

    n_octaves: int = 0
    n_semitones: int = 0
    f0_methods: list[F0Method] = [F0Method.RMVPE]
    index_rate: float = 0.5
    rms_mix_rate: float = 0.25
    protect_rate: float = 0.33
    autotune_audio: bool = False
    autotune_strength: float = 1.0
    sid: int = 0
    output_sr: SampleRate = SampleRate.HZ_44100
    output_format: AudioExt = AudioExt.MP3


class SongGenerationConfig(GenerationConfig):
    """Configuration options for song generation."""

    split_audio: bool = False
    clean_audio: bool = False
    room_size: float = 0.15
    wet_level: float = 0.2
    dry_level: float = 0.8
    damping: float = 0.7
    main_gain: int = 0
    inst_gain: int = 0
    backup_gain: int = 0


class SpeechGenerationConfig(GenerationConfig):
    """Configuration options for speech generation."""

    tts_pitch_shift: int = 0
    tts_speed_change: int = 0
    tts_volume_change: int = 0
    split_audio: bool = True
    clean_audio: bool = True
    output_gain: int = 0


class TrainingConfig(Config):
    """Configuration options for training."""

    sample_rate: TrainingSampleRate = TrainingSampleRate.HZ_40K
    filter_audio: bool = True
    clean_audio: bool = False
    split_method: AudioSplitMethod = AudioSplitMethod.AUTOMATIC
    chunk_len: float = 3.0
    overlap_len: float = 0.3
    f0_method: TrainingF0Method = TrainingF0Method.RMVPE
    include_mutes: int = 2
    extraction_acceleration: DeviceType = DeviceType.AUTOMATIC
    num_epochs: int = 500
    batch_size: int = 8
    detect_overtraining: bool = False
    overtraining_threshold: int = 50
    vocoder: Vocoder = Vocoder.HIFI_GAN
    index_algorithm: IndexAlgorithm = IndexAlgorithm.AUTO
    pretrained_type: PretrainedType = PretrainedType.DEFAULT
    save_interval: int = 10
    save_all_checkpoints: bool = False
    save_all_weights: bool = False
    clear_saved_data: bool = False
    upload_model: bool = False
    training_acceleration: DeviceType = DeviceType.AUTOMATIC
    preload_dataset: bool = False
    reduce_memory_usage: bool = False
