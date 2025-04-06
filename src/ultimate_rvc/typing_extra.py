"""Extra typing for the Ultimate RVC project."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum, StrEnum
from os import PathLike

from pydantic import BaseModel, ConfigDict, Field

import gradio as gr
from gradio.components import Component
from gradio.components.dropdown import DEFAULT_VALUE, DefaultValue

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


class BaseComponentConfig[T: Component](BaseModel):
    """Configuration for a component in the UI."""

    label: str = Field(default="", frozen=True)
    info: str = Field(default="", frozen=True)
    instantiation: T | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ComponentConfig[U, T: Component](BaseComponentConfig[T]):
    """Configuration for a component in the UI."""

    value: U


class ImmutableComponentConfig[U, T: Component](BaseComponentConfig[T]):
    value: U = Field(frozen=True)


class SliderConfig(ComponentConfig[float, gr.Slider]):
    pass


class CheckboxConfig(ComponentConfig[bool, gr.Checkbox]):
    pass


class NumberConfig(ComponentConfig[int, gr.Number]):
    pass


class RadioConfig(ComponentConfig[str | int, gr.Radio]):
    pass


class DropdownConfig(ComponentConfig[str | list[str] | int, gr.Dropdown]):
    pass


class ImmutableDropdownConfig(
    ImmutableComponentConfig[str | DefaultValue | None, gr.Dropdown],
):
    pass


class SongDirComponentConfig(ImmutableDropdownConfig):
    value: str | DefaultValue | None = DEFAULT_VALUE
    label: str = Field(default="song directory", frozen=True)
    info: str = Field(
        default=(
            "Directory where intermediate audio files are stored and loaded from"
            " locally. When a new song is retrieved, its directory is chosen by"
            " default."
        ),
        frozen=True,
    )


class BaseTabConfig(BaseModel):
    """Base class for configuration options."""

    clean_strength: SliderConfig = SliderConfig(
        label="Cleaning intensity",
        info=(
            "Higher values result in stronger cleaning, but may lead to a more"
            " compressed sound."
        ),
        value=5,
    )
    hop_length: SliderConfig = SliderConfig(
        label="Hop length",
        info=(
            "How often the CREPE-based pitch extraction method checks"
            " for pitch changes measured in milliseconds. Lower values"
            " lead to longer conversion times and a higher risk of"
            " voice cracks, but better pitch accuracy."
        ),
        value=128,
    )
    embedder_model: DropdownConfig = DropdownConfig(
        label="Embedder model",
        info="The model to use for generating speaker embeddings.",
        value=EmbedderModel.CONTENTVEC,
    )
    custom_embedder_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Custom embedder model",
        info="Select a custom embedder model from the dropdown.",
        value=DEFAULT_VALUE,
    )


class GenerationConfig(BaseTabConfig):
    """Base class for generation configuration options."""

    f0_methods: DropdownConfig = DropdownConfig(
        label="Pitch extraction algorithm(s)",
        info=(
            "If more than one method is selected, then the median of"
            " the pitch values extracted by each method is used. RMVPE"
            " is recommended for most cases and is the default when no"
            " method is selected."
        ),
        value=[F0Method.RMVPE],
    )
    index_rate: SliderConfig = SliderConfig(
        label="Index rate",
        info=(
            "Increase to bias the conversion towards the accent of the"
            " voice model. Decrease to potentially reduce artifacts"
            " coming from the voice model.<br><br><br>"
        ),
        value=0.5,
    )
    rms_mix_rate: SliderConfig = SliderConfig(
        label="RMS mix rate",
        info=(
            "How much to mimic the loudness (0) of the input voice or"
            " a fixed loudness (1). A value of 0.25 is recommended for"
            " most cases."
        ),
        value=0.25,
    )
    protect_rate: SliderConfig = SliderConfig(
        label="Protect rate",
        info=(
            "Controls the extent to which consonants and breathing"
            " sounds are protected from artifacts. A higher value"
            " offers more protection but may worsen the indexing"
            " effect."
        ),
        value=0.33,
    )
    autotune_voice: CheckboxConfig = CheckboxConfig(
        label="Autotune converted voice",
        info="Whether to apply autotune to the converted voice.<br><br>",
        value=False,
    )
    autotune_strength: SliderConfig = SliderConfig(
        label="Autotune intensity",
        info=(
            "Higher values result in stronger snapping to the chromatic grid and"
            " artifacting."
        ),
        value=1.0,
    )
    sid: NumberConfig = NumberConfig(
        label="Speaker ID",
        info="Speaker ID for multi-speaker-models.",
        value=0,
    )
    output_sr: DropdownConfig = DropdownConfig(
        label="Output sample rate",
        info="The sample rate of the mixed output track.",
        value=SampleRate.HZ_44100,
    )
    output_format: DropdownConfig = DropdownConfig(
        label="Output format",
        info="The audio format of the mixed output track.",
        value=AudioExt.MP3,
    )


class SongGenerationConfig(GenerationConfig):
    """Configuration options for song generation."""

    voice_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Voice model",
        info="Select a voice model to use for converting vocals.",
        value=None,
    )
    cached_song: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Source",
        info="Select a song from the list of cached songs.",
        value=DEFAULT_VALUE,
    )
    split_voice: CheckboxConfig = CheckboxConfig(
        label="Split input voice",
        info=(
            "Whether to split the input voice track into smaller"
            " segments before converting it. This can improve"
            " output quality for longer voice tracks."
        ),
        value=False,
    )
    clean_voice: CheckboxConfig = CheckboxConfig(
        label="Clean converted voice",
        info=(
            "Whether to clean the converted voice using noise reduction"
            " algorithms.<br><br>"
        ),
        value=False,
    )
    room_size: SliderConfig = SliderConfig(
        label="Room size",
        info=(
            "Size of the room which reverb effect simulates. Increase"
            " for longer reverb time."
        ),
        value=0.15,
    )
    wet_level: SliderConfig = SliderConfig(
        label="Wetness level",
        info="Loudness of converted vocals with reverb effect applied.",
        value=0.2,
    )
    dry_level: SliderConfig = SliderConfig(
        label="Dryness level",
        info="Loudness of converted vocals without reverb effect applied.",
        value=0.8,
    )
    damping: SliderConfig = SliderConfig(
        label="Damping level",
        info="Absorption of high frequencies in reverb effect.",
        value=0.75,
    )
    main_gain: SliderConfig = SliderConfig(
        label="Main gain",
        info="The gain to apply to the main vocals.",
        value=0,
    )
    inst_gain: SliderConfig = SliderConfig(
        label="Instrumentals gain",
        info="The gain to apply to the instrumentals.",
        value=0,
    )
    backup_gain: SliderConfig = SliderConfig(
        label="Backup gain",
        info="The gain to apply to the backup vocals.",
        value=0,
    )


class OneClickGenerationConfig(GenerationConfig):

    n_octaves: SliderConfig = SliderConfig(
        label="Vocal pitch shift",
        info=(
            "The number of octaves to shift the pitch of the converted"
            " vocals by. Use 1 for male-to-female and -1 for vice-versa."
        ),
        value=0,
    )

    n_semitones: SliderConfig = SliderConfig(
        label="Overall pitch shift",
        info=(
            "The number of semi-tones to shift the pitch of the converted"
            " vocals, instrumentals and backup vocals by."
        ),
        value=0,
    )


class MultiStepGenerationConfig(GenerationConfig):

    separate_audio_dir: SongDirComponentConfig = SongDirComponentConfig()
    convert_vocals_dir: SongDirComponentConfig = SongDirComponentConfig()
    postporcess_vocals_dir: SongDirComponentConfig = SongDirComponentConfig()
    pitch_shift_background_dir: SongDirComponentConfig = SongDirComponentConfig()
    mix_dir: SongDirComponentConfig = SongDirComponentConfig()
    separation_model: DropdownConfig = DropdownConfig(
        value=SeparationModel.UVR_MDX_NET_VOC_FT,
        label="Separation model",
        info="The model to use for audio separation.",
    )
    segment_size: RadioConfig = RadioConfig(
        value=SegmentSize.SEG_512,
        label="Segment size",
        info=(
            "Size of segments into which the audio is split. Larger consumes more"
            " resources, but may give better results."
        ),
    )
    n_octaves: SliderConfig = SliderConfig(
        label="Pitch shift (octaves)",
        info=(
            "The number of octaves to pitch-shift the converted voice by."
            " Use 1 for male-to-female and -1 for vice-versa."
        ),
        value=0,
    )
    n_semitones: SliderConfig = SliderConfig(
        label="Pitch shift (semi-tones)",
        info=(
            "The number of semi-tones to pitch-shift the converted vocals"
            " by. Altering this slightly reduces sound quality."
        ),
        value=0,
    )

    n_semitones_instrumentals: SliderConfig = SliderConfig(
        value=0,
        label="Instrumental pitch shift",
        info="The number of semi-tones to pitch-shift the instrumentals by",
    )
    n_semitones_backup_vocals: SliderConfig = SliderConfig(
        value=0,
        label="Backup vocal pitch shift",
        info="The number of semi-tones to pitch-shift the backup vocals by",
    )


class SpeechGenerationConfig(GenerationConfig):
    """Configuration options for speech generation."""

    edge_tts_voice: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Edge TTS voice",
        info="Select a voice to use for text to speech conversion.",
        value=DEFAULT_VALUE,
    )
    voice_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Voice model",
        info="Select a voice model to use for speech conversion.",
        value=None,
    )

    n_octaves: SliderConfig = SliderConfig(
        label="Octave shift",
        info=(
            "The number of octaves to pitch-shift the converted speech by."
            " Use 1 for male-to-female and -1 for vice-versa."
        ),
        value=0,
    )
    n_semitones: SliderConfig = SliderConfig(
        label="Semitone shift",
        info="The number of semi-tones to pitch-shift the converted speech by.",
        value=0,
    )
    tts_pitch_shift: SliderConfig = SliderConfig(
        label="Edge TTS pitch shift",
        info=(
            "The number of hertz to shift the pitch of the speech generated"
            " by Edge TTS."
        ),
        value=0,
    )
    tts_speed_change: SliderConfig = SliderConfig(
        label="TTS speed change",
        info="The percentual change to the speed of the speech generated by Edge TTS.",
        value=0,
    )
    tts_volume_change: SliderConfig = SliderConfig(
        label="TTS volume change",
        info="The percentual change to the volume of the speech generated by Edge TTS.",
        value=0,
    )
    split_voice: CheckboxConfig = CheckboxConfig(
        label="Split input voice",
        info=(
            "Whether to split the input voice track into smaller segments"
            " before converting it. This can improve output quality for"
            " longer input voice tracks."
        ),
        value=False,
    )
    clean_voice: CheckboxConfig = CheckboxConfig(
        label="Clean converted voice",
        info=(
            "Whether to clean the converted voice using noise reduction"
            " algorithms.<br><br>"
        ),
        value=False,
    )
    output_gain: SliderConfig = SliderConfig(
        label="Output gain",
        info="The gain to apply to the converted speech.<br><br>",
        value=0,
    )


class OneClickSpeechGenerationConfig(SpeechGenerationConfig):
    pass


class MultiStepSpeechGenerationConfig(SpeechGenerationConfig):

    pass


class TrainingConfig(BaseTabConfig):
    """Configuration options for training."""

    dataset: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Dataset path",
        info=(
            "The path to an existing dataset. Either select a path to a previously"
            " created dataset or provide a path to an external dataset."
        ),
        value=DEFAULT_VALUE,
    )
    preprocess_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Model name",
        info=(
            "Name of the model to preprocess the given dataset for. Either select"
            " an existing model from the dropdown or provide the name of a new"
            " model."
        ),
        value="My model",
    )
    extract_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Model name",
        info=(
            "Name of the model with an associated preprocessed dataset to extract"
            " training features from. When a new dataset is preprocessed, its"
            " associated model is selected by default."
        ),
        value=DEFAULT_VALUE,
    )
    train_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Model name",
        info=(
            "Name of the model to train. When training features are extracted for a"
            " new model, its name is selected by default."
        ),
        value=DEFAULT_VALUE,
    )

    custom_pretrained_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Custom pretrained model",
        info="Select a custom pretrained model to finetune from the dropdown.",
        value=DEFAULT_VALUE,
    )

    sample_rate: DropdownConfig = DropdownConfig(
        label="Sample rate",
        info="Target sample rate for the audio files in the provided dataset.",
        value=TrainingSampleRate.HZ_40K,
    )
    filter_audio: CheckboxConfig = CheckboxConfig(
        label="Filter audio",
        info=(
            "Whether to remove low-frequency sounds from the audio"
            " files in the provided dataset by applying a high-pass"
            " butterworth filter.<br><br>"
        ),
        value=True,
    )
    clean_audio: CheckboxConfig = CheckboxConfig(
        label="Clean audio",
        info=(
            "Whether to clean the audio files in the provided"
            " dataset using noise reduction algorithms.<br><br><br>"
        ),
        value=False,
    )
    split_method: DropdownConfig = DropdownConfig(
        label="Audio splitting method",
        info=(
            "The method to use for splitting the audio files in the provided dataset."
            " Use the Skip method to skip splitting if the audio files are already"
            " split. Use the Simple method if excessive silence has already been"
            " removed from the audio files. Use the Automatic method for automatic"
            " silence detection and splitting around it."
        ),
        value=AudioSplitMethod.AUTOMATIC,
    )
    chunk_len: SliderConfig = SliderConfig(
        label="Chunk length",
        info="Length of split audio chunks.",
        value=3.0,
    )
    overlap_len: SliderConfig = SliderConfig(
        label="Overlap length",
        info="Length of overlap between split audio chunks.",
        value=0.3,
    )
    f0_method: DropdownConfig = DropdownConfig(
        label="F0 method",
        info="The method to use for extracting pitch features.",
        value=TrainingF0Method.RMVPE,
    )
    include_mutes: SliderConfig = SliderConfig(
        label="Include mutes",
        info=(
            "The number of mute audio files to include in the generated"
            " training file list. Adding silent files enables the"
            " training model to handle pure silence in inferred audio"
            " files. If the preprocessed audio dataset already contains"
            " segments of pure silence, set this to 0."
        ),
        value=2,
    )
    extraction_acceleration: DropdownConfig = DropdownConfig(
        value=DeviceType.AUTOMATIC,
        label="Hardware acceleration",
        info=(
            "The type of hardware acceleration to use for feature"
            " extraction. 'Automatic' will automatically select the"
            " first available GPU and fall back to CPU if no GPUs"
            " are available."
        ),
    )
    num_epochs: SliderConfig = SliderConfig(
        label="Number of epochs",
        info=(
            "The number of epochs to train the voice model. A higher"
            " number can improve voice model performance but may lead"
            " to overtraining."
        ),
        value=500,
    )
    batch_size: SliderConfig = SliderConfig(
        label="Batch size",
        info=(
            "The number of samples in each training batch. It is"
            " advisable to align this value with the available VRAM of"
            " your GPU."
        ),
        value=8,
    )
    detect_overtraining: CheckboxConfig = CheckboxConfig(
        label="Detect overtraining",
        info=(
            "Whether to detect overtraining to prevent the voice model"
            " from learning the training data too well and losing the"
            " ability to generalize to new data."
        ),
        value=False,
    )
    overtraining_threshold: SliderConfig = SliderConfig(
        label="Overtraining threshold",
        info=(
            "The maximum number of epochs to continue training without"
            " any observed improvement in voice model performance."
        ),
        value=50,
    )
    vocoder: DropdownConfig = DropdownConfig(
        label="Vocoder",
        info=(
            "The vocoder to use for audio synthesis during"
            " training. HiFi-GAN provides basic audio fidelity,"
            " while RefineGAN provides the highest audio fidelity."
        ),
        value=Vocoder.HIFI_GAN,
    )
    index_algorithm: DropdownConfig = DropdownConfig(
        label="Index algorithm",
        info=(
            "The method to use for generating an index file for the"
            " trained voice model. KMeans is particularly useful"
            " for large datasets."
        ),
        value=IndexAlgorithm.AUTO,
    )
    pretrained_type: DropdownConfig = DropdownConfig(
        label="Pretrained model type",
        info=(
            "The type of pretrained model to finetune the voice"
            " model on. `None` will train the voice model from"
            " scratch, while `Default` will use a pretrained model"
            " tailored to the specific voice model architecture."
            " `Custom` will use a custom pretrained that you"
            " provide."
        ),
        value=PretrainedType.DEFAULT,
    )
    save_interval: SliderConfig = SliderConfig(
        label="Save interval",
        info=(
            "The epoch interval at which to to save voice model"
            " weights and checkpoints. The best model weights are"
            " always saved regardless of this setting."
        ),
        value=10,
    )
    save_all_checkpoints: CheckboxConfig = CheckboxConfig(
        label="Save all checkpoints",
        info=(
            "Whether to save a unique checkpoint at each save"
            " interval. If not enabled, only the latest checkpoint"
            " will be saved at each interval."
        ),
        value=False,
    )
    save_all_weights: CheckboxConfig = CheckboxConfig(
        label="Save all weights",
        info=(
            "Whether to save unique voice model weights at each"
            " save interval. If not enabled, only the best voice"
            " model weights will be saved."
        ),
        value=False,
    )
    clear_saved_data: CheckboxConfig = CheckboxConfig(
        label="Clear saved data",
        info=(
            "Whether to delete any existing training data"
            " associated with the voice model before training"
            " commences. Enable this setting only if you are"
            " training a new voice model from scratch or restarting"
            " training."
        ),
        value=False,
    )
    upload_model: CheckboxConfig = CheckboxConfig(
        label="Upload voice model",
        info=(
            "Whether to automatically upload the trained voice"
            " model so that it can be used for generation tasks"
            " within the Ultimate RVC app."
        ),
        value=False,
    )
    training_acceleration: DropdownConfig = DropdownConfig(
        label="Hardware acceleration",
        info=(
            "The type of hardware acceleration to use when training"
            " the voice model. 'Automatic' will select the first"
            " available GPU and fall back to CPU if no GPUs are"
            " available."
        ),
        value=DeviceType.AUTOMATIC,
    )
    preload_dataset: CheckboxConfig = CheckboxConfig(
        label="Preload dataset",
        info=(
            "Whether to preload all training data into GPU memory."
            " This can improve training speed but requires a lot of"
            " VRAM.<br><br>"
        ),
        value=False,
    )
    reduce_memory_usage: CheckboxConfig = CheckboxConfig(
        label="Reduce memory usage",
        info=(
            "Whether to reduce VRAM usage at the cost of slower"
            " training speed by enabling activation checkpointing."
            " This is useful for GPUs with limited memory (e.g.,"
            " <6GB VRAM) or when training with a batch size larger"
            " than what your GPU can normally accommodate."
        ),
        value=False,
    )


class MultiStepTrainingConfig(TrainingConfig):
    pass


class ManageModelsConfig(BaseModel):
    voice_models: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Voice models",
        value=DEFAULT_VALUE,
    )
    embedders: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Custom embedder models",
        value=DEFAULT_VALUE,
    )
    pretrained_models: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Custom pretrained models",
        value=DEFAULT_VALUE,
    )
    training_models: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Training models",
        value=DEFAULT_VALUE,
    )


class ManageAudioConfig(BaseModel):
    intermediate_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Song directories",
        info=(
            "Select one or more song directories containing intermediate audio"
            " files to delete."
        ),
        value=DEFAULT_VALUE,
    )
    speech_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Speech audio files",
        info="Select one or more speech audio files to delete.",
        value=DEFAULT_VALUE,
    )
    output_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Output audio files",
        info="Select one or more output audio files to delete.",
        value=DEFAULT_VALUE,
    )
    dataset_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Dataset audio files",
        info="Select one or more datasets containing audio files to delete.",
        value=DEFAULT_VALUE,
    )


class TotalSongGenerationConfig(BaseModel):
    one_click: OneClickGenerationConfig = OneClickGenerationConfig()
    multi_step: MultiStepGenerationConfig = MultiStepGenerationConfig()


class TotalSpeechGenerationConfig(BaseModel):
    one_click: OneClickSpeechGenerationConfig = OneClickSpeechGenerationConfig()
    multi_step: MultiStepSpeechGenerationConfig = MultiStepSpeechGenerationConfig()


class TotalTrainingConfig(BaseModel):
    multi_step: MultiStepTrainingConfig = MultiStepTrainingConfig()


class TotalManagementConfig(BaseModel):
    models: ManageModelsConfig = ManageModelsConfig()
    audio: ManageAudioConfig = ManageAudioConfig()


class TotalConfig(BaseModel):
    song: TotalSongGenerationConfig = TotalSongGenerationConfig()
    speech: TotalSpeechGenerationConfig = TotalSpeechGenerationConfig()
    training: TotalTrainingConfig = TotalTrainingConfig()
    management: TotalManagementConfig = TotalManagementConfig()
