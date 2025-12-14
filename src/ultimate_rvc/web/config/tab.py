"""Module defining common component configurations for UI tabs."""

from __future__ import annotations

from pydantic import BaseModel

from ultimate_rvc.typing_extra import (
    AudioExt,
    AudioNormalizationMode,
    AudioSplitMethod,
    EmbedderModel,
    F0Method,
    IndexAlgorithm,
    PrecisionType,
    PretrainedType,
    SampleRate,
    TrainingSampleRate,
    Vocoder,
)
from ultimate_rvc.web.config.component import (
    CheckboxConfig,
    DropdownConfig,
    NumberConfig,
    SliderConfig,
    TextboxConfig,
)
from ultimate_rvc.web.typing_extra import DatasetType, SongSourceType, SpeechSourceType


class BaseTabConfig(BaseModel):
    """
    Base model defining common component configuration settings for
    UI tabs.

    Attributes
    ----------
    embedder_model : DropdownConfig
        Configuration settings for an embedder model dropdown component.
    custom_embedder_model : DropdownConfig
        Configuration settings for a custom embedder model dropdown
        component.

    """

    embedder_model: DropdownConfig = DropdownConfig(
        label="Embedder model",
        info="The model to use for generating speaker embeddings.",
        value=EmbedderModel.CONTENTVEC,
        choices=list(EmbedderModel),
        exclude_value=True,
    )
    custom_embedder_model: DropdownConfig = DropdownConfig(
        label="Custom embedder model",
        info="Select a custom embedder model from the dropdown.",
        value=None,
        visible=False,
        render=False,
        exclude_value=True,
    )


class GenerationConfig(BaseTabConfig):
    """
    Common component configuration settings for generation tabs.

    voice_model : DropdownConfig
        Configuration settings for a voice model dropdown component.
    f0_method : DropdownConfig
        Configuration settings for a pitch extraction algorithm
        dropdown component.
    index_rate : SliderConfig
        Configuration settings for an index rate slider component.
    rms_mix_rate : SliderConfig
        Configuration settings for a RMS mix rate slider component.
    protect_rate : SliderConfig
        Configuration settings for a protect rate slider component.
    split_voice : CheckboxConfig
        Configuration settings for a split voice checkbox component.
    autotune_voice: CheckboxConfig
        Configuration settings for an autotune voice checkbox component.
    autotune_strength: SliderConfig
        Configuration settings for an autotune strength slider
        component.
    proposed_pitch: CheckboxConfig
        Configuration settings for a proposed pitch checkbox component.
    proposed_pitch_threshold: SliderConfig
        Configuration settings for a proposed pitch threshold slider
        component.
    sid : NumberConfig
        Configuration settings for a speaker ID number component.
    output_sr : DropdownConfig
        Configuration settings for an output sample rate dropdown
        component.
    output_format : DropdownConfig
        Configuration settings for an output format dropdown
        component.
    output_name : TextboxConfig
        Configuration settings for an output name textbox component.

    See Also
    --------
    BaseTabConfig
        Parent model defining common component configuration settings
        for UI tabs.

    """

    voice_model: DropdownConfig = DropdownConfig(
        label="Voice model",
        info="Select a model to use for voice conversion.",
        value=None,
        render=False,
        exclude_value=True,
    )
    f0_method: DropdownConfig = DropdownConfig(
        label="Pitch extraction algorithm",
        info="RMVPE is recommended for most cases and is the default.",
        value=F0Method.RMVPE,
        choices=list(F0Method),
        multiselect=False,
    )
    index_rate: SliderConfig = SliderConfig(
        label="Index rate",
        info=(
            "Increase to bias the conversion towards the accent of the voice model."
            " Decrease to potentially reduce artifacts coming from the voice"
            " model.<br><br><br>"
        ),
        value=0.3,
        minimum=0.0,
        maximum=1.0,
    )
    rms_mix_rate: SliderConfig = SliderConfig(
        label="RMS mix rate",
        info=(
            "How much to mimic the loudness (0) of the input voice or a fixed loudness"
            " (1). A value of 1 is recommended for most cases.<br><br>"
        ),
        value=1.0,
        minimum=0.0,
        maximum=1.0,
    )
    protect_rate: SliderConfig = SliderConfig(
        label="Protect rate",
        info=(
            "Controls the extent to which consonants and breathing sounds are protected"
            " from artifacts. A higher value offers more protection but may worsen the"
            " indexing effect.<br><br>"
        ),
        value=0.33,
        minimum=0.0,
        maximum=0.5,
    )

    split_voice: CheckboxConfig = CheckboxConfig(
        label="Split input voice",
        info=(
            "Whether to split the input voice track into smaller segments before"
            " converting it. This can improve output quality for longer voice tracks."
        ),
        value=False,
    )
    autotune_voice: CheckboxConfig = CheckboxConfig(
        label="Autotune converted voice",
        info="Whether to apply autotune to the converted voice.",
        value=False,
        exclude_value=True,
    )
    autotune_strength: SliderConfig = SliderConfig(
        label="Autotune intensity",
        info=(
            "Higher values result in stronger snapping to the chromatic grid and"
            " artifacting."
        ),
        value=1.0,
        minimum=0.0,
        maximum=1.0,
        visible=False,
    )
    proposed_pitch: CheckboxConfig = CheckboxConfig(
        label="Proposed pitch",
        info=(
            "Whether to adjust the pitch of the converted voice so that it matches the"
            " range of the voice model used."
        ),
        value=False,
        exclude_value=True,
    )
    proposed_pitch_threshold: SliderConfig = SliderConfig(
        label="Proposed pitch threshold",
        info=(
            "Male voice models typically use 155.0 and female voice models typically"
            " use 255.0."
        ),
        value=155.0,
        minimum=50.0,
        maximum=1200.0,
        visible=False,
    )
    sid: NumberConfig = NumberConfig(
        label="Speaker ID",
        info="Speaker ID for multi-speaker-models.",
        value=0,
        precision=0,
    )
    output_sr: DropdownConfig = DropdownConfig(
        label="Output sample rate",
        info="The sample rate of the mixed output track.",
        value=SampleRate.HZ_44K,
        choices=list(SampleRate),
    )
    output_format: DropdownConfig = DropdownConfig(
        label="Output format",
        info="The audio format of the mixed output track.",
        value=AudioExt.MP3,
        choices=list(AudioExt),
    )
    output_name: TextboxConfig = TextboxConfig(
        label="Output name",
        info="If no name is provided, a suitable name will be generated automatically.",
        value=None,
        placeholder="Ultimate RVC output",
        exclude_value=True,
    )


class SongGenerationConfig(GenerationConfig):
    """
    Common component configuration settings for song generation tabs.

    Attributes
    ----------
    source_type : DropdownConfig
        Configuration settings for a source type dropdown component.
    source : TextboxConfig
        Configuration settings for an input source textbox component.
    cached_song : DropdownConfig
        Configuration settings for a cached song dropdown component.
    clean_strength : SliderConfig
        Configuration settings for a clean strength slider component.
    clean_voice : CheckboxConfig
        Configuration settings for a clean voice checkbox component.
    room_size : SliderConfig
        Configuration settings for a room size slider component.
    wet_level : SliderConfig
        Configuration settings for a wetness level slider component.
    dry_level : SliderConfig
        Configuration settings for a dryness level slider component.
    damping : SliderConfig
        Configuration settings for a damping level slider component.
    main_gain : SliderConfig
        Configuration settings for a main gain slider component.
    inst_gain : SliderConfig
        Configuration settings for an instrumentals gain slider
        component.
    backup_gain : SliderConfig
        Configuration settings for a backup vocals gain slider
        component.

    See Also
    --------
    GenerationConfig
        Parent model defining common component configuration settings
        for song generation tabs.

    """

    source_type: DropdownConfig = DropdownConfig(
        label="Source type",
        info="The type of source to retrieve a song from.",
        value=SongSourceType.PATH,
        choices=list(SongSourceType),
        type="index",
        exclude_value=True,
    )
    source: TextboxConfig = TextboxConfig(
        label="Source",
        info="Link to a song on YouTube or the full path of a local audio file.",
        value=None,
        exclude_value=True,
    )
    cached_song: DropdownConfig = DropdownConfig(
        label="Source",
        info="Select a song from the list of cached songs.",
        value=None,
        visible=False,
        render=False,
        exclude_value=True,
    )
    clean_voice: CheckboxConfig = CheckboxConfig(
        label="Clean converted voice",
        info="Whether to clean the converted voice using noise reduction algorithms.",
        value=False,
        exclude_value=True,
    )
    clean_strength: SliderConfig = SliderConfig.clean_strength(visible=False)
    room_size: SliderConfig = SliderConfig(
        label="Room size",
        info=(
            "Size of the room which reverb effect simulates. Increase for longer reverb"
            " time."
        ),
        value=0.15,
        minimum=0.0,
        maximum=1.0,
    )
    wet_level: SliderConfig = SliderConfig(
        label="Wetness level",
        info="Loudness of converted vocals with reverb effect applied.",
        value=0.2,
        minimum=0.0,
        maximum=1.0,
    )
    dry_level: SliderConfig = SliderConfig(
        label="Dryness level",
        info="Loudness of converted vocals without reverb effect applied.",
        value=0.8,
        minimum=0.0,
        maximum=1.0,
    )
    damping: SliderConfig = SliderConfig(
        label="Damping level",
        info="Absorption of high frequencies in reverb effect.",
        value=0.7,
        minimum=0.0,
        maximum=1.0,
    )
    main_gain: SliderConfig = SliderConfig.gain(
        label="Main gain",
        info="The gain to apply to the main vocals.",
    )
    inst_gain: SliderConfig = SliderConfig.gain(
        label="Instrumentals gain",
        info="The gain to apply to the instrumentals.",
    )
    backup_gain: SliderConfig = SliderConfig.gain(
        label="Backup gain",
        info="The gain to apply to the backup vocals.",
    )


class SpeechGenerationConfig(GenerationConfig):
    """
    Common component configuration settings for speech generation tabs.

    Attributes
    ----------
    source_type : DropdownConfig
        Configuration settings for a source type dropdown component.
    source : TextboxConfig
        Configuration settings for an input source textbox component.
    edge_tts_voice : DropdownConfig
        Configuration settings for an Edge TTS voice dropdown
        component.
    n_octaves : SliderConfig
        Configuration settings for an octave pitch shift slider
        component.
    n_semitones : SliderConfig
        Configuration settings for a semitone pitch shift slider
        component.
    tts_pitch_shift : SliderConfig
        Configuration settings for a TTS pitch shift slider
        component.
    tts_speed_change : SliderConfig
        Configuration settings for a TTS speed change slider
        component.
    tts_volume_change : SliderConfig
        Configuration settings for a TTS volume change slider
        component.
    clean_voice : CheckboxConfig
        Configuration settings for a clean voice checkbox
        component.
    clean_strength : SliderConfig
        Configuration settings for a clean strength slider
        component.
    output_gain : GainSliderConfig
        Configuration settings for an output gain slider component.

    See Also
    --------
    GenerationConfig
        Parent model defining common component configuration settings
        for generation tabs.

    """

    source_type: DropdownConfig = DropdownConfig(
        label="Source type",
        info="The type of source to generate speech from.",
        value=SpeechSourceType.TEXT,
        choices=list(SpeechSourceType),
        type="index",
        exclude_value=True,
    )
    source: TextboxConfig = TextboxConfig(
        label="Source",
        info="Text to generate speech from",
        value=None,
        exclude_value=True,
    )
    edge_tts_voice: DropdownConfig = DropdownConfig(
        label="Edge TTS voice",
        info="Select a voice to use for text to speech conversion.",
        value=None,
        render=False,
        exclude_value=True,
    )
    n_octaves: SliderConfig = SliderConfig.octave_shift(
        label="Octave shift",
        info=(
            "The number of octaves to pitch-shift the converted speech by. Use 1 for"
            " male-to-female and -1 for vice-versa."
        ),
    )
    n_semitones: SliderConfig = SliderConfig.semitone_shift(
        label="Semitone shift",
        info="The number of semi-tones to pitch-shift the converted speech by.",
    )
    tts_pitch_shift: SliderConfig = SliderConfig(
        label="Edge TTS pitch shift",
        info=(
            "The number of hertz to shift the pitch of the speech generated by Edge"
            " TTS."
        ),
        value=0,
        minimum=-100,
        maximum=100,
        step=1,
    )
    tts_speed_change: SliderConfig = SliderConfig(
        label="TTS speed change",
        info="The percentual change to the speed of the speech generated by Edge TTS.",
        value=0,
        minimum=-50,
        maximum=100,
        step=1,
    )
    tts_volume_change: SliderConfig = SliderConfig(
        label="TTS volume change",
        info="The percentual change to the volume of the speech generated by Edge TTS.",
        value=0,
        minimum=-100,
        maximum=100,
        step=1,
    )
    clean_voice: CheckboxConfig = CheckboxConfig(
        label="Clean converted voice",
        info="Whether to clean the converted voice using noise reduction algorithms.",
        value=True,
        exclude_value=True,
    )
    clean_strength: SliderConfig = SliderConfig.clean_strength(visible=True)
    output_gain: SliderConfig = SliderConfig.gain(
        label="Output gain",
        info="The gain to apply to the converted speech.<br><br>",
    )


class TrainingConfig(BaseTabConfig):
    """
    Common component configuration settings for training tabs.

    Attributes
    ----------
    dataset_type : DropdownConfig
        Configuration settings for a dataset type dropdown component.
    dataset : DropdownConfig
        Configuration settings for a dataset dropdown component.
    dataset_name : TextboxConfig
        Configuration settings for a dataset name textbox component.
    preprocess_model : DropdownConfig
        Configuration settings for a model name dropdown component
        for audio preprocessing.
    sample_rate : DropdownConfig
        Configuration settings for a sample rate dropdown component.
    normalization_mode: DropdownConfig
        Configuration settings for a normalization mode dropdown
        component.
    filter_audio : CheckboxConfig
        Configuration settings for a filter audio checkbox component.
    clean_audio : CheckboxConfig
        Configuration settings for a clean audio checkbox component.
    clean_strength : SliderConfig
        Configuration settings for a clean strength slider component.
    split_method : DropdownConfig
        Configuration settings for an audio splitting method dropdown
        component.
    chunk_len : SliderConfig
        Configuration settings for a chunk length slider component.
    overlap_len : SliderConfig
        Configuration settings for an overlap length slider component.
    preprocess_cores : SliderConfig
        Configuration settings for a CPU cores slider component for
        preprocessing.
    extract_model : DropdownConfig
        Configuration settings for a model name dropdown component for
        feature extraction.
    f0_method : DropdownConfig
        Configuration settings for an F0 method dropdown component.
    include_mutes : SliderConfig
        Configuration settings for an include mutes slider component.
    extract_cores : SliderConfig
        Configuration settings for a CPU cores slider component for
        feature extraction.
    extraction_acceleration : HardwareAccelerationConfig
        Configuration settings for a hardware acceleration component for
        feature extraction.
    extraction_gpus : DropdownConfig
        Configuration settings for a GPU dropdown compoennt for feature
        extraction.
    train_model : DropdownConfig
        Configuration settings for a model name dropdown component for
        training.
    num_epochs : SliderConfig
        Configuration settings for a number of epochs slider component.
    batch_size : SliderConfig
        Configuration settings for a batch size slider component.
    detect_overtraining : CheckboxConfig
        Configuration settings for a detect overtraining checkbox
        component.
    overtraining_threshold : SliderConfig
        Configuration settings for an overtraining threshold slider
        component.
    vocoder : DropdownConfig
        Configuration settings for a vocoder dropdown component.
    index_algorithm : DropdownConfig
        Configuration settings for an index algorithm dropdown
        component.
    pretrained_type : DropdownConfig
        Configuration settings for a pretrained model type dropdown
        component.
    custom_pretrained_model : DropdownConfig
        Configuration settings for a custom pretrained model dropdown
        component.
    save_interval : SliderConfig
        Configuration settings for a save-interval slider component.
    save_all_checkpoints : CheckboxConfig
        Configuration settings for a save-all-checkpoints checkbox
        component.
    save_all_weights : CheckboxConfig
        Configuration settings for a save-all-weights checkbox
        component.
    clear_saved_data : CheckboxConfig
        Configuration settings for a clear-saved-data checkbox
        component.
    upload_model : CheckboxConfig
        Configuration settings for an upload voice model checkbox
        component.
    upload_name : TextboxConfig
        Configuration settings for an upload name textbox component.
    training_acceleration : HardwareAccelerationConfig
        Configuration settings for a hardware acceleration component for
        training.
    training_gpus : DropdownConfig
        Configuration settings for a GPU dropdown component for
        training.
    precision: DropdownConfig
        Configuration settings for a precision type dropdown component.
    preload_dataset : CheckboxConfig
        Configuration settings for a preload dataset checkbox component.
    reduce_memory_usage : CheckboxConfig
        Configuration settings for a reduce-memory-usage checkbox
        component.

    See Also
    --------
    BaseTabConfig
        Parent model defining common component configuration settings
        for UI tabs.

    """

    dataset_type: DropdownConfig = DropdownConfig(
        label="Dataset type",
        info="Select the type of dataset to preprocess.",
        value=DatasetType.NEW_DATASET,
        choices=list(DatasetType),
        exclude_value=True,
    )
    dataset: DropdownConfig = DropdownConfig(
        label="Dataset path",
        info=(
            "The path to an existing dataset. Either select a path to a previously"
            " created dataset or provide a path to an external dataset."
        ),
        value=None,
        allow_custom_value=True,
        visible=False,
        render=False,
        exclude_value=True,
    )
    dataset_name: TextboxConfig = TextboxConfig(
        label="Dataset name",
        info=(
            "The name of the new dataset. If the dataset already exists, the provided"
            " audio files will be added to it."
        ),
        value="My dataset",
        exclude_value=True,
    )
    preprocess_model: DropdownConfig = DropdownConfig(
        label="Model name",
        info=(
            "Name of the model to preprocess the given dataset for. Either select an"
            " existing model from the dropdown or provide the name of a new model."
        ),
        value="My model",
        allow_custom_value=True,
        render=False,
        exclude_value=True,
    )
    sample_rate: DropdownConfig = DropdownConfig(
        label="Sample rate",
        info="Target sample rate for the audio files in the provided dataset.",
        value=TrainingSampleRate.HZ_40K,
        choices=list(TrainingSampleRate),
    )
    normalization_mode: DropdownConfig = DropdownConfig(
        label="Normalization mode",
        info=(
            "The normalization method to use for the audio files in the provided"
            " dataset."
        ),
        value=AudioNormalizationMode.POST,
        choices=list(AudioNormalizationMode),
    )
    filter_audio: CheckboxConfig = CheckboxConfig(
        label="Filter audio",
        info=(
            "Whether to remove low-frequency sounds from the audio files in the"
            " provided dataset by applying a high-pass butterworth filter.<br><br>"
        ),
        value=True,
    )
    clean_audio: CheckboxConfig = CheckboxConfig(
        label="Clean audio",
        info=(
            "Whether to clean the audio files in the provided dataset using noise"
            " reduction algorithms.<br><br><br>"
        ),
        value=False,
        exclude_value=True,
    )
    clean_strength: SliderConfig = SliderConfig.clean_strength(visible=False)
    split_method: DropdownConfig = DropdownConfig(
        label="Audio splitting method",
        info=(
            "The method to use for splitting the audio files in the provided dataset."
            " Use the `Skip` method to skip splitting if the audio files are already"
            " split. Use the `Simple` method if excessive silence has already been"
            " removed from the audio files. Use the `Automatic` method for automatic"
            " silence detection and splitting around it."
        ),
        value=AudioSplitMethod.AUTOMATIC,
        choices=list(AudioSplitMethod),
        exclude_value=True,
    )
    chunk_len: SliderConfig = SliderConfig(
        label="Chunk length",
        info="Length of split audio chunks.",
        value=3.0,
        minimum=0.5,
        maximum=5.0,
        step=0.1,
        visible=False,
    )
    overlap_len: SliderConfig = SliderConfig(
        label="Overlap length",
        info="Length of overlap between split audio chunks.",
        value=0.3,
        minimum=0.0,
        maximum=0.4,
        step=0.1,
        visible=False,
    )
    preprocess_cores: SliderConfig = SliderConfig.cpu_cores()

    extract_model: DropdownConfig = DropdownConfig(
        label="Model name",
        info=(
            "Name of the model with an associated preprocessed dataset to extract"
            " training features from. When a new dataset is preprocessed, its"
            " associated model is selected by default."
        ),
        value=None,
        render=False,
        exclude_value=True,
    )
    f0_method: DropdownConfig = DropdownConfig(
        label="F0 method",
        info="The method to use for extracting pitch features.",
        value=F0Method.RMVPE,
        choices=list(F0Method),
        exclude_value=True,
    )

    include_mutes: SliderConfig = SliderConfig(
        label="Include mutes",
        info=(
            "The number of mute audio files to include in the generated training file"
            " list. Adding silent files enables the training model to handle pure"
            " silence in inferred audio files. If the preprocessed audio dataset"
            " already contains segments of pure silence, set this to 0."
        ),
        value=2,
        minimum=0,
        maximum=10,
        step=1,
    )
    extraction_cores: SliderConfig = SliderConfig.cpu_cores()
    extraction_acceleration: DropdownConfig = DropdownConfig.hardware_acceleration()
    extraction_gpus: DropdownConfig = DropdownConfig.gpu()

    train_model: DropdownConfig = DropdownConfig(
        label="Model name",
        info=(
            "Name of the model to train. When training features are extracted for a new"
            " model, its name is selected by default."
        ),
        value=None,
        render=False,
        exclude_value=True,
    )
    num_epochs: SliderConfig = SliderConfig(
        label="Number of epochs",
        info=(
            "The number of epochs to train the voice model. A higher number can improve"
            " voice model performance but may lead to overtraining."
        ),
        value=500,
        minimum=1,
        maximum=1000,
        step=1,
    )
    batch_size: SliderConfig = SliderConfig(
        label="Batch size",
        info=(
            "The number of samples in each training batch. It is advisable to align"
            " this value with the available VRAM of your GPU."
        ),
        value=8,
        minimum=1,
        maximum=64,
        step=1,
    )
    detect_overtraining: CheckboxConfig = CheckboxConfig(
        label="Detect overtraining",
        info=(
            "Whether to detect overtraining to prevent the voice model from learning"
            " the training data too well and losing the ability to generalize to new"
            " data."
        ),
        value=False,
        exclude_value=True,
    )
    overtraining_threshold: SliderConfig = SliderConfig(
        label="Overtraining threshold",
        info=(
            "The maximum number of epochs to continue training without any observed"
            " improvement in voice model performance."
        ),
        value=50,
        minimum=1,
        maximum=100,
        visible=False,
        step=1,
    )
    vocoder: DropdownConfig = DropdownConfig(
        label="Vocoder",
        info=(
            "The vocoder to use for audio synthesis during training. HiFi-GAN provides"
            " basic audio fidelity, while RefineGAN provides the highest audio"
            " fidelity."
        ),
        value=Vocoder.HIFI_GAN,
        choices=list(Vocoder),
    )
    index_algorithm: DropdownConfig = DropdownConfig(
        label="Index algorithm",
        info=(
            "The method to use for generating an index file for the trained voice"
            " model. `KMeans` is particularly useful for large datasets."
        ),
        value=IndexAlgorithm.AUTO,
        choices=list(IndexAlgorithm),
    )
    pretrained_type: DropdownConfig = DropdownConfig(
        label="Pretrained model type",
        info=(
            "The type of pretrained model to finetune the voice model on. `None` will"
            " train the voice model from scratch, while `Default` will use a pretrained"
            " model tailored to the specific voice model architecture. `Custom` will"
            " use a custom pretrained that you provide."
        ),
        value=PretrainedType.DEFAULT,
        choices=list(PretrainedType),
        exclude_value=True,
    )
    custom_pretrained_model: DropdownConfig = DropdownConfig(
        label="Custom pretrained model",
        info="Select a custom pretrained model to finetune from the dropdown.",
        value=None,
        visible=False,
        render=False,
        exclude_value=True,
    )
    save_interval: SliderConfig = SliderConfig(
        label="Save interval",
        info=(
            "The epoch interval at which to to save voice model weights and"
            " checkpoints. The best model weights are always saved regardless of this"
            " setting."
        ),
        value=10,
        minimum=1,
        maximum=100,
        step=1,
    )
    save_all_checkpoints: CheckboxConfig = CheckboxConfig(
        label="Save all checkpoints",
        info=(
            "Whether to save a unique checkpoint at each save interval. If not enabled,"
            " only the latest checkpoint will be saved at each interval."
        ),
        value=False,
    )
    save_all_weights: CheckboxConfig = CheckboxConfig(
        label="Save all weights",
        info=(
            "Whether to save unique voice model weights at each save interval. If not"
            " enabled, only the best voice model weights will be saved."
        ),
        value=False,
    )
    clear_saved_data: CheckboxConfig = CheckboxConfig(
        label="Clear saved data",
        info=(
            "Whether to delete any existing training data associated with the voice"
            " model before training commences. Enable this setting only if you are"
            " training a new voice model from scratch or restarting training."
        ),
        value=False,
    )
    upload_model: CheckboxConfig = CheckboxConfig(
        label="Upload voice model",
        info=(
            "Whether to automatically upload the trained voice model so that it can be"
            " used for generation tasks within the Ultimate RVC app."
        ),
        value=False,
        exclude_value=True,
    )
    upload_name: TextboxConfig = TextboxConfig(
        label="Upload name",
        info="The name to give the uploaded voice model.",
        value=None,
        visible=False,
        exclude_value=True,
    )
    training_acceleration: DropdownConfig = DropdownConfig.hardware_acceleration()
    training_gpus: DropdownConfig = DropdownConfig.gpu()
    precision: DropdownConfig = DropdownConfig(
        label="Precision",
        info=(
            "The precision type to use when training the voice model. FP16 and BF16 can"
            " reduce VRAM usage and speed up training on supported hardware."
        ),
        value=PrecisionType.FP32,
        choices=list(PrecisionType),
    )
    preload_dataset: CheckboxConfig = CheckboxConfig(
        label="Preload dataset",
        info=(
            "Whether to preload all training data into GPU memory. This can improve"
            " training speed but requires a lot of VRAM.<br><br>"
        ),
        value=False,
    )
    reduce_memory_usage: CheckboxConfig = CheckboxConfig(
        label="Reduce memory usage",
        info=(
            "Whether to reduce VRAM usage at the cost of slower training speed by"
            " enabling activation checkpointing. This is useful for GPUs with limited"
            " memory (e.g., <6GB VRAM) or when training with a batch size larger than"
            " what your GPU can normally accommodate."
        ),
        value=False,
    )
