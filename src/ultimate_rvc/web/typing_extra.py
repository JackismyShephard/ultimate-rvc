"""
Module which defines extra types for the web application of the Ultimate
RVC project.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from collections.abc import Callable, Sequence
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

import gradio as gr
from gradio.components import Component
from gradio.components.dropdown import DEFAULT_VALUE, DefaultValue

from ultimate_rvc.typing_extra import (
    AudioExt,
    AudioSplitMethod,
    DeviceType,
    EmbedderModel,
    F0Method,
    IndexAlgorithm,
    PretrainedType,
    SampleRate,
    SegmentSize,
    SeparationModel,
    TrainingF0Method,
    TrainingSampleRate,
    Vocoder,
)

type DropdownChoices = (
    Sequence[str | int | float | tuple[str, str | int | float]] | None
)

type DropdownValue = (
    str | int | float | Sequence[str | int | float] | Callable[..., Any] | None
)


class ConcurrencyId(StrEnum):
    """Enumeration of possible concurrency identifiers."""

    GPU = auto()


class SongSourceType(StrEnum):
    """The type of source providing the song to generate a cover of."""

    PATH = "YouTube link/local path"
    LOCAL_FILE = "Local file"
    CACHED_SONG = "Cached song"


class SpeechSourceType(StrEnum):
    """The type of source providing the text to generate speech from."""

    TEXT = "Text"
    LOCAL_FILE = "Local file"


class SongTransferOption(StrEnum):
    """Enumeration of possible song transfer options."""

    STEP_1_AUDIO = "Step 1: audio"
    STEP_2_VOCALS = "Step 2: vocals"
    STEP_3_VOCALS = "Step 3: vocals"
    STEP_4_INSTRUMENTALS = "Step 4: instrumentals"
    STEP_4_BACKUP_VOCALS = "Step 4: backup vocals"
    STEP_5_MAIN_VOCALS = "Step 5: main vocals"
    STEP_5_INSTRUMENTALS = "Step 5: instrumentals"
    STEP_5_BACKUP_VOCALS = "Step 5: backup vocals"


class SpeechTransferOption(StrEnum):
    """Enumeration of possible speech transfer options."""

    STEP_2_SPEECH = "Step 2: speech"
    STEP_3_SPEECH = "Step 3: speech"


class ComponentVisibilityKwArgs(TypedDict, total=False):
    """
    Keyword arguments for setting component visibility.

    Attributes
    ----------
    visible : bool
        Whether the component should be visible.
    value : Any
        The value of the component.

    """

    visible: bool
    value: Any


class UpdateDropdownKwArgs(TypedDict, total=False):
    """
    Keyword arguments for updating a dropdown component.

    Attributes
    ----------
    choices : DropdownChoices
        The updated choices for the dropdown component.
    value : DropdownValue
        The updated value for the dropdown component.

    """

    choices: DropdownChoices
    value: DropdownValue


class TextBoxKwArgs(TypedDict, total=False):
    """
    Keyword arguments for updating a textbox component.

    Attributes
    ----------
    value : str | None
        The updated value for the textbox component.
    placeholder : str | None
        The updated placeholder for the textbox component.

    """

    value: str | None
    placeholder: str | None


class UpdateAudioKwArgs(TypedDict, total=False):
    """
    Keyword arguments for updating an audio component.

    Attributes
    ----------
    value : str | None
        The updated value for the audio component.

    """

    value: str | None
    type: Literal["filepath", "numpy"]
    interactive: bool


class DatasetType(StrEnum):
    """The type of dataset to train a voice model."""

    NEW_DATASET = "New dataset"
    EXISTING_DATASET = "Existing dataset"


class BaseComponentConfig[T: Component](BaseModel):
    """Configuration for a component in the UI."""

    label: str = Field(default="", frozen=True)
    info: str = Field(default="", frozen=True)
    render: bool = Field(default=True, frozen=True)
    scale: int | None = Field(default=None, frozen=True)
    visible: bool = Field(default=True, frozen=False)
    _instance: T | None = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def instance(self) -> T:
        if self._instance is None:
            raise ValueError("Component instance has not been instantiated yet.")
        return self._instance


class ComponentConfig[U, T: Component](BaseComponentConfig[T]):
    """Configuration for a component in the UI."""

    value: U


class ImmutableComponentConfig[U, T: Component](BaseComponentConfig[T]):
    value: U = Field(frozen=True)


class SliderConfig(ComponentConfig[float, gr.Slider]):
    minimum: float = Field(default=0.0, frozen=True)
    maximum: float = Field(default=1.0, frozen=True)
    step: float | None = Field(default=None, frozen=True)

    def instantiate(self) -> None:
        self._instance = gr.Slider(
            minimum=self.minimum,
            maximum=self.maximum,
            step=self.step,
            label=self.label,
            info=self.info,
            value=self.value,
            show_reset_button=False,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
        )


class CheckboxConfig(ComponentConfig[bool, gr.Checkbox]):
    def instantiate(self) -> None:
        self._instance = gr.Checkbox(
            label=self.label,
            info=self.info,
            value=self.value,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
        )


class NumberConfig(ComponentConfig[int, gr.Number]):
    precision: int | None = Field(default=None, frozen=True)

    def instantiate(self) -> None:
        self._instance = gr.Number(
            precision=self.precision,
            label=self.label,
            info=self.info,
            value=self.value,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
        )


class RadioConfig(ComponentConfig[str | int, gr.Radio]):
    choices: Sequence[str | int] | None = Field(default=None, frozen=True)

    def instantiate(self) -> None:
        self._instance = gr.Radio(
            choices=self.choices,
            label=self.label,
            info=self.info,
            value=self.value,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
        )


class DropdownConfig(ComponentConfig[str | list[str] | int, gr.Dropdown]):
    choices: Sequence[str | int] | None = Field(default=None, frozen=True)
    multiselect: bool = Field(default=False, frozen=True)

    def instantiate(self) -> None:
        self._instance = gr.Dropdown(
            choices=self.choices,
            multiselect=self.multiselect,
            label=self.label,
            info=self.info,
            value=self.value,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
        )


class ImmutableDropdownConfig(
    ImmutableComponentConfig[str | DefaultValue | None, gr.Dropdown],
):
    render: bool = Field(default=False, frozen=True)
    choices: Sequence[str | int | tuple[str, str | int]] | None = Field(
        default=None,
        frozen=True,
    )
    multiselect: bool = Field(default=False, frozen=True)
    allow_custom_value: bool = Field(default=False, frozen=True)

    def instantiate(
        self,
        choices: Sequence[str | int | tuple[str, str | int]] | None = None,
    ) -> None:
        self._instance = gr.Dropdown(
            choices=choices,
            multiselect=self.multiselect,
            allow_custom_value=self.allow_custom_value,
            label=self.label,
            info=self.info,
            value=self.value,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
        )


class ImmutableTextboxConfig(ImmutableComponentConfig[str | None, gr.Textbox]):
    placeholder: str | None = Field(default=None, frozen=True)

    def instantiate(
        self,
        value: Callable[..., Any] | None = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
    ) -> None:
        self._instance = gr.Textbox(
            label=self.label,
            info=self.info,
            value=value or self.value,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
            inputs=inputs,
        )


class ImmutableAudioConfig(ImmutableComponentConfig[str | Path | None, gr.Audio]):
    value: str | Path | None = Field(default=None, frozen=True)
    interactive: bool | None = Field(default=None, frozen=True)

    def instantiate(self) -> None:
        self._instance = gr.Audio(
            label=self.label,
            value=self.value,
            type="filepath",
            interactive=self.interactive,
            render=self.render,
            visible=self.visible,
            scale=self.scale,
        )


class SongDirComponentConfig(ImmutableDropdownConfig):
    value: str | DefaultValue | None = Field(default=DEFAULT_VALUE, frozen=True)
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
        value=0.7,
        minimum=0.0,
        maximum=1.0,
        step=0.1,
        visible=False,
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
        minimum=1,
        maximum=512,
        step=1,
    )
    embedder_model: DropdownConfig = DropdownConfig(
        label="Embedder model",
        info="The model to use for generating speaker embeddings.",
        choices=list(EmbedderModel),
        value=EmbedderModel.CONTENTVEC,
    )
    custom_embedder_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Custom embedder model",
        info="Select a custom embedder model from the dropdown.",
        value=DEFAULT_VALUE,
        visible=False,
        render=False,
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
        choices=list(F0Method),
        value=[F0Method.RMVPE],
        multiselect=True,
    )
    index_rate: SliderConfig = SliderConfig(
        label="Index rate",
        info=(
            "Increase to bias the conversion towards the accent of the"
            " voice model. Decrease to potentially reduce artifacts"
            " coming from the voice model.<br><br><br>"
        ),
        minimum=0.0,
        maximum=1.0,
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
        minimum=0.0,
        maximum=1.0,
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
        minimum=0.0,
        maximum=0.5,
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
        minimum=0.0,
        maximum=1.0,
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
        choices=list(SampleRate),
        value=SampleRate.HZ_44100,
    )
    output_format: DropdownConfig = DropdownConfig(
        label="Output format",
        info="The audio format of the mixed output track.",
        choices=list(AudioExt),
        value=AudioExt.MP3,
    )


class SongGenerationConfig(GenerationConfig):
    """Configuration options for song generation."""

    source: ImmutableTextboxConfig = ImmutableTextboxConfig(
        label="Source",
        info="Link to a song on YouTube or the full path of a local audio file.",
        value="",
    )
    voice_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Voice model",
        info="Select a voice model to use for converting vocals.",
        value=None,
    )
    cached_song: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Source",
        info="Select a song from the list of cached songs.",
        value=DEFAULT_VALUE,
        visible=False,
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
        value=0.75,
        minimum=0.0,
        maximum=1.0,
    )
    main_gain: SliderConfig = SliderConfig(
        label="Main gain",
        info="The gain to apply to the main vocals.",
        value=0,
        minimum=-20,
        maximum=20,
        step=1,
    )
    inst_gain: SliderConfig = SliderConfig(
        label="Instrumentals gain",
        info="The gain to apply to the instrumentals.",
        value=0,
        minimum=-20,
        maximum=20,
        step=1,
    )
    backup_gain: SliderConfig = SliderConfig(
        label="Backup gain",
        info="The gain to apply to the backup vocals.",
        value=0,
        minimum=-20,
        maximum=20,
        step=1,
    )

    output_name: ImmutableTextboxConfig = ImmutableTextboxConfig(
        label="Output name",
        info="If no name is provided, a suitable name will be generated automatically.",
        placeholder="Ultimate RVC song cover",
        value=None,
    )


class OneClickGenerationConfig(SongGenerationConfig):

    n_octaves: SliderConfig = SliderConfig(
        label="Vocal pitch shift",
        info=(
            "The number of octaves to shift the pitch of the converted"
            " vocals by. Use 1 for male-to-female and -1 for vice-versa."
        ),
        value=0,
        minimum=-3,
        maximum=3,
        step=1,
    )

    n_semitones: SliderConfig = SliderConfig(
        label="Overall pitch shift",
        info=(
            "The number of semi-tones to shift the pitch of the converted"
            " vocals, instrumentals and backup vocals by."
        ),
        value=0,
        minimum=-12,
        maximum=12,
        step=1,
    )

    show_intermediate_audio: CheckboxConfig = CheckboxConfig(
        label="Show intermediate audio",
        info="Show intermediate audio tracks generated during song cover generation.",
        value=False,
    )
    song: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Song",
        interactive=False,
    )
    vocals_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Vocals",
        render=False,
    )
    instrumentals_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Instrumentals",
        render=False,
    )
    main_vocals_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Main vocals",
        render=False,
    )
    backup_vocals_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Backup vocals",
        render=False,
    )
    main_vocals_dereverbed_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="De-reverbed main vocals",
        render=False,
    )
    main_vocals_reverb_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Main vocals with reverb",
        render=False,
    )
    converted_vocals_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Converted vocals",
        render=False,
    )
    postprocessed_vocals_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Postprocessed vocals",
        render=False,
    )
    instrumentals_shifted_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Pitch-shifted instrumentals",
        render=False,
    )
    backup_vocals_shifted_track: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Pitch-shifted backup vocals",
        render=False,
    )

    @property
    def intermediate_audio_tracks(self) -> list[gr.Audio]:
        """List of intermediate audio tracks."""
        return [
            self.song.instance,
            self.vocals_track.instance,
            self.instrumentals_track.instance,
            self.main_vocals_track.instance,
            self.backup_vocals_track.instance,
            self.main_vocals_dereverbed_track.instance,
            self.main_vocals_reverb_track.instance,
            self.converted_vocals_track.instance,
            self.postprocessed_vocals_track.instance,
            self.instrumentals_shifted_track.instance,
            self.backup_vocals_shifted_track.instance,
        ]


class MultiStepGenerationConfig(SongGenerationConfig):
    separation_model: DropdownConfig = DropdownConfig(
        choices=list(SeparationModel),
        value=SeparationModel.UVR_MDX_NET_VOC_FT,
        label="Separation model",
        info="The model to use for audio separation.",
    )
    segment_size: RadioConfig = RadioConfig(
        choices=list(SegmentSize),
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
        minimum=-3,
        maximum=3,
        step=1,
    )
    n_semitones: SliderConfig = SliderConfig(
        label="Pitch shift (semi-tones)",
        info=(
            "The number of semi-tones to pitch-shift the converted vocals"
            " by. Altering this slightly reduces sound quality."
        ),
        value=0,
        minimum=-12,
        maximum=12,
        step=1,
    )

    n_semitones_instrumentals: SliderConfig = SliderConfig(
        value=0,
        minimum=-12,
        maximum=12,
        label="Instrumental pitch shift",
        info="The number of semi-tones to pitch-shift the instrumentals by",
    )
    n_semitones_backup_vocals: SliderConfig = SliderConfig(
        value=0,
        minimum=-12,
        maximum=12,
        label="Backup vocal pitch shift",
        info="The number of semi-tones to pitch-shift the backup vocals by",
    )
    separate_audio_dir: SongDirComponentConfig = SongDirComponentConfig()
    convert_vocals_dir: SongDirComponentConfig = SongDirComponentConfig()
    postprocess_vocals_dir: SongDirComponentConfig = SongDirComponentConfig()
    pitch_shift_background_dir: SongDirComponentConfig = SongDirComponentConfig()
    mix_dir: SongDirComponentConfig = SongDirComponentConfig()
    audio_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Audio",
        render=False,
    )
    vocals_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Vocals",
        render=False,
    )
    converted_vocals_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Vocals",
        render=False,
    )
    instrumentals_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Instrumentals",
        render=False,
    )
    backup_vocals_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Backup vocals",
        render=False,
    )
    main_vocals_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Main vocals",
        render=False,
    )
    shifted_instrumentals_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Instrumentals",
        render=False,
    )
    shifted_backup_vocals_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Backup vocals",
        render=False,
    )

    @property
    def song_dirs(self) -> list[gr.Dropdown]:
        """List of song directories."""
        return [
            self.separate_audio_dir.instance,
            self.convert_vocals_dir.instance,
            self.postprocess_vocals_dir.instance,
            self.pitch_shift_background_dir.instance,
            self.mix_dir.instance,
        ]

    @property
    def input_tracks(self) -> list[ImmutableAudioConfig]:
        """List of input audio tracks."""
        return [
            self.audio_track_input,
            self.vocals_track_input,
            self.converted_vocals_track_input,
            self.instrumentals_track_input,
            self.backup_vocals_track_input,
            self.main_vocals_track_input,
            self.shifted_instrumentals_track_input,
            self.shifted_backup_vocals_track_input,
        ]


class SpeechGenerationConfig(GenerationConfig):
    """Configuration options for speech generation."""

    source: ImmutableTextboxConfig = ImmutableTextboxConfig(
        label="Source",
        info="Text to generate speech from",
        value="",
    )
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
        minimum=-3,
        maximum=3,
        step=1,
    )
    n_semitones: SliderConfig = SliderConfig(
        label="Semitone shift",
        info="The number of semi-tones to pitch-shift the converted speech by.",
        value=0,
        minimum=-12,
        maximum=12,
        step=1,
    )
    tts_pitch_shift: SliderConfig = SliderConfig(
        label="Edge TTS pitch shift",
        info=(
            "The number of hertz to shift the pitch of the speech generated"
            " by Edge TTS."
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
    split_voice: CheckboxConfig = CheckboxConfig(
        label="Split input voice",
        info=(
            "Whether to split the input voice track into smaller segments"
            " before converting it. This can improve output quality for"
            " longer input voice tracks."
        ),
        value=True,
    )
    clean_voice: CheckboxConfig = CheckboxConfig(
        label="Clean converted voice",
        info=(
            "Whether to clean the converted voice using noise reduction"
            " algorithms.<br><br>"
        ),
        value=True,
    )
    clean_strength: SliderConfig = SliderConfig(
        label="Cleaning intensity",
        info=(
            "Higher values result in stronger cleaning, but may lead to a more"
            " compressed sound."
        ),
        value=0.7,
        minimum=0.0,
        maximum=1.0,
        step=0.1,
        visible=True,
    )
    output_gain: SliderConfig = SliderConfig(
        label="Output gain",
        info="The gain to apply to the converted speech.<br><br>",
        value=0,
        minimum=-20,
        maximum=20,
        step=1,
    )
    output_name: ImmutableTextboxConfig = ImmutableTextboxConfig(
        label="Output name",
        info="If no name is provided, a suitable name will be generated automatically.",
        placeholder="Ultimate RVC speech",
        value=None,
    )


class OneClickSpeechGenerationConfig(SpeechGenerationConfig):
    show_intermediate_audio: CheckboxConfig = CheckboxConfig(
        label="Show intermediate audio",
        info="Show intermediate audio tracks generated during speech generation.",
        value=False,
    )


class MultiStepSpeechGenerationConfig(SpeechGenerationConfig):

    speech_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Speech",
        render=False,
    )
    converted_speech_track_input: ImmutableAudioConfig = ImmutableAudioConfig(
        label="Converted speech",
        render=False,
    )

    @property
    def input_tracks(self) -> list[ImmutableAudioConfig]:
        """List of input audio tracks."""
        return [
            self.speech_track_input,
            self.converted_speech_track_input,
        ]


class TrainingConfig(BaseTabConfig):
    """Configuration options for training."""

    dataset: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Dataset path",
        info=(
            "The path to an existing dataset. Either select a path to a previously"
            " created dataset or provide a path to an external dataset."
        ),
        value=DEFAULT_VALUE,
        visible=False,
        allow_custom_value=True,
    )
    preprocess_model: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Model name",
        info=(
            "Name of the model to preprocess the given dataset for. Either select"
            " an existing model from the dropdown or provide the name of a new"
            " model."
        ),
        value="My model",
        allow_custom_value=True,
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
        choices=list(TrainingSampleRate),
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
        choices=list(AudioSplitMethod),
        value=AudioSplitMethod.AUTOMATIC,
    )
    chunk_len: SliderConfig = SliderConfig(
        label="Chunk length",
        info="Length of split audio chunks.",
        minimum=0.5,
        maximum=5.0,
        step=0.1,
        value=3.0,
        visible=False,
    )
    overlap_len: SliderConfig = SliderConfig(
        label="Overlap length",
        info="Length of overlap between split audio chunks.",
        minimum=0.0,
        maximum=0.4,
        value=0.3,
        step=0.1,
        visible=False,
    )
    f0_method: DropdownConfig = DropdownConfig(
        label="F0 method",
        info="The method to use for extracting pitch features.",
        choices=list(TrainingF0Method),
        value=TrainingF0Method.RMVPE,
    )

    hop_length: SliderConfig = SliderConfig(
        label="Hop length",
        info="The hop length to use for extracting pitch features.<br><br>",
        value=128,
        minimum=1,
        maximum=512,
        step=1,
        visible=False,
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
        minimum=0,
        maximum=10,
        value=2,
    )
    extraction_acceleration: DropdownConfig = DropdownConfig(
        choices=list(DeviceType),
        value=DeviceType.AUTOMATIC,
        label="Hardware acceleration",
        info=(
            "The type of hardware acceleration to use for feature"
            " extraction. 'Automatic' will automatically select the"
            " first available GPU and fall back to CPU if no GPUs"
            " are available."
        ),
    )
    extraction_gpus: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="GPU(s)",
        info="The GPU(s) to use for feature extraction.",
        value=DEFAULT_VALUE,
        render=True,
        multiselect=True,
        visible=False,
    )
    num_epochs: SliderConfig = SliderConfig(
        label="Number of epochs",
        info=(
            "The number of epochs to train the voice model. A higher"
            " number can improve voice model performance but may lead"
            " to overtraining."
        ),
        minimum=1,
        maximum=1000,
        step=1,
        value=500,
    )
    batch_size: SliderConfig = SliderConfig(
        label="Batch size",
        info=(
            "The number of samples in each training batch. It is"
            " advisable to align this value with the available VRAM of"
            " your GPU."
        ),
        minimum=1,
        maximum=64,
        value=8,
        step=1,
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
        minimum=1,
        maximum=100,
        value=50,
        visible=False,
    )
    vocoder: DropdownConfig = DropdownConfig(
        label="Vocoder",
        info=(
            "The vocoder to use for audio synthesis during"
            " training. HiFi-GAN provides basic audio fidelity,"
            " while RefineGAN provides the highest audio fidelity."
        ),
        value=Vocoder.HIFI_GAN,
        choices=list(Vocoder),
    )
    index_algorithm: DropdownConfig = DropdownConfig(
        label="Index algorithm",
        info=(
            "The method to use for generating an index file for the"
            " trained voice model. KMeans is particularly useful"
            " for large datasets."
        ),
        value=IndexAlgorithm.AUTO,
        choices=list(IndexAlgorithm),
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
        choices=list(PretrainedType),
        value=PretrainedType.DEFAULT,
    )
    save_interval: SliderConfig = SliderConfig(
        label="Save interval",
        info=(
            "The epoch interval at which to to save voice model"
            " weights and checkpoints. The best model weights are"
            " always saved regardless of this setting."
        ),
        minimum=1,
        maximum=100,
        step=1,
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
    upload_name: ImmutableTextboxConfig = ImmutableTextboxConfig(
        label="Upload name",
        info="The name to give the uploaded voice model.",
        visible=False,
        value=None,
    )
    training_acceleration: DropdownConfig = DropdownConfig(
        label="Hardware acceleration",
        info=(
            "The type of hardware acceleration to use when training"
            " the voice model. 'Automatic' will select the first"
            " available GPU and fall back to CPU if no GPUs are"
            " available."
        ),
        choices=list(DeviceType),
        value=DeviceType.AUTOMATIC,
    )
    training_gpus: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="GPU(s)",
        info="The GPU(s) to use for training the voice model.",
        value=DEFAULT_VALUE,
        render=True,
        multiselect=True,
        visible=False,
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


class ManageModelConfig(BaseModel):
    voice_models: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Voice models",
        value=DEFAULT_VALUE,
        multiselect=True,
    )
    embedders: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Custom embedder models",
        value=DEFAULT_VALUE,
        multiselect=True,
    )
    pretrained_models: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Custom pretrained models",
        value=DEFAULT_VALUE,
        multiselect=True,
    )
    training_models: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Training models",
        value=DEFAULT_VALUE,
        multiselect=True,
    )


class ManageAudioConfig(BaseModel):
    intermediate_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Song directories",
        info=(
            "Select one or more song directories containing intermediate audio"
            " files to delete."
        ),
        value=DEFAULT_VALUE,
        multiselect=True,
    )
    speech_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Speech audio files",
        info="Select one or more speech audio files to delete.",
        value=DEFAULT_VALUE,
        multiselect=True,
    )
    output_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Output audio files",
        info="Select one or more output audio files to delete.",
        value=DEFAULT_VALUE,
        multiselect=True,
    )
    dataset_audio: ImmutableDropdownConfig = ImmutableDropdownConfig(
        label="Dataset audio files",
        info="Select one or more datasets containing audio files to delete.",
        value=DEFAULT_VALUE,
        multiselect=True,
    )
    dummy_checkbox: CheckboxConfig = CheckboxConfig(
        value=False,
        visible=False,
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
    model: ManageModelConfig = ManageModelConfig()
    audio: ManageAudioConfig = ManageAudioConfig()


class TotalConfig(BaseModel):
    song: TotalSongGenerationConfig = TotalSongGenerationConfig()
    speech: TotalSpeechGenerationConfig = TotalSpeechGenerationConfig()
    training: TotalTrainingConfig = TotalTrainingConfig()
    management: TotalManagementConfig = TotalManagementConfig()
