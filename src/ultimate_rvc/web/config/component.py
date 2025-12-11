"""
Module defining models for representing UI component
configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    SerializerFunctionWrapHandler,
    model_serializer,
)

import gradio as gr
from gradio.components import Component

from ultimate_rvc.core.exceptions import ComponentNotInstatiatedError
from ultimate_rvc.typing_extra import DeviceType
from ultimate_rvc.web.typing_extra import (
    AnyCallable,
    BaseDropdownValue,
    BaseRadioValue,
    DropdownChoices,
    RadioChoices,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

U = TypeVar("U")
T = TypeVar("T", bound=Component)


# NOTE: pydantic does not yet seem to be fully compatible with the
# new generics syntax (3.12+) so we use the old one
class ComponentConfig(BaseModel, Generic[U, T]):  # noqa: UP046
    """
    Base model defining common fields and logic for configuration and
    storage of a UI component.

    Attributes
    ----------
    label : str | None, default=None
        The label of the component.
    value : U
        The default value of the component.
    visible : bool, default=True
        Whether the component is visible.
    scale : int | None, default=None
        The scale of the component.
    render : bool, default=True
        Whether to render the component when instantiated.
    exclude_value : bool, default=False
        If True, the default value of the component cannot be updated.
    _instance: T | None, default=None
        Internal attribute storing the component instance. Will be null
        until the component is instantiated.
    instance: T
        The component instance. Attempting to access this field before
        the component is instantiated will raise a NotInstantiatedError.

    """

    label: str | None = None
    value: U
    visible: bool = True
    scale: int | None = None
    render: bool = True
    exclude_value: bool = False
    _instance: T | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def instance(self) -> T:
        """
        Property which returns the component instance.

        Raises
        ------
        ComponentNotInstatiatedError
            If the component has not been instantiated yet.

        """
        if self._instance is None:
            raise ComponentNotInstatiatedError
        return self._instance

    @model_serializer(mode="wrap")
    def serialize(self, nxt: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """
        Serialize the component configuration to a dictionary using a
        custom field order.

        Parameters
        ----------
        nxt : SerializerFunctionWrapHandler
            The main serialization function to apply to the component
            configuration before applying the custom field order.

        Returns
        -------
        dict[str, Any]
            The serialized component configuration.

        """
        field_order = [
            "label",
            "info",
            "value",
            "minimum",
            "maximum",
            "step",
            "precision",
            "choices",
            "multiselect",
            "allow_custom_value",
            "type",
            "visible",
            "scale",
            "render",
            "exclude_value",
        ]
        config = nxt(self)

        ordered_mapping = {key: config[key] for key in field_order if key in config}
        remaining_mapping = {
            key: value for key, value in config.items() if key not in field_order
        }
        return ordered_mapping | remaining_mapping


type AnyComponentConfig = ComponentConfig[Any, Any]


class InfoComponentConfig(ComponentConfig[U, T], Generic[U, T]):  # noqa: UP046
    """
    Configuration settings for a component with an information text.

    Attributes
    ----------
    info : str | None, default=None
        The information text on the component.

    See Also
    --------
    ComponentConfig
        Parent model defining common configuration settings for a UI
        component.

    """

    info: str | None = None


class SliderConfig(InfoComponentConfig[float | None, gr.Slider]):
    """
    Configuration settings for a slider component.

    Attributes
    ----------
    minimum : float, default=0
        The minimum value of the slider component.
    maximum : float, default=100
        The maximum value of the slider component.
    step : float | None, default=None
        The step size of the slider component.

    See Also
    --------
    InfoComponentConfig
        Parent model defining common configuration settings for a UI
        component with an information text.

    """

    minimum: float = 0
    maximum: float = 100
    step: float | None = None

    def instantiate(
        self,
        maximum: float | None = None,
        value: float | None = None,
    ) -> None:
        """
        Instantiate the slider component.

        Parameters
        ----------
        maximum : float | None, default=None
            The maximum value of the slider component. If not provided,
            the maximum value specified in the configuration is used.
        value : float | None, default=None
            The initial value of the slider component. If not provided,
            the value specified in the configuration is used.

        """
        self._instance = gr.Slider(
            label=self.label,
            info=self.info,
            value=self.value if value is None else value,
            minimum=self.minimum,
            maximum=self.maximum if maximum is None else maximum,
            step=self.step,
            visible=self.visible,
            scale=self.scale,
            render=self.render,
            show_reset_button=False,
        )

    @classmethod
    def octave_shift(cls, label: str, info: str) -> SliderConfig:
        """
        Create a slider configuration for octave pitch shift.

        Parameters
        ----------
        label : str
            The label for the octave pitch shift slider.
        info : str
            The information text for the octave pitch shift slider.

        Returns
        -------
        SliderConfig
            A slider configuration for octave pitch shift.

        """
        return cls(label=label, info=info, value=0, minimum=-3, maximum=3, step=1)

    @classmethod
    def semitone_shift(cls, label: str, info: str) -> SliderConfig:
        """
        Create a slider configuration for semitone pitch shift.

        Parameters
        ----------
        label : str
            The label for the semitone pitch shift slider.
        info : str
            The information text for the semitone pitch shift slider.

        Returns
        -------
        SliderConfig
            A slider configuration for semitone pitch shift.

        """
        return cls(label=label, info=info, value=0, minimum=-12, maximum=12, step=1)

    @classmethod
    def clean_strength(cls, visible: bool) -> SliderConfig:
        """
        Create a slider configuration for clean strength.

        Parameters
        ----------
        visible : bool
            Whether the slider should be visible.

        Returns
        -------
        SliderConfig
            A slider configuration for clean strength.

        """
        return cls(
            label="Cleaning intensity",
            info=(
                "Higher values result in stronger cleaning, but may lead to a more"
                " compressed sound."
            ),
            value=0.7,
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            visible=visible,
        )

    @classmethod
    def gain(cls, label: str, info: str) -> SliderConfig:
        """
        Create a slider configuration for gain.

        Parameters
        ----------
        label : str
            The label for the gain slider.
        info : str
            The information text for the gain slider.

        Returns
        -------
        SliderConfig
            A slider configuration for gain.

        """
        return cls(label=label, info=info, value=0, minimum=-20, maximum=20, step=1)

    @classmethod
    def cpu_cores(cls) -> SliderConfig:
        """
        Create a slider configuration for CPU cores.

        Returns
        -------
        SliderConfig
            A slider configuration for CPU cores.

        """
        return cls(
            label="CPU cores",
            info="The number of CPU cores to use for multi-threading.",
            value=None,
            minimum=1,
            maximum=1,
            step=1,
            exclude_value=True,
        )


class CheckboxConfig(InfoComponentConfig[bool, gr.Checkbox]):
    """
    Configuration settings for a checkbox component.

    See Also
    --------
    InfoComponentConfig
        Parent model defining common configuration settings for a UI
        component with an information text.

    """

    def instantiate(self) -> None:
        """Instantiate the checkbox component."""
        self._instance = gr.Checkbox(
            label=self.label,
            info=self.info,
            value=self.value,
            visible=self.visible,
            scale=self.scale,
            render=self.render,
        )


class NumberConfig(InfoComponentConfig[int | None, gr.Number]):
    """
    Configuration settings for a number component.

    Attributes
    ----------
    precision : int | None, default=None
        The number of decimal places to display for the number
        component.

    See Also
    --------
    InfoComponentConfig
        Parent model defining common configuration settings for a UI
        component with an information text.

    """

    precision: int | None = None

    def instantiate(self) -> None:
        """Instantiate the number component."""
        self._instance = gr.Number(
            label=self.label,
            info=self.info,
            value=self.value,
            precision=self.precision,
            visible=self.visible,
            scale=self.scale,
            render=self.render,
        )


class RadioConfig(InfoComponentConfig[BaseRadioValue, gr.Radio]):
    """
    Configuration settings for a radio component.

    Attributes
    ----------
    choices : BaseRadioValue, default=None
        The selectable choices for the radio component.

    See Also
    --------
    InfoComponentConfig
        Parent model defining common configuration settings for a UI
        component with an information text.

    """

    choices: RadioChoices = None

    def instantiate(self) -> None:
        """Instantiate the radio component."""
        self._instance = gr.Radio(
            label=self.label,
            info=self.info,
            value=self.value,
            choices=self.choices,
            visible=self.visible,
            scale=self.scale,
            render=self.render,
        )


class DropdownConfig(InfoComponentConfig[BaseDropdownValue, gr.Dropdown]):
    """
    Configuration settings for a dropdown component.

    Attributes
    ----------
    choices : DropdownChoices, default=None
        The selectable choices for the dropdown component.
    multiselect : bool | None, default=None
        Whether the dropdown component allows multiple selections.

    See Also
    --------
    ComponentConfig
        Parent model defining common configuration settings for a UI
        component.

    """

    choices: DropdownChoices = None
    multiselect: bool | None = None
    allow_custom_value: bool = False
    type: Literal["value", "index"] = "value"

    def instantiate(
        self,
        value: BaseDropdownValue = None,
        choices: DropdownChoices = None,
    ) -> None:
        """
        Instantiate the dropdown component.

        Parameters
        ----------
        value : BaseDropdownValue, default=None
            The initial value of the dropdown component. If not
            provided, the value specified in the configuration is used.
        choices : DropdownChoices, default=None
            Custom choices to instantiate the dropdown component with.
            If not provided, the dropdown will be instantiated with the
            choices specified in the configuration.

        """
        self._instance = gr.Dropdown(
            label=self.label,
            info=self.info,
            value=self.value if value is None else value,
            choices=self.choices if choices is None else choices,
            multiselect=self.multiselect,
            allow_custom_value=self.allow_custom_value,
            type=self.type,
            visible=self.visible,
            scale=self.scale,
            render=self.render,
        )

    @classmethod
    def hardware_acceleration(cls) -> DropdownConfig:
        """
        Create a dropdown configuration for hardware acceleration.

        Returns
        -------
        DropdownConfig
            A dropdown configuration for hardware acceleration.

        """
        return cls(
            label="Hardware acceleration",
            info=(
                "The type of hardware acceleration to use. 'Automatic' will"
                " automatically select the first available GPU and fall back to CPU if"
                " no GPUs are available."
            ),
            value=DeviceType.AUTOMATIC,
            choices=list(DeviceType),
            exclude_value=True,
        )

    @classmethod
    def song_dir(cls) -> DropdownConfig:
        """
        Create a song directory dropdown configuration.

        Returns
        -------
        DropdownConfig
            A song directory dropdown configuration.

        """
        return cls(
            label="Song directory",
            info=(
                "Directory where intermediate audio files are stored and loaded from"
                " locally. When a new song is retrieved, its directory is chosen by"
                " default."
            ),
            value=None,
            render=False,
            exclude_value=True,
        )

    @classmethod
    def gpu(cls) -> DropdownConfig:
        """
        Create a GPU dropdown configuration.

        Returns
        -------
        DropdownConfig
            A GPU dropdown configuration.

        """
        return cls(
            label="GPU(s)",
            info="The GPU(s) to use for hardware acceleration.",
            value=None,
            multiselect=True,
            visible=False,
            exclude_value=True,
        )

    @classmethod
    def multi_delete(cls, label: str, info: str) -> DropdownConfig:
        """
        Create a multi-delete dropdown configuration.

        Parameters
        ----------
        label : str
            The label for the multi-delete dropdown.
        info : str
            The information text for the multi-delete dropdown.

        Returns
        -------
        DropdownConfig
            A multi-delete dropdown configuration.

        """
        return cls(
            label=label,
            info=info,
            value=None,
            multiselect=True,
            render=False,
            exclude_value=True,
        )


class TextboxConfig(InfoComponentConfig[str | None, gr.Textbox]):
    """
    Configuration settings for a textbox component.

    Attributes
    ----------
    placeholder : str | None, default=None
        The placeholder text for the textbox component.

    See Also
    --------
    InfoComponentConfig
        Parent model defining common configuration settings for a UI
        component with an information text.

    """

    placeholder: str | None = None

    def instantiate(
        self,
        value: AnyCallable | None = None,
        inputs: Component | Sequence[Component] | set[Component] | None = None,
    ) -> None:
        """
        Instantiate the textbox component.

        Parameters
        ----------
        value : AnyCallable | None, default=None
            A custom function to instantiate the component's value with.
            Meant to be used in conjunction with the `inputs` parameter.
            If not provided, the textbox will be instantiated with the
            default value specified in the configuration.
        inputs : Component | Sequence[Component] | set[Component] | None
            A sequence of components whose values will be passed to the
            function specified in the `value` parameter. If not provided
            , has no effect.

        """
        self._instance = gr.Textbox(
            label=self.label,
            info=self.info,
            value=self.value if value is None else value,
            placeholder=self.placeholder,
            visible=self.visible,
            scale=self.scale,
            render=self.render,
            inputs=inputs,
        )


class AudioConfig(ComponentConfig[str | Path | None, gr.Audio]):
    """
    Configuration settings for an audio component.

    Attributes
    ----------
    interactive : bool | None, default=None
        Whether the audio component is interactive.

    See Also
    --------
    ComponentConfig
        Parent model defining common configuration settings for a UI
        component.

    """

    interactive: bool | None = None

    def instantiate(self) -> None:
        """Instantiate the audio component."""
        self._instance = gr.Audio(
            label=self.label,
            value=self.value,
            interactive=self.interactive,
            type="filepath",
            visible=self.visible,
            render=self.render,
            scale=self.scale,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )

    @classmethod
    def intermediate(cls, label: str) -> AudioConfig:
        """
        Create an intermediate audio configuration.

        Returns
        -------
        AudioConfig
            An intermediate audio configuration.

        """
        return cls(label=label, value=None, exclude_value=True)

    @classmethod
    def input(cls, label: str) -> AudioConfig:
        """
        Create an input audio configuration.

        Returns
        -------
        AudioConfig
            An input audio configuration.

        """
        return cls(label=label, value=None, render=False, exclude_value=True)
