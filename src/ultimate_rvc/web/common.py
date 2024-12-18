"""
Module defining common utility functions and classes for the
web application of the Ultimate RVC project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Concatenate

import gradio as gr

from ultimate_rvc.core.exceptions import NotProvidedError
from ultimate_rvc.core.generate.song_cover import (
    get_named_song_dirs,
    get_song_cover_name,
)
from ultimate_rvc.core.generate.speech import get_speech_track_name
from ultimate_rvc.core.manage.audio import (
    get_saved_output_audio,
    get_saved_speech_audio,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ultimate_rvc.web.typing_extra import (
        ComponentVisibilityKwArgs,
        DropdownChoices,
        DropdownValue,
        TextBoxKwArgs,
        UpdateAudioKwArgs,
        UpdateDropdownKwArgs,
    )

PROGRESS_BAR = gr.Progress()


def exception_harness[T, **P](
    fn: Callable[P, T],
    info_msg: str | None = None,
) -> Callable[P, T]:
    """
    Wrap a function in a harness that catches exceptions and re-raises
    them as instances of `gradio.Error`.

    Parameters
    ----------
    fn : Callable[P, T]
        The function to wrap.

    info_msg : str, optional
        Message to display in an info-box pop-up after the function
        executes successfully.

    Returns
    -------
    Callable[P, T]
        The wrapped function.

    """

    def _wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            res = fn(*args, **kwargs)
        except gr.Error:
            raise
        except NotProvidedError as e:
            msg = e.ui_msg or e
            raise gr.Error(str(msg)) from None
        except Exception as e:
            raise gr.Error(str(e)) from e
        else:
            if info_msg:
                gr.Info(info_msg, duration=1)
            return res

    return _wrapped_fn


def confirmation_harness[T, **P](
    fn: Callable[P, T],
) -> Callable[Concatenate[bool, P], T]:
    """
    Wrap a function in a harness that requires a confirmation before
    executing and catches exceptions, re-raising them as instances of
    `gradio.Error`.

    Parameters
    ----------
    fn : Callable[P, T]
        The function to wrap.

    Returns
    -------
    Callable[Concatenate[bool, P], T]
        The wrapped function.

    """

    def _wrapped_fn(confirm: bool, *args: P.args, **kwargs: P.kwargs) -> T:
        if confirm:
            return exception_harness(fn)(*args, **kwargs)
        err_msg = "Confirmation missing!"
        raise gr.Error(err_msg)

    return _wrapped_fn


def render_msg(
    template: str,
    *args: str,
    display_info: bool = False,
    **kwargs: str,
) -> str:
    """
    Render a message template with the provided arguments.

    Parameters
    ----------
    template : str
        Message template to render.
    args : str
        Positional arguments to pass to the template.
    display_info : bool, default=False
        Whether to display the rendered message as an info message
        in addition to returning it.
    kwargs : str
        Keyword arguments to pass to the template.

    Returns
    -------
    str
        Rendered message.

    """
    msg = template.format(*args, **kwargs)
    if display_info:
        gr.Info(msg)
    return msg


def confirm_box_js(msg: str) -> str:
    """
    Generate a JavaScript code snippet which:
      * defines an anonymous function that takes one named parameter and
      zero or more unnamed parameters
      * renders a confirmation box
      * returns the choice selected by the user in that confirmation
      box in addition to any unnamed parameters passed to the function.

    Parameters
    ----------
    msg : str
        Message to display in the confirmation box rendered by the
        JavaScript code snippet.

    Returns
    -------
    str
        The JavaScript code snippet.

    """
    return f"(x, ...args) => [confirm('{msg}'), ...args]"


def update_value(x: str) -> dict[str, Any]:
    """
    Update the value of a component.

    Parameters
    ----------
    x : str
        New value for the component.

    Returns
    -------
    dict[str, Any]
        Dictionary which updates the value of the component.

    """
    return gr.update(value=x)


def toggle_visibility[T](value: T, target: T) -> dict[str, Any]:
    """
    Toggle the visibility of a component based on equality of
    a value and a target.

    Parameters
    ----------
    value : T
        The value to compare against the target.
    target : T
        The target to compare the value against.

    Returns
    -------
    dict[str, Any]
        Dictionary which updates the visibility of the component.

    """
    return gr.update(visible=value == target)


def update_dropdowns[**P](
    fn: Callable[P, DropdownChoices],
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
    *args: P.args,
    **kwargs: P.kwargs,
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices and optionally the value of one or more dropdown
    components.

    Parameters
    ----------
    fn : Callable[P, DropdownChoices]
        Function to get updated choices for the dropdown components.
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.
    args : P.args
        Positional arguments to pass to the function used to update
        dropdown choices.
    kwargs : P.kwargs
        Keyword arguments to pass to the function used to update
        dropdown choices.

    Returns
    -------
    gr.Dropdown | tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    Raises
    ------
    ValueError
        If not all provided indices are unique or if an index exceeds
        or is equal to the number of dropdown components.

    """
    if len(value_indices) != len(set(value_indices)):
        err_msg = "Value indices must be unique."
        raise ValueError(err_msg)
    if value_indices and max(value_indices) >= num_components:
        err_msg = (
            "Index of a dropdown component to update the value for exceeds the number"
            " of dropdown components to update."
        )
        raise ValueError(err_msg)
    updated_choices = fn(*args, **kwargs)
    update_args_list: list[UpdateDropdownKwArgs] = [
        {"choices": updated_choices} for _ in range(num_components)
    ]
    for index in value_indices:
        update_args_list[index]["value"] = value

    match update_args_list:
        case [update_args]:
            # NOTE This is a workaround as gradio does not support
            # singleton tuples for components.
            return gr.Dropdown(**update_args)
        case _:
            return tuple(gr.Dropdown(**update_args) for update_args in update_args_list)


def update_cached_songs(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices of one or more dropdown components to the set of
    currently cached songs.

    Optionally update the default value of one or more of these
    components.

    Parameters
    ----------
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for the dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.

    Returns
    -------
    gr.Dropdown | tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    """
    return update_dropdowns(get_named_song_dirs, num_components, value, value_indices)


def update_output_audio(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices of one or more dropdown components to the set of
    currently saved output audio files.

    Optionally update the default value of one or more of these
    components.

    Parameters
    ----------
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.

    Returns
    -------
    gr.Dropdown | tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    """
    return update_dropdowns(
        get_saved_output_audio,
        num_components,
        value,
        value_indices,
    )


def update_speech_audio(
    num_components: int,
    value: DropdownValue = None,
    value_indices: Sequence[int] = [],
) -> gr.Dropdown | tuple[gr.Dropdown, ...]:
    """
    Update the choices of one or more dropdown components to the set of
    currently saved speech audio files.

    Optionally update the default value of one or more of these
    components.

    Parameters
    ----------
    num_components : int
        Number of dropdown components to update.
    value : DropdownValue, optional
        New value for dropdown components.
    value_indices : Sequence[int], default=[]
        Indices of dropdown components to update the value for.

    Returns
    -------
    gr.Dropdown | tuple[gr.Dropdown,...]
        Updated dropdown component or components.

    """
    return update_dropdowns(
        get_saved_speech_audio,
        num_components,
        value,
        value_indices,
    )


def toggle_visible_component(
    num_components: int,
    visible_index: int,
) -> dict[str, Any] | tuple[dict[str, Any], ...]:
    """
    Reveal a single component from a set of components. All other
    components are hidden.

    Parameters
    ----------
    num_components : int
        Number of components to set visibility for.
    visible_index : int
        Index of the component to reveal.

    Returns
    -------
    dict[str, Any] | tuple[dict[str, Any], ...]
        A single dictionary or a tuple of dictionaries that update the
        visibility of the components.

    Raises
    ------
    ValueError
        If the visible index exceeds or is equal to the number of
        components to set visibility for.

    """
    if visible_index >= num_components:
        err_msg = (
            "Visible index must be less than the number of components to set visibility"
            " for."
        )
        raise ValueError(err_msg)
    update_args_list: list[ComponentVisibilityKwArgs] = [
        {"visible": False, "value": None} for _ in range(num_components)
    ]
    update_args_list[visible_index]["visible"] = True
    match update_args_list:
        case [update_args]:
            return gr.update(**update_args)
        case _:
            return tuple(gr.update(**update_args) for update_args in update_args_list)


def toggle_intermediate_audio(
    visible: bool,
    num_components: int,
) -> list[gr.Accordion]:
    """
    Toggle the visibility of intermediate audio accordions.

    Parameters
    ----------
    visible : bool
        Visibility status of the intermediate audio accordions.

    num_components : int
        Number of intermediate audio accordions to toggle visibility
        for.

    Returns
    -------
    list[gr.Accordion]
        The intermediate audio accordions.

    """
    accordions = [gr.Accordion(open=False) for _ in range(num_components)]
    return [gr.Accordion(visible=visible, open=False), *accordions]


def update_song_cover_name(
    effected_vocals_track: str | None = None,
    song_dir: str | None = None,
    model_name: str | None = None,
    update_placeholder: bool = False,
) -> gr.Textbox:
    """
    Update a textbox component so that it displays a suitable name for a
    cover of a given song.

    If the path of an existing song directory is provided, the name of
    the song is inferred from that directory. If the name of a voice
    model is not provided but the path of an existing song directory
    and the path of an effected vocals track in that directory are
    provided, then the voice model is inferred from the effected vocals
    track.


    Parameters
    ----------
    effected_vocals_track : str, optional
        The path to an effected vocals track.
    song_dir : str, optional
        The path to a song directory.
    model_name : str, optional
        The name of a voice model.
    update_placeholder : bool, default=False
        Whether to update the placeholder text instead of the value of
        the textbox component.

    Returns
    -------
    gr.Textbox
        Textbox component with updated value or placeholder text.

    """
    update_args: TextBoxKwArgs = {}
    update_key = "placeholder" if update_placeholder else "value"
    if effected_vocals_track or song_dir or model_name:
        song_cover_name = get_song_cover_name(
            effected_vocals_track,
            song_dir,
            model_name,
        )
        update_args[update_key] = song_cover_name
    else:
        update_args[update_key] = None
    return gr.Textbox(**update_args)


def update_speech_track_name(
    text: str | None = None,
    model_name: str | None = None,
    converted_voice_track: str | None = None,
    update_placeholder: bool = False,
) -> gr.Textbox:
    """
    Update a textbox component so that it displays a suitable name for a
    speech track generated from a given text source.

    Parameters
    ----------
    text : str, optional
        The text to generate a speech track from.
    model_name : str, optional
        The name of the voice model to use for converting the speech
        generated from text.
    converted_voice_track : str, optional
        The path to the voice track generated from the speech generated
        from text.
    update_placeholder : bool, default=False
        Whether to update the placeholder text instead of the value of
        the textbox component.

    Returns
    -------
    gr.Textbox
        Textbox component with updated value or placeholder text.

    """
    update_args: TextBoxKwArgs = {}
    update_key = "placeholder" if update_placeholder else "value"
    if text or model_name or converted_voice_track:
        speech_track_name = get_speech_track_name(
            source=text,
            model_name=model_name,
            converted_voice_track=converted_voice_track,
        )
        update_args[update_key] = speech_track_name
    else:
        update_args[update_key] = None
    return gr.Textbox(**update_args)


def update_audio(
    num_components: int,
    output_indices: Sequence[int],
    track: str | None,
    disallow_none: bool = True,
) -> gr.Audio | tuple[gr.Audio, ...]:
    """
    Update the value of a subset of `Audio` components to the given
    audio track.

    Parameters
    ----------
    num_components : int
        The total number of `Audio` components under consideration.
    output_indices : Sequence[int]
        Indices of `Audio` components to update the value for.
    track : str
        Path pointing to an audio track to update the value of the
        indexed `Audio` components with.
    disallow_none : bool, default=True
        Whether to disallow the value of the indexed components to be
        `None`.

    Returns
    -------
    gr.Audio | tuple[gr.Audio, ...]
        Each `Audio` component under consideration with the value of the
        indexed components updated to the given audio track.

    """
    update_args_list: list[UpdateAudioKwArgs] = [{} for _ in range(num_components)]
    for index in output_indices:
        if track or not disallow_none:
            update_args_list[index]["value"] = track
    match update_args_list:
        case [update_args]:
            return gr.Audio(**update_args)
        case _:
            return tuple(gr.Audio(**update_args) for update_args in update_args_list)
