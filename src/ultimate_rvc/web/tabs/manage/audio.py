"""Module which defines the code for the "Manage audio" tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import partial

import gradio as gr

from ultimate_rvc.core.generate.song_cover import get_named_song_dirs
from ultimate_rvc.core.manage.audio import (
    delete_all_audio,
    delete_all_dataset_audio,
    delete_all_intermediate_audio,
    delete_all_output_audio,
    delete_all_speech_audio,
    delete_dataset_audio,
    delete_intermediate_audio,
    delete_output_audio,
    delete_speech_audio,
    get_audio_datasets,
    get_named_audio_datasets,
    get_saved_output_audio,
    get_saved_speech_audio,
)
from ultimate_rvc.web.common import setup_delete_event, update_dropdowns
from ultimate_rvc.web.typing_extra import ManageAudioEventState

if TYPE_CHECKING:

    from ultimate_rvc.web.typing_extra import ManageAudioConfig, TotalConfig


def render(total_config: TotalConfig) -> None:
    """
    Render "Manage audio" tab.

    Parameters
    ----------
    total_config: TotalConfig
        The total configuration object.
        It contains all the configuration objects for the project.

    """
    tab_config = total_config.management.audio
    tab_config.dummy_checkbox.instantiate()
    event_state = ManageAudioEventState()

    with gr.Tab("Delete audio"):
        _render_intermediate_audio_accordion(tab_config, event_state)
        _render_speech_audio_accordion(tab_config, event_state)
        _render_output_audio_accordion(tab_config, event_state)
        _render_dataset_audio_accordion(tab_config, event_state)
        _render_all_audio_accordion(tab_config, event_state)

        _, _, all_audio_update = [
            click_event.success(
                partial(
                    update_dropdowns,
                    get_named_song_dirs,
                    3 + len(total_config.song.multi_step.song_dirs),
                    [],
                    [0],
                ),
                outputs=[
                    tab_config.intermediate_audio.instance,
                    total_config.song.one_click.cached_song.instance,
                    total_config.song.multi_step.cached_song.instance,
                    *total_config.song.multi_step.song_dirs,
                ],
                show_progress="hidden",
            )
            for click_event in [
                event_state.delete_intermediate_click.instance,
                event_state.delete_all_intermediate_click.instance,
                event_state.delete_all_click.instance,
            ]
        ]

        _, _, all_audio_update = [
            click_event.success(
                partial(update_dropdowns, get_saved_speech_audio, 1, [], [0]),
                outputs=tab_config.speech_audio.instance,
                show_progress="hidden",
            )
            for click_event in [
                event_state.delete_speech_click.instance,
                event_state.delete_all_speech_click.instance,
                all_audio_update,
            ]
        ]

        _, _, all_audio_update = [
            click_event.success(
                partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
                outputs=tab_config.output_audio.instance,
                show_progress="hidden",
            )
            for click_event in [
                event_state.delete_output_click.instance,
                event_state.delete_all_output_click.instance,
                all_audio_update,
            ]
        ]

        for click_event in [
            event_state.delete_dataset_click.instance,
            event_state.delete_all_dataset_click.instance,
            all_audio_update,
        ]:
            click_event.success(
                partial(update_dropdowns, get_named_audio_datasets, 1, [], [0]),
                outputs=tab_config.dataset_audio.instance,
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_audio_datasets, 1, [], [0]),
                outputs=total_config.training.multi_step.dataset.instance,
                show_progress="hidden",
            )


def _render_intermediate_audio_accordion(
    tab_config: ManageAudioConfig,
    event_state: ManageAudioEventState,
) -> None:
    with gr.Accordion("Intermediate audio", open=False), gr.Row():
        with gr.Column():
            tab_config.intermediate_audio.instance.render()
            intermediate_audio_btn = gr.Button(
                "Delete selected",
                variant="secondary",
            )
            all_intermediate_audio_btn = gr.Button(
                "Delete all",
                variant="primary",
            )
        with gr.Column():
            intermediate_audio_msg = gr.Textbox(
                label="Output message",
                interactive=False,
            )
        event_state.delete_intermediate_click.instance = setup_delete_event(
            intermediate_audio_btn,
            delete_intermediate_audio,
            [
                tab_config.dummy_checkbox.instance,
                tab_config.intermediate_audio.instance,
            ],
            intermediate_audio_msg,
            "Are you sure you want to delete the selected song directories?",
            "[-] Successfully deleted the selected song directories!",
        )
        event_state.delete_all_intermediate_click.instance = setup_delete_event(
            all_intermediate_audio_btn,
            delete_all_intermediate_audio,
            [tab_config.dummy_checkbox.instance],
            intermediate_audio_msg,
            "Are you sure you want to delete all intermediate audio files?",
            "[-] Successfully deleted all intermediate audio files!",
        )


def _render_speech_audio_accordion(
    tab_config: ManageAudioConfig,
    event_state: ManageAudioEventState,
) -> None:
    with gr.Accordion("Speech audio", open=False), gr.Row():
        with gr.Column():
            tab_config.speech_audio.instance.render()
            speech_audio_btn = gr.Button(
                "Delete selected",
                variant="secondary",
            )
            all_speech_audio_btn = gr.Button(
                "Delete all",
                variant="primary",
            )
        with gr.Column():
            speech_audio_msg = gr.Textbox(
                label="Output message",
                interactive=False,
            )

        event_state.delete_speech_click.instance = setup_delete_event(
            speech_audio_btn,
            delete_speech_audio,
            [tab_config.dummy_checkbox.instance, tab_config.speech_audio.instance],
            speech_audio_msg,
            "Are you sure you want to delete the selected speech audio files?",
            "[-] Successfully deleted the selected speech audio files!",
        )

        event_state.delete_all_speech_click.instance = setup_delete_event(
            all_speech_audio_btn,
            delete_all_speech_audio,
            [tab_config.dummy_checkbox.instance],
            speech_audio_msg,
            "Are you sure you want to delete all speech audio files?",
            "[-] Successfully deleted all speech audio files!",
        )


def _render_output_audio_accordion(
    tab_config: ManageAudioConfig,
    event_state: ManageAudioEventState,
) -> None:

    with gr.Accordion("Output audio", open=False), gr.Row():
        with gr.Column():
            tab_config.output_audio.instance.render()
            output_audio_btn = gr.Button(
                "Delete selected",
                variant="secondary",
            )
            all_output_audio_btn = gr.Button(
                "Delete all",
                variant="primary",
            )
        with gr.Column():
            output_audio_msg = gr.Textbox(
                label="Output message",
                interactive=False,
            )
        event_state.delete_output_click.instance = setup_delete_event(
            output_audio_btn,
            delete_output_audio,
            [tab_config.dummy_checkbox.instance, tab_config.output_audio.instance],
            output_audio_msg,
            "Are you sure you want to delete the selected output audio files?",
            "[-] Successfully deleted the selected output audio files!",
        )
        event_state.delete_all_output_click.instance = setup_delete_event(
            all_output_audio_btn,
            delete_all_output_audio,
            [tab_config.dummy_checkbox.instance],
            output_audio_msg,
            "Are you sure you want to delete all output audio files?",
            "[-] Successfully deleted all output audio files!",
        )


def _render_dataset_audio_accordion(
    tab_config: ManageAudioConfig,
    event_state: ManageAudioEventState,
) -> None:

    with gr.Accordion("Dataset audio", open=False), gr.Row():
        with gr.Column():
            tab_config.dataset_audio.instance.render()
            dataset_audio_btn = gr.Button(
                "Delete selected",
                variant="secondary",
            )
            all_dataset_audio_btn = gr.Button(
                "Delete all",
                variant="primary",
            )
        with gr.Column():
            dataset_audio_msg = gr.Textbox(
                label="Output message",
                interactive=False,
            )

        event_state.delete_dataset_click.instance = setup_delete_event(
            dataset_audio_btn,
            delete_dataset_audio,
            [tab_config.dummy_checkbox.instance, tab_config.dataset_audio.instance],
            dataset_audio_msg,
            "Are you sure you want to delete the selected dataset audio files?",
            "[-] Successfully deleted the selected dataset audio files!",
        )
        event_state.delete_all_dataset_click.instance = setup_delete_event(
            all_dataset_audio_btn,
            delete_all_dataset_audio,
            [tab_config.dummy_checkbox.instance],
            dataset_audio_msg,
            "Are you sure you want to delete all dataset audio files?",
            "[-] Successfully deleted all dataset audio files!",
        )


def _render_all_audio_accordion(
    tab_config: ManageAudioConfig,
    event_state: ManageAudioEventState,
) -> None:

    with gr.Accordion("All audio", open=True), gr.Row(equal_height=True):
        all_audio_btn = gr.Button("Delete", variant="primary")
        all_audio_msg = gr.Textbox(label="Output message", interactive=False)

    event_state.delete_all_click.instance = setup_delete_event(
        all_audio_btn,
        delete_all_audio,
        [tab_config.dummy_checkbox.instance],
        all_audio_msg,
        "Are you sure you want to delete all audio files?",
        "[-] Successfully deleted all audio files!",
    )
