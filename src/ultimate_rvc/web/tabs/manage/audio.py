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
from ultimate_rvc.web.common import (
    confirm_box_js,
    confirmation_harness,
    render_msg,
    update_dropdowns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def render(
    intermediate_audio: gr.Dropdown,
    speech_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
    dataset_audio: gr.Dropdown,
    cached_song_1click: gr.Dropdown,
    cached_song_multi: gr.Dropdown,
    song_dirs: Sequence[gr.Dropdown],
    dataset: gr.Dropdown,
) -> None:
    """
    Render "Manage audio" tab.

    Parameters
    ----------
    intermediate_audio : gr.Dropdown
        Dropdown for selecting intermediate audio files to delete in the
        "Delete audio" tab.
    speech_audio : gr.Dropdown
        Dropdown for selecting speech audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.
    dataset_audio : gr.Dropdown
        Dropdown for selecting dataset audio files to delete in the
        "Delete audio" tab.
    cached_song_1click : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Generate song covers - one-click generation" tab
    cached_song_multi : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Generate song covers - multi-step generation" tab
    song_dirs : Sequence[gr.Dropdown]
        Dropdown components for selecting song directories in the
        "Generate song covers - multi-step generation" tab.
    dataset : gr.Dropdown
        Dropdown to display available datasets in the "Train models -
        multi-step generation" tab.

    """
    dummy_checkbox = gr.Checkbox(visible=False)
    with gr.Tab("Delete audio"):
        with gr.Accordion("Intermediate audio", open=False), gr.Row():
            with gr.Column():
                intermediate_audio.render()
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
        with gr.Accordion("Speech audio", open=False), gr.Row():
            with gr.Column():
                speech_audio.render()
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
        with gr.Accordion("Output audio", open=False), gr.Row():
            with gr.Column():
                output_audio.render()
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
        with gr.Accordion("Dataset audio", open=False), gr.Row():
            with gr.Column():
                dataset_audio.render()
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

        with gr.Accordion("All audio", open=True), gr.Row(equal_height=True):
            all_audio_btn = gr.Button("Delete", variant="primary")
            all_audio_msg = gr.Textbox(label="Output message", interactive=False)

        intermediate_audio_click = intermediate_audio_btn.click(
            confirmation_harness(delete_intermediate_audio),
            inputs=[dummy_checkbox, intermediate_audio],
            outputs=intermediate_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected song directories?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted the selected song directories!",
            ),
            outputs=intermediate_audio_msg,
            show_progress="hidden",
        )

        all_intermediate_audio_click = all_intermediate_audio_btn.click(
            confirmation_harness(delete_all_intermediate_audio),
            inputs=dummy_checkbox,
            outputs=intermediate_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete all intermediate audio files?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted all intermediate audio files!",
            ),
            outputs=intermediate_audio_msg,
            show_progress="hidden",
        )

        speech_audio_click = speech_audio_btn.click(
            confirmation_harness(delete_speech_audio),
            inputs=[dummy_checkbox, speech_audio],
            outputs=speech_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected speech audio files?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted the selected speech audio files!",
            ),
            outputs=speech_audio_msg,
            show_progress="hidden",
        )

        all_speech_audio_click = all_speech_audio_btn.click(
            confirmation_harness(delete_all_speech_audio),
            inputs=dummy_checkbox,
            outputs=speech_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete all speech audio files?",
            ),
        ).success(
            partial(render_msg, "[-] Successfully deleted all speech audio files!"),
            outputs=speech_audio_msg,
            show_progress="hidden",
        )

        output_audio_click = output_audio_btn.click(
            confirmation_harness(delete_output_audio),
            inputs=[dummy_checkbox, output_audio],
            outputs=output_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected output audio files?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted the selected output audio files!",
            ),
            outputs=output_audio_msg,
            show_progress="hidden",
        )

        all_output_audio_click = all_output_audio_btn.click(
            confirmation_harness(delete_all_output_audio),
            inputs=dummy_checkbox,
            outputs=output_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete all output audio files?",
            ),
        ).success(
            partial(render_msg, "[-] Successfully deleted all output audio files!"),
            outputs=output_audio_msg,
            show_progress="hidden",
        )

        dataset_audio_click = dataset_audio_btn.click(
            confirmation_harness(delete_dataset_audio),
            inputs=[dummy_checkbox, dataset_audio],
            outputs=dataset_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete the selected datasets?",
            ),
        ).success(
            partial(
                render_msg,
                "[-] Successfully deleted the selected datasets!",
            ),
            outputs=dataset_audio_msg,
            show_progress="hidden",
        )

        all_dataset_audio_click = all_dataset_audio_btn.click(
            confirmation_harness(delete_all_dataset_audio),
            inputs=dummy_checkbox,
            outputs=dataset_audio_msg,
            js=confirm_box_js(
                "Are you sure you want to delete all dataset audio files?",
            ),
        ).success(
            partial(render_msg, "[-] Successfully deleted all dataset audio files!"),
            outputs=dataset_audio_msg,
            show_progress="hidden",
        )

        all_audio_click = all_audio_btn.click(
            confirmation_harness(delete_all_audio),
            inputs=dummy_checkbox,
            outputs=all_audio_msg,
            js=confirm_box_js("Are you sure you want to delete all audio files?"),
        ).success(
            partial(render_msg, "[-] Successfully deleted all audio files!"),
            outputs=all_audio_msg,
            show_progress="hidden",
        )

        _, _, all_audio_update = [
            click_event.success(
                partial(
                    update_dropdowns,
                    get_named_song_dirs,
                    3 + len(song_dirs),
                    [],
                    [0],
                ),
                outputs=[
                    intermediate_audio,
                    cached_song_1click,
                    cached_song_multi,
                    *song_dirs,
                ],
                show_progress="hidden",
            )
            for click_event in [
                intermediate_audio_click,
                all_intermediate_audio_click,
                all_audio_click,
            ]
        ]

        _, _, all_audio_update = [
            click_event.success(
                partial(update_dropdowns, get_saved_speech_audio, 1, [], [0]),
                outputs=speech_audio,
                show_progress="hidden",
            )
            for click_event in [
                speech_audio_click,
                all_speech_audio_click,
                all_audio_update,
            ]
        ]

        _, _, all_audio_update = [
            click_event.success(
                partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
                outputs=output_audio,
                show_progress="hidden",
            )
            for click_event in [
                output_audio_click,
                all_output_audio_click,
                all_audio_update,
            ]
        ]

        for click_event in [
            dataset_audio_click,
            all_dataset_audio_click,
            all_audio_update,
        ]:
            click_event.success(
                partial(update_dropdowns, get_named_audio_datasets, 1, [], [0]),
                outputs=dataset_audio,
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_audio_datasets, 1, [], [0]),
                outputs=dataset,
                show_progress="hidden",
            )
