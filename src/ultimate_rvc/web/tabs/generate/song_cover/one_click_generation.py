"""
Module which defines the code for the
"Generate song covers - one-click generation" tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import partial

import gradio as gr

from ultimate_rvc.core.generate.song_cover import (
    get_named_song_dirs,
    get_song_cover_name,
    run_pipeline,
)
from ultimate_rvc.core.manage.audio import get_saved_output_audio
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method, SampleRate
from ultimate_rvc.web.common import (
    PROGRESS_BAR,
    exception_harness,
    toggle_intermediate_audio,
    toggle_visibility,
    toggle_visible_component,
    update_dropdowns,
    update_output_name,
    update_value,
)
from ultimate_rvc.web.typing_extra import ConcurrencyId, SongSourceType

if TYPE_CHECKING:
    from collections.abc import Sequence


def render(
    voice_model: gr.Dropdown,
    custom_embedder_model: gr.Dropdown,
    cached_song_1click: gr.Dropdown,
    cached_song_multi: gr.Dropdown,
    song_dirs: Sequence[gr.Dropdown],
    intermediate_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
    cookiefile: str | None = None,
) -> None:
    """
    Render "Generate song covers - One-click generation" tab.

    Parameters
    ----------
    voice_model : gr.Dropdown
        Dropdown for selecting voice model in the
        "Generate song cover - One-click generation" tab.
    custom_embedder_model : gr.Dropdown
        Dropdown for selecting custom embedder model in the
        "Generate song cover - One-click generation" tab.
    cached_song_1click : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Generate song cover - One-click generation" tab
    cached_song_multi : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Generate song cover - Multi-step generation" tab
    song_dirs : Sequence[gr.Dropdown]
        Dropdowns for selecting song directories in the
        "Generate song cover - Multi-step generation" tab.
    intermediate_audio : gr.Dropdown
        Dropdown for selecting intermediate audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.
    cookiefile : str, optional
        The path to a file containing cookies to use when downloading
        audio from Youtube.

    """
    with gr.Tab("One-click generation"):
        with gr.Accordion("Main options"):
            with gr.Row():
                with gr.Column():
                    source_type = gr.Dropdown(
                        list(SongSourceType),
                        value=SongSourceType.PATH,
                        label="Source type",
                        type="index",
                        info="The type of source to retrieve a song from.",
                    )
                with gr.Column():
                    source = gr.Textbox(
                        label="Source",
                        info=(
                            "Link to a song on YouTube or the full path of a local"
                            " audio file."
                        ),
                    )
                    local_file = gr.Audio(
                        label="Source",
                        type="filepath",
                        visible=False,
                    )
                    cached_song_1click.render()
                source_type.input(
                    partial(toggle_visible_component, 3),
                    inputs=source_type,
                    outputs=[source, local_file, cached_song_1click],
                    show_progress="hidden",
                )

                local_file.change(
                    update_value,
                    inputs=local_file,
                    outputs=source,
                    show_progress="hidden",
                )
                cached_song_1click.input(
                    update_value,
                    inputs=cached_song_1click,
                    outputs=source,
                    show_progress="hidden",
                )
            with gr.Row():
                voice_model.render()
                n_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Vocal pitch shift",
                    info=(
                        "The number of octaves to shift the pitch of the converted"
                        " vocals by. Use 1 for male-to-female and -1 for vice-versa."
                    ),
                    show_reset_button=False,
                )
                n_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Overall pitch shift",
                    info=(
                        "The number of semi-tones to shift the pitch of the converted"
                        " vocals, instrumentals and backup vocals by."
                    ),
                    show_reset_button=False,
                )

        with gr.Accordion("Vocal conversion options", open=False):
            gr.Markdown("")
            with gr.Accordion("Voice synthesis settings", open=False):
                with gr.Row():
                    f0_methods = gr.Dropdown(
                        list(F0Method),
                        value=F0Method.RMVPE,
                        label="Pitch extraction algorithm(s)",
                        info=(
                            "If more than one method is selected, then the median of"
                            " the pitch values extracted by each method is used. RMVPE"
                            " is recommended for most cases and is the default when no"
                            " method is selected."
                        ),
                        multiselect=True,
                    )
                    index_rate = gr.Slider(
                        0,
                        1,
                        value=0.5,
                        label="Index rate",
                        info=(
                            "Increase to bias the conversion towards the accent of the"
                            " voice model. Decrease to potentially reduce artifacts"
                            " coming from the voice model.<br><br><br>"
                        ),
                        show_reset_button=False,
                    )
                with gr.Row():
                    rms_mix_rate = gr.Slider(
                        0,
                        1,
                        value=0.25,
                        label="RMS mix rate",
                        info=(
                            "How much to mimic the loudness (0) of the input vocals or"
                            " a fixed loudness (1). A value of 0.25 is recommended for"
                            " most cases."
                        ),
                        show_reset_button=False,
                    )
                    protect_rate = gr.Slider(
                        0,
                        0.5,
                        value=0.33,
                        label="Protect rate",
                        info=(
                            "Controls the extent to which consonants and breathing"
                            " sounds are protected from artifacts. A higher value"
                            " offers more protection but may worsen the indexing"
                            " effect."
                        ),
                        show_reset_button=False,
                    )
                    hop_length = gr.Slider(
                        1,
                        512,
                        value=128,
                        step=1,
                        label="Hop length",
                        info=(
                            "How often the CREPE-based pitch extraction method checks"
                            " for pitch changes measured in milliseconds. Lower values"
                            " lead to longer conversion times and a higher risk of"
                            " voice cracks, but better pitch accuracy."
                        ),
                        show_reset_button=False,
                    )
            with gr.Accordion("Vocal enrichment settings", open=False):
                with gr.Row():
                    with gr.Column():
                        split_vocals = gr.Checkbox(
                            label="Split vocals track",
                            info=(
                                "Whether to split the vocals track into smaller"
                                " segments before converting it. This can improve"
                                " output quality for longer vocal tracks."
                            ),
                        )
                    with gr.Column():
                        autotune_vocals = gr.Checkbox(
                            label="Autotune converted vocals",
                            info=(
                                "Whether to apply autotune to the converted"
                                " vocals.<br><br>"
                            ),
                        )
                        autotune_strength = gr.Slider(
                            0,
                            1,
                            value=1.0,
                            label="Autotune intensity",
                            info=(
                                "Higher values result in stronger snapping to the"
                                " chromatic grid and artifacting."
                            ),
                            visible=False,
                            show_reset_button=False,
                        )
                    with gr.Column():
                        clean_vocals = gr.Checkbox(
                            label="Clean converted vocals",
                            info=(
                                "Whether to clean the converted vocals using noise"
                                " reduction algorithms.<br><br>"
                            ),
                        )
                        clean_strength = gr.Slider(
                            0,
                            1,
                            value=0.7,
                            label="Cleaning intensity",
                            info=(
                                "Higher values result in stronger cleaning, but may"
                                " lead to a more compressed sound."
                            ),
                            visible=False,
                            show_reset_button=False,
                        )
                autotune_vocals.change(
                    partial(toggle_visibility, targets={True}, default=1.0),
                    inputs=autotune_vocals,
                    outputs=autotune_strength,
                    show_progress="hidden",
                )
                clean_vocals.change(
                    partial(toggle_visibility, targets={True}, default=0.7),
                    inputs=clean_vocals,
                    outputs=clean_strength,
                    show_progress="hidden",
                )
            with gr.Accordion("Speaker embedding settings", open=False):
                with gr.Row():
                    with gr.Column():
                        embedder_model = gr.Dropdown(
                            list(EmbedderModel),
                            value=EmbedderModel.CONTENTVEC,
                            label="Embedder model",
                            info="The model to use for generating speaker embeddings.",
                        )
                        custom_embedder_model.render()
                    sid = gr.Number(
                        label="Speaker ID",
                        info="Speaker ID for multi-speaker-models.",
                        precision=0,
                    )
                embedder_model.change(
                    partial(toggle_visibility, targets={EmbedderModel.CUSTOM}),
                    inputs=embedder_model,
                    outputs=custom_embedder_model,
                    show_progress="hidden",
                )
        with gr.Accordion("Audio mixing options", open=False):
            gr.Markdown("")
            with gr.Accordion("Reverb control on converted vocals", open=False):
                with gr.Row():
                    room_size = gr.Slider(
                        0,
                        1,
                        value=0.15,
                        label="Room size",
                        info=(
                            "Size of the room which reverb effect simulates. Increase"
                            " for longer reverb time."
                        ),
                        show_reset_button=False,
                    )
                with gr.Row():
                    wet_level = gr.Slider(
                        0,
                        1,
                        value=0.2,
                        label="Wetness level",
                        info="Loudness of converted vocals with reverb effect applied.",
                        show_reset_button=False,
                    )
                    dry_level = gr.Slider(
                        0,
                        1,
                        value=0.8,
                        label="Dryness level",
                        info=(
                            "Loudness of converted vocals without reverb effect"
                            " applied."
                        ),
                        show_reset_button=False,
                    )
                    damping = gr.Slider(
                        0,
                        1,
                        value=0.7,
                        label="Damping level",
                        info="Absorption of high frequencies in reverb effect.",
                        show_reset_button=False,
                    )

            with gr.Accordion("Volume controls (dB)", open=False), gr.Row():
                main_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Main vocals",
                    show_reset_button=False,
                )
                inst_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Instrumentals",
                    show_reset_button=False,
                )
                backup_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Backup vocals",
                    show_reset_button=False,
                )
        with gr.Accordion("Audio output options", open=False):
            with gr.Row():
                output_name = gr.Textbox(
                    value=partial(
                        update_output_name,
                        get_song_cover_name,
                        True,  # noqa: FBT003,,
                    ),
                    inputs=[cached_song_1click, cached_song_1click, voice_model],
                    label="Output name",
                    info=(
                        "If no name is provided, a suitable name will be generated"
                        " automatically."
                    ),
                    placeholder="Ultimate RVC song cover",
                )
                output_sr = gr.Dropdown(
                    choices=list(SampleRate),
                    value=SampleRate.HZ_44100,
                    label="Output sample rate",
                    info="The sample rate to save the generated song cover in.",
                )
                output_format = gr.Dropdown(
                    list(AudioExt),
                    value=AudioExt.MP3,
                    label="Output format",
                    info="The format to save the generated song cover in.",
                )
            with gr.Row():
                show_intermediate_audio = gr.Checkbox(
                    label="Show intermediate audio",
                    value=False,
                    info=(
                        "Show intermediate audio tracks generated during song cover"
                        " generation."
                    ),
                )

        intermediate_audio_accordions = [
            gr.Accordion(label, open=False, render=False)
            for label in [
                "Step 0: song retrieval",
                "Step 1a: vocals/instrumentals separation",
                "Step 1b: main vocals/ backup vocals separation",
                "Step 1c: main vocals cleanup",
                "Step 2: conversion of main vocals",
                "Step 3: post-processing of converted vocals",
                "Step 4: pitch shift of background tracks",
            ]
        ]
        (
            song_retrieval_accordion,
            vocals_separation_accordion,
            main_vocals_separation_accordion,
            vocal_cleanup_accordion,
            vocal_conversion_accordion,
            vocals_postprocessing_accordion,
            pitch_shift_accordion,
        ) = intermediate_audio_accordions
        intermediate_audio_tracks = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Song",
                "Vocals",
                "Instrumentals",
                "Main vocals",
                "Backup vocals",
                "De-reverbed main vocals",
                "Main vocals reverb",
                "Converted vocals",
                "Post-processed vocals",
                "Pitch-shifted instrumentals",
                "Pitch-shifted backup vocals",
            ]
        ]
        (
            song,
            vocals_track,
            instrumentals_track,
            main_vocals_track,
            backup_vocals_track,
            main_vocals_dereverbed_track,
            main_vocals_reverb_track,
            converted_vocals_track,
            postprocessed_vocals_track,
            instrumentals_shifted_track,
            backup_vocals_shifted_track,
        ) = intermediate_audio_tracks
        with gr.Accordion(
            "Intermediate audio tracks",
            open=False,
            visible=False,
        ) as intermediate_audio_accordion:
            song_retrieval_accordion.render()
            with song_retrieval_accordion:
                song.render()
            vocals_separation_accordion.render()
            with vocals_separation_accordion, gr.Row():
                vocals_track.render()
                instrumentals_track.render()
            main_vocals_separation_accordion.render()
            with main_vocals_separation_accordion, gr.Row():
                main_vocals_track.render()
                backup_vocals_track.render()
            vocal_cleanup_accordion.render()
            with vocal_cleanup_accordion, gr.Row():
                main_vocals_dereverbed_track.render()
                main_vocals_reverb_track.render()
            vocal_conversion_accordion.render()
            with vocal_conversion_accordion:
                converted_vocals_track.render()
            vocals_postprocessing_accordion.render()
            with vocals_postprocessing_accordion:
                postprocessed_vocals_track.render()
            pitch_shift_accordion.render()
            with pitch_shift_accordion, gr.Row():
                instrumentals_shifted_track.render()
                backup_vocals_shifted_track.render()

        show_intermediate_audio.change(
            partial(toggle_intermediate_audio, num_components=7),
            inputs=show_intermediate_audio,
            outputs=[
                intermediate_audio_accordion,
                *intermediate_audio_accordions,
            ],
            show_progress="hidden",
        )

        with gr.Row(equal_height=True):
            reset_btn = gr.Button(value="Reset settings", scale=2)
            generate_btn = gr.Button("Generate", scale=2, variant="primary")
            song_cover = gr.Audio(label="Song cover", scale=3)

        generate_btn.click(
            partial(
                exception_harness(
                    run_pipeline,
                    info_msg="Song cover generated successfully!",
                ),
                cookiefile=cookiefile,
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[
                source,
                voice_model,
                n_octaves,
                n_semitones,
                f0_methods,
                index_rate,
                rms_mix_rate,
                protect_rate,
                hop_length,
                split_vocals,
                autotune_vocals,
                autotune_strength,
                clean_vocals,
                clean_strength,
                embedder_model,
                custom_embedder_model,
                sid,
                room_size,
                wet_level,
                dry_level,
                damping,
                main_gain,
                inst_gain,
                backup_gain,
                output_sr,
                output_format,
                output_name,
            ],
            outputs=[song_cover, *intermediate_audio_tracks],
            concurrency_limit=1,
            concurrency_id=ConcurrencyId.GPU,
        ).success(
            partial(update_dropdowns, get_named_song_dirs, 3 + len(song_dirs), [], [2]),
            outputs=[
                cached_song_1click,
                cached_song_multi,
                intermediate_audio,
                *song_dirs,
            ],
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
            outputs=output_audio,
            show_progress="hidden",
        )
        reset_btn.click(
            lambda: [
                0,
                0,
                F0Method.RMVPE,
                0.5,
                0.25,
                0.33,
                128,
                False,
                False,
                1.0,
                False,
                0.7,
                EmbedderModel.CONTENTVEC,
                None,
                0,
                0.15,
                0.2,
                0.8,
                0.7,
                0,
                0,
                0,
                SampleRate.HZ_44100,
                AudioExt.MP3,
                False,
            ],
            outputs=[
                n_octaves,
                n_semitones,
                f0_methods,
                index_rate,
                rms_mix_rate,
                protect_rate,
                hop_length,
                split_vocals,
                autotune_vocals,
                autotune_strength,
                clean_vocals,
                clean_strength,
                embedder_model,
                custom_embedder_model,
                sid,
                room_size,
                wet_level,
                dry_level,
                damping,
                main_gain,
                inst_gain,
                backup_gain,
                output_sr,
                output_format,
                show_intermediate_audio,
            ],
            show_progress="hidden",
        )
