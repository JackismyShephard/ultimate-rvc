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
from ultimate_rvc.typing_extra import EmbedderModel
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
from ultimate_rvc.web.typing_extra import ConcurrencyId

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import OneClickSongGenerationConfig, TotalConfig


def render(total_config: TotalConfig, cookiefile: str | None = None) -> None:
    """
    Render "Generate song covers - One-click generation" tab.

    Parameters
    ----------
    total_config : TotalConfig
        Model containing all component configuration settings for the
        Ultimate RVC web UI.
    cookiefile : str, optional
        The path to a file containing cookies to use when downloading
        audio from Youtube.

    """
    with gr.Tab("One-click"):
        tab_config = total_config.song.one_click
        _render_input(tab_config)
        with gr.Accordion("Options", open=False):
            _render_main_options(tab_config)
            _render_conversion_options(tab_config)
            _render_mixing_options(tab_config)
            _render_output_options(tab_config)
            _render_intermediate_audio(tab_config)

        with gr.Row(equal_height=True):
            reset_btn = gr.Button(value="Reset options", scale=2)
            generate_btn = gr.Button("Generate", scale=2, variant="primary")
        song_cover = gr.Audio(
            label="Song cover",
            scale=3,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )
        song_dirs = total_config.song.multi_step.song_dirs.all
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
                tab_config.source.instance,
                tab_config.voice_model.instance,
                tab_config.n_octaves.instance,
                tab_config.n_semitones.instance,
                tab_config.f0_method.instance,
                tab_config.index_rate.instance,
                tab_config.rms_mix_rate.instance,
                tab_config.protect_rate.instance,
                tab_config.split_voice.instance,
                tab_config.autotune_voice.instance,
                tab_config.autotune_strength.instance,
                tab_config.proposed_pitch.instance,
                tab_config.proposed_pitch_threshold.instance,
                tab_config.clean_voice.instance,
                tab_config.clean_strength.instance,
                tab_config.embedder_model.instance,
                tab_config.custom_embedder_model.instance,
                tab_config.sid.instance,
                tab_config.room_size.instance,
                tab_config.wet_level.instance,
                tab_config.dry_level.instance,
                tab_config.damping.instance,
                tab_config.main_gain.instance,
                tab_config.inst_gain.instance,
                tab_config.backup_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                tab_config.output_name.instance,
            ],
            outputs=[song_cover, *tab_config.intermediate_audio.all],
            concurrency_limit=1,
            concurrency_id=ConcurrencyId.GPU,
        ).success(
            partial(update_dropdowns, get_named_song_dirs, 3 + len(song_dirs), [], [2]),
            outputs=[
                total_config.song.one_click.cached_song.instance,
                total_config.song.multi_step.cached_song.instance,
                total_config.management.audio.intermediate.instance,
                *song_dirs,
            ],
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
            outputs=total_config.management.audio.output.instance,
            show_progress="hidden",
        )
        reset_btn.click(
            lambda: [
                tab_config.n_octaves.value,
                tab_config.n_semitones.value,
                tab_config.f0_method.value,
                tab_config.index_rate.value,
                tab_config.rms_mix_rate.value,
                tab_config.protect_rate.value,
                tab_config.split_voice.value,
                tab_config.autotune_voice.value,
                tab_config.autotune_strength.value,
                tab_config.proposed_pitch.value,
                tab_config.proposed_pitch_threshold.value,
                tab_config.clean_voice.value,
                tab_config.clean_strength.value,
                tab_config.embedder_model.value,
                tab_config.sid.value,
                tab_config.room_size.value,
                tab_config.wet_level.value,
                tab_config.dry_level.value,
                tab_config.damping.value,
                tab_config.main_gain.value,
                tab_config.inst_gain.value,
                tab_config.backup_gain.value,
                tab_config.output_sr.value,
                tab_config.output_format.value,
                tab_config.show_intermediate_audio.value,
            ],
            outputs=[
                tab_config.n_octaves.instance,
                tab_config.n_semitones.instance,
                tab_config.f0_method.instance,
                tab_config.index_rate.instance,
                tab_config.rms_mix_rate.instance,
                tab_config.protect_rate.instance,
                tab_config.split_voice.instance,
                tab_config.autotune_voice.instance,
                tab_config.autotune_strength.instance,
                tab_config.proposed_pitch.instance,
                tab_config.proposed_pitch_threshold.instance,
                tab_config.clean_voice.instance,
                tab_config.clean_strength.instance,
                tab_config.embedder_model.instance,
                tab_config.sid.instance,
                tab_config.room_size.instance,
                tab_config.wet_level.instance,
                tab_config.dry_level.instance,
                tab_config.damping.instance,
                tab_config.main_gain.instance,
                tab_config.inst_gain.instance,
                tab_config.backup_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                tab_config.show_intermediate_audio.instance,
            ],
            show_progress="hidden",
        )


def _render_input(tab_config: OneClickSongGenerationConfig) -> None:
    with gr.Row():
        with gr.Column():
            tab_config.source_type.instantiate()
        with gr.Column():
            tab_config.source.instantiate()
            local_file = gr.Audio(
                label="Source",
                type="filepath",
                visible=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )
            tab_config.cached_song.instance.render()
        tab_config.source_type.instance.input(
            partial(toggle_visible_component, 3),
            inputs=tab_config.source_type.instance,
            outputs=[
                tab_config.source.instance,
                local_file,
                tab_config.cached_song.instance,
            ],
            show_progress="hidden",
        )

        local_file.change(
            update_value,
            inputs=local_file,
            outputs=tab_config.source.instance,
            show_progress="hidden",
        )
        tab_config.cached_song.instance.input(
            update_value,
            inputs=tab_config.cached_song.instance,
            outputs=tab_config.source.instance,
            show_progress="hidden",
        )

    with gr.Row():
        tab_config.voice_model.instance.render()


def _render_main_options(tab_config: OneClickSongGenerationConfig) -> None:
    with gr.Row():
        tab_config.n_octaves.instantiate()
        tab_config.n_semitones.instantiate()


def _render_conversion_options(tab_config: OneClickSongGenerationConfig) -> None:
    with gr.Accordion("Vocal conversion", open=False):
        gr.Markdown("")
        with gr.Accordion("Voice synthesis", open=False):
            with gr.Row():
                tab_config.f0_method.instantiate()
                tab_config.index_rate.instantiate()
            with gr.Row():
                tab_config.rms_mix_rate.instantiate()
                tab_config.protect_rate.instantiate()
        with gr.Accordion("Vocal enrichment", open=False):
            with gr.Row(), gr.Column():
                tab_config.split_voice.instantiate()
            with gr.Row():
                with gr.Column():
                    tab_config.autotune_voice.instantiate()
                    tab_config.autotune_strength.instantiate()
                with gr.Column():
                    tab_config.proposed_pitch.instantiate()
                    tab_config.proposed_pitch_threshold.instantiate()
                with gr.Column():
                    tab_config.clean_voice.instantiate()
                    tab_config.clean_strength.instantiate()
            tab_config.autotune_voice.instance.change(
                partial(toggle_visibility, targets={True}),
                inputs=tab_config.autotune_voice.instance,
                outputs=tab_config.autotune_strength.instance,
                show_progress="hidden",
            )
            tab_config.proposed_pitch.instance.change(
                partial(toggle_visibility, targets={True}),
                inputs=tab_config.proposed_pitch.instance,
                outputs=tab_config.proposed_pitch_threshold.instance,
                show_progress="hidden",
            )
            tab_config.clean_voice.instance.change(
                partial(toggle_visibility, targets={True}),
                inputs=tab_config.clean_voice.instance,
                outputs=tab_config.clean_strength.instance,
                show_progress="hidden",
            )
        with gr.Accordion("Speaker embedding", open=False):
            with gr.Row():
                with gr.Column():
                    tab_config.embedder_model.instantiate()
                    tab_config.custom_embedder_model.instance.render()
                tab_config.sid.instantiate()
            tab_config.embedder_model.instance.change(
                partial(toggle_visibility, targets={EmbedderModel.CUSTOM}),
                inputs=tab_config.embedder_model.instance,
                outputs=tab_config.custom_embedder_model.instance,
                show_progress="hidden",
            )


def _render_mixing_options(tab_config: OneClickSongGenerationConfig) -> None:
    with gr.Accordion("Audio mixing", open=False):
        gr.Markdown("")
        with gr.Accordion("Reverb control on converted vocals", open=False):
            with gr.Row():
                tab_config.room_size.instantiate()
            with gr.Row():
                tab_config.wet_level.instantiate()
                tab_config.dry_level.instantiate()
                tab_config.damping.instantiate()

        with gr.Accordion("Volume controls (dB)", open=False), gr.Row():
            tab_config.main_gain.instantiate()
            tab_config.inst_gain.instantiate()
            tab_config.backup_gain.instantiate()


def _render_output_options(tab_config: OneClickSongGenerationConfig) -> None:
    with gr.Accordion("Audio output", open=False):
        with gr.Row():
            tab_config.output_name.instantiate(
                value=partial(
                    update_output_name,
                    get_song_cover_name,
                    True,  # noqa: FBT003
                ),
                inputs=[
                    gr.State(None),
                    tab_config.cached_song.instance,
                    tab_config.voice_model.instance,
                ],
            )
            tab_config.output_sr.instantiate()
            tab_config.output_format.instantiate()
        with gr.Row():
            tab_config.show_intermediate_audio.instantiate()


def _render_intermediate_audio(tab_config: OneClickSongGenerationConfig) -> None:
    with gr.Accordion(
        "Intermediate audio tracks",
        open=False,
        visible=False,
    ) as intermediate_audio_accordion:
        with gr.Accordion(
            "Step 0: song retrieval",
            open=False,
        ) as song_retrieval_accordion:
            tab_config.intermediate_audio.song.instantiate()
        with (
            gr.Accordion(
                "Step 1a: vocals/instrumentals separation",
                open=False,
            ) as vocals_separation_accordion,
            gr.Row(),
        ):
            tab_config.intermediate_audio.vocals.instantiate()
            tab_config.intermediate_audio.instrumentals.instantiate()
        with (
            gr.Accordion(
                "Step 1b: main vocals/ backup vocals separation",
                open=False,
            ) as main_vocals_separation_accordion,
            gr.Row(),
        ):
            tab_config.intermediate_audio.main_vocals.instantiate()
            tab_config.intermediate_audio.backup_vocals.instantiate()
        with (
            gr.Accordion(
                "Step 1c: main vocals cleanup",
                open=False,
            ) as vocal_cleanup_accordion,
            gr.Row(),
        ):
            tab_config.intermediate_audio.main_vocals_dereverbed.instantiate()
            tab_config.intermediate_audio.main_vocals_reverb.instantiate()
        with gr.Accordion(
            "Step 2: conversion of main vocals",
            open=False,
        ) as vocal_conversion_accordion:
            tab_config.intermediate_audio.converted_vocals.instantiate()
        with gr.Accordion(
            "Step 3: post-processing of converted vocals",
            open=False,
        ) as vocals_postprocessing_accordion:
            tab_config.intermediate_audio.postprocessed_vocals.instantiate()
        with (
            gr.Accordion(
                "Step 4: pitch shift of background tracks",
                open=False,
            ) as pitch_shift_accordion,
            gr.Row(),
        ):
            tab_config.intermediate_audio.instrumentals_shifted.instantiate()
            tab_config.intermediate_audio.backup_vocals_shifted.instantiate()

    tab_config.show_intermediate_audio.instance.change(
        partial(toggle_intermediate_audio, num_components=7),
        inputs=tab_config.show_intermediate_audio.instance,
        outputs=[
            intermediate_audio_accordion,
            song_retrieval_accordion,
            vocals_separation_accordion,
            main_vocals_separation_accordion,
            vocal_cleanup_accordion,
            vocal_conversion_accordion,
            vocals_postprocessing_accordion,
            pitch_shift_accordion,
        ],
        show_progress="hidden",
    )
