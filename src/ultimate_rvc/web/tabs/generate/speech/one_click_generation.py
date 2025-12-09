"""
Module which defines the code for the "Generate speech - one-click
generation" tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import partial

import gradio as gr

from ultimate_rvc.core.generate.speech import get_mixed_speech_track_name, run_pipeline
from ultimate_rvc.core.manage.audio import (
    get_saved_output_audio,
    get_saved_speech_audio,
)
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
    from ultimate_rvc.web.config.main import OneClickSpeechGenerationConfig, TotalConfig


def render(total_config: TotalConfig) -> None:
    """
    Render "Generate speech - one-click generation" tab.

    Parameters
    ----------
    total_config : TotalConfig
        Model containing all component configuration settings for the
        Ultimate RVC web UI.

    """
    tab_config = total_config.speech.one_click
    with gr.Tab("One-click"):
        _render_input(tab_config)
        with gr.Accordion("Options", open=False):
            _render_tts_options(tab_config)
            _render_conversion_options(tab_config)
            _render_output_options(tab_config)
            _render_intermediate_audio(tab_config)

        with gr.Row(equal_height=True):
            reset_btn = gr.Button(value="Reeset settings", scale=2)
            generate_btn = gr.Button(value="Generate", scale=2, variant="primary")
        mixed_speech = gr.Audio(
            label="Mixed speech",
            scale=3,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )
        generate_btn.click(
            partial(
                exception_harness(
                    run_pipeline,
                    info_msg="Speech generated successfully!",
                ),
                progress_bar=PROGRESS_BAR,
            ),
            inputs=[
                tab_config.source.instance,
                tab_config.voice_model.instance,
                tab_config.edge_tts_voice.instance,
                tab_config.tts_pitch_shift.instance,
                tab_config.tts_speed_change.instance,
                tab_config.tts_volume_change.instance,
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
                tab_config.output_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                tab_config.output_name.instance,
            ],
            outputs=[mixed_speech, *tab_config.intermediate_audio.all],
            concurrency_limit=1,
            concurrency_id=ConcurrencyId.GPU,
        ).success(
            partial(update_dropdowns, get_saved_speech_audio, 1),
            outputs=total_config.management.audio.speech.instance,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_saved_output_audio, 1),
            outputs=total_config.management.audio.output.instance,
            show_progress="hidden",
        )
        reset_btn.click(
            lambda: [
                tab_config.tts_pitch_shift.value,
                tab_config.tts_speed_change.value,
                tab_config.tts_volume_change.value,
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
                tab_config.output_gain.value,
                tab_config.output_sr.value,
                tab_config.output_format.value,
                tab_config.show_intermediate_audio.value,
            ],
            outputs=[
                tab_config.tts_pitch_shift.instance,
                tab_config.tts_speed_change.instance,
                tab_config.tts_volume_change.instance,
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
                tab_config.output_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                tab_config.show_intermediate_audio.instance,
            ],
            show_progress="hidden",
        )


def _render_input(tab_config: OneClickSpeechGenerationConfig) -> None:
    with gr.Row():
        with gr.Column():
            tab_config.source_type.instantiate()
        with gr.Column():
            tab_config.source.instantiate()
            local_file = gr.File(
                label="Source",
                file_types=[".txt"],
                file_count="single",
                type="filepath",
                visible=False,
            )
        tab_config.source_type.instance.input(
            partial(toggle_visible_component, 2),
            inputs=tab_config.source_type.instance,
            outputs=[tab_config.source.instance, local_file],
            show_progress="hidden",
        )
        local_file.change(
            update_value,
            inputs=local_file,
            outputs=tab_config.source.instance,
            show_progress="hidden",
        )
    with gr.Row():
        tab_config.edge_tts_voice.instance.render()
        tab_config.voice_model.instance.render()


def _render_tts_options(tab_config: OneClickSpeechGenerationConfig) -> None:
    with gr.Accordion("Edge TTS", open=False), gr.Row():
        tab_config.tts_pitch_shift.instantiate()
        tab_config.tts_speed_change.instantiate()
        tab_config.tts_volume_change.instantiate()


def _render_conversion_options(tab_config: OneClickSpeechGenerationConfig) -> None:
    with gr.Accordion("Speech conversion", open=False):
        with gr.Row():
            tab_config.n_octaves.instantiate()
            tab_config.n_semitones.instantiate()
        with gr.Accordion("Voice synthesis", open=False):
            with gr.Row():
                tab_config.f0_method.instantiate()
                tab_config.index_rate.instantiate()
            with gr.Row():
                tab_config.rms_mix_rate.instantiate()
                tab_config.protect_rate.instantiate()
        with gr.Accordion("Speech enrichment", open=False):
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
            partial(toggle_visibility, targets={True}, update_default=False),
            inputs=tab_config.autotune_voice.instance,
            outputs=tab_config.autotune_strength.instance,
            show_progress="hidden",
        )
        tab_config.proposed_pitch.instance.change(
            partial(toggle_visibility, targets={True}, update_default=False),
            inputs=tab_config.proposed_pitch.instance,
            outputs=tab_config.proposed_pitch_threshold.instance,
            show_progress="hidden",
        )
        tab_config.clean_voice.instance.change(
            partial(toggle_visibility, targets={True}, update_default=False),
            inputs=tab_config.clean_voice.instance,
            outputs=tab_config.clean_strength.instance,
            show_progress="hidden",
        )
        with gr.Accordion("Speaker embedding", open=False), gr.Row():
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


def _render_output_options(tab_config: OneClickSpeechGenerationConfig) -> None:
    with gr.Accordion("Audio output", open=False):
        with gr.Row():
            tab_config.output_gain.instantiate()
            tab_config.output_sr.instantiate()
        with gr.Row():
            tab_config.output_name.instantiate(
                value=partial(
                    update_output_name,
                    get_mixed_speech_track_name,
                    True,  # noqa: FBT003
                ),
                inputs=[tab_config.source.instance, tab_config.voice_model.instance],
            )
            tab_config.output_format.instantiate()
        with gr.Row():
            tab_config.show_intermediate_audio.instantiate()


def _render_intermediate_audio(tab_config: OneClickSpeechGenerationConfig) -> None:
    with gr.Accordion(
        "Intermediate audio tracks",
        open=False,
        visible=False,
    ) as intermediate_audio_accordion:
        with gr.Accordion(
            "Step 1: text-to-speech conversion",
            open=False,
        ) as tts_accordion:
            tab_config.intermediate_audio.speech.instantiate()
        with gr.Accordion(
            "Step 2: speech conversion",
            open=False,
        ) as speech_conversion_accordion:
            tab_config.intermediate_audio.converted_speech.instantiate()
    tab_config.show_intermediate_audio.instance.change(
        partial(toggle_intermediate_audio, num_components=2),
        inputs=tab_config.show_intermediate_audio.instance,
        outputs=[
            intermediate_audio_accordion,
            tts_accordion,
            speech_conversion_accordion,
        ],
        show_progress="hidden",
    )
