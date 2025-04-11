"""
Module which defines the code for the "Generate speech - multi-step
generation" tab.
"""

from functools import partial

import gradio as gr

from ultimate_rvc.core.common import SPEECH_DIR
from ultimate_rvc.core.generate.common import convert
from ultimate_rvc.core.generate.speech import (
    get_mixed_speech_track_name,
    mix_speech,
    run_edge_tts,
)
from ultimate_rvc.core.manage.audio import (
    get_saved_output_audio,
    get_saved_speech_audio,
)
from ultimate_rvc.typing_extra import EmbedderModel, RVCContentType
from ultimate_rvc.web.common import (
    exception_harness,
    render_transfer_component,
    toggle_visibility,
    toggle_visible_component,
    update_audio,
    update_dropdowns,
    update_output_name,
    update_value,
)
from ultimate_rvc.web.typing_extra import (
    ConcurrencyId,
    SpeechSourceType,
    SpeechTransferOption,
    TotalConfig,
)


def render(total_config: TotalConfig) -> None:
    """
    Render "Generate speech - multi-step generation" tab.

    Parameters
    ----------
    total_config : TotalConfig
        The total configuration object.Contains all the configuration
        objects for the app.

    """
    tab_config = total_config.speech.multi_step
    for c in tab_config.input_tracks:
        c.instantiate()
    with gr.Tab("Multi-step generation"):

        _render_step_1(total_config)
        _render_step_2(total_config)
        _render_step_3(total_config)


def _render_step_1(total_config: TotalConfig) -> None:
    tab_config = total_config.speech.multi_step
    with gr.Accordion("Step 1: Text-to-speech conversion", open=True):
        gr.Markdown("")
        gr.Markdown("**Inputs**")
        with gr.Row():
            with gr.Column():
                source_type = gr.Dropdown(
                    list(SpeechSourceType),
                    value=SpeechSourceType.TEXT,
                    label="Source type",
                    type="index",
                    info="The type of source to generate speech from.",
                )
            with gr.Column():
                tab_config.source.instantiate()
                local_file = gr.File(
                    label="Source",
                    file_types=[".txt"],
                    file_count="single",
                    type="filepath",
                    visible=False,
                )
            source_type.input(
                partial(toggle_visible_component, 2),
                inputs=source_type,
                outputs=[tab_config.source.instance, local_file],
                show_progress="hidden",
            )
            local_file.change(
                update_value,
                inputs=local_file,
                outputs=tab_config.source.instance,
                show_progress="hidden",
            )
        tab_config.edge_tts_voice.instance.render()
        gr.Markdown("**Settings**")
        with gr.Row():
            tab_config.tts_pitch_shift.instantiate()
            tab_config.tts_speed_change.instantiate()
            tab_config.tts_volume_change.instantiate()
        speech_transfer = _render_speech_transfer(
            [SpeechTransferOption.STEP_2_SPEECH],
            "Speech",
        )

        gr.Markdown("**Outputs**")
        speech_track_output = gr.Audio(
            label="Generated speech",
            type="filepath",
            interactive=False,
        )
        gr.Markdown("**Controls**")
        tts_btn = gr.Button("Convert text", variant="primary")
        tts_transfer_btn = gr.Button("Transfer speech")
        tts_reset_btn = gr.Button("Reset settings")
        tts_reset_btn.click(
            lambda: [0, 0, 0, gr.Dropdown(value=[SpeechTransferOption.STEP_2_SPEECH])],
            outputs=[
                tab_config.tts_pitch_shift.instance,
                tab_config.tts_speed_change.instance,
                tab_config.tts_volume_change.instance,
                speech_transfer,
            ],
            show_progress="hidden",
        )
        tts_btn.click(
            exception_harness(
                run_edge_tts,
                info_msg="Text succesfully converted!",
            ),
            inputs=[
                tab_config.source.instance,
                tab_config.edge_tts_voice.instance,
                tab_config.tts_pitch_shift.instance,
                tab_config.tts_speed_change.instance,
                tab_config.tts_volume_change.instance,
            ],
            outputs=speech_track_output,
        ).then(
            partial(update_dropdowns, get_saved_speech_audio, 1, [], [0]),
            outputs=total_config.management.audio.speech_audio.instance,
            show_progress="hidden",
        )
        tts_transfer_btn.click(
            partial(update_audio, len(tab_config.input_tracks)),
            inputs=[speech_transfer, speech_track_output],
            outputs=[c.instance for c in tab_config.input_tracks],
            show_progress="hidden",
        )


def _render_step_2(total_config: TotalConfig) -> None:
    tab_config = total_config.speech.multi_step

    with gr.Accordion("Step 2: speech conversion", open=False):
        gr.Markdown("")
        gr.Markdown("**Inputs**")
        tab_config.speech_track_input.instance.render()
        tab_config.voice_model.instance.render()
        gr.Markdown("**Settings**")
        with gr.Accordion("Main settings", open=True), gr.Row():
            tab_config.n_octaves.instantiate()
            tab_config.n_semitones.instantiate()
        with gr.Accordion("Voice synthesis settings", open=False):
            with gr.Row():
                tab_config.f0_methods.instantiate()
                tab_config.index_rate.instantiate()
            with gr.Row():
                tab_config.rms_mix_rate.instantiate()
                tab_config.protect_rate.instantiate()
                tab_config.hop_length.instantiate()
        with gr.Accordion("Speech enrichment settings", open=False), gr.Row():
            with gr.Column():
                tab_config.split_voice.instantiate()
            with gr.Column():
                tab_config.autotune_voice.instantiate()
                tab_config.autotune_strength.instantiate()
            with gr.Column():
                tab_config.clean_voice.instantiate()
                tab_config.clean_strength.instantiate()
        tab_config.autotune_voice.instance.change(
            partial(
                toggle_visibility,
                targets={True},
                default=tab_config.autotune_voice.value,
            ),
            inputs=tab_config.autotune_voice.instance,
            outputs=tab_config.autotune_strength.instance,
            show_progress="hidden",
        )
        tab_config.clean_voice.instance.change(
            partial(
                toggle_visibility,
                targets={True},
                default=tab_config.clean_voice.value,
            ),
            inputs=tab_config.clean_voice.instance,
            outputs=tab_config.clean_strength.instance,
            show_progress="hidden",
        )
        with gr.Accordion("Speaker embedding settings", open=False), gr.Row():
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
        converted_speech_transfer = _render_speech_transfer(
            [SpeechTransferOption.STEP_3_SPEECH],
            "Converted speech",
        )
        gr.Markdown("**Outputs**")
        converted_speech_track_output = gr.Audio(
            label="Converted speech",
            type="filepath",
            interactive=False,
        )
        gr.Markdown("**Controls**")
        convert_speech_btn = gr.Button("Convert speech", variant="primary")
        converted_speech_transfer_btn = gr.Button("Transfer converted speech")
        convert_speech_reset_btn = gr.Button("Reset settings")

        convert_speech_reset_btn.click(
            lambda: [
                tab_config.n_octaves.value,
                tab_config.n_semitones.value,
                tab_config.f0_methods.value,
                tab_config.index_rate.value,
                tab_config.rms_mix_rate.value,
                tab_config.protect_rate.value,
                tab_config.hop_length.value,
                tab_config.split_voice.value,
                tab_config.autotune_voice.value,
                tab_config.autotune_strength.value,
                tab_config.clean_voice.value,
                tab_config.clean_strength.value,
                tab_config.embedder_model.value,
                tab_config.custom_embedder_model.value,
                tab_config.sid.value,
                gr.Dropdown(value=[SpeechTransferOption.STEP_3_SPEECH]),
            ],
            outputs=[
                tab_config.n_octaves.instance,
                tab_config.n_semitones.instance,
                tab_config.f0_methods.instance,
                tab_config.index_rate.instance,
                tab_config.rms_mix_rate.instance,
                tab_config.protect_rate.instance,
                tab_config.hop_length.instance,
                tab_config.split_voice.instance,
                tab_config.autotune_voice.instance,
                tab_config.autotune_strength.instance,
                tab_config.clean_voice.instance,
                tab_config.clean_strength.instance,
                tab_config.embedder_model.instance,
                tab_config.custom_embedder_model.instance,
                tab_config.sid.instance,
                converted_speech_transfer,
            ],
            show_progress="hidden",
        )

        convert_speech_btn.click(
            partial(
                exception_harness(
                    convert,
                    info_msg="Speech succesfully converted!",
                ),
                content_type=RVCContentType.SPEECH,
            ),
            inputs=[
                tab_config.speech_track_input.instance,
                gr.State(SPEECH_DIR),
                tab_config.voice_model.instance,
                tab_config.n_octaves.instance,
                tab_config.n_semitones.instance,
                tab_config.f0_methods.instance,
                tab_config.index_rate.instance,
                tab_config.rms_mix_rate.instance,
                tab_config.protect_rate.instance,
                tab_config.hop_length.instance,
                tab_config.split_voice.instance,
                tab_config.autotune_voice.instance,
                tab_config.autotune_strength.instance,
                tab_config.clean_voice.instance,
                tab_config.clean_strength.instance,
                tab_config.embedder_model.instance,
                tab_config.custom_embedder_model.instance,
                tab_config.sid.instance,
            ],
            outputs=converted_speech_track_output,
            concurrency_id=ConcurrencyId.GPU,
            concurrency_limit=1,
        ).then(
            partial(update_dropdowns, get_saved_speech_audio, 1, [], [0]),
            outputs=total_config.management.audio.speech_audio.instance,
            show_progress="hidden",
        )
        converted_speech_transfer_btn.click(
            partial(update_audio, len(tab_config.input_tracks)),
            inputs=[converted_speech_transfer, converted_speech_track_output],
            outputs=[c.instance for c in tab_config.input_tracks],
            show_progress="hidden",
        )


def _render_step_3(total_config: TotalConfig) -> None:
    tab_config = total_config.speech.multi_step
    with gr.Accordion("Step 3: speech mixing", open=False):
        gr.Markdown("")
        gr.Markdown("**Inputs**")
        tab_config.converted_speech_track_input.instance.render()
        gr.Markdown("**Settings**")
        with gr.Row():
            tab_config.output_gain.instantiate()
            tab_config.output_sr.instantiate()
        with gr.Row():
            tab_config.output_name.instantiate(
                value=partial(
                    update_output_name,
                    get_mixed_speech_track_name,
                    False,  # noqa: FBT003,,
                ),
                inputs=[
                    gr.State(None),
                    gr.State(None),
                    tab_config.converted_speech_track_input.instance,
                ],
            )
            tab_config.output_format.instantiate()
        mixed_speech_transfer = _render_speech_transfer(
            [],
            "Mixed speech",
        )
        gr.Markdown("**Outputs**")
        mixed_speech_track_output = gr.Audio(
            label="Mixed speech",
            type="filepath",
            interactive=False,
        )
        gr.Markdown("**Controls**")
        mix_speech_btn = gr.Button("Mix speech", variant="primary")
        mix_speech_transfer_btn = gr.Button("Transfer mixed speech")
        mix_speech_reset_btn = gr.Button("Reset settings")

        mix_speech_reset_btn.click(
            lambda: [
                tab_config.output_gain.value,
                tab_config.output_sr.value,
                tab_config.output_format.value,
                gr.Dropdown(value=[]),
            ],
            outputs=[
                tab_config.output_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                mixed_speech_transfer,
            ],
            show_progress="hidden",
        )
        mix_speech_btn.click(
            exception_harness(
                mix_speech,
                info_msg="Speech succesfully mixed!",
            ),
            inputs=[
                tab_config.converted_speech_track_input.instance,
                tab_config.output_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                tab_config.output_name.instance,
            ],
            outputs=mixed_speech_track_output,
        ).then(
            partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
            outputs=total_config.management.audio.output_audio.instance,
            show_progress="hidden",
        )
        mix_speech_transfer_btn.click(
            partial(update_audio, len(tab_config.input_tracks)),
            inputs=[mixed_speech_transfer, mixed_speech_track_output],
            outputs=[c.instance for c in tab_config.input_tracks],
            show_progress="hidden",
        )


def _render_speech_transfer(
    value: list[SpeechTransferOption],
    label_prefix: str,
) -> gr.Dropdown:
    return render_transfer_component(value, label_prefix, SpeechTransferOption)
