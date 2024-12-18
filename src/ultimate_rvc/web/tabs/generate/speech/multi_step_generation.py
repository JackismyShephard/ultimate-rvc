"""
Module which defines the code for the "Generate Speech - Multi-step
Generation" tab.
"""

from functools import partial

import gradio as gr

from ultimate_rvc.core.common import SPEECH_DIR
from ultimate_rvc.core.generate.common import convert
from ultimate_rvc.core.generate.speech import mix_voice, run_edge_tts
from ultimate_rvc.typing_extra import (
    AudioExt,
    EmbedderModel,
    F0Method,
    RVCContentType,
    SampleRate,
)
from ultimate_rvc.web.common import (
    PROGRESS_BAR,
    exception_harness,
    toggle_visibility,
    toggle_visible_component,
    update_audio,
    update_output_audio,
    update_speech_audio,
    update_speech_track_name,
    update_value,
)
from ultimate_rvc.web.typing_extra import ConcurrencyId, SpeechSourceType


def render(
    edge_tts_voice: gr.Dropdown,
    model_multi: gr.Dropdown,
    speech_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
) -> None:
    """
    Render "Generate Speech - Multi-step Generation" tab.

    Parameters
    ----------
    edge_tts_voice: gr.Dropdown
        Dropdown component for selecting a Edge TTS voice in the
        "Generate Speech - One Click Generation" tab.
    model_multi: gr.Dropdown
        Dropdown component for selecting a voice model in the
        "Generate Speech - Multi-step Generation" tab.
    speech_audio : gr.Dropdown
        Dropdown component for speech audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.

    """
    with gr.Tab("Multi-step generation"):

        input_tracks = [
            gr.Audio(label=label, type="filepath", render=False)
            for label in ["Speech", "Voice"]
        ]

        speech_track_input, voice_track_input = input_tracks

        output_tracks = [
            gr.Audio(label=label, type="filepath", render=False, interactive=False)
            for label in ["Speech", "Converted voice", "Mixed voice"]
        ]

        speech_track_output, converted_voice_track_output, mixed_voice_track_output = (
            output_tracks
        )

        transfer_defaults = [
            ["Step 2: speech"],
            ["Step 3: voice"],
            [],
        ]

        (
            speech_transfer_default,
            voice_transfer_default,
            mixed_voice_transfer_default,
        ) = transfer_defaults

        transfer = [
            gr.Dropdown(
                ["Step 2: speech", "Step 3: voice"],
                label=f"{label_prefix} destination",
                info=(
                    "select the input track(s) to transfer the "
                    f"{label_prefix.lower()} to when the 'Transfer"
                    f" {label_prefix.lower()}' button is clicked"
                ),
                render=False,
                type="index",
                multiselect=True,
                value=value,
            )
            for value, label_prefix in zip(
                transfer_defaults,
                ["Speech", "Voice", "Mixed voice"],
                strict=True,
            )
        ]
        speech_transfer, voice_transfer, mixed_voice_transfer = transfer

        with gr.Accordion("Step 1: Text to speech conversion", open=True):
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
                    source = gr.Textbox(
                        label="Source",
                        info="Text to generate speech from",
                    )
                    local_file = gr.File(
                        label="Source",
                        file_types=[".txt"],
                        file_count="single",
                        type="filepath",
                        visible=False,
                    )
                with gr.Column():
                    edge_tts_voice.render()
                source_type.input(
                    partial(toggle_visible_component, 2),
                    inputs=source_type,
                    outputs=[source, local_file],
                    show_progress="hidden",
                )
                local_file.change(
                    update_value,
                    inputs=local_file,
                    outputs=source,
                    show_progress="hidden",
                )
            gr.Markdown("**Settings**")
            with gr.Row():
                tts_pitch_shift = gr.Slider(
                    -100,
                    100,
                    value=0,
                    step=1,
                    label="Edge TTS pitch shift",
                    info=(
                        "The number of hertz to shift the pitch of the speech generated"
                        " by Edge TTS."
                    ),
                )
                tts_speed_change = gr.Slider(
                    -50,
                    100,
                    value=0,
                    step=1,
                    label="TTS speed change",
                    info=(
                        "The percentual change to the speed of the speech generated by"
                        " Edge TTS."
                    ),
                )
                tts_volume_change = gr.Slider(
                    -100,
                    100,
                    value=0,
                    step=1,
                    label="TTS volume change",
                    info=(
                        "The percentual change to the volume of the speech generated by"
                        " Edge TTS."
                    ),
                )
            speech_transfer.render()
            gr.Markdown("**Outputs**")
            speech_track_output.render()
            gr.Markdown("**Controls**")
            tts_btn = gr.Button("Convert text", variant="primary")
            tts_transfer_btn = gr.Button("Transfer speech")
            tts_reset_btn = gr.Button("Reset settings")
            tts_reset_btn.click(
                lambda: [0, 0, 0, gr.Dropdown(value=speech_transfer_default)],
                outputs=[
                    tts_pitch_shift,
                    tts_speed_change,
                    tts_volume_change,
                    speech_transfer,
                ],
                show_progress="hidden",
            )
            tts_btn.click(
                partial(
                    exception_harness(
                        run_edge_tts,
                        info_msg="Text succesfully converted",
                    ),
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=[
                    source,
                    edge_tts_voice,
                    tts_pitch_shift,
                    tts_speed_change,
                    tts_volume_change,
                ],
                outputs=[speech_track_output],
                concurrency_id=ConcurrencyId.GPU,
                concurrency_limit=1,
            ).then(
                partial(update_speech_audio, 1, [], [0]),
                outputs=speech_audio,
                show_progress="hidden",
            )
        with gr.Accordion("Step 2: voice conversion", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            speech_track_input.render()
            model_multi.render()
            gr.Markdown("**Settings**")
            with gr.Accordion("Main settings", open=True), gr.Row():
                n_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Octave shift",
                    info=(
                        "The number of octaves to pitch-shift the converted voice by."
                        " Use 1 for male-to-female and -1 for vice-versa."
                    ),
                )
                n_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Semitone shift",
                    info=(
                        "The number of semi-tones to pitch-shift the converted"
                        " voice by."
                    ),
                )
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
                            " coming from the voice model.<br><br>"
                        ),
                    )
                    filter_radius = gr.Slider(
                        0,
                        7,
                        value=3,
                        step=1,
                        label="Filter radius",
                        info=(
                            "If >=3: apply median filtering to extracted pitch values."
                            " Can help reduce breathiness in the converted"
                            " voice.<br><br>"
                        ),
                    )
                with gr.Row():
                    rms_mix_rate = gr.Slider(
                        0,
                        1,
                        value=0.25,
                        label="RMS mix rate",
                        info=(
                            "How much to mimic the loudness (0) of the input speech or"
                            " a fixed loudness (1).<br><br><br>"
                        ),
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
                            " effect.<br><br>"
                        ),
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
                    )
            with gr.Accordion("Voice enrichment settings", open=False), gr.Row():
                with gr.Column():
                    split_speech = gr.Checkbox(
                        value=True,
                        label="Split speech track",
                        info=(
                            "Whether to split the Edge TTS speech intosmaller segments"
                            " before converting it. This can improve output quality for"
                            " longer speech."
                        ),
                    )
                with gr.Column():
                    autotune_voice = gr.Checkbox(
                        label="Autotune converted voice",
                        info=(
                            "Whether to apply autotune to the converted voice.<br><br>"
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
                    )
                with gr.Column():
                    clean_voice = gr.Checkbox(
                        label="Clean converted voice",
                        info=(
                            "Whether to clean the converted voice using noise"
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
                    )
            autotune_voice.change(
                partial(toggle_visibility, target=True),
                inputs=autotune_voice,
                outputs=autotune_strength,
                show_progress="hidden",
            )
            clean_voice.change(
                partial(toggle_visibility, target=True),
                inputs=clean_voice,
                outputs=clean_strength,
                show_progress="hidden",
            )
            with gr.Accordion("Speaker embedding settings", open=False), gr.Row():
                with gr.Column():
                    embedder_model = gr.Dropdown(
                        list(EmbedderModel),
                        value=EmbedderModel.CONTENTVEC,
                        label="Embedder model",
                        info="The model to use for generating speaker embeddings.",
                    )
                    embedder_model_custom = gr.Textbox(
                        label="Custom embedder model",
                        info=(
                            "The path to a directory with a custom model to use for"
                            " generating speaker embeddings."
                        ),
                        visible=False,
                    )
                sid = gr.Number(
                    label="Speaker ID",
                    info="Speaker ID for multi-speaker-models.",
                    precision=0,
                )
            embedder_model.change(
                partial(toggle_visibility, target=EmbedderModel.CUSTOM),
                inputs=embedder_model,
                outputs=embedder_model_custom,
                show_progress="hidden",
            )
            voice_transfer.render()
            gr.Markdown("**Outputs**")
            converted_voice_track_output.render()
            gr.Markdown("**Controls**")
            convert_voice_btn = gr.Button("Convert voice", variant="primary")
            converted_voice_transfer_btn = gr.Button("Transfer converted voice")
            convert_voice_reset_btn = gr.Button("Reset settings")

            convert_voice_reset_btn.click(
                lambda: [
                    0,
                    0,
                    F0Method.RMVPE,
                    0.5,
                    3,
                    0.25,
                    0.33,
                    128,
                    True,
                    False,
                    1.0,
                    False,
                    0.7,
                    EmbedderModel.CONTENTVEC,
                    None,
                    0,
                    gr.Dropdown(value=voice_transfer_default),
                ],
                outputs=[
                    n_octaves,
                    n_semitones,
                    f0_methods,
                    index_rate,
                    filter_radius,
                    rms_mix_rate,
                    protect_rate,
                    hop_length,
                    split_speech,
                    autotune_voice,
                    autotune_strength,
                    clean_voice,
                    clean_strength,
                    embedder_model,
                    embedder_model_custom,
                    sid,
                    voice_transfer,
                ],
                show_progress="hidden",
            )

            convert_voice_btn.click(
                partial(
                    exception_harness(
                        convert,
                        info_msg="Voice succesfully converted!",
                    ),
                    directory=SPEECH_DIR,
                    content_type=RVCContentType.VOICE,
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=[
                    speech_track_input,
                    model_multi,
                    n_octaves,
                    n_semitones,
                    f0_methods,
                    index_rate,
                    filter_radius,
                    rms_mix_rate,
                    protect_rate,
                    hop_length,
                    split_speech,
                    autotune_voice,
                    autotune_strength,
                    clean_voice,
                    clean_strength,
                    embedder_model,
                    embedder_model_custom,
                    sid,
                ],
                outputs=converted_voice_track_output,
                concurrency_id=ConcurrencyId.GPU,
                concurrency_limit=1,
            ).then(
                partial(update_speech_audio, 1, [], [0]),
                outputs=speech_audio,
                show_progress="hidden",
            )
        with gr.Accordion("step 3: voice mixing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            voice_track_input.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                output_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Output gain",
                    info="The gain to apply to the converted voice.",
                )
                output_sr = gr.Dropdown(
                    choices=list(SampleRate),
                    value=SampleRate.HZ_44100,
                    label="Output sample rate",
                    info=(
                        "The sample rate of the mixed audio track containing the"
                        " converted voice."
                    ),
                )
            with gr.Row():
                output_name = gr.Textbox(
                    value=update_speech_track_name,
                    inputs=[source, model_multi],
                    label="Output name",
                    info=(
                        "If no name is provided, a suitable name will be generated"
                        " automatically."
                    ),
                    placeholder="Ultimate RVC speech",
                )
                output_format = gr.Dropdown(
                    list(AudioExt),
                    value=AudioExt.MP3,
                    label="Output format",
                    info=(
                        "The format of the mixed audio track containing the converted"
                        " voice."
                    ),
                )
            mixed_voice_transfer.render()
            gr.Markdown("**Outputs**")
            mixed_voice_track_output.render()
            gr.Markdown("**Controls**")
            mix_voice_btn = gr.Button("Mix voice", variant="primary")
            mix_voice_transfer_btn = gr.Button("Transfer mixed voice")
            mix_voice_reset_btn = gr.Button("Reset settings")

            mix_voice_reset_btn.click(
                lambda: [
                    0,
                    SampleRate.HZ_44100,
                    AudioExt.MP3,
                    gr.Dropdown(value=mixed_voice_transfer_default),
                ],
                outputs=[output_gain, output_sr, output_format, mixed_voice_transfer],
                show_progress="hidden",
            )
            mix_voice_btn.click(
                partial(
                    exception_harness(
                        mix_voice,
                        info_msg="Voice succesfully mixed!",
                    ),
                    progress_bar=PROGRESS_BAR,
                ),
                inputs=[
                    voice_track_input,
                    output_gain,
                    output_sr,
                    output_format,
                    output_name,
                ],
                outputs=mixed_voice_track_output,
            ).then(
                partial(update_output_audio, 1, [], [0]),
                outputs=output_audio,
                show_progress="hidden",
            )
        for btn, transfer, output in [
            (tts_transfer_btn, speech_transfer, speech_track_output),
            (
                converted_voice_transfer_btn,
                voice_transfer,
                converted_voice_track_output,
            ),
            (mix_voice_transfer_btn, mixed_voice_transfer, mixed_voice_track_output),
        ]:
            btn.click(
                partial(update_audio, len(input_tracks)),
                inputs=[transfer, output],
                outputs=input_tracks,
                show_progress="hidden",
            )
