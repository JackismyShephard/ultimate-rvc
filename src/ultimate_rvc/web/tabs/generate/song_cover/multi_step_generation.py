"""
Module which defines the code for the
"Generate song covers - multi-step generation" tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from functools import partial

import gradio as gr

from ultimate_rvc.core.generate.common import convert
from ultimate_rvc.core.generate.song_cover import (
    get_named_song_dirs,
    get_song_cover_name,
    mix_song,
    pitch_shift,
    postprocess,
    retrieve_song,
    separate_audio,
)
from ultimate_rvc.core.manage.audio import get_saved_output_audio
from ultimate_rvc.typing_extra import (
    AudioExt,
    EmbedderModel,
    F0Method,
    RVCContentType,
    SampleRate,
    SegmentSize,
    SeparationModel,
)
from ultimate_rvc.web.common import (
    exception_harness,
    toggle_visibility,
    toggle_visible_component,
    update_audio,
    update_dropdowns,
    update_output_name,
    update_value,
)
from ultimate_rvc.web.typing_extra import ConcurrencyId, SongSourceType

if TYPE_CHECKING:
    from collections.abc import Sequence


def _pair_audio_tracks_and_gain(
    audio_components: Sequence[gr.Audio],
    gain_components: Sequence[gr.Slider],
    data: dict[gr.Audio | gr.Slider, Any],
) -> list[tuple[str, int]]:
    """
    Pair audio tracks and gain levels stored in separate gradio
    components.

    This function is meant to first be partially applied to the sequence
    of audio components and the sequence of slider components containing
    the values that should be combined. The resulting function can then
    be called by an event listener whose inputs is a set containing
    those audio and slider components. The `data` parameter in that case
    will contain a mapping from each of those components to the value
    that the component stores.

    Parameters
    ----------
    audio_components : Sequence[gr.Audio]
        Audio components to pair with gain levels.
    gain_components : Sequence[gr.Slider]
        Gain level components to pair with audio tracks.
    data : dict[gr.Audio | gr.Slider, Any]
        Data from the audio and gain components.

    Returns
    -------
    list[tuple[str, int]]
        Paired audio tracks and gain levels.

    Raises
    ------
    ValueError
        If the number of audio tracks and gain levels are not the same.

    """
    audio_tracks = [data[component] for component in audio_components]
    gain_levels = [data[component] for component in gain_components]
    if len(audio_tracks) != len(gain_levels):
        err_msg = "Number of audio tracks and gain levels must be the same."
        raise ValueError(err_msg)
    return [
        (audio_track, gain_level)
        for audio_track, gain_level in zip(audio_tracks, gain_levels, strict=True)
        if audio_track
    ]


def render(
    voice_model: gr.Dropdown,
    custom_embedder_model: gr.Dropdown,
    cached_song_multi: gr.Dropdown,
    song_dirs: Sequence[gr.Dropdown],
    cached_song_1click: gr.Dropdown,
    intermediate_audio: gr.Dropdown,
    output_audio: gr.Dropdown,
    cookiefile: str | None = None,
) -> None:
    """
    Render "Generate song cover - multi-step generation" tab.

    Parameters
    ----------
    voice_model : gr.Dropdown
        Dropdown for selecting a voice model in the
        "Generate song covers - multi-step generation" tab.
    custom_embedder_model : gr.Dropdown
        Dropdown for selecting a custom embedder model in the
        "Generate song covers - multi-step generation" tab.
    cached_song_multi : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Generate song covers - multi-step generation" tab.
    song_dirs : Sequence[gr.Dropdown]
        Dropdowns for selecting song directories in the
        "Generate song covers - multi-step generation" tab.
    intermediate_audio : gr.Dropdown
        Dropdown for selecting intermediate audio files to delete in the
        "Delete audio" tab.
    cached_song_1click : gr.Dropdown
        Dropdown for selecting a cached song in the
        "Generate song covers - one-click generation" tab.
        Dropdown for selecting intermediate audio files to delete in the
        "Delete audio" tab.
    output_audio : gr.Dropdown
        Dropdown for selecting output audio files to delete in the
        "Delete audio" tab.
    cookiefile : str, optional
        The path to a file containing cookies to use when downloading
        audio from Youtube.

    """
    with gr.Tab("Multi-step generation"):
        (
            separate_audio_dir,
            convert_vocals_dir,
            postprocess_vocals_dir,
            pitch_shift_background_dir,
            mix_dir,
        ) = song_dirs
        current_song_dir = gr.State(None)

        input_tracks = [
            gr.Audio(label=label, type="filepath", render=False)
            for label in [
                "Audio",
                "Vocals",
                "Vocals",
                "Instrumentals",
                "Backup vocals",
                "Main vocals",
                "Instrumentals",
                "Backup vocals",
            ]
        ]
        (
            audio_track_input,
            vocals_track_input,
            converted_vocals_track_input,
            instrumentals_track_input,
            backup_vocals_track_input,
            main_vocals_track_input,
            shifted_instrumentals_track_input,
            shifted_backup_vocals_track_input,
        ) = input_tracks

        (
            song_output,
            primary_stem_output,
            secondary_stem_output,
            converted_vocals_track_output,
            effected_vocals_track_output,
            shifted_instrumentals_track_output,
            shifted_backup_vocals_track_output,
            song_cover_output,
        ) = [
            gr.Audio(label=label, type="filepath", interactive=False, render=False)
            for label in [
                "Song",
                "Primary stem",
                "Secondary stem",
                "Converted vocals",
                "Effected vocals",
                "Pitch-shifted instrumentals",
                "Pitch-shifted backup vocals",
                "Song cover",
            ]
        ]

        transfer_defaults = [
            ["Step 1: audio"],
            ["Step 2: vocals"],
            ["Step 4: instrumentals"],
            ["Step 3: vocals"],
            ["Step 5: main vocals"],
            ["Step 5: instrumentals"],
            ["Step 5: backup vocals"],
            [],
        ]

        (
            song_transfer_default,
            primary_stem_transfer_default,
            secondary_stem_transfer_default,
            converted_vocals_transfer_default,
            effected_vocals_transfer_default,
            shifted_instrumentals_transfer_default,
            shifted_backup_vocals_transfer_default,
            song_cover_transfer_default,
        ) = transfer_defaults

        (
            song_transfer,
            primary_stem_transfer,
            secondary_stem_transfer,
            converted_vocals_transfer,
            effected_vocals_transfer,
            shifted_instrumentals_transfer,
            shifted_backup_vocals_transfer,
            song_cover_transfer,
        ) = [
            gr.Dropdown(
                [
                    "Step 1: audio",
                    "Step 2: vocals",
                    "Step 3: vocals",
                    "Step 4: instrumentals",
                    "Step 4: backup vocals",
                    "Step 5: main vocals",
                    "Step 5: instrumentals",
                    "Step 5: backup vocals",
                ],
                label=f"{label_prefix} destination",
                info=(
                    "Select the input track(s) to transfer the"
                    f" {label_prefix.lower()} to when the 'Transfer"
                    f" {label_prefix.lower()}' button is clicked."
                ),
                render=False,
                type="index",
                multiselect=True,
                value=value,
            )
            for value, label_prefix in zip(
                transfer_defaults,
                [
                    "Song",
                    "Primary stem",
                    "Secondary stem",
                    "Converted vocals",
                    "Effected vocals",
                    "Pitch-shifted instrumentals",
                    "Pitch-shifted backup vocals",
                    "Song cover",
                ],
                strict=True,
            )
        ]

        with gr.Accordion("Step 0: song retrieval", open=True):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
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
                    cached_song_multi.render()

                source_type.input(
                    partial(toggle_visible_component, 3),
                    inputs=source_type,
                    outputs=[source, local_file, cached_song_multi],
                    show_progress="hidden",
                )

                local_file.change(
                    update_value,
                    inputs=local_file,
                    outputs=source,
                    show_progress="hidden",
                )
                cached_song_multi.input(
                    update_value,
                    inputs=cached_song_multi,
                    outputs=source,
                    show_progress="hidden",
                )
            gr.Markdown("**Settings**")
            song_transfer.render()
            gr.Markdown("**Outputs**")
            song_output.render()
            gr.Markdown("**Controls**")
            retrieve_song_btn = gr.Button("Retrieve song", variant="primary")
            song_transfer_btn = gr.Button("Transfer song")
            retrieve_song_reset_btn = gr.Button("Reset settings")

            retrieve_song_reset_btn.click(
                lambda: gr.Dropdown(value=song_transfer_default),
                outputs=song_transfer,
                show_progress="hidden",
            )
            retrieve_song_btn.click(
                partial(
                    exception_harness(
                        retrieve_song,
                        info_msg="Song retrieved successfully!",
                    ),
                    cookiefile=cookiefile,
                ),
                inputs=source,
                outputs=[song_output, current_song_dir],
            ).then(
                partial(
                    update_dropdowns,
                    get_named_song_dirs,
                    len(song_dirs) + 2,
                    value_indices=range(len(song_dirs)),
                ),
                inputs=current_song_dir,
                outputs=([*song_dirs, cached_song_multi, cached_song_1click]),
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_named_song_dirs, 1, [], [0]),
                outputs=intermediate_audio,
                show_progress="hidden",
            )

        with gr.Accordion("Step 1: vocal separation", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            audio_track_input.render()
            separate_audio_dir.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                separation_model = gr.Dropdown(
                    list(SeparationModel),
                    value=SeparationModel.UVR_MDX_NET_VOC_FT,
                    label="Separation model",
                    info="The model to use for audio separation.",
                )
                segment_size = gr.Radio(
                    list(SegmentSize),
                    value=SegmentSize.SEG_512,
                    label="Segment size",
                    info=(
                        "Size of segments into which the audio is split. Larger"
                        " consumes more resources, but may give better results."
                    ),
                )
            with gr.Row():
                primary_stem_transfer.render()
                secondary_stem_transfer.render()

            gr.Markdown("**Outputs**")
            with gr.Row():
                primary_stem_output.render()
                secondary_stem_output.render()
            gr.Markdown("**Controls**")
            separate_vocals_btn = gr.Button("Separate vocals", variant="primary")
            with gr.Row():
                primary_stem_transfer_btn = gr.Button("Transfer primary stem")
                secondary_stem_transfer_btn = gr.Button("Transfer secondary stem")
            separate_audio_reset_btn = gr.Button("Reset settings")

            separate_audio_reset_btn.click(
                lambda: [
                    SeparationModel.UVR_MDX_NET_VOC_FT,
                    SegmentSize.SEG_512,
                    gr.Dropdown(value=primary_stem_transfer_default),
                    gr.Dropdown(value=secondary_stem_transfer_default),
                ],
                outputs=[
                    separation_model,
                    segment_size,
                    primary_stem_transfer,
                    secondary_stem_transfer,
                ],
                show_progress="hidden",
            )
            separate_vocals_btn.click(
                exception_harness(
                    separate_audio,
                    info_msg="Vocals separated successfully!",
                ),
                inputs=[
                    audio_track_input,
                    separate_audio_dir,
                    separation_model,
                    segment_size,
                ],
                outputs=[primary_stem_output, secondary_stem_output],
                concurrency_limit=1,
                concurrency_id=ConcurrencyId.GPU,
            )
        with gr.Accordion("Step 2: vocal conversion", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            vocals_track_input.render()
            with gr.Row():
                convert_vocals_dir.render()
                voice_model.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                n_octaves = gr.Slider(
                    -3,
                    3,
                    value=0,
                    step=1,
                    label="Pitch shift (octaves)",
                    info=(
                        "The number of octaves to pitch-shift the converted vocals by."
                        " Use 1 for male-to-female and -1 for vice-versa."
                    ),
                    show_reset_button=False,
                )
                n_semitones = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Pitch shift (semi-tones)",
                    info=(
                        "The number of semi-tones to pitch-shift the converted vocals"
                        " by. Altering this slightly reduces sound quality."
                    ),
                    show_reset_button=False,
                )
            with gr.Accordion("Voice synthesis settings", open=False):
                with gr.Row():
                    f0_methods = gr.Dropdown(
                        list(F0Method),
                        value=F0Method.RMVPE,
                        label="Pitch extraction method(s)",
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
                            " coming from the voice model."
                        ),
                        show_reset_button=False,
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
                            " vocals.<br><br>"
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
            with gr.Accordion("Vocal enrichment settings", open=False), gr.Row():
                with gr.Column():
                    split_vocals = gr.Checkbox(
                        label="Split vocal track",
                        info=(
                            "Whether to split the vocals track into smaller segments"
                            " before converting it. This can improve output quality for"
                            " longer vocal tracks."
                        ),
                    )
                with gr.Column():
                    autotune_vocals = gr.Checkbox(
                        label="Autotune converted vocals",
                        info=(
                            "Whether to apply autotune to the converted vocals.<br><br>"
                        ),
                    )
                    autotune_strength = gr.Slider(
                        0,
                        1,
                        value=1.0,
                        label="Autotune intensity",
                        info=(
                            "Higher values result in stronger snapping to the chromatic"
                            " grid and artifacting."
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
                            "Higher values result in stronger cleaning, but may lead to"
                            " a more compressed sound."
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
            with gr.Accordion("Speaker embeddings settings", open=False), gr.Row():
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

            converted_vocals_transfer.render()
            gr.Markdown("**Outputs**")
            converted_vocals_track_output.render()
            gr.Markdown("**Controls**")
            convert_vocals_btn = gr.Button("Convert vocals", variant="primary")
            converted_vocals_transfer_btn = gr.Button("Transfer converted vocals")
            convert_vocals_reset_btn = gr.Button("Reset settings")

            convert_vocals_reset_btn.click(
                lambda: [
                    0,
                    0,
                    F0Method.RMVPE,
                    0.5,
                    3,
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
                    gr.Dropdown(value=converted_vocals_transfer_default),
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
                    split_vocals,
                    autotune_vocals,
                    autotune_strength,
                    clean_vocals,
                    clean_strength,
                    embedder_model,
                    custom_embedder_model,
                    sid,
                    converted_vocals_transfer,
                ],
                show_progress="hidden",
            )
            convert_vocals_btn.click(
                partial(
                    exception_harness(
                        convert,
                        info_msg="Vocals converted successfully!",
                    ),
                    content_type=RVCContentType.VOCALS,
                ),
                inputs=[
                    vocals_track_input,
                    convert_vocals_dir,
                    voice_model,
                    n_octaves,
                    n_semitones,
                    f0_methods,
                    index_rate,
                    filter_radius,
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
                ],
                outputs=converted_vocals_track_output,
                concurrency_id=ConcurrencyId.GPU,
                concurrency_limit=1,
            )
        with gr.Accordion("Step 3: vocal post-processing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            converted_vocals_track_input.render()
            postprocess_vocals_dir.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                room_size = gr.Slider(
                    0,
                    1,
                    value=0.15,
                    label="Room size",
                    info=(
                        "Size of the room which reverb effect simulates. Increase for"
                        " longer reverb time."
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
                    info="Loudness of converted vocals without reverb effect applied.",
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

            effected_vocals_transfer.render()
            gr.Markdown("**Outputs**")

            effected_vocals_track_output.render()
            gr.Markdown("**Controls**")
            postprocess_vocals_btn = gr.Button(
                "Post-process vocals",
                variant="primary",
            )
            effected_vocals_transfer_btn = gr.Button("Transfer effected vocals")
            postprocess_vocals_reset_btn = gr.Button("Reset settings")

            postprocess_vocals_reset_btn.click(
                lambda: [
                    0.15,
                    0.2,
                    0.8,
                    0.7,
                    gr.Dropdown(value=effected_vocals_transfer_default),
                ],
                outputs=[
                    room_size,
                    wet_level,
                    dry_level,
                    damping,
                    effected_vocals_transfer,
                ],
                show_progress="hidden",
            )
            postprocess_vocals_btn.click(
                exception_harness(
                    postprocess,
                    info_msg="Vocals post-processed successfully!",
                ),
                inputs=[
                    converted_vocals_track_input,
                    postprocess_vocals_dir,
                    room_size,
                    wet_level,
                    dry_level,
                    damping,
                ],
                outputs=effected_vocals_track_output,
            )
        with gr.Accordion("Step 4: pitch shift of background audio", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                instrumentals_track_input.render()
                backup_vocals_track_input.render()
            pitch_shift_background_dir.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                n_semitones_instrumentals = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Instrumental pitch shift",
                    info="The number of semi-tones to pitch-shift the instrumentals by",
                    show_reset_button=False,
                )
                n_semitones_backup_vocals = gr.Slider(
                    -12,
                    12,
                    value=0,
                    step=1,
                    label="Backup vocal pitch shift",
                    info="The number of semi-tones to pitch-shift the backup vocals by",
                    show_reset_button=False,
                )
            with gr.Row():
                shifted_instrumentals_transfer.render()
                shifted_backup_vocals_transfer.render()

            gr.Markdown("**Outputs**")
            with gr.Row():
                shifted_instrumentals_track_output.render()
                shifted_backup_vocals_track_output.render()
            gr.Markdown("**Controls**")
            with gr.Row():
                pitch_shift_instrumentals_btn = gr.Button(
                    "Pitch shift instrumentals",
                    variant="primary",
                )
                pitch_shift_backup_vocals_btn = gr.Button(
                    "Pitch shift backup vocals",
                    variant="primary",
                )
            with gr.Row():
                shifted_instrumentals_transfer_btn = gr.Button(
                    "Transfer pitch-shifted instrumentals",
                )
                shifted_backup_vocals_transfer_btn = gr.Button(
                    "Transfer pitch-shifted backup vocals",
                )
            pitch_shift_background_reset_btn = gr.Button("Reset settings")

            pitch_shift_background_reset_btn.click(
                lambda: [
                    0,
                    0,
                    gr.Dropdown(value=shifted_instrumentals_transfer_default),
                    gr.Dropdown(value=shifted_backup_vocals_transfer_default),
                ],
                outputs=[
                    n_semitones_instrumentals,
                    n_semitones_backup_vocals,
                    shifted_instrumentals_transfer,
                    shifted_backup_vocals_transfer,
                ],
                show_progress="hidden",
            )
            pitch_shift_instrumentals_btn.click(
                exception_harness(
                    pitch_shift,
                    info_msg="Instrumentals pitch-shifted successfully!",
                ),
                inputs=[
                    instrumentals_track_input,
                    pitch_shift_background_dir,
                    n_semitones_instrumentals,
                ],
                outputs=shifted_instrumentals_track_output,
            )
            pitch_shift_backup_vocals_btn.click(
                exception_harness(
                    pitch_shift,
                    info_msg="Backup vocals pitch-shifted successfully!",
                ),
                inputs=[
                    backup_vocals_track_input,
                    pitch_shift_background_dir,
                    n_semitones_backup_vocals,
                ],
                outputs=shifted_backup_vocals_track_output,
            )
        with gr.Accordion("Step 5: song mixing", open=False):
            gr.Markdown("")
            gr.Markdown("**Inputs**")
            with gr.Row():
                main_vocals_track_input.render()
                shifted_instrumentals_track_input.render()
                shifted_backup_vocals_track_input.render()
            mix_dir.render()
            gr.Markdown("**Settings**")
            with gr.Row():
                main_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Main gain",
                    info="The gain to apply to the main vocals.",
                    show_reset_button=False,
                )
                inst_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Instrumentals gain",
                    info="The gain to apply to the instrumentals.",
                    show_reset_button=False,
                )
                backup_gain = gr.Slider(
                    -20,
                    20,
                    value=0,
                    step=1,
                    label="Backup gain",
                    info="The gain to apply to the backup vocals.",
                    show_reset_button=False,
                )
            with gr.Row():
                output_name = gr.Textbox(
                    value=partial(
                        update_output_name,
                        get_song_cover_name,
                        False,  # noqa: FBT003,
                    ),
                    inputs=[main_vocals_track_input, mix_dir],
                    label="Output name",
                    placeholder="Ultimate RVC song cover",
                    info=(
                        "If no name is provided, a suitable name will be generated"
                        " automatically."
                    ),
                )
                output_sr = gr.Dropdown(
                    choices=list(SampleRate),
                    value=SampleRate.HZ_44100,
                    label="Output sample rate",
                    info="The sample rate to save the generated song in.",
                )
                output_format = gr.Dropdown(
                    list(AudioExt),
                    value=AudioExt.MP3,
                    label="Output format",
                    info="The format to save the generated song in.",
                )
            song_cover_transfer.render()
            gr.Markdown("**Outputs**")
            song_cover_output.render()
            gr.Markdown("**Controls**")
            mix_btn = gr.Button("Mix song cover", variant="primary")
            song_cover_transfer_btn = gr.Button("Transfer song cover")
            mix_reset_btn = gr.Button("Reset settings")

            mix_reset_btn.click(
                lambda: [
                    0,
                    0,
                    0,
                    SampleRate.HZ_44100,
                    AudioExt.MP3,
                    gr.Dropdown(value=song_cover_transfer_default),
                ],
                outputs=[
                    main_gain,
                    inst_gain,
                    backup_gain,
                    output_sr,
                    output_format,
                    song_cover_transfer,
                ],
                show_progress="hidden",
            )
            temp_audio_gains = gr.State()
            mix_btn.click(
                partial(
                    _pair_audio_tracks_and_gain,
                    [
                        main_vocals_track_input,
                        shifted_instrumentals_track_input,
                        shifted_backup_vocals_track_input,
                    ],
                    [main_gain, inst_gain, backup_gain],
                ),
                inputs={
                    main_vocals_track_input,
                    shifted_instrumentals_track_input,
                    shifted_backup_vocals_track_input,
                    main_gain,
                    inst_gain,
                    backup_gain,
                },
                outputs=temp_audio_gains,
            ).then(
                exception_harness(
                    mix_song,
                    info_msg="Song cover succesfully generated.",
                ),
                inputs=[
                    temp_audio_gains,
                    mix_dir,
                    output_sr,
                    output_format,
                    output_name,
                ],
                outputs=song_cover_output,
            ).then(
                partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
                outputs=output_audio,
                show_progress="hidden",
            )

        for btn, transfer, output in [
            (song_transfer_btn, song_transfer, song_output),
            (primary_stem_transfer_btn, primary_stem_transfer, primary_stem_output),
            (
                secondary_stem_transfer_btn,
                secondary_stem_transfer,
                secondary_stem_output,
            ),
            (
                converted_vocals_transfer_btn,
                converted_vocals_transfer,
                converted_vocals_track_output,
            ),
            (
                effected_vocals_transfer_btn,
                effected_vocals_transfer,
                effected_vocals_track_output,
            ),
            (
                shifted_instrumentals_transfer_btn,
                shifted_instrumentals_transfer,
                shifted_instrumentals_track_output,
            ),
            (
                shifted_backup_vocals_transfer_btn,
                shifted_backup_vocals_transfer,
                shifted_backup_vocals_track_output,
            ),
            (song_cover_transfer_btn, song_cover_transfer, song_cover_output),
        ]:
            btn.click(
                partial(update_audio, len(input_tracks)),
                inputs=[transfer, output],
                outputs=input_tracks,
                show_progress="hidden",
            )
