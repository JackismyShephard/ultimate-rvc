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
from ultimate_rvc.typing_extra import EmbedderModel, RVCContentType
from ultimate_rvc.web.common import (
    exception_harness,
    render_transfer_component,
    setup_transfer_event,
    toggle_visibility,
    toggle_visible_component,
    update_dropdowns,
    update_output_name,
    update_value,
)
from ultimate_rvc.web.typing_extra import ConcurrencyId, SongTransferOption

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ultimate_rvc.web.config.main import MultiStepSongGenerationConfig, TotalConfig


def render(total_config: TotalConfig, cookiefile: str | None = None) -> None:
    """
    Render "Generate song cover - multi-step generation" tab.

    Parameters
    ----------
    total_config : TotalConfig
        Model containing all component configuration settings for the
        Ultimate RVC web UI.
    cookiefile : str, optional
        The path to a file containing cookies to use when downloading
        audio from Youtube.

    """
    tab_config = total_config.song.multi_step
    for input_track in tab_config.input_audio.all:
        input_track.instantiate()
    with gr.Tab("Multi-step"):
        _render_step_0(total_config, cookiefile=cookiefile)
        _render_step_1(tab_config)
        _render_step_2(tab_config)
        _render_step_3(tab_config)
        _render_step_4(tab_config)
        _render_step_5(total_config, tab_config)


def _render_step_0(total_config: TotalConfig, cookiefile: str | None) -> None:
    tab_config = total_config.song.multi_step

    current_song_dir = gr.State(None)
    with gr.Accordion("Step 0: song retrieval", open=True):
        gr.Markdown("")
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
        with gr.Accordion("Options", open=False):
            song_transfer = _render_song_transfer(
                [SongTransferOption.STEP_1_AUDIO],
                "Song",
            )
        with gr.Row():
            retrieve_song_reset_btn = gr.Button("Reset options")
            retrieve_song_btn = gr.Button("Retrieve song", variant="primary")
        song_transfer_btn = gr.Button("Transfer song")
        song_output = gr.Audio(
            label="Song",
            type="filepath",
            interactive=False,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )

        retrieve_song_reset_btn.click(
            lambda: gr.Dropdown(value=[SongTransferOption.STEP_1_AUDIO]),
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
            inputs=tab_config.source.instance,
            outputs=[song_output, current_song_dir],
        ).then(
            partial(
                update_dropdowns,
                get_named_song_dirs,
                len(tab_config.song_dirs.all) + 2,
                value_indices=range(len(tab_config.song_dirs.all)),
            ),
            inputs=current_song_dir,
            outputs=[
                *tab_config.song_dirs.all,
                tab_config.cached_song.instance,
                total_config.song.one_click.cached_song.instance,
            ],
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_named_song_dirs, 1, [], [0]),
            outputs=total_config.management.audio.intermediate.instance,
            show_progress="hidden",
        )
        setup_transfer_event(
            song_transfer_btn,
            song_transfer,
            song_output,
            tab_config.input_audio.all,
        )


def _render_step_1(tab_config: MultiStepSongGenerationConfig) -> None:
    with gr.Accordion("Step 1: vocal separation", open=False):
        tab_config.input_audio.audio.instance.render()
        tab_config.song_dirs.separate_audio.instance.render()
        with gr.Accordion("Options", open=False):
            with gr.Row():
                tab_config.separation_model.instantiate()
                tab_config.segment_size.instantiate()
            with gr.Row():
                primary_stem_transfer = _render_song_transfer(
                    [SongTransferOption.STEP_2_VOCALS],
                    "Primary stem",
                )
                secondary_stem_transfer = _render_song_transfer(
                    [SongTransferOption.STEP_4_INSTRUMENTALS],
                    "Secondary stem",
                )
        with gr.Row():
            separate_audio_reset_btn = gr.Button("Reset options")
            separate_vocals_btn = gr.Button("Separate vocals", variant="primary")
        with gr.Row():
            primary_stem_transfer_btn = gr.Button("Transfer primary stem")
            secondary_stem_transfer_btn = gr.Button("Transfer secondary stem")

        with gr.Row():
            primary_stem_output = gr.Audio(
                label="Primary stem",
                type="filepath",
                interactive=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )
            secondary_stem_output = gr.Audio(
                label="Secondary stem",
                type="filepath",
                interactive=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )

        separate_audio_reset_btn.click(
            lambda: [
                tab_config.separation_model.value,
                tab_config.segment_size.value,
                gr.Dropdown(value=[SongTransferOption.STEP_2_VOCALS]),
                gr.Dropdown(value=[SongTransferOption.STEP_4_INSTRUMENTALS]),
            ],
            outputs=[
                tab_config.separation_model.instance,
                tab_config.segment_size.instance,
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
                tab_config.input_audio.audio.instance,
                tab_config.song_dirs.separate_audio.instance,
                tab_config.separation_model.instance,
                tab_config.segment_size.instance,
            ],
            outputs=[primary_stem_output, secondary_stem_output],
            concurrency_limit=1,
            concurrency_id=ConcurrencyId.GPU,
        )
        for btn, transfer, output in [
            (primary_stem_transfer_btn, primary_stem_transfer, primary_stem_output),
            (
                secondary_stem_transfer_btn,
                secondary_stem_transfer,
                secondary_stem_output,
            ),
        ]:
            setup_transfer_event(
                btn,
                transfer,
                output,
                tab_config.input_audio.all,
            )


def _render_step_2(tab_config: MultiStepSongGenerationConfig) -> None:
    with gr.Accordion("Step 2: vocal conversion", open=False):
        tab_config.input_audio.vocals.instance.render()
        tab_config.voice_model.instance.render()
        tab_config.song_dirs.convert_vocals.instance.render()
        with gr.Accordion("Options", open=False):
            with gr.Row():
                tab_config.n_octaves.instantiate()
                tab_config.n_semitones.instantiate()

            converted_vocals_transfer = _render_song_transfer(
                [SongTransferOption.STEP_3_VOCALS],
                "Converted vocals",
            )
            with gr.Accordion("Advanced", open=False):
                with gr.Accordion("Voice synthesis", open=False):
                    with gr.Row():
                        tab_config.f0_method.instantiate()
                        tab_config.index_rate.instantiate()
                    with gr.Row():
                        tab_config.rms_mix_rate.instantiate()
                        tab_config.protect_rate.instantiate()
                _render_step_2_vocal_enrichment(tab_config)
                with gr.Accordion("Speaker embeddings", open=False), gr.Row():
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
        with gr.Row():
            convert_vocals_reset_btn = gr.Button("Reset options")
            convert_vocals_btn = gr.Button("Convert vocals", variant="primary")
        converted_vocals_transfer_btn = gr.Button("Transfer converted vocals")
        converted_vocals_track_output = gr.Audio(
            label="Converted vocals",
            type="filepath",
            interactive=False,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )

        convert_vocals_reset_btn.click(
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
                gr.Dropdown(value=[SongTransferOption.STEP_3_VOCALS]),
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
                converted_vocals_transfer,
            ],
            show_progress="hidden",
        )
        convert_vocals_btn.click(
            partial(
                exception_harness(convert, info_msg="Vocals converted successfully!"),
                content_type=RVCContentType.VOCALS,
            ),
            inputs=[
                tab_config.input_audio.vocals.instance,
                tab_config.song_dirs.convert_vocals.instance,
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
            ],
            outputs=converted_vocals_track_output,
            concurrency_id=ConcurrencyId.GPU,
            concurrency_limit=1,
        )
        setup_transfer_event(
            converted_vocals_transfer_btn,
            converted_vocals_transfer,
            converted_vocals_track_output,
            tab_config.input_audio.all,
        )


def _render_step_2_vocal_enrichment(tab_config: MultiStepSongGenerationConfig) -> None:
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


def _render_step_3(tab_config: MultiStepSongGenerationConfig) -> None:
    with gr.Accordion("Step 3: vocal post-processing", open=False):
        tab_config.input_audio.converted_vocals.instance.render()
        tab_config.song_dirs.postprocess_vocals.instance.render()
        with gr.Accordion("Options", open=False):
            tab_config.room_size.instantiate()
            with gr.Row():
                tab_config.wet_level.instantiate()
                tab_config.dry_level.instantiate()
                tab_config.damping.instantiate()
            effected_vocals_transfer = _render_song_transfer(
                [SongTransferOption.STEP_5_MAIN_VOCALS],
                "Effected vocals",
            )
        with gr.Row():
            postprocess_vocals_reset_btn = gr.Button("Reset options")
            postprocess_vocals_btn = gr.Button("Post-process vocals", variant="primary")
        effected_vocals_transfer_btn = gr.Button("Transfer effected vocals")

        effected_vocals_track_output = gr.Audio(
            label="Effected vocals",
            type="filepath",
            interactive=False,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )

        postprocess_vocals_reset_btn.click(
            lambda: [
                tab_config.room_size.value,
                tab_config.wet_level.value,
                tab_config.dry_level.value,
                tab_config.damping.value,
                gr.Dropdown(value=[SongTransferOption.STEP_5_MAIN_VOCALS]),
            ],
            outputs=[
                tab_config.room_size.instance,
                tab_config.wet_level.instance,
                tab_config.dry_level.instance,
                tab_config.damping.instance,
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
                tab_config.input_audio.converted_vocals.instance,
                tab_config.song_dirs.postprocess_vocals.instance,
                tab_config.room_size.instance,
                tab_config.wet_level.instance,
                tab_config.dry_level.instance,
                tab_config.damping.instance,
            ],
            outputs=effected_vocals_track_output,
        )
        setup_transfer_event(
            effected_vocals_transfer_btn,
            effected_vocals_transfer,
            effected_vocals_track_output,
            tab_config.input_audio.all,
        )


def _render_step_4(tab_config: MultiStepSongGenerationConfig) -> None:
    with gr.Accordion("Step 4: pitch shift of background audio", open=False):
        with gr.Row():
            tab_config.input_audio.instrumentals.instance.render()
            tab_config.input_audio.backup_vocals.instance.render()
        with gr.Row():
            tab_config.n_semitones_instrumentals.instantiate()
            tab_config.n_semitones_backup_vocals.instantiate()
        tab_config.song_dirs.pitch_shift_background.instance.render()
        with gr.Accordion("Options", open=False), gr.Row():
            shifted_instrumentals_transfer = _render_song_transfer(
                [SongTransferOption.STEP_5_INSTRUMENTALS],
                "Pitch-shifted instrumentals",
            )
            shifted_backup_vocals_transfer = _render_song_transfer(
                [SongTransferOption.STEP_5_BACKUP_VOCALS],
                "Pitch-shifted backup vocals",
            )
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
                "Transfer shifted instrumentals",
            )
            shifted_backup_vocals_transfer_btn = gr.Button(
                "Transfer shifted backup vocals",
            )
        pitch_shift_background_reset_btn = gr.Button("Reset options")
        with gr.Row():
            shifted_instrumentals_track_output = gr.Audio(
                label="Pitch-shifted instrumentals",
                type="filepath",
                interactive=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )
            shifted_backup_vocals_track_output = gr.Audio(
                label="Pitch-shifted backup vocals",
                type="filepath",
                interactive=False,
                waveform_options=gr.WaveformOptions(show_recording_waveform=False),
            )

        pitch_shift_background_reset_btn.click(
            lambda: [
                tab_config.n_semitones_instrumentals.value,
                tab_config.n_semitones_backup_vocals.value,
                gr.Dropdown(value=[SongTransferOption.STEP_5_INSTRUMENTALS]),
                gr.Dropdown(value=[SongTransferOption.STEP_5_BACKUP_VOCALS]),
            ],
            outputs=[
                tab_config.n_semitones_instrumentals.instance,
                tab_config.n_semitones_backup_vocals.instance,
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
                tab_config.input_audio.instrumentals.instance,
                tab_config.song_dirs.pitch_shift_background.instance,
                tab_config.n_semitones_instrumentals.instance,
            ],
            outputs=shifted_instrumentals_track_output,
        )
        pitch_shift_backup_vocals_btn.click(
            exception_harness(
                pitch_shift,
                info_msg="Backup vocals pitch-shifted successfully!",
            ),
            inputs=[
                tab_config.input_audio.backup_vocals.instance,
                tab_config.song_dirs.pitch_shift_background.instance,
                tab_config.n_semitones_backup_vocals.instance,
            ],
            outputs=shifted_backup_vocals_track_output,
        )
        for btn, transfer, output in [
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
        ]:
            setup_transfer_event(
                btn,
                transfer,
                output,
                tab_config.input_audio.all,
            )


def _render_step_5(
    total_config: TotalConfig,
    tab_config: MultiStepSongGenerationConfig,
) -> None:
    with gr.Accordion("Step 5: song mixing", open=False):
        with gr.Row():
            tab_config.input_audio.main_vocals.instance.render()
            tab_config.input_audio.shifted_instrumentals.instance.render()
            tab_config.input_audio.shifted_backup_vocals.instance.render()
        tab_config.song_dirs.mix.instance.render()
        with gr.Accordion("Options", open=False):
            with gr.Row():
                tab_config.main_gain.instantiate()
                tab_config.inst_gain.instantiate()
                tab_config.backup_gain.instantiate()
            with gr.Row():
                tab_config.output_name.instantiate(
                    value=partial(
                        update_output_name,
                        get_song_cover_name,
                        False,  # noqa: FBT003,
                    ),
                    inputs=[
                        tab_config.input_audio.main_vocals.instance,
                        tab_config.song_dirs.mix.instance,
                    ],
                )
                tab_config.output_sr.instantiate()
                tab_config.output_format.instantiate()
            song_cover_transfer = _render_song_transfer([], "Song cover")
        with gr.Row():
            mix_reset_btn = gr.Button("Reset options")
            mix_btn = gr.Button("Mix song cover", variant="primary")
        song_cover_transfer_btn = gr.Button("Transfer song cover")
        song_cover_output = gr.Audio(
            label="Song cover",
            type="filepath",
            interactive=False,
            waveform_options=gr.WaveformOptions(show_recording_waveform=False),
        )
        mix_reset_btn.click(
            lambda: [
                tab_config.main_gain.value,
                tab_config.inst_gain.value,
                tab_config.backup_gain.value,
                tab_config.output_sr.value,
                tab_config.output_format.value,
                gr.Dropdown(value=[]),
            ],
            outputs=[
                tab_config.main_gain.instance,
                tab_config.inst_gain.instance,
                tab_config.backup_gain.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                song_cover_transfer,
            ],
            show_progress="hidden",
        )
        temp_audio_gains = gr.State()
        mix_btn.click(
            partial(
                _pair_audio_tracks_and_gain,
                [
                    tab_config.input_audio.main_vocals.instance,
                    tab_config.input_audio.shifted_instrumentals.instance,
                    tab_config.input_audio.shifted_backup_vocals.instance,
                ],
                [
                    tab_config.main_gain.instance,
                    tab_config.inst_gain.instance,
                    tab_config.backup_gain.instance,
                ],
            ),
            inputs={
                tab_config.input_audio.main_vocals.instance,
                tab_config.input_audio.shifted_instrumentals.instance,
                tab_config.input_audio.shifted_backup_vocals.instance,
                tab_config.main_gain.instance,
                tab_config.inst_gain.instance,
                tab_config.backup_gain.instance,
            },
            outputs=temp_audio_gains,
        ).then(
            exception_harness(mix_song, info_msg="Song cover succesfully generated."),
            inputs=[
                temp_audio_gains,
                tab_config.song_dirs.mix.instance,
                tab_config.output_sr.instance,
                tab_config.output_format.instance,
                tab_config.output_name.instance,
            ],
            outputs=song_cover_output,
        ).then(
            partial(update_dropdowns, get_saved_output_audio, 1, [], [0]),
            outputs=total_config.management.audio.output.instance,
            show_progress="hidden",
        )
        setup_transfer_event(
            song_cover_transfer_btn,
            song_cover_transfer,
            song_cover_output,
            tab_config.input_audio.all,
        )


def _render_song_transfer(
    value: list[SongTransferOption],
    label_prefix: str,
) -> gr.Dropdown:
    return render_transfer_component(value, label_prefix, SongTransferOption)


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
