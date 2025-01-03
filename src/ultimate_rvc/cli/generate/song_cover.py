"""
Module which defines the command-line interface for generating song
covers.
"""

from __future__ import annotations

from typing import Annotated

import time

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from ultimate_rvc.cli.common import (
    complete_audio_ext,
    complete_embedder_model,
    complete_f0_method,
    format_duration,
)
from ultimate_rvc.cli.typing_extra import PanelName
from ultimate_rvc.core.generate.song_cover import run_pipeline as _run_pipeline
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method

app = typer.Typer(
    name="song-cover",
    no_args_is_help=True,
    help="Generate song covers using RVC.",
    rich_markup_mode="markdown",
)


@app.command(no_args_is_help=True)
def run_pipeline(
    source: Annotated[
        str,
        typer.Argument(
            help=(
                "A Youtube URL, the path to a local audio file or the path to a"
                " song directory."
            ),
        ),
    ],
    model_name: Annotated[
        str,
        typer.Argument(help="The name of the voice model to use for vocal conversion."),
    ],
    n_octaves: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.MAIN_OPTIONS,
            help=(
                "The number of octaves to pitch-shift the converted vocals by. Use 1 "
                "for male-to-female and -1 for vice-versa."
            ),
        ),
    ] = 0,
    n_semitones: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.MAIN_OPTIONS,
            help=(
                "The number of semi-tones to pitch-shift the converted vocals,"
                " instrumentals, and backup vocals by. Altering this slightly reduces"
                " sound quality"
            ),
        ),
    ] = 0,
    f0_method: Annotated[
        list[F0Method] | None,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_f0_method,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "The method to use for pitch extraction during vocal conversion. This"
                " option can be provided multiple times to use multiple pitch"
                " extraction methods in combination. If not provided, will default to"
                " the rmvpe method, which is generally recommended."
            ),
        ),
    ] = None,
    index_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "The rate of influence of the index file. Increase to bias the vocal"
                " conversion towards the accent of the used voice model. Decrease to"
                " potentially reduce artifacts at the cost of accent accuracy."
            ),
        ),
    ] = 0.5,
    filter_radius: Annotated[
        int,
        typer.Option(
            min=0,
            max=7,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "A number which, if greater than 3, applies median filtering to the"
                " pitch values extracted during vocal conversion. Can help reduce"
                " breathiness in the converted vocals."
            ),
        ),
    ] = 3,
    rms_mix_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "Blending rate for the volume envelope of the vocals track. Controls"
                " how much to mimic the loudness of the input vocals (0) or a fixed"
                " loudness (1) during vocal conversion."
            ),
        ),
    ] = 0.25,
    protect_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=0.5,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "A coefficient which controls the extent to which consonants and"
                " breathing sounds are protected from artifacts during vocal"
                " conversion. A higher value offers more protection but may worsen the"
                " indexing effect."
            ),
        ),
    ] = 0.33,
    hop_length: Annotated[
        int,
        typer.Option(
            min=1,
            max=512,
            rich_help_panel=PanelName.VOICE_SYNTHESIS_OPTIONS,
            help=(
                "Controls how often the CREPE-based pitch extraction method checks for"
                " pitch changes during vocal conversion measured in milliseconds. Lower"
                " values lead to longer conversion times and a higher risk of voice"
                " cracks, but better pitch accuracy."
            ),
        ),
    ] = 128,
    split_vocals: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.VOCAL_ENRICHMENT_OPTIONS,
            help=(
                "Whether to split the main vocals track into smaller segments before"
                " converting it. This can improve output quality for longer main vocal"
                " tracks."
            ),
        ),
    ] = False,
    autotune_vocals: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.VOCAL_ENRICHMENT_OPTIONS,
            help="Whether to apply autotune to the converted vocals.",
        ),
    ] = False,
    autotune_strength: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOCAL_ENRICHMENT_OPTIONS,
            help=(
                "The intensity of the autotune effect to apply to the converted vocals."
                " Higher values result in stronger snapping to the chromatic grid."
            ),
        ),
    ] = 1.0,
    clean_vocals: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.VOCAL_ENRICHMENT_OPTIONS,
            help="Whether to apply noise reduction algorithms to the converted vocals.",
        ),
    ] = False,
    clean_strength: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOCAL_ENRICHMENT_OPTIONS,
            help=(
                "The intensity of the cleaning to apply to the converted vocals. Higher"
                " values result in stronger cleaning, but may lead to a more compressed"
                " sound."
            ),
        ),
    ] = 0.7,
    embedder_model: Annotated[
        EmbedderModel,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_embedder_model,
            rich_help_panel=PanelName.SPEAKER_EMBEDDINGS_OPTIONS,
            help=(
                "The model to use for generating speaker embeddings during vocal"
                " conversion."
            ),
        ),
    ] = EmbedderModel.CONTENTVEC,
    custom_embedder_model: Annotated[
        str | None,
        typer.Option(
            rich_help_panel=PanelName.SPEAKER_EMBEDDINGS_OPTIONS,
            help=(
                "The name of a custom embedder model to use for generating speaker"
                " embeddings during vocal conversion. Only applicable if"
                " `embedder-model` is set to `custom`."
            ),
        ),
    ] = None,
    sid: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.SPEAKER_EMBEDDINGS_OPTIONS,
            help="The id of the speaker to use for multi-speaker RVC models.",
        ),
    ] = 0,
    room_size: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOCAL_POST_PROCESSING_OPTIONS,
            help=(
                "The room size of the reverb effect to apply to the converted vocals."
                " Increase for longer reverb time. Should be a value between 0 and 1."
            ),
        ),
    ] = 0.15,
    wet_level: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOCAL_POST_PROCESSING_OPTIONS,
            help=(
                "The loudness of the converted vocals with reverb effect applied."
                " Should be a value between 0 and 1"
            ),
        ),
    ] = 0.2,
    dry_level: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOCAL_POST_PROCESSING_OPTIONS,
            help=(
                "The loudness of the converted vocals wihout reverb effect applied."
                " Should be a value between 0 and 1."
            ),
        ),
    ] = 0.8,
    damping: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.VOCAL_POST_PROCESSING_OPTIONS,
            help=(
                "The absorption of high frequencies in the reverb effect applied to the"
                " converted vocals. Should be a value between 0 and 1."
            ),
        ),
    ] = 0.7,
    main_gain: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help="The gain to apply to the post-processed vocals. Measured in dB.",
        ),
    ] = 0,
    inst_gain: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help=(
                "The gain to apply to the pitch-shifted instrumentals. Measured in dB."
            ),
        ),
    ] = 0,
    backup_gain: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help=(
                "The gain to apply to the pitch-shifted backup vocals. Measured in dB."
            ),
        ),
    ] = 0,
    output_sr: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help="The sample rate of the song cover.",
        ),
    ] = 44100,
    output_format: Annotated[
        AudioExt,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_audio_ext,
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help="The audio format of the song cover.",
        ),
    ] = AudioExt.MP3,
    output_name: Annotated[
        str | None,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help="The name of the song cover.",
        ),
    ] = None,
) -> None:
    """Run the song cover generation pipeline."""
    rprint()

    start_time = time.perf_counter()

    [song_cover_path, *intermediate_audio_file_paths] = _run_pipeline(
        source=source,
        model_name=model_name,
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_methods=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        split_vocals=split_vocals,
        autotune_vocals=autotune_vocals,
        autotune_strength=autotune_strength,
        clean_vocals=clean_vocals,
        clean_strength=clean_strength,
        embedder_model=embedder_model,
        custom_embedder_model=custom_embedder_model,
        sid=sid,
        room_size=room_size,
        wet_level=wet_level,
        dry_level=dry_level,
        damping=damping,
        main_gain=main_gain,
        inst_gain=inst_gain,
        backup_gain=backup_gain,
        output_sr=output_sr,
        output_format=output_format,
        output_name=output_name,
        progress_bar=None,
    )
    table = Table()
    table.add_column("Type")
    table.add_column("Path")
    for name, path in zip(
        [
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
        ],
        intermediate_audio_file_paths,
        strict=True,
    ):
        table.add_row(name, f"[green]{path}")
    rprint("[+] Song cover succesfully generated!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{song_cover_path}", title="Song Cover Path"))
    rprint(Panel(table, title="Intermediate Audio Files"))
