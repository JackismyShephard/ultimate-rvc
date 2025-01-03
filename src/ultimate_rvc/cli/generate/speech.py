"""
Module which defines the command-line interface for using RVC-based
TTS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

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
from ultimate_rvc.common import lazy_import
from ultimate_rvc.typing_extra import AudioExt, EmbedderModel, F0Method

if TYPE_CHECKING:
    from ultimate_rvc.core.generate import speech as generate_speech
else:
    generate_speech = lazy_import("ultimate_rvc.core.generate.speech")

app = typer.Typer(
    name="speech",
    no_args_is_help=True,
    help="Generate speech from text using RVC.",
    rich_markup_mode="markdown",
)


@app.command(no_args_is_help=True)
def run_edge_tts(
    source: Annotated[
        str,
        typer.Argument(
            help="A string or path to a file containing the text to be converted.",
        ),
    ],
    voice: Annotated[
        str,
        typer.Option(
            help=(
                "The short name of the Edge TTS voice which should speak the provided"
                " text. Use the `list-edge-voices` command to get a list of available"
                " Edge TTS voices."
            ),
        ),
    ] = "en-US-ChristopherNeural",
    pitch_shift: Annotated[
        int,
        typer.Option(
            help=(
                "The number of hertz to shift the pitch of the Edge TTS voice"
                " speaking the provided text."
            ),
        ),
    ] = 0,
    speed_change: Annotated[
        int,
        typer.Option(
            help=(
                "The percentual change to the speed of the Edge TTS voice speaking the"
                " provided text."
            ),
        ),
    ] = 0,
    volume_change: Annotated[
        int,
        typer.Option(
            help=(
                "The percentual change to the volume of the Efge TTS voice speaking the"
                " provided text."
            ),
        ),
    ] = 0,
) -> None:
    """Convert text to speech using Edge TTS."""
    start_time = time.perf_counter()

    rprint()

    audio_path = generate_speech.run_edge_tts(
        source,
        voice,
        pitch_shift,
        speed_change,
        volume_change,
    )

    rprint("[+] Text successfully converted to speech!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{audio_path}", title="Speech Path"))


@app.command()
def list_edge_tts_voices(
    locale: Annotated[
        str | None,
        typer.Option(
            help="The locale to filter Edge TTS voices by.",
        ),
    ] = None,
    content_category: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "The content category to filter Edge TTS voices by. This option can be"
                " supplied multiple times to filter by multiple content categories."
            ),
        ),
    ] = None,
    voice_personality: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "The voice personality to filter Edge TTS voices by. This option can be"
                " supplied multiple times to filter by multiple voice personalities."
            ),
        ),
    ] = None,
    offset: Annotated[
        int,
        typer.Option(
            min=0,
            help="The offset to start listing Edge TTS voices from.",
        ),
    ] = 0,
    limit: Annotated[
        int,
        typer.Option(
            min=0,
            help="The limit on how many Edge TTS voices to list.",
        ),
    ] = 20,
    include_status_info: Annotated[
        bool,
        typer.Option(
            help="Include status information for each Edge TTS voice.",
        ),
    ] = False,
    include_codec_info: Annotated[
        bool,
        typer.Option(
            help="Include codec information for each Edge TTS voice.",
        ),
    ] = False,
) -> None:
    """List Edge TTS voices based on provided filters."""
    start_time = time.perf_counter()

    rprint()
    rprint("[~] Retrieving information on all available edge TTS voices...")

    voices, keys = generate_speech.list_edge_tts_voices(
        locale=locale,
        content_categories=content_category,
        voice_personalities=voice_personality,
        offset=offset,
        limit=limit,
        include_status_info=include_status_info,
        include_codec_info=include_codec_info,
    )

    table = Table()
    for key in keys:
        table.add_column(key)
    for voice in voices:
        table.add_row(*[f"[green]{voice_attrib}" for voice_attrib in voice])

    rprint("[+] Information successfully retrieved!")
    rprint()

    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(table, title="Available Edge TTS Voices"))


@app.command(no_args_is_help=True)
def run_pipeline(
    source: Annotated[
        str,
        typer.Argument(
            help="A string or path to a file containing the text to be converted.",
        ),
    ],
    model_name: Annotated[
        str,
        typer.Argument(
            help="The name of the RVC model to use for speech conversion.",
        ),
    ],
    tts_voice: Annotated[
        str,
        typer.Option(
            help="The short name of the Edge TTS voice to use for text-to-speech"
            " conversion. Use the `list-edge-voices` command to get a list of available"
            " Edge TTS voices.",
            rich_help_panel=PanelName.EDGE_TTS_OPTIONS,
        ),
    ] = "en-US-ChristopherNeural",
    tts_pitch_shift: Annotated[
        int,
        typer.Option(
            help=(
                "The number of hertz to shift the pitch of the speech generated"
                " by Edge TTS."
            ),
            rich_help_panel=PanelName.EDGE_TTS_OPTIONS,
        ),
    ] = 0,
    tts_speed_change: Annotated[
        int,
        typer.Option(
            help=(
                "The percentual change to the speed of the speech generated by Edge"
                " TTS."
            ),
            rich_help_panel=PanelName.EDGE_TTS_OPTIONS,
        ),
    ] = 0,
    tts_volume_change: Annotated[
        int,
        typer.Option(
            help=(
                "The percentual change to the volume of the speech generated by Edge"
                " TTS."
            ),
            rich_help_panel=PanelName.EDGE_TTS_OPTIONS,
        ),
    ] = 0,
    n_octaves: Annotated[
        int,
        typer.Option(
            help=(
                "The number of octaves to shift the pitch of the speech converted using"
                " RVC. Use 1 for male-to-female and -1 for vice-versa."
            ),
            rich_help_panel=PanelName.RVC_MAIN_OPTIONS,
        ),
    ] = 0,
    n_semitones: Annotated[
        int,
        typer.Option(
            help=(
                "The number of semitones to shift the pitch of the speech converted"
                " using RVC. Altering this slightly reduces sound quality."
            ),
            rich_help_panel=PanelName.RVC_MAIN_OPTIONS,
        ),
    ] = 0,
    f0_method: Annotated[
        list[F0Method] | None,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_f0_method,
            rich_help_panel=PanelName.RVC_SYNTHESIS_OPTIONS,
            help=(
                "The method to use for pitch extraction during the RVC process. This"
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
            rich_help_panel=PanelName.RVC_SYNTHESIS_OPTIONS,
            help=(
                "The rate of influence of the RVC index file. Increase to"
                " bias the conversion towards the accent of the used voice model."
                " Decrease to potentially reduce artifacts."
            ),
        ),
    ] = 0.5,
    filter_radius: Annotated[
        int,
        typer.Option(
            min=0,
            max=7,
            rich_help_panel=PanelName.RVC_SYNTHESIS_OPTIONS,
            help=(
                "A number which, if greater than 3, applies median filtering to"
                " pitch values extracted during the RVC process. Can help reduce"
                " breathiness in the converted speech."
            ),
        ),
    ] = 3,
    rms_mix_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.RVC_SYNTHESIS_OPTIONS,
            help=(
                "Blending rate for the volume envelope of the speech converted using"
                " RVC. Controls how much to mimic the loudness of the given Edge TTS"
                " speech (0) or a fixed loudness (1)."
            ),
        ),
    ] = 0.25,
    protect_rate: Annotated[
        float,
        typer.Option(
            min=0,
            max=0.5,
            rich_help_panel=PanelName.RVC_SYNTHESIS_OPTIONS,
            help=(
                "A coefficient which controls the extent to which consonants and"
                " breathing sounds are protected from artifacts during the RVC"
                " process. A higher value offers more protection but may worsen the"
                " indexing effect."
            ),
        ),
    ] = 0.33,
    hop_length: Annotated[
        int,
        typer.Option(
            min=1,
            max=512,
            rich_help_panel=PanelName.RVC_SYNTHESIS_OPTIONS,
            help=(
                "Controls how often the CREPE-based pitch extraction method checks for"
                " pitch changes during the RVC process. Measured in milliseconds."
                " Lower values lead to longer conversion times and a higher risk of"
                " voice cracks, but better pitch accuracy."
            ),
        ),
    ] = 128,
    split_speech: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.RVC_ENRICHMENT_OPTIONS,
            help=(
                "Whether to split the Edge TTS speech into smaller segments before"
                " converting it using RVC. This can improve output quality for longer"
                " speech."
            ),
        ),
    ] = True,
    autotune_speech: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.RVC_ENRICHMENT_OPTIONS,
            help="Whether to apply autotune to the speech converted using RVC.",
        ),
    ] = False,
    autotune_strength: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.RVC_ENRICHMENT_OPTIONS,
            help=(
                "The intensity of the autotune effect to apply to the speech converted"
                " using RVC. Higher values result in stronger snapping to the chromatic"
                " grid."
            ),
        ),
    ] = 1.0,
    clean_speech: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.RVC_ENRICHMENT_OPTIONS,
            help=(
                "Whether to clean the speech converted using RVC using noise reduction"
                " algorithms"
            ),
        ),
    ] = True,
    clean_strength: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            rich_help_panel=PanelName.RVC_ENRICHMENT_OPTIONS,
            help=(
                "The intensity of the cleaning to apply to the speech converted using"
                " RVC. Higher values result in stronger cleaning, but may lead to a"
                " more compressed sound."
            ),
        ),
    ] = 0.7,
    embedder_model: Annotated[
        EmbedderModel,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_embedder_model,
            rich_help_panel=PanelName.RVC_EMBEDDINGS_OPTIONS,
            help=(
                "The model to use for generating speaker embeddings during the RVC"
                " process."
            ),
        ),
    ] = EmbedderModel.CONTENTVEC,
    custom_embedder_model: Annotated[
        str | None,
        typer.Option(
            rich_help_panel=PanelName.RVC_EMBEDDINGS_OPTIONS,
            help=(
                "The name of a custom embedder model to use for generating speaker"
                " embeddings during the RVC process. Only applicable if"
                " `embedder-model` is set to `custom`."
            ),
        ),
    ] = None,
    sid: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.RVC_EMBEDDINGS_OPTIONS,
            help="The id of the speaker to use for multi-speaker RVC models.",
        ),
    ] = 0,
    output_gain: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help="The gain to apply to the converted speech during mixing.",
        ),
    ] = 0,
    output_sr: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help="The sample rate of the mixed speech track.",
        ),
    ] = 44100,
    output_format: Annotated[
        AudioExt,
        typer.Option(
            case_sensitive=False,
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            autocompletion=complete_audio_ext,
            help="The format of the mixed speech track.",
        ),
    ] = AudioExt.MP3,
    output_name: Annotated[
        str | None,
        typer.Option(
            rich_help_panel=PanelName.AUDIO_MIXING_OPTIONS,
            help="The name of the mixed speech track.",
        ),
    ] = None,
) -> None:
    """
    Convert text to speech using a cascaded pipeline combining Edge TTS
    and RVC. The text is first converted to speech using Edge TTS, and
    then that speech is converted to a different voice using RVC.
    """
    start_time = time.perf_counter()

    rprint()

    [output_audio_path, *intermediate_audio_file_paths] = generate_speech.run_pipeline(
        source=source,
        model_name=model_name,
        tts_voice=tts_voice,
        tts_pitch_shift=tts_pitch_shift,
        tts_speed_change=tts_speed_change,
        tts_volume_change=tts_volume_change,
        n_octaves=n_octaves,
        n_semitones=n_semitones,
        f0_methods=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        protect_rate=protect_rate,
        hop_length=hop_length,
        split_speech=split_speech,
        autotune_speech=autotune_speech,
        autotune_strength=autotune_strength,
        clean_speech=clean_speech,
        clean_strength=clean_strength,
        embedder_model=embedder_model,
        custom_embedder_model=custom_embedder_model,
        sid=sid,
        output_gain=output_gain,
        output_sr=output_sr,
        output_format=output_format,
        output_name=output_name,
    )

    table = Table()
    table.add_column("Type")
    table.add_column("Path")
    for name, path in zip(
        [
            "Speech",
            "Converted Speech",
        ],
        intermediate_audio_file_paths,
        strict=True,
    ):
        table.add_row(name, f"[green]{path}")

    rprint("[+] Text successfully converted to speech!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{output_audio_path}", title="Mixed Speech Path"))
    rprint(Panel(table, title="Intermediate Audio Files"))
