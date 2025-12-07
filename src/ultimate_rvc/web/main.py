"""
Web application for the Ultimate RVC project.

Each tab of the application is defined in its own module in the
`web/tabs` directory. Components that are accessed across multiple
tabs are passed as arguments to the render functions in the respective
modules.
"""

from __future__ import annotations

from typing import Annotated

import os
from pathlib import Path

import gradio as gr

import typer

from ultimate_rvc.common import AUDIO_DIR, MODELS_DIR, TEMP_DIR
from ultimate_rvc.core.generate.song_cover import get_named_song_dirs
from ultimate_rvc.core.generate.speech import get_edge_tts_voice_names
from ultimate_rvc.core.manage.audio import (
    get_audio_datasets,
    get_named_audio_datasets,
    get_saved_output_audio,
    get_saved_speech_audio,
)
from ultimate_rvc.core.manage.config import get_config_names, load_config
from ultimate_rvc.core.manage.models import (
    get_custom_embedder_model_names,
    get_custom_pretrained_model_names,
    get_training_model_names,
    get_voice_model_names,
)
from ultimate_rvc.web.common import initialize_dropdowns
from ultimate_rvc.web.config.main import TotalConfig
from ultimate_rvc.web.tabs.generate.song_cover.multi_step_generation import (
    render as render_song_cover_multi_step_tab,
)
from ultimate_rvc.web.tabs.generate.song_cover.one_click_generation import (
    render as render_song_cover_one_click_tab,
)
from ultimate_rvc.web.tabs.generate.speech.multi_step_generation import (
    render as render_speech_multi_step_tab,
)
from ultimate_rvc.web.tabs.generate.speech.one_click_generation import (
    render as render_speech_one_click_tab,
)
from ultimate_rvc.web.tabs.manage.audio import render as render_audio_tab
from ultimate_rvc.web.tabs.manage.models import render as render_models_tab
from ultimate_rvc.web.tabs.manage.settings import render as render_settings_tab

config_name = os.environ.get("URVC_CONFIG")
cookiefile = os.environ.get("YT_COOKIEFILE")
total_config = load_config(config_name, TotalConfig) if config_name else TotalConfig()


def render_app() -> gr.Blocks:
    """
    Render the Ultimate RVC web application.

    Returns
    -------
    gr.Blocks
        The rendered web application.

    """
    css = """
    h1 { text-align: center; margin-top: 20px; margin-bottom: 20px; }

    #generate-tab-button { font-weight: bold !important;}
    #manage-tab-button { font-weight: bold !important;}
    #audio-tab-button { font-weight: bold !important;}
    #settings-tab-button { font-weight: bold !important;}
    """
    cache_delete_frequency = 86400  # every 24 hours check for files to delete
    cache_delete_cutoff = 86400  # and delete files older than 24 hours

    with gr.Blocks(
        title="Ultimate RVC",
        theme=gr.Theme.load(str(Path(__file__).parent / "config/theme.json")),
        css=css,
        delete_cache=(cache_delete_frequency, cache_delete_cutoff),
    ) as app:
        gr.HTML("<h1>Ultimate RVC 💙</h1>")
        for component_config in [
            total_config.song.one_click.voice_model,
            total_config.song.one_click.cached_song,
            total_config.song.one_click.custom_embedder_model,
            total_config.song.multi_step.voice_model,
            total_config.song.multi_step.cached_song,
            total_config.song.multi_step.custom_embedder_model,
            total_config.song.multi_step.song_dirs.separate_audio,
            total_config.song.multi_step.song_dirs.convert_vocals,
            total_config.song.multi_step.song_dirs.postprocess_vocals,
            total_config.song.multi_step.song_dirs.pitch_shift_background,
            total_config.song.multi_step.song_dirs.mix,
            total_config.speech.one_click.edge_tts_voice,
            total_config.speech.one_click.voice_model,
            total_config.speech.one_click.custom_embedder_model,
            total_config.speech.multi_step.edge_tts_voice,
            total_config.speech.multi_step.voice_model,
            total_config.speech.multi_step.custom_embedder_model,
            total_config.training.multi_step.dataset,
            total_config.training.multi_step.preprocess_model,
            total_config.training.multi_step.extract_model,
            total_config.training.multi_step.train_model,
            total_config.training.multi_step.custom_embedder_model,
            total_config.training.multi_step.custom_pretrained_model,
            total_config.management.audio.intermediate,
            total_config.management.audio.speech,
            total_config.management.audio.output,
            total_config.management.audio.dataset,
            total_config.management.model.voices,
            total_config.management.model.embedders,
            total_config.management.model.pretraineds,
            total_config.management.model.traineds,
            total_config.management.settings.load_config_name,
            total_config.management.settings.delete_config_names,
        ]:
            component_config.instantiate()
        # main tab
        with gr.Tab("Generate", elem_id="generate-tab"):
            with gr.Tab("Song covers"):
                render_song_cover_one_click_tab(total_config, cookiefile)
                render_song_cover_multi_step_tab(total_config, cookiefile)
            with gr.Tab("Speech"):
                render_speech_one_click_tab(total_config)
                render_speech_multi_step_tab(total_config)
        with gr.Tab("Models", elem_id="manage-tab"):
            render_models_tab(total_config)
        with gr.Tab("Audio", elem_id="audio-tab"):
            render_audio_tab(total_config)
        with gr.Tab("Settings", elem_id="settings-tab"):
            render_settings_tab(total_config)

        app.load(
            _init_dropdowns,
            outputs=[
                total_config.speech.one_click.edge_tts_voice.instance,
                total_config.speech.multi_step.edge_tts_voice.instance,
                total_config.song.one_click.voice_model.instance,
                total_config.song.multi_step.voice_model.instance,
                total_config.speech.one_click.voice_model.instance,
                total_config.speech.multi_step.voice_model.instance,
                total_config.management.model.voices.instance,
                total_config.song.one_click.custom_embedder_model.instance,
                total_config.song.multi_step.custom_embedder_model.instance,
                total_config.speech.one_click.custom_embedder_model.instance,
                total_config.speech.multi_step.custom_embedder_model.instance,
                total_config.training.multi_step.custom_embedder_model.instance,
                total_config.management.model.embedders.instance,
                total_config.training.multi_step.custom_pretrained_model.instance,
                total_config.management.model.pretraineds.instance,
                total_config.training.multi_step.extract_model.instance,
                total_config.training.multi_step.train_model.instance,
                total_config.training.multi_step.preprocess_model.instance,
                total_config.management.model.traineds.instance,
                total_config.song.one_click.cached_song.instance,
                total_config.song.multi_step.cached_song.instance,
                total_config.song.multi_step.song_dirs.separate_audio.instance,
                total_config.song.multi_step.song_dirs.convert_vocals.instance,
                total_config.song.multi_step.song_dirs.postprocess_vocals.instance,
                total_config.song.multi_step.song_dirs.pitch_shift_background.instance,
                total_config.song.multi_step.song_dirs.mix.instance,
                total_config.management.audio.intermediate.instance,
                total_config.training.multi_step.dataset.instance,
                total_config.management.audio.speech.instance,
                total_config.management.audio.output.instance,
                total_config.management.audio.dataset.instance,
                total_config.management.settings.load_config_name.instance,
                total_config.management.settings.delete_config_names.instance,
            ],
            show_progress="hidden",
        )
    return app


def _init_dropdowns() -> list[gr.Dropdown]:
    """
    Initialize the Ultimate RVC web application by updating the choices
    and default values of non-static dropdown components.

    Returns
    -------
    tuple[gr.Dropdown, ...]
        A tuple of gr.Dropdown components with updated choices and
        default values.

    """
    # Initialize model dropdowns
    edge_tts_models = initialize_dropdowns(
        get_edge_tts_voice_names,
        2,
        "en-US-ChristopherNeural",
        range(2),
    )
    voice_models = initialize_dropdowns(
        get_voice_model_names,
        5,
        value_indices=range(4),
    )
    custom_embedder_models = initialize_dropdowns(
        get_custom_embedder_model_names,
        6,
        value_indices=range(5),
    )
    custom_pretrained_models = initialize_dropdowns(
        get_custom_pretrained_model_names,
        2,
        value_indices=range(1),
    )
    training_models = initialize_dropdowns(
        get_training_model_names,
        4,
        value_indices=range(2),
    )
    song_dirs = initialize_dropdowns(
        get_named_song_dirs,
        8,
        value_indices=range(7),
    )
    dataset = gr.Dropdown(get_audio_datasets())
    speech_delete = gr.Dropdown(get_saved_speech_audio())
    output_delete = gr.Dropdown(get_saved_output_audio())
    dataset_delete = gr.Dropdown(get_named_audio_datasets())
    configs = initialize_dropdowns(get_config_names, 2, value_indices=range(1))
    return [
        *edge_tts_models,
        *voice_models,
        *custom_embedder_models,
        *custom_pretrained_models,
        *training_models,
        *song_dirs,
        dataset,
        speech_delete,
        output_delete,
        dataset_delete,
        *configs,
    ]


app = render_app()
app_wrapper = typer.Typer()


@app_wrapper.command()
def start_app(
    share: Annotated[
        bool,
        typer.Option("--share", "-s", help="Enable sharing"),
    ] = False,
    listen: Annotated[
        bool,
        typer.Option(
            "--listen",
            "-l",
            help="Make the web application reachable from your local network.",
        ),
    ] = False,
    listen_host: Annotated[
        str | None,
        typer.Option(
            "--listen-host",
            "-h",
            help="The hostname that the server will use.",
        ),
    ] = None,
    listen_port: Annotated[
        int | None,
        typer.Option(
            "--listen-port",
            "-p",
            help="The listening port that the server will use.",
        ),
    ] = None,
    ssr_mode: Annotated[
        bool,
        typer.Option(
            "--ssr-mode",
            help="Enable server-side rendering mode.",
        ),
    ] = False,
) -> None:
    """Run the Ultimate RVC web application."""
    os.environ["GRADIO_TEMP_DIR"] = str(TEMP_DIR)
    gr.set_static_paths([MODELS_DIR, AUDIO_DIR])
    app.queue()
    app.launch(
        share=share,
        server_name=(None if not listen else (listen_host or "0.0.0.0")),  # noqa: S104
        server_port=listen_port,
        ssr_mode=ssr_mode,
    )


if __name__ == "__main__":
    app_wrapper()
