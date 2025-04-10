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
from ultimate_rvc.core.manage.config import load_config
from ultimate_rvc.core.manage.models import (
    get_custom_embedder_model_names,
    get_custom_pretrained_model_names,
    get_training_model_names,
    get_voice_model_names,
)
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
from ultimate_rvc.web.tabs.manage.audio import render as render_manage_audio_tab
from ultimate_rvc.web.tabs.manage.models import render as render_manage_models_tab
from ultimate_rvc.web.tabs.manage.settings import render as render_settings_tab
from ultimate_rvc.web.tabs.train.multi_step_generation import (
    render as render_train_multi_step_tab,
)

app_wrapper = typer.Typer()

default_config = load_config(os.getenv("MY_ENV_VAR", ""))


def _init_app() -> list[gr.Dropdown]:
    """
    Initialize the Ultimate RVC web application by updating the choices
    of all dropdown components.

    Returns
    -------
    tuple[gr.Dropdown, ...]
        Updated dropdowns for selecting edge tts voices, RVC voice
        models, cached songs, and output audio files.

    """
    # Initialize model dropdowns
    edge_tts_voice_1click, edge_tts_voice_multi = [
        gr.Dropdown(
            choices=get_edge_tts_voice_names(),
            value="en-US-ChristopherNeural",
        )
        for _ in range(2)
    ]
    voice_model_names = get_voice_model_names()
    voice_models = [
        gr.Dropdown(
            choices=voice_model_names,
            value=None if not voice_model_names else voice_model_names[0],
        )
        for _ in range(4)
    ]
    voice_model_delete = gr.Dropdown(choices=voice_model_names)
    custom_embedder_models = [
        gr.Dropdown(choices=get_custom_embedder_model_names()) for _ in range(6)
    ]

    custom_pretrained_models = [
        gr.Dropdown(choices=get_custom_pretrained_model_names()) for _ in range(2)
    ]
    training_models = [
        gr.Dropdown(choices=get_training_model_names()) for _ in range(4)
    ]

    # Initialize audio dropdowns
    named_song_dirs = get_named_song_dirs()
    cached_songs = [gr.Dropdown(choices=named_song_dirs) for _ in range(3)]
    song_dirs = [
        gr.Dropdown(
            choices=named_song_dirs,
            value=None if not named_song_dirs else named_song_dirs[0][1],
        )
        for _ in range(5)
    ]
    speech_audio = gr.Dropdown(choices=get_saved_speech_audio())
    output_audio = gr.Dropdown(choices=get_saved_output_audio())
    dataset = gr.Dropdown(choices=get_audio_datasets())
    dataset_audio = gr.Dropdown(choices=get_named_audio_datasets())
    return [
        edge_tts_voice_1click,
        edge_tts_voice_multi,
        *voice_models,
        voice_model_delete,
        *custom_embedder_models,
        *custom_pretrained_models,
        *training_models,
        *cached_songs,
        *song_dirs,
        speech_audio,
        output_audio,
        dataset,
        dataset_audio,
    ]


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
    """
    cache_delete_frequency = 86400  # every 24 hours check for files to delete
    cache_delete_cutoff = 86400  # and delete files older than 24 hours

    with gr.Blocks(
        title="Ultimate RVC",
        css=css,
        delete_cache=(cache_delete_frequency, cache_delete_cutoff),
    ) as app:
        gr.HTML("<h1>Ultimate RVC 🧡</h1>")

        # Define model dropdown components
        default_config.speech.one_click.edge_tts_voice.instantiate()
        default_config.speech.multi_step.edge_tts_voice.instantiate()
        for component_config in [
            default_config.song.one_click.voice_model,
            default_config.song.multi_step.voice_model,
            default_config.speech.one_click.voice_model,
            default_config.speech.multi_step.voice_model,
        ]:
            component_config.instantiate(choices=get_voice_model_names())
        default_config.management.model.voice_models.instantiate()
        for component_config in [
            default_config.song.one_click.custom_embedder_model,
            default_config.song.multi_step.custom_embedder_model,
            default_config.speech.one_click.custom_embedder_model,
            default_config.speech.multi_step.custom_embedder_model,
            default_config.training.multi_step.embedder_model,
        ]:
            component_config.instantiate()
        default_config.management.model.embedders.instantiate()
        default_config.training.multi_step.custom_pretrained_model.instantiate()
        default_config.management.model.pretrained_models.instantiate()
        default_config.training.multi_step.preprocess_model.instantiate()
        default_config.training.multi_step.extract_model.instantiate()
        default_config.training.multi_step.train_model.instantiate()
        default_config.management.model.training_models.instantiate()

        # Define audio dropdown components
        default_config.song.one_click.cached_song.instantiate()
        default_config.song.multi_step.cached_song.instantiate()
        default_config.management.audio.intermediate_audio.instantiate()
        named_song_dirs = get_named_song_dirs()
        for component_config in [
            default_config.song.multi_step.separate_audio_dir,
            default_config.song.multi_step.convert_vocals_dir,
            default_config.song.multi_step.postprocess_vocals_dir,
            default_config.song.multi_step.pitch_shift_background_dir,
            default_config.song.multi_step.mix_dir,
        ]:
            component_config.instantiate(choices=named_song_dirs)
        default_config.management.audio.speech_audio.instantiate()
        default_config.management.audio.output_audio.instantiate()
        default_config.management.audio.dataset_audio.instantiate()
        default_config.training.multi_step.dataset.instantiate()
        cookiefile = os.environ.get("YT_COOKIEFILE")
        # main tab
        with gr.Tab("Generate song covers"):
            render_song_cover_one_click_tab(default_config, cookiefile)
            render_song_cover_multi_step_tab(default_config, cookiefile)
        with gr.Tab("Generate speech"):
            render_speech_one_click_tab(default_config)
            render_speech_multi_step_tab(default_config)
        with gr.Tab("Train voice models"):
            render_train_multi_step_tab(default_config)
        with gr.Tab("Manage models"):
            render_manage_models_tab(default_config)
        with gr.Tab("Manage audio"):
            render_manage_audio_tab(default_config)
        with gr.Tab("Settings"):
            render_settings_tab()

        app.load(
            _init_app,
            outputs=[
                default_config.speech.one_click.edge_tts_voice.instance,
                default_config.speech.multi_step.edge_tts_voice.instance,
                default_config.song.one_click.voice_model.instance,
                default_config.song.multi_step.voice_model.instance,
                default_config.speech.one_click.voice_model.instance,
                default_config.speech.multi_step.voice_model.instance,
                default_config.management.model.voice_models.instance,
                default_config.song.one_click.embedder_model.instance,
                default_config.song.multi_step.embedder_model.instance,
                default_config.speech.one_click.embedder_model.instance,
                default_config.speech.multi_step.embedder_model.instance,
                default_config.training.multi_step.embedder_model.instance,
                default_config.management.model.embedders.instance,
                default_config.training.multi_step.custom_pretrained_model.instance,
                default_config.management.model.pretrained_models.instance,
                default_config.training.multi_step.preprocess_model.instance,
                default_config.training.multi_step.extract_model.instance,
                default_config.training.multi_step.train_model.instance,
                default_config.management.model.training_models.instance,
                default_config.management.audio.intermediate_audio.instance,
                default_config.song.one_click.cached_song.instance,
                default_config.song.multi_step.cached_song.instance,
                default_config.song.multi_step.separate_audio_dir.instance,
                default_config.song.multi_step.convert_vocals_dir.instance,
                default_config.song.multi_step.postprocess_vocals_dir.instance,
                default_config.song.multi_step.pitch_shift_background_dir.instance,
                default_config.song.multi_step.mix_dir.instance,
                default_config.management.audio.speech_audio.instance,
                default_config.management.audio.output_audio.instance,
                default_config.training.multi_step.dataset.instance,
                default_config.management.audio.dataset_audio.instance,
            ],
            show_progress="hidden",
        )
    return app


app = render_app()


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
