"""Module which defines the code for the "Models" tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from collections.abc import Sequence
from functools import partial

import gradio as gr

# NOTE gradio uses pandas for more than typechecking so we need to
# import it here
import pandas as pd  # noqa: TC002

from ultimate_rvc.core.manage.models import (
    PRETRAINED_MODELS_TABLE,
    delete_all_custom_embedder_models,
    delete_all_custom_pretrained_models,
    delete_all_models,
    delete_all_training_models,
    delete_all_voice_models,
    delete_custom_embedder_models,
    delete_custom_pretrained_models,
    delete_training_models,
    delete_voice_models,
    download_pretrained_model,
    download_voice_model,
    filter_public_models_table,
    get_custom_embedder_model_names,
    get_custom_pretrained_model_names,
    get_public_model_tags,
    get_training_model_names,
    get_voice_model_names,
    load_public_models_table,
    upload_custom_embedder_model,
    upload_voice_model,
)
from ultimate_rvc.web.common import (
    exception_harness,
    render_msg,
    setup_delete_event,
    update_dropdowns,
)
from ultimate_rvc.web.config.event import ManageModelEventState
from ultimate_rvc.web.tabs.train.multi_step_generation import (
    render as _render_train_multi_step_tab,
)

if TYPE_CHECKING:
    from ultimate_rvc.web.config.main import ModelManagementConfig, TotalConfig


def render(total_config: TotalConfig) -> None:
    """

    Render "Models" tab.

    Parameters
    ----------
    total_config : TotalConfig
        Model containing all component configuration settings for the
        Ultimate RVC web UI.

    """
    tab_config = total_config.management.model
    tab_config.dummy_checkbox.instantiate()
    event_state = ManageModelEventState()

    _render_download_tab(event_state)
    _render_upload_tab(event_state)
    with gr.Tab("Train", elem_id="train-tab"):
        _render_train_multi_step_tab(total_config)
    _render_delete_tab(tab_config, event_state)

    *_, all_model_update = [
        click_event.success(
            partial(update_dropdowns, get_voice_model_names, 5, [], [4]),
            outputs=[
                total_config.song.one_click.voice_model.instance,
                total_config.song.multi_step.voice_model.instance,
                total_config.speech.one_click.voice_model.instance,
                total_config.speech.multi_step.voice_model.instance,
                tab_config.voices.instance,
            ],
            show_progress="hidden",
        )
        for click_event in [
            event_state.download_voice_click.instance,
            event_state.upload_voice_click.instance,
            event_state.delete_voice_click.instance,
            event_state.delete_all_voices_click.instance,
            event_state.delete_all_click.instance,
        ]
    ]

    *_, all_model_update = [
        click_event.success(
            partial(update_dropdowns, get_custom_embedder_model_names, 6, [], [5]),
            outputs=[
                total_config.song.one_click.custom_embedder_model.instance,
                total_config.song.multi_step.custom_embedder_model.instance,
                total_config.speech.one_click.custom_embedder_model.instance,
                total_config.speech.multi_step.custom_embedder_model.instance,
                total_config.training.multi_step.custom_embedder_model.instance,
                tab_config.embedders.instance,
            ],
            show_progress="hidden",
        )
        for click_event in [
            event_state.upload_embedder_click.instance,
            event_state.delete_embedder_click.instance,
            event_state.delete_all_embedders_click.instance,
            all_model_update,
        ]
    ]

    *_, all_model_update = [
        click_event.success(
            partial(update_dropdowns, get_custom_pretrained_model_names, 2, [], [1]),
            outputs=[
                total_config.training.multi_step.custom_pretrained_model.instance,
                tab_config.pretraineds.instance,
            ],
            show_progress="hidden",
        )
        for click_event in [
            event_state.download_pretrained_click.instance,
            event_state.delete_pretrained_click.instance,
            event_state.delete_all_pretraineds_click.instance,
            all_model_update,
        ]
    ]

    for click_event in [
        event_state.delete_trained_click.instance,
        event_state.delete_all_trained_click.instance,
        all_model_update,
    ]:
        click_event.success(
            partial(update_dropdowns, get_training_model_names, 4, [], [0, 3]),
            outputs=[
                total_config.training.multi_step.preprocess_model.instance,
                total_config.training.multi_step.extract_model.instance,
                total_config.training.multi_step.train_model.instance,
                tab_config.traineds.instance,
            ],
            show_progress="hidden",
        )


def _render_download_tab(event_state: ManageModelEventState) -> None:
    with gr.Tab("Download"):
        with gr.Accordion("Voice models"):
            with gr.Accordion("View public models table", open=False):
                with gr.Accordion("HOW TO USE", open=False):
                    gr.Markdown("")
                    gr.Markdown(
                        "- Filter voice models by selecting one or more tags and/or"
                        " providing a search query.",
                    )
                    gr.Markdown(
                        "- Select a row in the table to autofill the name and URL for"
                        " the given voice model in the form fields below.",
                    )
                with gr.Row():
                    search_query = gr.Textbox(label="Search query")
                    tags = gr.CheckboxGroup(
                        label="Tags",
                        value=[],
                        choices=get_public_model_tags(),
                    )
                with gr.Row():
                    public_models_table = gr.Dataframe(
                        label="Public models table",
                        value=load_public_models_table([]),
                        headers=[
                            "Name",
                            "Description",
                            "Tags",
                            "Credit",
                            "Added",
                            "URL",
                        ],
                        interactive=False,
                    )
                # We are updating the table here instead of doing it
                # implicitly using value=_filter_public_models_table and
                # inputs=[tags, search_query] when instantiating
                # gr.Dataframe because that does not work with reload
                # mode due to a bug.
                gr.on(  # type: ignore[reportUnknownMemberType]
                    triggers=[search_query.change, tags.change],
                    fn=_filter_public_models_table,
                    inputs=[tags, search_query],
                    outputs=public_models_table,
                )

            with gr.Row():
                voice_model_url = gr.Textbox(
                    label="Model URL",
                    info=(
                        "Should point to a zip file containing a .pth model file and"
                        " optionally also an .index file."
                    ),
                )
                voice_model_name = gr.Textbox(
                    label="Model name",
                    info="Enter a unique name for the voice model.",
                )

            with gr.Row(equal_height=True):
                download_voice_btn = gr.Button(
                    "Download 🌐",
                    variant="primary",
                    scale=19,
                )
                download_voice_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                    scale=20,
                )

            public_models_table.select(
                _autofill_model_name_and_url,
                inputs=public_models_table,
                outputs=[voice_model_name, voice_model_url],
                show_progress="hidden",
            )

            event_state.download_voice_click.instance = download_voice_btn.click(
                exception_harness(download_voice_model),
                inputs=[voice_model_url, voice_model_name],
                outputs=download_voice_msg,
            ).success(
                partial(render_msg, "[+] Succesfully downloaded voice model!"),
                outputs=download_voice_msg,
                show_progress="hidden",
            )
        with gr.Accordion("Pretrained models", open=False):
            with gr.Row():
                pretrained_model = gr.Dropdown(
                    label="Pretrained model",
                    info="Select the pretrained model you want to download.",
                    value=PRETRAINED_MODELS_TABLE.default_name,
                    choices=PRETRAINED_MODELS_TABLE.names,
                )
                pretrained_sample_rate = gr.Dropdown(
                    label="Sample rate",
                    info="Select the sample rate for the pretrained model.",
                    value=PRETRAINED_MODELS_TABLE.default_sample_rate,
                    choices=PRETRAINED_MODELS_TABLE.default_sample_rates,
                )

                pretrained_model.change(
                    _update_pretrained_sample_rates,
                    inputs=pretrained_model,
                    outputs=pretrained_sample_rate,
                    show_progress="hidden",
                )
            with gr.Row(equal_height=True):
                download_pretrained_btn = gr.Button(
                    "Download 🌐",
                    variant="primary",
                    scale=19,
                )
                download_pretrained_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                    scale=20,
                )
            event_state.download_pretrained_click.instance = (
                download_pretrained_btn.click(
                    exception_harness(download_pretrained_model),
                    inputs=[pretrained_model, pretrained_sample_rate],
                    outputs=download_pretrained_msg,
                ).success(
                    partial(render_msg, "[+] Succesfully downloaded pretrained model!"),
                    outputs=download_pretrained_msg,
                    show_progress="hidden",
                )
            )


def _render_upload_tab(event_state: ManageModelEventState) -> None:
    with gr.Tab("Upload"):
        with gr.Accordion("Voice models", open=True):
            with gr.Accordion("HOW TO USE", open=False):
                gr.Markdown("")
                gr.Markdown(
                    "1. Find the .pth file for a locally trained RVC model (e.g. in"
                    " your local weights folder) and optionally also a corresponding"
                    " .index file (e.g. in your logs/[name] folder)",
                )
                gr.Markdown(
                    "2. Upload the files directly or save them to a folder, then"
                    " compress that folder and upload the resulting .zip file",
                )
                gr.Markdown("3. Enter a unique name for the uploaded model")
                gr.Markdown("4. Click 'Upload'")

            with gr.Row():
                voice_model_files = gr.File(
                    label="Files",
                    file_count="multiple",
                    file_types=[".zip", ".pth", ".index"],
                )

                local_voice_model_name = gr.Textbox(label="Model name")

            with gr.Row(equal_height=True):
                upload_voice_btn = gr.Button("Upload", variant="primary", scale=19)
                upload_voice_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                    scale=20,
                )
                event_state.upload_voice_click.instance = upload_voice_btn.click(
                    exception_harness(upload_voice_model),
                    inputs=[voice_model_files, local_voice_model_name],
                    outputs=upload_voice_msg,
                ).success(
                    partial(render_msg, "[+] Successfully uploaded voice model!"),
                    outputs=upload_voice_msg,
                    show_progress="hidden",
                )
        with gr.Accordion("Custom embedder models", open=False):
            with gr.Accordion("HOW TO USE", open=False):
                gr.Markdown("")
                gr.Markdown(
                    "1. Find the config.json file and pytorch_model.bin file for a"
                    " custom embedder model stored locally.",
                )
                gr.Markdown(
                    "2. Upload the files directly or save them to a folder, then"
                    " compress that folder and upload the resulting .zip file",
                )
                gr.Markdown("3. Enter a unique name for the uploaded embedder model")
                gr.Markdown("4. Click 'Upload'")

            with gr.Row():
                embedder_files = gr.File(
                    label="Files",
                    file_count="multiple",
                    file_types=[".zip", ".json", ".bin"],
                )

                local_embedder_name = gr.Textbox(label="Model name")

            with gr.Row(equal_height=True):
                upload_embedder_btn = gr.Button("Upload", variant="primary", scale=19)
                upload_embedder_msg = gr.Textbox(
                    label="Output message",
                    interactive=False,
                    scale=20,
                )
                event_state.upload_embedder_click.instance = upload_embedder_btn.click(
                    exception_harness(upload_custom_embedder_model),
                    inputs=[embedder_files, local_embedder_name],
                    outputs=upload_embedder_msg,
                ).success(
                    partial(
                        render_msg,
                        "[+] Successfully uploaded custom embedder model!",
                    ),
                    outputs=upload_embedder_msg,
                    show_progress="hidden",
                )


def _render_delete_tab(
    tab_config: ModelManagementConfig,
    event_state: ManageModelEventState,
) -> None:
    with gr.Tab("Delete"):
        _render_voices_accordion(tab_config, event_state)
        _render_embedders_accordion(tab_config, event_state)
        _render_pretraineds_accordion(tab_config, event_state)
        _render_traineds_accordion(tab_config, event_state)
        _render_all_accordion(tab_config, event_state)


def _render_voices_accordion(
    tab_config: ModelManagementConfig,
    event_state: ManageModelEventState,
) -> None:
    with gr.Accordion("Voice models", open=False), gr.Row():
        with gr.Column():
            tab_config.voices.instance.render()
            delete_voice_btn = gr.Button("Delete selected", variant="secondary")
            delete_all_voice_btn = gr.Button("Delete all", variant="primary")
        with gr.Column():
            delete_voice_msg = gr.Textbox(label="Output message", interactive=False)

    event_state.delete_voice_click.instance = setup_delete_event(
        delete_voice_btn,
        delete_voice_models,
        [tab_config.dummy_checkbox.instance, tab_config.voices.instance],
        delete_voice_msg,
        "Are you sure you want to delete the selected voice models?",
        "[-] Successfully deleted selected voice models!",
    )

    event_state.delete_all_voices_click.instance = setup_delete_event(
        delete_all_voice_btn,
        delete_all_voice_models,
        [tab_config.dummy_checkbox.instance],
        delete_voice_msg,
        "Are you sure you want to delete all voice models?",
        "[-] Successfully deleted all voice models!",
    )


def _render_embedders_accordion(
    tab_config: ModelManagementConfig,
    event_state: ManageModelEventState,
) -> None:
    with gr.Accordion("Custom embedder models", open=False), gr.Row():
        with gr.Column():
            tab_config.embedders.instance.render()
            delete_embedder_btn = gr.Button("Delete selected", variant="secondary")
            delete_all_embedder_btn = gr.Button("Delete all", variant="primary")
        with gr.Column():
            delete_embedder_msg = gr.Textbox(label="Output message", interactive=False)

    event_state.delete_embedder_click.instance = setup_delete_event(
        delete_embedder_btn,
        delete_custom_embedder_models,
        [tab_config.dummy_checkbox.instance, tab_config.embedders.instance],
        delete_embedder_msg,
        "Are you sure you want to delete the selected custom embedder models?",
        "[-] Successfully deleted selected custom embedder models!",
    )

    event_state.delete_all_embedders_click.instance = setup_delete_event(
        delete_all_embedder_btn,
        delete_all_custom_embedder_models,
        [tab_config.dummy_checkbox.instance],
        delete_embedder_msg,
        "Are you sure you want to delete all custom embedder models?",
        "[-] Successfully deleted all custom embedder models!",
    )


def _render_pretraineds_accordion(
    tab_config: ModelManagementConfig,
    event_state: ManageModelEventState,
) -> None:
    with gr.Accordion("Custom pretrained models", open=False), gr.Row():
        with gr.Column():
            tab_config.pretraineds.instance.render()
            delete_pretrained_btn = gr.Button("Delete selected", variant="secondary")
            delete_all_pretrained_btn = gr.Button("Delete all", variant="primary")
        with gr.Column():
            delete_pretrained_msg = gr.Textbox(
                label="Output message",
                interactive=False,
            )

    event_state.delete_pretrained_click.instance = setup_delete_event(
        delete_pretrained_btn,
        delete_custom_pretrained_models,
        [tab_config.dummy_checkbox.instance, tab_config.pretraineds.instance],
        delete_pretrained_msg,
        "Are you sure you want to delete the selected custom pretrained models?",
        "[-] Successfully deleted selected custom pretrained models!",
    )
    event_state.delete_all_pretraineds_click.instance = setup_delete_event(
        delete_all_pretrained_btn,
        delete_all_custom_pretrained_models,
        [tab_config.dummy_checkbox.instance],
        delete_pretrained_msg,
        "Are you sure you want to delete all custom pretrained models?",
        "[-] Successfully deleted all custom pretrained models!",
    )


def _render_traineds_accordion(
    tab_config: ModelManagementConfig,
    event_state: ManageModelEventState,
) -> None:
    with gr.Accordion("Training models", open=False), gr.Row():
        with gr.Column():
            tab_config.traineds.instance.render()
            delete_train_btn = gr.Button("Delete selected", variant="secondary")
            delete_all_train_btn = gr.Button("Delete all", variant="primary")
        with gr.Column():
            delete_train_msg = gr.Textbox(label="Output message", interactive=False)

    event_state.delete_trained_click.instance = setup_delete_event(
        delete_train_btn,
        delete_training_models,
        [tab_config.dummy_checkbox.instance, tab_config.traineds.instance],
        delete_train_msg,
        "Are you sure you want to delete the selected training models?",
        "[-] Successfully deleted selected training models!",
    )

    event_state.delete_all_trained_click.instance = setup_delete_event(
        delete_all_train_btn,
        delete_all_training_models,
        [tab_config.dummy_checkbox.instance],
        delete_train_msg,
        "Are you sure you want to delete all training models?",
        "[-] Successfully deleted all training models!",
    )


def _render_all_accordion(
    tab_config: ModelManagementConfig,
    event_state: ManageModelEventState,
) -> None:
    with gr.Accordion("All models"), gr.Row(equal_height=True):
        delete_all_btn = gr.Button("Delete", variant="primary")
        delete_all_msg = gr.Textbox(label="Output message", interactive=False)

    event_state.delete_all_click.instance = setup_delete_event(
        delete_all_btn,
        delete_all_models,
        [tab_config.dummy_checkbox.instance],
        delete_all_msg,
        "Are you sure you want to delete all models?",
        "[-] Successfully deleted all models!",
    )


def _filter_public_models_table(tags: Sequence[str], query: str) -> gr.Dataframe:
    """
    Filter table containing metadata of public voice models by tags and
    a search query.

    Parameters
    ----------
    tags : Sequence[str]
        Tags to filter the metadata table by.
    query : str
        Search query to filter the metadata table by.

    Returns
    -------
    gr.Dataframe
        The filtered table rendered in a Gradio dataframe.

    """
    models_table = filter_public_models_table(tags, query)
    return gr.Dataframe(value=models_table)


def _autofill_model_name_and_url(
    public_models_table: pd.DataFrame,
    select_event: gr.SelectData,
) -> tuple[gr.Textbox, gr.Textbox]:
    """
    Autofill two textboxes with respectively the name and URL that is
    saved in the currently selected row of the public models table.

    Parameters
    ----------
    public_models_table : pd.DataFrame
        The public models table saved in a Pandas dataframe.
    select_event : gr.SelectData
        Event containing the index of the currently selected row in the
        public models table.

    Returns
    -------
    name : gr.Textbox
        The textbox containing the model name.

    url : gr.Textbox
        The textbox containing the model URL.

    Raises
    ------
    TypeError
        If the index in the provided event is not a sequence.

    """
    event_index: int | Sequence[int] = select_event.index
    if not isinstance(event_index, Sequence):
        err_msg = (
            f"Expected a sequence of indices but got {type(event_index)} from the"
            " provided event."
        )
        raise TypeError(err_msg)
    event_index = event_index[0]
    url = public_models_table.loc[event_index, "URL"]
    name = public_models_table.loc[event_index, "Name"]
    if isinstance(url, str) and isinstance(name, str):
        return gr.Textbox(value=name), gr.Textbox(value=url)
    err_msg = (
        "Expected model name and URL to be strings but got"
        f" {type(name)} and {type(url)} respectively."
    )
    raise TypeError(err_msg)


def _update_pretrained_sample_rates(name: str) -> gr.Dropdown:
    """
    Update the dropdown for pretrained sample rates based on the
    selected pretrained model.

    Parameters
    ----------
    name : str
        The name of the selected pretrained model.

    Returns
    -------
    pretrained_sample_rate : gr.Dropdown
        The updated dropdown for pretrained sample rates.

    """
    default, sample_rates = PRETRAINED_MODELS_TABLE.get_sample_rates_with_default(name)
    return gr.Dropdown(value=default, choices=sample_rates)
