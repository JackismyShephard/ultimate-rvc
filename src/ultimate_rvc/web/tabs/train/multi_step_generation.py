"""
Module which defines the code for the
"Train models - multi-step generation" tab.
"""

from __future__ import annotations

from typing import Any

from functools import partial
from multiprocessing import cpu_count

import gradio as gr

from ultimate_rvc.core.manage.audio import get_audio_datasets, get_named_audio_datasets
from ultimate_rvc.core.manage.models import (
    get_training_model_names,
    get_voice_model_names,
)
from ultimate_rvc.core.train.common import get_gpu_info
from ultimate_rvc.core.train.extract import extract_features
from ultimate_rvc.core.train.prepare import (
    populate_dataset,
    preprocess_dataset,
)
from ultimate_rvc.core.train.train import (
    get_trained_model_files,
    run_training,
    stop_training,
)
from ultimate_rvc.typing_extra import (
    AudioExt,
    AudioSplitMethod,
    DeviceType,
    EmbedderModel,
    PretrainedType,
    TrainingF0Method,
)
from ultimate_rvc.web.common import (
    exception_harness,
    render_msg,
    toggle_visibilities,
    toggle_visibility,
    toggle_visible_component,
    update_dropdowns,
    update_value,
)
from ultimate_rvc.web.typing_extra import (
    ComponentVisibilityKwArgs,
    ConcurrencyId,
    DatasetType,
    MultiStepTrainingConfig,
    TotalConfig,
)


def render(total_config: TotalConfig) -> None:
    """
    Render the "Train models - multi-step generation" tab.

    Parameters
    ----------
    total_config : TotalConfig
        The total configuration object containing the current settings.

    """
    gpu_choices = get_gpu_info()
    with gr.Tab("Multi-step generation"):
        _render_step_1(total_config)
        _render_step_2(total_config, gpu_choices)
        _render_step_3(total_config, gpu_choices)


def _render_step_1(total_config: TotalConfig) -> None:
    tab_config = total_config.training.multi_step

    current_dataset = gr.State()
    with gr.Accordion("Step 1: dataset preprocessing", open=True):
        with gr.Row():
            dataset_type = gr.Dropdown(
                choices=list(DatasetType),
                label="Dataset type",
                info="Select the type of dataset to preprocess.",
                value=DatasetType.NEW_DATASET,
            )
            tab_config.dataset.instance.render()
            dataset_name = gr.Textbox(
                label="Dataset name",
                info=(
                    "The name of the new dataset. If the dataset already"
                    " exists, the provided audio files will be added to it."
                ),
                value="My dataset",
            )
        audio_files = gr.File(
            file_count="multiple",
            label="Audio files",
            file_types=[f".{e.value}" for e in AudioExt],
        )

        dataset_type.change(
            partial(_toggle_dataset_type),
            inputs=dataset_type,
            outputs=[dataset_name, audio_files, tab_config.dataset.instance],
            show_progress="hidden",
        )

        audio_files.upload(
            exception_harness(
                populate_dataset,
                info_msg=(
                    "[+] Audio files successfully added to the dataset with the"
                    " provided name!"
                ),
            ),
            inputs=[dataset_name, audio_files],
            outputs=current_dataset,
        ).then(
            partial(update_value, None),
            outputs=audio_files,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_audio_datasets, 1, value_indices=[0]),
            inputs=current_dataset,
            outputs=tab_config.dataset.instance,
            show_progress="hidden",
        ).then(
            partial(update_dropdowns, get_named_audio_datasets, 1, [], [0]),
            outputs=total_config.management.audio.dataset_audio.instance,
            show_progress="hidden",
        )
        with gr.Row():
            tab_config.preprocess_model.instance.render()
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                with gr.Column():
                    tab_config.sample_rate.instantiate()
                with gr.Column():
                    tab_config.filter_audio.instantiate()
                with gr.Column():
                    tab_config.clean_audio.instantiate()
                    tab_config.clean_strength.instantiate()
                    tab_config.clean_audio.instance.change(
                        partial(
                            toggle_visibility,
                            targets={True},
                            default=tab_config.clean_strength.value,
                        ),
                        inputs=tab_config.clean_audio.instance,
                        outputs=tab_config.clean_strength.instance,
                        show_progress="hidden",
                    )
            with gr.Row():
                tab_config.split_method.instantiate()
            with gr.Row():
                tab_config.chunk_len.instantiate()
                tab_config.overlap_len.instantiate()
            tab_config.split_method.instance.change(
                partial(
                    toggle_visibilities,
                    targets={AudioSplitMethod.SIMPLE},
                    defaults=[tab_config.chunk_len.value, tab_config.overlap_len.value],
                ),
                inputs=tab_config.split_method.instance,
                outputs=[
                    tab_config.chunk_len.instance,
                    tab_config.overlap_len.instance,
                ],
                show_progress="hidden",
            )
            with gr.Row():
                cpu_cores_preprocess = gr.Slider(
                    1,
                    cpu_count(),
                    cpu_count(),
                    step=1,
                    label="CPU cores",
                    info="The number of CPU cores to use for preprocessing.",
                    show_reset_button=False,
                )
        with gr.Row(equal_height=True):
            reset_preprocess_btn = gr.Button(
                "Reset settings",
                variant="secondary",
                scale=2,
            )
            preprocess_btn = gr.Button(
                "Preprocess dataset",
                variant="primary",
                scale=2,
            )
            preprocess_msg = gr.Textbox(
                label="Output message",
                interactive=False,
                scale=3,
            )
            preprocess_btn.click(
                exception_harness(preprocess_dataset),
                inputs=[
                    tab_config.preprocess_model.instance,
                    tab_config.dataset.instance,
                    tab_config.sample_rate.instance,
                    tab_config.split_method.instance,
                    tab_config.chunk_len.instance,
                    tab_config.overlap_len.instance,
                    tab_config.filter_audio.instance,
                    tab_config.clean_audio.instance,
                    tab_config.clean_strength.instance,
                    cpu_cores_preprocess,
                ],
                outputs=preprocess_msg,
                concurrency_limit=1,
                concurrency_id=ConcurrencyId.GPU,
            ).success(
                partial(render_msg, "[+] Dataset successfully preprocessed!"),
                outputs=preprocess_msg,
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_training_model_names, 3, [], [2]),
                outputs=[
                    tab_config.preprocess_model.instance,
                    tab_config.extract_model.instance,
                    total_config.management.model.training_models.instance,
                ],
                show_progress="hidden",
            ).then(
                _normalize_and_update,
                inputs=tab_config.preprocess_model.instance,
                outputs=tab_config.preprocess_model.instance,
                show_progress="hidden",
            ).then(
                update_value,
                inputs=tab_config.preprocess_model.instance,
                outputs=tab_config.extract_model.instance,
                show_progress="hidden",
            )
            reset_preprocess_btn.click(
                lambda: [
                    tab_config.sample_rate.value,
                    tab_config.filter_audio.value,
                    tab_config.clean_audio.value,
                    tab_config.clean_strength.value,
                    tab_config.split_method.value,
                    tab_config.chunk_len.value,
                    tab_config.overlap_len.value,
                    cpu_count(),
                ],
                outputs=[
                    tab_config.sample_rate.instance,
                    tab_config.filter_audio.instance,
                    tab_config.clean_audio.instance,
                    tab_config.clean_strength.instance,
                    tab_config.split_method.instance,
                    tab_config.chunk_len.instance,
                    tab_config.overlap_len.instance,
                    cpu_cores_preprocess,
                ],
                show_progress="hidden",
            )


def _render_step_2(
    total_config: TotalConfig,
    gpu_choices: list[tuple[str, int]],
) -> None:
    tab_config = total_config.training.multi_step
    with gr.Accordion("Step 2: feature extraction", open=True):
        with gr.Row():
            tab_config.extract_model.instance.render()
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                with gr.Column():
                    tab_config.f0_method.instantiate()
                    tab_config.hop_length.instantiate()
                tab_config.f0_method.instance.change(
                    partial(
                        toggle_visibility,
                        targets={
                            TrainingF0Method.CREPE,
                            TrainingF0Method.CREPE_TINY,
                        },
                        default=128,
                    ),
                    inputs=tab_config.f0_method.instance,
                    outputs=tab_config.hop_length.instance,
                    show_progress="hidden",
                )
                with gr.Column():
                    tab_config.embedder_model.instantiate()
                    tab_config.custom_embedder_model.instance.render()

                tab_config.embedder_model.instance.change(
                    partial(toggle_visibility, targets={EmbedderModel.CUSTOM}),
                    inputs=tab_config.embedder_model.instance,
                    outputs=tab_config.custom_embedder_model.instance,
                    show_progress="hidden",
                )
            with gr.Row():
                tab_config.include_mutes.instantiate()
            with gr.Row():
                with gr.Column():
                    cpu_cores_extract = gr.Slider(
                        1,
                        cpu_count(),
                        cpu_count(),
                        step=1,
                        label="CPU cores",
                        info=(
                            "The number of CPU cores to use for feature"
                            " extraction.<br><br>"
                        ),
                        show_reset_button=False,
                    )
                with gr.Column():
                    tab_config.extraction_acceleration.instantiate()
                    tab_config.extraction_gpus.instantiate(
                        choices=gpu_choices,
                    )
            tab_config.extraction_acceleration.instance.change(
                partial(
                    toggle_visibility,
                    targets={DeviceType.GPU},
                    default=gpu_choices[0][1] if gpu_choices else None,
                ),
                inputs=tab_config.extraction_acceleration.instance,
                outputs=tab_config.extraction_gpus.instance,
                show_progress="hidden",
            )
        with gr.Row(equal_height=True):
            reset_extract_btn = gr.Button(
                "Reset settings",
                variant="secondary",
                scale=2,
            )
            extract_btn = gr.Button("Extract features", variant="primary", scale=2)
            extract_msg = gr.Textbox(
                label="Output message",
                interactive=False,
                scale=3,
            )
            extract_btn.click(
                exception_harness(extract_features),
                inputs=[
                    tab_config.extract_model.instance,
                    tab_config.f0_method.instance,
                    tab_config.hop_length.instance,
                    tab_config.embedder_model.instance,
                    tab_config.custom_embedder_model.instance,
                    tab_config.include_mutes.instance,
                    cpu_cores_extract,
                    tab_config.extraction_acceleration.instance,
                    tab_config.extraction_gpus.instance,
                ],
                outputs=extract_msg,
                concurrency_limit=1,
                concurrency_id=ConcurrencyId.GPU,
            ).success(
                partial(render_msg, "[+] Features successfully extracted!"),
                outputs=extract_msg,
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_training_model_names, 1),
                outputs=tab_config.train_model.instance,
                show_progress="hidden",
            ).then(
                update_value,
                inputs=tab_config.extract_model.instance,
                outputs=tab_config.train_model.instance,
                show_progress="hidden",
            )
            reset_extract_btn.click(
                lambda: [
                    tab_config.f0_method.value,
                    tab_config.hop_length.value,
                    tab_config.embedder_model.value,
                    tab_config.custom_embedder_model.value,
                    tab_config.include_mutes.value,
                    tab_config.extraction_acceleration.value,
                    cpu_count(),
                    gpu_choices[0][1] if gpu_choices else None,
                ],
                outputs=[
                    tab_config.f0_method.instance,
                    tab_config.hop_length.instance,
                    tab_config.embedder_model.instance,
                    tab_config.custom_embedder_model.instance,
                    tab_config.include_mutes.instance,
                    cpu_cores_extract,
                    tab_config.extraction_acceleration.instance,
                    tab_config.extraction_gpus.instance,
                ],
                show_progress="hidden",
            )


def _render_step_3(
    total_config: TotalConfig,
    gpu_choices: list[tuple[str, int]],
) -> None:
    tab_config = total_config.training.multi_step
    with gr.Accordion("Step 3: model training"):
        with gr.Row():
            tab_config.train_model.instance.render()
            _render_step_3_main_settings(tab_config)
            _render_step_3_algorithmic_settings(tab_config)
            _render_step_3_data_storage_settings(tab_config)
            _render_step_3_device_settings(tab_config, gpu_choices)
        with gr.Accordion("Output", open=True):
            voice_model_files = gr.File(
                label="Voice model files",
                interactive=False,
            )
        with gr.Row(equal_height=True):
            reset_train_btn = gr.Button(
                "Reset settings",
                variant="secondary",
                scale=2,
            )
            train_btn = gr.Button("Train voice model", variant="primary", scale=2)
            stop_train_btn = gr.Button(
                "Stop training",
                variant="primary",
                scale=2,
                visible=False,
            )
            train_msg = gr.Textbox(
                label="Output message",
                interactive=False,
                scale=3,
            )
            train_btn.click(
                partial(toggle_visible_component, 2, 1, reset_values=False),
                outputs=[train_btn, stop_train_btn],
                show_progress="hidden",
            )
            train_btn_click = train_btn.click(
                partial(
                    exception_harness(run_training),
                ),
                inputs=[
                    tab_config.train_model.instance,
                    tab_config.num_epochs.instance,
                    tab_config.batch_size.instance,
                    tab_config.detect_overtraining.instance,
                    tab_config.overtraining_threshold.instance,
                    tab_config.vocoder.instance,
                    tab_config.index_algorithm.instance,
                    tab_config.pretrained_type.instance,
                    tab_config.custom_pretrained_model.instance,
                    tab_config.save_interval.instance,
                    tab_config.save_all_checkpoints.instance,
                    tab_config.save_all_weights.instance,
                    tab_config.clear_saved_data.instance,
                    tab_config.upload_model.instance,
                    tab_config.upload_name.instance,
                    tab_config.training_acceleration.instance,
                    tab_config.training_gpus.instance,
                    tab_config.preload_dataset.instance,
                    tab_config.reduce_memory_usage.instance,
                ],
                outputs=train_msg,
                concurrency_limit=1,
                concurrency_id=ConcurrencyId.GPU,
            )

            train_btn_click.then(
                partial(toggle_visible_component, 2, 0, reset_values=False),
                outputs=[train_btn, stop_train_btn],
                show_progress="hidden",
            )

            train_btn_click.success(
                partial(render_msg, "[+] Voice model successfully trained!"),
                outputs=train_msg,
                show_progress="hidden",
            ).then(
                get_trained_model_files,
                inputs=tab_config.train_model.instance,
                outputs=voice_model_files,
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_voice_model_names, 5, [], [4]),
                outputs=[
                    total_config.song.one_click.voice_model.instance,
                    total_config.song.multi_step.voice_model.instance,
                    total_config.speech.one_click.voice_model.instance,
                    total_config.speech.multi_step.voice_model.instance,
                    total_config.management.model.voice_models.instance,
                ],
                show_progress="hidden",
            )

            stop_train_btn.click(
                stop_training,
                inputs=tab_config.train_model.instance,
                show_progress="hidden",
            )
            reset_train_btn.click(
                lambda: [
                    tab_config.num_epochs.value,
                    tab_config.batch_size.value,
                    tab_config.detect_overtraining.value,
                    tab_config.overtraining_threshold.value,
                    tab_config.vocoder.value,
                    tab_config.index_algorithm.value,
                    tab_config.pretrained_type.value,
                    tab_config.custom_pretrained_model.value,
                    tab_config.save_interval.value,
                    tab_config.save_all_checkpoints.value,
                    tab_config.save_all_weights.value,
                    tab_config.clear_saved_data.value,
                    tab_config.upload_model.value,
                    tab_config.upload_name.value,
                    tab_config.training_acceleration.value,
                    gpu_choices[0][1] if gpu_choices else None,
                    tab_config.preload_dataset.value,
                    tab_config.reduce_memory_usage.value,
                ],
                outputs=[
                    tab_config.num_epochs.instance,
                    tab_config.batch_size.instance,
                    tab_config.detect_overtraining.instance,
                    tab_config.overtraining_threshold.instance,
                    tab_config.vocoder.instance,
                    tab_config.index_algorithm.instance,
                    tab_config.pretrained_type.instance,
                    tab_config.custom_pretrained_model.instance,
                    tab_config.save_interval.instance,
                    tab_config.save_all_checkpoints.instance,
                    tab_config.save_all_weights.instance,
                    tab_config.clear_saved_data.instance,
                    tab_config.upload_model.instance,
                    tab_config.upload_name.instance,
                    tab_config.training_acceleration.instance,
                    tab_config.training_gpus.instance,
                    tab_config.preload_dataset.instance,
                    tab_config.reduce_memory_usage.instance,
                ],
                show_progress="hidden",
            )


def _render_step_3_main_settings(tab_config: MultiStepTrainingConfig) -> None:
    with gr.Accordion("Settings", open=False):

        with gr.Row():
            tab_config.num_epochs.instantiate()
            tab_config.batch_size.instantiate()
        with gr.Column():
            tab_config.detect_overtraining.instantiate()
            tab_config.overtraining_threshold.instantiate()
        tab_config.detect_overtraining.instance.change(
            partial(toggle_visibility, targets={True}, default=50),
            inputs=tab_config.detect_overtraining.instance,
            outputs=tab_config.overtraining_threshold.instance,
            show_progress="hidden",
        )


def _render_step_3_algorithmic_settings(tab_config: MultiStepTrainingConfig) -> None:
    with gr.Accordion("Algorithmic settings", open=False):
        with gr.Row():
            tab_config.vocoder.instantiate()
            tab_config.index_algorithm.instantiate()
        with gr.Column():
            tab_config.pretrained_type.instantiate()
            tab_config.custom_pretrained_model.instance.render()

        tab_config.pretrained_type.instance.change(
            partial(toggle_visibility, targets={PretrainedType.CUSTOM}),
            inputs=tab_config.pretrained_type.instance,
            outputs=tab_config.custom_pretrained_model.instance,
            show_progress="hidden",
        )


def _render_step_3_data_storage_settings(tab_config: MultiStepTrainingConfig) -> None:

    with gr.Accordion("Data storage settings", open=False):
        with gr.Row():
            tab_config.save_interval.instantiate()
        with gr.Row():
            tab_config.save_all_checkpoints.instantiate()
            tab_config.save_all_weights.instantiate()
            tab_config.clear_saved_data.instantiate()

        with gr.Column():
            tab_config.upload_model.instantiate()
            tab_config.upload_name.instantiate(
                inputs=tab_config.upload_model.instance,
            )
        tab_config.train_model.instance.change(
            update_value,
            inputs=tab_config.train_model.instance,
            outputs=tab_config.upload_name.instance,
            show_progress="hidden",
        )
        tab_config.train_model.instance.change(
            partial(
                toggle_visibility,
                targets={True},
                update_default=False,
            ),
            inputs=tab_config.train_model.instance,
            outputs=tab_config.upload_name.instance,
            show_progress="hidden",
        )


def _render_step_3_device_settings(
    tab_config: MultiStepTrainingConfig,
    gpu_choices: list[tuple[str, int]],
) -> None:
    with gr.Accordion("Device and memory settings", open=False):
        with gr.Column():
            tab_config.training_acceleration.instantiate()
            tab_config.training_gpus.instantiate(
                choices=gpu_choices,
            )
        tab_config.training_acceleration.instance.change(
            partial(
                toggle_visibility,
                targets={DeviceType.GPU},
                default=gpu_choices[0][1] if gpu_choices else None,
            ),
            inputs=tab_config.training_acceleration.instance,
            outputs=tab_config.training_gpus.instance,
            show_progress="hidden",
        )
        with gr.Row():
            tab_config.preload_dataset.instantiate()
            tab_config.reduce_memory_usage.instantiate()


def _toggle_dataset_type(dataset_type: DatasetType) -> tuple[dict[str, Any], ...]:
    """
    Toggle the visibility of three different dataset input components
    based on the selected dataset type.

    Parameters
    ----------
    dataset_type : DatasetType
        The type of dataset to preprocess.

    Returns
    -------
    tuple[dict[str, Any], ...]
        A tuple of dictionaries which update the visibility of the
        the three dataset input components.

    """
    update_args_list: list[ComponentVisibilityKwArgs] = [
        {"visible": False, "value": None} for _ in range(3)
    ]
    match dataset_type:
        case DatasetType.NEW_DATASET:
            update_args_list[0]["visible"] = True
            update_args_list[0]["value"] = "My dataset"
            update_args_list[1]["visible"] = True
        case DatasetType.EXISTING_DATASET:
            update_args_list[2]["visible"] = True
    return tuple(gr.update(**update_args) for update_args in update_args_list)


def _normalize_and_update(value: str) -> gr.Dropdown:
    """
    Normalize the value of the given string and update the dropdown.

    Parameters
    ----------
    value : str
        The value to normalize and update.

    Returns
    -------
    gr.Dropdown
        The updated dropdown.

    """
    return gr.Dropdown(value=value.strip())
