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
    IndexAlgorithm,
    PretrainedType,
    TrainingF0Method,
    TrainingSampleRate,
    Vocoder,
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
)


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


def render(
    dataset: gr.Dropdown,
    preprocess_model: gr.Dropdown,
    custom_embedder_model: gr.Dropdown,
    extract_model: gr.Dropdown,
    custom_pretrained_model: gr.Dropdown,
    train_model: gr.Dropdown,
    song_cover_voice_model_1click: gr.Dropdown,
    song_cover_voice_model_multi: gr.Dropdown,
    speech_voice_model_1click: gr.Dropdown,
    speech_voice_model_multi: gr.Dropdown,
    training_model_delete: gr.Dropdown,
    voice_model_delete: gr.Dropdown,
    dataset_audio: gr.Dropdown,
) -> None:
    """
    Render the "Train models - multi-step generation" tab.

    Parameters
    ----------
    dataset : gr.Dropdown
        Dropdown to display available datasets in the "Train models -
        multi-step generation" tab.
    preprocess_model : gr.Dropdown
        Dropdown for selecting a voice model to preprocess a
        dataset for in the "Train models - multi-step generation" tab.
    custom_embedder_model : gr.Dropdown
        Dropdown for selecting a custom embedder model to use for
        extracting audio embeddings in the "Train models - multi-step
        generation" tab.
    extract_model : gr.Dropdown
        Dropdown for selecting a voice model with an associated
        preprocessed dataset to extract features from in the
        "Train models - multi-step generation" tab
    custom_pretrained_model : gr.Dropdown
        Dropdown for selecting a custom pretrained model to use for
        training in the "Train models - multi-step generation" tab.
    train_model : gr.Dropdown
        Dropdown for selecting a voice model to train in the "Train
        models - multi-step generation" tab.
    song_cover_voice_model_1click : gr.Dropdown
        Dropdown for selecting a voice model to use for generating
        song covers in the "1-click generation" tab.
    song_cover_voice_model_multi : gr.Dropdown
        Dropdown for selecting a voice model to use for generating
        song covers in the "Multi-step generation" tab.
    speech_voice_model_1click : gr.Dropdown
        Dropdown for selecting a voice model to use for generating
        speech in the "1-click generation" tab.
    speech_voice_model_multi : gr.Dropdown
        Dropdown for selecting a voice model to use for generating
        speech in the "Multi-step generation" tab.
    training_model_delete : gr.Dropdown
        Dropdown for selecting training models to delete in the
        "Delete models" tab.
    voice_model_delete : gr.Dropdown
        Dropdown for selecting voice models to delete in the "Delete
        models" tab.
    dataset_audio : gr.Dropdown
        Dropdown for selecting dataset audio files to delete in the
        "Delete audio" tab.

    """
    current_dataset = gr.State()
    with gr.Tab("Multi-step generation"):
        with gr.Accordion("Step 1: dataset preprocessing", open=True):
            with gr.Row():
                dataset_type = gr.Dropdown(
                    choices=list(DatasetType),
                    label="Dataset type",
                    info="Select the type of dataset to preprocess.",
                    value=DatasetType.NEW_DATASET,
                )
                dataset.render()
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
                outputs=[dataset_name, audio_files, dataset],
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
                outputs=dataset,
                show_progress="hidden",
            ).then(
                partial(update_dropdowns, get_named_audio_datasets, 1, [], [0]),
                outputs=dataset_audio,
                show_progress="hidden",
            )
            with gr.Row():
                preprocess_model.render()
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    with gr.Column():
                        sample_rate = gr.Dropdown(
                            choices=list(TrainingSampleRate),
                            label="Sample rate",
                            info=(
                                "Target sample rate for the audio files in the provided"
                                " dataset."
                            ),
                            value=TrainingSampleRate.HZ_40K,
                        )
                    with gr.Column():
                        filter_audio = gr.Checkbox(
                            value=True,
                            label="Filter audio",
                            info=(
                                "Whether to remove low-frequency sounds from the audio"
                                " files in the provided dataset by applying a high-pass"
                                " butterworth filter.<br><br>"
                            ),
                        )
                    with gr.Column():
                        clean_audio = gr.Checkbox(
                            label="Clean audio",
                            info=(
                                "Whether to clean the audio files in the provided"
                                " dataset using noise reduction algorithms.<br><br><br>"
                            ),
                        )
                        clean_strength = gr.Slider(
                            0.0,
                            1.0,
                            0.7,
                            step=0.1,
                            label="Clean strength",
                            info="The strength of the audio cleaning process.",
                            visible=False,
                            show_reset_button=False,
                        )
                        clean_audio.change(
                            partial(toggle_visibility, targets={True}, default=0.7),
                            inputs=clean_audio,
                            outputs=clean_strength,
                            show_progress="hidden",
                        )
                with gr.Row():
                    split_method = gr.Dropdown(
                        choices=list(AudioSplitMethod),
                        value=AudioSplitMethod.AUTOMATIC,
                        label="Audio splitting method",
                        info=(
                            "The method to use for splitting the audio files in the"
                            " provided dataset. Use the Skip method to skip"
                            " splitting if the audio files are already split. Use"
                            " the Simple method if excessive silence has already"
                            " been removed from the audio files. Use the"
                            " Automatic method for automatic silence detection"
                            " and splitting around it."
                        ),
                    )
                with gr.Row():
                    chunk_len = gr.Slider(
                        0.5,
                        5.0,
                        3.0,
                        step=0.1,
                        label="Chunk length",
                        info="Length of split audio chunks.",
                        visible=False,
                        show_reset_button=False,
                    )
                    overlap_len = gr.Slider(
                        0.0,
                        0.4,
                        0.3,
                        step=0.1,
                        label="Overlap length",
                        info="Length of overlap between split audio chunks.",
                        visible=False,
                        show_reset_button=False,
                    )
                split_method.change(
                    partial(
                        toggle_visibilities,
                        targets={AudioSplitMethod.SIMPLE},
                        defaults=[3.0, 0.3],
                    ),
                    inputs=split_method,
                    outputs=[chunk_len, overlap_len],
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
                        preprocess_model,
                        dataset,
                        sample_rate,
                        split_method,
                        chunk_len,
                        overlap_len,
                        filter_audio,
                        clean_audio,
                        clean_strength,
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
                        preprocess_model,
                        extract_model,
                        training_model_delete,
                    ],
                    show_progress="hidden",
                ).then(
                    _normalize_and_update,
                    inputs=preprocess_model,
                    outputs=preprocess_model,
                    show_progress="hidden",
                ).then(
                    update_value,
                    inputs=preprocess_model,
                    outputs=extract_model,
                    show_progress="hidden",
                )
                reset_preprocess_btn.click(
                    lambda: [
                        TrainingSampleRate.HZ_40K,
                        True,
                        False,
                        0.7,
                        AudioSplitMethod.AUTOMATIC,
                        3.0,
                        0.3,
                        cpu_count(),
                    ],
                    outputs=[
                        sample_rate,
                        filter_audio,
                        clean_audio,
                        clean_strength,
                        split_method,
                        chunk_len,
                        overlap_len,
                        cpu_cores_preprocess,
                    ],
                    show_progress="hidden",
                )
        with gr.Accordion("Step 2: feature extraction", open=True):
            with gr.Row():
                extract_model.render()
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    with gr.Column():
                        f0_method = gr.Dropdown(
                            choices=list(TrainingF0Method),
                            label="F0 method",
                            info="The method to use for extracting pitch features.",
                            value=TrainingF0Method.RMVPE,
                        )
                        hop_length = gr.Slider(
                            1,
                            512,
                            128,
                            step=1,
                            label="Hop length",
                            info=(
                                "The hop length to use for extracting pitch"
                                " features.<br><br>"
                            ),
                            visible=False,
                            show_reset_button=False,
                        )
                    f0_method.change(
                        partial(
                            toggle_visibility,
                            targets={
                                TrainingF0Method.CREPE,
                                TrainingF0Method.CREPE_TINY,
                            },
                            default=128,
                        ),
                        inputs=f0_method,
                        outputs=hop_length,
                        show_progress="hidden",
                    )
                    with gr.Column():
                        embedder_model = gr.Dropdown(
                            choices=list(EmbedderModel),
                            label="Embedder model",
                            info="The model to use for extracting audio embeddings.",
                            value=EmbedderModel.CONTENTVEC,
                        )
                        custom_embedder_model.render()

                    embedder_model.change(
                        partial(toggle_visibility, targets={EmbedderModel.CUSTOM}),
                        inputs=embedder_model,
                        outputs=custom_embedder_model,
                        show_progress="hidden",
                    )
                with gr.Row():
                    include_mutes = gr.Slider(
                        0,
                        10,
                        2,
                        step=1,
                        label="Include mutes",
                        info=(
                            "The number of mute audio files to include in the generated"
                            " training file list. Adding silent files enables the"
                            " training model to handle pure silence in inferred audio"
                            " files. If the preprocessed audio dataset already contains"
                            " segments of pure silence, set this to 0."
                        ),
                        show_reset_button=False,
                    )
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
                        gpu_choices = get_gpu_info()
                        extraction_acceleration = gr.Dropdown(
                            choices=list(DeviceType),
                            value=DeviceType.AUTOMATIC,
                            label="Hardware acceleration",
                            info=(
                                "The type of hardware acceleration to use for feature"
                                " extraction. 'Automatic' will automatically select the"
                                " first available GPU and fall back to CPU if no GPUs"
                                " are available."
                            ),
                        )
                        extraction_gpus = gr.Dropdown(
                            choices=gpu_choices,
                            label="GPU(s)",
                            info="The GPU(s) to use for feature extraction.",
                            multiselect=True,
                            visible=False,
                        )
                extraction_acceleration.change(
                    partial(
                        toggle_visibility,
                        targets={DeviceType.GPU},
                        default=gpu_choices[0][1] if gpu_choices else None,
                    ),
                    inputs=extraction_acceleration,
                    outputs=extraction_gpus,
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
                        extract_model,
                        f0_method,
                        hop_length,
                        embedder_model,
                        custom_embedder_model,
                        include_mutes,
                        cpu_cores_extract,
                        extraction_acceleration,
                        extraction_gpus,
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
                    outputs=train_model,
                    show_progress="hidden",
                ).then(
                    update_value,
                    inputs=extract_model,
                    outputs=train_model,
                    show_progress="hidden",
                )
                reset_extract_btn.click(
                    lambda: [
                        TrainingF0Method.RMVPE,
                        128,
                        EmbedderModel.CONTENTVEC,
                        None,
                        2,
                        cpu_count(),
                        DeviceType.AUTOMATIC,
                        gpu_choices[0][1] if gpu_choices else None,
                    ],
                    outputs=[
                        f0_method,
                        hop_length,
                        embedder_model,
                        custom_embedder_model,
                        include_mutes,
                        cpu_cores_extract,
                        extraction_acceleration,
                        extraction_gpus,
                    ],
                    show_progress="hidden",
                )
        with gr.Accordion("Step 3: model training"):
            with gr.Row():
                train_model.render()

            with gr.Accordion("Settings", open=False):

                with gr.Row():
                    num_epochs = gr.Slider(
                        1,
                        10000,
                        500,
                        step=1,
                        label="Number of epochs",
                        info=(
                            "The number of epochs to train the voice model. A higher"
                            " number can improve voice model performance but may lead"
                            " to overtraining."
                        ),
                        show_reset_button=False,
                    )
                    batch_size = gr.Slider(
                        1,
                        64,
                        8,
                        step=1,
                        label="Batch size",
                        info=(
                            "The number of samples in each training batch. It is"
                            " advisable to align this value with the available VRAM of"
                            " your GPU."
                        ),
                        show_reset_button=False,
                    )
                with gr.Column():
                    detect_overtraining = gr.Checkbox(
                        label="Detect overtraining",
                        info=(
                            "Whether to detect overtraining to prevent the voice model"
                            " from learning the training data too well and losing the"
                            " ability to generalize to new data."
                        ),
                    )
                    overtraining_threshold = gr.Slider(
                        1,
                        100,
                        50,
                        step=1,
                        label="Overtraining threshold",
                        info=(
                            "The maximum number of epochs to continue training without"
                            " any observed improvement in voice model performance."
                        ),
                        visible=False,
                        show_reset_button=False,
                    )
                detect_overtraining.change(
                    partial(toggle_visibility, targets={True}, default=50),
                    inputs=detect_overtraining,
                    outputs=overtraining_threshold,
                    show_progress="hidden",
                )
                with gr.Accordion("Algorithmic settings", open=False):
                    with gr.Row():
                        vocoder = gr.Dropdown(
                            choices=list(Vocoder),
                            label="Vocoder",
                            info=(
                                "The vocoder to use for audio synthesis during"
                                " training. HiFi-GAN provides basic audio fidelity,"
                                " while RefineGAN provides the highest audio fidelity."
                            ),
                            value=Vocoder.HIFI_GAN,
                        )
                        index_algorithm = gr.Dropdown(
                            choices=list(IndexAlgorithm),
                            label="Index algorithm",
                            info=(
                                "The method to use for generating an index file for the"
                                " trained voice model. KMeans is particularly useful"
                                " for large datasets."
                            ),
                            value=IndexAlgorithm.AUTO,
                        )
                    with gr.Column():
                        pretrained_type = gr.Dropdown(
                            choices=list(PretrainedType),
                            label="Pretrained model type",
                            info=(
                                "The type of pretrained model to finetune the voice"
                                " model on. `None` will train the voice model from"
                                " scratch, while `Default` will use a pretrained model"
                                " tailored to the specific voice model architecture."
                                " `Custom` will use a custom pretrained that you"
                                " provide."
                            ),
                            value=PretrainedType.DEFAULT,
                        )
                        custom_pretrained_model.render()

                    pretrained_type.change(
                        partial(toggle_visibility, targets={PretrainedType.CUSTOM}),
                        inputs=pretrained_type,
                        outputs=custom_pretrained_model,
                        show_progress="hidden",
                    )

                with gr.Accordion("Data storage settings", open=False):
                    with gr.Row():

                        save_interval = gr.Slider(
                            1,
                            100,
                            10,
                            step=1,
                            label="Save interval",
                            info=(
                                "The epoch interval at which to to save voice model"
                                " weights and checkpoints. The best model weights are"
                                " always saved regardless of this setting."
                            ),
                            show_reset_button=False,
                        )
                    with gr.Row():

                        save_all_checkpoints = gr.Checkbox(
                            label="Save all checkpoints",
                            info=(
                                "Whether to save a unique checkpoint at each save"
                                " interval. If not enabled, only the latest checkpoint"
                                " will be saved at each interval."
                            ),
                        )
                        save_all_weights = gr.Checkbox(
                            label="Save all weights",
                            info=(
                                "Whether to save unique voice model weights at each"
                                " save interval. If not enabled, only the best voice"
                                " model weights will be saved."
                            ),
                        )

                        clear_saved_data = gr.Checkbox(
                            label="Clear saved data",
                            info=(
                                "Whether to delete any existing training data"
                                " associated with the voice model before training"
                                " commences. Enable this setting only if you are"
                                " training a new voice model from scratch or restarting"
                                " training."
                            ),
                        )

                    with gr.Column():
                        upload_model = gr.Checkbox(
                            label="Upload voice model",
                            info=(
                                "Whether to automatically upload the trained voice"
                                " model so that it can be used for generation tasks"
                                " within the Ultimate RVC app."
                            ),
                        )
                        upload_name = gr.Textbox(
                            label="Upload name",
                            inputs=train_model,
                            info="The name to give the uploaded voice model.",
                            visible=False,
                        )
                    train_model.change(
                        update_value,
                        inputs=train_model,
                        outputs=upload_name,
                        show_progress="hidden",
                    )
                    upload_model.change(
                        partial(
                            toggle_visibility,
                            targets={True},
                            update_default=False,
                        ),
                        inputs=upload_model,
                        outputs=upload_name,
                        show_progress="hidden",
                    )
                with gr.Accordion("Device and memory settings", open=False):
                    with gr.Column():
                        training_acceleration = gr.Dropdown(
                            choices=list(DeviceType),
                            value=DeviceType.AUTOMATIC,
                            label="Hardware acceleration",
                            info=(
                                "The type of hardware acceleration to use when training"
                                " the voice model. 'Automatic' will select the first"
                                " available GPU and fall back to CPU if no GPUs are"
                                " available."
                            ),
                        )
                        training_gpus = gr.Dropdown(
                            choices=gpu_choices,
                            label="GPU(s)",
                            info="The GPU(s) to use for training the voice model.",
                            multiselect=True,
                            visible=False,
                        )
                    training_acceleration.change(
                        partial(
                            toggle_visibility,
                            targets={DeviceType.GPU},
                            default=gpu_choices[0][1] if gpu_choices else None,
                        ),
                        inputs=training_acceleration,
                        outputs=training_gpus,
                        show_progress="hidden",
                    )
                    with gr.Row():
                        preload_dataset = gr.Checkbox(
                            label="Preload dataset",
                            info=(
                                "Whether to preload all training data into GPU memory."
                                " This can improve training speed but requires a lot of"
                                " VRAM.<br><br>"
                            ),
                        )
                        reduce_memory_usage = gr.Checkbox(
                            label="Reduce memory usage",
                            info=(
                                "Whether to reduce VRAM usage at the cost of slower"
                                " training speed by enabling activation checkpointing."
                                " This is useful for GPUs with limited memory (e.g.,"
                                " <6GB VRAM) or when training with a batch size larger"
                                " than what your GPU can normally accommodate."
                            ),
                        )
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
                        train_model,
                        num_epochs,
                        batch_size,
                        detect_overtraining,
                        overtraining_threshold,
                        vocoder,
                        index_algorithm,
                        pretrained_type,
                        custom_pretrained_model,
                        save_interval,
                        save_all_checkpoints,
                        save_all_weights,
                        clear_saved_data,
                        upload_model,
                        upload_name,
                        training_acceleration,
                        training_gpus,
                        preload_dataset,
                        reduce_memory_usage,
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
                    inputs=train_model,
                    outputs=voice_model_files,
                    show_progress="hidden",
                ).then(
                    partial(update_dropdowns, get_voice_model_names, 5, [], [4]),
                    outputs=[
                        song_cover_voice_model_1click,
                        song_cover_voice_model_multi,
                        speech_voice_model_1click,
                        speech_voice_model_multi,
                        voice_model_delete,
                    ],
                    show_progress="hidden",
                )

                stop_train_btn.click(
                    stop_training,
                    inputs=train_model,
                    show_progress="hidden",
                )
                reset_train_btn.click(
                    lambda: [
                        500,
                        8,
                        False,
                        50,
                        Vocoder.HIFI_GAN,
                        IndexAlgorithm.AUTO,
                        PretrainedType.DEFAULT,
                        None,
                        10,
                        False,
                        False,
                        False,
                        False,
                        DeviceType.AUTOMATIC,
                        gpu_choices[0][1] if gpu_choices else None,
                        False,
                        False,
                    ],
                    outputs=[
                        num_epochs,
                        batch_size,
                        detect_overtraining,
                        overtraining_threshold,
                        vocoder,
                        index_algorithm,
                        pretrained_type,
                        custom_pretrained_model,
                        save_interval,
                        save_all_checkpoints,
                        save_all_weights,
                        clear_saved_data,
                        upload_model,
                        training_acceleration,
                        training_gpus,
                        preload_dataset,
                        reduce_memory_usage,
                    ],
                    show_progress="hidden",
                )
