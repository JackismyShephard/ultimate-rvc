"""
Module which defines the command-line interface for training voice
models using RVC.
"""

from __future__ import annotations

from typing import Annotated

import time
from multiprocessing import cpu_count
from pathlib import Path  # noqa: TC003

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from ultimate_rvc.cli.common import (
    complete_audio_split_method,
    complete_embedder_model,
    complete_f0_method,
    complete_index_algorithm,
    complete_rvc_version,
    complete_training_sample_rate,
    complete_vocoder,
    format_duration,
)
from ultimate_rvc.cli.typing_extra import PanelName
from ultimate_rvc.core.train.common import get_gpu_info as _get_gpu_info
from ultimate_rvc.core.train.extract import extract_features as _extract_features
from ultimate_rvc.core.train.prepare import populate_dataset as _populate_dataset
from ultimate_rvc.core.train.prepare import preprocess_dataset as _preprocess_dataset
from ultimate_rvc.core.train.train import run_training as _run_training
from ultimate_rvc.typing_extra import (
    AudioSplitMethod,
    EmbedderModel,
    IndexAlgorithm,
    RVCVersion,
    TrainingF0Method,
    TrainingSampleRate,
    Vocoder,
)

app = typer.Typer(
    name="train",
    no_args_is_help=True,
    help="Train voice models using RVC",
    rich_markup_mode="markdown",
)

CORES = cpu_count()


@app.command(no_args_is_help=True)
def populate_dataset(
    name: Annotated[
        str,
        typer.Argument(help="The name of the dataset to populate."),
    ],
    audio_files: Annotated[
        list[Path],
        typer.Argument(
            help="The audio files to populate the dataset with.",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            file_okay=True,
        ),
    ],
) -> None:
    """
    Populate the dataset with the provided name with the provided audio
    files.
    """
    start_time = time.perf_counter()

    dataset_path = _populate_dataset(name, audio_files)

    rprint("[+] Dataset succesfully populated with the provided audio files!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{dataset_path}", title="Dataset Path"))


@app.command(no_args_is_help=True)
def preprocess_dataset(
    model_name: Annotated[
        str,
        typer.Argument(
            help=(
                "The name of the voice model to train. If no voice model with the"
                " provided name exists for training, a new voice model for training"
                " will be created with the provided name. If a voice model with the"
                " provided name already exists for training, then its currently"
                " associated dataset will be replaced with the provided dataset."
            ),
        ),
    ],
    dataset: Annotated[
        Path,
        typer.Argument(
            help="The path to the dataset to preprocess",
            exists=True,
            resolve_path=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
    sample_rate: Annotated[
        TrainingSampleRate,
        typer.Option(
            autocompletion=complete_training_sample_rate,
            help="The target sample rate for the audio files in the provided dataset",
        ),
    ] = TrainingSampleRate.HZ_40K,
    split_method: Annotated[
        AudioSplitMethod,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_audio_split_method,
            help=(
                "The method to use for splitting the audio files in the provided"
                " dataset. Use the `Skip` method to skip splitting if the audio files"
                " are already split. Use the `Simple` method if excessive silence has"
                " already been removed from the audio files. Use the `Automatic` method"
                " for automatic silence detection and splitting around it."
            ),
        ),
    ] = AudioSplitMethod.AUTOMATIC,
    chunk_len: Annotated[
        float,
        typer.Option(
            min=0.5,
            max=5.0,
            help="Length of split audio chunks when using the `Simple` split method.",
        ),
    ] = 3.0,
    overlap_len: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=0.4,
            help=(
                "Length of overlap between split audio chunks when using the `Simple`"
                " split method."
            ),
        ),
    ] = 0.3,
    filter_audio: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to remove low-frequency sounds from the audio files in the"
                " provided dataset by applying a high-pass butterworth filter."
            ),
        ),
    ] = True,
    clean_audio: Annotated[
        bool,
        typer.Option(
            help=(
                "Whether to clean the audio files in the provided dataset using noise"
                " reduction algorithms."
            ),
        ),
    ] = False,
    clean_strength: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help=(
                "The intensity of the cleaning to apply to the audio files in the"
                " provided dataset."
            ),
        ),
    ] = 0.7,
    cpu_cores: Annotated[
        int,
        typer.Option(
            min=1,
            max=CORES,
            help="The number of CPU cores to use for preprocessing",
        ),
    ] = CORES,
) -> None:
    """
    Preprocess a dataset of audio files for training a voice
    model.
    """
    start_time = time.perf_counter()

    _preprocess_dataset(
        model_name=model_name,
        dataset=dataset,
        sample_rate=sample_rate,
        split_method=split_method,
        chunk_len=chunk_len,
        overlap_len=overlap_len,
        filter_audio=filter_audio,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        cpu_cores=cpu_cores,
    )

    rprint("[+] Dataset succesfully preprocessed!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))


@app.command()
def get_gpu_information() -> None:
    """Retrieve information on locally available GPUs."""
    start_time = time.perf_counter()
    rprint("[+] Retrieving GPU Information...")
    gpu_infos = _get_gpu_info()

    rprint("[+] GPU Information successfully retrieved!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))

    table = Table()
    table.add_column("Name", style="green")
    table.add_column("Index", style="green")

    for gpu_name, gpu_index in gpu_infos:
        table.add_row(gpu_name, str(gpu_index))

    rprint(table)


@app.command(no_args_is_help=True)
def extract_features(
    model_name: Annotated[
        str,
        typer.Argument(help="The name of the voice model to be trained."),
    ],
    rvc_version: Annotated[
        RVCVersion,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_rvc_version,
            help="Version of RVC to use for training the voice model.",
        ),
    ] = RVCVersion.V2,
    f0_method: Annotated[
        TrainingF0Method,
        typer.Option(
            case_sensitive=False,
            autocompletion=complete_f0_method,
            help="The method to use for extracting pitch features.",
        ),
    ] = TrainingF0Method.RMVPE,
    hop_length: Annotated[
        int,
        typer.Option(
            min=1,
            max=512,
            help=(
                "The hop length to use for extracting pitch features. Only used"
                " with the CREPE pitch extraction method."
            ),
        ),
    ] = 128,
    embedder_model: Annotated[
        EmbedderModel,
        typer.Option(
            autocompletion=complete_embedder_model,
            help="The model to use for extracting audio embeddings.",
            case_sensitive=False,
        ),
    ] = EmbedderModel.CONTENTVEC,
    custom_embedder_model: Annotated[
        str | None,
        typer.Option(
            exists=True,
            resolve_path=True,
            dir_okay=True,
            file_okay=False,
            help="The path to a custom model to use for extracting audio embeddings.",
        ),
    ] = None,
    include_mutes: Annotated[
        int,
        typer.Option(
            help=(
                "The number of mute audio files to include in the generated"
                " training file list. Adding silent files enables the voice model to"
                " handle pure silence in inferred audio files. If the preprocessed"
                " audio dataset already contains segments of pure silence, set this"
                " to 0."
            ),
            min=0,
            max=10,
        ),
    ] = 2,
    cpu_cores: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.DEVICE_OPTIONS,
            help="The number of CPU cores to use for feature extraction.",
            min=1,
            max=cpu_count(),
        ),
    ] = cpu_count(),
    gpus: Annotated[
        list[int] | None,
        typer.Option(
            rich_help_panel=PanelName.DEVICE_OPTIONS,
            min=0,
            help="The device ids of the GPUs to use for extracting audio embeddings.",
        ),
    ] = None,
) -> None:
    """
    Extract features from the preprocessed dataset associated with a
    voice model to be trained.
    """
    start_time = time.perf_counter()

    gpu_set = set(gpus) if gpus is not None else None
    _extract_features(
        model_name=model_name,
        rvc_version=rvc_version,
        f0_method=f0_method,
        hop_length=hop_length,
        embedder_model=embedder_model,
        custom_embedder_model=custom_embedder_model,
        include_mutes=include_mutes,
        cpu_cores=cpu_cores,
        gpus=gpu_set,
    )

    rprint("[+] Dataset features succesfully extracted!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))


@app.command(no_args_is_help=True)
def run_training(
    model_name: Annotated[
        str,
        typer.Argument(
            help="The name of the voice model to train.",
        ),
    ],
    vocoder: Annotated[
        Vocoder,
        typer.Option(
            rich_help_panel=PanelName.MAIN_OPTIONS,
            case_sensitive=False,
            autocompletion=complete_vocoder,
            help=(
                "The vocoder to use for audio synthesis during training. HiFi-GAN"
                " provides basic audio fidelity, while RefineGAN provides the highest"
                " audio fidelity."
            ),
        ),
    ] = Vocoder.HIFI_GAN,
    index_algorithm: Annotated[
        IndexAlgorithm,
        typer.Option(
            rich_help_panel=PanelName.MAIN_OPTIONS,
            case_sensitive=False,
            autocompletion=complete_index_algorithm,
            help=(
                "The method to use for generating an index file for the trained voice"
                " model. KMeans is a clustering algorithm that divides the dataset"
                " into K clusters. This setting is particularly useful for large"
                " datasets."
            ),
        ),
    ] = IndexAlgorithm.AUTO,
    num_epochs: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.TRAINING_OPTIONS,
            help="The number of epochs to train the voice model.",
            min=1,
        ),
    ] = 500,
    batch_size: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.TRAINING_OPTIONS,
            help=(
                "The number of samples to include in each training batch. It is"
                " advisable to align this value with the available VRAM of your GPU. A"
                " setting of 4 offers improved accuracy but slower processing, while 8"
                " provides faster and standard results."
            ),
            min=1,
        ),
    ] = 8,
    detect_overtraining: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.TRAINING_OPTIONS,
            help=(
                "Whether to detect overtraining to prevent the voice model from"
                " learning the training data too well and losing the ability to"
                " generalize to new data."
            ),
        ),
    ] = False,
    overtraining_threshold: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.TRAINING_OPTIONS,
            help=(
                "The maximum number of epochs to continue training without any observed"
                " improvement in model performance."
            ),
            min=1,
        ),
    ] = 50,
    save_interval: Annotated[
        int,
        typer.Option(
            rich_help_panel=PanelName.SAVE_OPTIONS,
            help=(
                "The epoch interval at which to save voice model weights and"
                " checkpoints. The best model weights and latest model checkpoint are"
                " always saved regardless of this setting."
            ),
            min=1,
        ),
    ] = 10,
    save_all_checkpoints: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.SAVE_OPTIONS,
            help=(
                "Whether to save a voice model checkpoint at each save interval. If"
                " False, only the latest voice model checkpoint will be saved."
            ),
        ),
    ] = False,
    save_all_weights: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.SAVE_OPTIONS,
            help=(
                "Whether to save voice model weights at each save interval. If False,"
                " only the best voice model weights will be saved."
            ),
        ),
    ] = False,
    clear_saved_data: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.SAVE_OPTIONS,
            help=(
                "Whether to delete any existing saved training data associated with the"
                " voice model before starting a new training session. Enable this"
                " settingonly if you are training a new voice model from scratch or"
                " restarting training."
            ),
        ),
    ] = False,
    use_pretrained: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.PRETRAINED_MODEL_OPTIONS,
            help=(
                "Whether to use a pretrained model for training. This reduces training"
                " time and improves overall voice model performance."
            ),
        ),
    ] = True,
    custom_pretrained: Annotated[
        str | None,
        typer.Option(
            rich_help_panel=PanelName.PRETRAINED_MODEL_OPTIONS,
            help=(
                "The name of a custom pretrained model to use for training. Using a"
                " custom generator/discriminator model can lead to superior results, as"
                " selecting the most suitable pretrained modeltailored to the specific"
                " use case can significantly enhanceperformance."
            ),
        ),
    ] = None,
    preload_dataset: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.MEMORY_OPTIONS,
            help=(
                "Whether to preload all training data into GPU memory. This can improve"
                " training speed but requires a lot of VRAM."
            ),
        ),
    ] = False,
    save_memory: Annotated[
        bool,
        typer.Option(
            rich_help_panel=PanelName.MEMORY_OPTIONS,
            help=(
                "Whether to reduce VRAM usage at the cost of slower training speed by"
                " enabling activation checkpointing. This is useful for GPUs with"
                " limited memory (e.g., <6GB VRAM) or when training with a batch size"
                " larger than what your GPU can normally accommodate."
            ),
        ),
    ] = False,
    gpus: Annotated[
        list[int] | None,
        typer.Option(
            rich_help_panel=PanelName.DEVICE_OPTIONS,
            help=(
                "The device ids of the GPUs to use for training. If None, only CPU will"
                " be used."
            ),
        ),
    ] = None,
) -> None:
    """
    Train a voice model using its associated preprocessed dataset and
    extracted features.
    """
    start_time = time.perf_counter()

    gpu_set = set(gpus) if gpus is not None else None
    model_file, index_file = _run_training(
        model_name=model_name,
        vocoder=vocoder,
        index_algorithm=index_algorithm,
        num_epochs=num_epochs,
        batch_size=batch_size,
        detect_overtraining=detect_overtraining,
        overtraining_threshold=overtraining_threshold,
        save_interval=save_interval,
        save_all_checkpoints=save_all_checkpoints,
        save_all_weights=save_all_weights,
        clear_saved_data=clear_saved_data,
        use_pretrained=use_pretrained,
        custom_pretrained=custom_pretrained,
        preload_dataset=preload_dataset,
        save_memory=save_memory,
        gpus=gpu_set,
    )

    rprint("[+] Voice model succesfully trained!")
    rprint()
    rprint("Elapsed time:", format_duration(time.perf_counter() - start_time))
    rprint(Panel(f"[green]{model_file}", title="Model File"))
    rprint(Panel(f"[green]{index_file}", title="Index File"))
