"""
Module which exposes functionality for extracting training features from
audio datasets.
"""

from __future__ import annotations

from multiprocessing import cpu_count

from ultimate_rvc.core.common import (
    display_progress,
    get_combined_file_hash,
    validate_model_exists,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    ModelAsssociatedEntityNotFoundError,
    Step,
)
from ultimate_rvc.core.train.common import validate_devices
from ultimate_rvc.typing_extra import (
    DeviceType,
    EmbedderModel,
    TrainingF0Method,
)


def extract_features(
    model_name: str,
    f0_method: TrainingF0Method = TrainingF0Method.RMVPE,
    hop_length: int = 128,
    embedder_model: EmbedderModel = EmbedderModel.CONTENTVEC,
    custom_embedder_model: str | None = None,
    include_mutes: int = 2,
    cpu_cores: int = cpu_count(),
    hardware_acceleration: DeviceType = DeviceType.AUTOMATIC,
    gpu_ids: set[int] | None = None,
) -> None:
    """
    Extract features from the preprocessed dataset associated with a
    voice model to be trained.

    Parameters
    ----------
    model_name : str
        The name of the voice model to be trained.
    f0_method : TrainingF0Method, default=TrainingF0Method.RMVPE
        The method to use for extracting pitch features.
    hop_length : int, default=128
        The hop length to use for extracting pitch features. Only used
        with the CREPE pitch extraction method.
    embedder_model : EmbedderModel, default=EmbedderModel.CONTENTVEC
        The model to use for extracting audio embeddings.
    custom_embedder_model : StrPath, optional
        The name of the custom embedder model to use for extracting
        audio embeddings.
    include_mutes : int, default=2
        The number of mute audio files to include in the generated
        training file list. Adding silent files enables the voice model
        to handle pure silence in inferred audio files. If the
        preprocessed audio dataset already contains segments of pure
        silence, set this to 0.
    cpu_cores : int, default=cpu_count()
        The number of CPU cores to use for feature extraction.
    hardware_acceleration : DeviceType, default=DeviceType.AUTOMATIC
        The type of hardware acceleration to use for feature extraction.
        `AUTOMATIC` will select the first available GPU and fall back to
        CPU if no GPUs are available.
    gpu_ids : set[int], optional
        Set of ids of the GPUs to use for feature extraction when `GPU`
        is selected for hardware acceleration.

    Raises
    ------
    ModelAsssociatedEntityNotFoundError
        If no preprocessed dataset audio files are associated with the
        voice model identified by the provided name.

    """
    model_path = validate_model_exists(model_name, Entity.TRAINING_MODEL)
    sliced_audios16k_path = model_path / "sliced_audios_16k"
    if not sliced_audios16k_path.is_dir() or not any(sliced_audios16k_path.iterdir()):
        raise ModelAsssociatedEntityNotFoundError(
            Entity.PREPROCESSED_AUDIO_DATASET_FILES,
            model_name,
            Step.DATASET_PREPROCESSING,
        )

    custom_embedder_model_path, combined_file_hash = None, None
    chosen_embedder_model, embedder_model_id = [embedder_model] * 2
    if embedder_model == EmbedderModel.CUSTOM:
        custom_embedder_model_path = validate_model_exists(
            custom_embedder_model,
            Entity.CUSTOM_EMBEDDER_MODEL,
        )
        json_file = custom_embedder_model_path / "config.json"
        bin_path = custom_embedder_model_path / "pytorch_model.bin"

        combined_file_hash = get_combined_file_hash([json_file, bin_path])
        chosen_embedder_model = str(custom_embedder_model_path)
        embedder_model_id = f"custom_{combined_file_hash}"

    f0_method_id = f0_method
    if f0_method in {TrainingF0Method.CREPE, TrainingF0Method.CREPE_TINY}:
        f0_method_id = f"{f0_method}_{hop_length}"

    device_type, device_ids = validate_devices(hardware_acceleration, gpu_ids)

    devices = (
        [f"{device_type}:{device_id}" for device_id in device_ids]
        if device_ids
        else [device_type]
    )
    # NOTE The lazy_import function does not work with the package below
    # so we import it here manually
    from ultimate_rvc.rvc.train.extract import extract  # noqa: PLC0415

    file_infos = extract.initialize_extraction(
        str(model_path),
        f0_method_id,
        embedder_model_id,
    )
    extract.update_model_info(
        str(model_path),
        chosen_embedder_model,
        combined_file_hash,
    )
    display_progress("[~] Extracting pitch features...")
    extract.run_pitch_extraction(file_infos, devices, f0_method, hop_length, cpu_cores)
    display_progress("[~] Extracting audio embeddings...")
    extract.run_embedding_extraction(
        file_infos,
        devices,
        embedder_model,
        (
            str(custom_embedder_model_path)
            if custom_embedder_model_path is not None
            else None
        ),
        cpu_cores,
    )
    # NOTE The lazy_import function does not work with the package below
    # so we import it here manually
    from ultimate_rvc.rvc.train.extract import preparing_files  # noqa: PLC0415

    preparing_files.generate_config(str(model_path))
    preparing_files.generate_filelist(
        str(model_path),
        include_mutes,
        f0_method_id,
        embedder_model_id,
    )
