"""
Module which exposes functionality for training voice conversion
models.
"""

from __future__ import annotations

import logging
import os
import re
import signal

from ultimate_rvc.core.common import (
    TRAINING_MODELS_DIR,
    VOICE_MODELS_DIR,
    copy_files_to_new_dir,
    json_dump,
    json_load,
    validate_model_exists,
)
from ultimate_rvc.core.exceptions import (
    Entity,
    IncompatiblePretrainedModelError,
    IncompatibleVocoderError,
    ModelAsssociatedEntityNotFoundError,
    ModelExistsError,
    NotProvidedError,
    Step,
)
from ultimate_rvc.core.train.common import validate_devices
from ultimate_rvc.core.train.typing_extra import ModelInfo, TrainingInfo
from ultimate_rvc.typing_extra import (
    DeviceType,
    IndexAlgorithm,
    PretrainedType,
    Vocoder,
)

logger = logging.getLogger(__name__)


def _get_pretrained_model(
    pretrained_type: PretrainedType,
    vocoder: Vocoder,
    sample_rate: int,
    custom_pretrained: str | None = None,
) -> tuple[str, str]:
    """
    Get the pretrained model to finetune a voice model on.

    Parameters
    ----------
    pretrained_type : PretrainedType
        The type of pretrained model to finetune the voice model on
    vocoder : str
        The vocoder to use for audio synthesis when training the voice
        model.
    sample_rate : int
        The sample rate of the preprocessed dataset associated with the
        voice model to be trained.
    custom_pretrained : str, optional
        The name of a custom pretrained model to finetune the voice
        model on

    Returns
    -------
    pg : str
        The path to the generator of the pretrained model to finetune.
    pd : str
        The path to the discriminator of the pretrained model to
        finetune.

    Raises
    ------
    ModelAsssociatedEntityNotFoundError
        If the voice model to be trained does not have an associated
        dataset file list or if a custom pretrained
        generator/discriminator model does not have an associated
        generator or discriminator.
    IncompatiblePretrainedModelError
        if a custom pretrained model is not compatible with the sample
        rate of the preprocessed dataset associated with the voice model
        to be trained.
    IncompatibleVocoderError
        If the default pretrained model is not compatible with the
        vocoder to be used for audio synthesis when training the voice
        model.

    """
    match pretrained_type:
        case PretrainedType.NONE:
            pg, pd = "", ""
        case PretrainedType.DEFAULT:
            from ultimate_rvc.rvc.lib.tools.pretrained_selector import (  # noqa: PLC0415
                pretrained_selector,
            )

            if not vocoder == Vocoder.HIFI_GAN:
                raise IncompatibleVocoderError(vocoder)

            pg, pd = pretrained_selector(vocoder, sample_rate=sample_rate)
        case PretrainedType.CUSTOM:
            custom_pretrained_path = validate_model_exists(
                custom_pretrained,
                Entity.CUSTOM_PRETRAINED_MODEL,
            )
            # NOTE simply done to appease the type checker
            custom_pretrained = custom_pretrained_path.name

            # TODO need to make this cleaner
            custom_pretrained_sample_rate = int(custom_pretrained[-3:-1]) * 1000
            if not custom_pretrained_sample_rate == sample_rate:
                raise IncompatiblePretrainedModelError(custom_pretrained, sample_rate)

            pg = next(
                (
                    str(path)
                    for path in custom_pretrained_path.iterdir()
                    if re.match(r"^(G|f0G).*\.pth$|.*G\.pth$", path.name)
                ),
                None,
            )
            if pg is None:
                raise ModelAsssociatedEntityNotFoundError(
                    Entity.GENERATOR,
                    custom_pretrained,
                )
            pd = next(
                (
                    str(path)
                    for path in custom_pretrained_path.iterdir()
                    if re.match(r"^(D|f0D).*\.pth$|.*D\.pth$", path.name)
                ),
                None,
            )
            if pd is None:
                raise ModelAsssociatedEntityNotFoundError(
                    Entity.DISCRIMINATOR,
                    custom_pretrained,
                )

    return pg, pd


def run_training(
    model_name: str,
    num_epochs: int = 500,
    batch_size: int = 8,
    detect_overtraining: bool = False,
    overtraining_threshold: int = 50,
    vocoder: Vocoder = Vocoder.HIFI_GAN,
    index_algorithm: IndexAlgorithm = IndexAlgorithm.AUTO,
    pretrained_type: PretrainedType = PretrainedType.DEFAULT,
    custom_pretrained: str | None = None,
    save_interval: int = 10,
    save_all_checkpoints: bool = False,
    save_all_weights: bool = False,
    clear_saved_data: bool = False,
    upload_model: bool = False,
    upload_name: str | None = None,
    hardware_acceleration: DeviceType = DeviceType.AUTOMATIC,
    gpu_ids: set[int] | None = None,
    preload_dataset: bool = False,
    reduce_memory_usage: bool = False,
) -> None:
    """

    Train a voice model using its associated preprocessed dataset and
    extracted features.

    Parameters
    ----------
    model_name : str
        The name of the voice model to train.
    num_epochs : int, default=500
        The number of epochs to train the voice model. A higher number
        can improve voice model performance but may lead to
        overtraining.
    batch_size : int, default=8
        The number of samples to include in each training batch. It is
        advisable to align this value with the available VRAM of your
        GPU. A setting of 4 offers improved accuracy but slower
        processing, while 8 provides faster and standard results.
    detect_overtraining : bool, default=False
        Whether to detect overtraining to prevent the voice model from
        learning the training data too well and losing the ability to
        generalize to new data.
    overtraining_threshold : int, default=50
        The maximum number of epochs to continue training without any
        observed improvement in voice model performance.
    vocoder : Vocoder, default=Vocoder.HIFI_GAN
        The vocoder to use for audio synthesis during training. HiFi-GAN
        provides basic audio fidelity, while RefineGAN provides the
        highest audio fidelity.
    index_algorithm : IndexAlgorithm, default=IndexAlgorithm.AUTO
        The method to use for generating an index file for the trained
        voice model. KMeans is particularly useful for large datasets.
    pretrained_type : PretrainedType, default=PretrainedType.DEFAULT
        The type of pretrained model to finetune the voice model on.
        "None" will train the voice model from scratch, while
        "Default" will use a pretrained model tailored to the specific
        voice model architecture. "Custom" will use a custom pretrained
        model that you provide.
    custom_pretrained: str, optional
        The name of a custom pretrained model to finetune the voice
        model on.
    save_interval : int, default=10
        The epoch interval at which to to save voice model weights and
        checkpoints. The best model weights are always saved regardless
        of this setting.
    save_all_checkpoints : bool, default=False
        Whether to save a unique checkpoint at each save interval. If
        not enabled, only the latest checkpoint will be saved at each
        interval.
    save_all_weights : bool, default=False
        Whether to save unique voice model weights at each save
        interval. If not enabled, only the best voice model weights will
        be saved.
    clear_saved_data : bool, default=False
        Whether to delete any existing training data associated
        with the voice model before training commences. Enable this
        setting only if you are training a new voice model from scratch
        or restarting training.
    upload_model : bool, default=False
        Whether to automatically upload the trained voice model so that
        it can be used for audio generation tasks within the Ultimate
        RVC app.
    upload_name : str, optional
        The name to give the uploaded voice model.
    hardware_acceleration : DeviceType, default=DeviceType.AUTOMATIC
        The type of hardware acceleration to use when training the voice
        model. `AUTOMATIC` will select the first available GPU and fall
        back to CPU if no GPUs are available.
    gpu_ids : set[int], optional
        Set of ids of the GPUs to use for training the voice model when
        `GPU` is selected for hardware acceleration.
    preload_dataset : bool, default=False
        Whether to preload all training data into GPU memory. This can
        improve training speed but requires a lot of VRAM.
    reduce_memory_usage : bool, default=False
        Whether to reduce VRAM usage at the cost of slower training
        speed by enabling activation checkpointing. This is useful for
        GPUs with limited memory (e.g., <6GB VRAM) or when training with
        a batch size larger than what your GPU can normally accommodate.

    Raises
    ------
    ModelAsssociatedEntityNotFoundError
        If the voice model to be trained does not have an associated
        dataset file list.
    NotProvidedError
        If an upload name is not provided when the upload parameter is
        set
    ModelExistsError
        If a voice with the provided upload name already exists when the
        upload parameter is set


    """
    model_path = validate_model_exists(model_name, Entity.TRAINING_MODEL)
    filelist_path = model_path / "filelist.txt"
    if not filelist_path.is_file():
        raise ModelAsssociatedEntityNotFoundError(
            Entity.DATASET_FILE_LIST,
            model_name,
            Step.FEATURE_EXTRACTION,
        )
    upload_model_path = None
    if upload_model:
        if not upload_name:
            raise NotProvidedError(Entity.UPLOAD_NAME)
        upload_model_path = VOICE_MODELS_DIR / upload_name.strip()
        if upload_model_path.is_dir():
            raise ModelExistsError(Entity.VOICE_MODEL, upload_name)

    model_info_dict = json_load(model_path / "model_info.json")

    model_info = ModelInfo.model_validate(model_info_dict)
    sample_rate = model_info.sample_rate

    pg, pd = _get_pretrained_model(
        pretrained_type,
        vocoder,
        sample_rate,
        custom_pretrained,
    )

    from ultimate_rvc.rvc.train.train import main as train_main  # noqa: PLC0415

    device_type, device_ids = validate_devices(hardware_acceleration, gpu_ids)

    train_main(
        model_name,
        sample_rate,
        vocoder,
        num_epochs,
        batch_size,
        save_interval,
        not save_all_checkpoints,
        save_all_weights,
        pg,
        pd,
        detect_overtraining,
        overtraining_threshold,
        clear_saved_data,
        preload_dataset,
        reduce_memory_usage,
        device_type,
        device_ids,
    )

    model_file = model_path / f"{model_name}_best.pth"

    if not model_file.is_file():
        return

    from ultimate_rvc.rvc.train.process.extract_index import (  # noqa: PLC0415
        main as extract_index_main,
    )

    extract_index_main(str(model_path), index_algorithm)

    index_file = model_path / f"{model_name}.index"
    if upload_model_path and model_file.is_file() and index_file.is_file():
        copy_files_to_new_dir([index_file, model_file], upload_model_path)


def get_trained_model_files(model_name: str) -> list[str] | None:
    """
    Get the best weights file and the index file for a trained voice
    model, if they exist.

    Parameters
    ----------
    model_name : str
        The name of the trained voice model.

    Returns
    -------
    list[str] | None
        A list containing the paths to the best weights file and the
        index file for the trained voice model, if they exist.
        Otherwise, None.

    """
    model_path = TRAINING_MODELS_DIR / model_name
    model_file = model_path / f"{model_name}_best.pth"
    index_file = model_path / f"{model_name}.index"
    if not model_file.is_file() or not index_file.is_file():
        return None
    return [str(model_file), str(index_file)]


def stop_training(model_name: str) -> None:
    """
    Stop the training of a voice model.

    Parameters
    ----------
    model_name : str
        The name of the voice model to stop training for.

    """
    training_info_path = TRAINING_MODELS_DIR / model_name / "config.json"
    try:
        training_info_dict = json_load(training_info_path)
        training_info = TrainingInfo.model_validate(training_info_dict)
        process_ids = training_info.process_pids
        for pid in process_ids:
            os.kill(pid, signal.SIGTERM)
        training_info.process_pids = []
        updated_training_info_dict = training_info.model_dump()
        json_dump(updated_training_info_dict, training_info_path)
    except Exception as e:
        logger.error("Error stopping training: %s", e)  # noqa: TRY400
