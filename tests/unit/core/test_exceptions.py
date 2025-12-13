"""Comprehensive unit tests for ultimate_rvc.core.exceptions module."""

from __future__ import annotations

from pathlib import Path

import pytest

from ultimate_rvc.core.exceptions import (
    ComponentNotInstatiatedError,
    ConfigExistsError,
    ConfigNotFoundError,
    Entity,
    EntityExistsError,
    EntityNotFoundError,
    EventNotInstantiatedError,
    GPUNotFoundError,
    HttpUrlError,
    InvalidAudioFormatError,
    InvalidLocationError,
    Location,
    ModelAsssociatedEntityNotFoundError,
    ModelEntity,
    ModelExistsError,
    ModelNotFoundError,
    NotFoundError,
    NotInstantiatedError,
    NotProvidedError,
    PretrainedModelExistsError,
    PretrainedModelIncompatibleError,
    PretrainedModelNotAvailableError,
    Step,
    UIMessage,
    UploadLimitError,
    UploadTypeError,
    YoutubeUrlError,
)
from ultimate_rvc.typing_extra import TrainingSampleRate


class TestEntityEnum:
    """Test cases for Entity enumeration."""

    def test_entity_enum_values_exist(self) -> None:
        """Test that all expected Entity enum values exist."""
        # General entities
        assert Entity.FILE == "file"
        assert Entity.FILES == "files"
        assert Entity.DIRECTORY == "directory"
        assert Entity.DIRECTORIES == "directories"

        # Model entities
        assert Entity.MODEL_NAME == "model name"
        assert Entity.MODEL_NAMES == "model names"
        assert Entity.UPLOAD_NAME == "upload name"
        assert Entity.MODEL == "model"
        assert Entity.VOICE_MODEL == "voice model"
        assert Entity.TRAINING_MODEL == "training model"
        assert Entity.CUSTOM_EMBEDDER_MODEL == "custom embedder model"
        assert Entity.CUSTOM_PRETRAINED_MODEL == "custom pretrained model"
        assert Entity.GENERATOR == "generator"
        assert Entity.DISCRIMINATOR == "discriminator"
        assert Entity.MODEL_FILE == "model file"
        assert Entity.MODEL_BIN_FILE == "pytorch_model.bin file"
        assert Entity.CONFIG_JSON_FILE == "config.json file"

        # Audio entities
        assert Entity.AUDIO_TRACK == "audio track"
        assert Entity.AUDIO_TRACK_GAIN_PAIRS == "pairs of audio track and gain"
        assert Entity.VOICE_TRACK == "voice track"
        assert Entity.SPEECH_TRACK == "speech track"
        assert Entity.VOCALS_TRACK == "vocals track"
        assert Entity.SONG_DIR == "song directory"
        assert Entity.DATASET == "dataset"
        assert Entity.DATASETS == "datasets"
        assert Entity.DATASET_NAME == "dataset name"
        assert Entity.DATASET_FILE_LIST == "dataset file list"
        assert (
            Entity.PREPROCESSED_AUDIO_DATASET_FILES
            == "preprocessed dataset audio files"
        )

        # Source entities
        assert Entity.SOURCE == "source"
        assert Entity.URL == "URL"

        # GPU entities
        assert Entity.GPU_IDS == "GPU IDs"

        # Config entities
        assert Entity.CONFIG == "configuration"
        assert Entity.CONFIG_NAME == "configuration name"
        assert Entity.CONFIG_NAMES == "configuration names"
        assert Entity.EVENT == "event"
        assert Entity.COMPONENT == "component"

    def test_entity_enum_is_str_enum(self) -> None:
        """Test that Entity inherits from StrEnum."""
        assert isinstance(Entity.FILE, str)
        assert str(Entity.FILE) == "file"

    def test_entity_enum_comparison(self) -> None:
        """Test Entity enum string comparison."""
        assert Entity.FILE == "file"
        assert Entity.FILE != "files"
        assert Entity.FILE != Entity.FILES


class TestLocationEnum:
    """Test cases for Location enumeration."""

    def test_location_enum_values_exist(self) -> None:
        """Test that all expected Location enum values exist."""
        # Audio locations
        assert Location.AUDIO_ROOT == "the root of the audio base directory"
        assert (
            Location.INTERMEDIATE_AUDIO_ROOT
            == "the root of the intermediate audio base directory"
        )
        assert Location.SPEECH_AUDIO_ROOT == "the root of the speech audio directory"
        assert (
            Location.TRAINING_AUDIO_ROOT == "the root of the training audio directory"
        )
        assert Location.OUTPUT_AUDIO_ROOT == "the root of the output audio directory"

        # Model locations
        assert Location.EXTRACTED_ZIP_FILE == "extracted zip file"

    def test_location_enum_is_str_enum(self) -> None:
        """Test that Location inherits from StrEnum."""
        assert isinstance(Location.AUDIO_ROOT, str)
        assert str(Location.AUDIO_ROOT) == "the root of the audio base directory"


class TestStepEnum:
    """Test cases for Step enumeration."""

    def test_step_enum_values_exist(self) -> None:
        """Test that all expected Step enum values exist."""
        assert Step.DATASET_PREPROCESSING == "dataset preprocessing"
        assert Step.FEATURE_EXTRACTION == "feature extraction"

    def test_step_enum_is_str_enum(self) -> None:
        """Test that Step inherits from StrEnum."""
        assert isinstance(Step.DATASET_PREPROCESSING, str)
        assert str(Step.DATASET_PREPROCESSING) == "dataset preprocessing"


class TestUIMessageEnum:
    """Test cases for UIMessage enumeration."""

    def test_ui_message_enum_values_exist(self) -> None:
        """Test that all expected UIMessage enum values exist."""
        # General messages
        assert UIMessage.NO_UPLOADED_FILES == "No files selected."

        # Audio messages
        assert UIMessage.NO_AUDIO_TRACK == "No audio tracks provided."
        assert (
            UIMessage.NO_SPEECH_AUDIO_FILES
            == "No files selected. Please select one or more speech audio files to"
            " delete."
        )
        assert (
            UIMessage.NO_OUTPUT_AUDIO_FILES
            == "No files selected. Please select one or more output audio files to"
            " delete."
        )
        assert UIMessage.NO_SONG_DIR == "No song directory selected."
        assert (
            UIMessage.NO_SONG_DIRS
            == "No song directories selected. Please select one or more song"
            " directories"
            " containing intermediate audio files to delete."
        )
        assert (
            UIMessage.NO_DATASETS
            == "No datasets selected. Please select one or more datasets containing"
            " audio"
            " files to delete."
        )

        # Model messages
        assert UIMessage.NO_MODEL == "No model selected."
        assert UIMessage.NO_MODELS == "No models selected."
        assert UIMessage.NO_VOICE_MODEL == "No voice model selected."
        assert UIMessage.NO_VOICE_MODELS == "No voice models selected."
        assert UIMessage.NO_TRAINING_MODELS == "No training models selected."
        assert (
            UIMessage.NO_CUSTOM_EMBEDDER_MODEL == "No custom embedder model selected."
        )
        assert (
            UIMessage.NO_CUSTOM_EMBEDDER_MODELS == "No custom embedder models selected."
        )
        assert (
            UIMessage.NO_CUSTOM_PRETRAINED_MODELS
            == "No custom pretrained models selected."
        )
        assert (
            UIMessage.NO_CUSTOM_PRETRAINED_MODEL
            == "No custom pretrained model selected."
        )

        # Source messages
        assert (
            UIMessage.NO_AUDIO_SOURCE
            == "No source provided. Please provide a valid Youtube URL, local audio"
            " file"
            " or song directory."
        )
        assert (
            UIMessage.NO_TEXT_SOURCE
            == "No source provided. Please provide a valid text string or path to a"
            " text file."
        )

        # GPU messages
        assert UIMessage.NO_GPUS == "No GPUs selected."

        # Config messages
        assert UIMessage.NO_CONFIG == "No configuration selected."
        assert UIMessage.NO_CONFIGS == "No configurations selected."

    def test_ui_message_enum_is_str_enum(self) -> None:
        """Test that UIMessage inherits from StrEnum."""
        assert isinstance(UIMessage.NO_UPLOADED_FILES, str)
        assert str(UIMessage.NO_UPLOADED_FILES) == "No files selected."


class TestTypeAliases:
    """Test cases for type aliases."""

    def test_audio_file_entity_type_alias(self) -> None:
        """Test AudioFileEntity type alias contains correct entities."""
        # This is mainly for documentation - type aliases can't be
        # tested at runtime
        # But we can verify the entities exist
        assert Entity.AUDIO_TRACK in Entity
        assert Entity.VOICE_TRACK in Entity
        assert Entity.SPEECH_TRACK in Entity
        assert Entity.VOCALS_TRACK in Entity
        assert Entity.FILE in Entity

    def test_audio_directory_entity_type_alias(self) -> None:
        """Test AudioDirectoryEntity type alias has correct entities."""
        assert Entity.SONG_DIR in Entity
        assert Entity.DATASET in Entity
        assert Entity.DIRECTORY in Entity

    def test_model_entity_type_alias(self) -> None:
        """Test ModelEntity type alias contains correct entities."""
        assert Entity.MODEL in Entity
        assert Entity.VOICE_MODEL in Entity
        assert Entity.TRAINING_MODEL in Entity
        assert Entity.CUSTOM_EMBEDDER_MODEL in Entity
        assert Entity.CUSTOM_PRETRAINED_MODEL in Entity

    def test_config_entity_type_alias(self) -> None:
        """Test ConfigEntity type alias contains correct entities."""
        assert Entity.EVENT in Entity
        assert Entity.COMPONENT in Entity


class TestNotProvidedError:
    """Test cases for NotProvidedError exception."""

    def test_not_provided_error_basic(self) -> None:
        """Test basic NotProvidedError functionality."""
        error = NotProvidedError(Entity.FILE)

        assert isinstance(error, ValueError)
        assert str(error) == "No file provided."
        assert error.ui_msg is None

    def test_not_provided_error_with_ui_message(self) -> None:
        """Test NotProvidedError with UI message."""
        error = NotProvidedError(Entity.VOICE_MODEL, UIMessage.NO_VOICE_MODEL)

        assert str(error) == "No voice model provided."
        assert error.ui_msg == UIMessage.NO_VOICE_MODEL

    @pytest.mark.parametrize(
        ("entity", "expected_message"),
        [
            (Entity.FILE, "No file provided."),
            (Entity.VOICE_MODEL, "No voice model provided."),
            (Entity.DATASET, "No dataset provided."),
            (Entity.URL, "No URL provided."),
            (Entity.CONFIG, "No configuration provided."),
        ],
    )
    def test_not_provided_error_different_entities(
        self, entity: Entity, expected_message: str
    ) -> None:
        """Test NotProvidedError with different entities."""
        error = NotProvidedError(entity)
        assert str(error) == expected_message

    def test_not_provided_error_inheritance(self) -> None:
        """Test NotProvidedError inheritance."""
        error = NotProvidedError(Entity.FILE)
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)


class TestNotFoundError:
    """Test cases for NotFoundError exception."""

    def test_not_found_error_basic_path(self) -> None:
        """Test basic NotFoundError functionality with path."""
        location = Path("/some/path/file.txt")
        error = NotFoundError(Entity.FILE, location)

        assert isinstance(error, OSError)
        assert str(error) == f"File not found at: {location}"

    def test_not_found_error_with_location_enum(self) -> None:
        """Test NotFoundError with Location enum."""
        error = NotFoundError(Entity.DATASET, Location.AUDIO_ROOT, is_path=False)

        assert str(error) == "Dataset not found in the root of the audio base directory"

    def test_not_found_error_capitalization(self) -> None:
        """Test NotFoundError capitalizes entity names correctly."""
        # Test normal capitalization
        error1 = NotFoundError(Entity.FILE, "/path")
        assert str(error1) == "File not found at: /path"

        # Test entity already uppercase
        error2 = NotFoundError(Entity.URL, "/path")
        assert str(error2) == "URL not found at: /path"

    @pytest.mark.parametrize(
        ("entity", "location", "is_path", "expected_message"),
        [
            (Entity.FILE, "/test/path", True, "File not found at: /test/path"),
            (
                Entity.VOICE_MODEL,
                "/models/voice",
                True,
                "Voice model not found at: /models/voice",
            ),
            (
                Entity.DATASET,
                Location.AUDIO_ROOT,
                False,
                "Dataset not found in the root of the audio base directory",
            ),
            (
                Entity.CONFIG,
                Location.EXTRACTED_ZIP_FILE,
                False,
                "Configuration not found in extracted zip file",
            ),
        ],
    )
    def test_not_found_error_parametrized(
        self,
        entity: Entity,
        location: str | Location,
        is_path: bool,
        expected_message: str,
    ) -> None:
        """Test NotFoundError with various parameters."""
        error = NotFoundError(entity, location, is_path)
        assert str(error) == expected_message

    def test_not_found_error_inheritance(self) -> None:
        """Test NotFoundError inheritance."""
        error = NotFoundError(Entity.FILE, "/path")
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestEntityNotFoundError:
    """Test cases for EntityNotFoundError exception."""

    def test_entity_not_found_error_basic(self) -> None:
        """Test basic EntityNotFoundError functionality."""
        error = EntityNotFoundError(Entity.VOICE_MODEL, "test_model")

        assert isinstance(error, OSError)
        assert str(error) == "Voice model with name 'test_model' not found."

    def test_entity_not_found_error_capitalization(self) -> None:
        """Test EntityNotFoundError capitalizes entity names."""
        error = EntityNotFoundError(Entity.FILE, "test_file")
        assert str(error) == "File with name 'test_file' not found."

    @pytest.mark.parametrize(
        ("entity", "name", "expected_message"),
        [
            (Entity.VOICE_MODEL, "model1", "Voice model with name 'model1' not found."),
            (Entity.CONFIG, "config1", "Configuration with name 'config1' not found."),
            (Entity.DATASET, "data1", "Dataset with name 'data1' not found."),
        ],
    )
    def test_entity_not_found_error_parametrized(
        self, entity: Entity, name: str, expected_message: str
    ) -> None:
        """
        Test EntityNotFoundError with different entities and
        names.
        """
        error = EntityNotFoundError(entity, name)
        assert str(error) == expected_message

    def test_entity_not_found_error_inheritance(self) -> None:
        """Test EntityNotFoundError inheritance."""
        error = EntityNotFoundError(Entity.FILE, "test")
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestModelNotFoundError:
    """Test cases for ModelNotFoundError exception."""

    def test_model_not_found_error_basic(self) -> None:
        """Test basic ModelNotFoundError functionality."""
        error = ModelNotFoundError(Entity.VOICE_MODEL, "test_model")

        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, OSError)
        assert str(error) == "Voice model with name 'test_model' not found."

    @pytest.mark.parametrize(
        ("entity", "name", "expected_message"),
        [
            (Entity.VOICE_MODEL, "voice1", "Voice model with name 'voice1' not found."),
            (
                Entity.TRAINING_MODEL,
                "train1",
                "Training model with name 'train1' not found.",
            ),
            (
                Entity.CUSTOM_EMBEDDER_MODEL,
                "embed1",
                "Custom embedder model with name 'embed1' not found.",
            ),
            (
                Entity.CUSTOM_PRETRAINED_MODEL,
                "pretrained1",
                "Custom pretrained model with name 'pretrained1' not found.",
            ),
            (Entity.MODEL, "model1", "Model with name 'model1' not found."),
        ],
    )
    def test_model_not_found_error_parametrized(
        self, entity: ModelEntity, name: str, expected_message: str
    ) -> None:
        """Test ModelNotFoundError with different model entities."""
        error = ModelNotFoundError(entity, name)
        assert str(error) == expected_message

    def test_model_not_found_error_inheritance(self) -> None:
        """Test ModelNotFoundError inheritance chain."""
        error = ModelNotFoundError(Entity.VOICE_MODEL, "test")
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestConfigNotFoundError:
    """Test cases for ConfigNotFoundError exception."""

    def test_config_not_found_error_basic(self) -> None:
        """Test basic ConfigNotFoundError functionality."""
        error = ConfigNotFoundError("test_config")

        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, OSError)
        assert str(error) == "Configuration with name 'test_config' not found."

    def test_config_not_found_error_inheritance(self) -> None:
        """Test ConfigNotFoundError inheritance chain."""
        error = ConfigNotFoundError("test")
        assert isinstance(error, EntityNotFoundError)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize(
        "config_name",
        ["config1", "my_config", "test-config", "config_with_underscores"],
    )
    def test_config_not_found_error_different_names(self, config_name: str) -> None:
        """
        Test ConfigNotFoundError with different configuration
        names.
        """
        error = ConfigNotFoundError(config_name)
        assert str(error) == f"Configuration with name '{config_name}' not found."


class TestPretrainedModelNotAvailableError:
    """Test cases for PretrainedModelNotAvailableError exception."""

    def test_pretrained_model_not_available_error_basic(self) -> None:
        """Test basic PretrainedModelNotAvailableError functionality."""
        error = PretrainedModelNotAvailableError("test_model")

        assert isinstance(error, OSError)
        assert (
            str(error)
            == "Pretrained model with name 'test_model' is not available for download."
        )

    def test_pretrained_model_not_available_error_with_sample_rate(self) -> None:
        """Test PretrainedModelNotAvailableError with sample rate."""
        error = PretrainedModelNotAvailableError(
            "test_model", TrainingSampleRate.HZ_40K
        )

        assert (
            str(error)
            == "Pretrained model with name 'test_model' and sample rate 40000 is not"
            " available for download."
        )

    @pytest.mark.parametrize(
        ("name", "sample_rate", "expected_message"),
        [
            (
                "model1",
                None,
                "Pretrained model with name 'model1' is not available for download.",
            ),
            (
                "model2",
                TrainingSampleRate.HZ_32K,
                (
                    "Pretrained model with name 'model2' and sample rate 32000 is not"
                    " available for download."
                ),
            ),
            (
                "model3",
                TrainingSampleRate.HZ_48K,
                (
                    "Pretrained model with name 'model3' and sample rate 48000 is not"
                    " available for download."
                ),
            ),
        ],
    )
    def test_pretrained_model_not_available_error_parametrized(
        self, name: str, sample_rate: TrainingSampleRate | None, expected_message: str
    ) -> None:
        """
        Test PretrainedModelNotAvailableError with different
        parameters.
        """
        error = PretrainedModelNotAvailableError(name, sample_rate)
        assert str(error) == expected_message

    def test_pretrained_model_not_available_error_inheritance(self) -> None:
        """Test PretrainedModelNotAvailableError inheritance."""
        error = PretrainedModelNotAvailableError("test")
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestIncompatiblePretrainedModelError:
    """Test cases for IncompatiblePretrainedModelError exception."""

    def test_incompatible_pretrained_model_error_basic(self) -> None:
        """Test basic IncompatiblePretrainedModelError functionality."""
        error = PretrainedModelIncompatibleError(
            "test_model", TrainingSampleRate.HZ_40K
        )

        assert isinstance(error, OSError)
        assert (
            str(error)
            == "Pretrained model with name 'test_model' is incompatible with sample"
            " rate 44000."
        )

    @pytest.mark.parametrize(
        ("name", "sample_rate", "expected_message"),
        [
            (
                "model2",
                TrainingSampleRate.HZ_32K,
                (
                    "Pretrained model with name 'model2' is incompatible with sample"
                    " rate 32000."
                ),
            ),
            (
                "model3",
                TrainingSampleRate.HZ_48K,
                (
                    "Pretrained model with name 'model3' is incompatible with sample"
                    " rate 48000."
                ),
            ),
        ],
    )
    def test_incompatible_pretrained_model_error_parametrized(
        self, name: str, sample_rate: TrainingSampleRate, expected_message: str
    ) -> None:
        """
        Test IncompatiblePretrainedModelError with different
        parameters.
        """
        error = PretrainedModelIncompatibleError(name, sample_rate)
        assert str(error) == expected_message

    def test_incompatible_pretrained_model_error_inheritance(self) -> None:
        """Test IncompatiblePretrainedModelError inheritance."""
        error = PretrainedModelIncompatibleError("test", TrainingSampleRate.HZ_32K)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestGPUNotFoundError:
    """Test cases for GPUNotFoundError exception."""

    def test_gpu_not_found_error_basic(self) -> None:
        """Test basic GPUNotFoundError functionality."""
        error = GPUNotFoundError(0)

        assert isinstance(error, OSError)
        assert str(error) == "No GPU with id 0 found."

    def test_gpu_not_found_error_none_id(self) -> None:
        """Test GPUNotFoundError with None device_id."""
        error = GPUNotFoundError(None)

        assert str(error) == "No GPU with id None found."

    def test_gpu_not_found_error_no_args(self) -> None:
        """Test GPUNotFoundError with no arguments."""
        error = GPUNotFoundError()

        assert str(error) == "No GPU with id None found."

    @pytest.mark.parametrize(
        ("device_id", "expected_message"),
        [
            (0, "No GPU with id 0 found."),
            (1, "No GPU with id 1 found."),
            (None, "No GPU with id None found."),
            (-1, "No GPU with id -1 found."),
        ],
    )
    def test_gpu_not_found_error_parametrized(
        self, device_id: int | None, expected_message: str
    ) -> None:
        """Test GPUNotFoundError with different device IDs."""
        error = GPUNotFoundError(device_id)
        assert str(error) == expected_message

    def test_gpu_not_found_error_inheritance(self) -> None:
        """Test GPUNotFoundError inheritance."""
        error = GPUNotFoundError(0)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestModelAssociatedEntityNotFoundError:
    """Test cases for ModelAsssociatedEntityNotFoundError exception."""

    def test_model_associated_entity_not_found_error_basic(self) -> None:
        """Test basic ModelAsssociatedEntityNotFoundError function."""
        error = ModelAsssociatedEntityNotFoundError(Entity.DATASET, "test_model")

        assert isinstance(error, OSError)
        assert (
            str(error) == "No Dataset associated with the model with name test_model. "
        )

    def test_model_associated_entity_not_found_error_with_step(self) -> None:
        """Test ModelAsssociatedEntityNotFoundError with step."""
        error = ModelAsssociatedEntityNotFoundError(
            Entity.DATASET, "test_model", Step.DATASET_PREPROCESSING
        )

        assert (
            str(error)
            == "No Dataset associated with the model with name test_model. Please run"
            " dataset preprocessing first."
        )

    @pytest.mark.parametrize(
        ("entity", "model_name", "required_step", "expected_message"),
        [
            (
                Entity.DATASET,
                "model1",
                None,
                "No Dataset associated with the model with name model1. ",
            ),
            (
                Entity.CONFIG,
                "model2",
                Step.FEATURE_EXTRACTION,
                (
                    "No Configuration associated with the model with name model2."
                    " Please run feature extraction first."
                ),
            ),
            (
                Entity.FILE,
                "model3",
                Step.DATASET_PREPROCESSING,
                (
                    "No File associated with the model with name model3. Please run"
                    " dataset preprocessing first."
                ),
            ),
        ],
    )
    def test_model_associated_entity_not_found_error_parametrized(
        self,
        entity: Entity,
        model_name: str,
        required_step: Step | None,
        expected_message: str,
    ) -> None:
        """
        Test ModelAsssociatedEntityNotFoundError with different
        parameters.
        """
        error = ModelAsssociatedEntityNotFoundError(entity, model_name, required_step)
        assert str(error) == expected_message

    def test_model_associated_entity_not_found_error_inheritance(self) -> None:
        """Test ModelAsssociatedEntityNotFoundError inheritance."""
        error = ModelAsssociatedEntityNotFoundError(Entity.DATASET, "test")
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestEntityExistsError:
    """Test cases for EntityExistsError exception."""

    def test_entity_exists_error_basic(self) -> None:
        """Test basic EntityExistsError functionality."""
        error = EntityExistsError(Entity.VOICE_MODEL, "test_model")

        assert isinstance(error, OSError)
        assert (
            str(error)
            == "Voice model with name 'test_model' already exists. Please provide a"
            " different name for your voice model."
        )

    @pytest.mark.parametrize(
        ("entity", "name", "expected_message"),
        [
            (
                Entity.VOICE_MODEL,
                "model1",
                (
                    "Voice model with name 'model1' already exists. Please provide a"
                    " different name for your voice model."
                ),
            ),
            (
                Entity.CONFIG,
                "config1",
                (
                    "Configuration with name 'config1' already exists. Please provide a"
                    " different name for your configuration."
                ),
            ),
            (
                Entity.DATASET,
                "data1",
                (
                    "Dataset with name 'data1' already exists. Please provide a"
                    " different name for your dataset."
                ),
            ),
        ],
    )
    def test_entity_exists_error_parametrized(
        self, entity: Entity, name: str, expected_message: str
    ) -> None:
        """Test EntityExistsError with different entities and names."""
        error = EntityExistsError(entity, name)
        assert str(error) == expected_message

    def test_entity_exists_error_inheritance(self) -> None:
        """Test EntityExistsError inheritance."""
        error = EntityExistsError(Entity.FILE, "test")
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestModelExistsError:
    """Test cases for ModelExistsError exception."""

    def test_model_exists_error_basic(self) -> None:
        """Test basic ModelExistsError functionality."""
        error = ModelExistsError(Entity.VOICE_MODEL, "test_model")

        assert isinstance(error, EntityExistsError)
        assert isinstance(error, OSError)
        assert (
            str(error)
            == "Voice model with name 'test_model' already exists. Please provide a"
            " different name for your voice model."
        )

    @pytest.mark.parametrize(
        ("entity", "name", "expected_message"),
        [
            (
                Entity.VOICE_MODEL,
                "voice1",
                (
                    "Voice model with name 'voice1' already exists. Please provide a"
                    " different name for your voice model."
                ),
            ),
            (
                Entity.TRAINING_MODEL,
                "train1",
                (
                    "Training model with name 'train1' already exists. Please provide a"
                    " different name for your training model."
                ),
            ),
            (
                Entity.CUSTOM_EMBEDDER_MODEL,
                "embed1",
                (
                    "Custom embedder model with name 'embed1' already exists. Please"
                    " provide a different name for your custom embedder model."
                ),
            ),
            (
                Entity.CUSTOM_PRETRAINED_MODEL,
                "pretrained1",
                (
                    "Custom pretrained model with name 'pretrained1' already exists."
                    " Please provide a different name for your custom pretrained model."
                ),
            ),
            (
                Entity.MODEL,
                "model1",
                (
                    "Model with name 'model1' already exists. Please provide a"
                    " different name for your model."
                ),
            ),
        ],
    )
    def test_model_exists_error_parametrized(
        self, entity: ModelEntity, name: str, expected_message: str
    ) -> None:
        """Test ModelExistsError with different model entities."""
        error = ModelExistsError(entity, name)
        assert str(error) == expected_message

    def test_model_exists_error_inheritance(self) -> None:
        """Test ModelExistsError inheritance chain."""
        error = ModelExistsError(Entity.VOICE_MODEL, "test")
        assert isinstance(error, EntityExistsError)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestConfigExistsError:
    """Test cases for ConfigExistsError exception."""

    def test_config_exists_error_basic(self) -> None:
        """Test basic ConfigExistsError functionality."""
        error = ConfigExistsError("test_config")

        assert isinstance(error, EntityExistsError)
        assert isinstance(error, OSError)
        assert (
            str(error)
            == "Configuration with name 'test_config' already exists. Please provide a"
            " different name for your configuration."
        )

    def test_config_exists_error_inheritance(self) -> None:
        """Test ConfigExistsError inheritance chain."""
        error = ConfigExistsError("test")
        assert isinstance(error, EntityExistsError)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize(
        "config_name",
        ["config1", "my_config", "test-config", "config_with_underscores"],
    )
    def test_config_exists_error_different_names(self, config_name: str) -> None:
        """Test ConfigExistsError with different configuration names."""
        error = ConfigExistsError(config_name)
        assert (
            str(error)
            == f"Configuration with name '{config_name}' already exists. Please provide"
            " a different name for your configuration."
        )


class TestPretrainedModelExistsError:
    """Test cases for PretrainedModelExistsError exception."""

    def test_pretrained_model_exists_error_basic(self) -> None:
        """Test basic PretrainedModelExistsError functionality."""
        error = PretrainedModelExistsError("test_model", TrainingSampleRate.HZ_40K)

        assert isinstance(error, OSError)
        assert (
            str(error)
            == "Pretrained model with name 'test_model' and sample rate 40000 already"
            " exists."
        )

    @pytest.mark.parametrize(
        ("name", "sample_rate", "expected_message"),
        [
            (
                "model1",
                TrainingSampleRate.HZ_32K,
                (
                    "Pretrained model with name 'model1' and sample rate 32000 already"
                    " exists."
                ),
            ),
            (
                "model2",
                TrainingSampleRate.HZ_40K,
                (
                    "Pretrained model with name 'model2' and sample rate 40000 already"
                    " exists."
                ),
            ),
            (
                "model3",
                TrainingSampleRate.HZ_48K,
                (
                    "Pretrained model with name 'model3' and sample rate 48000 already"
                    " exists."
                ),
            ),
        ],
    )
    def test_pretrained_model_exists_error_parametrized(
        self, name: str, sample_rate: TrainingSampleRate, expected_message: str
    ) -> None:
        """Test PretrainedModelExistsError with different parameters."""
        error = PretrainedModelExistsError(name, sample_rate)
        assert str(error) == expected_message

    def test_pretrained_model_exists_error_inheritance(self) -> None:
        """Test PretrainedModelExistsError inheritance."""
        error = PretrainedModelExistsError("test", TrainingSampleRate.HZ_40K)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestInvalidLocationError:
    """Test cases for InvalidLocationError exception."""

    def test_invalid_location_error_basic(self) -> None:
        """Test basic InvalidLocationError functionality."""
        error = InvalidLocationError(Entity.FILE, Location.AUDIO_ROOT, "/wrong/path")

        assert isinstance(error, OSError)
        assert (
            str(error)
            == "File should be located in the root of the audio base directory but"
            " found at: /wrong/path"
        )

    def test_invalid_location_error_capitalization(self) -> None:
        """
        Test InvalidLocationError capitalizes entity names
        correctly.
        """
        # Test normal capitalization
        error1 = InvalidLocationError(Entity.FILE, Location.AUDIO_ROOT, "/path")
        assert (
            str(error1)
            == "File should be located in the root of the audio base directory but"
            " found at: /path"
        )

        # Test entity already uppercase
        error2 = InvalidLocationError(Entity.URL, Location.AUDIO_ROOT, "/path")
        assert (
            str(error2)
            == "URL should be located in the root of the audio base directory but found"
            " at: /path"
        )

    @pytest.mark.parametrize(
        ("entity", "location", "path", "expected_message"),
        [
            (
                Entity.FILE,
                Location.AUDIO_ROOT,
                "/wrong/path",
                (
                    "File should be located in the root of the audio base directory but"
                    " found at: /wrong/path"
                ),
            ),
            (
                Entity.DATASET,
                Location.TRAINING_AUDIO_ROOT,
                "/bad/location",
                (
                    "Dataset should be located in the root of the training audio"
                    " directory but found at: /bad/location"
                ),
            ),
            (
                Entity.CONFIG,
                Location.EXTRACTED_ZIP_FILE,
                "/incorrect/path",
                (
                    "Configuration should be located in extracted zip file but found"
                    " at: /incorrect/path"
                ),
            ),
        ],
    )
    def test_invalid_location_error_parametrized(
        self, entity: Entity, location: Location, path: str, expected_message: str
    ) -> None:
        """Test InvalidLocationError with different parameters."""
        error = InvalidLocationError(entity, location, path)
        assert str(error) == expected_message

    def test_invalid_location_error_inheritance(self) -> None:
        """Test InvalidLocationError inheritance."""
        error = InvalidLocationError(Entity.FILE, Location.AUDIO_ROOT, "/path")
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestHttpUrlError:
    """Test cases for HttpUrlError exception."""

    def test_http_url_error_basic(self) -> None:
        """Test basic HttpUrlError functionality."""
        error = HttpUrlError("invalid-url")

        assert isinstance(error, OSError)
        assert str(error) == "Invalid HTTP-based URL: invalid-url"

    @pytest.mark.parametrize(
        ("url", "expected_message"),
        [
            ("invalid-url", "Invalid HTTP-based URL: invalid-url"),
            ("ftp://example.com", "Invalid HTTP-based URL: ftp://example.com"),
            ("not-a-url", "Invalid HTTP-based URL: not-a-url"),
            ("", "Invalid HTTP-based URL: "),
        ],
    )
    def test_http_url_error_parametrized(self, url: str, expected_message: str) -> None:
        """Test HttpUrlError with different URLs."""
        error = HttpUrlError(url)
        assert str(error) == expected_message

    def test_http_url_error_inheritance(self) -> None:
        """Test HttpUrlError inheritance."""
        error = HttpUrlError("invalid")
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestYoutubeUrlError:
    """Test cases for YoutubeUrlError exception."""

    def test_youtube_url_error_basic(self) -> None:
        """Test basic YoutubeUrlError functionality."""
        error = YoutubeUrlError("invalid-url", playlist=False)

        assert isinstance(error, OSError)
        assert str(error) == "Not able to access Youtube video at: invalid-url"

    def test_youtube_url_error_with_playlist(self) -> None:
        """Test YoutubeUrlError with playlist option."""
        error = YoutubeUrlError("invalid-url", playlist=True)

        assert (
            str(error) == "Not able to access Youtube video or playlist at: invalid-url"
        )

    @pytest.mark.parametrize(
        ("url", "playlist", "expected_message"),
        [
            ("invalid-url", False, "Not able to access Youtube video at: invalid-url"),
            (
                "invalid-url",
                True,
                "Not able to access Youtube video or playlist at: invalid-url",
            ),
            (
                "http://example.com",
                False,
                "Not able to access Youtube video at: http://example.com",
            ),
            (
                "http://example.com",
                True,
                "Not able to access Youtube video or playlist at: http://example.com",
            ),
        ],
    )
    def test_youtube_url_error_parametrized(
        self, url: str, playlist: bool, expected_message: str
    ) -> None:
        """Test YoutubeUrlError with different parameters."""
        error = YoutubeUrlError(url, playlist=playlist)
        assert str(error) == expected_message

    def test_youtube_url_error_inheritance(self) -> None:
        """Test YoutubeUrlError inheritance."""
        error = YoutubeUrlError("invalid", playlist=False)
        assert isinstance(error, OSError)
        assert isinstance(error, Exception)


class TestUploadLimitError:
    """Test cases for UploadLimitError exception."""

    def test_upload_limit_error_basic(self) -> None:
        """Test basic UploadLimitError functionality."""
        error = UploadLimitError(Entity.FILES, 5)

        assert isinstance(error, ValueError)
        assert str(error) == "At most 5 files can be uploaded."

    def test_upload_limit_error_string_limit(self) -> None:
        """Test UploadLimitError with string limit."""
        error = UploadLimitError(Entity.FILES, "10MB")

        assert str(error) == "At most 10MB files can be uploaded."

    @pytest.mark.parametrize(
        ("entity", "limit", "expected_message"),
        [
            (Entity.FILES, 5, "At most 5 files can be uploaded."),
            (Entity.FILES, 10.5, "At most 10.5 files can be uploaded."),
            (Entity.VOICE_MODEL, "50MB", "At most 50MB voice model can be uploaded."),
            (Entity.DATASET, 1, "At most 1 dataset can be uploaded."),
        ],
    )
    def test_upload_limit_error_parametrized(
        self, entity: Entity, limit: str | float, expected_message: str
    ) -> None:
        """Test UploadLimitError with different parameters."""
        error = UploadLimitError(entity, limit)
        assert str(error) == expected_message

    def test_upload_limit_error_inheritance(self) -> None:
        """Test UploadLimitError inheritance."""
        error = UploadLimitError(Entity.FILES, 5)
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)


class TestUploadTypeError:
    """Test cases for UploadTypeError exception."""

    def test_upload_type_error_single(self) -> None:
        """
        Test basic UploadTypeError functionality for single
        upload.
        """
        error = UploadTypeError(
            Entity.FILES, [".wav", ".mp3"], "formats", multiple=False
        )

        assert isinstance(error, ValueError)
        assert (
            str(error)
            == "Only files with the following formats can be uploaded by themselves:"
            " .wav, .mp3."
        )

    def test_upload_type_error_multiple(self) -> None:
        """Test UploadTypeError for multiple uploads."""
        error = UploadTypeError(
            Entity.FILES, ["model.pth", "config.json"], "names", multiple=True
        )

        assert (
            str(error)
            == "Only files with the following names can be uploaded together (at most"
            " one of each): model.pth, config.json."
        )

    @pytest.mark.parametrize(
        ("entity", "valid_types", "type_class", "multiple", "expected_message"),
        [
            (
                Entity.FILES,
                [".wav", ".mp3"],
                "formats",
                False,
                (
                    "Only files with the following formats can be uploaded by"
                    " themselves: .wav, .mp3."
                ),
            ),
            (
                Entity.FILES,
                ["model.pth", "config.json"],
                "names",
                True,
                (
                    "Only files with the following names can be uploaded together (at"
                    " most one of each): model.pth, config.json."
                ),
            ),
            (
                Entity.VOICE_MODEL,
                [".pth"],
                "formats",
                False,
                (
                    "Only voice model with the following formats can be uploaded by"
                    " themselves: .pth."
                ),
            ),
            (
                Entity.CONFIG,
                ["config.yaml", "config.json"],
                "names",
                True,
                (
                    "Only configuration with the following names can be uploaded"
                    " together (at most one of each): config.yaml, config.json."
                ),
            ),
        ],
    )
    def test_upload_type_error_parametrized(
        self,
        entity: Entity,
        valid_types: list[str],
        type_class: str,
        multiple: bool,
        expected_message: str,
    ) -> None:
        """Test UploadTypeError with different parameters."""
        error = UploadTypeError(entity, valid_types, type_class, multiple=multiple)  # type: ignore[arg-type]
        assert str(error) == expected_message

    def test_upload_type_error_inheritance(self) -> None:
        """Test UploadTypeError inheritance."""
        error = UploadTypeError(Entity.FILES, [".wav"], "formats", multiple=False)
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)


class TestInvalidAudioFormatError:
    """Test cases for InvalidAudioFormatError exception."""

    def test_invalid_audio_format_error_basic(self) -> None:
        """Test basic InvalidAudioFormatError functionality."""
        error = InvalidAudioFormatError("/path/to/file.txt", [".wav", ".mp3"])

        assert isinstance(error, ValueError)
        assert (
            str(error)
            == "Invalid audio file format: /path/to/file.txt. Supported formats are:"
            " .wav, .mp3."
        )

    def test_invalid_audio_format_error_path_object(self) -> None:
        """Test InvalidAudioFormatError with Path object."""
        path = Path("/path/to/file.txt")
        error = InvalidAudioFormatError(path, [".wav", ".mp3"])

        assert (
            str(error)
            == f"Invalid audio file format: {path}. Supported formats are: .wav, .mp3."
        )

    @pytest.mark.parametrize(
        ("path", "formats", "expected_message"),
        [
            (
                "/file.txt",
                [".wav"],
                "Invalid audio file format: /file.txt. Supported formats are: .wav.",
            ),
            (
                "/file.doc",
                [".wav", ".mp3", ".flac"],
                (
                    "Invalid audio file format: /file.doc. Supported formats are: .wav,"
                    " .mp3, .flac."
                ),
            ),
            (
                "file.unknown",
                [".ogg"],
                "Invalid audio file format: file.unknown. Supported formats are: .ogg.",
            ),
        ],
    )
    def test_invalid_audio_format_error_parametrized(
        self, path: str, formats: list[str], expected_message: str
    ) -> None:
        """Test InvalidAudioFormatError with different parameters."""
        error = InvalidAudioFormatError(path, formats)
        assert str(error) == expected_message

    def test_invalid_audio_format_error_inheritance(self) -> None:
        """Test InvalidAudioFormatError inheritance."""
        error = InvalidAudioFormatError("/path", [".wav"])
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)


class TestNotInstantiatedError:
    """Test cases for NotInstantiatedError exception."""

    def test_not_instantiated_error_basic(self) -> None:
        """Test basic NotInstantiatedError functionality."""
        error = NotInstantiatedError(Entity.COMPONENT)

        assert isinstance(error, ValueError)
        assert str(error) == "component has not been instantiated."

    @pytest.mark.parametrize(
        ("entity", "expected_message"),
        [
            (Entity.COMPONENT, "component has not been instantiated."),
            (Entity.EVENT, "event has not been instantiated."),
            (Entity.CONFIG, "configuration has not been instantiated."),
            (Entity.MODEL, "model has not been instantiated."),
        ],
    )
    def test_not_instantiated_error_parametrized(
        self, entity: Entity, expected_message: str
    ) -> None:
        """Test NotInstantiatedError with different entities."""
        error = NotInstantiatedError(entity)
        assert str(error) == expected_message

    def test_not_instantiated_error_inheritance(self) -> None:
        """Test NotInstantiatedError inheritance."""
        error = NotInstantiatedError(Entity.COMPONENT)
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)


class TestComponentNotInstantiatedError:
    """Test cases for ComponentNotInstatiatedError exception."""

    def test_component_not_instantiated_error_basic(self) -> None:
        """Test basic ComponentNotInstatiatedError functionality."""
        error = ComponentNotInstatiatedError()

        assert isinstance(error, NotInstantiatedError)
        assert isinstance(error, ValueError)
        assert str(error) == "component has not been instantiated."

    def test_component_not_instantiated_error_inheritance(self) -> None:
        """Test ComponentNotInstatiatedError inheritance chain."""
        error = ComponentNotInstatiatedError()
        assert isinstance(error, NotInstantiatedError)
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)


class TestEventNotInstantiatedError:
    """Test cases for EventNotInstantiatedError exception."""

    def test_event_not_instantiated_error_basic(self) -> None:
        """Test basic EventNotInstantiatedError functionality."""
        error = EventNotInstantiatedError()

        assert isinstance(error, NotInstantiatedError)
        assert isinstance(error, ValueError)
        assert str(error) == "event has not been instantiated."

    def test_event_not_instantiated_error_inheritance(self) -> None:
        """Test EventNotInstantiatedError inheritance chain."""
        error = EventNotInstantiatedError()
        assert isinstance(error, NotInstantiatedError)
        assert isinstance(error, ValueError)
        assert isinstance(error, Exception)
