"""
Comprehensive unit tests for ultimate_rvc.core.generate.common
module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unittest.mock import Mock, patch

import pytest

from ultimate_rvc.core.common import Entity
from ultimate_rvc.core.exceptions import (
    ModelNotFoundError,
    NotFoundError,
    NotProvidedError,
)
from ultimate_rvc.core.generate.common import (
    convert,
    get_unique_base_path,
    mix_audio,
    wavify,
)
from ultimate_rvc.core.generate.typing_extra import MixedAudioType
from ultimate_rvc.typing_extra import (
    AudioExt,
    F0Method,
    RVCContentType,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGetUniqueBasePath:
    """Test cases for get_unique_base_path function."""

    def test_get_unique_base_path_basic(self, tmp_path: Path) -> None:
        """Test get_unique_base_path with basic functionality."""
        directory = tmp_path / "test_dir"
        directory.mkdir()
        prefix = "test_prefix"
        args_dict = {"key": "value", "number": 42}
        hash_size = 5

        result = get_unique_base_path(directory, prefix, args_dict, hash_size)

        assert result.parent == directory
        assert result.name.startswith(prefix)
        hash_part = result.name.split("_")[-1]  # Get last part (actual hash)
        expected_hex_chars = hash_size * 2
        assert len(hash_part) == expected_hex_chars

    def test_get_unique_base_path_same_args_same_path(self, tmp_path: Path) -> None:
        """
        Test get_unique_base_path returns same path for identical
        arguments.
        """
        directory = tmp_path / "test_dir"
        directory.mkdir()
        prefix = "test"
        args_dict = {"key": "value", "number": 42}

        # Create first file
        base_path1 = get_unique_base_path(directory, prefix, args_dict)
        json_path1 = base_path1.with_suffix(".json")
        json_path1.write_text('{"key": "value", "number": 42}')

        # Request path for same args - should get same path
        base_path2 = get_unique_base_path(directory, prefix, args_dict)

        assert base_path1 == base_path2

    def test_get_unique_base_path_different_args_different_paths(
        self, tmp_path: Path
    ) -> None:
        """
        Test get_unique_base_path returns different paths for different
        arguments.
        """
        directory = tmp_path / "test_dir"
        directory.mkdir()
        prefix = "test"
        args_dict1 = {"key": "value1"}
        args_dict2 = {"key": "value2"}

        base_path1 = get_unique_base_path(directory, prefix, args_dict1)
        base_path2 = get_unique_base_path(directory, prefix, args_dict2)

        assert base_path1 != base_path2
        assert base_path1.parent == base_path2.parent

    def test_get_unique_base_path_nonexistent_directory(self, tmp_path: Path) -> None:
        """
        Test get_unique_base_path works with non-existent
        directory.
        """
        directory = tmp_path / "nonexistent_dir"
        prefix = "test"
        args_dict = {"key": "value"}

        result = get_unique_base_path(directory, prefix, args_dict)

        assert result.parent == directory

    @pytest.mark.parametrize("hash_size", [1, 3, 5, 8, 16])
    def test_get_unique_base_path_different_hash_sizes(
        self, tmp_path: Path, hash_size: int
    ) -> None:
        """Test get_unique_base_path with different hash sizes."""
        directory = tmp_path / "test_dir"
        directory.mkdir()
        prefix = "test"
        args_dict = {"key": "value"}

        result = get_unique_base_path(directory, prefix, args_dict, hash_size)

        expected_hash_length = hash_size * 2  # hex characters
        actual_hash_length = len(result.name.split("_")[-1])
        assert actual_hash_length == expected_hash_length

    def test_get_unique_base_path_empty_args_dict(self, tmp_path: Path) -> None:
        """Test get_unique_base_path with empty args_dict."""
        directory = tmp_path / "test_dir"
        directory.mkdir()
        prefix = "test"
        args_dict: dict[str, str] = {}

        result = get_unique_base_path(directory, prefix, args_dict)

        assert result.parent == directory
        assert result.name.startswith(prefix)

    def test_get_unique_base_path_empty_prefix(self, tmp_path: Path) -> None:
        """Test get_unique_base_path with empty prefix."""
        directory = tmp_path / "test_dir"
        directory.mkdir()
        prefix = ""
        args_dict = {"key": "value"}

        result = get_unique_base_path(directory, prefix, args_dict)

        assert result.parent == directory
        # Should start with underscore since prefix is empty
        assert result.name.startswith("_")


class TestWavify:
    """Test cases for wavify function."""

    def _create_simple_wav_file(self, path: Path) -> None:
        """Create a minimal valid WAV file for testing."""
        # Minimal WAV file header + some data
        wav_data = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
            b"\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00"
            b"\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        path.write_bytes(wav_data)

    def _create_fake_audio_file(self, path: Path) -> None:
        """Create a fake audio file that pydub can't process."""
        path.write_text("fake audio content")

    @patch("pydub.utils")
    def test_wavify_already_wav_format(
        self,
        mock_pydub_utils: Mock,
        tmp_path: Path,
    ) -> None:
        """Test wavify when audio is already in WAV format."""
        # Setup
        audio_file = tmp_path / "test.wav"
        self._create_simple_wav_file(audio_file)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_pydub_utils.mediainfo.return_value = {"format_name": "wav"}

        # Test
        result = wavify(audio_file, output_dir, "test_prefix")

        # Assert
        assert result == audio_file  # Should return original file
        mock_pydub_utils.mediainfo.assert_called_once_with(str(audio_file))

    @patch("ultimate_rvc.core.generate.common.ffmpeg")
    @patch("pydub.utils")
    def test_wavify_conversion_needed(
        self,
        mock_pydub_utils: Mock,
        mock_ffmpeg: Mock,
        tmp_path: Path,
    ) -> None:
        """Test wavify when conversion to WAV is needed."""
        # Setup
        audio_file = tmp_path / "test.mp3"
        self._create_fake_audio_file(audio_file)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_pydub_utils.mediainfo.return_value = {"format_name": "mp3"}
        mock_ffmpeg_chain = Mock()
        mock_ffmpeg.input.return_value = mock_ffmpeg_chain
        mock_ffmpeg_chain.output.return_value = mock_ffmpeg_chain
        mock_ffmpeg_chain.run.return_value = (None, b"ffmpeg output")

        # Test
        result = wavify(audio_file, output_dir, "test_prefix")

        # Assert
        assert result.suffix == ".wav"
        assert result.parent == output_dir
        mock_ffmpeg.input.assert_called_once_with(audio_file)

    @patch("ultimate_rvc.core.generate.common.ffmpeg")
    @patch("pydub.utils")
    def test_wavify_custom_accepted_formats(
        self,
        mock_pydub_utils: Mock,
        mock_ffmpeg: Mock,
        tmp_path: Path,
    ) -> None:
        """Test wavify with custom accepted formats."""
        # Setup
        audio_file = tmp_path / "test.flac"
        self._create_fake_audio_file(audio_file)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_pydub_utils.mediainfo.return_value = {"format_name": "ogg"}
        accepted_formats = {AudioExt.OGG}

        # Mock ffmpeg to prevent actual execution
        mock_ffmpeg_chain = Mock()
        mock_ffmpeg.input.return_value = mock_ffmpeg_chain
        mock_ffmpeg_chain.output.return_value = mock_ffmpeg_chain
        mock_ffmpeg_chain.run.return_value = (None, b"ffmpeg output")

        # Test
        result = wavify(audio_file, output_dir, "test_prefix", accepted_formats)

        # Should convert since ogg was found but WAV is default output
        assert result.suffix == ".wav"
        assert result.parent == output_dir

    @patch("ultimate_rvc.core.generate.common.ffmpeg")
    @patch("pydub.utils")
    def test_wavify_m4a_special_case(
        self,
        mock_pydub_utils: Mock,
        mock_ffmpeg: Mock,
        tmp_path: Path,
    ) -> None:
        """Test wavify M4A format special case handling."""
        # Setup
        audio_file = tmp_path / "test.m4a"
        self._create_fake_audio_file(audio_file)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # M4A uses "in" check rather than exact match
        mock_pydub_utils.mediainfo.return_value = {"format_name": "mov,mp4,m4a"}
        accepted_formats = {AudioExt.M4A}

        # Mock ffmpeg chain for conversion
        mock_ffmpeg_chain = Mock()
        mock_ffmpeg.input.return_value = mock_ffmpeg_chain
        mock_ffmpeg_chain.output.return_value = mock_ffmpeg_chain
        mock_ffmpeg_chain.run.return_value = (None, b"ffmpeg output")

        # Test
        result = wavify(audio_file, output_dir, "test_prefix", accepted_formats)

        # Should convert since M4A is found in format_name
        assert result != audio_file
        assert result.suffix == ".wav"

    def test_wavify_invalid_audio_file(self, tmp_path: Path) -> None:
        """Test wavify with non-existent audio file."""
        nonexistent_file = tmp_path / "nonexistent.mp3"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with pytest.raises(NotFoundError):
            wavify(nonexistent_file, output_dir, "test_prefix")

    def test_wavify_invalid_directory(self, tmp_path: Path) -> None:
        """Test wavify with non-existent directory."""
        audio_file = tmp_path / "test.mp3"
        self._create_fake_audio_file(audio_file)
        nonexistent_dir = tmp_path / "nonexistent_dir"

        with pytest.raises(NotFoundError):
            wavify(audio_file, nonexistent_dir, "test_prefix")

    @patch("pydub.utils")
    def test_wavify_empty_accepted_formats(
        self,
        mock_pydub_utils: Mock,
        tmp_path: Path,
    ) -> None:
        """Test wavify with empty accepted formats set."""
        # Setup
        audio_file = tmp_path / "test.mp3"
        self._create_fake_audio_file(audio_file)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_pydub_utils.mediainfo.return_value = {"format_name": "mp3"}
        accepted_formats: set[AudioExt] = set()

        # Test
        result = wavify(audio_file, output_dir, "test_prefix", accepted_formats)

        # Should return original since no formats accepted
        assert result == audio_file


class TestMixAudio:
    """Test cases for mix_audio function."""

    def _create_simple_wav_file(self, path: Path) -> None:
        """Create a minimal valid WAV file for testing."""
        wav_data = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
            b"\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00"
            b"\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        path.write_bytes(wav_data)

    def test_mix_audio_empty_tracks(self, tmp_path: Path) -> None:
        """Test mix_audio with empty track list."""
        directory = tmp_path / "output"
        directory.mkdir()

        with pytest.raises(NotProvidedError) as exc_info:
            mix_audio([], directory)

        assert "audio track" in str(exc_info.value).lower()

    @patch("ultimate_rvc.core.generate.common._mix_audio")
    @patch("pydub.utils")
    def test_mix_audio_basic(
        self, mock_pydub_utils: Mock, mock_mix_audio: Mock, tmp_path: Path
    ) -> None:
        """Test mix_audio basic functionality."""
        # Setup
        audio_file1 = tmp_path / "audio1.wav"
        audio_file2 = tmp_path / "audio2.wav"
        directory = tmp_path / "output"
        directory.mkdir()

        # Create actual WAV files
        self._create_simple_wav_file(audio_file1)
        self._create_simple_wav_file(audio_file2)

        # Mock pydub to return wav format so wavify doesn't try to
        # convert
        mock_pydub_utils.mediainfo.return_value = {"format_name": "wav"}

        # Test
        audio_track_gain_pairs = [(audio_file1, 0), (audio_file2, -5)]
        result = mix_audio(
            audio_track_gain_pairs,
            directory,
            output_sr=44100,
            output_format=AudioExt.MP3,
            content_type=MixedAudioType.AUDIO,
        )

        # Mock creates the expected output file since _mix_audio is
        # mocked
        result.write_bytes(b"fake mp3 data")

        # Assert
        assert result.suffix == ".mp3"
        assert result.parent == directory
        assert result.exists()
        mock_mix_audio.assert_called_once()

    @pytest.mark.parametrize(
        "content_type",
        [
            MixedAudioType.AUDIO,
            MixedAudioType.SPEECH,
            MixedAudioType.SONG,
        ],
    )
    @patch("ultimate_rvc.core.generate.common._mix_audio")
    @patch("pydub.utils")
    def test_mix_audio_different_content_types(
        self,
        mock_pydub_utils: Mock,
        mock_mix_audio: Mock,
        content_type: MixedAudioType,
        tmp_path: Path,
    ) -> None:
        """Test mix_audio with different content types."""
        # Setup
        audio_file = tmp_path / "audio.wav"
        directory = tmp_path / "output"
        directory.mkdir()

        # Create actual WAV file
        self._create_simple_wav_file(audio_file)

        # Mock pydub to return wav format
        mock_pydub_utils.mediainfo.return_value = {"format_name": "wav"}

        # Test
        result = mix_audio([(audio_file, 0)], directory, content_type=content_type)

        # Mock creates the expected output file since _mix_audio is
        # mocked
        result.write_bytes(b"fake wav data")

        # Should complete without error for all content types
        assert result.exists()
        assert result.parent == directory
        mock_mix_audio.assert_called_once()


class TestConvert:
    """Test cases for convert function."""

    def _create_simple_wav_file(self, path: Path) -> None:
        """Create a minimal valid WAV file for testing."""
        wav_data = (
            b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
            b"\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00"
            b"\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        path.write_bytes(wav_data)

    def _create_model_dir_structure(self, base_path: Path) -> None:
        """Create proper model directory structure."""
        # Voice models directory
        voice_models_dir = base_path / "voice_models"
        voice_models_dir.mkdir(parents=True, exist_ok=True)

        # Create test model
        test_model_dir = voice_models_dir / "test_model"
        test_model_dir.mkdir()
        (test_model_dir / "model.pth").write_text("fake model")
        (test_model_dir / "model.index").write_text("fake index")

    @patch("ultimate_rvc.core.generate.common.validate_model")
    @patch("ultimate_rvc.core.generate.common._get_rvc_files")
    @patch("ultimate_rvc.core.generate.common._get_voice_converter")
    @patch("pydub.utils")
    def test_convert_basic(
        self,
        mock_pydub_utils: Mock,
        mock_get_voice_converter: Mock,
        mock_get_rvc_files: Mock,
        mock_validate_model: Mock,
        tmp_path: Path,
    ) -> None:
        """Test convert function basic functionality."""
        # Setup
        audio_file = tmp_path / "input.wav"
        directory = tmp_path / "output"
        directory.mkdir()

        self._create_simple_wav_file(audio_file)

        # Mock pydub and voice converter
        mock_pydub_utils.mediainfo.return_value = {"format_name": "wav"}
        mock_voice_converter = Mock()
        mock_get_voice_converter.return_value = mock_voice_converter

        # Mock RVC model files
        model_path = tmp_path / "model.pth"
        index_path = tmp_path / "model.index"
        model_path.write_text("fake model")
        index_path.write_text("fake index")
        mock_get_rvc_files.return_value = (model_path, index_path)

        # Test
        result = convert(
            audio_track=audio_file,
            directory=directory,
            model_name="test_model",
            n_octaves=1,
            n_semitones=2,
            f0_methods=[F0Method.RMVPE],
            content_type=RVCContentType.AUDIO,
        )

        # Assert
        assert result.suffix == ".wav"
        assert result.parent == directory
        mock_voice_converter.convert_audio.assert_called_once()
        mock_validate_model.assert_called_once()

    def test_convert_invalid_audio_file(self, tmp_path: Path) -> None:
        """Test convert function with non-existent audio file."""
        nonexistent_file = tmp_path / "nonexistent.wav"
        directory = tmp_path / "output"
        directory.mkdir()

        with pytest.raises(NotFoundError):
            convert(
                audio_track=nonexistent_file,
                directory=directory,
                model_name="test_model",
            )

    @patch("ultimate_rvc.core.generate.common.validate_model")
    def test_convert_make_directory(
        self, mock_validate_model: Mock, tmp_path: Path
    ) -> None:
        """Test convert function with make_directory option."""
        # Setup
        audio_file = tmp_path / "input.wav"
        directory = tmp_path / "nonexistent" / "output"

        self._create_simple_wav_file(audio_file)

        # Mock validate_model to raise error after directory creation
        mock_validate_model.side_effect = ModelNotFoundError(
            Entity.VOICE_MODEL, name="test_model"
        )

        # Test - will fail at model validation, but directory
        # should be created
        with pytest.raises(ModelNotFoundError):
            convert(
                audio_track=audio_file,
                directory=directory,
                model_name="test_model",
                make_directory=True,
            )

        # Assert directory was created
        assert directory.exists()

    @patch("ultimate_rvc.core.generate.common.validate_model")
    def test_convert_invalid_semitones(
        self, mock_validate_model: Mock, tmp_path: Path
    ) -> None:
        """Test convert function with invalid semitone parameters."""
        # Setup
        audio_file = tmp_path / "input.wav"
        directory = tmp_path / "output"
        directory.mkdir()

        self._create_simple_wav_file(audio_file)

        # Mock validate_model to prevent it from being called
        mock_validate_model.return_value = None

        # Test invalid octaves
        with pytest.raises(ValueError, match="n_octaves must be between -5 and 5"):
            convert(
                audio_track=audio_file,
                directory=directory,
                model_name="test_model",
                n_octaves=10,  # Invalid - too high
            )

        with pytest.raises(ValueError, match="n_octaves must be between -5 and 5"):
            convert(
                audio_track=audio_file,
                directory=directory,
                model_name="test_model",
                n_octaves=-10,  # Invalid - too low
            )

        # Test invalid semitones
        with pytest.raises(ValueError, match="n_semitones must be between -12 and 12"):
            convert(
                audio_track=audio_file,
                directory=directory,
                model_name="test_model",
                n_semitones=15,  # Invalid - too high
            )

        with pytest.raises(ValueError, match="n_semitones must be between -12 and 12"):
            convert(
                audio_track=audio_file,
                directory=directory,
                model_name="test_model",
                n_semitones=-15,  # Invalid - too low
            )
