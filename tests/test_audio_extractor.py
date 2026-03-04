"""
test_audio_extractor.py — unit-тести для AudioExtractor.

Стратегія:
- subprocess.run замокований (не викликаємо ffmpeg/ffprobe реально)
- перевіряємо побудову команди, обробку помилок, перевірку файлів
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

# Додаємо src/ до PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_extractor import AudioExtractor, AudioExtractorError
from config import ProjectConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> ProjectConfig:
    cfg = ProjectConfig(project_dir=tmp_path / "project")
    cfg.create_dirs()
    return cfg


@pytest.fixture
def extractor(tmp_project: ProjectConfig) -> AudioExtractor:
    return AudioExtractor(tmp_project)


@pytest.fixture
def fake_input(tmp_path: Path) -> Path:
    """Створює порожній відеофайл для тестів."""
    p = tmp_path / "video.mp4"
    p.write_bytes(b"\x00" * 16)
    return p


# ---------------------------------------------------------------------------
# _build_ffmpeg_cmd
# ---------------------------------------------------------------------------

class TestBuildFfmpegCmd:
    def test_correct_args_order(self, extractor: AudioExtractor, fake_input: Path, tmp_project: ProjectConfig):
        cmd = extractor._build_ffmpeg_cmd(fake_input, tmp_project.clean_audio_path)
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-i" in cmd
        assert str(fake_input) in cmd
        assert "-vn" in cmd
        assert "-acodec" in cmd
        assert "pcm_s16le" in cmd
        assert "-ar" in cmd
        assert "16000" in cmd
        assert "-ac" in cmd
        assert "1" in cmd

    def test_output_path_is_last(self, extractor: AudioExtractor, fake_input: Path, tmp_project: ProjectConfig):
        cmd = extractor._build_ffmpeg_cmd(fake_input, tmp_project.clean_audio_path)
        assert cmd[-1] == str(tmp_project.clean_audio_path)


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------

class TestExtract:
    def test_raises_file_not_found_for_missing_input(self, extractor: AudioExtractor, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="не знайдено"):
            extractor.extract(tmp_path / "nonexistent.mp4")

    @patch("audio_extractor.subprocess.run")
    def test_successful_extraction(
        self, mock_run: MagicMock, extractor: AudioExtractor, fake_input: Path, tmp_project: ProjectConfig
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # ffprobe теж мокується — для get_duration
        output_path = tmp_project.clean_audio_path
        output_path.write_bytes(b"\x00" * 32)  # симулюємо що файл створено

        with patch.object(extractor, "_get_duration", return_value=5.0):
            result = extractor.extract(fake_input)

        assert result == output_path
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args[0]

    @patch("audio_extractor.subprocess.run")
    def test_raises_on_nonzero_returncode(
        self, mock_run: MagicMock, extractor: AudioExtractor, fake_input: Path
    ):
        mock_run.return_value = MagicMock(returncode=1, stderr="some error")
        with pytest.raises(AudioExtractorError, match="завершився з кодом"):
            extractor.extract(fake_input)

    @patch("audio_extractor.subprocess.run")
    def test_raises_when_ffmpeg_not_found(
        self, mock_run: MagicMock, extractor: AudioExtractor, fake_input: Path
    ):
        mock_run.side_effect = FileNotFoundError
        with pytest.raises(AudioExtractorError, match="ffmpeg не знайдено"):
            extractor.extract(fake_input)

    @patch("audio_extractor.subprocess.run")
    def test_raises_when_output_not_created(
        self, mock_run: MagicMock, extractor: AudioExtractor, fake_input: Path
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # Файл НЕ створюємо — перевіряємо що кидається помилка
        with pytest.raises(AudioExtractorError, match="файл не створено"):
            extractor.extract(fake_input)


# ---------------------------------------------------------------------------
# _get_duration()
# ---------------------------------------------------------------------------

class TestGetDuration:
    @patch("audio_extractor.subprocess.run")
    def test_returns_float_on_success(self, mock_run: MagicMock, extractor: AudioExtractor, tmp_path: Path):
        fake_wav = tmp_path / "test.wav"
        fake_wav.write_bytes(b"\x00")
        mock_run.return_value = MagicMock(returncode=0, stdout="12.345\n")
        result = extractor._get_duration(fake_wav)
        assert result == pytest.approx(12.345)

    @patch("audio_extractor.subprocess.run")
    def test_returns_zero_on_ffprobe_error(self, mock_run: MagicMock, extractor: AudioExtractor, tmp_path: Path):
        fake_wav = tmp_path / "test.wav"
        fake_wav.write_bytes(b"\x00")
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = extractor._get_duration(fake_wav)
        assert result == 0.0

    @patch("audio_extractor.subprocess.run")
    def test_returns_zero_when_ffprobe_missing(self, mock_run: MagicMock, extractor: AudioExtractor, tmp_path: Path):
        fake_wav = tmp_path / "test.wav"
        fake_wav.write_bytes(b"\x00")
        mock_run.side_effect = FileNotFoundError
        result = extractor._get_duration(fake_wav)
        assert result == 0.0
