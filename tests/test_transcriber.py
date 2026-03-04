"""
test_transcriber.py — unit-тести для Transcriber.

Стратегія:
- whisperx імпорт замокований через sys.modules
- перевіряємо _aggregate_segments, _save, load_saved
- перевіряємо очищення VRAM після транскрипції
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transcriber import Transcriber, TranscriberError
from config import ProjectConfig, WhisperConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> ProjectConfig:
    cfg = ProjectConfig(project_dir=tmp_path / "project")
    cfg.create_dirs()
    return cfg


@pytest.fixture
def whisper_config() -> WhisperConfig:
    return WhisperConfig(
        model_name="large-v3",
        language="en",
        device="cpu",
        compute_type="float32",
        batch_size=4,
        hf_token="fake-token",
    )


@pytest.fixture
def transcriber(tmp_project: ProjectConfig, whisper_config: WhisperConfig) -> Transcriber:
    return Transcriber(tmp_project, whisper_config)


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "clean.wav"
    p.write_bytes(b"\x00" * 64)
    return p


# ---------------------------------------------------------------------------
# _aggregate_segments
# ---------------------------------------------------------------------------

class TestAggregateSegments:
    def test_basic_aggregation(self, transcriber: Transcriber):
        raw = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello"},
            {"speaker": "SPEAKER_00", "start": 2.1, "end": 4.0, "text": "world"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 7.0, "text": "Hi there"},
        ]
        result = transcriber._aggregate_segments(raw)
        # Два сегменти SPEAKER_00 об'єднуються (пауза < 0.5 сек)
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert "Hello world" in result[0]["text"]
        assert result[1]["speaker"] == "SPEAKER_01"

    def test_does_not_merge_different_speakers(self, transcriber: Transcriber):
        raw = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello"},
            {"speaker": "SPEAKER_01", "start": 2.1, "end": 4.0, "text": "world"},
        ]
        result = transcriber._aggregate_segments(raw)
        assert len(result) == 2

    def test_does_not_merge_with_large_gap(self, transcriber: Transcriber):
        raw = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello"},
            {"speaker": "SPEAKER_00", "start": 5.0, "end": 7.0, "text": "world"},
        ]
        result = transcriber._aggregate_segments(raw)
        assert len(result) == 2

    def test_skips_empty_text(self, transcriber: Transcriber):
        raw = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": ""},
            {"speaker": "SPEAKER_00", "start": 3.0, "end": 5.0, "text": "Hello"},
        ]
        result = transcriber._aggregate_segments(raw)
        assert len(result) == 1

    def test_fallback_speaker(self, transcriber: Transcriber):
        raw = [{"start": 0.0, "end": 2.0, "text": "Test"}]  # без speaker
        result = transcriber._aggregate_segments(raw)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_sorted_by_start(self, transcriber: Transcriber):
        raw = [
            {"speaker": "SPEAKER_00", "start": 5.0, "end": 7.0, "text": "Second"},
            {"speaker": "SPEAKER_01", "start": 0.0, "end": 2.0, "text": "First"},
        ]
        result = transcriber._aggregate_segments(raw)
        assert result[0]["start"] < result[1]["start"]


# ---------------------------------------------------------------------------
# _save / load_saved
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_json(self, transcriber: Transcriber, tmp_project: ProjectConfig):
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5, "text": "Hello"},
        ]
        transcriber._save(segments)
        assert tmp_project.transcription_path.exists()

        with open(tmp_project.transcription_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == segments

    def test_load_saved_returns_segments(self, transcriber: Transcriber, tmp_project: ProjectConfig):
        segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5, "text": "Test"}]
        transcriber._save(segments)
        result = transcriber.load_saved()
        assert result == segments

    def test_load_saved_raises_if_missing(self, transcriber: Transcriber):
        with pytest.raises(FileNotFoundError):
            transcriber.load_saved()


# ---------------------------------------------------------------------------
# transcribe() — з мокованим whisperx
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_raises_file_not_found(self, transcriber: Transcriber, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(tmp_path / "nonexistent.wav")

    def test_full_pipeline_with_mock_whisperx(
        self, transcriber: Transcriber, fake_audio: Path, tmp_project: ProjectConfig
    ):
        """Мокуємо весь whisperx і перевіряємо що transcribe() повертає коректний список."""
        fake_segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.0, "text": "Hello"},
            {"speaker": "SPEAKER_01", "start": 4.0, "end": 6.0, "text": "World"},
        ]

        mock_whisperx = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "language": "en",
            "segments": fake_segments,
        }
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = MagicMock()
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
        mock_whisperx.align.return_value = {"segments": fake_segments}

        mock_diarize = MagicMock()
        mock_diarize.return_value = MagicMock()
        mock_whisperx.DiarizationPipeline.return_value = mock_diarize
        mock_whisperx.assign_word_speakers.return_value = {"segments": fake_segments}

        with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
            with patch("torch.cuda.is_available", return_value=False):
                result = transcriber.transcribe(fake_audio)

        assert isinstance(result, list)
        assert len(result) > 0
        assert tmp_project.transcription_path.exists()

    def test_raises_transcriber_error_on_whisperx_exception(
        self, transcriber: Transcriber, fake_audio: Path
    ):
        mock_whisperx = MagicMock()
        mock_whisperx.load_model.side_effect = RuntimeError("CUDA error")

        with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
            with patch("torch.cuda.is_available", return_value=False):
                with pytest.raises(TranscriberError, match="Помилка WhisperX"):
                    transcriber.transcribe(fake_audio)

    def test_clears_vram_even_on_error(
        self, transcriber: Transcriber, fake_audio: Path
    ):
        mock_whisperx = MagicMock()
        mock_whisperx.load_model.side_effect = RuntimeError("CUDA error")

        with patch.dict(sys.modules, {"whisperx": mock_whisperx}):
            with patch("torch.cuda.is_available", return_value=True) as mock_cuda:
                with patch("torch.cuda.empty_cache") as mock_empty:
                    with pytest.raises(TranscriberError):
                        transcriber.transcribe(fake_audio)
                    mock_empty.assert_called()
