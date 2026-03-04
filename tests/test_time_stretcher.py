"""
test_time_stretcher.py — unit-тести для TimeStretcher.

Стратегія:
- Генеруємо реальні WAV-файли через numpy + soundfile (без GPU)
- pyrubberband замокований, щоб не потребував rubberband binary
- Перевіряємо логіку clamping, padding, trimming
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from time_stretcher import TimeStretcher, TimeStretcherError
from config import ProjectConfig, TimeStretchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wav(path: Path, duration_sec: float, sample_rate: int = 22050) -> Path:
    """Генерує синусоїдальний WAV-файл заданої тривалості."""
    import soundfile as sf
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    sf.write(str(path), audio, sample_rate)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> ProjectConfig:
    cfg = ProjectConfig(project_dir=tmp_path / "project")
    cfg.create_dirs()
    return cfg


@pytest.fixture
def stretch_config() -> TimeStretchConfig:
    return TimeStretchConfig(min_rate=0.5, max_rate=2.0, clamp_strategy="clamp")


@pytest.fixture
def stretcher(tmp_project: ProjectConfig, stretch_config: TimeStretchConfig) -> TimeStretcher:
    return TimeStretcher(tmp_project, stretch_config)


# ---------------------------------------------------------------------------
# stretch_file — нормальний випадок
# ---------------------------------------------------------------------------

class TestStretchFile:
    def test_stretch_to_longer_duration(self, stretcher: TimeStretcher, tmp_path: Path):
        """Розтягуємо 1 сек аудіо до 2 сек."""
        import soundfile as sf
        from unittest.mock import patch

        input_wav = make_wav(tmp_path / "input.wav", duration_sec=1.0)
        output_wav = tmp_path / "output.wav"

        audio_data, sr = sf.read(str(input_wav))

        # Мокуємо pyrubberband.time_stretch — повертає масив потрібної довжини
        stretched_audio = np.zeros(int(2.0 * sr), dtype=np.float32)

        with patch("time_stretcher.pyrb.time_stretch", return_value=stretched_audio):
            result_duration = stretcher.stretch_file(input_wav, output_wav, target_duration_sec=2.0)

        assert output_wav.exists()
        assert result_duration == pytest.approx(2.0, abs=0.05)

    def test_stretch_to_shorter_duration(self, stretcher: TimeStretcher, tmp_path: Path):
        """Стискаємо 2 сек аудіо до 1 сек."""
        import soundfile as sf
        from unittest.mock import patch

        input_wav = make_wav(tmp_path / "input.wav", duration_sec=2.0)
        output_wav = tmp_path / "output.wav"

        _, sr = sf.read(str(input_wav))
        stretched_audio = np.zeros(int(1.0 * sr), dtype=np.float32)

        with patch("time_stretcher.pyrb.time_stretch", return_value=stretched_audio):
            result_duration = stretcher.stretch_file(input_wav, output_wav, target_duration_sec=1.0)

        assert result_duration == pytest.approx(1.0, abs=0.05)

    def test_raises_if_input_missing(self, stretcher: TimeStretcher, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            stretcher.stretch_file(
                tmp_path / "nonexistent.wav",
                tmp_path / "output.wav",
                target_duration_sec=1.0,
            )

    def test_stereo_converted_to_mono(self, stretcher: TimeStretcher, tmp_path: Path):
        """Стерео-файл повинен бути сконвертований у моно перед обробкою."""
        import soundfile as sf
        from unittest.mock import patch

        sr = 22050
        n = sr  # 1 секунда
        stereo = np.stack([
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        ], axis=1)
        input_wav = tmp_path / "stereo.wav"
        sf.write(str(input_wav), stereo, sr)
        output_wav = tmp_path / "output.wav"

        stretched = np.zeros(sr, dtype=np.float32)
        with patch("time_stretcher.pyrb.time_stretch", return_value=stretched) as mock_ts:
            stretcher.stretch_file(input_wav, output_wav, target_duration_sec=1.0)
            # Переданий аудіо-масив повинен бути 1D
            call_audio = mock_ts.call_args[0][0]
            assert call_audio.ndim == 1

    def test_nearly_empty_file_skips_stretch(self, stretcher: TimeStretcher, tmp_path: Path):
        """Майже порожній файл (< 10 мс) не повинен проходити через pyrubberband."""
        import soundfile as sf
        from unittest.mock import patch

        sr = 22050
        tiny = np.zeros(50, dtype=np.float32)  # ~2 мс
        input_wav = tmp_path / "tiny.wav"
        sf.write(str(input_wav), tiny, sr)
        output_wav = tmp_path / "output.wav"

        with patch("time_stretcher.pyrb.time_stretch") as mock_ts:
            stretcher.stretch_file(input_wav, output_wav, target_duration_sec=1.0)
            mock_ts.assert_not_called()


# ---------------------------------------------------------------------------
# _fit_to_length
# ---------------------------------------------------------------------------

class TestFitToLength:
    def test_trims_longer_audio(self, stretcher: TimeStretcher):
        audio = np.ones(100, dtype=np.float32)
        result = stretcher._fit_to_length(audio, target_samples=60, sr=22050)
        assert len(result) == 60

    def test_pads_shorter_audio(self, stretcher: TimeStretcher):
        audio = np.ones(40, dtype=np.float32)
        result = stretcher._fit_to_length(audio, target_samples=100, sr=22050)
        assert len(result) == 100
        # Доданий хвіст — тиша
        assert np.all(result[40:] == 0.0)

    def test_exact_length_unchanged(self, stretcher: TimeStretcher):
        audio = np.ones(100, dtype=np.float32)
        result = stretcher._fit_to_length(audio, target_samples=100, sr=22050)
        assert len(result) == 100
        np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# stretch_all
# ---------------------------------------------------------------------------

class TestStretchAll:
    def test_processes_multiple_speakers(self, stretcher: TimeStretcher, tmp_path: Path):
        import soundfile as sf
        from unittest.mock import patch

        sr = 22050
        seg_duration = 2.0

        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": seg_duration, "text": "Hello"},
            {"speaker": "SPEAKER_01", "start": 3.0, "end": 3.0 + seg_duration, "text": "Hi"},
        ]

        # Створюємо TTS файли
        tts_dir_00 = tmp_path / "tts" / "SPEAKER_00"
        tts_dir_01 = tmp_path / "tts" / "SPEAKER_01"
        tts_dir_00.mkdir(parents=True)
        tts_dir_01.mkdir(parents=True)

        file_00 = tts_dir_00 / "phrase_0000.wav"
        file_01 = tts_dir_01 / "phrase_0001.wav"
        make_wav(file_00, duration_sec=1.5, sample_rate=sr)
        make_wav(file_01, duration_sec=2.5, sample_rate=sr)

        tts_results = {
            "SPEAKER_00": [{"segment_index": 0, "path": str(file_00), "duration_sec": 1.5}],
            "SPEAKER_01": [{"segment_index": 1, "path": str(file_01), "duration_sec": 2.5}],
        }

        stretched_audio = np.zeros(int(seg_duration * sr), dtype=np.float32)
        with patch("time_stretcher.pyrb.time_stretch", return_value=stretched_audio):
            results = stretcher.stretch_all(segments, tts_results)

        assert "SPEAKER_00" in results
        assert "SPEAKER_01" in results
        assert results["SPEAKER_00"][0]["target_duration_sec"] == seg_duration
        assert results["SPEAKER_01"][0]["target_duration_sec"] == seg_duration
