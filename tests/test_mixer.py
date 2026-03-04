"""
test_mixer.py — unit-тести для Mixer.

Стратегія:
- Генеруємо реальні WAV-файли через numpy + soundfile
- Перевіряємо розміщення фраз на таймлайні, нормалізацію, ресемплінг
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mixer import Mixer, MixerError
from config import ProjectConfig, MixerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wav(path: Path, duration_sec: float, sample_rate: int = 44100, value: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_sec * sample_rate)
    audio = np.full(n, value, dtype=np.float32)
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
def mixer_config() -> MixerConfig:
    return MixerConfig(output_sample_rate=44100, output_channels=1)


@pytest.fixture
def mixer(tmp_project: ProjectConfig, mixer_config: MixerConfig) -> Mixer:
    return Mixer(tmp_project, mixer_config)


# ---------------------------------------------------------------------------
# mix()
# ---------------------------------------------------------------------------

class TestMix:
    def test_creates_output_file(self, mixer: Mixer, tmp_path: Path, tmp_project: ProjectConfig):
        sr = 44100
        phrase = make_wav(tmp_path / "phrase_0000.wav", duration_sec=2.0, sample_rate=sr)

        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello"},
        ]
        stretched_results = {
            "SPEAKER_00": [{"segment_index": 0, "path": str(phrase), "duration_sec": 2.0}],
        }

        output = mixer.mix(segments, stretched_results)
        assert output.exists()

    def test_output_duration_matches_timeline(self, mixer: Mixer, tmp_path: Path, tmp_project: ProjectConfig):
        """Фінальний файл повинен тривати не менше ніж кінець останнього сегмента."""
        sr = 44100
        phrase = make_wav(tmp_path / "phrase_0000.wav", duration_sec=3.0, sample_rate=sr)

        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello"},
        ]
        stretched_results = {
            "SPEAKER_00": [{"segment_index": 0, "path": str(phrase), "duration_sec": 3.0}],
        }

        output = mixer.mix(segments, stretched_results)
        audio, out_sr = sf.read(str(output))
        duration = len(audio) / out_sr
        assert duration >= 5.0  # +1 сек запасу

    def test_phrase_placed_at_correct_position(self, mixer: Mixer, tmp_path: Path, tmp_project: ProjectConfig):
        """Фраза розміщується з правильним відступом (start)."""
        sr = 44100
        phrase = make_wav(tmp_path / "phrase_0000.wav", duration_sec=1.0, sample_rate=sr, value=1.0)

        start_sec = 3.0
        segments = [
            {"speaker": "SPEAKER_00", "start": start_sec, "end": start_sec + 1.0, "text": "Hi"},
        ]
        stretched_results = {
            "SPEAKER_00": [{"segment_index": 0, "path": str(phrase), "duration_sec": 1.0}],
        }

        output = mixer.mix(segments, stretched_results)
        audio, out_sr = sf.read(str(output))

        start_sample = int(start_sec * out_sr)
        # Перед стартом — тиша (< 0.01)
        silence_part = audio[:start_sample]
        assert np.max(np.abs(silence_part)) < 0.01

        # Після старту — сигнал > 0
        signal_part = audio[start_sample: start_sample + out_sr]
        assert np.max(np.abs(signal_part)) > 0.01

    def test_raises_when_no_stretched_files(self, mixer: Mixer):
        segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hi"}]
        with pytest.raises(MixerError, match="Немає жодного"):
            mixer.mix(segments, {})

    def test_skips_missing_phrase_files(self, mixer: Mixer, tmp_path: Path, tmp_project: ProjectConfig):
        """Якщо файл фрази відсутній — пропускаємо без помилки."""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello"},
        ]
        stretched_results = {
            "SPEAKER_00": [{"segment_index": 0, "path": str(tmp_path / "missing.wav"), "duration_sec": 2.0}],
        }
        # Не кидає, просто пропускає
        output = mixer.mix(segments, stretched_results)
        assert output.exists()

    def test_stereo_output(self, tmp_path: Path, tmp_project: ProjectConfig):
        """Тест для stereo виходу (channels=2)."""
        config = MixerConfig(output_sample_rate=44100, output_channels=2)
        stereo_mixer = Mixer(tmp_project, config)

        sr = 44100
        phrase = make_wav(tmp_path / "phrase_0000.wav", duration_sec=1.0, sample_rate=sr)
        segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "Hi"}]
        stretched_results = {
            "SPEAKER_00": [{"segment_index": 0, "path": str(phrase), "duration_sec": 1.0}],
        }
        output = stereo_mixer.mix(segments, stretched_results)
        audio, _ = sf.read(str(output))
        assert audio.ndim == 2
        assert audio.shape[1] == 2

    def test_multiple_speakers_mixed(self, mixer: Mixer, tmp_path: Path, tmp_project: ProjectConfig):
        """Два спікери на різних таймах — обидва повинні з'явитись у фінальному файлі."""
        sr = 44100
        p1 = make_wav(tmp_path / "p1.wav", duration_sec=1.0, sample_rate=sr, value=0.5)
        p2 = make_wav(tmp_path / "p2.wav", duration_sec=1.0, sample_rate=sr, value=0.8)

        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "Hello"},
            {"speaker": "SPEAKER_01", "start": 2.0, "end": 3.0, "text": "World"},
        ]
        stretched_results = {
            "SPEAKER_00": [{"segment_index": 0, "path": str(p1), "duration_sec": 1.0}],
            "SPEAKER_01": [{"segment_index": 1, "path": str(p2), "duration_sec": 1.0}],
        }
        output = mixer.mix(segments, stretched_results)
        audio, out_sr = sf.read(str(output))
        # Сигнал присутній у позиції обох спікерів
        assert np.max(np.abs(audio[:out_sr])) > 0.01       # SPEAKER_00 (0-1 сек)
        assert np.max(np.abs(audio[out_sr*2:out_sr*3])) > 0.01  # SPEAKER_01 (2-3 сек)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_normalizes_to_headroom(self):
        audio = np.array([0.0, 2.0, -2.0], dtype=np.float32)
        result = Mixer._normalize(audio, headroom_db=-1.0)
        peak = np.max(np.abs(result))
        expected_peak = 10 ** (-1.0 / 20.0)
        assert peak == pytest.approx(expected_peak, rel=0.01)

    def test_silent_audio_unchanged(self):
        audio = np.zeros(100, dtype=np.float32)
        result = Mixer._normalize(audio)
        np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# _resample
# ---------------------------------------------------------------------------

class TestResample:
    def test_upsampling(self):
        audio = np.ones(100, dtype=np.float32)
        result = Mixer._resample(audio, orig_sr=100, target_sr=200)
        assert len(result) == 200

    def test_downsampling(self):
        audio = np.ones(200, dtype=np.float32)
        result = Mixer._resample(audio, orig_sr=200, target_sr=100)
        assert len(result) == 100

    def test_same_sr_unchanged(self):
        audio = np.arange(100, dtype=np.float32)
        result = Mixer._resample(audio, orig_sr=44100, target_sr=44100)
        np.testing.assert_array_equal(result, audio)
