"""
audio_extractor.py — витягування чистого аудіо з відео/аудіо файлу через ffmpeg.

Вхід:  будь-який відео або аудіо файл (mp4, mkv, avi, mp3, m4a тощо)
Вихід: project_01/audio/clean.wav (16-bit PCM, 16kHz моно — оптимально для WhisperX)
"""

import subprocess
import logging
from pathlib import Path

from config import ProjectConfig

logger = logging.getLogger(__name__)


class AudioExtractorError(Exception):
    """Базова помилка модуля витягування аудіо."""


class AudioExtractor:
    """
    Витягує чисте аудіо з вхідного файлу за допомогою ffmpeg.

    Параметри виходу зафіксовані під WhisperX:
      - sample rate: 16000 Hz
      - channels:    1 (моно)
      - codec:       pcm_s16le (WAV 16-bit)
    """

    OUTPUT_SAMPLE_RATE = 16000
    OUTPUT_CHANNELS = 1
    OUTPUT_CODEC = "pcm_s16le"

    def __init__(self, project: ProjectConfig):
        self.project = project

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def extract(self, input_path: str | Path) -> Path:
        """
        Витягує аудіо з input_path та зберігає у project.clean_audio_path.

        Args:
            input_path: шлях до відео або аудіо файлу.

        Returns:
            Path до збереженого WAV-файлу.

        Raises:
            FileNotFoundError: якщо input_path не існує.
            AudioExtractorError: якщо ffmpeg завершився з помилкою.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Вхідний файл не знайдено: {input_path}")

        output_path = self.project.clean_audio_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Витягування аудіо: %s → %s", input_path, output_path)

        cmd = self._build_ffmpeg_cmd(input_path, output_path)
        self._run(cmd)

        if not output_path.exists():
            raise AudioExtractorError(
                f"ffmpeg завершився без помилки, але файл не створено: {output_path}"
            )

        duration = self._get_duration(output_path)
        logger.info("Готово. Тривалість аудіо: %.2f сек", duration)

        return output_path

    def get_duration(self, audio_path: str | Path) -> float:
        """Повертає тривалість аудіо файлу у секундах."""
        return self._get_duration(Path(audio_path))

    # ------------------------------------------------------------------
    # Приватні методи
    # ------------------------------------------------------------------

    def _build_ffmpeg_cmd(self, input_path: Path, output_path: Path) -> list[str]:
        """Будує список аргументів для ffmpeg."""
        return [
            "ffmpeg",
            "-y",                          # перезаписати якщо існує
            "-i", str(input_path),
            "-vn",                         # без відеодоріжки
            "-acodec", self.OUTPUT_CODEC,
            "-ar", str(self.OUTPUT_SAMPLE_RATE),
            "-ac", str(self.OUTPUT_CHANNELS),
            str(output_path),
        ]

    def _run(self, cmd: list[str]) -> None:
        """Запускає команду ffmpeg та перехоплює помилки."""
        logger.debug("Виконання: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            raise AudioExtractorError(
                "ffmpeg не знайдено. Переконайтесь що ffmpeg встановлений та доступний у PATH."
            )

        if result.returncode != 0:
            raise AudioExtractorError(
                f"ffmpeg завершився з кодом {result.returncode}.\n"
                f"stderr: {result.stderr}"
            )

    def _get_duration(self, audio_path: Path) -> float:
        """Використовує ffprobe для отримання тривалості файлу."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (FileNotFoundError, ValueError):
            pass
        return 0.0
