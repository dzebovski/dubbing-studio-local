"""
mixer.py — зведення всіх аудіо-шматків у фінальну доріжку.

Вхід:  - список реплік [{speaker, start, end, ...}]
        - stretched_results {speaker_id: [{segment_index, path, duration_sec}]}
Вихід: - project_01/output/final.wav

Алгоритм:
  1. Визначаємо загальну тривалість (кінець останнього сегмента)
  2. Створюємо порожню тишу потрібної довжини
  3. Для кожного сегмента вставляємо stretched WAV у відповідну позицію таймлайну
  4. Записуємо фінальний WAV
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore

from config import ProjectConfig, MixerConfig

logger = logging.getLogger(__name__)

Segment = dict[str, Any]


class MixerError(Exception):
    """Базова помилка модуля зведення."""


class Mixer:
    """
    Збирає всі підігнані аудіо-фрагменти на таймлайн оригінальних таймінгів.

    Кожна фраза розміщується точно за координатами (start, end) з WhisperX.
    Проміжки між репліками заповнюються тишею.
    Якщо фрази перекриваються — вони мікшуються (сумуються з clip-захистом).
    """

    def __init__(self, project: ProjectConfig, config: MixerConfig):
        self.project = project
        self.config = config

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def mix(
        self,
        segments: list[Segment],
        stretched_results: dict[str, list[dict]],
        on_progress: Any = None,  # callable(current, total) | None
    ) -> Path:
        """
        Зводить всі фрагменти у фінальний WAV.

        Args:
            segments:         список реплік [{speaker, start, end, ...}].
            stretched_results: {speaker_id: [{segment_index, path, duration_sec}]}.
            on_progress:      callback прогресу.

        Returns:
            Path до збереженого final.wav.

        Raises:
            MixerError: при будь-якій помилці зведення.
        """
        sr = self.config.output_sample_rate

        # Будуємо lookup: segment_index → stretched file info
        phrase_by_idx: dict[int, dict] = {}
        for phrase_list in stretched_results.values():
            for item in phrase_list:
                phrase_by_idx[item["segment_index"]] = item

        if not phrase_by_idx:
            raise MixerError("Немає жодного stretched файлу для зведення.")

        # Визначаємо загальну тривалість таймлайну
        total_duration = max(seg["end"] for seg in segments)
        total_samples = int(total_duration * sr) + sr  # +1 сек запасу
        timeline = np.zeros(total_samples, dtype=np.float32)

        logger.info(
            "Зведення: %.2f сек таймлайн, %d фраз...",
            total_duration, len(phrase_by_idx),
        )

        placed = 0
        skipped = 0

        for seg_idx, seg in enumerate(segments):
            item = phrase_by_idx.get(seg_idx)
            if item is None:
                logger.warning("Фраза для сегмента %d відсутня, заповнюємо тишею.", seg_idx)
                skipped += 1
                continue

            phrase_path = Path(item["path"])
            if not phrase_path.exists():
                logger.warning("Файл не знайдено: %s, пропускаємо.", phrase_path)
                skipped += 1
                continue

            try:
                audio, file_sr = sf.read(str(phrase_path), dtype="float32")
            except Exception as e:
                raise MixerError(f"Не вдалось прочитати {phrase_path}: {e}") from e

            # Конвертуємо моно якщо потрібно
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # Ресемплінг якщо sample rate не збігається
            if file_sr != sr:
                audio = self._resample(audio, file_sr, sr)

            # Визначаємо позицію на таймлайні
            start_sample = int(seg["start"] * sr)
            end_sample = start_sample + len(audio)

            # Якщо фраза виходить за межі таймлайну — розширюємо
            if end_sample > len(timeline):
                extra = np.zeros(end_sample - len(timeline), dtype=np.float32)
                timeline = np.concatenate([timeline, extra])

            # Мікшуємо (додаємо до таймлайну)
            timeline[start_sample:end_sample] += audio
            placed += 1

            if on_progress:
                on_progress(placed + skipped, len(segments))

        logger.info("Розміщено %d фраз, пропущено %d.", placed, skipped)

        # Нормалізація щоб уникнути clipping
        timeline = self._normalize(timeline)

        # Зберігаємо
        output_path = self.project.final_output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        channels = self.config.output_channels
        if channels == 2:
            timeline = np.stack([timeline, timeline], axis=1)

        try:
            sf.write(str(output_path), timeline, sr)
        except Exception as e:
            raise MixerError(f"Не вдалось записати фінальний файл: {e}") from e

        actual_duration = len(timeline if timeline.ndim == 1 else timeline[:, 0]) / sr
        logger.info("Фінальний файл збережено: %s (%.2f сек)", output_path, actual_duration)

        return output_path

    # ------------------------------------------------------------------
    # Приватні методи
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Простий ресемплінг через numpy (лінійна інтерполяція).
        Для продакшн-якості рекомендується librosa.resample().
        """
        if orig_sr == target_sr:
            return audio

        orig_len = len(audio)
        target_len = int(orig_len * target_sr / orig_sr)
        indices = np.linspace(0, orig_len - 1, target_len)
        return np.interp(indices, np.arange(orig_len), audio).astype(np.float32)

    @staticmethod
    def _normalize(audio: np.ndarray, headroom_db: float = -1.0) -> np.ndarray:
        """
        Нормалізує гучність до headroom_db від піку.
        Запобігає clipping при мікшуванні перекритих фраз.
        """
        peak = np.max(np.abs(audio))
        if peak < 1e-6:
            return audio
        target_peak = 10 ** (headroom_db / 20.0)
        return (audio / peak * target_peak).astype(np.float32)
