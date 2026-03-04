"""
time_stretcher.py — підгонка тривалості TTS-фрази під оригінальний тайминг.

Вхід:  - згенеровані WAV-файли (з tts_engine.py)
        - цільова тривалість кожного сегмента (з WhisperX: end - start)
Вихід: - project_01/stretched/SPEAKER_XX/phrase_XXXX.wav

Алгоритм:
  stretch_rate = original_duration / generated_duration
  - stretch_rate < 1.0 → пришвидшуємо (стискаємо)
  - stretch_rate > 1.0 → сповільнюємо (розтягуємо)
  - stretch_rate виходить за межі [0.5, 2.0] → clamp + pad тишею або crop

pyrubberband: high-quality time-stretching без зміни pitch (не як atempo у ffmpeg).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore
import pyrubberband as pyrb  # type: ignore

from config import ProjectConfig, TimeStretchConfig

logger = logging.getLogger(__name__)

Segment = dict[str, Any]


class TimeStretcherError(Exception):
    """Базова помилка модуля time-stretching."""


class TimeStretcher:
    """
    Підганяє тривалість кожного TTS-файлу під оригінальний тайминг сегмента.

    Використовує pyrubberband (обгортка над Rubber Band Library).
    """

    def __init__(self, project: ProjectConfig, config: TimeStretchConfig):
        self.project = project
        self.config = config

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def stretch_all(
        self,
        segments: list[Segment],
        tts_results: dict[str, list[dict]],
        on_progress: Any = None,  # callable(current, total) | None
    ) -> dict[str, list[dict]]:
        """
        Обробляє всі TTS-файли для всіх спікерів.

        Args:
            segments:    список реплік [{speaker, start, end, ...}].
            tts_results: результати TTS {speaker_id: [{segment_index, path, duration_sec}]}.
            on_progress: callback прогресу.

        Returns:
            Аналогічний до tts_results словник, але з оновленими path (після stretch).
        """
        # Будуємо швидкий lookup: segment_index → segment
        seg_by_idx = {i: seg for i, seg in enumerate(segments)}

        total = sum(len(v) for v in tts_results.values())
        done = 0
        stretched_results: dict[str, list[dict]] = {}

        for speaker_id, phrase_list in tts_results.items():
            speaker_stretched = []

            for item in phrase_list:
                seg_idx = item["segment_index"]
                tts_path = Path(item["path"])
                seg = seg_by_idx.get(seg_idx)

                if seg is None:
                    logger.warning("Сегмент %d не знайдено, пропускаємо.", seg_idx)
                    speaker_stretched.append(item)
                    continue

                target_duration = seg["end"] - seg["start"]
                out_path = self.project.stretched_dir_for(speaker_id) / tts_path.name

                stretched_duration = self.stretch_file(
                    input_path=tts_path,
                    output_path=out_path,
                    target_duration_sec=target_duration,
                )

                speaker_stretched.append({
                    "segment_index": seg_idx,
                    "path": str(out_path),
                    "duration_sec": stretched_duration,
                    "target_duration_sec": target_duration,
                })

                done += 1
                if on_progress:
                    on_progress(done, total)

            stretched_results[speaker_id] = speaker_stretched

        logger.info("Time-stretching завершено. Оброблено %d файлів.", done)
        return stretched_results

    def stretch_file(
        self,
        input_path: Path,
        output_path: Path,
        target_duration_sec: float,
    ) -> float:
        """
        Підганяє один WAV-файл під задану тривалість.

        Args:
            input_path:         вхідний WAV (TTS-вихід).
            output_path:        куди зберегти результат.
            target_duration_sec: бажана тривалість у секундах.

        Returns:
            Фактична тривалість збереженого файлу (секунди).

        Raises:
            FileNotFoundError: якщо input_path не існує.
            TimeStretcherError: при помилці обробки.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"TTS файл не знайдено: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            audio, sr = sf.read(str(input_path), dtype="float32")
        except Exception as e:
            raise TimeStretcherError(f"Не вдалось прочитати {input_path}: {e}") from e

        # Конвертуємо стерео → моно якщо потрібно
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        current_duration = len(audio) / sr

        if current_duration < 0.01:
            logger.warning("Файл %s майже порожній (%.3f сек), пропускаємо stretch.", input_path, current_duration)
            sf.write(str(output_path), audio, sr)
            return current_duration

        stretch_rate = target_duration_sec / current_duration

        logger.debug(
            "%s: %.3f сек → %.3f сек (rate=%.3f)",
            input_path.name, current_duration, target_duration_sec, stretch_rate,
        )

        # Clamping: обмежуємо rate у допустимих межах
        clamped_rate = max(self.config.min_rate, min(self.config.max_rate, stretch_rate))
        was_clamped = (clamped_rate != stretch_rate)

        if was_clamped:
            logger.debug(
                "  Rate %.3f → %.3f (clamped до [%.1f, %.1f])",
                stretch_rate, clamped_rate,
                self.config.min_rate, self.config.max_rate,
            )

        # Виконуємо time-stretching через pyrubberband
        try:
            stretched = pyrb.time_stretch(audio, sr, clamped_rate)
        except Exception as e:
            raise TimeStretcherError(
                f"pyrubberband помилка для {input_path.name}: {e}"
            ) from e

        # Якщо rate був обмежений — доганяємо тишею або обрізаємо
        target_samples = int(target_duration_sec * sr)
        stretched = self._fit_to_length(stretched, target_samples, sr)

        try:
            sf.write(str(output_path), stretched, sr)
        except Exception as e:
            raise TimeStretcherError(f"Не вдалось зберегти {output_path}: {e}") from e

        actual_duration = len(stretched) / sr
        return actual_duration

    # ------------------------------------------------------------------
    # Приватні методи
    # ------------------------------------------------------------------

    def _fit_to_length(
        self, audio: np.ndarray, target_samples: int, sr: int
    ) -> np.ndarray:
        """
        Підрізає або доповнює тишею масив аудіо до точної кількості семплів.
        """
        current_len = len(audio)

        if current_len == target_samples:
            return audio

        if current_len > target_samples:
            # Обрізаємо з кінця
            return audio[:target_samples]

        # Доповнюємо тишею
        padding = np.zeros(target_samples - current_len, dtype=audio.dtype)
        return np.concatenate([audio, padding])
