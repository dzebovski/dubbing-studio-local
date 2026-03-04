"""
reference_collector.py — нарізка аудіо-семплів для кожного спікера.

Вхід:  - project_01/audio/clean.wav
        - список реплік [{speaker, start, end, text}] (з transcriber.py)
Вихід: - project_01/references/SPEAKER_XX/sample_001.wav
        - project_01/references/SPEAKER_XX/sample_002.wav ...

Логіка відбору семплів:
  - Беремо репліки тривалістю від min_sample_duration до max_sample_duration сек
  - Відбираємо найдовші репліки (більше матеріалу — краща якість TTS)
  - Сумарна тривалість для кожного спікера зберігається для рішення F5-TTS vs GPT-SoVITS
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydub import AudioSegment  # type: ignore

from config import ProjectConfig, ReferenceConfig

logger = logging.getLogger(__name__)

Segment = dict[str, Any]


class ReferenceCollectorError(Exception):
    """Базова помилка модуля збору референсів."""


class ReferenceCollector:
    """
    Нарізає найкращі аудіо-семпли для кожного спікера з clean.wav.

    Результати:
      - WAV-файли у project/references/SPEAKER_XX/
      - JSON-файл з метаданими: project/references/speakers_meta.json
    """

    META_FILENAME = "speakers_meta.json"

    def __init__(self, project: ProjectConfig, config: ReferenceConfig):
        self.project = project
        self.config = config

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def collect(
        self,
        segments: list[Segment],
        audio_path: str | Path | None = None,
    ) -> dict[str, dict]:
        """
        Нарізає семпли для всіх спікерів.

        Args:
            segments:   список реплік [{speaker, start, end, text}].
            audio_path: шлях до WAV. Якщо None — project.clean_audio_path.

        Returns:
            Словник {speaker_id: {total_duration_sec, sample_paths, tts_method}}
            де tts_method = "f5_tts" або "gpt_sovits"

        Raises:
            FileNotFoundError: якщо аудіо файл не існує.
            ReferenceCollectorError: при помилці нарізки.
        """
        audio_path = Path(audio_path) if audio_path else self.project.clean_audio_path

        if not audio_path.exists():
            raise FileNotFoundError(f"Аудіо файл не знайдено: {audio_path}")

        logger.info("Завантаження аудіо для нарізки семплів: %s", audio_path)
        try:
            audio = AudioSegment.from_wav(str(audio_path))
        except Exception as e:
            raise ReferenceCollectorError(f"Не вдалось завантажити аудіо: {e}") from e

        # Групуємо репліки по спікерам
        by_speaker = self._group_by_speaker(segments)

        speakers_meta: dict[str, dict] = {}

        for speaker_id, speaker_segments in by_speaker.items():
            logger.info("Збір семплів для %s (%d реплік)...", speaker_id, len(speaker_segments))
            meta = self._collect_for_speaker(speaker_id, speaker_segments, audio)
            speakers_meta[speaker_id] = meta
            logger.info(
                "  %s: %d семплів, %.1f сек загалом → %s",
                speaker_id,
                len(meta["sample_paths"]),
                meta["total_duration_sec"],
                meta["tts_method"],
            )

        self._save_meta(speakers_meta)
        return speakers_meta

    def load_meta(self) -> dict[str, dict]:
        """Завантажує збережені метадані спікерів з JSON."""
        path = self.project.references_dir / self.META_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Метадані референсів не знайдено: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Приватні методи
    # ------------------------------------------------------------------

    def _group_by_speaker(self, segments: list[Segment]) -> dict[str, list[Segment]]:
        """Групує репліки по speaker_id."""
        groups: dict[str, list[Segment]] = {}
        for seg in segments:
            sid = seg.get("speaker", "SPEAKER_00")
            groups.setdefault(sid, []).append(seg)
        return groups

    def _collect_for_speaker(
        self,
        speaker_id: str,
        segments: list[Segment],
        audio: "AudioSegment",
    ) -> dict:
        """
        Відбирає найкращі семпли для одного спікера та нарізає WAV-файли.

        Returns:
            {total_duration_sec, sample_paths, tts_method}
        """
        min_dur = self.config.min_sample_duration_sec
        max_dur = self.config.max_sample_duration_sec

        # Фільтруємо репліки за тривалістю
        valid = [
            seg for seg in segments
            if min_dur <= (seg["end"] - seg["start"]) <= max_dur
        ]

        if not valid:
            # Якщо немає підходящих — беремо всі що є (без фільтру)
            valid = segments
            logger.warning(
                "%s: немає реплік у діапазоні [%.1f, %.1f] сек, беремо всі.",
                speaker_id, min_dur, max_dur,
            )

        # Сортуємо за тривалістю (найдовші — перші)
        valid_sorted = sorted(valid, key=lambda s: s["end"] - s["start"], reverse=True)

        out_dir = self.project.references_dir_for(speaker_id)
        sample_paths: list[str] = []
        total_duration = 0.0

        for i, seg in enumerate(valid_sorted):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            duration_sec = seg["end"] - seg["start"]

            chunk = audio[start_ms:end_ms]

            # Нормалізація гучності
            chunk = chunk.normalize()

            # Конвертуємо до потрібного sample rate
            chunk = chunk.set_frame_rate(self.config.sample_rate).set_channels(1)

            filename = f"sample_{i + 1:03d}.{self.config.audio_format}"
            out_path = out_dir / filename
            chunk.export(str(out_path), format=self.config.audio_format)

            sample_paths.append(str(out_path))
            total_duration += duration_sec

            logger.debug(
                "  Збережено: %s (%.2f сек, текст: %s)",
                filename, duration_sec, seg.get("text", "")[:40],
            )

        tts_method = (
            "gpt_sovits"
            if total_duration >= self.config.gpt_sovits_threshold_sec
            else "f5_tts"
        )

        return {
            "speaker_id": speaker_id,
            "total_duration_sec": round(total_duration, 2),
            "sample_count": len(sample_paths),
            "sample_paths": sample_paths,
            "tts_method": tts_method,
        }

    def _save_meta(self, speakers_meta: dict[str, dict]) -> None:
        """Зберігає метадані всіх спікерів у JSON."""
        out_path = self.project.references_dir / self.META_FILENAME
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(speakers_meta, f, ensure_ascii=False, indent=2)
        logger.info("Метадані референсів збережено: %s", out_path)
