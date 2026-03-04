"""
transcriber.py — транскрипція та діаризація аудіо через WhisperX.

Вхід:  project_01/audio/clean.wav
Вихід: project_01/transcription/result.json

Формат виходу (список реплік):
[
  {
    "speaker": "SPEAKER_00",
    "start":   0.0,
    "end":     3.5,
    "text":    "Hello world"
  },
  ...
]

Важливо: після завершення обов'язково викликати torch.cuda.empty_cache()
щоб звільнити VRAM для наступних моделей.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch

from config import ProjectConfig, WhisperConfig

logger = logging.getLogger(__name__)


# Тип однієї репліки
Segment = dict[str, Any]  # {speaker, start, end, text}


class TranscriberError(Exception):
    """Базова помилка модуля транскрипції."""


class Transcriber:
    """
    Запускає WhisperX (large-v3) для транскрипції та діаризації.

    Послідовність кроків всередині:
      1. Завантаження WhisperX-моделі
      2. Транскрипція аудіо → сегменти з таймінгами на рівні слів
      3. Вирівнювання (alignment) для точних мілісекундних таймінгів
      4. Діаризація (diarize) → прив'язка speaker_id до кожного слова
      5. Агрегація слів у репліки по speaker_id
      6. Збереження у JSON
      7. Очищення VRAM
    """

    def __init__(self, project: ProjectConfig, config: WhisperConfig):
        self.project = project
        self.config = config

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str | Path | None = None) -> list[Segment]:
        """
        Транскрибує та діаризує аудіо.

        Args:
            audio_path: шлях до WAV. Якщо None — використовує project.clean_audio_path.

        Returns:
            Список реплік [{speaker, start, end, text}], відсортованих за start.

        Raises:
            FileNotFoundError: якщо аудіо файл не існує.
            TranscriberError: при будь-якій помилці WhisperX.
        """
        audio_path = Path(audio_path) if audio_path else self.project.clean_audio_path

        if not audio_path.exists():
            raise FileNotFoundError(f"Аудіо файл не знайдено: {audio_path}")

        logger.info("Початок транскрипції: %s", audio_path)

        try:
            segments = self._run_whisperx(audio_path)
        except ImportError as e:
            raise TranscriberError(
                "whisperx не встановлений. Виконай: pip install whisperx"
            ) from e
        except Exception as e:
            raise TranscriberError(f"Помилка WhisperX: {e}") from e
        finally:
            self._clear_vram()

        logger.info("Транскрипція завершена. Знайдено %d реплік.", len(segments))

        self._save(segments)
        return segments

    def load_saved(self) -> list[Segment]:
        """
        Завантажує збережений результат транскрипції з JSON.

        Raises:
            FileNotFoundError: якщо result.json не існує.
        """
        path = self.project.transcription_path
        if not path.exists():
            raise FileNotFoundError(
                f"Збережений результат транскрипції не знайдено: {path}"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Приватні методи
    # ------------------------------------------------------------------

    def _run_whisperx(self, audio_path: Path) -> list[Segment]:
        """Основна логіка WhisperX: завантаження, транскрипція, alignment, diarize."""
        import whisperx  # type: ignore

        device = self.config.device
        compute_type = self.config.compute_type

        # 1. Завантаження моделі WhisperX
        logger.info("Завантаження WhisperX %s (device=%s)...", self.config.model_name, device)
        model = whisperx.load_model(
            self.config.model_name,
            device=device,
            compute_type=compute_type,
        )

        # 2. Транскрипція
        logger.info("Транскрипція...")
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(
            audio,
            batch_size=self.config.batch_size,
            language=self.config.language,
        )

        # Вивантажуємо транскрипційну модель перед alignment
        del model
        self._clear_vram()

        # 3. Вирівнювання (alignment) — точні тайминги на рівні слів
        logger.info("Alignment (word-level timings)...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )
        del model_a
        self._clear_vram()

        # 4. Діаризація — прив'язка speaker_id
        if self.config.hf_token:
            logger.info("Діаризація (speaker diarization)...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.config.hf_token,
                device=device,
            )
            diarize_segments = diarize_model(
                audio,
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
            del diarize_model
            self._clear_vram()
        else:
            logger.warning(
                "HF_TOKEN не задано — діаризація пропущена. "
                "Всі репліки будуть позначені як SPEAKER_00."
            )

        # 5. Агрегація слів у репліки
        return self._aggregate_segments(result["segments"])

    def _aggregate_segments(self, raw_segments: list[dict]) -> list[Segment]:
        """
        Перетворює сегменти WhisperX у стандартний формат {speaker, start, end, text}.

        WhisperX повертає сегменти де speaker може бути вказаний або відсутній.
        Об'єднуємо послідовні сегменти одного спікера якщо між ними < 0.5 сек.
        """
        segments: list[Segment] = []

        for raw in raw_segments:
            speaker = raw.get("speaker", "SPEAKER_00")
            start = float(raw.get("start", 0.0))
            end = float(raw.get("end", 0.0))
            text = raw.get("text", "").strip()

            if not text:
                continue

            # Об'єднати з попереднім сегментом якщо той самий спікер і пауза < 0.5 сек
            if (
                segments
                and segments[-1]["speaker"] == speaker
                and (start - segments[-1]["end"]) < 0.5
            ):
                segments[-1]["end"] = end
                segments[-1]["text"] += " " + text
            else:
                segments.append({
                    "speaker": speaker,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                })

        return sorted(segments, key=lambda s: s["start"])

    def _save(self, segments: list[Segment]) -> None:
        """Зберігає результат у JSON."""
        out_path = self.project.transcription_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        logger.info("Транскрипція збережена: %s", out_path)

    @staticmethod
    def _clear_vram() -> None:
        """Очищує GPU-пам'ять між завантаженнями моделей."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("VRAM очищено.")
