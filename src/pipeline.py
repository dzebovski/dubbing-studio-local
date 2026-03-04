"""
pipeline.py — оркестратор пайплайну авто-дубляжу.

Координує послідовний виклик всіх модулів:
  1. AudioExtractor  — витягування аудіо
  2. Transcriber     — WhisperX транскрипція + діаризація
  3. ReferenceCollector — нарізка семплів
  4. Translator      — переклад через Ollama
  5. TTSEngine       — генерація озвучки
  6. TimeStretcher   — підгонка тривалості
  7. Mixer           — зведення фінального WAV

Зберігає стан між кроками у JSON (pipeline_state.json),
що дозволяє продовжити з будь-якого кроку після збою або паузи.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from config import (
    AppConfig,
    ProjectConfig,
    DEFAULT_CONFIG,
    get_project_config,
)
from audio_extractor import AudioExtractor
from transcriber import Transcriber
from reference_collector import ReferenceCollector
from translator import Translator
from tts_engine import TTSEngine
from time_stretcher import TimeStretcher
from mixer import Mixer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Перелік кроків пайплайну
# ---------------------------------------------------------------------------

class PipelineStep(str, Enum):
    IDLE           = "idle"
    EXTRACT_AUDIO  = "extract_audio"
    TRANSCRIBE     = "transcribe"
    COLLECT_REFS   = "collect_refs"
    TRANSLATE      = "translate"
    GENERATE_TTS   = "generate_tts"
    STRETCH        = "stretch"
    MIX            = "mix"
    DONE           = "done"
    ERROR          = "error"


# Порядок виконання кроків
STEP_ORDER = [
    PipelineStep.EXTRACT_AUDIO,
    PipelineStep.TRANSCRIBE,
    PipelineStep.COLLECT_REFS,
    PipelineStep.TRANSLATE,
    PipelineStep.GENERATE_TTS,
    PipelineStep.STRETCH,
    PipelineStep.MIX,
]


# ---------------------------------------------------------------------------
# Стан пайплайну (серіалізується у JSON)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Зберігає поточний стан виконання пайплайну."""

    input_file: str = ""
    current_step: str = PipelineStep.IDLE
    last_completed_step: str = ""

    # Дані між кроками
    audio_path: str = ""
    segments: list = field(default_factory=list)          # [{speaker, start, end, text}]
    speakers_meta: dict = field(default_factory=dict)     # {speaker_id: {...}}
    translated_segments: list = field(default_factory=list)
    tts_choices: dict = field(default_factory=dict)       # {speaker_id: "f5_tts"|"gpt_sovits"}
    tts_results: dict = field(default_factory=dict)       # {speaker_id: [{segment_index, path}]}
    stretched_results: dict = field(default_factory=dict)
    final_output: str = ""

    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Callbacks типи
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[PipelineStep, int, int], None]   # (step, current, total)
LogCallback = Callable[[str], None]                            # (message)


class PipelineError(Exception):
    """Загальна помилка пайплайну."""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Оркестратор авто-дубляжу.

    Підтримує:
    - Збереження/відновлення стану між кроками
    - Пропуск вже виконаних кроків (resume)
    - Callback-и для відображення прогресу в UI
    - Паузу після COLLECT_REFS для вибору TTS користувачем

    Типовий сценарій:
        pipeline = Pipeline(project_dir, input_file)
        pipeline.run_until_tts_choice()     # зупиняється, чекає вибір TTS
        pipeline.set_tts_choices({...})     # користувач обирає
        pipeline.run_from_tts()             # продовжує до кінця
    """

    STATE_FILENAME = "pipeline_state.json"

    def __init__(
        self,
        project_dir: str | Path,
        input_file: str | Path | None = None,
        config: AppConfig | None = None,
        on_progress: ProgressCallback | None = None,
        on_log: LogCallback | None = None,
    ):
        self.project: ProjectConfig = get_project_config(project_dir)
        self.config: AppConfig = config or DEFAULT_CONFIG
        self.on_progress = on_progress
        self.on_log = on_log

        # Ініціалізуємо або завантажуємо стан
        state_path = self.project.project_dir / self.STATE_FILENAME
        if state_path.exists():
            self.state = self._load_state(state_path)
            self._log(f"Завантажено стан пайплайну. Останній завершений крок: {self.state.last_completed_step}")
        else:
            self.state = PipelineState()
            if input_file:
                self.state.input_file = str(input_file)

        if input_file:
            self.state.input_file = str(input_file)

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def run_until_tts_choice(self) -> PipelineState:
        """
        Виконує кроки до COLLECT_REFS включно, потім зупиняється.
        Після цього UI повинен показати спікерів і дочекатись вибору TTS.

        Returns:
            Поточний стан (містить speakers_meta для побудови UI).
        """
        steps_to_run = [
            PipelineStep.EXTRACT_AUDIO,
            PipelineStep.TRANSCRIBE,
            PipelineStep.COLLECT_REFS,
            PipelineStep.TRANSLATE,
        ]
        self._run_steps(steps_to_run)
        return self.state

    def run_from_tts(self) -> Path:
        """
        Продовжує пайплайн після вибору TTS до самого кінця.

        Returns:
            Path до фінального WAV-файлу.
        """
        if not self.state.tts_choices:
            raise PipelineError(
                "tts_choices не задані. Спочатку викличте set_tts_choices()."
            )

        steps_to_run = [
            PipelineStep.GENERATE_TTS,
            PipelineStep.STRETCH,
            PipelineStep.MIX,
        ]
        self._run_steps(steps_to_run)

        return Path(self.state.final_output)

    def run_all(self, tts_choices: dict[str, str] | None = None) -> Path:
        """
        Запускає весь пайплайн від початку до кінця.

        Args:
            tts_choices: якщо None — автоматично визначається з speakers_meta.

        Returns:
            Path до фінального WAV.
        """
        self.run_until_tts_choice()

        if tts_choices is not None:
            self.set_tts_choices(tts_choices)
        else:
            self._auto_assign_tts_choices()

        return self.run_from_tts()

    def set_tts_choices(self, choices: dict[str, str]) -> None:
        """
        Зберігає вибір TTS від користувача.

        Args:
            choices: {speaker_id: "f5_tts" | "gpt_sovits"}
        """
        self.state.tts_choices = choices
        self._save_state()
        self._log(f"TTS вибір збережено: {choices}")

    def reset(self) -> None:
        """Скидає стан пайплайну (починає з нуля)."""
        input_file = self.state.input_file
        self.state = PipelineState(input_file=input_file)
        self._save_state()
        self._log("Стан пайплайну скинуто.")

    @property
    def speakers_meta(self) -> dict:
        """Метадані спікерів після збору референсів."""
        return self.state.speakers_meta

    # ------------------------------------------------------------------
    # Виконання кроків
    # ------------------------------------------------------------------

    def _run_steps(self, steps: list[PipelineStep]) -> None:
        """Послідовно виконує список кроків (пропускає вже виконані)."""
        for step in steps:
            if self._is_step_done(step):
                self._log(f"Крок {step} вже виконаний — пропускаємо.")
                continue

            self._set_current_step(step)
            try:
                self._execute_step(step)
                self._mark_step_done(step)
            except Exception as e:
                self.state.current_step = PipelineStep.ERROR
                self.state.error_message = str(e)
                self._save_state()
                raise PipelineError(f"Помилка на кроці {step}: {e}") from e

    def _execute_step(self, step: PipelineStep) -> None:
        """Диспетчер: викликає відповідний метод для кожного кроку."""
        dispatch = {
            PipelineStep.EXTRACT_AUDIO: self._step_extract_audio,
            PipelineStep.TRANSCRIBE:    self._step_transcribe,
            PipelineStep.COLLECT_REFS:  self._step_collect_refs,
            PipelineStep.TRANSLATE:     self._step_translate,
            PipelineStep.GENERATE_TTS:  self._step_generate_tts,
            PipelineStep.STRETCH:       self._step_stretch,
            PipelineStep.MIX:           self._step_mix,
        }
        handler = dispatch.get(step)
        if handler:
            handler()

    # ------------------------------------------------------------------
    # Кроки пайплайну
    # ------------------------------------------------------------------

    def _step_extract_audio(self) -> None:
        self._log("Витягування аудіо...")
        extractor = AudioExtractor(self.project)
        audio_path = extractor.extract(self.state.input_file)
        self.state.audio_path = str(audio_path)
        self._save_state()

    def _step_transcribe(self) -> None:
        self._log("Транскрипція та діаризація (WhisperX)...")
        transcriber = Transcriber(self.project, self.config.whisper)
        segments = transcriber.transcribe(self.state.audio_path or None)
        self.state.segments = segments
        self._save_state()
        self._log(f"Знайдено {len(segments)} реплік.")

    def _step_collect_refs(self) -> None:
        self._log("Збір референсних семплів...")
        collector = ReferenceCollector(self.project, self.config.reference)
        speakers_meta = collector.collect(
            self.state.segments,
            audio_path=self.state.audio_path or None,
        )
        self.state.speakers_meta = speakers_meta
        self._save_state()
        for sid, meta in speakers_meta.items():
            self._log(
                f"  {sid}: {meta['sample_count']} семплів, "
                f"{meta['total_duration_sec']:.1f} сек → {meta['tts_method']}"
            )

    def _step_translate(self) -> None:
        self._log("Переклад через Ollama...")
        translator = Translator(self.project, self.config.ollama)
        translated = translator.translate(
            self.state.segments,
            on_progress=lambda cur, tot: self._progress(PipelineStep.TRANSLATE, cur, tot),
        )
        self.state.translated_segments = translated
        self._save_state()
        self._log(f"Переклад завершено: {len(translated)} реплік.")

    def _step_generate_tts(self) -> None:
        self._log("Генерація TTS...")
        engine = TTSEngine(
            self.project,
            self.config.f5_tts,
            self.config.gpt_sovits,
        )
        tts_results = engine.generate_all(
            self.state.translated_segments,
            self.state.speakers_meta,
            self.state.tts_choices,
            on_progress=lambda sid, cur, tot: self._progress(PipelineStep.GENERATE_TTS, cur, tot),
        )
        self.state.tts_results = tts_results
        self._save_state()

    def _step_stretch(self) -> None:
        self._log("Time-stretching...")
        stretcher = TimeStretcher(self.project, self.config.time_stretch)
        stretched = stretcher.stretch_all(
            self.state.translated_segments,
            self.state.tts_results,
            on_progress=lambda cur, tot: self._progress(PipelineStep.STRETCH, cur, tot),
        )
        self.state.stretched_results = stretched
        self._save_state()

    def _step_mix(self) -> None:
        self._log("Зведення фінального WAV...")
        mixer = Mixer(self.project, self.config.mixer)
        output_path = mixer.mix(
            self.state.translated_segments,
            self.state.stretched_results,
            on_progress=lambda cur, tot: self._progress(PipelineStep.MIX, cur, tot),
        )
        self.state.final_output = str(output_path)
        self.state.current_step = PipelineStep.DONE
        self._save_state()
        self._log(f"Готово! Фінальний файл: {output_path}")

    # ------------------------------------------------------------------
    # Стан кроків
    # ------------------------------------------------------------------

    def _is_step_done(self, step: PipelineStep) -> bool:
        """Перевіряє чи крок вже виконаний (для resume)."""
        if not self.state.last_completed_step:
            return False
        try:
            done_idx = STEP_ORDER.index(PipelineStep(self.state.last_completed_step))
            step_idx = STEP_ORDER.index(step)
            return step_idx <= done_idx
        except ValueError:
            return False

    def _set_current_step(self, step: PipelineStep) -> None:
        self.state.current_step = step
        self._save_state()

    def _mark_step_done(self, step: PipelineStep) -> None:
        self.state.last_completed_step = step
        self._save_state()

    def _auto_assign_tts_choices(self) -> None:
        """Автоматично призначає метод TTS на основі тривалості семплів."""
        choices = {}
        for sid, meta in self.state.speakers_meta.items():
            choices[sid] = meta.get("tts_method", "f5_tts")
        self.state.tts_choices = choices
        self._save_state()
        self._log(f"Автоматичний вибір TTS: {choices}")

    # ------------------------------------------------------------------
    # Збереження/завантаження стану
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """Серіалізує стан у JSON."""
        path = self.project.project_dir / self.STATE_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _load_state(path: Path) -> PipelineState:
        """Завантажує стан з JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return PipelineState.from_dict(data)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        logger.info(message)
        if self.on_log:
            self.on_log(message)

    def _progress(self, step: PipelineStep, current: int, total: int) -> None:
        if self.on_progress:
            self.on_progress(step, current, total)
