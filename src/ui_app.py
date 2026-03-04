"""
ui_app.py — Textual TUI двопанельний дашборд для авто-дубляжу.

Ліва панель:
  - Вхідний файл
  - Вибір мови перекладу
  - Вибір LLM моделі
  - Динамічний список спікерів (з'являється після WhisperX)
  - Кнопки управління

Права панель:
  - Прогрес-бар
  - Системний лог (Rich markup)
"""

import logging
import sys
from pathlib import Path
from threading import Event

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
)
from textual.worker import Worker

# Додаємо src/ до шляхів імпорту (щоб ui_app.py можна було запустити напряму)
sys.path.insert(0, str(Path(__file__).parent))

from config import AppConfig, DEFAULT_CONFIG, OllamaConfig, get_project_config
from pipeline import Pipeline, PipelineStep, PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Константи
# ---------------------------------------------------------------------------

TOTAL_STEPS = 7   # кількість кроків пайплайну для прогрес-бара

# Відображення кроку → % прогресу
STEP_PROGRESS = {
    PipelineStep.EXTRACT_AUDIO: 10,
    PipelineStep.TRANSCRIBE:    25,
    PipelineStep.COLLECT_REFS:  40,
    PipelineStep.TRANSLATE:     55,
    PipelineStep.GENERATE_TTS:  75,
    PipelineStep.STRETCH:       90,
    PipelineStep.MIX:           100,
    PipelineStep.DONE:          100,
}

LANG_OPTIONS = [("Російська (RU)", "ru"), ("Українська (UK)", "uk"), ("Англійська (EN)", "en")]
LLM_OPTIONS  = [("Qwen-2.5-14B", "qwen2.5:14b"), ("Llama-3.1-8B", "llama3.1:8b")]

PROJECT_DIR = Path(__file__).resolve().parent.parent / "project_01"


# ---------------------------------------------------------------------------
# Субвіджет: картка спікера
# ---------------------------------------------------------------------------

class SpeakerCard(Vertical):
    """
    Картка одного спікера з вибором методу TTS.
    З'являється динамічно після завершення діаризації.
    """

    DEFAULT_CSS = """
    SpeakerCard {
        border: round #444;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }
    SpeakerCard .card-title {
        color: cyan;
        text-style: bold;
    }
    """

    def __init__(self, speaker_id: str, total_duration_sec: float, tts_method: str, **kwargs):
        super().__init__(**kwargs)
        self.speaker_id = speaker_id
        self.total_duration_sec = total_duration_sec
        self.tts_method = tts_method

    def compose(self) -> ComposeResult:
        dur = self.total_duration_sec
        yield Label(
            f"Speaker: [bold]{self.speaker_id}[/bold]  ({dur:.0f} сек аудіо)",
            classes="card-title",
        )
        use_sovits = (self.tts_method == "gpt_sovits")
        yield RadioSet(
            RadioButton(
                "F5-TTS (Zero-shot, швидко)",
                id=f"f5_{self.speaker_id}",
                value=not use_sovits,
            ),
            RadioButton(
                "GPT-SoVITS (Fine-tune, якісно)",
                id=f"sovits_{self.speaker_id}",
                value=use_sovits,
            ),
            id=f"radioset_{self.speaker_id}",
        )

    def get_choice(self) -> str:
        """Повертає поточний вибір: 'f5_tts' або 'gpt_sovits'."""
        radioset = self.query_one(RadioSet)
        pressed = radioset.pressed_button
        if pressed and pressed.id and pressed.id.startswith("sovits_"):
            return "gpt_sovits"
        return "f5_tts"


# ---------------------------------------------------------------------------
# Головний додаток
# ---------------------------------------------------------------------------

class DubbingApp(App):
    """Textual TUI для AI Auto-Dubbing Pipeline."""

    TITLE = "AI Auto-Dubbing Studio"
    SUB_TITLE = "RTX 4080 | EN → RU"

    CSS = """
    Screen {
        layout: horizontal;
    }

    #left-panel {
        width: 42%;
        height: 100%;
        border-right: solid #444;
        padding: 1 2;
    }

    #right-panel {
        width: 58%;
        height: 100%;
        padding: 1 2;
    }

    .section-title {
        text-style: bold;
        color: yellow;
        margin-bottom: 1;
        margin-top: 1;
    }

    #speakers-scroll {
        height: auto;
        max-height: 18;
        margin-top: 1;
    }

    #speakers-container {
        height: auto;
    }

    #btn-run {
        width: 100%;
        margin-top: 1;
    }

    #btn-continue {
        width: 100%;
        margin-top: 1;
        display: none;
    }

    #btn-reset {
        width: 100%;
        margin-top: 1;
    }

    RichLog {
        border: round #555;
        height: 1fr;
        margin-top: 1;
    }

    ProgressBar {
        margin-top: 1;
        margin-bottom: 1;
    }

    #status-label {
        color: green;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Вийти"),
        Binding("c", "clear_log", "Очистити лог"),
        Binding("r", "reset_pipeline", "Скинути"),
    ]

    def __init__(self, config: AppConfig | None = None, **kwargs):
        super().__init__(**kwargs)
        self.app_config: AppConfig = config or DEFAULT_CONFIG
        self._pipeline: Pipeline | None = None
        self._tts_choice_ready = Event()
        self._last_log_message: str | None = None  # для тестів

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            # ---- Ліва панель ----
            with VerticalScroll(id="left-panel"):
                yield Label("Вхідний файл:", classes="section-title")
                yield Input(
                    placeholder="/шлях/до/відео.mp4",
                    id="input-file",
                )

                yield Label("Мова перекладу:", classes="section-title")
                yield Select(LANG_OPTIONS, value="ru", id="lang-select")

                yield Label("LLM модель:", classes="section-title")
                yield Select(LLM_OPTIONS, value="qwen2.5:14b", id="llm-select")

                yield Label("Спікери (після діаризації):", classes="section-title")
                with VerticalScroll(id="speakers-scroll"):
                    yield Vertical(id="speakers-container")

                yield Button(
                    "ЗАПУСТИТИ ПАЙПЛАЙН",
                    id="btn-run",
                    variant="success",
                )
                yield Button(
                    "ПРОДОВЖИТИ (вибір TTS збережено)",
                    id="btn-continue",
                    variant="primary",
                )
                yield Button(
                    "СКИНУТИ СТАН",
                    id="btn-reset",
                    variant="error",
                )

            # ---- Права панель ----
            with Vertical(id="right-panel"):
                yield Label("Прогрес:", classes="section-title")
                yield ProgressBar(total=100, id="progress-bar", show_eta=False)
                yield Label("", id="status-label")
                yield Label("Системний лог:", classes="section-title")
                yield RichLog(id="sys-log", highlight=True, markup=True)

        yield Footer()

    # ------------------------------------------------------------------
    # on_mount
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._log("[bold green]AI Auto-Dubbing Studio готовий.[/bold green]")
        self._log(f"[dim]Робоча папка: {PROJECT_DIR}[/dim]")

        # Перевіряємо чи є збережений стан
        state_path = PROJECT_DIR / "pipeline_state.json"
        if state_path.exists():
            self._log("[yellow]Знайдено збережений стан. Можна продовжити з попередньої точки.[/yellow]")

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        if btn_id == "btn-run":
            self._handle_run()
        elif btn_id == "btn-continue":
            self._handle_continue()
        elif btn_id == "btn-reset":
            self._handle_reset()

    def _handle_run(self) -> None:
        file_path = self.query_one("#input-file", Input).value.strip()
        if not file_path:
            self._log("[bold red]Помилка:[/bold red] Вкажіть шлях до файлу!")
            return

        if not Path(file_path).exists():
            self._log(f"[bold red]Файл не знайдено:[/bold red] {file_path}")
            return

        # Оновлюємо config з вибраних параметрів
        lang_select = self.query_one("#lang-select", Select)
        llm_select  = self.query_one("#llm-select", Select)

        if lang_select.value and lang_select.value != Select.BLANK:
            self.app_config.ollama.target_language = {
                "ru": "Russian", "uk": "Ukrainian", "en": "English"
            }.get(str(lang_select.value), "Russian")

        if llm_select.value and llm_select.value != Select.BLANK:
            self.app_config.ollama.model = str(llm_select.value)

        PROJECT_DIR.mkdir(parents=True, exist_ok=True)

        self.query_one("#btn-run", Button).disabled = True
        self.query_one("#btn-continue").display = False
        self._set_status("")

        # Запускаємо фонового воркера
        self.run_worker(
            self._worker_run_until_tts(file_path),
            thread=True,
            name="pipeline-pre-tts",
        )

    def _handle_continue(self) -> None:
        """Зчитує вибір TTS і продовжує пайплайн."""
        if self._pipeline is None:
            return

        choices = self._collect_tts_choices()
        self._pipeline.set_tts_choices(choices)

        self.query_one("#btn-continue").display = False
        self.query_one("#btn-run", Button).disabled = True

        self.run_worker(
            self._worker_run_from_tts(),
            thread=True,
            name="pipeline-post-tts",
        )

    def _handle_reset(self) -> None:
        if self._pipeline:
            self._pipeline.reset()
        self._log("[yellow]Стан скинуто. Можна починати заново.[/yellow]")
        self._clear_speakers()
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-continue").display = False
        self.query_one("#progress-bar", ProgressBar).update(progress=0)
        self._set_status("")

    # ------------------------------------------------------------------
    # Workers (фонові задачі)
    # ------------------------------------------------------------------

    async def _worker_run_until_tts(self, file_path: str) -> None:
        """Кроки 1-4: extract → transcribe → collect_refs → translate.
        Виконується в окремому треді через run_worker(..., thread=True).
        Всі оновлення UI — тільки через call_from_thread.
        """
        try:
            self._pipeline = Pipeline(
                project_dir=PROJECT_DIR,
                input_file=file_path,
                config=self.app_config,
                on_progress=self._on_pipeline_progress,
                on_log=self._on_pipeline_log,
            )

            # Синхронний виклик — ми вже в окремому треді
            state = self._pipeline.run_until_tts_choice()

            # Показуємо спікерів у UI (через call_from_thread → schedule coroutine)
            self.call_from_thread(self._mount_speaker_cards, state.speakers_meta)

        except Exception as e:
            self.call_from_thread(self._log, f"[bold red]Помилка:[/bold red] {e}")
            self.call_from_thread(
                lambda: setattr(self.query_one("#btn-run", Button), "disabled", False)
            )

    async def _worker_run_from_tts(self) -> None:
        """Кроки 5-7: generate_tts → stretch → mix.
        Виконується в окремому треді через run_worker(..., thread=True).
        """
        try:
            output_path = self._pipeline.run_from_tts()
            self.call_from_thread(
                self._log,
                f"[bold green]ГОТОВО![/bold green] Файл збережено: [cyan]{output_path}[/cyan]",
            )
            self.call_from_thread(self._set_status, "Завершено!")
        except Exception as e:
            self.call_from_thread(self._log, f"[bold red]Помилка:[/bold red] {e}")
        finally:
            self.call_from_thread(
                lambda: setattr(self.query_one("#btn-run", Button), "disabled", False)
            )

    # ------------------------------------------------------------------
    # UI helpers (викликаються з потоку воркера через call_from_thread)
    # ------------------------------------------------------------------

    def _mount_speaker_cards(self, speakers_meta: dict) -> None:
        """
        Планує монтування карток спікерів через post_message / mount.
        Викликається через call_from_thread → виконується в головному треді.
        """
        self._clear_speakers()
        container = self.query_one("#speakers-container", Vertical)
        for sid, meta in speakers_meta.items():
            card = SpeakerCard(
                speaker_id=sid,
                total_duration_sec=meta.get("total_duration_sec", 0),
                tts_method=meta.get("tts_method", "f5_tts"),
                id=f"card_{sid}",
            )
            container.mount(card)

        self._log(
            f"[cyan]Знайдено {len(speakers_meta)} спікерів.[/cyan] "
            "Перевірте вибір TTS і натисніть «Продовжити»."
        )
        self._show_continue_button()

    def _clear_speakers(self) -> None:
        """Видаляє всі картки спікерів."""
        container = self.query_one("#speakers-container", Vertical)
        for card in container.query(SpeakerCard):
            card.remove()

    def _show_continue_button(self) -> None:
        self.query_one("#btn-continue").display = True

    def _collect_tts_choices(self) -> dict[str, str]:
        """Зчитує вибір TTS з усіх карток спікерів."""
        choices = {}
        for card in self.query(SpeakerCard):
            choices[card.speaker_id] = card.get_choice()
        return choices

    def _set_status(self, text: str) -> None:
        self.query_one("#status-label", Label).update(text)

    def _set_status_from_thread(self, text: str) -> None:
        self.call_from_thread(self._set_status, text)

    # ------------------------------------------------------------------
    # Pipeline callbacks
    # ------------------------------------------------------------------

    def _on_pipeline_progress(self, step: PipelineStep, current: int, total: int) -> None:
        """Оновлює прогрес-бар з потоку пайплайну."""
        step_end = STEP_PROGRESS.get(step, 0)
        step_start = STEP_PROGRESS.get(self._prev_step(step), 0)
        step_range = step_end - step_start

        if total > 0:
            progress = step_start + (current / total) * step_range
        else:
            progress = step_end

        self.call_from_thread(
            self.query_one("#progress-bar", ProgressBar).update,
            progress=int(min(progress, 100)),
        )

    def _on_pipeline_log(self, message: str) -> None:
        """Пише лог-повідомлення з пайплайну у RichLog (викликається з треду)."""
        self.call_from_thread(self._log, f"[dim]{message}[/dim]")

    def _prev_step(self, step: PipelineStep) -> PipelineStep:
        """Повертає попередній крок у ланцюжку."""
        from pipeline import STEP_ORDER
        try:
            idx = STEP_ORDER.index(step)
            return STEP_ORDER[idx - 1] if idx > 0 else step
        except ValueError:
            return step

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """
        Запис у RichLog.
        Якщо викликається з не-головного треду — використовуй call_from_thread(_log, msg).
        Якщо з головного треду — можна викликати напряму.
        """
        self._last_log_message = message
        try:
            rich_log = self.query_one("#sys-log", RichLog)
            rich_log.write(message)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_clear_log(self) -> None:
        self.query_one("#sys-log", RichLog).clear()

    def action_reset_pipeline(self) -> None:
        self._handle_reset()


# ---------------------------------------------------------------------------
# Точка входу
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = DubbingApp()
    app.run()


if __name__ == "__main__":
    main()
