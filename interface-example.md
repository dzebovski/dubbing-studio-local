
Для створення дійсно гарного ТУІ (Terminal User Interface), який виглядатиме як повноцінна програма (з підтримкою мишки, віджетами, прогрес-барами та логами), найкраще підходить сучасна Python-бібліотека Textual. Вона дозволяє зробити інтерфейс у терміналі, який виглядає майже як веб-додаток.

Нижче я накидав повноцінний макет інтерфейсу. Він має панель налаштувань, динамічну появу знайдених спікерів (де ти обираєш F5-TTS або GPT-SoVITS) та вікно логів для контролю пайплайну.

📦 1. Встановлення бібліотеки
Щоб цей код запрацював, тобі потрібно встановити Textual:

code
Bash
pip install textual
💻 2. Код інтерфейсу (файл ui_app.py)
Скопіюй цей код у файл ui_app.py та запусти (python ui_app.py). Ти побачиш інтерактивний дашборд.

code
Python
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Button, Input, Label, RichLog, ProgressBar, Select, RadioSet, RadioButton
from textual.worker import Worker, WorkerState
import time

class SpeakerConfig(Vertical):
    """Віджет для налаштування конкретного спікера після діаризації"""
    def __init__(self, speaker_name: str, audio_length: int, **kwargs):
        super().__init__(**kwargs)
        self.speaker_name = speaker_name
        self.audio_length = audio_length # У секундах

    def compose(self) -> ComposeResult:
        yield Label(f"🎙 {self.speaker_name} (Знайдено: {self.audio_length} сек. аудіо)", classes="speaker-title")
        yield RadioSet(
            RadioButton("F5-TTS (Zero-shot) - Швидко", id="f5", value=True if self.audio_length < 60 else False),
            RadioButton("GPT-SoVITS (Fine-tune) - Тренування", id="sovits", value=True if self.audio_length >= 60 else False),
            id=f"radioset_{self.speaker_name}"
        )

class DubbingApp(App):
    """Головний додаток TUI для авто-дубляжу"""
    
    CSS = """
    Screen { layout: horizontal; }
    #left-panel { width: 45%; height: 100%; border-right: solid white; padding: 1; }
    #right-panel { width: 55%; height: 100%; padding: 1; }
    .section-title { text-style: bold; color: yellow; margin-bottom: 1; margin-top: 1; }
    .speaker-title { color: cyan; text-style: bold; margin-top: 1; }
    SpeakerConfig { border: round #333; padding: 1; margin-bottom: 1; height: auto; }
    #speakers-container { height: auto; margin-top: 1; }
    #start-btn { width: 100%; margin-top: 2; variant: success; }
    RichLog { border: round gray; height: 1fr; }
    ProgressBar { margin-top: 1; margin-bottom: 1; }
    """

    BINDINGS =[
        ("q", "quit", "Вийти"),
        ("c", "clear_log", "Очистити лог")
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        # Ліва панель: Управління
        with VerticalScroll(id="left-panel"):
            yield Label("📁 Вхідний файл:", classes="section-title")
            yield Input(placeholder="/шлях/до/відео.mp4", id="file_input")
            
            yield Label("🌍 Мова перекладу:", classes="section-title")
            yield Select((("Російська", "ru"), ("Англійська", "en"), ("Українська", "uk")), value="ru", id="lang_select")
            
            yield Label("🤖 LLM для перекладу:", classes="section-title")
            yield Select((("Qwen-2.5-14B", "qwen"), ("Llama-3.1-8B", "llama")), value="qwen", id="llm_select")
            
            # Контейнер для спікерів (заповниться після WhisperX)
            yield Vertical(id="speakers-container")
            
            yield Button("🚀 ЗАПУСТИТИ ПАЙПЛАЙН", id="start-btn")

        # Права панель: Логи та Прогрес
        with Vertical(id="right-panel"):
            yield Label("📊 Статус виконання:", classes="section-title")
            yield ProgressBar(total=100, id="main_progress", show_eta=True)
            yield Label("📝 Системний лог:", classes="section-title")
            yield RichLog(id="sys_log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        self.log_msg("[bold green]Система авто-дубляжу готова![/bold green] Очікування файлу...")
        self.log_msg("[dim]Знайдено GPU: NVIDIA RTX 4080 (16GB VRAM)[/dim]")

    def log_msg(self, message: str) -> None:
        """Допоміжна функція для запису в лог"""
        rich_log = self.query_one(RichLog)
        rich_log.write(message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            file_path = self.query_one("#file_input", Input).value
            if not file_path:
                self.log_msg("[bold red]Помилка:[/bold red] Вкажіть шлях до файлу!")
                return
            
            # Деактивуємо кнопку, щоб не натиснули двічі
            event.button.disabled = True
            self.log_msg(f"[bold blue]Запуск обробки файлу:[/bold blue] {file_path}")
            
            # Запускаємо імітацію фонової роботи пайплайну
            self.run_worker(self.mock_pipeline_execution(), thread=True)

    def action_clear_log(self) -> None:
        self.query_one(RichLog).clear()

    async def mock_pipeline_execution(self) -> None:
        """Тут буде реальний виклик моделей. Зараз це імітація для демонстрації UI."""
        pb = self.query_one(ProgressBar)
        speakers_container = self.query_one("#speakers-container")
        
        # 1. Транскрипція WhisperX
        pb.update(progress=10)
        self.log_msg("[yellow]▶ Етап 1/5: Запуск WhisperX (Транскрипція та Діаризація)...[/yellow]")
        time.sleep(2) # Імітація роботи
        self.log_msg("[green]✔ WhisperX завершив роботу. VRAM очищено.[/green]")
        
        # Динамічно додаємо знайдених спікерів в UI
        self.log_msg("[cyan]Знайдено 2 спікерів. Виберіть метод TTS для кожного в лівій панелі.[/cyan]")
        await self.app.call_from_thread(speakers_container.mount, SpeakerConfig("SPEAKER_00", 120))
        await self.app.call_from_thread(speakers_container.mount, SpeakerConfig("SPEAKER_01", 15))
        
        # Чекаємо 3 секунди, ніби користувач вибирає налаштування (в реальності тут треба пауза/кнопка "Продовжити")
        time.sleep(3) 
        
        # 2. Переклад
        pb.update(progress=40)
        self.log_msg("[yellow]▶ Етап 2/5: Переклад субтитрів через Ollama (isochrony)...[/yellow]")
        time.sleep(2)
        self.log_msg("[green]✔ Переклад завершено.[/green]")

        # 3. TTS
        pb.update(progress=60)
        self.log_msg("[yellow]▶ Етап 3/5: Генерація голосу (TTS)...[/yellow]")
        time.sleep(1)
        self.log_msg("  [dim]- Тренування та генерація SPEAKER_00 (GPT-SoVITS)...[/dim]")
        time.sleep(2)
        self.log_msg("  [dim]- Генерація SPEAKER_01 (F5-TTS)...[/dim]")
        time.sleep(1)
        self.log_msg("[green]✔ Всі аудіофайли згенеровано.[/green]")

        # 4. Time-Stretching
        pb.update(progress=80)
        self.log_msg("[yellow]▶ Етап 4/5: Синхронізація (Time-Stretching)...[/yellow]")
        time.sleep(1.5)
        self.log_msg("[green]✔ Таймінги підігнано під оригінал.[/green]")

        # 5. Зведення
        pb.update(progress=100)
        self.log_msg("[yellow]▶ Етап 5/5: Зведення доріжок...[/yellow]")
        time.sleep(1)
        self.log_msg("[bold green]🎉 ГОТОВО! Фінальний файл збережено у папці /output[/bold green]")
        
        # Відновлюємо кнопку
        btn = self.query_one("#start-btn", Button)
        await self.app.call_from_thread(setattr, btn, "disabled", False)

if __name__ == "__main__":
    app = DubbingApp()
    app.run()
🎨 Як працює цей інтерфейс:
Ліва панель (Налаштування):

Ти вводиш шлях до відео.

Вибираєш мову і модель (наприклад, Qwen).

Коли ти натискаєш "Запустити", скрипт візуально починає роботу.

Динамічний вибір TTS (Магія):

Коли завершиться етап WhisperX (імітація в коді займає 2 сек), у лівій панелі автоматично з'являться налаштування для спікерів.

Якщо скрипт бачить, що у SPEAKER_00 аж 120 секунд голосу, він за замовчуванням перемикає радіокнопку на GPT-SoVITS.

Якщо у SPEAKER_01 лише 15 секунд — він автоматично пропонує F5-TTS. (Ти можеш клікнути мишкою і змінити це).

Права панель (Логи та прогрес):

Прогрес-бар плавно заповнюється.

У красивий лог (з підтримкою кольорів) виводяться всі повідомлення про те, яка модель зараз завантажена і що вона робить. VRAM звільняється — ти бачиш це в лозі.