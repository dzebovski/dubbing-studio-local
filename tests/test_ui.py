"""
test_ui.py — тести компонентів Textual TUI.

Стратегія:
- Використовуємо textual.testing.Pilot для симуляції взаємодії
- Перевіряємо SpeakerCard, основні елементи DubbingApp
- Pipeline замокований — не звертаємось до GPU/диску
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# SpeakerCard
# ---------------------------------------------------------------------------

class TestSpeakerCard:
    @pytest.mark.asyncio
    async def test_default_choice_f5_for_short_audio(self):
        """Якщо tts_method='f5_tts' → за замовчуванням обрано F5-TTS."""
        from ui_app import SpeakerCard
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield SpeakerCard(
                    speaker_id="SPEAKER_01",
                    total_duration_sec=15.0,
                    tts_method="f5_tts",
                    id="card_SPEAKER_01",
                )

        async with TestApp().run_test() as pilot:
            card = pilot.app.query_one(SpeakerCard)
            assert card.get_choice() == "f5_tts"

    @pytest.mark.asyncio
    async def test_default_choice_sovits_for_long_audio(self):
        """Якщо tts_method='gpt_sovits' → за замовчуванням обрано GPT-SoVITS."""
        from ui_app import SpeakerCard
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield SpeakerCard(
                    speaker_id="SPEAKER_00",
                    total_duration_sec=120.0,
                    tts_method="gpt_sovits",
                    id="card_SPEAKER_00",
                )

        async with TestApp().run_test() as pilot:
            card = pilot.app.query_one(SpeakerCard)
            assert card.get_choice() == "gpt_sovits"

    @pytest.mark.asyncio
    async def test_speaker_id_in_label(self):
        """Назва спікера відображається у тексті мітки."""
        from ui_app import SpeakerCard
        from textual.app import App, ComposeResult
        from textual.widgets import Label

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield SpeakerCard(
                    speaker_id="SPEAKER_42",
                    total_duration_sec=30.0,
                    tts_method="f5_tts",
                    id="card",
                )

        async with TestApp().run_test() as pilot:
            card = pilot.app.query_one(SpeakerCard)
            label = card.query_one(Label)
            # Textual 8.x: текст доступний через .content (Static.content)
            label_text = str(label.content)
            assert "SPEAKER_42" in label_text


# ---------------------------------------------------------------------------
# DubbingApp — базові перевірки UI
# ---------------------------------------------------------------------------

class TestDubbingApp:
    @pytest.mark.asyncio
    async def test_app_starts_without_errors(self):
        """Додаток запускається і містить базові віджети."""
        from ui_app import DubbingApp
        from textual.widgets import Input, Button, RichLog, ProgressBar

        async with DubbingApp().run_test() as pilot:
            assert pilot.app.query_one("#input-file", Input) is not None
            assert pilot.app.query_one("#btn-run", Button) is not None
            assert pilot.app.query_one("#sys-log", RichLog) is not None
            assert pilot.app.query_one("#progress-bar", ProgressBar) is not None

    @pytest.mark.asyncio
    async def test_run_button_disabled_without_file(self):
        """Натискання Run без файлу не запускає пайплайн — пишеться помилка в лог."""
        from ui_app import DubbingApp
        from textual.widgets import Button

        async with DubbingApp().run_test(size=(180, 50)) as pilot:
            await pilot.click("#btn-run")
            await pilot.pause()

            btn = pilot.app.query_one("#btn-run", Button)
            assert not btn.disabled

            log_text = pilot.app._last_log_message
            assert log_text is not None
            assert "Вкажіть" in log_text or "Помилка" in log_text

    @pytest.mark.asyncio
    async def test_run_button_disabled_with_nonexistent_file(self, tmp_path: Path):
        """Якщо файл не існує — в лог виводиться помилка."""
        from ui_app import DubbingApp
        from textual.widgets import Input

        async with DubbingApp().run_test(size=(180, 50)) as pilot:
            file_input = pilot.app.query_one("#input-file", Input)
            file_input.value = str(tmp_path / "nonexistent.mp4")

            await pilot.click("#btn-run")
            await pilot.pause()

            log_text = pilot.app._last_log_message
            assert log_text is not None
            assert "не знайдено" in log_text or "Файл" in log_text

    @pytest.mark.asyncio
    async def test_reset_button_clears_ui(self):
        """Кнопка Reset скидає стан і розблоковує Run."""
        from ui_app import DubbingApp
        from textual.widgets import Button

        async with DubbingApp().run_test(size=(180, 50)) as pilot:
            btn_run = pilot.app.query_one("#btn-run", Button)
            btn_run.disabled = True

            await pilot.click("#btn-reset")
            await pilot.pause()

            assert not btn_run.disabled

    @pytest.mark.asyncio
    async def test_clear_log_action(self):
        """Комбінація 'c' очищає лог (не кидає винятків)."""
        from ui_app import DubbingApp

        async with DubbingApp().run_test() as pilot:
            pilot.app._log("Test line 1")
            pilot.app._log("Test line 2")
            await pilot.pause()

            # Натискаємо 'c' — action_clear_log() не повинен кидати
            await pilot.press("c")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_continue_button_hidden_initially(self):
        """Кнопка 'Продовжити' прихована на старті."""
        from ui_app import DubbingApp
        from textual.widgets import Button

        async with DubbingApp().run_test() as pilot:
            btn = pilot.app.query_one("#btn-continue", Button)
            assert not btn.display


# ---------------------------------------------------------------------------
# _collect_tts_choices
# ---------------------------------------------------------------------------

class TestCollectTtsChoices:
    @pytest.mark.asyncio
    async def test_collects_choices_from_cards(self):
        """_collect_tts_choices() зчитує вибір з усіх SpeakerCard."""
        from ui_app import DubbingApp, SpeakerCard
        from textual.app import ComposeResult
        from textual.widgets import Button, Input, RichLog, ProgressBar, Label, Select
        from textual.containers import Horizontal, Vertical, VerticalScroll

        class TestApp(DubbingApp):
            def compose(self) -> ComposeResult:
                # Мінімальний compose з потрібними id
                with Horizontal():
                    with VerticalScroll(id="left-panel"):
                        yield Input(id="input-file")
                        yield Select([("RU", "ru")], value="ru", id="lang-select")
                        yield Select([("Qwen", "qwen2.5:14b")], value="qwen2.5:14b", id="llm-select")
                        with VerticalScroll(id="speakers-scroll"):
                            with Vertical(id="speakers-container"):
                                yield SpeakerCard("SPEAKER_00", 120.0, "gpt_sovits", id="card_SPEAKER_00")
                                yield SpeakerCard("SPEAKER_01", 15.0, "f5_tts", id="card_SPEAKER_01")
                        yield Button("Run", id="btn-run", variant="success")
                        yield Button("Continue", id="btn-continue", variant="primary")
                        yield Button("Reset", id="btn-reset", variant="error")
                    with Vertical(id="right-panel"):
                        yield ProgressBar(total=100, id="progress-bar")
                        yield Label("", id="status-label")
                        yield RichLog(id="sys-log")

        async with TestApp().run_test() as pilot:
            choices = pilot.app._collect_tts_choices()
            assert choices["SPEAKER_00"] == "gpt_sovits"
            assert choices["SPEAKER_01"] == "f5_tts"
