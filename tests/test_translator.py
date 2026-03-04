"""
test_translator.py — unit-тести для Translator.

Стратегія:
- requests.get / requests.post замоковані
- перевіряємо _check_connection, _translate_one, translate()
- перевіряємо retry-логіку
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator import Translator, TranslatorError, OllamaConnectionError
from config import ProjectConfig, OllamaConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> ProjectConfig:
    cfg = ProjectConfig(project_dir=tmp_path / "project")
    cfg.create_dirs()
    return cfg


@pytest.fixture
def ollama_config() -> OllamaConfig:
    return OllamaConfig(
        base_url="http://localhost:11434",
        model="qwen2.5:14b",
        target_language="Russian",
        timeout_seconds=10,
    )


@pytest.fixture
def translator(tmp_project: ProjectConfig, ollama_config: OllamaConfig) -> Translator:
    return Translator(tmp_project, ollama_config)


@pytest.fixture
def sample_segments() -> list[dict]:
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "Hello world"},
        {"speaker": "SPEAKER_01", "start": 3.0, "end": 5.0, "text": "How are you?"},
    ]


# ---------------------------------------------------------------------------
# _check_connection
# ---------------------------------------------------------------------------

class TestCheckConnection:
    @patch("translator.requests.get")
    def test_passes_when_ollama_available(self, mock_get: MagicMock, translator: Translator):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        translator._check_connection()  # не повинно кидати

    @patch("translator.requests.get")
    def test_raises_on_connection_error(self, mock_get: MagicMock, translator: Translator):
        mock_get.side_effect = requests.exceptions.ConnectionError
        with pytest.raises(OllamaConnectionError, match="Не вдалось підключитись"):
            translator._check_connection()

    @patch("translator.requests.get")
    def test_raises_on_timeout(self, mock_get: MagicMock, translator: Translator):
        mock_get.side_effect = requests.exceptions.Timeout
        with pytest.raises(OllamaConnectionError, match="timeout"):
            translator._check_connection()

    @patch("translator.requests.get")
    def test_raises_on_http_error(self, mock_get: MagicMock, translator: Translator):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_get.return_value = resp
        with pytest.raises(OllamaConnectionError):
            translator._check_connection()


# ---------------------------------------------------------------------------
# _translate_one
# ---------------------------------------------------------------------------

class TestTranslateOne:
    @patch("translator.requests.post")
    def test_returns_translated_text(self, mock_post: MagicMock, translator: Translator):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "Привет мир"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        result = translator._translate_one("Hello world")
        assert result == "Привет мир"

    @patch("translator.requests.post")
    def test_retries_on_timeout(self, mock_post: MagicMock, translator: Translator):
        good_response = MagicMock(
            status_code=200,
            json=lambda: {"response": "Привет"},
        )
        good_response.raise_for_status = MagicMock()

        mock_post.side_effect = [
            requests.exceptions.Timeout,
            requests.exceptions.Timeout,
            good_response,
        ]

        with patch("translator.time.sleep"):
            result = translator._translate_one("Hello")

        assert result == "Привет"
        assert mock_post.call_count == 3

    @patch("translator.requests.post")
    def test_raises_after_max_retries(self, mock_post: MagicMock, translator: Translator):
        mock_post.side_effect = requests.exceptions.Timeout

        with patch("translator.time.sleep"):
            with pytest.raises(TranslatorError, match="Не вдалось перекласти"):
                translator._translate_one("Hello")

        assert mock_post.call_count == Translator.MAX_RETRIES

    @patch("translator.requests.post")
    def test_prompt_contains_text(self, mock_post: MagicMock, translator: Translator):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": "Тест"},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        translator._translate_one("Test phrase")

        payload = mock_post.call_args[1]["json"]
        assert "Test phrase" in payload["prompt"]
        assert "Russian" in payload["prompt"]


# ---------------------------------------------------------------------------
# translate()
# ---------------------------------------------------------------------------

class TestTranslate:
    @patch("translator.requests.post")
    @patch("translator.requests.get")
    def test_translate_adds_translated_text(
        self,
        mock_get: MagicMock,
        mock_post: MagicMock,
        translator: Translator,
        sample_segments: list[dict],
        tmp_project: ProjectConfig,
    ):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()

        responses = ["Привет мир", "Как дела?"]
        call_count = {"n": 0}

        def fake_post(*args, **kwargs):
            # Захоплюємо поточний індекс у момент виклику POST
            idx = call_count["n"]
            call_count["n"] += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = lambda: {"response": responses[idx]}
            return resp

        mock_post.side_effect = fake_post

        result = translator.translate(sample_segments)

        assert len(result) == 2
        assert result[0]["translated_text"] == "Привет мир"
        assert result[1]["translated_text"] == "Как дела?"
        # Оригінальні поля збережені
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[0]["text"] == "Hello world"

    @patch("translator.requests.post")
    @patch("translator.requests.get")
    def test_saves_result_to_json(
        self,
        mock_get: MagicMock,
        mock_post: MagicMock,
        translator: Translator,
        sample_segments: list[dict],
        tmp_project: ProjectConfig,
    ):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = lambda: {"response": "Перевод"}
        mock_post.return_value = mock_resp

        translator.translate(sample_segments)
        assert tmp_project.translation_path.exists()

    @patch("translator.requests.post")
    @patch("translator.requests.get")
    def test_calls_on_progress(
        self,
        mock_get: MagicMock,
        mock_post: MagicMock,
        translator: Translator,
        sample_segments: list[dict],
    ):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = lambda: {"response": "OK"}
        mock_post.return_value = mock_resp

        progress_calls = []
        translator.translate(sample_segments, on_progress=lambda c, t: progress_calls.append((c, t)))

        assert progress_calls == [(1, 2), (2, 2)]

    def test_load_saved_raises_if_missing(self, translator: Translator):
        with pytest.raises(FileNotFoundError):
            translator.load_saved()
