"""
translator.py — переклад реплік з EN→RU через Ollama API з isochrony-промптом.

Вхід:  список реплік [{speaker, start, end, text}] (з transcriber.py)
Вихід: той самий список + ключ "translated_text" у кожній репліці
       зберігається у project_01/translation/result.json

Isochrony: промпт просить LLM зберегти приблизно ту саму кількість складів,
щоб переклад вліз у відведений таймінг без надмірного time-stretching.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests  # type: ignore

from config import ProjectConfig, OllamaConfig

logger = logging.getLogger(__name__)

Segment = dict[str, Any]


class TranslatorError(Exception):
    """Базова помилка модуля перекладу."""


class OllamaConnectionError(TranslatorError):
    """Ollama недоступна."""


class Translator:
    """
    Перекладає репліки через локальний Ollama API.

    Особливості:
    - Кожна репліка перекладається окремим запитом (для точного контролю isochrony)
    - Retry при тимчасових збоях (до 3 спроб з паузою)
    - Прогрес логується через callback on_progress(current, total)
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SEC = 2.0

    def __init__(self, project: ProjectConfig, config: OllamaConfig):
        self.project = project
        self.config = config

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def translate(
        self,
        segments: list[Segment],
        on_progress: Any = None,  # callable(current: int, total: int) | None
    ) -> list[Segment]:
        """
        Перекладає всі репліки та повертає збагачений список.

        Args:
            segments:    список [{speaker, start, end, text}].
            on_progress: опціональний callback для відображення прогресу в UI.

        Returns:
            Той самий список з доданим ключем "translated_text".

        Raises:
            OllamaConnectionError: якщо Ollama недоступна.
            TranslatorError: при інших помилках.
        """
        self._check_connection()

        total = len(segments)
        result: list[Segment] = []

        for i, seg in enumerate(segments):
            translated = self._translate_one(seg["text"])
            new_seg = {**seg, "translated_text": translated}
            result.append(new_seg)

            if on_progress:
                on_progress(i + 1, total)

            logger.debug(
                "[%d/%d] %s → %s",
                i + 1, total,
                seg["text"][:50],
                translated[:50],
            )

        logger.info("Переклад завершено. Оброблено %d реплік.", total)
        self._save(result)
        return result

    def load_saved(self) -> list[Segment]:
        """Завантажує збережений результат перекладу з JSON."""
        path = self.project.translation_path
        if not path.exists():
            raise FileNotFoundError(f"Збережений переклад не знайдено: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Приватні методи
    # ------------------------------------------------------------------

    def _check_connection(self) -> None:
        """Перевіряє доступність Ollama перед початком перекладу."""
        try:
            resp = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(
                f"Не вдалось підключитись до Ollama: {self.config.base_url}\n"
                "Переконайтесь що Ollama запущена: ollama serve"
            )
        except requests.exceptions.HTTPError as e:
            raise OllamaConnectionError(f"Ollama відповіла з помилкою: {e}")
        except requests.exceptions.Timeout:
            raise OllamaConnectionError("Ollama не відповідає (timeout).")

    def _translate_one(self, text: str) -> str:
        """
        Перекладає одну фразу з retry-логікою.

        Args:
            text: оригінальний текст (англійська).

        Returns:
            Перекладений текст.
        """
        prompt = self.config.prompt_template.format(
            target_language=self.config.target_language,
            text=text,
        )

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,   # низька температура для стабільного перекладу
                "top_p": 0.9,
            },
        }

        last_error: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.config.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "").strip()

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning("Спроба %d/%d: timeout Ollama.", attempt, self.MAX_RETRIES)
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning("Спроба %d/%d: помилка запиту: %s", attempt, self.MAX_RETRIES, e)

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY_SEC)

        raise TranslatorError(
            f"Не вдалось перекласти після {self.MAX_RETRIES} спроб. "
            f"Остання помилка: {last_error}"
        )

    def _save(self, segments: list[Segment]) -> None:
        """Зберігає результат перекладу у JSON."""
        out_path = self.project.translation_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        logger.info("Переклад збережено: %s", out_path)
