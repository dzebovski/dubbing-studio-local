"""
tts_engine.py — генерація озвучки через F5-TTS або GPT-SoVITS.

Вхід:  - список реплік з перекладом [{speaker, start, end, text, translated_text}]
        - метадані спікерів {speaker_id: {tts_method, sample_paths, ...}}
        - вибір методу TTS (може бути перевизначений користувачем через UI)
Вихід: - project_01/tts_output/SPEAKER_XX/phrase_001.wav
        - project_01/tts_output/SPEAKER_XX/phrase_002.wav ...

F5-TTS:    zero-shot, викликається через CLI (f5-tts_infer-cli)
GPT-SoVITS: fine-tune + inference, викликається через subprocess
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from config import ProjectConfig, F5TTSConfig, GPTSoVITSConfig

logger = logging.getLogger(__name__)

Segment = dict[str, Any]


class TTSError(Exception):
    """Базова помилка TTS модуля."""


class TTSEngine:
    """
    Оркеструє генерацію TTS для всіх спікерів.

    Логіка:
    - Для кожного спікера використовує метод визначений у tts_choices
      (F5-TTS або GPT-SoVITS)
    - Після кожного спікера очищує VRAM
    - Результат: dict {speaker_id: [{"segment_index": i, "path": str}]}
    """

    def __init__(
        self,
        project: ProjectConfig,
        f5_config: F5TTSConfig,
        sovits_config: GPTSoVITSConfig,
    ):
        self.project = project
        self.f5_config = f5_config
        self.sovits_config = sovits_config

    # ------------------------------------------------------------------
    # Публічний API
    # ------------------------------------------------------------------

    def generate_all(
        self,
        segments: list[Segment],
        speakers_meta: dict[str, dict],
        tts_choices: dict[str, str],  # {speaker_id: "f5_tts" | "gpt_sovits"}
        on_progress: Any = None,      # callable(speaker_id, current, total) | None
    ) -> dict[str, list[dict]]:
        """
        Генерує озвучку для всіх реплік усіх спікерів.

        Args:
            segments:      список реплік з translated_text.
            speakers_meta: метадані спікерів (з reference_collector).
            tts_choices:   вибір методу TTS для кожного спікера (від користувача).
            on_progress:   callback прогресу.

        Returns:
            {speaker_id: [{"segment_index": i, "path": str, "duration_sec": float}]}
        """
        # Групуємо репліки по спікерам (зберігаємо оригінальний індекс)
        by_speaker: dict[str, list[tuple[int, Segment]]] = {}
        for idx, seg in enumerate(segments):
            sid = seg.get("speaker", "SPEAKER_00")
            by_speaker.setdefault(sid, []).append((idx, seg))

        results: dict[str, list[dict]] = {}

        for speaker_id, indexed_segs in by_speaker.items():
            method = tts_choices.get(speaker_id, "f5_tts")
            meta = speakers_meta.get(speaker_id, {})
            sample_paths = meta.get("sample_paths", [])

            if not sample_paths:
                raise TTSError(
                    f"Немає референсних семплів для {speaker_id}. "
                    "Спочатку запустіть reference_collector."
                )

            logger.info(
                "Генерація TTS для %s (%s): %d реплік...",
                speaker_id, method, len(indexed_segs),
            )

            if method == "gpt_sovits":
                speaker_results = self._generate_sovits(
                    speaker_id, indexed_segs, sample_paths, on_progress
                )
            else:
                speaker_results = self._generate_f5(
                    speaker_id, indexed_segs, sample_paths, on_progress
                )

            results[speaker_id] = speaker_results
            self._clear_vram()

        self._save_results_meta(results)
        return results

    # ------------------------------------------------------------------
    # F5-TTS
    # ------------------------------------------------------------------

    def _generate_f5(
        self,
        speaker_id: str,
        indexed_segs: list[tuple[int, Segment]],
        sample_paths: list[str],
        on_progress: Any,
    ) -> list[dict]:
        """
        Генерує озвучку через F5-TTS (zero-shot).

        Використовує перший семпл як reference audio.
        Викликає CLI: f5-tts_infer-cli --ref_audio ... --ref_text ... --gen_text ...
        """
        reference_audio = sample_paths[0]
        out_dir = self.project.tts_output_dir_for(speaker_id)
        results = []

        for i, (seg_idx, seg) in enumerate(indexed_segs):
            text = seg.get("translated_text") or seg.get("text", "")
            out_path = out_dir / f"phrase_{seg_idx:04d}.wav"

            cmd = [
                self.f5_config.executable,
                "--ref_audio", reference_audio,
                "--ref_text", "",        # F5-TTS може авто-транскрибувати референс
                "--gen_text", text,
                "--output_file", str(out_path),
                "--device", self.f5_config.device,
            ]

            logger.debug("F5-TTS [%d/%d]: %s", i + 1, len(indexed_segs), text[:50])
            self._run_subprocess(cmd, f"F5-TTS phrase {seg_idx}")

            duration = self._get_wav_duration(out_path)
            results.append({
                "segment_index": seg_idx,
                "path": str(out_path),
                "duration_sec": duration,
            })

            if on_progress:
                on_progress(speaker_id, i + 1, len(indexed_segs))

        return results

    # ------------------------------------------------------------------
    # GPT-SoVITS
    # ------------------------------------------------------------------

    def _generate_sovits(
        self,
        speaker_id: str,
        indexed_segs: list[tuple[int, Segment]],
        sample_paths: list[str],
        on_progress: Any,
    ) -> list[dict]:
        """
        Тренує GPT-SoVITS на семплах спікера, потім генерує озвучку.

        Кроки:
        1. Підготовка даних (список WAV + txt файл)
        2. Запуск тренування (subprocess → train.py)
        3. Генерація кожної фрази (subprocess → inference_cli.py)
        """
        out_dir = self.project.tts_output_dir_for(speaker_id)
        model_dir = out_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Підготовка: пишемо список семплів у файл
        list_file = model_dir / "train_list.txt"
        self._write_train_list(list_file, sample_paths)

        # 2. Тренування
        logger.info("GPT-SoVITS: починаємо тренування для %s...", speaker_id)
        self._run_sovits_training(speaker_id, list_file, model_dir)
        logger.info("GPT-SoVITS: тренування %s завершено.", speaker_id)

        # 3. Генерація реплік
        results = []
        for i, (seg_idx, seg) in enumerate(indexed_segs):
            text = seg.get("translated_text") or seg.get("text", "")
            out_path = out_dir / f"phrase_{seg_idx:04d}.wav"

            self._run_sovits_inference(
                speaker_id=speaker_id,
                text=text,
                reference_audio=sample_paths[0],
                model_dir=model_dir,
                out_path=out_path,
            )

            duration = self._get_wav_duration(out_path)
            results.append({
                "segment_index": seg_idx,
                "path": str(out_path),
                "duration_sec": duration,
            })

            if on_progress:
                on_progress(speaker_id, i + 1, len(indexed_segs))

        return results

    def _write_train_list(self, list_file: Path, sample_paths: list[str]) -> None:
        """Пише список WAV-файлів для тренування GPT-SoVITS."""
        with open(list_file, "w", encoding="utf-8") as f:
            for p in sample_paths:
                # Формат: шлях|мова|текст (текст опціональний для SoVITS)
                f.write(f"{p}|ru|\n")

    def _run_sovits_training(
        self, speaker_id: str, list_file: Path, model_dir: Path
    ) -> None:
        """Запускає тренування GPT-SoVITS через subprocess."""
        root = self.sovits_config.root_dir
        train_script = root / self.sovits_config.train_script

        cmd = [
            sys.executable,
            str(train_script),
            "--train_list", str(list_file),
            "--output_dir", str(model_dir),
            "--gpt_epochs", str(self.sovits_config.gpt_epochs),
            "--sovits_epochs", str(self.sovits_config.sovits_epochs),
            "--device", self.sovits_config.device,
        ]

        self._run_subprocess(cmd, f"GPT-SoVITS training ({speaker_id})", timeout=7200)

    def _run_sovits_inference(
        self,
        speaker_id: str,
        text: str,
        reference_audio: str,
        model_dir: Path,
        out_path: Path,
    ) -> None:
        """Генерує одну фразу через GPT-SoVITS inference CLI."""
        root = self.sovits_config.root_dir
        infer_script = root / self.sovits_config.infer_script

        # Шукаємо останній чекпоінт
        gpt_ckpt = self._find_latest_ckpt(model_dir, "*.ckpt")
        sovits_ckpt = self._find_latest_ckpt(model_dir, "*.pth")

        cmd = [
            sys.executable,
            str(infer_script),
            "--gpt_model", str(gpt_ckpt),
            "--sovits_model", str(sovits_ckpt),
            "--ref_audio", reference_audio,
            "--ref_text", "",
            "--text", text,
            "--text_lang", "ru",
            "--output_path", str(out_path),
        ]

        self._run_subprocess(cmd, f"GPT-SoVITS inference ({speaker_id})")

    # ------------------------------------------------------------------
    # Допоміжні методи
    # ------------------------------------------------------------------

    def _find_latest_ckpt(self, model_dir: Path, pattern: str) -> Path:
        """Знаходить найновіший файл чекпоінту за патерном."""
        files = list(model_dir.glob(pattern))
        if not files:
            raise TTSError(
                f"Не знайдено чекпоінт {pattern} у {model_dir}. "
                "Можливо тренування не завершилось."
            )
        return max(files, key=lambda p: p.stat().st_mtime)

    def _run_subprocess(
        self, cmd: list[str], label: str, timeout: int = 300
    ) -> None:
        """Запускає subprocess, логує stdout/stderr, кидає TTSError при помилці."""
        logger.debug("Запуск [%s]: %s", label, " ".join(str(c) for c in cmd))
        try:
            result = subprocess.run(
                [str(c) for c in cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise TTSError(f"[{label}] Timeout ({timeout} сек) перевищено.")
        except FileNotFoundError as e:
            raise TTSError(f"[{label}] Виконуваний файл не знайдено: {e}")

        if result.stdout:
            logger.debug("[%s] stdout: %s", label, result.stdout[-500:])
        if result.returncode != 0:
            raise TTSError(
                f"[{label}] завершився з кодом {result.returncode}.\n"
                f"stderr: {result.stderr[-1000:]}"
            )

    def _get_wav_duration(self, path: Path) -> float:
        """Повертає тривалість WAV-файлу у секундах."""
        try:
            import soundfile as sf  # type: ignore
            info = sf.info(str(path))
            return info.duration
        except Exception:
            return 0.0

    def _save_results_meta(self, results: dict[str, list[dict]]) -> None:
        """Зберігає метадані згенерованих файлів у JSON."""
        out_path = self.project.tts_output_dir / "results_meta.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("TTS результати збережено: %s", out_path)

    @staticmethod
    def _clear_vram() -> None:
        """Очищує GPU пам'ять після обробки кожного спікера."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("VRAM очищено.")
