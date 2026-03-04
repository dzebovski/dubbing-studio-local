"""
audio_export.py — експорт WAV у стиснутий формат (MP3) через pydub.

Потрібен ffmpeg у PATH для кодека MP3.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_wav_to_mp3(
    wav_path: str | Path,
    mp3_path: str | Path | None = None,
    bitrate: str = "192k",
) -> Path:
    """
    Конвертує WAV у MP3 за допомогою pydub (використовує ffmpeg).

    Args:
        wav_path: шлях до вихідного WAV.
        mp3_path: шлях для MP3; якщо None — той самий каталог, ім'я з розширенням .mp3.
        bitrate: бітрейт MP3, напр. "192k", "256k".

    Returns:
        Path до створеного MP3-файлу.

    Raises:
        FileNotFoundError: якщо WAV не існує.
        RuntimeError: якщо pydub/ffmpeg не вдалося виконати конвертацію.
    """
    from pydub import AudioSegment  # type: ignore

    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV не знайдено: {wav_path}")

    if mp3_path is None:
        mp3_path = wav_path.with_suffix(".mp3")
    else:
        mp3_path = Path(mp3_path)

    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Експорт MP3: %s → %s (bitrate %s)", wav_path, mp3_path, bitrate)
    try:
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(mp3_path), format="mp3", bitrate=bitrate)
    except Exception as e:
        raise RuntimeError(
            f"Не вдалося конвертувати в MP3: {e}. Переконайтесь, що ffmpeg встановлений і в PATH."
        ) from e

    if not mp3_path.exists():
        raise RuntimeError(f"Експорт завершився, але файл не створено: {mp3_path}")

    logger.info("MP3 збережено: %s", mp3_path)
    return mp3_path
