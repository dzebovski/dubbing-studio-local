"""
config.py — централізована конфігурація проекту.
Всі шляхи, константи та налаштування моделей знаходяться тут.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Кореневі шляхи
# ---------------------------------------------------------------------------

# Корінь репозиторію (dubbing-studio/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Папка з вихідним кодом
SRC_DIR = ROOT_DIR / "src"


# ---------------------------------------------------------------------------
# Завантаження .env (без зовнішніх залежностей)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """
    Читає .env з кореня репозиторію та встановлює змінні середовища.
    Не перезаписує змінні які вже задані (наприклад через систему).
    Не потребує python-dotenv.
    """
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Не перезаписувати якщо вже задано
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


# ---------------------------------------------------------------------------
# Конфігурація проекту (project_01, project_02 тощо)
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """
    Описує всі шляхи для конкретного проекту (під-папки для одного відео).
    Створює директорії автоматично при ініціалізації.
    """
    project_dir: Path

    # Під-директорії — заповнюються в __post_init__
    input_dir:        Path = field(init=False)
    audio_dir:        Path = field(init=False)
    transcription_dir: Path = field(init=False)
    references_dir:   Path = field(init=False)
    translation_dir:  Path = field(init=False)
    tts_output_dir:   Path = field(init=False)
    stretched_dir:    Path = field(init=False)
    output_dir:       Path = field(init=False)

    def __post_init__(self):
        self.project_dir = Path(self.project_dir)
        self.input_dir         = self.project_dir / "input"
        self.audio_dir         = self.project_dir / "audio"
        self.transcription_dir = self.project_dir / "transcription"
        self.references_dir    = self.project_dir / "references"
        self.translation_dir   = self.project_dir / "translation"
        self.tts_output_dir    = self.project_dir / "tts_output"
        self.stretched_dir     = self.project_dir / "stretched"
        self.output_dir        = self.project_dir / "output"

    def create_dirs(self) -> None:
        """Створює всі необхідні папки якщо їх не існує."""
        for d in [
            self.input_dir,
            self.audio_dir,
            self.transcription_dir,
            self.references_dir,
            self.translation_dir,
            self.tts_output_dir,
            self.stretched_dir,
            self.output_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def references_dir_for(self, speaker_id: str) -> Path:
        """Повертає папку семплів для конкретного спікера."""
        p = self.references_dir / speaker_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def tts_output_dir_for(self, speaker_id: str) -> Path:
        """Повертає папку TTS-виходів для конкретного спікера."""
        p = self.tts_output_dir / speaker_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stretched_dir_for(self, speaker_id: str) -> Path:
        """Повертає папку stretched-файлів для конкретного спікера."""
        p = self.stretched_dir / speaker_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Стандартні шляхи до файлів
    @property
    def clean_audio_path(self) -> Path:
        return self.audio_dir / "clean.wav"

    @property
    def transcription_path(self) -> Path:
        return self.transcription_dir / "result.json"

    @property
    def translation_path(self) -> Path:
        return self.translation_dir / "result.json"

    @property
    def final_output_path(self) -> Path:
        return self.output_dir / "final.wav"


# ---------------------------------------------------------------------------
# Налаштування WhisperX
# ---------------------------------------------------------------------------

@dataclass
class WhisperConfig:
    model_name: str = "large-v3"
    language: str = "en"
    device: str = "cuda"
    compute_type: str = "float16"        # float16 — оптимально для RTX 4080
    batch_size: int = 16
    # HuggingFace токен для діаризації (pyannote/speaker-diarization)
    # Отримати на: https://huggingface.co/settings/tokens
    hf_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_TOKEN")
    )
    min_speakers: int = 1
    max_speakers: int = 10


# ---------------------------------------------------------------------------
# Налаштування Ollama / LLM перекладу
# ---------------------------------------------------------------------------

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:14b"           # або "llama3.1:8b"
    target_language: str = "Russian"
    timeout_seconds: int = 120
    # Промпт для isochrony-перекладу
    prompt_template: str = (
        "Translate the following English phrase to {target_language}.\n"
        "Keep the translation roughly the same length (number of syllables) as the original.\n"
        "Respond with ONLY the translated text, no explanations.\n"
        "Original: \"{text}\""
    )


# ---------------------------------------------------------------------------
# Налаштування референс-семплів
# ---------------------------------------------------------------------------

@dataclass
class ReferenceConfig:
    min_sample_duration_sec: float = 3.0
    max_sample_duration_sec: float = 30.0
    # Мінімальна загальна тривалість щоб вибрати GPT-SoVITS (замість F5-TTS)
    gpt_sovits_threshold_sec: float = 60.0
    sample_rate: int = 22050
    audio_format: str = "wav"


# ---------------------------------------------------------------------------
# Налаштування TTS
# ---------------------------------------------------------------------------

@dataclass
class F5TTSConfig:
    # Шлях до встановленого F5-TTS (якщо не в PATH)
    executable: str = "f5-tts_infer-cli"
    device: str = "cuda"
    sample_rate: int = 24000


@dataclass
class GPTSoVITSConfig:
    # Шлях до кореневої папки GPT-SoVITS
    root_dir: Path = Path(os.environ.get("GPT_SOVITS_DIR", "C:/GPT-SoVITS"))
    # Скрипти тренування та інференсу
    train_script: str = "GPT_SoVITS/scripts/train.py"
    infer_script: str = "GPT_SoVITS/inference_cli.py"
    device: str = "cuda"
    sample_rate: int = 32000
    # Кількість епох для fine-tuning
    gpt_epochs: int = 5
    sovits_epochs: int = 8


# ---------------------------------------------------------------------------
# Налаштування Time-Stretching
# ---------------------------------------------------------------------------

@dataclass
class TimeStretchConfig:
    # Ліміти стретчу: менше 0.5x або більше 2.0x — артефакти
    min_rate: float = 0.5
    max_rate: float = 2.0
    # Якщо вийшли за ліміт — обрізати тишею або обрізати аудіо
    clamp_strategy: str = "clamp"   # "clamp" або "pad"


# ---------------------------------------------------------------------------
# Налаштування Mixer
# ---------------------------------------------------------------------------

@dataclass
class MixerConfig:
    output_sample_rate: int = 44100
    output_channels: int = 1          # моно
    output_format: str = "wav"
    # Якщо між репліками є тиша — заповнюємо silence
    silence_fill: bool = True


# ---------------------------------------------------------------------------
# Головний об'єкт конфігурації
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """Зведена конфігурація всього додатку."""
    whisper:      WhisperConfig      = field(default_factory=WhisperConfig)
    ollama:       OllamaConfig       = field(default_factory=OllamaConfig)
    reference:    ReferenceConfig    = field(default_factory=ReferenceConfig)
    f5_tts:       F5TTSConfig        = field(default_factory=F5TTSConfig)
    gpt_sovits:   GPTSoVITSConfig    = field(default_factory=GPTSoVITSConfig)
    time_stretch: TimeStretchConfig  = field(default_factory=TimeStretchConfig)
    mixer:        MixerConfig        = field(default_factory=MixerConfig)


# Глобальний екземпляр конфігурації (можна перевизначити в тестах)
DEFAULT_CONFIG = AppConfig()


# ---------------------------------------------------------------------------
# Допоміжні функції
# ---------------------------------------------------------------------------

def get_project_config(project_path: str | Path) -> ProjectConfig:
    """
    Фабрика: повертає ProjectConfig для вказаної папки проекту.
    Автоматично створює всі під-директорії.
    """
    cfg = ProjectConfig(project_dir=Path(project_path))
    cfg.create_dirs()
    return cfg
