# AI Auto-Dubbing Studio

Локальний Python-додаток для автоматичного дубляжу відео з англійської на російську мову зі збереженням оригінальних голосів, інтонацій та таймінгів реплік. Працює повністю офлайн на RTX 4080 16GB.

---

## Що реалізовано

### Пайплайн (7 кроків)

```
відео/аудіо
    │
    ▼
[1] AudioExtractor     — витягує чисте аудіо через ffmpeg (16kHz WAV моно)
    │
    ▼
[2] Transcriber        — WhisperX large-v3: транскрипція + word-level alignment
    │                    + діаризація (pyannote) → [{speaker, start, end, text}]
    │                    → очищення VRAM після завершення
    ▼
[3] ReferenceCollector — нарізає найкращі аудіо-семпли для кожного SPEAKER_XX
    │                    автоматично вирішує: F5-TTS або GPT-SoVITS
    ▼
[4] Translator         — переклад EN→RU через Ollama API (isochrony-промпт:
    │                    зберігає кількість складів для кращого tімінгу)
    │
    │   ← ПАУЗА: користувач перевіряє вибір TTS для кожного спікера →
    │
    ▼
[5] TTSEngine          — генерація озвучки:
    │                    • F5-TTS (zero-shot, якщо референс < 60 сек)
    │                    • GPT-SoVITS (fine-tuning + inference, якщо ≥ 60 сек)
    │                    → очищення VRAM після кожного спікера
    ▼
[6] TimeStretcher      — pyrubberband: стискає/розтягує кожну фразу під
    │                    оригінальний тайминг без зміни тональності (pitch)
    │                    Ліміти: 0.5x – 2.0x, решта — pad тишею або crop
    ▼
[7] Mixer              — збирає всі фрази на numpy-таймлайн за координатами
                         WhisperX, нормалізує гучність → final.wav
```

### Модулі (`src/`)

| Файл | Опис |
|------|------|
| `config.py` | Централізована конфігурація: шляхи, dataclass-конфіги для кожного модуля, авто-завантаження `.env` |
| `audio_extractor.py` | ffmpeg wrapper: відео → `audio/clean.wav` (16kHz, моно, PCM 16-bit) |
| `transcriber.py` | WhisperX: транскрипція → alignment → діаризація → агрегація реплік → `transcription/result.json` |
| `reference_collector.py` | Нарізка WAV-семплів для кожного спікера, рішення F5/SoVITS за тривалістю, `references/speakers_meta.json` |
| `translator.py` | Ollama REST API, isochrony-промпт, retry (3 спроби), `translation/result.json` |
| `tts_engine.py` | F5-TTS через CLI; GPT-SoVITS через subprocess: підготовка даних → тренування → inference |
| `time_stretcher.py` | pyrubberband time-stretch, clamping [0.5x–2.0x], pad/trim до точної тривалості |
| `mixer.py` | numpy-таймлайн, авто-ресемплінг, нормалізація піку, підтримка моно/стерео |
| `pipeline.py` | Оркестратор: послідовний виклик кроків, збереження стану в `pipeline_state.json` (resume після збою), callbacks для UI |
| `ui_app.py` | Textual TUI: двопанельний дашборд, динамічні картки спікерів, прогрес-бар, системний лог |

### Особливості архітектури

- **VRAM-менеджмент**: `torch.cuda.empty_cache()` після кожної важкої моделі — WhisperX, TTS
- **Resume**: пайплайн зберігає стан після кожного кроку у `pipeline_state.json`. При перезапуску — продовжує з місця зупинки
- **Пауза для вибору TTS**: після діаризації UI показує картки спікерів із автоматичним вибором (F5/SoVITS), користувач може змінити і натиснути "Продовжити"
- **Isochrony**: Ollama отримує промпт зі збереженням кількості складів, що зменшує необхідний коефіцієнт стретчу

---

## Структура папок

```
dubbing-studio/
├── .env                    ← змінні середовища (HF_TOKEN, KMP_DUPLICATE_LIB_OK)
├── requirements.txt
├── run.bat                 ← запуск будь-якого скрипта через GPU Python
├── run_tests.bat           ← запуск тестів
├── dev-plan.md             ← детальний план розробки
│
├── src/                    ← весь код
├── tests/                  ← 80 тестів (unit + інтеграційні)
│
└── project_01/             ← робоча папка одного відео
    ├── input/              ← вхідне відео або аудіо
    ├── audio/              ← clean.wav (витягнуте аудіо)
    ├── transcription/      ← result.json (репліки)
    ├── references/         ← WAV-семпли для кожного SPEAKER_XX
    ├── translation/        ← result.json (з перекладом)
    ├── tts_output/         ← згенеровані фрази
    ├── stretched/          ← після time-stretching
    └── output/             ← final.wav
```

---

## Тести

80 тестів, всі проходять (`80 passed`):

| Файл | Тип | Що тестує |
|------|-----|-----------|
| `test_audio_extractor.py` | unit | ffmpeg команди, обробка помилок, mock subprocess |
| `test_transcriber.py` | unit | агрегація реплік, збереження/завантаження JSON, mock WhisperX |
| `test_translator.py` | unit | Ollama API, retry-логіка, isochrony-промпт, mock requests |
| `test_time_stretcher.py` | unit | clamping, pad/trim, стерео→моно, mock pyrubberband |
| `test_mixer.py` | unit | розміщення фраз на таймлайні, нормалізація, ресемплінг, реальні WAV |
| `test_pipeline.py` | інтеграційний | resume-логіка, збереження стану, end-to-end з усіма моками |
| `test_ui.py` | Textual pilot | SpeakerCard, DubbingApp компоненти, кнопки, розміри вікна |

---

## Налаштування середовища

### Вимоги
- Windows 10/11
- NVIDIA RTX 4080 (16GB VRAM)
- Miniconda
- ffmpeg у PATH

### Python середовище
Використовується conda env `transcription` з GPU PyTorch (`torch 2.8.0+cu128`).

```powershell
# Активація (стандартний conda activate не працює в Nushell)
# Використовуй повний шлях або run.bat

# Перевірка GPU
KMP_DUPLICATE_LIB_OK=TRUE C:\Users\makro\miniconda3\envs\transcription\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
```

### `.env` файл
```env
HF_TOKEN=<твій HuggingFace token>       # потрібен для pyannote діаризації
OLLAMA_HOST=http://localhost:11434       # за замовчуванням
KMP_DUPLICATE_LIB_OK=TRUE               # усуває OMP-конфлікт на Windows
GPT_SOVITS_DIR=C:/GPT-SoVITS           # шлях до встановленого GPT-SoVITS
```

### Встановлення залежностей
```powershell
# Основні пакети
C:\Users\makro\miniconda3\envs\transcription\python.exe -m pip install -r requirements.txt

# WhisperX (окремо — тягне специфічний torch)
C:\Users\makro\miniconda3\envs\transcription\python.exe -m pip install git+https://github.com/m-bain/whisperX.git

# F5-TTS
C:\Users\makro\miniconda3\envs\transcription\python.exe -m pip install git+https://github.com/SWivid/F5-TTS.git
```

### Ollama
```powershell
winget install Ollama.Ollama
ollama serve              # запустити сервер (окреме вікно)
ollama pull qwen2.5:14b  # завантажити модель (~9GB)
```

---

## Запуск

### Тести
```powershell
run_tests.bat
# або
KMP_DUPLICATE_LIB_OK=TRUE C:\Users\makro\miniconda3\envs\transcription\python.exe -m pytest tests/ -v
```

### UI
```powershell
run.bat src\ui_app.py
# або
KMP_DUPLICATE_LIB_OK=TRUE C:\Users\makro\miniconda3\envs\transcription\python.exe src\ui_app.py
```

### Використання

1. Поклади відео у `project_01/input/`
2. Запусти `run.bat src\ui_app.py`
3. Вкажи шлях до файлу в інтерфейсі
4. Натисни **ЗАПУСТИТИ ПАЙПЛАЙН**
5. Дочекайся діаризації — з'являться картки спікерів з автовибором TTS
6. Перевір/змін вибір TTS для кожного спікера
7. Натисни **ПРОДОВЖИТИ**
8. Результат: `project_01/output/final.wav`

---

## Зовнішні залежності (встановлюються окремо)

| Залежність | Призначення | Посилання |
|-----------|-------------|-----------|
| WhisperX | STT + діаризація | github.com/m-bain/whisperX |
| F5-TTS | Zero-shot TTS | github.com/SWivid/F5-TTS |
| GPT-SoVITS | Fine-tune TTS | github.com/RVC-Boss/GPT-SoVITS |
| ffmpeg | Аудіо екстракція | ffmpeg.org |
| Ollama | LLM inference | ollama.com |
| Rubber Band | Time-stretching (для pyrubberband) | breakfastquay.com/rubberband |

---

## Поточний статус

- [x] Всі 10 модулів написані
- [x] 80/80 тестів проходять
- [x] Середовище налаштоване (GPU torch, ffmpeg, whisperx)
- [ ] Ollama: модель ще не завантажена (`ollama pull qwen2.5:14b`)
- [ ] F5-TTS: потребує встановлення в `transcription` env
- [ ] GPT-SoVITS: потребує окремого клонування репо
- [ ] Реальне тестування на відео файлі
