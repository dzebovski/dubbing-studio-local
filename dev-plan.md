# Dev Plan — AI Auto-Dubbing Studio

## Контекст проекту
Локальний Python-додаток для автодубляжу відео (EN→RU).
Апаратура: RTX 4080 16GB VRAM.
TUI: Textual (двопанельний дашборд).
Вихід: аудіофайл WAV.

## Прийняті рішення
- Мова перекладу: EN → RU
- TUI: Textual (interface-example.md як основа)
- GPT-SoVITS: тренуємо в пайплайні якщо референс > 1 хв
- Після діаризації: пауза + кнопка "Продовжити" для вибору TTS
- Вихід: тільки аудіофайл (WAV)
- Робоча папка: project_01/ всередині dubbing-studio/

## Структура файлів

```
dubbing-studio/
├── dev-plan.md                 ← цей файл
├── plan.md
├── interface-example.md
│
├── src/
│   ├── config.py               ← шляхи, константи, налаштування
│   ├── pipeline.py             ← оркестратор (викликає всі модулі)
│   ├── ui_app.py               ← Textual TUI
│   ├── audio_extractor.py      ← ffmpeg: video → clean audio
│   ├── transcriber.py          ← WhisperX: audio → [{speaker, start, end, text}]
│   ├── reference_collector.py  ← нарізка семплів для кожного SPEAKER_XX
│   ├── translator.py           ← Ollama EN→RU з isochrony-промптом
│   ├── tts_engine.py           ← F5-TTS / GPT-SoVITS генерація
│   ├── time_stretcher.py       ← pyrubberband: підгонка під оригінальний тайминг
│   └── mixer.py                ← pydub: збирає всі шматки → фінальний WAV
│
├── tests/
│   ├── test_audio_extractor.py
│   ├── test_transcriber.py     ← мок WhisperX
│   ├── test_translator.py      ← мок Ollama API
│   ├── test_time_stretcher.py
│   ├── test_mixer.py
│   ├── test_pipeline.py        ← end-to-end з моками
│   └── test_ui.py              ← Textual компоненти
│
└── project_01/                 ← робоча папка для конкретного відео
    ├── input/                  ← сюди кладеш відео
    ├── audio/                  ← витягнуте чисте аудіо
    ├── transcription/          ← JSON з репліками WhisperX
    ├── references/
    │   ├── SPEAKER_00/         ← нарізані семпли спікера
    │   └── SPEAKER_01/
    ├── translation/            ← JSON з перекладеними репліками
    ├── tts_output/             ← згенеровані аудіо-фрази
    ├── stretched/              ← після time-stretching
    └── output/                 ← фінальний WAV
```

## Формат даних між модулями

### Транскрипція (transcription/result.json)
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 3.5,
    "text": "Hello world"
  }
]
```

### Переклад (translation/result.json)
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.0,
    "end": 3.5,
    "text": "Hello world",
    "translated_text": "Привет мир"
  }
]
```

### Вибір TTS (зберігається в pipeline state)
```json
{
  "SPEAKER_00": "gpt_sovits",
  "SPEAKER_01": "f5_tts"
}
```

## Стек залежностей
- `textual` — TUI
- `ffmpeg-python` — витягування аудіо
- `whisperx` — STT + діаризація
- `torch` — GPU, очищення VRAM між моделями
- `requests` — Ollama API
- `pydub` — нарізка семплів, зведення треків
- `pyrubberband` — time-stretching без pitch shift
- `soundfile` — читання/запис WAV

## TTS моделі
- **F5-TTS**: zero-shot, якщо референс < 60 сек
- **GPT-SoVITS**: fine-tuning, якщо референс >= 60 сек; запускається через subprocess CLI

## Ollama промпт (isochrony)
```
Translate the following English phrase to Russian.
Keep the translation roughly the same length (number of syllables) as the original.
Respond with ONLY the translated text, no explanations.
Original: "{text}"
```

## VRAM стратегія
- Після кожної важкої моделі: `torch.cuda.empty_cache()`
- Послідовне завантаження: WhisperX → вивантажити → Ollama (CPU) → TTS → вивантажити

## Порядок розробки

| # | Статус | Модуль | Нотатки |
|---|--------|--------|---------|
| 1 | [x] | config.py | шляхи відносно project_dir |
| 2 | [x] | audio_extractor.py | ffmpeg, вихід: audio/clean.wav |
| 3 | [x] | transcriber.py | WhisperX large-v3, HuggingFace token для діаризації |
| 4 | [x] | reference_collector.py | pydub, мін. семпл 3 сек, макс. 30 сек |
| 5 | [x] | translator.py | Ollama localhost:11434 |
| 6 | [x] | tts_engine.py | F5-TTS + GPT-SoVITS subprocess |
| 7 | [x] | time_stretcher.py | pyrubberband, ліміт стретчу 0.5x–2.0x |
| 8 | [x] | mixer.py | soundfile+numpy, вихід: output/final.wav |
| 9 | [x] | pipeline.py | оркестратор + збереження стану (pipeline_state.json) |
| 10 | [x] | ui_app.py | Textual, двопанельний дашборд |

## Порядок тестування

| # | Статус | Тест | Тип |
|---|--------|------|-----|
| 1 | [x] | test_audio_extractor.py | unit, мок subprocess |
| 2 | [x] | test_transcriber.py | unit, мок WhisperX |
| 3 | [x] | test_translator.py | unit, мок requests |
| 4 | [x] | test_time_stretcher.py | unit, реальний WAV + мок pyrubberband |
| 5 | [x] | test_mixer.py | unit, реальні WAV-шматки |
| 6 | [x] | test_pipeline.py | інтеграційний, всі зовнішні моки |
| 7 | [x] | test_ui.py | Textual pilot, компоненти |

## Нотатки для наступної сесії
- Продовжувати з першого незавершеного модуля в таблиці вище
- project_01/ — тестова робоча папка
- Моделі WhisperX та GPT-SoVITS мають бути встановлені окремо (не через pip у requirements.txt)
- HuggingFace token потрібен для діаризації (pyannote/speaker-diarization)
